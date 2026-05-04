"""Unified white-box probe interface.

A `Probe` consumes per-token hidden states from a transformer's residual stream
at one or more layers and emits a per-token deception score (higher = more
deceptive). Callers run a single forward pass that captures every layer needed
by every probe, then dispatch each probe over the captured hidden states.

This replaces the half-dozen one-off `eval_*.py` scripts that each:
  1. Loaded Llama 70B
  2. Built a suspect/user/argument dialogue
  3. Did a forward pass with `output_hidden_states=True`
  4. Pulled out their probe-specific layers
  5. Applied probe-specific math (LR detector, MMS, linear head, RepE-LAT, ...)

Step 5 was the only thing that varied. Everything else is shared by `ProbeEngine`.

## Probe types implemented

- ApolloLRProbe          — Apollo's `LogisticRegressionDetector` checkpoints
                           (roleplaying / followup / instructed_pairs / sae_rp / our
                           custom-trained geometry_of_truth, wmdp_full_arg,
                           wmdp_answer_only — all use the same .pt format)
- ApolloMMSProbe         — Apollo's `MMSDetector` checkpoints (descriptive)
- BalcellsLinearHead     — Balcells linear head (Linear(hidden, 1)) at config
                           layer; optionally with a LoRA adapter wrapping the base
                           model. Output is sigmoid(P(hallucination)) per token.
- RepELATProbe           — Zou et al. 2023 PCA-per-layer with sign assignment.
                           Score is sum over (sign * projection) across the
                           detector layers, negated so higher = lying.

## Adding a new probe

Subclass `Probe`, declare:
  - `name`: str (identifier shown in result JSONs)
  - `required_layers`: tuple[int, ...]
  - `needs_lora_adapter`: bool (and if True, set `lora_adapter_dir`)
  - `score_per_token(hidden_by_layer) -> Tensor[seq]`
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# liars-bench probe utilities (LogisticRegressionDetector, MMSDetector, etc.)
_LIARS_BENCH_PROBES = (
    Path(__file__).resolve().parent.parent / "liars-bench" / "src" / "probes"
)
if _LIARS_BENCH_PROBES.exists() and str(_LIARS_BENCH_PROBES) not in sys.path:
    sys.path.insert(0, str(_LIARS_BENCH_PROBES))


@dataclass(frozen=True)
class PerTokenScore:
    """Output of probe.score_per_token, with token strings and scores aligned."""

    tokens: list[str]
    scores: list[float]

    def pool(self, mode: str) -> float:
        if not self.scores:
            return float("nan")
        if mode == "mean":
            return float(np.mean(self.scores))
        if mode == "max":
            return float(np.max(self.scores))
        if mode == "last":
            return float(self.scores[-1])
        raise ValueError(f"unknown pool mode {mode!r}")


class Probe(ABC):
    """Stateful probe object. Subclasses load weights in `__init__` and
    expose `score_per_token` to consume hidden states already extracted by
    the engine."""

    name: str
    required_layers: tuple[int, ...]

    # If True, the engine wraps the base model in `PeftModel.from_pretrained(
    # base, lora_adapter_dir)` BEFORE running the forward pass. Only one probe
    # in a batch may need a LoRA adapter (engine refuses otherwise).
    needs_lora_adapter: bool = False
    lora_adapter_dir: str | None = None

    @abstractmethod
    def score_per_token(
        self, hidden_by_layer: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Score every token in the detection window.

        Args:
            hidden_by_layer: {layer_idx: tensor[n_detect_tokens, hidden_size]}
                in float32 on CPU. Includes only the layers in
                `self.required_layers`.

        Returns:
            tensor[n_detect_tokens] of deception scores. Higher = more
            deceptive. Sign convention is the probe's responsibility.
        """
        ...


# ---------------------------------------------------------------------------
# Apollo-style probes (liars-bench LogisticRegressionDetector / MMSDetector)
# ---------------------------------------------------------------------------


class ApolloLRProbe(Probe):
    """Wraps liars-bench's `LogisticRegressionDetector` (the .pt format used
    by Apollo's published probes and our custom-trained ones).

    Apollo's scoring formula (verified in `trash/flash_apollo/apollo_probe.py`):
        standardized = (act - scaler_mean) / scaler_scale
        per_tok_logit = standardized @ direction
    """

    def __init__(self, name: str, detector_pt: Path) -> None:
        from deception_detection.detectors import LogisticRegressionDetector

        self.name = name
        self._detector = LogisticRegressionDetector.load(detector_pt)
        layers = list(self._detector.layers)
        if len(layers) != 1:
            raise ValueError(
                f"ApolloLRProbe expects single-layer detectors, got layers={layers}"
            )
        self.required_layers = (layers[0],)
        # Pre-cache as float32 on CPU. LR detectors always carry direction +
        # scaler tensors; assert so a corrupted .pt fails loudly here, not
        # later inside score_per_token.
        d = self._detector
        assert d.directions is not None, f"LR detector {detector_pt} missing directions"
        assert d.scaler_mean is not None, f"LR detector {detector_pt} missing scaler_mean"
        assert d.scaler_scale is not None, f"LR detector {detector_pt} missing scaler_scale"
        self._direction = d.directions.squeeze(0).float()
        self._scaler_mean = d.scaler_mean.squeeze(0).float()
        self._scaler_scale = d.scaler_scale.squeeze(0).float()
        self._normalize = bool(getattr(d, "normalize", True))

    def score_per_token(
        self, hidden_by_layer: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        hs = hidden_by_layer[self.required_layers[0]]  # [n, hidden] cpu f32
        if self._normalize:
            hs = (hs - self._scaler_mean) / self._scaler_scale
        return hs @ self._direction


class ApolloMMSProbe(Probe):
    """Wraps liars-bench's `MMSDetector` (e.g. Apollo's `descriptive` probe).

    MMS direction is `pos_mean - neg_mean` and is NOT scaled — there's no
    scaler attribute on the class (verified against
    liars-bench/src/probes/deception_detection/detectors.py:207).
    """

    def __init__(self, name: str, detector_pt: Path) -> None:
        from deception_detection.detectors import MMSDetector

        self.name = name
        self._detector = MMSDetector.load(detector_pt)
        layers = list(self._detector.layers)
        if len(layers) != 1:
            raise ValueError(
                f"ApolloMMSProbe expects single-layer detectors, got layers={layers}"
            )
        self.required_layers = (layers[0],)
        assert self._detector.directions is not None, f"MMS detector {detector_pt} missing directions"
        self._direction = self._detector.directions.squeeze(0).float()

    def score_per_token(
        self, hidden_by_layer: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        hs = hidden_by_layer[self.required_layers[0]]
        return hs @ self._direction


# ---------------------------------------------------------------------------
# Balcells linear-head probes
# ---------------------------------------------------------------------------


class BalcellsLinearHead(Probe):
    """Linear(hidden_size, 1) head on the post-block residual stream at a
    configured layer. Supports the LoRA-adapted variants by setting
    `needs_lora_adapter=True` when an adapter_config.json is present.

    Output: per-token logit. (Apply sigmoid downstream if you want
    probability; but rank-AUROC doesn't care.)
    """

    def __init__(self, name: str, probe_dir: Path) -> None:
        cfg = json.loads((probe_dir / "probe_config.json").read_text())
        layer_idx = int(cfg["layer_idx"])
        hidden_size = int(cfg["hidden_size"])
        self.name = name
        self.required_layers = (layer_idx,)
        adapter_cfg = probe_dir / "adapter_config.json"
        if adapter_cfg.exists():
            self.needs_lora_adapter = True
            self.lora_adapter_dir = str(probe_dir)
        head = torch.nn.Linear(hidden_size, 1, device="cpu", dtype=torch.float32)
        sd = torch.load(
            probe_dir / "probe_head.bin", map_location="cpu", weights_only=True
        )
        head.load_state_dict({k: v.float() for k, v in sd.items()})
        head.eval()
        self._head = head

    def score_per_token(
        self, hidden_by_layer: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        hs = hidden_by_layer[self.required_layers[0]]
        with torch.no_grad():
            return self._head(hs).squeeze(-1)


# ---------------------------------------------------------------------------
# RepE-LAT probe (PCA per layer + sign assignment, sum across middle layers)
# ---------------------------------------------------------------------------


class RepELATProbe(Probe):
    """Zou et al. 2023 RepE-LAT honesty probe.

    Stored format (on disk): a .pt with
        {
          "directions": {layer_idx: np.ndarray[hidden]},
          "signs":      {layer_idx: float},
          "detector_layers": list[int],
        }
    Per token, score = sum over detector_layers of (-sign[l] * (h_l @ d_l)).
    Higher = more deceptive (we negate honesty).
    """

    def __init__(self, name: str, probe_path: Path) -> None:
        blob = torch.load(probe_path, map_location="cpu", weights_only=False)
        directions: dict[int, np.ndarray] = blob["directions"]
        signs: dict[int, float] = blob["signs"]
        detector_layers: list[int] = list(blob["detector_layers"])
        self.name = name
        self.required_layers = tuple(detector_layers)
        # Pre-stack to a single matrix [n_layers, hidden] for fast eval.
        self._detector_layers = detector_layers
        self._directions_t: dict[int, torch.Tensor] = {
            l: torch.tensor(directions[l], dtype=torch.float32)
            for l in detector_layers
        }
        self._signs: dict[int, float] = {l: float(signs[l]) for l in detector_layers}

    def score_per_token(
        self, hidden_by_layer: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        first_layer = self._detector_layers[0]
        n = hidden_by_layer[first_layer].shape[0]
        cumulative = torch.zeros(n, dtype=torch.float32)
        for l in self._detector_layers:
            hs = hidden_by_layer[l]  # [n, hidden] cpu f32
            proj = hs @ self._directions_t[l]
            cumulative = cumulative + (-self._signs[l] * proj)  # negate for lie
        return cumulative


# ---------------------------------------------------------------------------
# Probe registry / loader
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Apollo / custom LR probes — all share LogisticRegressionDetector format.
_APOLLO_LR_PROBES = {
    "apollo_roleplaying": REPO_ROOT
    / "liars-bench/src/probes/example_results/roleplaying/detector.pt",
    "apollo_followup": REPO_ROOT
    / "liars-bench/src/probes/example_results/followup/detector.pt",
    "apollo_instructed_pairs": REPO_ROOT
    / "liars-bench/src/probes/example_results/instructed_pairs/detector.pt",
    "apollo_sae_rp": REPO_ROOT
    / "liars-bench/src/probes/example_results/sae_rp/detector.pt",
    # Custom-trained probes (LR-detector format, all layer 22).
    "geometry_of_truth": REPO_ROOT / "probes/geometry_of_truth_llama70b_layer22.pt",
    "wmdp_full_arg": REPO_ROOT / "probes/wmdp_bio_interrogation_custom_layer22.pt",
    "wmdp_answer_only": REPO_ROOT / "probes/wmdp_bio_answer_only_layer22.pt",
}

_APOLLO_MMS_PROBES = {
    "apollo_descriptive": REPO_ROOT
    / "liars-bench/src/probes/example_results/descriptive/detector.pt",
}

_BALCELLS_PROBES = {
    "balcells_linear": REPO_ROOT / "probes/balcells/llama3_3_70b_linear",
    "balcells_lora_kl": REPO_ROOT / "probes/balcells/llama3_3_70b_lora_lambda_kl_0_05",
}

_REPE_LAT_PROBES = {
    "repe_lat": REPO_ROOT / "probes/repe_lat_llama70b.pt",
}


def list_available_probes() -> list[str]:
    """Names of probes whose weight files exist on disk right now."""
    out: list[str] = []
    for d in (_APOLLO_LR_PROBES, _APOLLO_MMS_PROBES, _BALCELLS_PROBES, _REPE_LAT_PROBES):
        for name, p in d.items():
            if p.exists():
                out.append(name)
    return out


def load_probe(name: str) -> Probe:
    """Construct a Probe from its registered name. Raises if weights missing.

    Use `list_available_probes()` to see what's loadable in the current repo
    state. Probe weights live under `probes/` and `liars-bench/src/probes/
    example_results/`.
    """
    if name in _APOLLO_LR_PROBES:
        path = _APOLLO_LR_PROBES[name]
        if not path.exists():
            raise FileNotFoundError(f"Probe {name!r} weights missing: {path}")
        return ApolloLRProbe(name=name, detector_pt=path)
    if name in _APOLLO_MMS_PROBES:
        path = _APOLLO_MMS_PROBES[name]
        if not path.exists():
            raise FileNotFoundError(f"Probe {name!r} weights missing: {path}")
        return ApolloMMSProbe(name=name, detector_pt=path)
    if name in _BALCELLS_PROBES:
        path = _BALCELLS_PROBES[name]
        if not (path / "probe_config.json").exists():
            raise FileNotFoundError(f"Probe {name!r} weights missing: {path}")
        return BalcellsLinearHead(name=name, probe_dir=path)
    if name in _REPE_LAT_PROBES:
        path = _REPE_LAT_PROBES[name]
        if not path.exists():
            raise FileNotFoundError(
                f"RepE-LAT probe {name!r} not found at {path}. "
                "Train one and persist it with `save_repe_lat_probe(...)` from "
                "this module. The legacy training script is at "
                "trash/eval_repe_lat.py — it has the contrast-pair + PCA pipeline."
            )
        return RepELATProbe(name=name, probe_path=path)
    raise KeyError(
        f"Unknown probe {name!r}. Available: "
        f"{sorted(list(_APOLLO_LR_PROBES) + list(_APOLLO_MMS_PROBES) + list(_BALCELLS_PROBES) + list(_REPE_LAT_PROBES))}"
    )


def save_repe_lat_probe(
    out_path: Path,
    directions: dict[int, np.ndarray],
    signs: dict[int, float],
    detector_layers: list[int],
) -> None:
    """Persist a trained RepE-LAT probe in the format expected by `RepELATProbe`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "directions": directions,
            "signs": signs,
            "detector_layers": detector_layers,
        },
        out_path,
    )
