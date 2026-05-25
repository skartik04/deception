"""Single-pass probe runner.

`ProbeEngine` owns:
  - the Llama 70B base model + tokenizer (loaded strict-GPU per project policy)
  - optionally a LoRA adapter wrapping the base
  - a list of `Probe`s sharing the same forward pass

Workflow:

    engine = ProbeEngine(
        model_path="/workspace/models/llama-3.3-70b-instruct",
        probes=[load_probe("apollo_roleplaying"), load_probe("balcells_linear")],
        suspect_system=SUSPECT_PROMPTS["debate_coach"],
    )
    engine.load()  # heavy: ~140GB load + LoRA wrap
    out = engine.score_argument(meta, argument)  # one forward, all probes scored

`out[probe_name]` is a `PerTokenScore` with `.tokens` and `.scores`.

The engine works identically inside a Flash worker (see `flash_app/main.py`
which simply imports `ProbeEngine` and wraps `score_argument` with @Endpoint).
There's no Flash-specific code in this module — keeping the deployment shim
thin is the whole point.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from lib.model_loading import assert_no_cpu_params, load_llama_strict
from lib.probes import PerTokenScore, Probe
from lib.prompts import _SUSPECT_USER
from lib.wmdp_data import format_question_block

# Ensure liars-bench tokenization helpers are importable.
_LIARS_BENCH = Path(__file__).resolve().parent.parent / "liars-bench" / "src" / "probes"
if _LIARS_BENCH.exists() and str(_LIARS_BENCH) not in sys.path:
    sys.path.insert(0, str(_LIARS_BENCH))


PADDING = {"llama": {"left": 0, "right": 0}}


class ProbeEngine:
    def __init__(
        self,
        model_path: str,
        probes: list[Probe],
        suspect_system: str,
        max_length: int = 4096,
        per_gpu_gib: int = 70,
    ) -> None:
        if not probes:
            raise ValueError("ProbeEngine needs at least one probe")
        lora_probes = [p for p in probes if p.needs_lora_adapter]
        if len(lora_probes) > 1:
            raise ValueError(
                "ProbeEngine supports at most ONE LoRA-wrapped probe per pass; "
                f"got {[p.name for p in lora_probes]}. Run them in separate engines."
            )
        self.model_path = model_path
        self.probes = probes
        self.suspect_system = suspect_system
        self.max_length = max_length
        self.per_gpu_gib = per_gpu_gib
        self._lora_probe = lora_probes[0] if lora_probes else None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        self._model, self._tokenizer = load_llama_strict(
            self.model_path, per_gpu_gib=self.per_gpu_gib
        )
        if self._lora_probe is not None:
            assert self._lora_probe.lora_adapter_dir is not None
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(
                self._model, self._lora_probe.lora_adapter_dir
            )
            assert_no_cpu_params(self._model)
            self._model.eval()
        # Pre-compute the union of layers we need to capture.
        layers: set[int] = set()
        for p in self.probes:
            layers.update(p.required_layers)
        self._needed_layers = sorted(layers)
        self._loaded = True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_argument(
        self, metadata: dict[str, Any], argument: str
    ) -> dict[str, PerTokenScore]:
        """Build the (system, user, assistant) dialogue from WMDP metadata
        + argument text and run all probes in one forward pass."""
        suspect_user = _SUSPECT_USER.format(
            question_block=format_question_block(
                str(metadata["question"]), list(metadata["choices"])
            ),
            argue_target=metadata["argue_target"],
            argue_target_text=metadata["argue_target_text"],
        )
        return self.score_dialogue(
            system=self.suspect_system, user=suspect_user, assistant=argument
        )

    def score_dialogue(
        self, system: str, user: str, assistant: str
    ) -> dict[str, PerTokenScore]:
        """Score arbitrary system/user/assistant dialogue. The detection
        window is the assistant tokens."""
        if not self._loaded:
            self.load()
        from deception_detection.data.base import Message
        from deception_detection.tokenized_data import TokenizedDataset
        from deception_detection.types import Dialogue

        dialogue: Dialogue = [
            Message("system", system, False),
            Message("user", user, False),
            Message("assistant", assistant, True),
        ]
        toks = TokenizedDataset.from_dialogue_list(
            [dialogue], self._tokenizer, padding=PADDING, max_length=self.max_length
        )
        device = next(self._model.parameters()).device
        input_ids = toks.tokens.to(device)
        mask = toks.detection_mask
        assert mask is not None
        with torch.no_grad():
            out = self._model(input_ids, output_hidden_states=True, use_cache=False)
        detect_mask = mask[0].bool()
        n_seq = min(detect_mask.shape[0], out.hidden_states[1].shape[1])
        detect_mask = detect_mask[:n_seq]
        # Build hidden_by_layer for the detection-window tokens only.
        hidden_by_layer: dict[int, torch.Tensor] = {}
        for layer_idx in self._needed_layers:
            # hidden_states[i] is the residual stream AFTER layer i-1, i.e. the
            # input to layer i — except hidden_states[layer+1] is the standard
            # convention used by both Apollo and Balcells. We follow it here.
            hs = out.hidden_states[layer_idx + 1][0][:n_seq]  # [n_seq, hidden]
            kept = hs[detect_mask]
            hidden_by_layer[layer_idx] = kept.to("cpu", dtype=torch.float32)
        # Token strings for the detection window.
        all_str_tokens = toks.str_tokens[0][:n_seq]
        kept_tokens = [
            all_str_tokens[i] for i in range(n_seq) if bool(detect_mask[i].item())
        ]
        # Dispatch each probe.
        result: dict[str, PerTokenScore] = {}
        for probe in self.probes:
            scores = probe.score_per_token(hidden_by_layer)
            result[probe.name] = PerTokenScore(
                tokens=list(kept_tokens), scores=scores.cpu().tolist()
            )
        return result
