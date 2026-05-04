"""Flash deployment of `ProbeEngine`.

Single-endpoint replacement for the old `flash_apollo/` and `flash_repe_lat/`
projects. Takes a payload describing which probe(s) to run and a (system, user,
assistant) dialogue or a WMDP metadata + argument, returns per-token scores.

Why this file is so short
-------------------------
All the heavy lifting — model load, tokenization, forward pass, probe dispatch —
lives in `lib.probe_engine.ProbeEngine`. This file does only two things:

  1. Wraps the engine in an `@Endpoint` decorator with the right GPU / volume /
     dependency config (see `docs/flash_handover.md` for why each dep is listed).
  2. Persists the engine across warm-worker invocations.

When you need to add a new probe, do NOT change this file — register it in
`lib/probes.py` and pass its name in the payload.

Local invocation (no Flash):
    from lib.probe_engine import ProbeEngine
    from lib.probes import load_probe
    from lib.prompts import SUSPECT_PROMPTS
    engine = ProbeEngine(
        model_path="/workspace/models/llama-3.3-70b-instruct",
        probes=[load_probe("apollo_roleplaying")],
        suspect_system=SUSPECT_PROMPTS["debate_coach"],
    )
    engine.load()
    out = engine.score_argument(metadata, argument)

Flash invocation:
    POST /v2/<endpoint>/run
    payload = {"probes": ["apollo_roleplaying"], "metadata": {...}, "argument": "..."}
    -> {"per_token_scores": {"apollo_roleplaying": {"tokens": [...], "scores": [...]}}}
"""

from __future__ import annotations

import os
from typing import Any

from runpod_flash import DataCenter, Endpoint, GpuType, NetworkVolume, PodTemplate

HF_TOKEN = os.environ.get("HF_TOKEN", "")

apollo_volume = NetworkVolume(
    name="apollo-cache",
    size=200,
    datacenter=DataCenter.US_CA_2,
)


@Endpoint(
    name="probe-engine",
    gpu=[
        GpuType.NVIDIA_A100_80GB_PCIe,
        GpuType.NVIDIA_A100_SXM4_80GB,
        GpuType.NVIDIA_H100_80GB_HBM3,
    ],
    datacenter=DataCenter.US_CA_2,
    volume=apollo_volume,
    template=PodTemplate(containerDiskInGb=80),
    workers=(0, 1),
    idle_timeout=600,
    execution_timeout_ms=4_800_000,
    env={
        "HF_TOKEN": HF_TOKEN,
        "HF_HOME": "/runpod-volume/hf",
        "TRANSFORMERS_CACHE": "/runpod-volume/hf",
    },
    dependencies=[
        # Same minimal list as the legacy workers (see docs/flash_handover.md).
        "transformers",
        "accelerate",
        "tokenizers<=0.23.0",
        "huggingface-hub",
        "safetensors",
        "regex",
        "packaging",
        "pyyaml",
        "tqdm",
        "psutil",
        "filelock",
        "fsspec",
        "hf-xet",
        "httpx",
        "typing-extensions",
        "typer",
        "scikit-learn",
        "numpy",
        "peft",  # only needed if balcells_lora_kl is requested
    ],
)
async def score(payload: dict[str, Any] = {}, debug: bool = False) -> dict[str, Any]:
    """Score an argument with one or more probes.

    Payload keys:
      - probes: list[str]               — names from lib.probes.list_available_probes()
      - dialogue: dict[str,str] OR
        metadata: dict + argument: str  — exactly one of the two

    For probes whose weights aren't bundled (e.g. RepE-LAT), the payload may
    include `repe_lat_directions` / `repe_lat_signs` / `repe_lat_layers` to
    instantiate `RepELATProbe` from in-memory state. (The bundled flash app
    doesn't yet support that — the legacy `flash_repe_lat/main.py` does, see
    `trash/flash_repe_lat/main.py` for that pattern.)
    """
    if debug:
        import sys

        info: dict[str, Any] = {"app_listing": sorted(os.listdir("/app"))[:30]}
        for pkg in ("torch", "transformers", "tokenizers", "huggingface_hub", "sklearn", "numpy", "peft"):
            try:
                m = __import__(pkg)
                info[pkg] = {"version": getattr(m, "__version__", "?")}
            except Exception as e:
                info[pkg] = {"err": str(e)}
        info["sys_path"] = sys.path[:5]
        return info

    # Lazy imports keep cold-start scope minimal.
    from lib.probe_engine import ProbeEngine
    from lib.probes import load_probe
    from lib.prompts import SUSPECT_PROMPTS

    global _engine, _engine_key  # type: ignore[name-defined]

    probe_names = list(payload.get("probes", ["apollo_roleplaying"]))
    suspect_key = str(payload.get("suspect_system_key", "debate_coach"))
    suspect_system = SUSPECT_PROMPTS[suspect_key]

    # Cache the engine across invocations IFF same probes + same suspect prompt.
    key = (tuple(probe_names), suspect_key)
    if "_engine" not in globals() or _engine_key != key:  # type: ignore[name-defined]
        _engine = ProbeEngine(
            model_path="meta-llama/Llama-3.3-70B-Instruct",
            probes=[load_probe(n) for n in probe_names],
            suspect_system=suspect_system,
        )
        _engine.load()
        _engine_key = key  # type: ignore[name-defined]

    if "dialogue" in payload:
        d = payload["dialogue"]
        out = _engine.score_dialogue(d["system"], d["user"], d["assistant"])
    else:
        out = _engine.score_argument(payload["metadata"], payload["argument"])

    return {
        "per_token_scores": {
            name: {"tokens": ps.tokens, "scores": ps.scores} for name, ps in out.items()
        },
        "probes": probe_names,
    }
