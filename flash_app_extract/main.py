"""Flash worker — per-sentence activation extractor for probe training.

Endpoint signature: takes a list of labeled samples and a list of layers,
forwards each through Llama 3.3 70B, returns per-sentence pooled
activations (mean over the sentence's argument tokens) + the labels.

The resulting tensor (small enough to ship over HTTP) is then used to
train a linear probe locally (CPU is fine for ~few hundred sentences ×
8192 dims).

This is intentionally self-contained — no `lib/` imports — because Flash
bundles only the files alongside this script. The extraction logic
duplicates pieces of `ProbeEngine.score_dialogue` inline.

Payload schema:
    {
        "samples": [
            {
                "sample_id": "1",
                "system": "...",
                "user": "...",
                "assistant": "...",
                "sentence_spans": [[start, end], ...],  # char offsets in `assistant`
                "labels": [0.0, 0.5, 1.0, ...]           # one per sentence
            },
            ...
        ],
        "layers": [22, 50, 76]
    }

Response:
    {
        "per_sentence": [
            {
                "sample_id": "1",
                "sentence_idx": 0,
                "label": 0.0,
                "activations": {"22": [8192 floats], "50": [...], "76": [...]}
            },
            ...
        ],
        "n_layers": 3,
        "hidden_size": 8192,
        "load_time_s": 16.0,
        "extract_time_s": 480.0
    }
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
    name="extract-activations",
    gpu=[
        GpuType.NVIDIA_A100_80GB_PCIe,
        GpuType.NVIDIA_A100_SXM4_80GB,
        GpuType.NVIDIA_H100_80GB_HBM3,
    ],
    datacenter=DataCenter.US_CA_2,
    volume=apollo_volume,
    template=PodTemplate(containerDiskInGb=80),
    workers=(0, 1),
    gpu_count=2,  # Llama 3.3 70B BF16 = 140 GB; needs 2× 80 GB GPU.
    idle_timeout=600,
    execution_timeout_ms=7_200_000,  # 2 hours
    env={
        "HF_TOKEN": HF_TOKEN,
        "HF_HOME": "/runpod-volume/hf",
        "TRANSFORMERS_CACHE": "/runpod-volume/hf",
    },
    dependencies=[
        # Same minimal list as legacy workers (see docs/flash_handover.md)
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
        "numpy",
    ],
)
async def extract_per_sentence(
    payload: dict[str, Any] = {}, debug: bool = False
) -> dict[str, Any]:
    """Forward each sample through Llama 3.3 70B, return per-sentence
    pooled activations at the requested layers.
    """
    import time

    if debug:
        import sys

        info: dict[str, Any] = {"app_listing": sorted(os.listdir("/app"))[:30]}
        for pkg in ("torch", "transformers", "tokenizers", "huggingface_hub", "numpy"):
            try:
                m = __import__(pkg)
                info[pkg] = {"version": getattr(m, "__version__", "?")}
            except Exception as e:
                info[pkg] = {"err": str(e)}
        info["sys_path"] = sys.path[:5]
        return info

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HOME", "/runpod-volume/hf")
    os.makedirs("/runpod-volume/hf", exist_ok=True)

    # Persist model across worker invocations.
    global _model, _tokenizer  # type: ignore[name-defined]
    first_call = "_model" not in globals()
    t_load = time.perf_counter()
    if first_call:
        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        _tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
            _tokenizer.pad_token_id = _tokenizer.eos_token_id

        # Strict GPU-only load (project policy — see CLAUDE.local.md)
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No CUDA devices available")
        max_memory = {i: "78GiB" for i in range(n_gpus)}
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
        )
        _model.eval()
        cpu_params = [
            n for n, p in _model.named_parameters() if p.device.type != "cuda"
        ]
        if cpu_params:
            raise RuntimeError(
                f"CPU spillover on {len(cpu_params)} params (first: {cpu_params[0]!r})."
            )
    load_time = time.perf_counter() - t_load

    samples = payload.get("samples", [])
    layers: list[int] = list(payload.get("layers", [22, 50, 76]))
    if not samples:
        return {"per_sentence": [], "n_layers": len(layers), "first_call": first_call}

    results: list[dict[str, Any]] = []
    t_extract = time.perf_counter()

    for s in samples:
        # Build the chat-format dialogue and find the assistant token range.
        msgs = [
            {"role": "system", "content": s["system"]},
            {"role": "user", "content": s["user"]},
            {"role": "assistant", "content": s["assistant"]},
        ]
        formatted: str = _tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        # Find character span of the assistant content inside the formatted string.
        argument = s["assistant"].strip()
        arg_start_char = formatted.rfind(argument)
        if arg_start_char < 0:
            continue
        arg_end_char = arg_start_char + len(argument)

        enc = _tokenizer(formatted, return_offsets_mapping=True, padding=False)
        input_ids = torch.tensor([enc["input_ids"]], device=next(_model.parameters()).device)
        offsets: list[tuple[int, int]] = enc["offset_mapping"]

        # Compute per-token char offsets RELATIVE to the assistant content.
        # token belongs to assistant range iff its formatted-char-span overlaps [arg_start_char, arg_end_char).
        rel_offsets: list[tuple[int, int]] = []
        token_in_arg: list[bool] = []
        for tok_start, tok_end in offsets:
            if tok_start < arg_end_char and tok_end > arg_start_char:
                rel_offsets.append(
                    (
                        max(0, tok_start - arg_start_char),
                        min(len(argument), tok_end - arg_start_char),
                    )
                )
                token_in_arg.append(True)
            else:
                rel_offsets.append((0, 0))
                token_in_arg.append(False)

        with torch.no_grad():
            out = _model(input_ids, output_hidden_states=True, use_cache=False)

        # Pool per-sentence: mean of tokens whose char-overlap is within the span.
        spans: list[tuple[int, int]] = [tuple(sp) for sp in s["sentence_spans"]]  # type: ignore[misc]
        labels = list(s["labels"])
        for sent_idx, ((ss, se), lbl) in enumerate(zip(spans, labels)):
            tok_idx_in_sent: list[int] = []
            for ti, ((rs, re), in_arg) in enumerate(zip(rel_offsets, token_in_arg)):
                if not in_arg:
                    continue
                if rs < se and re > ss:
                    tok_idx_in_sent.append(ti)
            if not tok_idx_in_sent:
                continue
            import base64

            import numpy as np

            acts_per_layer: dict[str, str] = {}
            for layer_idx in layers:
                hs = out.hidden_states[layer_idx + 1][0]  # [seq, hidden]
                pooled = (
                    hs[tok_idx_in_sent].mean(dim=0).to("cpu", dtype=torch.float16)
                )
                buf = pooled.numpy().astype(np.float16).tobytes()
                acts_per_layer[str(layer_idx)] = base64.b64encode(buf).decode("ascii")
            results.append(
                {
                    "sample_id": s["sample_id"],
                    "sentence_idx": sent_idx,
                    "label": float(lbl),
                    "activations_b64_f16": acts_per_layer,
                }
            )

    extract_time = time.perf_counter() - t_extract
    # hidden_size = number of float16 entries in the first activation buffer.
    if results:
        import base64

        first_b = base64.b64decode(results[0]["activations_b64_f16"][str(layers[0])])
        hidden_size = len(first_b) // 2  # float16 = 2 bytes
    else:
        hidden_size = 0
    return {
        "per_sentence": results,
        "n_layers": len(layers),
        "hidden_size": hidden_size,
        "load_time_s": round(load_time, 1),
        "extract_time_s": round(extract_time, 1),
        "first_call": first_call,
    }
