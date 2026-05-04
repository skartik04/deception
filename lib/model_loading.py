"""Strict model loaders that REFUSE to spill to CPU.

Project policy: CPU offload of Llama 70B is forbidden. Forward passes through
CPU-resident layers serialize over PCIe and produce 5–10x slowdowns that are
easy to miss in a logged run. Any script that needs the 70B model must use
`load_llama_strict()` (or its lower-level variants) so a too-small GPU pool
fails immediately rather than silently accepting CPU spillover.

The standard HF call

    AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        device_map="auto", max_memory={0: "70GiB", "cpu": "200GiB"},
    )

is exactly the pattern this loader replaces. The "cpu": "200GiB" key is what
permits silent offload; we drop it and additionally verify post-load that no
parameter sits on the host.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _detect_gpu_budget(per_gpu_gib: int) -> dict[int, str]:
    """Build a max_memory dict for every visible GPU. NO 'cpu' key — that's
    the whole point. If the model doesn't fit, HF raises ValueError on load."""
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError(
            "load_llama_strict requires CUDA. No GPUs visible. CPU offload of "
            "Llama 70B is project-policy forbidden (see lib/model_loading.py)."
        )
    return {i: f"{per_gpu_gib}GiB" for i in range(n)}


def assert_no_cpu_params(model: Any) -> None:
    """After load, fail if any parameter ended up on CPU."""
    cpu_params = [name for name, p in model.named_parameters() if p.device.type != "cuda"]
    if cpu_params:
        raise RuntimeError(
            f"CPU spillover detected on {len(cpu_params)} parameters "
            f"(first: {cpu_params[0]!r}). Project policy forbids CPU offload "
            f"of Llama 70B — increase per_gpu_gib or use more GPUs."
        )


def load_llama_strict(
    model_path: str,
    *,
    per_gpu_gib: int = 70,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = True,
) -> tuple[Any, Any]:
    """Load Llama (or any HF causal LM) across all visible GPUs in BF16.

    Raises ValueError (from HF) if weights don't fit, or RuntimeError if any
    parameter ends up on CPU after load. Never silently spills.

    Args:
        model_path: HF id or local path. e.g. /workspace/models/llama-3.3-70b-instruct
        per_gpu_gib: VRAM budget per visible GPU in GiB. Default 70 leaves
            ~10 GB activation headroom on an 80 GB card.
        dtype: Default bfloat16 (~140 GB for 70B). Use float16 for the same.
        local_files_only: True for offline/staged loads.

    Returns:
        (model, tokenizer)
    """
    max_memory = _detect_gpu_budget(per_gpu_gib)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        local_files_only=local_files_only,
    )
    assert_no_cpu_params(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer
