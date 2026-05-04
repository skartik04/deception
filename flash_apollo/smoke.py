"""Smoke-test endpoint: validates Flash deploy pipeline before committing to 70B."""
from runpod_flash import Endpoint, GpuGroup


@Endpoint(
    name="apollo-smoke",
    gpu=GpuGroup.AMPERE_80,
    workers=(0, 1),
    idle_timeout=30,
    dependencies=["torch"],
)
async def smoke(payload: dict) -> dict:
    import os
    import platform
    import torch

    gpu_count = torch.cuda.device_count()
    gpus = []
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpus.append(
            {
                "index": i,
                "name": props.name,
                "vram_gib": round(props.total_memory / (1024**3), 1),
            }
        )
    # Probe what's already in the base image
    preinstalled = {}
    for pkg in ("transformers", "accelerate", "tokenizers", "huggingface_hub", "safetensors", "bitsandbytes", "vllm"):
        try:
            mod = __import__(pkg)
            preinstalled[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            preinstalled[pkg] = None
    return {
        "echo": payload,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpus": gpus,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "runpod_volume_exists": os.path.isdir("/runpod-volume"),
        "preinstalled": preinstalled,
    }


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(smoke({"hello": "world"})))
