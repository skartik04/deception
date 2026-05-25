# Flash deployment handover — Apollo probe on Llama-3.3-70B

What I learned getting the Apollo deception probe running on RunPod Flash for a 140 GB model. Documented for the next person/session who needs to deploy a large-model GPU workload on Flash. Specific to this project's setup but most lessons generalize.

## TL;DR — recommended workflow

1. Iterate locally with `flash run` first. The dev server validates dependency resolution, imports, and decorator config in seconds. Almost every bug below the model layer (string formatting, decorator args, dep mismatches) reproduces locally without a GPU.
2. Once `flash run` is clean, `flash deploy --no-deps` for the first remote test. See dependency section for why `--no-deps`.
3. Use the CLIs (`flash`, `runpodctl`) before reaching for REST/GraphQL. Tool-preference rule is in /workspace/gergely_deception/CLAUDE.local.md:11.
4. Expect cold-start friction: region pinning, capacity allocation, volume DC-lock are all real constraints; budget time accordingly.

## Dependency packaging

### The 1.5 GB bundle limit

Flash uploads a tarball (`.flash/artifact.tar.gz`) to RunPod containing your code and Python deps. The limit is **1.5 GB**. The base image already has torch + CUDA, so torch/torchvision/torchaudio/triton are auto-excluded. But `accelerate` declares `torch>=2.0.0` as a hard dep, and pip will pull all transitive deps from PyPI, which drags in the `nvidia/*` CUDA wheels (~4.4 GB combined). Bundle blows past the limit.

### `--no-deps` is the right escape hatch

`flash deploy --no-deps` tells pip to install only the packages you list, no transitive resolution. This drops the bundle from ~2 GB to ~19 MB. But you must list every pure-Python transitive dep yourself or the worker will `ImportError`. The runtime "auto-install missing deps" safety net mentioned in the docs only catches **top-level missing packages**, not transitive ones.

### Working dep list for transformers + accelerate (Llama-3.3-70B forward pass)

```python
dependencies=[
    "transformers",
    "accelerate",
    "tokenizers<=0.23.0",     # transformers 5.7 pins this
    "huggingface-hub",
    "safetensors",
    "regex",                  # transformers transitive
    "packaging", "pyyaml", "tqdm", "psutil",
    "filelock", "fsspec", "hf-xet",
    "httpx", "typing-extensions", "typer",
],
```

Source: walked the `requires` field of each top-level dep, kept the non-extra entries, dropped `torch` and `numpy` (in base image). The version pin on `tokenizers` is critical — pip otherwise resolves to 0.23.1 which transformers refuses at import.

### Verifying the bundle locally

```bash
tar tzf .flash/artifact.tar.gz | grep -i tokenizers
ls /tmp/extracted/tokenizers-*.dist-info  # confirm version
```

### Verifying what's installed on the worker

Add a debug branch to your @Endpoint that returns versions and import paths. Useful when worker behavior diverges from local install.

```python
if debug:
    info = {}
    for pkg in ("tokenizers", "transformers", "huggingface_hub"):
        try:
            m = __import__(pkg)
            info[pkg] = {"version": m.__version__, "file": m.__file__}
        except Exception as e:
            info[pkg] = {"err": str(e)}
    return info
```

## Base image (`runpod/flash:py3.12-latest`)

Confirmed pre-installed: `torch 2.9.1+cu128`, `huggingface_hub` (some version), CUDA 12.8.

NOT pre-installed: `transformers`, `accelerate`, `tokenizers`, `safetensors`, `bitsandbytes`, `vllm`. Numpy is, but I didn't probe a full list.

Don't put torch in your `dependencies=[]` — it's auto-excluded from the bundle and would just slow build resolution.

## NetworkVolume for large models

### Why it's needed

Llama-3.3-70B in bf16 = ~140 GB across 31 safetensors shards. Container disk is capped well below this for serverless workers. Volume is the only place 140 GB fits.

Volume also caches weights across cold-starts. Without it, every fresh worker re-downloads from HuggingFace (~$0.30 in GPU time per cold-start). With it, second cold-start loads from network NVMe in ~16s.

### Volume gotchas

1. **Region-locked**. The volume binds to one DC, the endpoint must use the same DC, so GPU allocation is gated by that single DC's capacity. Saw this concretely: US-KS-2 stayed `IN_QUEUE` indefinitely with 0 workers because A100 80GB stock was empty there. Pivoted to US-CA-2.

2. **Not all DCs support volumes.** `flash deploy` will fail at provision time with the list of supported DCs. Currently: CA-MTL-3, CA-MTL-4, EU-CZ-1, EU-NL-1, EU-RO-1, EUR-IS-1, EUR-IS-3, EUR-NO-1, US-CA-2, US-GA-2, US-IL-1, US-KS-2, US-MO-1, US-MO-2, US-NC-2, US-NE-1, US-TX-3, US-WA-1.

3. **`flash undeploy` does NOT delete volumes.** It explicitly errors: `NetworkVolume undeploy is not yet supported. Network volumes must be manually deleted via RunPod UI or API.` Workflow:
   ```bash
   yes | uv run flash undeploy apollo-probe          # delete endpoint first
   curl -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
     https://rest.runpod.io/v1/networkvolumes/<id>   # then delete volume
   ```
   The volume can't be deleted while any endpoint references it, so order matters.

4. **`.flash/resources.pkl` caches volume state locally.** If you switch DC, Flash sees the old volume in the cached state and tries to "undeploy" it (which fails). Wipe `rm .flash/resources.pkl .flash/flash_manifest.json` before redeploying to a new DC.

5. **Mount path is fixed at `/runpod-volume`** regardless of which volume the worker landed on. So function code can hardcode `HF_HOME=/runpod-volume/hf` — only the @Endpoint decorator binds to a specific volume.

### Multi-DC volumes (untested)

The Flash manifest uses plural `networkVolumes` and the SDK probably accepts a list. If you need to escape region-pinning while keeping the cache, replicate the same `apollo-cache` volume across 2-3 DCs and pass them as a list. Verify with `flash deploy --help` or by inspecting the SDK before relying on this.

## Worker lifecycle quirks

### Warm workers don't pick up new code on `flash deploy`

This was the #1 source of confusion in this session. The flow:

1. `flash deploy` uploads a new bundle.
2. RunPod stores the new bundle as the endpoint's "current" artifact.
3. **Warm workers keep their original `/app` from when they started.** They don't restart on deploy.
4. New invocations route to the warm worker, which runs OLD code.

Symptom: you fix a bug, redeploy, re-invoke, see the same bug. Worker ID stays the same across attempts.

**Fix: drain the warm worker.** Set `workersMax=0`, wait for scale-down, set back to 1:
```bash
curl -X PATCH -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  https://rest.runpod.io/v1/endpoints/<id> \
  -d '{"workersMax":0}'
# wait for /health to show 0 workers
curl -X PATCH ... -d '{"workersMax":1}'
# now invoke; new worker gets the new bundle
```

The `flash` CLI does NOT cover this. REST is the only path.

### Cold-start timing (apollo-probe in US-CA-2)

| Phase | Duration |
|---|---|
| Worker provisioning (queue → IN_PROGRESS) | ~2-3 min |
| Initial download of 140 GB to volume | not measured separately; happened lazily inside `from_pretrained` |
| Re-load model from cached volume | ~16 s |
| Forward pass + probe scoring (~50 tokens) | ~31 s |
| Generation (~250 tokens with bf16 + CPU offload) | >5 min — abandoned, too slow without vLLM |

CPU offload at 70B is painfully slow for autoregressive generation. For chat use vLLM or a smaller model.

### `idle_timeout` doesn't always fire predictably

I saw a worker stay warm well past the configured `idle_timeout=600s`. RunPod's flashboot/keep-alive may extend it. Don't rely on natural scale-to-zero for stale-state cleanup; use the drain pattern above.

## GPU type fallback

A100 80GB is genuinely scarce on RunPod. Specifying just one GPU type risks indefinite queueing in any single DC. Pass a list to fall back across the 80 GB tier:

```python
gpu=[
    GpuType.NVIDIA_A100_80GB_PCIe,
    GpuType.NVIDIA_A100_SXM4_80GB,
    GpuType.NVIDIA_H100_80GB_HBM3,
],
```

Check current stock per DC: `runpodctl datacenter list` returns each DC's `gpuAvailability` array with `stockStatus` of "High"/"Medium"/"Low"/"" (empty = none). Cross-reference against the volume-supporting DC list above before pinning.

## CLI vs API

Use CLIs when they cover the operation. Use REST/GraphQL only for things they genuinely don't.

| Operation | Right tool |
|---|---|
| `flash deploy`, `flash undeploy`, `flash run`, `flash env` | `flash` CLI |
| Pod create/list/stop/remove, datacenter/gpu list, balance | `runpodctl` |
| Endpoint config inspection | `flash undeploy list` for names; REST `GET /v1/endpoints` for full config (templateId, gpuTypeIds, etc.) |
| Drain warm workers (scale to zero) | REST `PATCH /v1/endpoints/<id> {"workersMax":N}` |
| Invoke a deployed endpoint | REST `POST /v2/<id>/run` (sync) or `/runsync` (immediate, <30 s) |
| Job status | REST `GET /v2/<id>/status/<job-id>` |
| Health/worker pool inspection | REST `GET /v2/<id>/health` |
| Delete a NetworkVolume | REST `DELETE /v1/networkvolumes/<id>` (only after `flash undeploy` of any using endpoint) |

The flash CLI prints the right `curl` shape for invocation after each `flash deploy`. Copy it.

## Useful files in this project

- /workspace/gergely_deception/flash_apollo/apollo_probe.py — the probe scoring endpoint with optional `mode="chat"` branch.
- /workspace/gergely_deception/flash_apollo/smoke.py — small smoke endpoint that probes worker capabilities and base image deps. Useful template for environment debugging.
- /workspace/gergely_deception/flash_apollo/.flash/flash_manifest.json — generated manifest after deploy. Inspect this to see what Flash actually serialized (env vars, volume bindings, GPU types).
- /workspace/gergely_deception/flash_apollo/probe_weights/roleplaying.pt — Apollo's pretrained probe (layer-22 directions + scaler).
- /workspace/.claude/skills/flash/SKILL.md — official runpod-flash skill (symlink to npx-installed copy at /root/.agents/skills/flash).

## Open questions / things I didn't validate

- Does `volumes=[v1, v2, v3]` actually work in `@Endpoint`? Would let us escape region-pinning while keeping the weights cache.
- Are there serverless-specific GPU stock APIs that report different numbers from the pod-pool `runpodctl datacenter list`? My queries used the pod numbers, which may overstate or understate serverless capacity.
- Best path for chat/generation on this stack — vLLM via Flash @Endpoint? Custom Docker image with vLLM preinstalled? Hosted on a separate small endpoint, not co-located with the probe?
