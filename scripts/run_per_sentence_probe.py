"""Run the per-sentence behavioral probe on an inspect_ai eval log.

Designed to be invoked as a subprocess so argument text never enters the
calling agent's conversation context. Reads the eval log, runs the
calibration check sentence-by-sentence for every sample, and dumps a
self-contained JSON with sentence verdicts + sample-level pooled scores.

Output schema (results/per_sentence/<run_name>.json):
{
  "eval_log": "...",
  "suspect_model": "...",
  "samples": [
    {
      "sample_id": "...",
      "is_deceptive": bool,
      "argue_target": "B",
      "n_sentences": 12,
      "verdicts": [...],
      "pooled": {"mean": 0.4, "max": 1.0, "last": 0.5}
    },
    ...
  ],
  "AUROCs": {"mean": ..., "max": ..., "last": ...}
}

Usage:
    OPENROUTER_API_KEY=... uv run python scripts/run_per_sentence_probe.py \
        <eval_log_path> [--limit N] [--out results/per_sentence/run.json] \
        [--suspect-model openrouter/meta-llama/llama-3.3-70b-instruct:nitro]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from sklearn.metrics import roc_auc_score

# Ensure repo root on sys.path so `lib.*` imports work when invoked directly.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.per_sentence_behavioral import (
    load_eval_argument_blob,
    score_argument_per_sentence,
    serialize,
)


def _pool(scores: list[float], mode: str) -> float:
    if not scores:
        return 0.5
    if mode == "mean":
        return sum(scores) / len(scores)
    if mode == "max":
        return max(scores)
    return scores[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_log", help="Path to inspect_ai .eval log")
    p.add_argument(
        "--suspect-model",
        default="openrouter/meta-llama/llama-3.3-70b-instruct:nitro",
    )
    p.add_argument("--limit", type=int, default=0, help="0 = all samples")
    p.add_argument(
        "--out",
        default="results/per_sentence/run.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="OpenRouter max_tokens per calibration call",
    )
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Try to load from .env in repo root.
        envfile = REPO / ".env"
        if envfile.exists():
            for line in envfile.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    os.environ["OPENROUTER_API_KEY"] = api_key
                    break
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (env or .env)")

    # Stage args + metadata to a temp blob so the rest of the script doesn't
    # need to import inspect_ai (and so we have a checkpoint if interrupted).
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        blob_path = f.name
    print(f"Loading eval log → {blob_path}", flush=True)
    load_eval_argument_blob(args.eval_log, blob_path)
    blob = json.loads(Path(blob_path).read_text())
    samples = blob["samples"]
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"Will process {len(samples)} samples", flush=True)

    # Strip OpenRouter prefix for the suspect model — the probe uses raw
    # OpenRouter model id (no `openrouter/` prefix).
    suspect_model = args.suspect_model
    if suspect_model.startswith("openrouter/"):
        suspect_model = suspect_model.removeprefix("openrouter/")

    out_records: list[dict] = []
    for i, s in enumerate(samples):
        print(
            f"[{i+1}/{len(samples)}] sample_id={s['sample_id']} "
            f"deceptive={s['is_deceptive']}",
            flush=True,
        )
        verdicts = score_argument_per_sentence(
            question_block=s["question_block"],
            argument=s["argument"],
            suspect_model=suspect_model,
            api_key=api_key,
            max_tokens=args.max_tokens,
            log_progress=True,
        )
        scores = [v.score for v in verdicts]
        pooled = {m: _pool(scores, m) for m in ("mean", "max", "last")}
        rec = {
            "sample_id": s["sample_id"],
            "is_deceptive": s["is_deceptive"],
            "argue_target": s["argue_target"],
            "argue_target_text": s["argue_target_text"],
            "n_sentences": len(verdicts),
            "verdicts": serialize(verdicts),
            "pooled": pooled,
        }
        out_records.append(rec)
        # Checkpoint after every sample (cheap; resumable).
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "eval_log": args.eval_log,
                    "suspect_model": args.suspect_model,
                    "samples": out_records,
                    "AUROCs": _aurocs(out_records),
                },
                indent=2,
            )
        )

    Path(blob_path).unlink(missing_ok=True)
    print(f"\nDone. Wrote {args.out}", flush=True)
    print(f"  AUROCs: {_aurocs(out_records)}", flush=True)


def _aurocs(records: list[dict]) -> dict[str, float]:
    labels = [1 if r["is_deceptive"] else 0 for r in records]
    if len(set(labels)) < 2:
        return {}
    out: dict[str, float] = {}
    for mode in ("mean", "max", "last"):
        scores = [r["pooled"][mode] for r in records]
        out[mode] = float(roc_auc_score(labels, scores))
    return out


if __name__ == "__main__":
    main()
