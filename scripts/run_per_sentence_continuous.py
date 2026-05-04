"""Run the continuous-score variant of the per-sentence probe.

Output schema mirrors `run_per_sentence_probe.py` but verdicts have a
`score` only (0..1, higher = more deceptive).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.per_sentence_behavioral import load_eval_argument_blob
from lib.per_sentence_continuous import score_argument_continuous, serialize


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
    p.add_argument("eval_log")
    p.add_argument(
        "--suspect-model",
        default="openrouter/meta-llama/llama-3.3-70b-instruct:nitro",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="results/per_sentence/continuous_run.json")
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        envfile = REPO / ".env"
        if envfile.exists():
            for line in envfile.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    os.environ["OPENROUTER_API_KEY"] = api_key
                    break
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        blob_path = f.name
    load_eval_argument_blob(args.eval_log, blob_path)
    blob = json.loads(Path(blob_path).read_text())
    samples = blob["samples"]
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"Will process {len(samples)} samples (continuous variant)", flush=True)

    suspect_model = args.suspect_model
    if suspect_model.startswith("openrouter/"):
        suspect_model = suspect_model.removeprefix("openrouter/")

    out_records: list[dict] = []
    aurocs: dict[str, float] = {}
    for i, s in enumerate(samples):
        print(
            f"[{i+1}/{len(samples)}] sample_id={s['sample_id']} "
            f"deceptive={s['is_deceptive']}",
            flush=True,
        )
        verdicts = score_argument_continuous(
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
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        labels = [1 if r["is_deceptive"] else 0 for r in out_records]
        aurocs = {}
        if len(set(labels)) == 2:
            for mode in ("mean", "max", "last"):
                aurocs[mode] = float(
                    roc_auc_score(labels, [r["pooled"][mode] for r in out_records])
                )
        out_path.write_text(
            json.dumps(
                {
                    "eval_log": args.eval_log,
                    "suspect_model": args.suspect_model,
                    "samples": out_records,
                    "AUROCs": aurocs,
                },
                indent=2,
            )
        )
    Path(blob_path).unlink(missing_ok=True)
    print(f"\nDone. AUROCs: {aurocs}", flush=True)


if __name__ == "__main__":
    main()
