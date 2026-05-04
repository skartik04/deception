"""Plot the trained linear probe's per-sentence predictions, broadcast to tokens.

Reuses the cached activations from `train_per_sentence_probe.py` to score
each sentence with the trained probe (no Flash needed at this stage), then
plots per-token bars in the same format as `plot_per_sentence.py`.

Usage:
    uv run python scripts/plot_trained_probe.py \
      --cache results/per_sentence/activations_cache.npz \
      --probe-summary probes/per_sentence_behavioral.summary.json \
      --probe probes/per_sentence_behavioral.pt \
      --labels results/per_sentence/main_run.json \
      --eval-log logs/...eval \
      --out-dir present/per_sentence_probe/trained
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "liars-bench" / "src" / "probes"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/per_sentence/activations_cache.npz")
    p.add_argument("--probe", default="probes/per_sentence_behavioral.pt")
    p.add_argument(
        "--probe-summary", default="probes/per_sentence_behavioral.summary.json"
    )
    p.add_argument("--labels", default="results/per_sentence/main_run.json")
    p.add_argument("--out-dir", default="present/per_sentence_probe/trained")
    p.add_argument("--n-honest", type=int, default=5)
    p.add_argument("--n-deceptive", type=int, default=5)
    args = p.parse_args()

    summary = json.loads(Path(args.probe_summary).read_text())
    best_layer = summary["best_layer"]
    print(f"Loading probe at layer {best_layer}", flush=True)

    from deception_detection.detectors import LogisticRegressionDetector

    det = LogisticRegressionDetector.load(args.probe)
    assert det.directions is not None
    direction = det.directions.squeeze(0).float().numpy()
    assert det.scaler_mean is not None
    scaler_mean = det.scaler_mean.squeeze(0).float().numpy()
    assert det.scaler_scale is not None
    scaler_scale = det.scaler_scale.squeeze(0).float().numpy()

    cache = np.load(args.cache, allow_pickle=True)
    per_sentence: list[dict] = list(cache["per_sentence"])

    import base64

    def _decode(r: dict, layer: int) -> np.ndarray:
        if "activations_b64_f16" in r:
            buf = base64.b64decode(r["activations_b64_f16"][str(layer)])
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        return np.array(r["activations"][str(layer)], dtype=np.float32)

    # Score each sentence using the probe (sigmoid of scaled-direction-dot-act).
    scores_by_sample: dict[str, list[tuple[int, float]]] = {}
    for r in per_sentence:
        sid = str(r["sample_id"])
        sidx = int(r["sentence_idx"])
        v = _decode(r, best_layer)
        std = (v - scaler_mean) / scaler_scale
        logit = float(std @ direction)
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        scores_by_sample.setdefault(sid, []).append((sidx, prob))

    # Pull sentence text + sample metadata from the labels JSON.
    labels = json.loads(Path(args.labels).read_text())
    by_id = {str(s["sample_id"]): s for s in labels["samples"]}

    # Build records the existing plot_per_sentence.plot_sample expects.
    out_records: list[dict] = []
    for sid, scores in scores_by_sample.items():
        meta = by_id[sid]
        verdicts = meta["verdicts"]
        score_by_idx = {sidx: sc for sidx, sc in scores}
        # Build per-sentence records — only sentences with activations (drop INSUFFICIENT).
        new_verdicts: list[dict] = []
        for i, v in enumerate(verdicts):
            sc = score_by_idx.get(i)
            if sc is None:
                # Sentences excluded from training (label==0.5) — show as gray
                new_verdicts.append(
                    {
                        "sentence": v["sentence"],
                        "start": v["start"],
                        "end": v["end"],
                        "verdict": "INSUFFICIENT_INFO",
                        "score": 0.5,
                    }
                )
                continue
            verdict = "FALSE" if sc > 0.5 else "TRUE"
            new_verdicts.append(
                {
                    "sentence": v["sentence"],
                    "start": v["start"],
                    "end": v["end"],
                    "verdict": verdict,
                    "score": sc,
                }
            )
        # pooled
        nums = [v["score"] for v in new_verdicts]
        pooled = {
            "mean": float(np.mean(nums)) if nums else 0.5,
            "max": float(np.max(nums)) if nums else 0.5,
            "last": float(nums[-1]) if nums else 0.5,
        }
        out_records.append(
            {
                "sample_id": int(sid) if str(sid).isdigit() else sid,
                "is_deceptive": meta["is_deceptive"],
                "argue_target": meta["argue_target"],
                "verdicts": new_verdicts,
                "pooled": pooled,
            }
        )

    # Reuse plot_per_sentence's plot_sample
    from scripts.plot_per_sentence import plot_sample

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    honest = [r for r in out_records if not r["is_deceptive"]][: args.n_honest]
    decept = [r for r in out_records if r["is_deceptive"]][: args.n_deceptive]
    for r in honest + decept:
        label = "deceptive" if r["is_deceptive"] else "honest"
        path = (
            out_dir
            / f"trained_probe_sample{r['sample_id']}_{label}_argue{r['argue_target']}.png"
        )
        plot_sample(
            sample=r,
            out_path=path,
            probe_title=f"trained linear probe (Llama 70B layer {best_layer}, sigmoid)",
        )


if __name__ == "__main__":
    main()
