"""Final evaluation of the trained per-sentence behavioral probe.

Reads the cached activations + labels and reports:
  - Sample-level AUROC (mean / max / last pooling of per-sentence sigmoid scores)
  - Per-sentence locality vs Claude-haiku ground truth (FALSE-recall, TRUE-specificity)
  - Side-by-side comparison with the raw behavioral probe (the labels themselves)
    and the whole-arg baseline.

Output: results/per_sentence/trained_probe_eval.json + console summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

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
    p.add_argument("--baseline", default="results/per_sentence/whole_arg_baseline.json")
    p.add_argument("--groundtruth", default="results/per_sentence/groundtruth.json")
    p.add_argument("--out", default="results/per_sentence/trained_probe_eval.json")
    args = p.parse_args()

    summary = json.loads(Path(args.probe_summary).read_text())
    best_layer = summary["best_layer"]

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

    # Per-sentence trained-probe scores.
    by_sample: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for r in per_sentence:
        v = _decode(r, best_layer)
        std = (v - scaler_mean) / scaler_scale
        logit = float(std @ direction)
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        by_sample[str(r["sample_id"])].append(
            (int(r["sentence_idx"]), float(r["label"]), prob)
        )

    labels = json.loads(Path(args.labels).read_text())
    base = json.loads(Path(args.baseline).read_text())
    base_by_id = {str(s["sample_id"]): s for s in base["samples"]}

    # ----------------------- Sample-level AUROC -----------------------
    samples = labels["samples"]
    is_d_by_id = {str(s["sample_id"]): bool(s["is_deceptive"]) for s in samples}
    common = sorted(set(by_sample.keys()) & set(is_d_by_id.keys()))
    y = [1 if is_d_by_id[i] else 0 for i in common]

    pooled: dict[str, dict[str, list[float]]] = {
        "trained_probe": defaultdict(list),
        "behavioral_labels": defaultdict(list),
    }
    for sid in common:
        scores = [x[2] for x in by_sample[sid]]
        bel = [x[1] for x in by_sample[sid]]  # behavioral labels (raw target)
        for mode, fn in (
            ("mean", np.mean),
            ("max", np.max),
            ("last", lambda v: v[-1]),
        ):
            pooled["trained_probe"][mode].append(float(fn(scores)) if scores else 0.5)
            pooled["behavioral_labels"][mode].append(float(fn(bel)) if bel else 0.5)

    sample_aurocs: dict[str, dict[str, float]] = {}
    for probe_name, mode_scores in pooled.items():
        sample_aurocs[probe_name] = {}
        for mode, sc in mode_scores.items():
            try:
                sample_aurocs[probe_name][mode] = float(roc_auc_score(y, sc))
            except ValueError:
                sample_aurocs[probe_name][mode] = float("nan")

    # Whole-arg baseline AUROC on the same samples.
    base_scores = [base_by_id[i]["score"] for i in common if i in base_by_id]
    base_y = [
        1 if is_d_by_id[i] else 0 for i in common if i in base_by_id
    ]
    if len(set(base_y)) == 2:
        sample_aurocs["whole_arg_baseline"] = {
            "single": float(roc_auc_score(base_y, base_scores))
        }

    # ----------------------- Per-sentence locality vs Claude-haiku -----------------------
    locality: dict[str, object] = {}
    gt_path = Path(args.groundtruth)
    if gt_path.exists():
        gt = json.loads(gt_path.read_text())
        # Build sentence-level ground-truth labels
        ts: list[tuple[float, str]] = []  # (probe_prob, gt_verdict)
        for s in gt["samples"]:
            sid = str(s["sample_id"])
            sc_by_idx = {idx: prob for idx, _, prob in by_sample.get(sid, [])}
            for r in s["per_sentence"]:
                idx = r["sentence_idx"]
                gv = r["groundtruth_verdict"]
                if idx not in sc_by_idx:
                    continue
                ts.append((sc_by_idx[idx], gv))
        n_evald = len(ts)
        # Probe says FALSE if prob > 0.5; TRUE otherwise
        false_recall_n = 0
        false_recall_d = 0
        true_spec_n = 0
        true_spec_d = 0
        for prob, gv in ts:
            probe_says_false = prob > 0.5
            if gv == "FALSE":
                false_recall_d += 1
                if probe_says_false:
                    false_recall_n += 1
            elif gv == "TRUE":
                true_spec_d += 1
                if not probe_says_false:
                    true_spec_n += 1
        # Also: AUROC at the per-sentence level using Claude's TRUE/FALSE (drop NEUTRAL/PARSE_FAIL)
        binary = [(p, 1 if g == "FALSE" else 0) for (p, g) in ts if g in ("TRUE", "FALSE")]
        if binary:
            yb = [b for (_, b) in binary]
            sb = [p for (p, _) in binary]
            try:
                per_sent_auroc = float(roc_auc_score(yb, sb))
            except ValueError:
                per_sent_auroc = float("nan")
        else:
            per_sent_auroc = float("nan")
        locality = {
            "n_sentences_evaluated": n_evald,
            "false_recall": false_recall_n / false_recall_d if false_recall_d else None,
            "true_specificity": true_spec_n / true_spec_d if true_spec_d else None,
            "per_sentence_auroc": per_sent_auroc,
            "judge_model": gt.get("judge_model"),
        }

    out = {
        "best_layer": best_layer,
        "n_samples": len(common),
        "sample_AUROCs": sample_aurocs,
        "trained_probe_per_sentence_locality": locality,
        "training_summary": summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"=== Trained-probe eval (layer={best_layer}, n={len(common)}) ===\n")
    print("Sample-level AUROC:")
    for probe_name, modes in sample_aurocs.items():
        print(f"  {probe_name}:")
        for m, v in modes.items():
            print(f"    {m:<10} {v:.4f}")
    if locality:
        print("\nPer-sentence locality vs Claude-haiku (n="
              f"{locality.get('n_sentences_evaluated')}):")
        print(f"  false_recall    = {locality.get('false_recall')}")
        print(f"  true_specificity = {locality.get('true_specificity')}")
        print(f"  per_sentence_AUROC = {locality.get('per_sentence_auroc')}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
