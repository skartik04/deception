"""Analyze per-sentence probe results vs whole-argument behavioral baseline.

Reads:
  - results/per_sentence/main_run.json          (per-sentence probe)
  - results/per_sentence/whole_arg_baseline.json (whole-arg baseline)
  - results/per_sentence/groundtruth.json        (Claude judge, optional)

Reports:
  1. Sample-level AUROCs side-by-side
  2. Combined-score AUROC (max of per-sentence-max, whole-arg)
  3. Per-sentence locality stats:
       - confusion matrix per_sentence_probe vs ground truth (when available)
       - rate at which probe correctly identifies FALSE-by-Claude sentences in
         deceptive samples (locality TPR)
  4. Verdict distribution by sample-label (TRUE/FALSE/INSUFFICIENT counts)

Output: results/per_sentence/analysis.json + console summary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import roc_auc_score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default="results/per_sentence/main_run.json")
    p.add_argument(
        "--continuous",
        default="results/per_sentence/continuous_run.json",
        help="Optional continuous-score variant — included if file exists",
    )
    p.add_argument("--baseline", default="results/per_sentence/whole_arg_baseline.json")
    p.add_argument("--groundtruth", default="results/per_sentence/groundtruth.json")
    p.add_argument("--out", default="results/per_sentence/analysis.json")
    args = p.parse_args()

    probe = json.loads(Path(args.probe).read_text())
    base = json.loads(Path(args.baseline).read_text())
    gt_path = Path(args.groundtruth)
    gt = json.loads(gt_path.read_text()) if gt_path.exists() else None
    cont_path = Path(args.continuous)
    cont = (
        json.loads(cont_path.read_text())
        if cont_path.exists() and cont_path.stat().st_size > 0
        else None
    )

    # ---------------------------------------------------------------
    # 1. Sample-level AUROCs
    # ---------------------------------------------------------------
    probe_by_id = {s["sample_id"]: s for s in probe["samples"]}
    base_by_id = {s["sample_id"]: s for s in base["samples"]}
    common = sorted(set(probe_by_id) & set(base_by_id))
    labels = [1 if probe_by_id[i]["is_deceptive"] else 0 for i in common]

    psh_mean = [probe_by_id[i]["pooled"]["mean"] for i in common]
    psh_max = [probe_by_id[i]["pooled"]["max"] for i in common]
    psh_last = [probe_by_id[i]["pooled"]["last"] for i in common]
    base_scores = [base_by_id[i]["score"] for i in common]
    combined = [max(probe_by_id[i]["pooled"]["max"], base_by_id[i]["score"]) for i in common]

    def _auroc(scores: list[float]) -> float:
        return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")

    aurocs = {
        "per_sentence_ternary_mean": _auroc(psh_mean),
        "per_sentence_ternary_max": _auroc(psh_max),
        "per_sentence_ternary_last": _auroc(psh_last),
        "whole_arg_baseline": _auroc(base_scores),
        "combined_max(ternary_max, whole_arg)": _auroc(combined),
    }
    if cont is not None:
        cont_by_id = {s["sample_id"]: s for s in cont["samples"]}
        cont_common = [i for i in common if i in cont_by_id]
        if len(cont_common) >= 2:
            cont_labels = [
                1 if probe_by_id[i]["is_deceptive"] else 0 for i in cont_common
            ]
            cont_mean = [cont_by_id[i]["pooled"]["mean"] for i in cont_common]
            cont_max = [cont_by_id[i]["pooled"]["max"] for i in cont_common]
            cont_last = [cont_by_id[i]["pooled"]["last"] for i in cont_common]
            base_for_cont = [base_by_id[i]["score"] for i in cont_common]
            cont_combined = [
                max(cont_by_id[i]["pooled"]["max"], base_by_id[i]["score"])
                for i in cont_common
            ]

            def _auroc_c(scores: list[float]) -> float:
                return float(roc_auc_score(cont_labels, scores)) if len(set(cont_labels)) == 2 else float("nan")

            aurocs["per_sentence_continuous_mean"] = _auroc_c(cont_mean)
            aurocs["per_sentence_continuous_max"] = _auroc_c(cont_max)
            aurocs["per_sentence_continuous_last"] = _auroc_c(cont_last)
            aurocs["combined_max(continuous_max, whole_arg)"] = _auroc_c(cont_combined)
            aurocs["_continuous_n_samples"] = float(len(cont_common))

    # ---------------------------------------------------------------
    # 2. Verdict distribution by sample-label
    # ---------------------------------------------------------------
    verdict_counts: dict[str, Counter] = {
        "honest_arguments": Counter(),
        "deceptive_arguments": Counter(),
    }
    for i in common:
        s = probe_by_id[i]
        bucket = "deceptive_arguments" if s["is_deceptive"] else "honest_arguments"
        for v in s["verdicts"]:
            verdict_counts[bucket][v["verdict"]] += 1

    # ---------------------------------------------------------------
    # 3. Per-sentence locality (vs Claude ground truth)
    # ---------------------------------------------------------------
    locality: dict[str, object] = {}
    if gt is not None:
        # Confusion matrix: probe verdict vs Claude verdict
        conf: Counter = Counter()
        false_recall_num = 0  # probe correctly flagged Claude-FALSE
        false_recall_den = 0
        true_specificity_num = 0  # probe correctly didn't flag Claude-TRUE
        true_specificity_den = 0
        for s in gt["samples"]:
            for r in s["per_sentence"]:
                pv = r["probe_verdict"]
                gv = r["groundtruth_verdict"]
                conf[(pv, gv)] += 1
                if gv == "FALSE":
                    false_recall_den += 1
                    if pv == "FALSE":
                        false_recall_num += 1
                elif gv == "TRUE":
                    true_specificity_den += 1
                    if pv != "FALSE":
                        true_specificity_num += 1
        locality = {
            "confusion_probe_vs_gt": {f"{p}|{g}": n for (p, g), n in conf.items()},
            "false_recall": (
                false_recall_num / false_recall_den if false_recall_den else None
            ),
            "true_specificity": (
                true_specificity_num / true_specificity_den
                if true_specificity_den
                else None
            ),
            "n_sentences_evaluated": sum(conf.values()),
            "judge_model": gt.get("judge_model"),
        }

    out = {
        "n_samples": len(common),
        "AUROCs": aurocs,
        "verdict_distribution": {k: dict(v) for k, v in verdict_counts.items()},
        "locality": locality,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"=== Analysis (n={len(common)}) ===")
    print("\nAUROCs (sample level):")
    for k, v in aurocs.items():
        print(f"  {k:<40} {v:.4f}")
    print("\nVerdict distribution (per-sentence):")
    for bucket, c in verdict_counts.items():
        total = sum(c.values())
        print(f"  {bucket} (n={total} sentences):")
        for v, n in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"    {v:<22} {n:>4}  ({100*n/max(total,1):.1f}%)")
    if locality:
        print("\nPer-sentence locality (probe vs Claude judge):")
        print(
            f"  false_recall   (probe says FALSE | Claude says FALSE): "
            f"{locality.get('false_recall')}"
        )
        print(
            f"  true_specificity (probe NOT FALSE | Claude says TRUE): "
            f"{locality.get('true_specificity')}"
        )
        print(f"  n sentences evaluated: {locality.get('n_sentences_evaluated')}")
    print(f"\nSaved analysis to {args.out}")


if __name__ == "__main__":
    main()
