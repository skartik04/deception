"""Plot per-sentence probe verdicts side-by-side with Claude-haiku ground truth.

For each sample in `groundtruth.json`, render two stacked rows of small
colored squares — top row = probe verdicts, bottom row = ground-truth
verdicts. Highlights agreement / disagreement at sentence granularity.
This is the locality eval visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


VERDICT_COLOR = {
    "TRUE": "tab:blue",
    "T": "tab:blue",
    "FALSE": "tab:red",
    "F": "tab:red",
    "INSUFFICIENT_INFO": "tab:gray",
    "INSUFFICIENT": "tab:gray",
    "I": "tab:gray",
    "NEUTRAL": "tab:gray",
    "N": "tab:gray",
    "PARSE_FAIL": "black",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="results/per_sentence/groundtruth.json")
    p.add_argument("--probe", default="results/per_sentence/main_run.json")
    p.add_argument("--out", default="present/per_sentence_probe/probe_vs_gt.png")
    args = p.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    probe = json.loads(Path(args.probe).read_text())
    probe_by_id = {s["sample_id"]: s for s in probe["samples"]}

    samples = gt["samples"]
    n_samples = len(samples)
    if n_samples == 0:
        print("No GT samples to plot")
        return
    max_sents = max(s["n_sentences"] for s in samples)

    fig_h = max(2.5 + n_samples * 0.95, 4.0)
    fig_w = max(8.0, max_sents * 0.36 + 4.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    cell_size = 0.9
    pad = 0.1
    row_y = {}
    for r, s in enumerate(samples):
        sid = s["sample_id"]
        argue = s["argue_target"]
        pooled = probe_by_id[sid]["pooled"]["mean"]
        # two rows per sample: probe (upper), GT (lower)
        y_probe = 2 * r + 1
        y_gt = 2 * r
        row_y[sid] = (y_probe, y_gt)
        for j, rec in enumerate(s["per_sentence"]):
            x = j
            pv = rec["probe_verdict"]
            gv = rec["groundtruth_verdict"]
            ax.add_patch(
                Rectangle(
                    (x + pad, y_probe + pad),
                    cell_size,
                    cell_size,
                    color=VERDICT_COLOR.get(pv, "white"),
                    ec="black",
                    linewidth=0.4,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x + pad, y_gt + pad),
                    cell_size,
                    cell_size,
                    color=VERDICT_COLOR.get(gv, "white"),
                    ec="black",
                    linewidth=0.4,
                )
            )
            # Sentence index label every 5 sentences
            if j % 5 == 0:
                ax.text(
                    x + cell_size / 2 + pad,
                    y_probe + cell_size + pad + 0.05,
                    f"S{j+1}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                )

        ax.text(
            -0.6,
            y_probe + cell_size / 2 + pad,
            "probe",
            ha="right",
            va="center",
            fontsize=8,
            color="black",
        )
        ax.text(
            -0.6,
            y_gt + cell_size / 2 + pad,
            "GT",
            ha="right",
            va="center",
            fontsize=8,
            color="black",
        )
        ax.text(
            max_sents + 0.5,
            y_probe + cell_size / 2 + pad,
            f"sample {sid}\n(argue={argue})\nmean={pooled:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            color="black",
        )

    ax.set_xlim(-2, max_sents + 4)
    ax.set_ylim(-0.5, 2 * n_samples + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    legend = [
        Patch(color="tab:blue", label="TRUE / T"),
        Patch(color="tab:red", label="FALSE / F"),
        Patch(color="tab:gray", label="INSUFFICIENT / NEUTRAL"),
        Patch(facecolor="white", edgecolor="black", label="(no data)"),
    ]
    ax.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=10,
        frameon=False,
    )
    fig.suptitle(
        "Per-sentence locality: probe (top row) vs Claude-haiku ground truth (bottom row)\n"
        f"{n_samples} deceptive samples — judge: {gt.get('judge_model', '?')}",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
