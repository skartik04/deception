"""Plot per-token (broadcast from per-sentence) scores for selected samples.

Reads the JSON written by `run_per_sentence_probe.py`. For each sample
selected via `--ids`, generates a horizontal bar plot where:
  - x-axis = token index
  - y-axis = sentence-level deception score (0..1) broadcast to tokens
  - red = score > 0.5 (FALSE / INSUFFICIENT_INFO leaning deceptive)
  - blue = score < 0.5 (TRUE)
  - sentence boundaries marked with dotted vertical lines + sentence index
  - title shows is_deceptive + argue_target + pooled mean/max

Importantly, this script tokenizes with a tokenizer NOT the actual Llama 70B
one (we don't have it locally) — we use a simple regex tokenizer so the
plot is interpretable on this CPU-only pod. Token strings displayed are the
exact substrings from the source argument.

Usage:
    uv run python scripts/plot_per_sentence.py results/per_sentence/main_run.json \
        --ids 1 2 3 --out-dir present/per_sentence_probe
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Simple tokenizer: keep words and punctuation as separate tokens.
_TOK_RE = re.compile(r"\S+")


def tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for m in _TOK_RE.finditer(text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def plot_sample(
    *,
    sample: dict,
    out_path: Path,
    probe_title: str = "per-sentence behavioral probe",
) -> None:
    """sample shape from run_per_sentence_probe output."""
    # Reconstruct argument from concatenating sentences (preserves original text;
    # see lib.per_sentence_behavioral.split_sentences).
    verdicts = sample["verdicts"]
    if not verdicts:
        print(f"  sample {sample['sample_id']}: no verdicts, skipping")
        return
    # Boundaries are stored as char offsets into the original argument; the
    # original argument isn't here, so reconstruct it as the concatenated
    # sentence text with single spaces (close enough for visualization).
    # We keep per-sentence offsets relative to that reconstruction.
    rebuilt = ""
    sent_spans: list[tuple[int, int]] = []
    sent_scores: list[float] = []
    sent_verdicts: list[str] = []
    for v in verdicts:
        s = v["sentence"]
        if rebuilt:
            rebuilt += " "
        start = len(rebuilt)
        rebuilt += s
        end = len(rebuilt)
        sent_spans.append((start, end))
        sent_scores.append(v["score"])
        sent_verdicts.append(v["verdict"])

    tokens, tok_offsets = tokenize(rebuilt)
    per_tok_scores: list[float] = []
    per_tok_sentence_idx: list[int] = []
    for (ts, te) in tok_offsets:
        score = 0.0
        sidx = -1
        for i, (ss, se) in enumerate(sent_spans):
            if ts < se and te > ss:
                score = sent_scores[i]
                sidx = i
                break
        per_tok_scores.append(score)
        per_tok_sentence_idx.append(sidx)

    # Wrap into rows of TOKS_PER_ROW tokens each so the figure stays readable.
    TOKS_PER_ROW = 60
    n = len(tokens)
    n_rows = (n + TOKS_PER_ROW - 1) // TOKS_PER_ROW
    fig_h = max(2.5 * n_rows + 1.0, 4.0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(18.0, fig_h), squeeze=False)

    for row in range(n_rows):
        ax = axes[row][0]
        lo = row * TOKS_PER_ROW
        hi = min(n, lo + TOKS_PER_ROW)
        seg_scores = per_tok_scores[lo:hi]
        seg_tokens = tokens[lo:hi]
        seg_sidx = per_tok_sentence_idx[lo:hi]
        x = list(range(hi - lo))
        # Colored background strip: full height, categorical color, so TRUE
        # (score 0) sentences are still visible. Bar height encodes the actual
        # score on top, in a darker shade.
        bg_colors = [
            "#fbb4b4" if s > 0.5 else "#bbd9f5" if s < 0.5 else "#d9d9d9"
            for s in seg_scores
        ]
        fg_colors = [
            "tab:red" if s > 0.5 else "tab:blue" if s < 0.5 else "tab:gray"
            for s in seg_scores
        ]
        # Background: full-height strip
        ax.bar(x, [1.0] * len(x), color=bg_colors, width=1.0, alpha=0.6)
        # Foreground: actual score height
        ax.bar(x, seg_scores, color=fg_colors, width=0.85)
        ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 1.15)
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.set_ylabel("score", fontsize=9)

        # Sentence-boundary markers + verdict label per sentence
        last_sidx = -1
        for i, sidx in enumerate(seg_sidx):
            if sidx != last_sidx:
                ax.axvline(i - 0.5, color="black", linewidth=0.4, alpha=0.25)
                if sidx >= 0:
                    v = sent_verdicts[sidx]
                    color = {"TRUE": "tab:blue", "FALSE": "tab:red"}.get(v, "tab:gray")
                    ax.text(
                        i + 0.2,
                        1.05,
                        f"S{sidx+1}:{v[:1]}",
                        fontsize=8,
                        ha="left",
                        va="bottom",
                        color=color,
                        weight="bold",
                    )
                last_sidx = sidx

        ax.set_xticks(x)
        ax.set_xticklabels(
            [t if len(t) <= 12 else t[:11] + "…" for t in seg_tokens],
            rotation=80,
            fontsize=7.5,
            family="monospace",
        )
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    target = sample["argue_target"]
    is_d = sample["is_deceptive"]
    pooled = sample["pooled"]
    label = "DECEPTIVE" if is_d else "HONEST"
    title_color = "tab:red" if is_d else "tab:blue"
    fig.suptitle(
        f"sample {sample['sample_id']}  —  argue_target = {target}  —  {label} argument\n"
        f"{probe_title}  —  pooled mean = {pooled['mean']:.2f}   "
        f"max = {pooled['max']:.2f}   last = {pooled['last']:.2f}",
        fontsize=13,
        color=title_color,
        weight="bold",
    )
    # Legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color="tab:red", label="FALSE (per-sentence verdict, score=1.0)"),
        Patch(color="tab:gray", label="INSUFFICIENT_INFO (score=0.5)"),
        Patch(color="tab:blue", label="TRUE (score=0.0)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("json_path")
    p.add_argument("--ids", nargs="*", default=[])
    p.add_argument("--n-honest", type=int, default=5)
    p.add_argument("--n-deceptive", type=int, default=5)
    p.add_argument("--out-dir", default="present/per_sentence_probe")
    args = p.parse_args()

    blob = json.loads(Path(args.json_path).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected: list[dict] = []
    if args.ids:
        wanted = set(map(str, args.ids))
        selected = [s for s in blob["samples"] if str(s["sample_id"]) in wanted]
    else:
        honest = [s for s in blob["samples"] if not s["is_deceptive"]]
        decept = [s for s in blob["samples"] if s["is_deceptive"]]
        selected = honest[: args.n_honest] + decept[: args.n_deceptive]

    for s in selected:
        label = "deceptive" if s["is_deceptive"] else "honest"
        path = out_dir / f"per_sentence_sample{s['sample_id']}_{label}_argue{s['argue_target']}.png"
        plot_sample(sample=s, out_path=path)


if __name__ == "__main__":
    main()
