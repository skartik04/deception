#!/usr/bin/env python3
"""Pivot eval log statistics by metadata variables.

Usage:
    uv run python scripts/pivot_eval.py <eval_file> [--pivot key1,key2,...]

Examples:
    # Split by honest vs deceptive:
    uv run python scripts/pivot_eval.py logs/curated/single-turn_oracle-detector_gemma27b-judges_llama70b-argues_n200.eval --pivot is_deceptive

    # Split by multiple variables:
    uv run python scripts/pivot_eval.py some.eval --pivot is_deceptive,assigned_letter

    # No pivot — just show overall scores:
    uv run python scripts/pivot_eval.py some.eval
"""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path


def load_samples(eval_path: Path) -> list[dict[str, object]]:
    """Load all samples from an .eval zip file."""
    samples = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if name.startswith("samples/") and name.endswith(".json"):
                with zf.open(name) as sf:
                    samples.append(json.load(sf))
    return samples


def score_to_numeric(value: object) -> float | None:
    """Convert a score value to a number for averaging."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        mapping = {"C": 1.0, "I": 0.0, "N": 0.0, "P": 0.5}
        return mapping.get(value)
    return None


def format_pct(val: float, n: int) -> str:
    """Format a value as percentage with count."""
    if val != val:  # NaN
        return f"  NaN (n={n})"
    return f"{val * 100:5.1f}% (n={n})"


def print_table(
    rows: list[dict[str, str]],
    columns: list[str],
    title: str,
    fmt: str = "markdown",
) -> None:
    """Print a formatted table (markdown or ascii)."""
    if fmt == "markdown":
        print(f"\n### {title}\n")
        print("| " + " | ".join(columns) + " |")
        print("|" + "|".join("---" for _ in columns) + "|")
        for row in rows:
            print("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    else:
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
        sep = "+-" + "-+-".join("-" * widths[c] for c in columns) + "-+"
        header = "| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |"
        print(f"\n{'=' * len(sep)}")
        print(f" {title}")
        print(sep)
        print(header)
        print(sep)
        for row in rows:
            line = "| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns) + " |"
            print(line)
        print(sep)


def analyze(eval_path: Path, pivot_keys: list[str], fmt: str = "markdown") -> None:
    """Analyze an eval file, printing per-score tables pivoted by metadata keys."""
    samples = load_samples(eval_path)
    if not samples:
        print(f"No samples found in {eval_path}")
        return

    # Discover score names and metadata
    first_scores = samples[0].get("scores")
    assert isinstance(first_scores, dict)
    score_names = list(first_scores.keys())
    first_meta = samples[0].get("metadata")
    assert isinstance(first_meta, dict)
    available_meta = set(first_meta.keys())

    print(f"File: {eval_path.name}")
    print(f"Samples: {len(samples)}")
    print(f"Scores: {', '.join(score_names)}")
    print(f"Available metadata keys: {', '.join(sorted(available_meta))}")

    bad_keys = [k for k in pivot_keys if k not in available_meta]
    if bad_keys:
        print(f"WARNING: pivot keys not in metadata: {bad_keys}")
        pivot_keys = [k for k in pivot_keys if k in available_meta]

    # Build a composite pivot key per sample
    def get_meta(sample: dict[str, object]) -> dict[str, object]:
        meta = sample.get("metadata")
        assert isinstance(meta, dict)
        return meta

    def pivot_label(sample: dict[str, object]) -> str:
        meta = get_meta(sample)
        if not pivot_keys:
            return "all"
        parts = []
        for k in pivot_keys:
            v = meta.get(k)
            if isinstance(v, bool):
                parts.append(f"{k}={'yes' if v else 'no'}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    # Collect per-group scores
    for score_name in score_names:
        groups: dict[str, list[float]] = defaultdict(list)
        value_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for sample in samples:
            label = pivot_label(sample)
            all_scores = sample.get("scores")
            assert isinstance(all_scores, dict)
            score_data = all_scores.get(score_name, {})
            assert isinstance(score_data, dict)
            raw_value = score_data.get("value")
            numeric = score_to_numeric(raw_value)

            # Track raw value distribution
            value_counts[label][str(raw_value)] = value_counts[label].get(str(raw_value), 0) + 1

            if numeric is not None:
                groups[label].append(numeric)

        # Build table rows
        rows: list[dict[str, str]] = []
        columns = ["group", "accuracy", "n", "C", "I", "P", "N"]

        for label in sorted(groups.keys()):
            vals = groups[label]
            counts = value_counts[label]
            n_total = sum(value_counts[label].values())
            avg = sum(vals) / len(vals) if vals else float("nan")

            rows.append({
                "group": label,
                "accuracy": f"{avg * 100:.1f}%",
                "n": str(n_total),
                "C": str(counts.get("C", 0)),
                "I": str(counts.get("I", 0)),
                "P": str(counts.get("0.5", 0)),
                "N": str(counts.get("N", 0)),
            })

        # Add total row
        all_vals = [v for vs in groups.values() for v in vs]
        all_counts: dict[str, int] = defaultdict(int)
        for vc in value_counts.values():
            for k, c in vc.items():
                all_counts[k] += c
        n_all = sum(all_counts.values())
        avg_all = sum(all_vals) / len(all_vals) if all_vals else float("nan")
        rows.append({
            "group": "TOTAL",
            "accuracy": f"{avg_all * 100:.1f}%",
            "n": str(n_all),
            "C": str(all_counts.get("C", 0)),
            "I": str(all_counts.get("I", 0)),
            "P": str(all_counts.get("0.5", 0)),
            "N": str(all_counts.get("N", 0)),
        })

        print_table(rows, columns, f"Score: {score_name}", fmt=fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pivot eval statistics by metadata")
    parser.add_argument("eval_file", type=Path, help="Path to .eval file")
    parser.add_argument(
        "--pivot", type=str, default="",
        help="Comma-separated metadata keys to split by (e.g. is_deceptive,assigned_letter)",
    )
    parser.add_argument(
        "--fmt", type=str, default="markdown", choices=["markdown", "ascii"],
        help="Output format: 'markdown' (default) or 'ascii'",
    )
    args = parser.parse_args()

    pivot_keys = [k.strip() for k in args.pivot.split(",") if k.strip()]
    analyze(args.eval_file, pivot_keys, fmt=args.fmt)


if __name__ == "__main__":
    main()
