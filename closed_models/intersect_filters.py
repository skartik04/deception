"""Intersect suspect and judge filter JSONs per subset.

Finds questions where:
  - Suspect (Llama 70B) knows the answer: correct_count >= suspect_threshold
  - Judge (Gemma 27B) doesn't know the answer: correct_count <= judge_threshold

Reads the per-subset JSONs produced by run_filter.py and writes one
intersected JSON per subset into results/. The output format matches what
gergely_deception's expanded_dataset_filtered() expects.

Usage:
    uv run python -m closed_models.intersect_filters [--dry-run]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import PROJECT_ROOT, load_config


def sanitize_model_name(model: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]", "-", model).strip("-")


def find_filter_json(role: str, model: str, subset: str, threshold: int) -> Path | None:
    """Find the filter JSON for a given role/model/subset/threshold."""
    safe_model = sanitize_model_name(model)
    path = PROJECT_ROOT / "results" / f"{role}_{safe_model}_{subset}_thresh{threshold}.json"
    return path if path.exists() else None


def intersect_filters(dry_run: bool = False) -> None:
    config = load_config()
    suspect_model = config.suspect.model
    judge_model = config.judge.model
    suspect_threshold = config.suspect.threshold
    judge_threshold = config.judge.threshold
    subsets = config.dataset.subsets

    print(f"Suspect: {suspect_model} (threshold >= {suspect_threshold}/10)")
    print(f"Judge:   {judge_model} (threshold <= {judge_threshold}/10)")
    print()

    safe_suspect = sanitize_model_name(suspect_model)
    safe_judge = sanitize_model_name(judge_model)

    total_suspect = 0
    total_judge = 0
    total_intersect = 0

    for subset in subsets:
        suspect_path = find_filter_json("suspect", suspect_model, subset, suspect_threshold)
        judge_path = find_filter_json("judge", judge_model, subset, judge_threshold)

        if suspect_path is None:
            print(f"[{subset}] MISSING suspect filter: run_filter --role suspect first")
            continue
        if judge_path is None:
            print(f"[{subset}] MISSING judge filter: run_filter --role judge first")
            continue

        suspect_data = json.loads(suspect_path.read_text())
        judge_data = json.loads(judge_path.read_text())

        suspect_ids = set(suspect_data["qualifying_ids"])
        judge_ids = set(judge_data["qualifying_ids"])
        intersect_ids = sorted(suspect_ids & judge_ids)

        total_suspect += len(suspect_ids)
        total_judge += len(judge_ids)
        total_intersect += len(intersect_ids)

        print(f"[{subset}]")
        print(f"  Suspect qualifying: {len(suspect_ids)}")
        print(f"  Judge qualifying:   {len(judge_ids)}")
        print(f"  Intersection:       {len(intersect_ids)}")

        if dry_run:
            print(f"  (dry-run, not writing)")
            print()
            continue

        out = {
            "suspect_model": suspect_model,
            "judge_model": judge_model,
            "suspect_threshold": suspect_threshold,
            "judge_threshold": judge_threshold,
            "dataset": {"subsets": [subset]},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "qualifying_ids": intersect_ids,
            "qualifying_count": len(intersect_ids),
            "stats": {
                "suspect_qualifying": len(suspect_ids),
                "judge_qualifying": len(judge_ids),
                "intersection": len(intersect_ids),
            },
        }

        out_name = (
            f"intersect_{safe_suspect}_thresh{suspect_threshold}"
            f"_x_{safe_judge}_thresh{judge_threshold}"
            f"_{subset}.json"
        )
        out_path = PROJECT_ROOT / "results" / out_name
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  Written: {out_path}")
        print()

    print(f"Total: {total_intersect} intersected questions "
          f"(from {total_suspect} suspect, {total_judge} judge) "
          f"across {len(subsets)} subsets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without writing files")
    args = parser.parse_args()
    intersect_filters(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
