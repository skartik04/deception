"""Compute the intersection of suspect-pass and judge-fail qualifying IDs.

Usage:
    uv run python -m closed_models.intersect \
        --suspect results/suspect_openrouter-openai-gpt-oss-120b.json \
        --judge results/judge_openrouter-openai-gpt-oss-20b.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import PROJECT_ROOT


def intersect(suspect_path: str, judge_path: str) -> None:
    with open(suspect_path) as f:
        suspect = json.load(f)
    with open(judge_path) as f:
        judge = json.load(f)

    suspect_ids = set(suspect["qualifying_ids"])
    judge_ids = set(judge["qualifying_ids"])
    common = sorted(suspect_ids & judge_ids)

    result = {
        "suspect_model": suspect["model"],
        "judge_model": judge["model"],
        "suspect_file": suspect_path,
        "judge_file": judge_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filtered_ids": common,
        "stats": {
            "suspect_qualifying": len(suspect_ids),
            "judge_qualifying": len(judge_ids),
            "intersection": len(common),
        },
    }

    output_path = PROJECT_ROOT / "results" / "filtered_questions.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Suspect qualifying: {len(suspect_ids)}")
    print(f"Judge qualifying:   {len(judge_ids)}")
    print(f"Intersection:       {len(common)}")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Intersect suspect and judge qualifying IDs")
    parser.add_argument("--suspect", required=True, help="Path to suspect results JSON")
    parser.add_argument("--judge", required=True, help="Path to judge results JSON")
    args = parser.parse_args()
    intersect(args.suspect, args.judge)


if __name__ == "__main__":
    main()
