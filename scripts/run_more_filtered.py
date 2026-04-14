"""Run interrogation_single_turn_detector with coin_flip detector on the
more-filtered dataset (suspect knows & judge doesn't, across bio/chem/cyber).

Runs 4 conditions × 2 temperatures:
  - honest + coin_flip, temp=0
  - honest + coin_flip, temp=1
  - deceptive + coin_flip, temp=0
  - deceptive + coin_flip, temp=1

Actually runs the full expanded dataset (all 4 argue_targets) at both temps,
which gives all conditions in 2 evals. Logs go to logs/more_filtered_04_08/.

Usage:
    uv run python scripts/run_more_filtered.py [--dry-run]

Requires:
    kartik_deception/results/intersect_*_{subset}.json for each subset.
"""

import argparse
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / "kartik_deception" / ".env")

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KARTIK_RESULTS = PROJECT_ROOT.parent / "kartik_deception" / "results"
LOG_DIR = PROJECT_ROOT / "logs" / "more_filtered_04_08"

SUSPECT_MODEL = "openrouter/meta-llama/llama-3.3-70b-instruct:nitro"
JUDGE_MODEL = "openrouter/google/gemma-3-27b-it:nitro"
TEMPERATURES = [0.0, 1.0]

# Import task here so inspect_ai can find it
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from tasks.interrogation_single_turn_detector import interrogation_single_turn_detector
from tasks.judge_baseline import judge_baseline_no_idk


def find_intersect_files() -> list[str]:
    """Find all intersect JSON stems in kartik results dir."""
    files = sorted(KARTIK_RESULTS.glob("intersect_*_wmdp-*.json"))
    if not files:
        raise FileNotFoundError(
            f"No intersect_*.json files found in {KARTIK_RESULTS}. "
            "Run kartik_deception/closed_models/intersect_filters.py first."
        )
    return [f.stem for f in files]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running evals")
    args = parser.parse_args()

    filter_files = find_intersect_files()
    filter_files_str = ",".join(filter_files)

    print(f"Filter files ({len(filter_files)}):")
    for f in filter_files:
        print(f"  {f}")
    print(f"Log dir: {LOG_DIR}")
    print(f"Temperatures: {TEMPERATURES}")
    print()

    if args.dry_run:
        print("Dry run — not launching evals.")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    MAX_CONNECTIONS = 1000

    print("--- Baseline (no detector, no IDK) ---")
    baseline_task = judge_baseline_no_idk(filter_files=filter_files_str)
    inspect_eval(
        baseline_task,
        model=JUDGE_MODEL,
        epochs=1,
        log_dir=str(LOG_DIR),
        max_samples=MAX_CONNECTIONS,
        max_connections=MAX_CONNECTIONS,
        temperature=0.0,
        max_tokens=1024,
    )
    print()

    for temp in TEMPERATURES:
        print(f"--- Temperature {temp} ---")
        task = interrogation_single_turn_detector(
            suspect_model=SUSPECT_MODEL,
            detector_name="coin_flip",
            allow_idk=False,
            answer_focused=False,
            temperature=temp,
            filter_files=filter_files_str,
        )
        inspect_eval(
            task,
            model=JUDGE_MODEL,
            epochs=1,
            log_dir=str(LOG_DIR),
            max_samples=MAX_CONNECTIONS,
            max_connections=MAX_CONNECTIONS,
            temperature=temp,
            max_tokens=1024,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
