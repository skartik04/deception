"""Run an interrogation sweep over the filtered usable IDs.

This script runs whichever conditions are enabled in RUN_CONDITIONS below.
The CLI --limit is interpreted as a number of source questions, not expanded
Inspect samples. Questions are taken in deterministic order from:
  1. usable_ids_wmdp-bio
  2. usable_ids_wmdp-chem
  3. usable_ids_wmdp-cyber

Since each question expands to 4 argue-target variants, the Inspect sample
limit is automatically set to 4 * question_limit.

Usage:
    uv run python scripts/run_interrogation_suite.py --limit 10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_ROOT = PROJECT_ROOT / "logs"
FILTERINGS_DIR = PROJECT_ROOT / "data" / "filterings"
load_dotenv(PROJECT_ROOT / ".env")

from tasks.interrogation_single_turn import interrogation_single_turn
from tasks.interrogation_single_turn_detector import interrogation_single_turn_detector

RUN_CONDITIONS: dict[str, bool] = {
    "oracle_true": True,
    "oracle_random50": True,
    "no_detector": True,
    "llm_score": True,
    "llm_report": True,
}

SUSPECT_MODEL = "openrouter/meta-llama/llama-3.3-70b-instruct:nitro"
JUDGE_MODEL = "openrouter/google/gemma-3-27b-it:nitro"
DETECTOR_MODEL = "openrouter/google/gemma-3-27b-it:nitro"
FILTER_FILES = [f.stem for f in sorted(FILTERINGS_DIR.glob("usable_ids_wmdp-*.json"))]
SUSPECT_PROMPT = "debate_coach"
JUDGE_PROMPT = "informed"
TEMPERATURE = 0.0


def _load_filter_file(stem: str) -> dict:
    path = FILTERINGS_DIR / f"{stem}.json"
    return json.loads(path.read_text())


def _selected_question_ids(question_limit: int) -> dict[str, list[int]]:
    remaining = question_limit
    selected: dict[str, list[int]] = {}
    for stem in FILTER_FILES:
        data = _load_filter_file(stem)
        subset = data["dataset"]["subsets"][0]
        ids = list(data["qualifying_ids"])
        take = ids[:remaining]
        selected[subset] = take
        remaining -= len(take)
        if remaining <= 0:
            break
    return selected


def _available_question_count() -> int:
    total = 0
    for stem in FILTER_FILES:
        data = _load_filter_file(stem)
        total += len(data["qualifying_ids"])
    return total


def _build_task(condition: str):
    filter_files = ",".join(FILTER_FILES)

    if condition == "oracle_true":
        return interrogation_single_turn_detector(
            suspect_model=SUSPECT_MODEL,
            suspect_prompt=SUSPECT_PROMPT,
            detector_name="oracle",
            filter_files=filter_files,
            temperature=TEMPERATURE,
            allow_idk=False,
        )
    if condition == "oracle_random50":
        return interrogation_single_turn_detector(
            suspect_model=SUSPECT_MODEL,
            suspect_prompt=SUSPECT_PROMPT,
            detector_name="always_50",
            filter_files=filter_files,
            temperature=TEMPERATURE,
            allow_idk=False,
        )
    if condition == "no_detector":
        return interrogation_single_turn(
            suspect_model=SUSPECT_MODEL,
            suspect_prompt=SUSPECT_PROMPT,
            judge_prompt=JUDGE_PROMPT,
            filter_files=filter_files,
            allow_idk=False,
        )
    if condition == "llm_score":
        return interrogation_single_turn_detector(
            suspect_model=SUSPECT_MODEL,
            suspect_prompt=SUSPECT_PROMPT,
            detector_name="llm_score",
            detector_model=DETECTOR_MODEL,
            filter_files=filter_files,
            temperature=TEMPERATURE,
            allow_idk=False,
        )
    if condition == "llm_report":
        return interrogation_single_turn_detector(
            suspect_model=SUSPECT_MODEL,
            suspect_prompt=SUSPECT_PROMPT,
            detector_name="llm_report",
            detector_model=DETECTOR_MODEL,
            filter_files=filter_files,
            temperature=TEMPERATURE,
            allow_idk=False,
        )
    raise ValueError(f"Unknown condition: {condition}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Number of source questions to run. Converted internally to 4x samples.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without executing them.",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer number of questions.")

    enabled_conditions = [name for name, enabled in RUN_CONDITIONS.items() if enabled]
    if not enabled_conditions:
        raise ValueError("No conditions enabled in RUN_CONDITIONS.")

    available_questions = _available_question_count()
    effective_questions = min(args.limit, available_questions)
    sample_limit = effective_questions * 4

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S")
    run_root = LOGS_ROOT / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "requested_question_limit": args.limit,
        "effective_question_limit": effective_questions,
        "sample_limit": sample_limit,
        "enabled_conditions": enabled_conditions,
        "filter_files": FILTER_FILES,
        "selected_question_ids": _selected_question_ids(effective_questions),
        "suspect_model": SUSPECT_MODEL,
        "judge_model": JUDGE_MODEL,
        "detector_model": DETECTOR_MODEL,
        "suspect_prompt": SUSPECT_PROMPT,
        "judge_prompt": JUDGE_PROMPT,
        "temperature": TEMPERATURE,
    }
    (run_root / "run_config.json").write_text(json.dumps(config, indent=2))

    print(f"Run root: {run_root}")
    print(f"Questions: {effective_questions}")
    print(f"Samples per condition: {sample_limit}")
    print(f"Conditions: {enabled_conditions}")
    print(f"Ordered filter files: {FILTER_FILES}")
    print()

    if args.dry_run:
        print("Dry run only. No evals launched.")
        return

    for condition in enabled_conditions:
        condition_dir = run_root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Running {condition} ===")
        task = _build_task(condition)
        inspect_eval(
            task,
            model=JUDGE_MODEL,
            limit=sample_limit,
            log_dir=str(condition_dir),
            log_level="warning",
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
