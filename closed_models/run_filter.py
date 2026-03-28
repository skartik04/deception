"""CLI entry point: run WMDP filtering for suspect or judge model.

Usage:
    uv run python -m closed_models.run_filter --role suspect
    uv run python -m closed_models.run_filter --role judge
"""

import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig

# Load .env before anything else
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.config import PROJECT_ROOT, load_config
from closed_models.filter_task import build_filter_task
from closed_models.compute_qualifying import compute_qualifying, save_results


def sanitize_model_name(model: str) -> str:
    """Turn model string into a safe filename fragment."""
    return re.sub(r"[^a-zA-Z0-9]", "-", model).strip("-")


def run_filtering(role: str) -> None:
    config = load_config()

    if role == "suspect":
        model_cfg = config.suspect
    elif role == "judge":
        model_cfg = config.judge
    else:
        print(f"Unknown role: {role}. Use 'suspect' or 'judge'.")
        sys.exit(1)

    model = model_cfg.model
    threshold = model_cfg.threshold
    framings = config.eval.framings
    temperature = config.eval.temperature

    print(f"Running filtering for {role} model: {model}")
    print(f"Threshold: {'>=' if role == 'suspect' else '<='} {threshold}/{config.total_runs}")
    print(f"Framings: {framings}  ({config.total_runs} attempts per question)")
    print()

    log_paths: list[str] = []

    for subset in config.dataset.subsets:
        print(f"  Evaluating {subset} ({config.dataset.num_questions} questions x {config.total_runs} attempts) ...")

        task = build_filter_task(
            subset=subset,
            limit=config.dataset.num_questions,
            framings=framings,
            role=role,
        )

        results = inspect_eval(
            task,
            model=model,
            epochs=1,
            log_dir=str(PROJECT_ROOT / "logs"),
            generate_config=GenerateConfig(temperature=temperature),
        )

        for result in results:
            if result.location:
                # Rename: {model_name}_{subset}_{num_questions}_{hash}.eval
                model_short = sanitize_model_name(model.split("/")[-1])
                hash_part = old_path.stem.split("_")[-1]
                new_name = f"{model_short}_{subset}_{config.dataset.num_questions}_{hash_part}.eval"
                old_path = Path(result.location)
                new_path = old_path.parent / new_name
                old_path.rename(new_path)
                log_paths.append(str(new_path))
                print(f"  Log: {new_path}")

    print(f"\nEval complete. Computing qualifying indices ...")

    config_snapshot = {
        "dataset": {"subsets": config.dataset.subsets, "num_questions": config.dataset.num_questions},
        "framings": framings,
    }

    qualifying = compute_qualifying(
        log_paths=log_paths,
        threshold=threshold,
        role=role,
        model=model,
        config_snapshot=config_snapshot,
    )

    safe_name = sanitize_model_name(model)
    output_path = PROJECT_ROOT / "results" / f"{role}_{safe_name}.json"
    save_results(qualifying, output_path)

    print(f"\nDone. {qualifying['stats']['qualifying_count']}/{qualifying['stats']['total_questions']} questions qualify.")
    print(f"Per-framing accuracy: {qualifying['stats']['per_framing_accuracy']}")


def main():
    parser = argparse.ArgumentParser(description="Run WMDP filtering for suspect or judge")
    parser.add_argument("--role", required=True, choices=["suspect", "judge"])
    args = parser.parse_args()
    run_filtering(args.role)


if __name__ == "__main__":
    main()
