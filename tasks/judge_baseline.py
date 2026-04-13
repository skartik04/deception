"""Inspect AI tasks: judge answers WMDP-bio questions cold (no suspect argument).

Two tasks in one file:
  - judge_baseline_yes_idk: IDK is valid (default for delta measurement).
  - judge_baseline_no_idk: IDK counts as incorrect (raw accuracy / model selection).

Usage:
    # Delta baseline — pair with interrogation/debate using same --limit
    uv run inspect eval tasks/judge_baseline.py@judge_baseline_yes_idk --limit 100 --log-level warning \
      --model openrouter/google/gemma-3-27b-it:nitro

    # Raw model accuracy (for judge selection)
    uv run inspect eval tasks/judge_baseline.py@judge_baseline_no_idk --limit 100 --log-level warning \
      --model openrouter/google/gemma-3-4b-it
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message

from lib.prompts import JudgeBaselinePromptsNoIDK, JudgeBaselinePromptsWithIDK
from lib.wmdp_data import (
    answer_scorer,
    answer_scorer_strict,
    balanced_expanded_dataset,
    format_question_block,
    load_root_dataset,
    original_dataset_filtered,
)
from inspect_ai.dataset import Sample, MemoryDataset


@task
def judge_baseline_yes_idk(
    balanced: bool = False,
    n_each: int = 50,
) -> Task:
    """Judge answers questions cold; IDK is valid (scores 0.5). --model is the judge.

    Use as the control measurement paired with interrogation or debate tasks.
    Run with the same --limit and balanced/n_each as the experiment.

    Args:
        balanced: If True, use the same filtered+balanced dataset as interrogation tasks.
        n_each: Number of honest and deceptive samples each when balanced=True (default 50).
    """
    prompts = JudgeBaselinePromptsWithIDK()
    if balanced:
        # Use balanced_expanded_dataset for question selection and is_deceptive metadata,
        # but replace the suspect-formatted input with a neutral question block so the
        # judge sees no argue_target hint.
        raw = balanced_expanded_dataset(n_each=n_each)
        neutral_samples = [
            Sample(
                input=format_question_block(
                    str((s.metadata or {})["question"]),
                    list((s.metadata or {})["choices"]),  # type: ignore[arg-type]
                ),
                target=s.target,
                metadata=s.metadata,
            )
            for s in raw.samples
        ]
        dataset = MemoryDataset(samples=neutral_samples, name="wmdp-bio-balanced-baseline")
    else:
        dataset = load_root_dataset()
    return Task(
        dataset=dataset,
        solver=[
            system_message(prompts.judge_system),
            generate(),
        ],
        scorer=answer_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=1024),
    )


@task
def judge_baseline_no_idk(filter_files: str = "") -> Task:
    """Judge answers questions cold; IDK is not valid (scores 0). --model is the model under test.

    Use for raw capability measurement and judge selection.

    Args:
        filter_files: Comma-separated list of Kartik filter JSON stems. If set,
            loads questions from those filter files instead of the full wmdp-bio root dataset.
    """
    prompts = JudgeBaselinePromptsNoIDK()
    if filter_files:
        files = [f.strip() for f in filter_files.split(",")]
        dataset = original_dataset_filtered(files)
    else:
        dataset = load_root_dataset()
    return Task(
        dataset=dataset,
        solver=[
            system_message(prompts.judge_system),
            generate(),
        ],
        scorer=answer_scorer_strict(),
        config=GenerateConfig(temperature=0, max_tokens=1024),
    )
