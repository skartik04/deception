"""Inspect task for WMDP question filtering.

Builds one combined dataset with all framings expanded inline.
Each question appears N times (once per framing run), so a single
eval with epochs=1 covers all attempts. One log file per model.
"""

from datasets import load_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from core.framings import FRAMINGS


def _load_wmdp_raw(subset: str, limit: int) -> list[dict]:
    """Load raw WMDP records from HuggingFace as plain dicts."""
    ds = load_dataset("cais/wmdp", subset, split="test")
    records = []
    for i, row in enumerate(ds):
        if limit > 0 and i >= limit:
            break
        records.append({
            "index": i,
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        })
    return records


def build_filter_task(subset: str, limit: int, framings: dict[str, int], role: str = "") -> Task:
    """Build one combined inspect Task covering all framings.

    Each question is expanded into sum(framings.values()) samples —
    e.g. 4 MCQ + 3 T/F + 3 Negated = 10 samples per question.
    Run with epochs=1: one log file, all attempts in one place.
    """
    raw_records = _load_wmdp_raw(subset, limit)

    samples = []
    for rec in raw_records:
        for framing_name, count in framings.items():
            framing_fn = FRAMINGS[framing_name]
            for run_idx in range(count):
                sample = framing_fn(rec, rec["index"])
                # Make ID unique across runs of the same framing
                sample = sample.model_copy(
                    update={"id": f"{rec['index']}_{framing_name}_{run_idx}"}
                )
                samples.append(sample)

    task_name = f"{role}_{subset}" if role else f"wmdp_{subset}"
    return Task(
        name=task_name,
        dataset=MemoryDataset(samples=samples, name=task_name),
        solver=[multiple_choice()],
        scorer=choice(),
    )
