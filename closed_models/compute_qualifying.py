"""Post-process inspect eval logs to find qualifying question indices.

Reads one or more eval log files, groups results by original question ID,
counts correct answers across all framings and epochs, and filters by threshold.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import read_eval_log


def _extract_scores(log_path: str) -> dict[int, list[bool]]:
    """Extract per-question correctness from a single eval log.

    Returns: {original_id: [True, False, True, ...]} for each epoch/sample.
    """
    log = read_eval_log(log_path)
    scores: dict[int, list[bool]] = defaultdict(list)

    if log.samples is None:
        return scores

    for sample in log.samples:
        meta = sample.metadata or {}
        original_id = meta.get("original_id")
        if original_id is None:
            continue

        correct = False
        if sample.scores:
            score = next(iter(sample.scores.values()), None)
            if score and score.value:
                correct = score.value == "C"

        scores[original_id].append(correct)

    return scores


def compute_qualifying(
    log_paths: list[str],
    threshold: int,
    role: str,
    model: str,
    config_snapshot: dict,
) -> dict:
    """Aggregate scores across all logs and determine qualifying questions.

    For suspect: qualifying means correct_count >= threshold
    For judge: qualifying means correct_count <= threshold
    """
    all_scores: dict[int, list[bool]] = defaultdict(list)

    for lp in log_paths:
        per_question = _extract_scores(lp)
        for qid, results in per_question.items():
            all_scores[qid].extend(results)

    # Also track per-framing accuracy
    framing_scores: dict[str, list[bool]] = defaultdict(list)
    for lp in log_paths:
        log = read_eval_log(lp)
        if log.samples is None:
            continue
        for sample in log.samples:
            meta = sample.metadata or {}
            framing = meta.get("framing", "unknown")
            correct = False
            if sample.score and sample.score.value:
                correct = sample.score.value == "C"
            framing_scores[framing].append(correct)

    qualifying_ids = []
    per_question_scores = {}

    for qid, results in sorted(all_scores.items()):
        correct_count = sum(results)
        total = len(results)
        per_question_scores[str(qid)] = {"total": total, "correct": correct_count}

        if role == "suspect" and correct_count >= threshold:
            qualifying_ids.append(qid)
        elif role == "judge" and correct_count <= threshold:
            qualifying_ids.append(qid)

    per_framing_accuracy = {
        framing: sum(vals) / len(vals) if vals else 0.0
        for framing, vals in framing_scores.items()
    }

    return {
        "model": model,
        "role": role,
        "threshold": threshold,
        "dataset": config_snapshot.get("dataset", {}),
        "framings": config_snapshot.get("framings", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "qualifying_ids": qualifying_ids,
        "stats": {
            "total_questions": len(all_scores),
            "qualifying_count": len(qualifying_ids),
            "per_question_scores": per_question_scores,
            "per_framing_accuracy": per_framing_accuracy,
        },
    }


def save_results(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results['stats']['qualifying_count']} qualifying IDs to {output_path}")
