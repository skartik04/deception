"""Reproduce the published AUROC for the instructed_pairs probe on the
roleplaying__plain eval rollouts (Llama 3.3 70B, BF16, layer 22, mean-pool).

Reported: 0.9637 (liars-bench/src/probes/example_results/instructed_pairs/results_table.csv).

Two checks:
  (1) recompute AUROC from the saved scores.json (no forward pass) — validates
      the metric definition matches.
  (2) re-extract activations + score with our pipeline (full forward passes),
      mean-pool, recompute AUROC — validates the probe pipeline reproduces
      published numbers.
"""

import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "liars-bench" / "src" / "probes"))

from deception_detection.activations import Activations
from deception_detection.data.base import Message
from deception_detection.detectors import LogisticRegressionDetector
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Dialogue

MODEL_PATH = "/workspace/models/llama-3.3-70b-instruct"
LAYER = 22
PADDING: dict[str, dict[str, int]] = {"llama": {"left": 0, "right": 0}}
MAX_LENGTH = 4096

PROBE_DIR = Path(__file__).parent / "liars-bench/src/probes/example_results/instructed_pairs"
PROBE_PATH = PROBE_DIR / "detector.pt"
SAVED_SCORES_PATH = PROBE_DIR / "scores.json"
ROLLOUTS_PATH = (
    Path(__file__).parent
    / "liars-bench/src/probes/data/rollouts/roleplaying__plain__llama-70b-3.3.json"
)


def auroc_from_scores(scores: list[float], labels: list[str]) -> float:
    y_true: list[int] = []
    y_score: list[float] = []
    for s, l in zip(scores, labels):
        if l == "DECEPTIVE":
            y_true.append(1)
            y_score.append(s)
        elif l == "HONEST":
            y_true.append(0)
            y_score.append(s)
    return float(roc_auc_score(y_true, y_score))


def step1_recompute_from_saved() -> None:
    saved = json.load(open(SAVED_SCORES_PATH))["roleplaying__plain"]
    per_token_scores: list[list[float]] = saved["scores"]
    labels: list[str] = saved["labels"]
    pooled = [sum(s) / len(s) for s in per_token_scores]
    auroc = auroc_from_scores(pooled, labels)
    print(f"[step 1] AUROC from saved scores.json (mean-pool): {auroc:.4f}")
    print(f"         reported (results_table.csv):              0.9637")


def build_dialogue(rollout: dict) -> Dialogue:
    msgs: list[Message] = []
    seen_assistant = False
    for m in rollout["input_messages"]:
        role = m["role"]
        content = m["content"]
        if role == "assistant" and not seen_assistant:
            assistant_content = rollout["output_strs"][0]
            msgs.append(Message("assistant", content + assistant_content, True))
            seen_assistant = True
            continue
        if role == "assistant":
            continue
        msgs.append(Message(role, content, False))
    return msgs


def step2_recompute_from_forward() -> None:
    print("\n[step 2] Loading model (BF16)...")
    from lib.model_loading import load_llama_strict
    model, tokenizer = load_llama_strict(MODEL_PATH, per_gpu_gib=78)
    probe = LogisticRegressionDetector.load(PROBE_PATH)

    rollouts = json.load(open(ROLLOUTS_PATH))["rollouts"]
    print(f"  {len(rollouts)} rollouts")

    saved = json.load(open(SAVED_SCORES_PATH))["roleplaying__plain"]
    saved_labels: list[str] = saved["labels"]

    dialogues: list[Dialogue] = [build_dialogue(r) for r in rollouts]

    pooled_scores: list[float] = []
    print("  scoring batches of 4...")
    BATCH = 4
    for start in range(0, len(dialogues), BATCH):
        chunk = dialogues[start : start + BATCH]
        toks = TokenizedDataset.from_dialogue_list(
            chunk, tokenizer, padding=PADDING, max_length=MAX_LENGTH
        )
        acts = Activations.from_model(
            model, toks, batch_size=BATCH, layers=[LAYER], verbose=False
        )
        scores = probe.score(acts).scores
        for s in scores:
            pooled_scores.append(float(s.mean().item()))
        if (start // BATCH) % 10 == 0:
            print(f"    {start + len(chunk)}/{len(dialogues)}")

    auroc = auroc_from_scores(pooled_scores, saved_labels)
    print(f"\n[step 2] AUROC from re-extracted activations (mean-pool): {auroc:.4f}")
    print(f"         reported (results_table.csv):                       0.9637")


def main() -> None:
    step1_recompute_from_saved()
    step2_recompute_from_forward()


if __name__ == "__main__":
    main()
