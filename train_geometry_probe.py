"""Train and evaluate a linear probe on Llama 3.3 70B using Geometry of Truth datasets.

Usage:
    cd /root/gergely_deception && uv run python train_geometry_probe.py

Requires ~140 GB VRAM (2× H100/A100 80 GB) for bfloat16.
Data: liars-bench/src/probes/data/geometry_of_truth/ (11 subsets).
"""

import csv
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent / "liars-bench" / "src" / "probes"))

from deception_detection.activations import Activations
from deception_detection.data.base import Message
from deception_detection.detectors import LogisticRegressionDetector
from deception_detection.scores import Scores
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Dialogue

# Hyperparameters (from liars-bench roleplaying config)
LAYER = 22
REG_COEFF = 10.0
NORMALIZE = True
MAX_ITER = 1000
TRAIN_PER_CLASS = 250       # 250 honest + 250 deceptive total
TRAIN_PER_SUBSET = 23       # per-class quota per subset; 23 × 11 = 253, trimmed to 250
TEST_PER_SUBSET = 23        # per-class quota per subset for test set
RANDOM_SEED = 42
PADDING: dict[str, dict[str, int]] = {"llama": {"left": 0, "right": 0}}
MAX_LENGTH = 512

DATA_DIR = (
    Path(__file__).parent / "liars-bench" / "src" / "probes" / "data" / "geometry_of_truth"
)
SUBSET_FILES = [
    "cities.csv",
    "cities_cities_conj.csv",
    "cities_cities_disj.csv",
    "common_claim_true_false.csv",
    "companies_true_false.csv",
    "counterfact_true_false.csv",
    "larger_than.csv",
    "neg_cities.csv",
    "smaller_than.csv",
    "sp_en_trans.csv",
    "neg_sp_en_trans.csv",
]


def load_subset(csv_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (honest_rows, deceptive_rows) from a GT CSV (label=1 honest, label=0 deceptive)."""
    honest: list[dict[str, str]] = []
    deceptive: list[dict[str, str]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            (honest if int(row["label"]) == 1 else deceptive).append(dict(row))
    return honest, deceptive


def make_dialogue(row: dict[str, str]) -> Dialogue:
    """Plain GT format: probe on the final period token after the statement."""
    statement = row["statement"].rstrip(".")
    return [
        Message("user", "Please state a fact.", False),
        Message("assistant", statement, False),
        Message("assistant", ".", True),  # detect=True → probe target
    ]


def get_activations(
    rows: list[dict[str, str]],
    model: Any,
    tokenizer: Any,
) -> Activations:
    dialogues: list[Dialogue] = [make_dialogue(r) for r in rows]
    toks = TokenizedDataset.from_dialogue_list(
        dialogues, tokenizer, padding=PADDING, max_length=MAX_LENGTH
    )
    return Activations.from_model(model, toks, batch_size=1, layers=[LAYER], verbose=True)


def classify(scores: Scores) -> list[bool]:
    """Return True (predicted deceptive) if mean logit > 0, False (predicted honest) otherwise."""
    return [bool(s.mean().item() > 0) for s in scores.scores]


def main() -> None:
    random.seed(RANDOM_SEED)

    # --- Build train / test pools from all 11 subsets ---
    train_honest: list[dict[str, str]] = []
    train_deceptive: list[dict[str, str]] = []
    test_honest: list[dict[str, str]] = []
    test_deceptive: list[dict[str, str]] = []

    for fname in SUBSET_FILES:
        honest, deceptive = load_subset(DATA_DIR / fname)
        random.shuffle(honest)
        random.shuffle(deceptive)

        n_train = min(TRAIN_PER_SUBSET, len(honest), len(deceptive))
        train_honest.extend(honest[:n_train])
        train_deceptive.extend(deceptive[:n_train])

        n_test = min(TEST_PER_SUBSET, len(honest) - n_train, len(deceptive) - n_train)
        test_honest.extend(honest[n_train : n_train + n_test])
        test_deceptive.extend(deceptive[n_train : n_train + n_test])

    # Trim train pools to exactly TRAIN_PER_CLASS (shuffle first to avoid subset bias)
    random.shuffle(train_honest)
    random.shuffle(train_deceptive)
    train_honest = train_honest[:TRAIN_PER_CLASS]
    train_deceptive = train_deceptive[:TRAIN_PER_CLASS]

    print(f"Train: {len(train_honest)} honest, {len(train_deceptive)} deceptive")
    print(f"Test:  {len(test_honest)} honest,  {len(test_deceptive)} deceptive")

    # --- Load model (8-bit quantization: ~70 GB, fits in 96 GB GPU) ---
    print("\nLoading model (8-bit quantization)...")
    model_path = "/root/models/llama-3.3-70b-instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
    tokenizer.pad_token_id = tokenizer.bos_token_id
    model.eval()

    # --- Extract train activations ---
    print("\nExtracting train honest activations...")
    train_honest_acts = get_activations(train_honest, model, tokenizer)
    print("Extracting train deceptive activations...")
    train_deceptive_acts = get_activations(train_deceptive, model, tokenizer)

    # --- Train probe ---
    print("\nTraining probe...")
    detector = LogisticRegressionDetector(
        layers=[LAYER],
        reg_coeff=REG_COEFF,
        normalize=NORMALIZE,
        max_iter=MAX_ITER,
    )
    # Convention: positive_acts = deceptive (y=1), negative_acts = honest (y=0)
    detector.fit(positive_acts=train_deceptive_acts, negative_acts=train_honest_acts)

    # --- Extract test activations ---
    print("\nExtracting test honest activations...")
    test_honest_acts = get_activations(test_honest, model, tokenizer)
    print("Extracting test deceptive activations...")
    test_deceptive_acts = get_activations(test_deceptive, model, tokenizer)

    # --- Evaluate ---
    honest_preds = classify(detector.score(test_honest_acts))
    deceptive_preds = classify(detector.score(test_deceptive_acts))

    # Honest → should predict False (logit < 0); Deceptive → should predict True (logit > 0)
    n_honest_correct = sum(1 for p in honest_preds if not p)
    n_deceptive_correct = sum(1 for p in deceptive_preds if p)
    n_total = len(honest_preds) + len(deceptive_preds)
    n_correct = n_honest_correct + n_deceptive_correct

    print(f"\n{'='*50}")
    print(f"Test results  (layer {LAYER}, reg_coeff={REG_COEFF})")
    print(
        f"  Honest accuracy:    "
        f"{n_honest_correct}/{len(honest_preds)} = "
        f"{n_honest_correct/len(honest_preds):.1%}"
    )
    print(
        f"  Deceptive accuracy: "
        f"{n_deceptive_correct}/{len(deceptive_preds)} = "
        f"{n_deceptive_correct/len(deceptive_preds):.1%}"
    )
    print(f"  Overall accuracy:   {n_correct}/{n_total} = {n_correct/n_total:.1%}")
    print(f"{'='*50}")

    # --- Save probe ---
    output_path = Path(__file__).parent / "probes" / "geometry_of_truth_llama70b_layer22.pt"
    output_path.parent.mkdir(exist_ok=True)
    detector.save(output_path)
    print(f"\nSaved probe to {output_path}")


if __name__ == "__main__":
    main()
