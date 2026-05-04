"""Test Apollo's `descriptive` probe (MMS detector trained on
`descriptive_qa__pair`, layer 22). Unlike the LR-trained roleplaying /
instructed_pairs / followup probes, this one is a single Mass-Mean-Shift
direction in residual space.

Same WMDP-bio test set as our previous probe evals (25 honest + 25 deceptive
slice from logs/2026-05-03T20-52-41-00-00...).

Usage:
    PYTHONPATH=. uv run python eval_descriptive_probe.py <wmdp_eval_log_path>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from inspect_ai.log import read_eval_log
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "liars-bench" / "src" / "probes"))

from deception_detection.activations import Activations
from deception_detection.data.base import Message
from deception_detection.detectors import (
    LogisticRegressionDetector,
    MMSDetector,
)
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Dialogue
from lib.prompts import SUSPECT_PROMPTS, _SUSPECT_USER  # type: ignore
from lib.wmdp_data import format_question_block

MODEL_PATH = "/workspace/models/llama-3.3-70b-instruct"
PADDING: dict[str, dict[str, int]] = {"llama": {"left": 0, "right": 0}}
MAX_LENGTH = 4096
SUSPECT_SYSTEM = SUSPECT_PROMPTS["debate_coach"]
LAYER = 22

PROBE_DIR_ROOT = Path(__file__).parent / "liars-bench/src/probes/example_results"
DESCRIPTIVE = PROBE_DIR_ROOT / "descriptive" / "detector.pt"


def extract_argument(judge_user_content: str) -> str:
    parts = judge_user_content.split("---\n", 1)
    rest = parts[1]
    end = rest.rsplit("\n---", 1)
    return end[0]


def make_dialogue(suspect_user: str, argument: str) -> Dialogue:
    return [
        Message("system", SUSPECT_SYSTEM, False),
        Message("user", suspect_user, False),
        Message("assistant", argument, True),
    ]


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: eval_descriptive_probe.py <wmdp_eval_log_path>", file=sys.stderr)
        sys.exit(1)
    log = read_eval_log(sys.argv[1])
    assert log.samples is not None
    honest = [s for s in log.samples if not s.metadata["is_deceptive"]]
    deceptive = [s for s in log.samples if s.metadata["is_deceptive"]]
    n = min(len(honest), len(deceptive))
    n_test = max(20, n // 4)
    n_train = n - n_test
    test_h = honest[n_train : n_train + n_test]
    test_d = deceptive[n_train : n_train + n_test]
    print(f"Test set: {len(test_h)} honest + {len(test_d)} deceptive")

    print("\nLoading Llama 3.3 70B (BF16, 2× H100)...")
    from lib.model_loading import load_llama_strict
    model, tokenizer = load_llama_strict(MODEL_PATH, per_gpu_gib=75)
    descriptive = MMSDetector.load(DESCRIPTIVE)
    print(f"descriptive (MMS) probe loaded at layer {descriptive.layers}")

    def build(sample: Any) -> Dialogue:
        meta = sample.metadata
        qb = format_question_block(str(meta["question"]), list(meta["choices"]))
        suspect_user = _SUSPECT_USER.format(
            question_block=qb,
            argue_target=meta["argue_target"],
            argue_target_text=meta["argue_target_text"],
        )
        judge_user = next(m.content for m in sample.messages if m.role == "user")
        argument = extract_argument(str(judge_user))
        return make_dialogue(suspect_user, argument)

    def extract_acts(samples: list[Any], desc: str) -> Activations:
        dialogues = [build(s) for s in samples]
        toks = TokenizedDataset.from_dialogue_list(
            dialogues, tokenizer, padding=PADDING, max_length=MAX_LENGTH
        )
        print(f"  {desc}: {len(dialogues)} dialogues")
        return Activations.from_model(
            model, toks, batch_size=1, layers=[LAYER], verbose=True
        )

    test_h_acts = extract_acts(test_h, "test honest")
    test_d_acts = extract_acts(test_d, "test deceptive")

    pooled: dict[str, list[float]] = defaultdict(list)
    labels_list: list[str] = []
    for label, acts in [("HONEST", test_h_acts), ("DECEPTIVE", test_d_acts)]:
        scores_obj = descriptive.score(acts)
        for s in scores_obj.scores:
            labels_list.append(label)
            pooled["mean"].append(float(s.mean().item()))
            pooled["max"].append(float(s.max().item()))
            pooled["last"].append(float(s[-1].item()))

    print(f"\n{'Pool':<8} {'AUROC':>8}")
    print("-" * 18)
    aurocs: dict[str, float] = {}
    for mode in ("mean", "max", "last"):
        y = [1 if l == "DECEPTIVE" else 0 for l in labels_list]
        aurocs[mode] = float(roc_auc_score(y, pooled[mode]))
        print(f"{mode:<8} {aurocs[mode]:>8.4f}")

    out_json = Path(__file__).parent / "descriptive_probe_results.json"
    json.dump({
        "n_test_honest": len(test_h),
        "n_test_deceptive": len(test_d),
        "AUROCs": aurocs,
        "probe": "Apollo descriptive (MMS, layer 22, train_data=descriptive_qa__pair)",
    }, open(out_json, "w"), indent=2)
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
