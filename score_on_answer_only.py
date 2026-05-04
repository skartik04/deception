"""Score the (already-trained) probes on test samples but apply detect-mask
ONLY over the suspect's `ANSWER: X` region. This isolates "does the residual
at the commitment moment encode honest vs deceptive" without diluting via
mean-over-the-whole-argument.

Probes evaluated:
  - 3 pretrained (roleplaying / followup / instructed_pairs)
  - CUSTOM (all-token), trained on n=30/30 with detect=True over full argument
  - CUSTOM (ANSWER-only), trained on n=75/75 with detect=True only on ANSWER line

Usage:
    PYTHONPATH=. uv run python score_on_answer_only.py <wmdp_eval_log_path>
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
from deception_detection.detectors import LogisticRegressionDetector
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Dialogue
from lib.prompts import SUSPECT_PROMPTS, _SUSPECT_USER  # type: ignore
from lib.wmdp_data import format_question_block

MODEL_PATH = "/workspace/models/llama-3.3-70b-instruct"
LAYER = 22
PADDING: dict[str, dict[str, int]] = {"llama": {"left": 0, "right": 0}}
MAX_LENGTH = 4096
SUSPECT_SYSTEM = SUSPECT_PROMPTS["debate_coach"]

PROBE_PATHS: dict[str, Path] = {
    "roleplaying":          Path(__file__).parent / "liars-bench/src/probes/example_results/roleplaying/detector.pt",
    "followup":             Path(__file__).parent / "liars-bench/src/probes/example_results/followup/detector.pt",
    "instructed_pairs":     Path(__file__).parent / "liars-bench/src/probes/example_results/instructed_pairs/detector.pt",
    "CUSTOM (all-token)":   Path(__file__).parent / "probes/wmdp_bio_interrogation_custom_layer22.pt",
    "CUSTOM (ANSWER-only)": Path(__file__).parent / "probes/wmdp_bio_answer_only_layer22.pt",
}


def extract_argument(judge_user_content: str) -> str:
    parts = judge_user_content.split("---\n", 1)
    rest = parts[1]
    end = rest.rsplit("\n---", 1)
    return end[0]


def split_argument(argument: str) -> tuple[str, str]:
    needle = "ANSWER:"
    idx = argument.rfind(needle)
    if idx == -1:
        return argument[:-30], argument[-30:]
    start = idx
    while start > 0 and argument[start - 1] in (" ", "\n", "\t"):
        start -= 1
    return argument[:start], argument[start:]


def make_dialogue_answer_only(suspect_user: str, argument: str) -> Dialogue:
    before, answer = split_argument(argument)
    msgs: list[Message] = [
        Message("system", SUSPECT_SYSTEM, False),
        Message("user", suspect_user, False),
    ]
    if before.strip():
        msgs.append(Message("assistant", before, False))
    msgs.append(Message("assistant", answer, True))
    return msgs


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: score_on_answer_only.py <wmdp_eval_log_path>", file=sys.stderr)
        sys.exit(1)
    log = read_eval_log(sys.argv[1])
    assert log.samples is not None
    honest = [s for s in log.samples if not s.metadata["is_deceptive"]]
    deceptive = [s for s in log.samples if s.metadata["is_deceptive"]]
    n = min(len(honest), len(deceptive))
    n_test = max(20, n // 4)
    n_train = n - n_test
    test_h = honest[n_train:n_train + n_test]
    test_d = deceptive[n_train:n_train + n_test]
    print(f"Test set: {len(test_h)} honest + {len(test_d)} deceptive")

    def make(sample: Any) -> Dialogue:
        meta = sample.metadata
        question_block = format_question_block(str(meta["question"]), list(meta["choices"]))
        suspect_user = _SUSPECT_USER.format(
            question_block=question_block,
            argue_target=meta["argue_target"],
            argue_target_text=meta["argue_target_text"],
        )
        judge_user = next(m.content for m in sample.messages if m.role == "user")
        argument = extract_argument(str(judge_user))
        return make_dialogue_answer_only(suspect_user, argument)

    print("Loading model (BF16)...")
    from lib.model_loading import load_llama_strict
    model, tokenizer = load_llama_strict(MODEL_PATH, per_gpu_gib=78)
    detectors = {n: LogisticRegressionDetector.load(p) for n, p in PROBE_PATHS.items()}

    def extract_acts(samples: list[Any], desc: str) -> Activations:
        dialogues = [make(s) for s in samples]
        toks = TokenizedDataset.from_dialogue_list(
            dialogues, tokenizer, padding=PADDING, max_length=MAX_LENGTH
        )
        print(f"  {desc}: {len(dialogues)} dialogues")
        return Activations.from_model(model, toks, batch_size=1, layers=[LAYER], verbose=True)

    test_h_acts = extract_acts(test_h, "test honest")
    test_d_acts = extract_acts(test_d, "test deceptive")

    table: dict[str, dict[str, float]] = defaultdict(dict)
    print(f"\n{'Probe':<24} {'Pool':<6} {'Test AUROC':>12}")
    print("-" * 47)
    for name, det in detectors.items():
        h_obj = det.score(test_h_acts)
        d_obj = det.score(test_d_acts)
        for mode in ("mean", "max", "last"):
            def pool_t(t: torch.Tensor) -> float:
                if mode == "mean":
                    return t.mean().item()
                if mode == "max":
                    return t.max().item()
                return t[-1].item()
            scores = [pool_t(s) for s in h_obj.scores] + [pool_t(s) for s in d_obj.scores]
            y = [0] * len(h_obj.scores) + [1] * len(d_obj.scores)
            auroc = float(roc_auc_score(y, scores))
            table[name][mode] = auroc
            print(f"{name:<24} {mode:<6} {auroc:>12.4f}")

    # Diagnostic: how many tokens in the ANSWER mask, sample 0?
    h_mask = test_h_acts.tokenized_dataset.detection_mask
    d_mask = test_d_acts.tokenized_dataset.detection_mask
    assert h_mask is not None and d_mask is not None
    n_h0 = int(h_mask[0].sum().item())
    n_d0 = int(d_mask[0].sum().item())
    print(f"\nANSWER-region token count, honest sample 0: {n_h0}; deceptive sample 0: {n_d0}")

    out_json = Path(__file__).parent / "score_on_answer_only_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_test_honest": len(test_h),
            "n_test_deceptive": len(test_d),
            "answer_mask_tokens_sample0": {"honest": n_h0, "deceptive": n_d0},
            "AUROCs": dict(table),
        }, f, indent=2)
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
