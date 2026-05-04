"""Train a custom probe whose detect mask covers ONLY the suspect's `ANSWER: X`
region (the final commitment line), and evaluate it across the whole argument.

For each sample:
  - TRAIN dialogue: assistant message split into two — body (detect=False) +
    `ANSWER: X` + eot (detect=True). Probe learns from these tokens only.
  - TEST dialogue: detect_all=True. Probe (and others) score every token. AUROC
    computed with mean/max/last pooling. Per-token plots cover the full argument.

Usage:
    PYTHONPATH=. uv run python train_answer_only_probe.py <wmdp_eval_log_path>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

PROBES_PRETRAINED: dict[str, Path] = {
    "roleplaying":      Path(__file__).parent / "liars-bench/src/probes/example_results/roleplaying/detector.pt",
    "followup":         Path(__file__).parent / "liars-bench/src/probes/example_results/followup/detector.pt",
    "instructed_pairs": Path(__file__).parent / "liars-bench/src/probes/example_results/instructed_pairs/detector.pt",
}
EXISTING_CUSTOM = Path(__file__).parent / "probes/wmdp_bio_interrogation_custom_layer22.pt"
NEW_CUSTOM_PATH = Path(__file__).parent / "probes/wmdp_bio_answer_only_layer22.pt"


def extract_argument(judge_user_content: str) -> str:
    parts = judge_user_content.split("---\n", 1)
    rest = parts[1]
    end = rest.rsplit("\n---", 1)
    return end[0]


def split_argument(argument: str) -> tuple[str, str]:
    """Split argument into (before_answer_line, answer_line). Final answer line
    is the last 'ANSWER:' occurrence onward (typically just `ANSWER: X`)."""
    needle = "ANSWER:"
    idx = argument.rfind(needle)
    if idx == -1:
        # Fallback: use last 30 characters as the "answer region"
        return argument[:-30], argument[-30:]
    # Include any leading whitespace immediately before ANSWER for cleaner tokenization
    start = idx
    while start > 0 and argument[start - 1] in (" ", "\n", "\t"):
        start -= 1
    return argument[:start], argument[start:]


def make_train_dialogue(suspect_user: str, argument: str) -> Dialogue:
    before, answer = split_argument(argument)
    msgs: list[Message] = [
        Message("system", SUSPECT_SYSTEM, False),
        Message("user", suspect_user, False),
    ]
    if before.strip():
        msgs.append(Message("assistant", before, False))
    msgs.append(Message("assistant", answer, True))
    return msgs


def make_test_dialogue(suspect_user: str, argument: str) -> Dialogue:
    return [
        Message("system", SUSPECT_SYSTEM, False),
        Message("user", suspect_user, False),
        Message("assistant", argument, True),
    ]


def auroc_from(scores: list[float], labels: list[str]) -> float:
    y_true = [1 if l == "DECEPTIVE" else 0 for l in labels]
    return float(roc_auc_score(y_true, scores))


def pool(t: torch.Tensor, mode: str) -> float:
    if mode == "mean":
        return t.mean().item()
    if mode == "max":
        return t.max().item()
    return t[-1].item()


def plot_sample(
    sample_label: str,
    tokens: list[str],
    scores: dict[str, list[float]],
    out_path: Path,
) -> None:
    n_probes = len(scores)
    n_tokens = max(len(tokens), max(len(v) for v in scores.values()))
    fig_w = max(12.0, n_tokens * 0.18)
    fig, axes = plt.subplots(
        n_probes, 1, figsize=(fig_w, 2.5 * n_probes), sharex=True
    )
    if n_probes == 1:
        axes = [axes]
    for ax, (probe_name, ys) in zip(axes, scores.items()):
        x = list(range(len(ys)))
        colors = ["tab:red" if y > 0 else "tab:blue" for y in ys]
        ax.bar(x, ys, color=colors, width=0.85)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(probe_name, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    n_label = min(len(tokens), n_tokens)
    axes[-1].set_xticks(list(range(n_label)))
    axes[-1].set_xticklabels(
        [t.replace("\n", "\\n") for t in tokens[:n_label]],
        rotation=90,
        fontsize=6,
        family="monospace",
    )
    fig.suptitle(
        f"{sample_label}: per-token probe scores (positive = deceptive)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: train_answer_only_probe.py <wmdp_eval_log_path>", file=sys.stderr)
        sys.exit(1)
    log = read_eval_log(sys.argv[1])
    assert log.samples is not None
    honest = [s for s in log.samples if not s.metadata["is_deceptive"]]
    deceptive = [s for s in log.samples if s.metadata["is_deceptive"]]
    print(f"WMDP eval: {len(honest)} honest, {len(deceptive)} deceptive")

    n = min(len(honest), len(deceptive))
    n_test = max(20, n // 4)
    n_train = n - n_test
    train_h, test_h = honest[:n_train], honest[n_train:n_train + n_test]
    train_d, test_d = deceptive[:n_train], deceptive[n_train:n_train + n_test]
    print(f"Split: {n_train} train_h + {n_train} train_d / {len(test_h)} test_h + {len(test_d)} test_d")

    def build(sample: Any, dialogue_fn: Any) -> Dialogue:
        meta = sample.metadata
        question_block = format_question_block(str(meta["question"]), list(meta["choices"]))
        suspect_user = _SUSPECT_USER.format(
            question_block=question_block,
            argue_target=meta["argue_target"],
            argue_target_text=meta["argue_target_text"],
        )
        judge_user = next(m.content for m in sample.messages if m.role == "user")
        argument = extract_argument(str(judge_user))
        return dialogue_fn(suspect_user, argument)

    print("\nLoading model (BF16)...")
    from lib.model_loading import load_llama_strict
    model, tokenizer = load_llama_strict(MODEL_PATH, per_gpu_gib=78)
    def extract(samples: list[Any], dialogue_fn: Any, desc: str) -> Activations:
        dialogues = [build(s, dialogue_fn) for s in samples]
        toks = TokenizedDataset.from_dialogue_list(
            dialogues, tokenizer, padding=PADDING, max_length=MAX_LENGTH
        )
        print(f"  {desc}: {len(dialogues)} dialogues")
        return Activations.from_model(
            model, toks, batch_size=1, layers=[LAYER], verbose=True
        )

    print("\nExtracting train activations (ANSWER region only)...")
    train_h_acts = extract(train_h, make_train_dialogue, "train honest")
    train_d_acts = extract(train_d, make_train_dialogue, "train deceptive")

    print("\nTraining new probe (ANSWER-only) ...")
    new_probe = LogisticRegressionDetector(layers=[LAYER], normalize=True, reg_coeff=1e3)
    new_probe.fit(positive_acts=train_d_acts, negative_acts=train_h_acts)
    NEW_CUSTOM_PATH.parent.mkdir(exist_ok=True)
    new_probe.save(str(NEW_CUSTOM_PATH))
    print(f"  saved {NEW_CUSTOM_PATH}")

    print("\nExtracting test activations (full mask)...")
    test_h_acts = extract(test_h, make_test_dialogue, "test honest")
    test_d_acts = extract(test_d, make_test_dialogue, "test deceptive")

    print("\n=== Test AUROC ===")
    detectors = {
        "roleplaying":      LogisticRegressionDetector.load(PROBES_PRETRAINED["roleplaying"]),
        "followup":         LogisticRegressionDetector.load(PROBES_PRETRAINED["followup"]),
        "instructed_pairs": LogisticRegressionDetector.load(PROBES_PRETRAINED["instructed_pairs"]),
        "CUSTOM (all-token)": LogisticRegressionDetector.load(EXISTING_CUSTOM),
        "CUSTOM (ANSWER-only)": new_probe,
    }
    table: dict[str, dict[str, float]] = defaultdict(dict)
    print(f"\n{'Probe':<24} {'Pool':<6} {'Test AUROC':>12}")
    print("-" * 47)
    for name, det in detectors.items():
        h_scores_obj = det.score(test_h_acts)
        d_scores_obj = det.score(test_d_acts)
        for mode in ("mean", "max", "last"):
            h_pooled = [pool(s, mode) for s in h_scores_obj.scores]
            d_pooled = [pool(s, mode) for s in d_scores_obj.scores]
            scores_all = h_pooled + d_pooled
            labels_all = ["HONEST"] * len(h_pooled) + ["DECEPTIVE"] * len(d_pooled)
            auroc = auroc_from(scores_all, labels_all)
            table[name][mode] = auroc
            print(f"{name:<24} {mode:<6} {auroc:>12.4f}")

    print("\n=== Plotting ===")
    for label, sample, acts in [
        ("honest", test_h[0], test_h_acts),
        ("deceptive", test_d[0], test_d_acts),
    ]:
        idx = 0
        mask = acts.tokenized_dataset.detection_mask
        assert mask is not None
        str_tokens = acts.tokenized_dataset.str_tokens[idx]
        mask_list = mask[idx].tolist()
        n_pair = min(len(str_tokens), len(mask_list))
        kept_tokens = [str_tokens[i] for i in range(n_pair) if mask_list[i]]
        sample_scores: dict[str, list[float]] = {}
        for probe_name, det in detectors.items():
            sample_scores[probe_name] = det.score(acts).scores[idx].tolist()
        out_png = Path(__file__).parent / f"answer_only_{label}_argue{sample.metadata['argue_target']}.png"
        plot_sample(f"WMDP {label} (ANSWER-only-trained probe vs others)", kept_tokens, sample_scores, out_png)

    out_json = Path(__file__).parent / "answer_only_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_train_per_class": n_train,
            "n_test_honest": len(test_h),
            "n_test_deceptive": len(test_d),
            "AUROCs": dict(table),
        }, f, indent=2)
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
