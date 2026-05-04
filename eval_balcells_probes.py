"""Evaluate Balcells hallucination probes (Llama 3.3 70B) on our WMDP-bio
interrogation test set. Score per-token, compute AUROC under mean/max/last
pooling, and dump per-token scores + plots for inspection.

Variants tested:
  - llama3_3_70b_linear (a single Linear(8192,1) at layer 76 residual)
  - (optional) llama3_3_70b_lora_lambda_kl_0_05 — LoRA + linear head

Usage:
    PYTHONPATH=. uv run python eval_balcells_probes.py <wmdp_eval_log_path>
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
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Dialogue
from lib.prompts import SUSPECT_PROMPTS, _SUSPECT_USER  # type: ignore
from lib.wmdp_data import format_question_block

MODEL_PATH = "/workspace/models/llama-3.3-70b-instruct"
PADDING: dict[str, dict[str, int]] = {"llama": {"left": 0, "right": 0}}
MAX_LENGTH = 4096
SUSPECT_SYSTEM = SUSPECT_PROMPTS["debate_coach"]
BATCH = 1

BALCELLS_DIR = Path(__file__).parent / "probes/balcells"


class BalcellsLinearProbe:
    """Single Linear(hidden_size, 1) head applied to a specific layer's
    post-residual stream. Output is sigmoid (= P(hallucination))."""

    def __init__(self, probe_dir: Path) -> None:
        cfg = json.load(open(probe_dir / "probe_config.json"))
        self.layer_idx: int = cfg["layer_idx"]
        self.hidden_size: int = cfg["hidden_size"]
        # Keep probe on CPU as float32 for portability; move per-call to match acts.
        self.head = torch.nn.Linear(self.hidden_size, 1, device="cpu", dtype=torch.float32)
        sd = torch.load(probe_dir / "probe_head.bin", map_location="cpu", weights_only=True)
        self.head.load_state_dict({k: v.float() for k, v in sd.items()})
        self.head.eval()

    @torch.no_grad()
    def score_per_token(
        self, hidden_states_post_layer: torch.Tensor
    ) -> torch.Tensor:
        """hidden_states_post_layer: [seq_len, hidden_size] residual stream
        AFTER the configured layer's processing. Returns [seq_len] of P(hallucinated)."""
        hs = hidden_states_post_layer.to("cpu", dtype=torch.float32)
        logits = self.head(hs).squeeze(-1)
        return torch.sigmoid(logits)


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


def per_sample_token_scores(
    model: Any,
    tokenizer: Any,
    sample: Any,
    probe: BalcellsLinearProbe,
) -> tuple[list[str], list[float]]:
    meta = sample.metadata
    question_block = format_question_block(str(meta["question"]), list(meta["choices"]))
    suspect_user = _SUSPECT_USER.format(
        question_block=question_block,
        argue_target=meta["argue_target"],
        argue_target_text=meta["argue_target_text"],
    )
    judge_user = next(m.content for m in sample.messages if m.role == "user")
    argument = extract_argument(str(judge_user))
    dialogue = make_dialogue(suspect_user, argument)
    toks = TokenizedDataset.from_dialogue_list(
        [dialogue], tokenizer, padding=PADDING, max_length=MAX_LENGTH
    )
    input_ids = toks.tokens.to(model.device)
    mask = toks.detection_mask
    assert mask is not None
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    # hidden_states[i] = residual stream after layer i-1 (or input to layer i)
    # We want post-layer probe.layer_idx → hidden_states[layer_idx + 1]
    hs = out.hidden_states[probe.layer_idx + 1][0]  # [seq, hidden]
    detect_mask = mask[0].bool()
    n = min(detect_mask.shape[0], hs.shape[0])
    detect_mask = detect_mask[:n]
    hs = hs[:n]
    kept_hs = hs[detect_mask]
    probs = probe.score_per_token(kept_hs)
    str_tokens = toks.str_tokens[0][:n]
    kept_tokens = [str_tokens[i] for i in range(n) if detect_mask[i].item()]
    return kept_tokens, probs.cpu().tolist()


def auroc(scores: list[float], labels: list[str]) -> float:
    y = [1 if l == "DECEPTIVE" else 0 for l in labels]
    return float(roc_auc_score(y, scores))


def pool(scores: list[float], mode: str) -> float:
    if mode == "mean":
        return sum(scores) / len(scores)
    if mode == "max":
        return max(scores)
    return scores[-1]


def plot_sample(
    sample_label: str,
    tokens: list[str],
    scores: list[float],
    probe_name: str,
    out_path: Path,
) -> None:
    n = len(tokens)
    fig_w = max(12.0, n * 0.18)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 4.0))
    x = list(range(n))
    colors = ["tab:red" if s > 0.5 else "tab:blue" for s in scores]
    ax.bar(x, scores, color=colors, width=0.85)
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"{probe_name}\nP(hallucination)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("\n", "\\n") for t in tokens],
        rotation=90,
        fontsize=6,
        family="monospace",
    )
    fig.suptitle(
        f"{sample_label}: per-token Balcells {probe_name} score",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: eval_balcells_probes.py <wmdp_eval_log_path>", file=sys.stderr)
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

    print("\nLoading Llama 3.3 70B (BF16)...")
    from lib.model_loading import load_llama_strict
    model, tokenizer = load_llama_strict(MODEL_PATH, per_gpu_gib=75)
    dm = getattr(model, "hf_device_map", {})
    print(f"  hf_device_map sample: {list(dm.values())[:5] if dm else 'single device'}")

    probe = BalcellsLinearProbe(BALCELLS_DIR / "llama3_3_70b_linear")
    print(f"Balcells linear probe loaded: layer_idx={probe.layer_idx} hidden_size={probe.hidden_size}")

    all_outputs: list[dict[str, Any]] = []
    for label, samples in [("honest", test_h), ("deceptive", test_d)]:
        for i, s in enumerate(samples):
            print(f"  {label} {i+1}/{len(samples)}", end="\r")
            tokens, scores = per_sample_token_scores(model, tokenizer, s, probe)
            all_outputs.append({
                "label": label,
                "sample_id": s.id,
                "argue_target": s.metadata["argue_target"],
                "tokens": tokens,
                "scores": scores,
            })
        print()

    # Compute AUROC
    pooled_scores: dict[str, list[float]] = defaultdict(list)
    label_list: list[str] = []
    for o in all_outputs:
        label_list.append("DECEPTIVE" if o["label"] == "deceptive" else "HONEST")
        for mode in ("mean", "max", "last"):
            pooled_scores[mode].append(pool(o["scores"], mode))

    print(f"\n{'Pool':<8} {'AUROC':>8}")
    print("-" * 18)
    auroc_table: dict[str, float] = {}
    for mode in ("mean", "max", "last"):
        a = auroc(pooled_scores[mode], label_list)
        auroc_table[mode] = a
        print(f"{mode:<8} {a:>8.4f}")

    # Plot 1 honest + 1 deceptive
    out_dir = Path(__file__).parent
    for o in all_outputs:
        if o["label"] == "honest":
            plot_sample(
                f"WMDP HONEST argue={o['argue_target']}",
                o["tokens"], o["scores"],
                "linear (layer 76)",
                out_dir / f"balcells_linear_honest_argue{o['argue_target']}.png",
            )
            break
    for o in all_outputs:
        if o["label"] == "deceptive":
            plot_sample(
                f"WMDP DECEPTIVE argue={o['argue_target']}",
                o["tokens"], o["scores"],
                "linear (layer 76)",
                out_dir / f"balcells_linear_deceptive_argue{o['argue_target']}.png",
            )
            break

    out_json = out_dir / "balcells_linear_results.json"
    json.dump({
        "n_test_honest": len(test_h),
        "n_test_deceptive": len(test_d),
        "AUROCs_balcells_linear": auroc_table,
        "per_sample": all_outputs,
    }, open(out_json, "w"), indent=2)
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
