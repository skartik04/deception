"""Evaluate Balcells LoRA-KL hallucination probe (Llama 3.3 70B + LoRA adapter
+ linear head at layer 76). Same WMDP-bio test set as eval_balcells_probes.py.

Usage:
    PYTHONPATH=. uv run python eval_balcells_lora.py <wmdp_eval_log_path>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from inspect_ai.log import read_eval_log
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "liars-bench" / "src" / "probes"))

from deception_detection.data.base import Message
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Dialogue
from lib.prompts import SUSPECT_PROMPTS, _SUSPECT_USER  # type: ignore
from lib.wmdp_data import format_question_block

MODEL_PATH = "/workspace/models/llama-3.3-70b-instruct"
PADDING: dict[str, dict[str, int]] = {"llama": {"left": 0, "right": 0}}
MAX_LENGTH = 4096
SUSPECT_SYSTEM = SUSPECT_PROMPTS["debate_coach"]
BALCELLS_DIR = Path(__file__).parent / "probes/balcells/llama3_3_70b_lora_lambda_kl_0_05"


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
        print("usage: eval_balcells_lora.py <wmdp_eval_log_path>", file=sys.stderr)
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

    print("\nLoading Llama 3.3 70B (BF16, GPU-only)...")
    from lib.model_loading import assert_no_cpu_params, load_llama_strict
    base, tokenizer = load_llama_strict(MODEL_PATH, per_gpu_gib=75)

    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(BALCELLS_DIR))
    assert_no_cpu_params(model)
    model.eval()

    cfg = json.load(open(BALCELLS_DIR / "probe_config.json"))
    layer_idx: int = cfg["layer_idx"]
    hidden_size: int = cfg["hidden_size"]
    head = torch.nn.Linear(hidden_size, 1, device="cpu", dtype=torch.float32)
    sd = torch.load(BALCELLS_DIR / "probe_head.bin", map_location="cpu", weights_only=True)
    head.load_state_dict({k: v.float() for k, v in sd.items()})
    head.eval()

    all_outputs: list[dict[str, Any]] = []
    for label, samples in [("honest", test_h), ("deceptive", test_d)]:
        for i, s in enumerate(samples):
            print(f"  {label} {i+1}/{len(samples)}", end="\r")
            meta = s.metadata
            qb = format_question_block(str(meta["question"]), list(meta["choices"]))
            suspect_user = _SUSPECT_USER.format(
                question_block=qb,
                argue_target=meta["argue_target"],
                argue_target_text=meta["argue_target_text"],
            )
            judge_user = next(m.content for m in s.messages if m.role == "user")
            argument = extract_argument(str(judge_user))
            dialogue = make_dialogue(suspect_user, argument)
            toks = TokenizedDataset.from_dialogue_list(
                [dialogue], tokenizer, padding=PADDING, max_length=MAX_LENGTH
            )
            input_ids = toks.tokens.to(base.device)
            mask = toks.detection_mask
            assert mask is not None
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[layer_idx + 1][0]
            detect_mask = mask[0].bool()
            n_pair = min(detect_mask.shape[0], hs.shape[0])
            kept_hs = hs[:n_pair][detect_mask[:n_pair]]
            kept_hs_cpu = kept_hs.to("cpu", dtype=torch.float32)
            with torch.no_grad():
                logits = head(kept_hs_cpu).squeeze(-1)
                probs = torch.sigmoid(logits).tolist()
            all_outputs.append({
                "label": label,
                "argue_target": meta["argue_target"],
                "scores": probs,
            })
        print()

    pooled: dict[str, list[float]] = defaultdict(list)
    labels_list: list[str] = []
    for o in all_outputs:
        labels_list.append("DECEPTIVE" if o["label"] == "deceptive" else "HONEST")
        sc = o["scores"]
        pooled["mean"].append(sum(sc) / len(sc))
        pooled["max"].append(max(sc))
        pooled["last"].append(sc[-1])

    print(f"\n{'Pool':<8} {'AUROC':>8}")
    print("-" * 18)
    aurocs: dict[str, float] = {}
    for mode in ("mean", "max", "last"):
        y = [1 if l == "DECEPTIVE" else 0 for l in labels_list]
        aurocs[mode] = float(roc_auc_score(y, pooled[mode]))
        print(f"{mode:<8} {aurocs[mode]:>8.4f}")

    out_json = Path(__file__).parent / "balcells_lora_kl_results.json"
    json.dump({
        "n_test_honest": len(test_h),
        "n_test_deceptive": len(test_d),
        "AUROCs": aurocs,
        "per_sample_scores": [o["scores"] for o in all_outputs],
        "per_sample_labels": [o["label"] for o in all_outputs],
    }, open(out_json, "w"), indent=2)
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
