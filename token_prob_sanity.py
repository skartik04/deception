"""Pick one suspect trajectory and forward-pass it locally.
Compute per-token statistics under our local BF16 Llama 70B for the
OpenRouter-generated text:
  - log-probability of the actually-emitted token at each position
  - rank of the actual token (top-1, top-5, top-10, ...)
  - perplexity over the suspect's response

If OR-API and local BF16 are the same effective model, most tokens should
be top-1 with high probability (low perplexity). If providers serve a
quantized variant (FP8 etc.), expect more divergence.

Usage:
    PYTHONPATH=. uv run python token_prob_sanity.py <eval_log_path> [sample_idx]
"""

import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from inspect_ai.log import read_eval_log
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "liars-bench" / "src" / "probes"))

from lib.prompts import SUSPECT_PROMPTS, _SUSPECT_USER  # type: ignore
from lib.wmdp_data import format_question_block

MODEL_PATH = "/workspace/models/llama-3.3-70b-instruct"


def extract_argument(judge_user_content: str) -> str:
    parts = judge_user_content.split("---\n", 1)
    rest = parts[1]
    end = rest.rsplit("\n---", 1)
    return end[0]


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: token_prob_sanity.py <eval_log_path> [sample_idx]", file=sys.stderr)
        sys.exit(1)
    eval_log_path = sys.argv[1]
    idx = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    log = read_eval_log(eval_log_path)
    assert log.samples is not None
    sample = log.samples[idx]
    meta = sample.metadata
    is_deceptive = bool(meta["is_deceptive"])
    print(f"Sample {idx}: argue={meta['argue_target']} deceptive={is_deceptive}")

    question_block = format_question_block(str(meta["question"]), list(meta["choices"]))
    suspect_user = _SUSPECT_USER.format(
        question_block=question_block,
        argue_target=meta["argue_target"],
        argue_target_text=meta["argue_target_text"],
    )
    judge_user = next(m.content for m in sample.messages if m.role == "user")
    argument = extract_argument(str(judge_user))
    print(f"  argument length: {len(argument)} chars")

    print("\nLoading model (BF16)...")
    from lib.model_loading import load_llama_strict
    model, tokenizer = load_llama_strict(MODEL_PATH, per_gpu_gib=78)
    suspect_messages = [
        {"role": "system", "content": SUSPECT_PROMPTS["debate_coach"]},
        {"role": "user", "content": suspect_user},
    ]
    prompt_ids: torch.Tensor = tokenizer.apply_chat_template(
        suspect_messages, add_generation_prompt=True, return_tensors="pt"
    )
    arg_ids: torch.Tensor = tokenizer(
        argument, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    full_ids = torch.cat([prompt_ids, arg_ids], dim=1).to(model.device)
    n_prompt = prompt_ids.shape[1]
    n_arg = arg_ids.shape[1]
    print(f"  prompt tokens: {n_prompt}, argument tokens: {n_arg}")

    print("Forward pass...")
    with torch.no_grad():
        out = model(full_ids)
    logits = out.logits[0]  # [seq_len, vocab]
    # We want: at position p (0-indexed), the model predicted the token at p+1.
    # Argument tokens occupy positions [n_prompt, n_prompt + n_arg).
    # The logits that PREDICT them are at positions [n_prompt - 1, n_prompt + n_arg - 1).
    pred_logits = logits[n_prompt - 1 : n_prompt + n_arg - 1, :]  # [n_arg, vocab]
    target_ids = full_ids[0, n_prompt : n_prompt + n_arg]  # [n_arg]

    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    target_logp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    sorted_logp, sorted_ids = torch.sort(log_probs, dim=-1, descending=True)
    target_rank = (sorted_ids == target_ids.unsqueeze(1)).int().argmax(dim=1)

    target_logp_list = target_logp.cpu().tolist()
    target_rank_list = target_rank.cpu().tolist()

    n = len(target_logp_list)
    mean_logp = statistics.mean(target_logp_list)
    median_logp = statistics.median(target_logp_list)
    perplexity = math.exp(-mean_logp)
    top1 = sum(1 for r in target_rank_list if r == 0) / n
    top5 = sum(1 for r in target_rank_list if r < 5) / n
    top10 = sum(1 for r in target_rank_list if r < 10) / n
    top100 = sum(1 for r in target_rank_list if r < 100) / n
    fraction_logp_lt_neg_5 = sum(1 for x in target_logp_list if x < -5) / n
    fraction_logp_lt_neg_10 = sum(1 for x in target_logp_list if x < -10) / n
    worst5_idx = sorted(range(n), key=lambda i: target_logp_list[i])[:5]
    worst5 = []
    for i in worst5_idx:
        actual_tok = tokenizer.decode([int(target_ids[i].item())])
        local_top1 = tokenizer.decode([int(sorted_ids[i, 0].item())])
        worst5.append({
            "pos": i,
            "actual": actual_tok,
            "local_top1": local_top1,
            "logp": target_logp_list[i],
            "rank": target_rank_list[i],
        })

    print(f"\n=== Token-level statistics over the suspect's {n} argument tokens ===")
    print(f"Mean log-prob (under local BF16):       {mean_logp:.4f}")
    print(f"Median log-prob:                        {median_logp:.4f}")
    print(f"Perplexity (= exp(-mean_logp)):         {perplexity:.3f}")
    print(f"Fraction top-1 (= local model's choice): {top1:.1%}")
    print(f"Fraction top-5:                          {top5:.1%}")
    print(f"Fraction top-10:                         {top10:.1%}")
    print(f"Fraction top-100:                        {top100:.1%}")
    print(f"Fraction log-prob < -5:                  {fraction_logp_lt_neg_5:.1%}")
    print(f"Fraction log-prob < -10:                 {fraction_logp_lt_neg_10:.1%}")
    print(f"\nLowest-prob tokens (most surprising under local BF16):")
    for w in worst5:
        print(
            f"  pos {w['pos']:>4}  actual={w['actual']!r:<15} "
            f"local_top1={w['local_top1']!r:<15} "
            f"logp={w['logp']:>7.3f} rank={w['rank']:>5}"
        )


if __name__ == "__main__":
    main()
