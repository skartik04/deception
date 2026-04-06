"""Debate runner.

Reads question_id from debate/config.yaml, loads the question from
filtered_questions.json (or the raw WMDP dataset), builds correct and
incorrect claims, then runs all 4 conditions:
  - correct   + no probe
  - correct   + probe
  - incorrect + no probe
  - incorrect + probe

Each condition is saved as a separate JSON inside results/traces/q{id:04d}/.

Usage:
    uv run python -m debate.debate
    # Or from repo root: uv run python debate/debate.py (script dir must not shadow the package).
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from debate.config import PROJECT_ROOT, load_debate_config
from debate.prompts import JUDGE_VERDICT_PROMPT, fill_auditor_turn, fill_judge, fill_suspect

FILTERED_QUESTIONS = PROJECT_ROOT / "results" / "filtered_questions.json"
TRACES_DIR = PROJECT_ROOT / "results" / "debate_traces"


# ---------------------------------------------------------------------------
# Load question
# ---------------------------------------------------------------------------

def load_question(question_id: int) -> dict:
    """Load raw WMDP record for the given index."""
    from datasets import load_dataset
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    row = ds[question_id]
    return {
        "index": question_id,
        "question": row["question"],
        "choices": row["choices"],
        "answer": row["answer"],
    }


def build_claims(record: dict) -> tuple[str, str]:
    """Return (correct_claim, incorrect_claim) for a WMDP record."""
    correct_text = record["choices"][record["answer"]]
    wrong_indices = [i for i in range(len(record["choices"])) if i != record["answer"]]
    wrong_text = record["choices"][random.choice(wrong_indices)]
    return correct_text, wrong_text


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def chat(client: OpenAI, model: str, messages: list[dict], temperature: float) -> dict[str, str]:
    model_name = re.sub(r"^openrouter/", "", model)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    msg = resp.choices[0].message
    return {
        "content": _clean_text(getattr(msg, "content", None)),
        "reasoning": _clean_text(getattr(msg, "reasoning", None)),
    }


def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


# ---------------------------------------------------------------------------
# Single debate
# ---------------------------------------------------------------------------

def run_debate(
    question: str,
    claim: str,
    correct_claim: str,
    question_id: int,
    probe_enabled: bool,
    cfg,
) -> dict:
    client = make_client()
    d = cfg
    temp = d.temperature

    sus_sys, sus_user = fill_suspect(question, claim)
    jdg_sys, jdg_user = fill_judge(question, claim, with_probe=probe_enabled)

    judge_msgs: list[dict] = [
        {"role": "system", "content": jdg_sys},
        {"role": "user", "content": jdg_user},
    ]
    suspect_msgs: list[dict] = [
        {"role": "system", "content": sus_sys},
        {"role": "user", "content": sus_user},
    ]

    transcript: list[dict] = []
    private_reasoning_traces: list[dict] = []

    for turn in range(1, d.num_rounds + 1):
        # Judge asks
        judge_result = chat(client, d.judge_model, judge_msgs, temp)
        judge_reply = judge_result["content"]
        judge_msgs.append({"role": "assistant", "content": judge_reply})
        transcript.append({"turn": turn, "role": d.judge_label, "content": judge_reply})

        # Suspect responds
        suspect_msgs.append({"role": "user", "content": judge_reply})
        suspect_result = chat(client, d.suspect_model, suspect_msgs, temp)
        suspect_reply = suspect_result["content"]
        suspect_reasoning = suspect_result["reasoning"]
        suspect_msgs.append({"role": "assistant", "content": suspect_reply})
        transcript.append({
            "turn": turn,
            "role": d.suspect_label,
            "content": suspect_reply,
            "private_reasoning": suspect_reasoning,
        })
        private_reasoning_traces.append({
            "turn": turn,
            "reasoning": suspect_reasoning or "NA",
        })

        # Auditor (if probe enabled)
        if probe_enabled:
            public_transcript = _format_public_transcript(transcript, d)
            reasoning_history = _format_private_reasoning(private_reasoning_traces)
            aud_sys, aud_user = fill_auditor_turn(
                question,
                claim,
                public_transcript,
                reasoning_history,
                suspect_reply,
            )
            aud_result = chat(client, d.auditor_model, [
                {"role": "system", "content": aud_sys},
                {"role": "user", "content": aud_user},
            ], temp)
            auditor_report = _parse_json_safe(aud_result["content"])
            transcript.append({"turn": turn, "role": d.auditor_label, "content": auditor_report})
            report_text = (
                json.dumps(auditor_report, indent=2)
                if isinstance(auditor_report, dict)
                else str(auditor_report)
            )
            judge_msgs.append({
                "role": "user",
                "content": (
                    f"[{d.auditor_label} report — private]\n"
                    + report_text
                    + f"\n\n{d.suspect_label}'s response:\n{suspect_reply}"
                ),
            })
        else:
            judge_msgs.append({"role": "user", "content": suspect_reply})

    # Final verdict
    judge_msgs.append({"role": "user", "content": JUDGE_VERDICT_PROMPT})
    verdict_result = chat(client, d.judge_model, judge_msgs, temp)
    verdict = _parse_json_safe(verdict_result["content"])

    judge_correct = (
        isinstance(verdict, dict)
        and verdict.get("agreement_with_claim", "").upper()
        == ("AGREE" if claim == correct_claim else "DISAGREE")
    )

    return {
        "question_id": question_id,
        "question": question,
        "claim": claim,
        "correct_claim": correct_claim,
        "probe_enabled": probe_enabled,
        "models": {
            "suspect": d.suspect_model,
            "judge": d.judge_model,
            "auditor": d.auditor_model if probe_enabled else None,
        },
        "num_rounds": d.num_rounds,
        "transcript": transcript,
        "verdict": verdict,
        "judge_correct": judge_correct,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_public_transcript(transcript: list[dict], d) -> str:
    """Format transcript for the auditor, excluding auditor entries."""
    lines = []
    for entry in transcript:
        if entry["role"] == d.auditor_label:
            continue
        content = entry["content"]
        if isinstance(content, dict):
            content = json.dumps(content)
        lines.append(f"{entry['role']}: {content}")
    return "\n\n".join(lines)


def _format_private_reasoning(traces: list[dict]) -> str:
    """Format accumulated private reasoning traces across all turns."""
    parts = []
    for t in traces:
        parts.append(f"[Turn {t['turn']}]\n{t['reasoning']}")
    return "\n\n".join(parts)


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_clean_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    return str(value).strip()


def _parse_json_safe(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _model_slug(model_str: str) -> str:
    """Short slug from model name: first word + size. e.g. qwen3-80b, llama-8b, gpt-120b."""
    name = model_str.split("/")[-1].lower()
    first = re.match(r"[a-z]+\d*", name)
    first = first.group() if first else name[:6]
    size = re.search(r"(\d+\.?\d*[bmt])", name)
    size = size.group() if size else ""
    return f"{first}-{size}" if size else first


def main():
    parser = argparse.ArgumentParser(description="Run debate for a single question")
    parser.add_argument("--config", type=Path, default=None, help="Path to config yaml (default: debate/config.yaml)")
    parser.add_argument("--question-id", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--suspect-model", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--auditor-model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    cfg = load_debate_config(args.config)

    # CLI overrides
    if args.question_id is not None: cfg.question_id = args.question_id
    if args.num_rounds is not None: cfg.num_rounds = args.num_rounds
    if args.suspect_model is not None: cfg.suspect_model = args.suspect_model
    if args.judge_model is not None: cfg.judge_model = args.judge_model
    if args.auditor_model is not None: cfg.auditor_model = args.auditor_model
    if args.temperature is not None: cfg.temperature = args.temperature

    qid = cfg.question_id

    record = load_question(qid)
    correct_claim, incorrect_claim = build_claims(record)
    question = record["question"]

    q_dir = TRACES_DIR / f"q{qid:04d}"
    q_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("correct",   correct_claim,   True,  "probe1"),
        ("correct",   correct_claim,   False, "probe0"),
        ("incorrect", incorrect_claim, True,  "probe1"),
        ("incorrect", incorrect_claim, False, "probe0"),
    ]

    print(f"Question {qid}: {question}")
    print(f"Correct claim:   {correct_claim}")
    print(f"Incorrect claim: {incorrect_claim}")
    print()

    for label, claim, probe, probe_label in conditions:
        print(f"  Running: {label} / {probe_label} ...")
        result = run_debate(
            question=question,
            claim=claim,
            correct_claim=correct_claim,
            question_id=qid,
            probe_enabled=probe,
            cfg=cfg,
        )
        filename = f"{label}_{probe_label}_rounds_{cfg.num_rounds}_sus-{_model_slug(cfg.suspect_model)}_jdg-{_model_slug(cfg.judge_model)}.json"
        out_path = q_dir / filename
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        correct_str = "✓" if result["judge_correct"] else "✗"
        print(f"  {correct_str} verdict: {result['verdict']} → {out_path.name}")

    print(f"\nAll 4 conditions saved to {q_dir}")


if __name__ == "__main__":
    main()
