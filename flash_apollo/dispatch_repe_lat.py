"""Local dispatcher: build payload, POST to /run, poll /status, save result.

Run from /workspace/gergely_deception/flash_apollo:
    uv run --with httpx --with inspect-ai --with pandas python dispatch_repe_lat.py
"""

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import httpx

ENDPOINT_ID = os.environ.get("REPE_LAT_ENDPOINT_ID")  # set after flash deploy
API_KEY = os.environ["RUNPOD_API_KEY"]

REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE_FALSE_CSV = REPO_ROOT / "liars-bench/src/probes/data/repe/true_false_facts.csv"
WMDP_LOG = REPO_ROOT / "logs/2026-05-03T20-52-41-00-00_interrogation-single-turn_GAWkrSWEdSrTh7V9DiYPzZ.eval"


def build_payload() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "liars-bench/src/probes"))
    from inspect_ai.log import read_eval_log
    sys.path.insert(0, str(REPO_ROOT))
    from lib.prompts import SUSPECT_PROMPTS, _SUSPECT_USER  # type: ignore
    from lib.wmdp_data import format_question_block

    with open(TRUE_FALSE_CSV) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    true_stmts = [r["statement"] for r in rows if r["label"] == "1"]

    log = read_eval_log(str(WMDP_LOG))
    assert log.samples is not None
    h = [s for s in log.samples if not s.metadata["is_deceptive"]]
    d = [s for s in log.samples if s.metadata["is_deceptive"]]
    n = min(len(h), len(d))
    n_test = max(20, n // 4)
    n_skip = n - n_test
    test_h = h[n_skip : n_skip + n_test]
    test_d = d[n_skip : n_skip + n_test]
    suspect_system = SUSPECT_PROMPTS["debate_coach"]

    def serialize(sample) -> dict:
        meta = sample.metadata
        qb = format_question_block(str(meta["question"]), list(meta["choices"]))
        suspect_user = _SUSPECT_USER.format(
            question_block=qb,
            argue_target=meta["argue_target"],
            argue_target_text=meta["argue_target_text"],
        )
        judge_user = next(m.content for m in sample.messages if m.role == "user")
        parts = str(judge_user).split("---\n", 1)
        argument = parts[1].rsplit("\n---", 1)[0]
        return {
            "argue_target": str(meta["argue_target"]),
            "system": suspect_system,
            "user": suspect_user,
            "argument": argument,
        }

    return {
        "true_statements": true_stmts,
        "test_h": [serialize(s) for s in test_h],
        "test_d": [serialize(s) for s in test_d],
        "n_train_pairs": 256,
        "detector_layers": list(range(30, 50)),
    }


async def main() -> None:
    if not ENDPOINT_ID:
        print("set REPE_LAT_ENDPOINT_ID env var (from flash deploy output)", file=sys.stderr)
        sys.exit(1)
    payload = build_payload()
    print(
        f"[local] payload: {len(payload['true_statements'])} stmts, "
        f"{len(payload['test_h'])} test_h, {len(payload['test_d'])} test_d"
    )

    base = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=120.0)) as client:
        resp = await client.post(f"{base}/run", json={"input": {"payload": payload}}, headers=headers)
        resp.raise_for_status()
        job_id = resp.json()["id"]
        print(f"[local] job_id={job_id}; polling /status")

        start = time.time()
        last_status = ""
        while True:
            r = await client.get(f"{base}/status/{job_id}", headers=headers)
            r.raise_for_status()
            data = r.json()
            status = data.get("status", "UNKNOWN")
            elapsed = int(time.time() - start)
            if status != last_status:
                print(f"  [{elapsed:>5}s] status={status}")
                last_status = status
            else:
                print(f"  [{elapsed:>5}s] status={status}", end="\r", flush=True)
            if status in ("COMPLETED", "FAILED", "CANCELLED"):
                print()
                break
            await asyncio.sleep(15)

    if status != "COMPLETED":
        print(f"[local] job ended with status={status}")
        print(json.dumps(data, indent=2)[:2000])
        sys.exit(1)

    result = data.get("output", {})
    out_path = REPO_ROOT / "repe_lat_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[local] saved {out_path}")
    print(f"[local] AUROCs: {result.get('AUROCs')}")


if __name__ == "__main__":
    asyncio.run(main())
