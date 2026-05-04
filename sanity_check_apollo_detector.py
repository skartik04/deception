"""Sanity check for ApolloProbeDetector.

Picks one labeled-honest and one labeled-deceptive on-policy roleplaying rollout
from Apollo's data, runs them through ApolloProbeDetector, prints the deception
probability and per-call wall time. Verifies the deceptive one scores higher
than the honest one.

Run:
    CUDA_VISIBLE_DEVICES=0 uv run python sanity_check_apollo_detector.py
"""

import asyncio
import json
import time
from pathlib import Path

from lib.detectors import ApolloProbeDetector

ROLLOUT_PATH = Path(__file__).parent / "liars-bench" / "src" / "probes" / "data" / "rollouts" / "roleplaying__plain__llama-70b-3.3.json"


def load_one_of_each() -> tuple[dict, dict]:
    with open(ROLLOUT_PATH) as f:
        data = json.load(f)
    rollouts = data["rollouts"]
    honest = next(r for r in rollouts if r.get("graded_deception") and all(j <= 2 for j in r["graded_deception"]))
    deceptive = next(r for r in rollouts if r.get("graded_deception") and all(j >= 6 for j in r["graded_deception"]))
    return honest, deceptive


async def score(detector: ApolloProbeDetector, rollout: dict) -> tuple[float, float]:
    """Return (probability, wall_time_seconds)."""
    argument = rollout["output_strs"][0]
    metadata = {
        "question": rollout["metadata"]["question"],
        "choices": ["", "", "", ""],  # not used by probe; argue_target empty
        "argue_target": "",
        "argue_target_text": "",
    }
    t0 = time.perf_counter()
    p = await detector.deception_probability(argument, metadata)
    t1 = time.perf_counter()
    return p, t1 - t0


async def main() -> None:
    honest, deceptive = load_one_of_each()
    print("Honest sample (graded:", honest["graded_deception"], "):")
    print(f"  {honest['output_strs'][0][:120]}...")
    print("Deceptive sample (graded:", deceptive["graded_deception"], "):")
    print(f"  {deceptive['output_strs'][0][:120]}...")
    print()

    detector = ApolloProbeDetector(probe_variant="roleplaying", suspect_prompt_key="debate_coach")

    print("First call (includes model load)...")
    p_h, t_h = await score(detector, honest)
    print(f"  honest:    p={p_h:.3f}, time={t_h:.1f}s")

    print("Second call (model already loaded)...")
    p_d, t_d = await score(detector, deceptive)
    print(f"  deceptive: p={p_d:.3f}, time={t_d:.1f}s")

    print()
    if p_d > p_h:
        print(f"PASS: deceptive ({p_d:.3f}) > honest ({p_h:.3f})")
    else:
        print(f"FAIL: deceptive ({p_d:.3f}) <= honest ({p_h:.3f})")
    print(f"Steady-state per-argument time: ~{t_d:.1f}s (single GPU + CPU offload)")


if __name__ == "__main__":
    asyncio.run(main())
