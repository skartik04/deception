"""Unified probe evaluation: load Llama 70B once, run any subset of registered
probes against the WMDP-bio interrogation test split (or any inspect_ai eval
log with the same metadata schema), report AUROCs under mean/max/last pooling,
and dump per-token scores to a JSON for downstream plotting.

Replaces: eval_balcells_probes.py, eval_balcells_lora.py, eval_descriptive_probe.py,
eval_repe_lat.py, eval_probes_bfloat16.py, eval_probes_comparison.py, probe_sweep.py,
score_on_answer_only.py, sanity_check_apollo_detector.py, combined_probe_run.py.

Usage:
    PYTHONPATH=. uv run python eval_probes.py <wmdp_eval_log_path> \
        --probes apollo_roleplaying balcells_linear

    # All available probes (slow):
    PYTHONPATH=. uv run python eval_probes.py <wmdp_eval_log_path> --all

LoRA-wrapped probes (e.g. balcells_lora_kl) are run in their own pass — the
engine refuses to mix them with non-LoRA probes (different forward graph).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log
from sklearn.metrics import roc_auc_score

from lib.probe_engine import ProbeEngine
from lib.probes import PerTokenScore, list_available_probes, load_probe
from lib.prompts import SUSPECT_PROMPTS

MODEL_PATH = "/workspace/models/llama-3.3-70b-instruct"
SUSPECT_SYSTEM = SUSPECT_PROMPTS["debate_coach"]


def _extract_argument(judge_user_content: str) -> str:
    parts = judge_user_content.split("---\n", 1)
    rest = parts[1]
    end = rest.rsplit("\n---", 1)
    return end[0]


def _split_test(log_path: str) -> tuple[list[Any], list[Any]]:
    log = read_eval_log(log_path)
    assert log.samples is not None
    honest = [s for s in log.samples if not s.metadata["is_deceptive"]]
    deceptive = [s for s in log.samples if s.metadata["is_deceptive"]]
    n = min(len(honest), len(deceptive))
    n_test = max(20, n // 4)
    n_train = n - n_test
    return honest[n_train : n_train + n_test], deceptive[n_train : n_train + n_test]


def _argument_for(sample: Any) -> str:
    judge_user = next(m.content for m in sample.messages if m.role == "user")
    return _extract_argument(str(judge_user))


def _run_engine(
    engine: ProbeEngine, samples: list[Any], label: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        print(f"  {label} {i+1}/{len(samples)}", end="\r", flush=True)
        scored: dict[str, PerTokenScore] = engine.score_argument(
            s.metadata, _argument_for(s)
        )
        out.append(
            {
                "label": label,
                "sample_id": s.id,
                "argue_target": s.metadata["argue_target"],
                "is_deceptive": bool(s.metadata["is_deceptive"]),
                "tokens": next(iter(scored.values())).tokens,
                "scores": {name: ps.scores for name, ps in scored.items()},
            }
        )
    print()
    return out


def _aurocs(per_sample: list[dict[str, Any]], probe_names: list[str]) -> dict[
    str, dict[str, float]
]:
    table: dict[str, dict[str, float]] = {}
    labels = [1 if r["is_deceptive"] else 0 for r in per_sample]
    if len(set(labels)) < 2:
        return table
    for probe in probe_names:
        per_pool: dict[str, list[float]] = defaultdict(list)
        for r in per_sample:
            ps = PerTokenScore(tokens=r["tokens"], scores=r["scores"][probe])
            for mode in ("mean", "max", "last"):
                per_pool[mode].append(ps.pool(mode))
        table[probe] = {
            mode: float(roc_auc_score(labels, per_pool[mode])) for mode in ("mean", "max", "last")
        }
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_log", help="Path to inspect_ai .eval log file")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--probes", nargs="+", help="Probe names to run (see --list)"
    )
    grp.add_argument("--all", action="store_true", help="Run every available probe")
    grp.add_argument(
        "--list", action="store_true", help="Show available probe names and exit"
    )
    p.add_argument(
        "--out",
        default="results/eval_probes.json",
        help="Output JSON path (relative to repo root)",
    )
    p.add_argument("--per-gpu-gib", type=int, default=70)
    p.add_argument("--max-length", type=int, default=4096)
    args = p.parse_args()

    if args.list:
        for n in list_available_probes():
            print(n)
        return

    requested = list_available_probes() if args.all else list(args.probes)
    print(f"Probes requested: {requested}")
    probes = [load_probe(n) for n in requested]

    # Split into LoRA / non-LoRA passes.
    lora_probes = [p for p in probes if p.needs_lora_adapter]
    plain_probes = [p for p in probes if not p.needs_lora_adapter]
    passes: list[list] = []
    if plain_probes:
        passes.append(plain_probes)
    for lp in lora_probes:
        passes.append([lp])
    print(f"Forward-pass groups: {[[p.name for p in g] for g in passes]}")

    test_h, test_d = _split_test(args.eval_log)
    print(f"Test split: {len(test_h)} honest + {len(test_d)} deceptive")

    all_per_sample: list[dict[str, Any]] = []
    all_probe_names: list[str] = []
    for group_i, group in enumerate(passes):
        print(f"\n=== Pass {group_i+1}/{len(passes)}: {[p.name for p in group]} ===")
        engine = ProbeEngine(
            model_path=MODEL_PATH,
            probes=group,
            suspect_system=SUSPECT_SYSTEM,
            max_length=args.max_length,
            per_gpu_gib=args.per_gpu_gib,
        )
        engine.load()
        h_results = _run_engine(engine, test_h, "honest")
        d_results = _run_engine(engine, test_d, "deceptive")
        # Merge into the global per_sample list. Keyed by sample_id+label.
        if not all_per_sample:
            # First pass — initialize.
            all_per_sample = h_results + d_results
        else:
            # Subsequent passes — merge probe scores into existing rows.
            by_key = {(r["sample_id"], r["label"]): r for r in all_per_sample}
            for r in h_results + d_results:
                key = (r["sample_id"], r["label"])
                if key in by_key:
                    by_key[key]["scores"].update(r["scores"])
                else:
                    all_per_sample.append(r)
        all_probe_names.extend(p.name for p in group)
        # Free the engine before loading the next.
        del engine
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    aurocs = _aurocs(all_per_sample, all_probe_names)
    print(f"\n{'Probe':<28} {'mean':>7} {'max':>7} {'last':>7}")
    print("-" * 52)
    for name, row in aurocs.items():
        print(f"{name:<28} {row['mean']:>7.4f} {row['max']:>7.4f} {row['last']:>7.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "n_test_honest": len(test_h),
                "n_test_deceptive": len(test_d),
                "AUROCs": aurocs,
                "per_sample": all_per_sample,
                "model_path": MODEL_PATH,
                "suspect_system_key": "debate_coach",
                "eval_log": args.eval_log,
            },
            indent=2,
        )
    )
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
