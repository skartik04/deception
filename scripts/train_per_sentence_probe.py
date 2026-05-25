"""Train a linear probe on per-sentence activations.

Pipeline:
  1. Read per-sentence labels from a `run_per_sentence_probe.py` JSON.
  2. Reconstruct each sample's (system, user, assistant, sentence_spans, labels).
  3. POST to a deployed Flash endpoint (`extract-activations`) which returns
     per-sentence pooled activations at chosen layers.
  4. Train a logistic-regression probe per layer.
  5. Save the best layer's probe in liars-bench `LogisticRegressionDetector`
     .pt format so it loads through `lib.probes.ApolloLRProbe`.

Usage:
    OPENROUTER_API_KEY=... RUNPOD_API_KEY=... uv run python \
      scripts/train_per_sentence_probe.py \
        --labels results/per_sentence/main_run_prefix.json \
        --eval-log logs/2026-05-03T...eval \
        --endpoint-id <runpod-endpoint-id> \
        --layers 22 50 76 \
        --out probes/per_sentence_behavioral.pt \
        --train-frac 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.per_sentence_behavioral import load_eval_argument_blob
from lib.prompts import SUSPECT_PROMPTS, _SUSPECT_USER
from lib.wmdp_data import format_question_block

SUSPECT_SYSTEM_KEY = "debate_coach"


def _build_assistant_payload(probe_json: dict, eval_blob: dict) -> list[dict[str, Any]]:
    """Reconstruct (system, user, assistant, sentence_spans, labels) per sample."""
    by_sid = {s["sample_id"]: s for s in eval_blob["samples"]}
    out: list[dict[str, Any]] = []
    suspect_system = SUSPECT_PROMPTS[SUSPECT_SYSTEM_KEY]
    for s in probe_json["samples"]:
        sid = s["sample_id"]
        ev = by_sid.get(sid)
        if ev is None:
            continue
        suspect_user = _SUSPECT_USER.format(
            question_block=ev["question_block"],
            argue_target=ev["argue_target"],
            argue_target_text=ev["argue_target_text"],
        )
        spans: list[list[int]] = []
        labels: list[float] = []
        for v in s["verdicts"]:
            spans.append([int(v["start"]), int(v["end"])])
            labels.append(float(v["score"]))
        out.append(
            {
                "sample_id": str(sid),
                "system": suspect_system,
                "user": suspect_user,
                "assistant": ev["argument"],
                "sentence_spans": spans,
                "labels": labels,
            }
        )
    return out


def _post_flash(
    *,
    endpoint_id: str,
    api_key: str,
    samples_chunk: list[dict],
    layers: list[int],
    timeout_s: float = 7200.0,
) -> dict[str, Any]:
    """POST a batch to the extract-activations endpoint, poll until done."""
    base = f"https://api.runpod.ai/v2/{endpoint_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"input": {"payload": {"samples": samples_chunk, "layers": layers}}}
    with httpx.Client(timeout=60.0) as c:
        r = c.post(f"{base}/run", headers=headers, json=body)
        r.raise_for_status()
        job = r.json()
    job_id = job["id"]
    print(f"  job {job_id} queued; polling...", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(15)
        with httpx.Client(timeout=30.0) as c:
            r = c.get(f"{base}/status/{job_id}", headers=headers)
            r.raise_for_status()
            st = r.json()
        status = st.get("status")
        print(f"  status={status}", flush=True)
        if status == "COMPLETED":
            if "output" not in st:
                # Some completions don't include output in /status — try /stream.
                with httpx.Client(timeout=60.0) as c:
                    sr = c.get(f"{base}/stream/{job_id}", headers=headers)
                    sr.raise_for_status()
                    sd = sr.json()
                if sd.get("stream"):
                    return sd["stream"][-1]
                raise RuntimeError(
                    f"Flash job {job_id} COMPLETED but no output (likely too large): {st}"
                )
            return st["output"]
        if status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Flash job {job_id} {status}: {st}")
    raise TimeoutError(f"Flash job {job_id} timed out after {timeout_s}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--labels",
        default="results/per_sentence/main_run_prefix.json",
        help="JSON output of run_per_sentence_probe.py with sentence verdicts",
    )
    p.add_argument(
        "--eval-log",
        default="logs/2026-05-03T21-37-34-00-00_interrogation-single-turn-detector_avAxSJGz85ooGeday4Tuzy.eval",
    )
    p.add_argument("--endpoint-id", required=True, help="Runpod endpoint id")
    p.add_argument("--layers", nargs="+", type=int, default=[22, 50, 76])
    p.add_argument("--chunk-size", type=int, default=10, help="Samples per Flash call")
    p.add_argument("--out", default="probes/per_sentence_behavioral.pt")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cache-acts",
        default="results/per_sentence/activations_cache.npz",
        help="Where to save extracted activations (so reruns skip Flash)",
    )
    p.add_argument(
        "--reuse-acts",
        action="store_true",
        help="If set, skip Flash and load cached activations from --cache-acts",
    )
    args = p.parse_args()

    runpod_key = os.environ.get("RUNPOD_API_KEY")
    if not runpod_key and not args.reuse_acts:
        envfile = REPO / ".env"
        if envfile.exists():
            for line in envfile.read_text().splitlines():
                if line.startswith("RUNPOD_API_KEY="):
                    runpod_key = line.split("=", 1)[1].strip()
                    os.environ["RUNPOD_API_KEY"] = runpod_key
                    break
    if not runpod_key and not args.reuse_acts:
        raise SystemExit("RUNPOD_API_KEY not set (env or .env)")

    # ---------- 1. Load labels + reconstruct argument text ----------
    probe_json = json.loads(Path(args.labels).read_text())
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        blob_path = f.name
    load_eval_argument_blob(args.eval_log, blob_path)
    eval_blob = json.loads(Path(blob_path).read_text())
    Path(blob_path).unlink(missing_ok=True)

    payload_samples = _build_assistant_payload(probe_json, eval_blob)
    print(f"Built payload: {len(payload_samples)} samples", flush=True)

    # ---------- 2. Get activations (Flash or cached) ----------
    cache_path = Path(args.cache_acts)
    if args.reuse_acts and cache_path.exists():
        print(f"Loading cached activations from {cache_path}", flush=True)
        cache = np.load(cache_path, allow_pickle=True)
        per_sentence: list[dict] = list(cache["per_sentence"])
        layers = list(cache["layers"])
    else:
        per_sentence = []
        layers = args.layers
        # Chunk to avoid 30MB+ HTTP responses.
        for i in range(0, len(payload_samples), args.chunk_size):
            chunk = payload_samples[i : i + args.chunk_size]
            print(
                f"Flash call {i//args.chunk_size + 1}/"
                f"{(len(payload_samples) + args.chunk_size - 1)//args.chunk_size} "
                f"({len(chunk)} samples)",
                flush=True,
            )
            out = _post_flash(
                endpoint_id=args.endpoint_id,
                api_key=runpod_key or "",
                samples_chunk=chunk,
                layers=layers,
            )
            per_sentence.extend(out["per_sentence"])
            print(f"  got {len(out['per_sentence'])} sentence activations", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            per_sentence=np.array(per_sentence, dtype=object),
            layers=np.array(layers),
        )
        print(f"Cached activations to {cache_path}", flush=True)

    # ---------- 3. Build feature matrix per layer ----------
    # Drop INSUFFICIENT_INFO (label==0.5) — train as binary TRUE/FALSE.
    used = [r for r in per_sentence if r["label"] in (0.0, 1.0)]
    print(
        f"Sentences: total {len(per_sentence)}; used {len(used)} after dropping "
        f"label==0.5; pos (FALSE) = {sum(1 for r in used if r['label']==1.0)}",
        flush=True,
    )
    if len(used) < 10:
        raise SystemExit("Too few binary-labeled sentences to train a probe")

    # Group sentences by sample_id so train/test split is at the sample level
    # (otherwise we leak per-argument context across train/test).
    sids = sorted({r["sample_id"] for r in used})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sids)
    n_train = int(len(sids) * args.train_frac)
    train_sids = set(sids[:n_train])
    test_sids = set(sids[n_train:])
    print(
        f"Sample-level split: {len(train_sids)} train / {len(test_sids)} test samples",
        flush=True,
    )

    results_per_layer: dict[int, dict] = {}
    best_layer: int | None = None
    best_test_auroc = -1.0

    import base64

    def _decode_acts(r: dict, layer: int) -> np.ndarray:
        # Worker returns either {"activations": {"<layer>": [floats]}} (legacy)
        # or {"activations_b64_f16": {"<layer>": "<b64>"}}.
        if "activations_b64_f16" in r:
            buf = base64.b64decode(r["activations_b64_f16"][str(layer)])
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        return np.array(r["activations"][str(layer)], dtype=np.float32)

    for layer in layers:
        Xtr, ytr, Xte, yte = [], [], [], []
        for r in used:
            v = _decode_acts(r, layer)
            if r["sample_id"] in train_sids:
                Xtr.append(v)
                ytr.append(r["label"])
            else:
                Xte.append(v)
                yte.append(r["label"])
        if not Xtr or not Xte:
            print(f"Layer {layer}: empty split, skipping")
            continue
        Xtr_a = np.stack(Xtr)
        Xte_a = np.stack(Xte)
        ytr_a = np.array(ytr, dtype=np.float32)
        yte_a = np.array(yte, dtype=np.float32)

        scaler = StandardScaler().fit(Xtr_a)
        Xtr_s = scaler.transform(Xtr_a)
        Xte_s = scaler.transform(Xte_a)

        clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
        clf.fit(Xtr_s, ytr_a)

        train_auroc = float(roc_auc_score(ytr_a, clf.decision_function(Xtr_s)))
        if len(set(yte_a.tolist())) >= 2:
            test_auroc = float(roc_auc_score(yte_a, clf.decision_function(Xte_s)))
        else:
            test_auroc = float("nan")
        print(
            f"Layer {layer}: train_AUROC={train_auroc:.4f} "
            f"test_AUROC={test_auroc:.4f} (n_train={len(ytr_a)}, n_test={len(yte_a)})",
            flush=True,
        )
        results_per_layer[layer] = {
            "train_auroc": train_auroc,
            "test_auroc": test_auroc,
            "scaler_mean": np.asarray(scaler.mean_).tolist(),  # type: ignore[union-attr]
            "scaler_scale": np.asarray(scaler.scale_).tolist(),  # type: ignore[union-attr]
            "direction": np.asarray(clf.coef_)[0].tolist(),
            "intercept": float(np.asarray(clf.intercept_).flatten()[0]),
            "n_train": int(len(ytr_a)),
            "n_test": int(len(yte_a)),
        }
        if not np.isnan(test_auroc) and test_auroc > best_test_auroc:
            best_test_auroc = test_auroc
            best_layer = layer

    if best_layer is None:
        raise SystemExit("No layer produced a valid test AUROC")

    print(f"\nBest layer: {best_layer} (test_AUROC={best_test_auroc:.4f})", flush=True)

    # ---------- 4. Save in liars-bench LR-detector format ----------
    import torch as _torch

    sys.path.insert(0, str(REPO / "liars-bench" / "src" / "probes"))
    from deception_detection.detectors import LogisticRegressionDetector

    direction = _torch.tensor(
        [results_per_layer[best_layer]["direction"]], dtype=_torch.float32
    )
    scaler_mean = _torch.tensor(
        [results_per_layer[best_layer]["scaler_mean"]], dtype=_torch.float32
    )
    scaler_scale = _torch.tensor(
        [results_per_layer[best_layer]["scaler_scale"]], dtype=_torch.float32
    )

    det = LogisticRegressionDetector(layers=[best_layer])
    det.directions = direction
    det.scaler_mean = scaler_mean
    det.scaler_scale = scaler_scale
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    det.save(out_path)
    print(f"Saved probe to {out_path}", flush=True)

    # Also save full results JSON for debugging.
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "best_layer": best_layer,
                "best_test_auroc": best_test_auroc,
                "per_layer": {
                    k: {kk: v for kk, v in vv.items() if kk not in ("direction", "scaler_mean", "scaler_scale")}
                    for k, vv in results_per_layer.items()
                },
                "n_train_samples": len(train_sids),
                "n_test_samples": len(test_sids),
                "labels_source": args.labels,
                "eval_log": args.eval_log,
            },
            indent=2,
        )
    )
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
