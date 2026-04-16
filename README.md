# Deception Bench

A benchmark environment for measuring how much a lie detector helps a judge model identify correct answers when an untrusted suspect model is arguing for potentially wrong answers.

## Core idea

A **suspect** model is assigned an answer to argue for (which may or may not be the correct one). A **judge** model sees the argument and optionally a **lie detector report**, then picks what it thinks is the correct answer. We measure how much the detector improves the judge's accuracy compared to seeing the argument alone.

The dataset is WMDP-bio (bioweapons/biosecurity MCQ). Questions are filtered so that the suspect model knows the correct answer and the judge model doesn't — creating genuine information asymmetry.

---

## Repository layout

```
tasks/          Inspect AI eval tasks (one per file)
lib/            Shared code (data loading, prompts, detectors)
data/           Per-model belief CSVs from liars-bench
notebooks/      Analysis scripts and ad-hoc exploration
scripts/        Utility scripts
core/           Config loaders and prompt templates (filtering pipeline)
closed_models/  Filtering pipeline (API-based, no GPU needed)
debate/         Debate environment (multi-turn cross-examination)
open_models/    (Placeholder) White-box probe code, requires GPU
results/        Output data (filter results, debate traces)
pyproject.toml  Python project / dependency spec
```

---

## Setup

**Requirements:** Python 3.12+, `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/skartik04/deception.git
cd deception
uv sync
cp .env.example .env  # add your OPENROUTER_API_KEY
```

---

## Tasks (deception-bench)

### `tasks/interrogation_single_turn.py`

Single-turn interrogation without a lie detector. The suspect argues for its assigned answer; the judge reads the argument and picks an answer.

```bash
uv run python -m inspect_ai eval tasks/interrogation_single_turn.py \
  --limit 100 --log-level warning \
  -T suspect_model=openrouter/meta-llama/llama-3.3-70b-instruct:nitro \
  -T judge_prompt=informed \
  --model openrouter/google/gemma-3-27b-it:nitro
```

### `tasks/interrogation_single_turn_detector.py`

Same as above but a lie detector report is appended to the judge's prompt.

Supported detectors (`-T detector_name=...`): `oracle`, `coin_flip`, `behavioral`, `behavioral_hardened`

```bash
uv run python -m inspect_ai eval tasks/interrogation_single_turn_detector.py \
  --limit 100 --log-level warning \
  -T detector_name=oracle \
  -T suspect_model=openrouter/meta-llama/llama-3.3-70b-instruct:nitro \
  --model openrouter/google/gemma-3-27b-it:nitro
```

---

## Filtering Pipeline (debate/closed_models)

Find questions where the suspect model knows the answer and the judge doesn't.

```bash
for role in suspect judge; do uv run python -m closed_models.run_filter --role $role; done

uv run python -m closed_models.intersect \
  --suspect results/suspect_*.json \
  --judge results/judge_*.json
```

---

## Models

| Role | Model |
|---|---|
| Suspect | `openrouter/meta-llama/llama-3.3-70b-instruct:nitro` |
| Primary judge | `openrouter/google/gemma-3-27b-it:nitro` |

---

## Viewing results

```bash
uv run python -m inspect_ai view
```
