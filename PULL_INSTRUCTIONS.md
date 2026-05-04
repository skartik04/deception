# Pull this RunPod project to local machine

Hi Claude (on the user's local machine). The user wants to rsync the current
state of the `gergely_deception` project + their Claude config back from a
remote RunPod pod to local. **Do not pull `/workspace/models/`** — that's
~140 GB of Llama weights, not needed locally.

## Pod details (active right now)

- Public IP: `205.196.144.42`
- SSH port: `11677`
- User: `root`
- SSH key (already on local): `~/.ssh/mac_key`
- Pod ID: `4rbufwyl8tvcm9` (RunPod, datacenter US-MO-2)

Connection sanity check:

```bash
ssh -i ~/.ssh/mac_key -p 11677 root@205.196.144.42 'echo ok && uptime'
```

## Layout on the pod

The pod uses `/workspace` as the persistent root (RunPod network volume).
- `/workspace/gergely_deception/` — the project (this is what to pull)
- `/workspace/.claude-kris/` — Claude config (commands, skills, projects DB)
- `/workspace/.claude/` — symlink/bind to `.claude-kris`; treat both names as the same dir
- `/workspace/.bashrc` — shell setup
- `/workspace/models/` — **DO NOT PULL** (140 GB Llama weights, not needed locally)

## Pull commands

Run these from the local machine. The corresponding local destinations match
what the user's existing setup expects (`/root/.claude*` and `/root/gergely_deception/`).

```bash
POD_SSH="ssh -i ~/.ssh/mac_key -p 11677"
POD="root@205.196.144.42"

# Project directory (skip the venv — local will recreate it via `uv sync`)
rsync -av --exclude='.venv' --exclude='__pycache__' \
  -e "$POD_SSH" \
  $POD:/workspace/gergely_deception/ \
  /root/gergely_deception/

# Claude config (skills, commands, projects DB)
rsync -av -e "$POD_SSH" \
  $POD:/workspace/.claude-kris/ \
  /root/.claude/
```

Adjust `/root/...` to wherever the user keeps these on local (likely `/root/`
on a Linux dev box, `~/` on macOS). Ask the user if you're unsure.

## After pulling

1. `cd /root/gergely_deception` (or local path).
2. Run `uv sync` to recreate the venv from `uv.lock`.
3. Note that scripts referencing `/workspace/models/llama-3.3-70b-instruct/`
   will not work locally — model paths are pod-specific. The probes folder
   `probes/` (small `.pt` files) is preserved.

## Context: what was done in this session

- Reproduced Apollo's roleplaying probe AUC = **0.983** (their reported 0.9825) by running their `experiment.py` directly with `CUDA_VISIBLE_DEVICES=0` (single-GPU + CPU offload). Multi-GPU + bf16 + default attention on Blackwell produces NaN; single-GPU avoids it.
- Pinned `transformers==4.52.4` and `torch==2.7.1` to match Apollo's lockfile (lock at `liars-bench/src/probes/uv.lock`). Local `uv sync` will use whatever is in the project's `pyproject.toml` — may need to pin manually if reproducing.
- Confirmed pre-trained probes (roleplaying, followup, instructed_pairs) collapse to ~50% on geometry_of_truth — OOD failure, not a setup bug.
- Started a final stress test (`recompute` on `got_mixed__plain` through their pipeline) before the user paused. State at pause:
  - Scores cached: `liars-bench/src/probes/results/roleplaying_minimal_lr__roleplaying__plain__llama-70b-3.3__20260430_180729/scores.json`
  - That folder has the trained roleplaying probe (`detector.pt`) + roleplaying val scores (matching Apollo's 0.98)
  - The recompute job for `got_mixed__plain` was kicked off but not yet verified — log at `/tmp/their_got_eval.log` on the pod.

## Gotchas / things to know

- The pod's `liars-bench/src/probes/deception_detection/models.py:142` was patched to point Llama to `/workspace/models/llama-3.3-70b-instruct/` (was `/root/models/...`). The patched path won't exist locally; that's fine, just a reminder.
- `liars-bench/src/probes/deception_detection/scripts/experiment.py results()` indexes `cfg.control_data[0]` — crashes if `control_data: []`. Workaround: provide a dummy or call `recompute` instead of `run`.
