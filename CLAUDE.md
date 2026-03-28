# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project under UCLA Algoverse studying whether deception detection probes help a judge LLM identify lying in adversarial debates. The core question: when a knowledgeable (large) model argues for a position it knows is false, can probes given to a less-knowledgeable (small) judge model help it catch the lie?

This bridges two research areas that haven't been combined before:
- **Deception probes** (Apollo Research, Liars' Bench): classification of deceptive vs honest LLM outputs
- **Social deduction / debate games** (WOLF, Werewolf Arena): LLMs in adversarial multi-turn settings

The contribution is testing whether probes are *useful to a player* in an interactive setting, not just accurate as classifiers.

## Research Setup: WMDP Expert Knowledge Debate

### Models
- **Suspect model**: Llama-3.3-70B-Instruct (large, knows the answers). Apollo's pre-trained probes are built for this exact model (layer 22 residual stream activations).
- **Judge model**: Smaller model (e.g. Llama-8B). Must be weak enough on WMDP that it genuinely doesn't know the answers.

### Dataset: WMDP (Weapons of Mass Destruction Proxy)
Hard multiple-choice questions about biosecurity, chemical weapons, cybersecurity. Used to create information asymmetry between suspect and judge.

### Question Filtering Pipeline
1. Ask the suspect model each WMDP question N times (varied framings: MCQ, true/false, open-ended, negated, knowledge-report). Keep questions where suspect answers correctly 9+/N times.
2. Ask the judge model the same filtered questions. Keep only questions where the judge fails (gets it right <=2/N times).
3. Surviving questions form the filtered dataset with guaranteed information asymmetry.

### Experiment Structure (per question)
For each question in the filtered set, run 4 conditions:
1. Suspect told: "argue {fact} is TRUE" — judge debates WITHOUT probe
2. Suspect told: "argue {fact} is FALSE" — judge debates WITHOUT probe
3. Suspect told: "argue {fact} is TRUE" — judge debates WITH probe
4. Suspect told: "argue {fact} is FALSE" — judge debates WITH probe

Only one of true/false is correct. The suspect knows which, the judge doesn't. Conditions 1-2 are baselines; 3-4 test whether the probe helps the judge catch the lie. Multiple iterations per condition to reduce variance.

### Probe Types to Evaluate
- **White-box** (open models only): Linear probes on residual stream activations (Apollo-style). Apollo's pre-trained probes for Llama-3.3-70B are available at https://github.com/ApolloResearch/deception-detection
- **Black-box** (works on closed models too): LLM-based oversight, multi-agent inspection, behavioral analysis
- Goal is to compare probe types and find which helps the judge most

## Key References
- Apollo deception detection: https://github.com/ApolloResearch/deception-detection (paper: arxiv 2502.03407)
- Liars' Bench (mentor Walter Laurito's group, Cadenza Labs): arxiv 2511.16035
- WMDP dataset for question sourcing

## Mentor Guidance (Walter Laurito)
- Focus on environment setup first, detectors second
- Use Llama-3.3-70B so Apollo probes work out of the box
- Start with Apollo probes as baseline, train custom probes later
- The point is NOT classification — it's ranking detectors by usefulness to a player in an environment
- Cloud compute via Lambda (Walter setting up access)
