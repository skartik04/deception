# Debate Design: Open Questions

Before building more, we need to align on five decisions. Everything else follows from these.

---

## 1. Debate Format

**Option A — One suspect, judge interrogates (current)**
Suspect argues a claim. Judge cross-examines. Judge gives verdict.
Problem: judge has no domain knowledge, so "is this true?" is unanswerable. The judge can only assess "did this hold up under pressure?"

**Option B — Two suspects, judge observes**
One argues true, one argues false. Judge picks the winner.
Removes domain knowledge problem entirely — judge just does comparative evaluation.

Which one? Or both?

---

## 2. Judge's Goal

| Framing | What judge does | Domain knowledge needed? |
|---|---|---|
| "Find the lie" | Binary: lying or not | Yes (can't do it) |
| "Does this hold up?" | Evaluate argument quality | No (current direction) |
| "Pick the honest one" | Compare two suspects | No (requires Option B) |

We've moved toward "does this hold up?" — is that agreed?

---

## 3. Where Does the Probe Get Information the Judge Can't See?

**This is the central constraint:** a probe reading the same text as the judge adds nothing. The judge could do everything the probe does. A probe is only useful if it has access to something the judge structurally cannot have.

| Probe type | Information source | Actually new info? | Notes |
|---|---|---|---|
| Oracle | Ground truth injected | Yes (trivially) | Ceiling test only, not a real detector |
| CoT / reasoning traces | Suspect's private chain-of-thought | **Yes** | Requires thinking model. Auditor compares private reasoning vs public answer |
| Parallel interrogation | Second independent conversation with suspect | **Yes** | Compares answers across two conversations for consistency |
| Commitment forcing | Pre-debate factual commitments | Somewhat | Suspect can hedge strategically |
| Text-only behavioral analysis | Same public transcript as judge | **No** | Division of labor, not new signal. Not a valid probe |

The only two real options right now: **CoT traces** and **parallel interrogation**.

Open question on CoT: does the suspect's reasoning actually show internal conflict when lying, or does it just plan how to argue? Needs empirical checking.

---

## 4. Probe Timing

- **Every turn** (current) — judge can course-correct mid-debate
- **Once at end** — simpler, one consolidated report
- **On demand** — judge invokes auditor when stuck (harder to analyze)

---

## 5. Calibration — Run These Before Anything Else

| Condition | Expected | What it proves |
|---|---|---|
| No probe | ~50% accuracy | Judge is unbiased, no default-true/false |
| Oracle (perfect signal) | ~95-100% | Environment responds to information |
| Coinflip (random signal) | ~50% | Random signal doesn't inflate accuracy |
| Real probe (CoT / parallel) | Between 50-100% | Probe actually helps |

If no-probe isn't ~50%, the judge prompt is biased — fix that first.
If oracle isn't ~100%, the judge can't use signals — fix that first.
**We haven't run any of these yet.**

---

## Align on These

1. Format: one suspect or two?
2. Judge goal: "hold up?" or "pick the honest one?"
3. Probe source: CoT traces, parallel interrogation, or both?
4. Probe timing: every turn or once?
5. Calibration first: oracle + no-probe baseline before comparing detectors.

Once these five have answers, the experiment is fully specified.
