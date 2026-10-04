# Prompt Improvement

This context tests whether a prompt can be improved and decides which version to return.

## Language

**Round**:
One attempt to beat the prompt: candidates are written with rewrite strategies, tested against the success tests on the weak panel and the strong model, and a changed winner is picked or the run reports an unverified improvement.
_Avoid_: pass, iteration, optimization

**Deep pass**:
Further rounds at the Deep tier for a run whose lower-tier rounds kept the original prompt, continuing the same run.
_Avoid_: deep run, re-run, escalation run

**Rejection cause label**:
A Jev suggestion for why a user rejected a completed prompt result, inferred from the user's optional note and retained for evaluation.
_Avoid_: ground-truth label, Round signal

**Improvement not verified**:
The failure a run reports when its bounded rounds end without a verified changed prompt. The original is returned and the run's status is failed; it is not a successful no-change result.
_Avoid_: no change, kept original, unchanged success

## Policy

A successful enhancement returns a prompt that changed. Under the always-improve
policy every valid prompt is rewritten: the absence of confirmed gaps opens
whole-prompt latitude (all Jev support, meaning, grading, screen, and strong
checks still apply), ties are broken toward a changed candidate, and a run that
ends on the original input carries an `improvement_not_verified` failure with
status `failed` — never a silent success.
