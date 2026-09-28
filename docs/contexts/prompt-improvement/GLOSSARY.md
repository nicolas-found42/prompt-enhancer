# Prompt Improvement

This context tests whether a prompt can be improved and decides which version to return.

## Language

**Round**:
One attempt to beat the prompt: candidates are written with rewrite strategies, tested against the success tests on the weak panel and the strong model, and a winner is picked or the prompt is kept.
_Avoid_: pass, iteration, optimization

**Deep pass**:
Further rounds at the Deep tier for a run whose lower-tier rounds kept the original prompt, continuing the same run.
_Avoid_: deep run, re-run, escalation run
