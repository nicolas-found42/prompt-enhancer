# Prompt Enhancer

A local engine that diagnoses a user's prompt with Jev, tests rewrites on weak models, and returns the original or a verified improvement.

## Language

**Gateway**:
The single way the engine reaches any model: the writer, the weak panel, the strong check, and Jev. Live, scripted, recorded, and replayed runs differ only in which gateway they use.
_Avoid_: client, provider, model API

**Round**:
One attempt to beat the prompt: candidates are written with rewrite strategies, tested against the success tests on the weak panel and the strong model, and a winner is picked or the prompt is kept.
_Avoid_: pass, iteration, optimization

**Deep pass**:
Further rounds at the Deep tier for a run whose lower-tier rounds kept the original prompt, continuing the same run.
_Avoid_: deep run, re-run, escalation run
