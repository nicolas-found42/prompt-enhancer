# Live prompt health

The draft editor checks a nonempty prompt 600 ms after typing stops when the
pinned Jev model and a server-side OpenRouter credential are available. The
checkbox beside the editor persists its on/off preference in browser storage.
The page cancels superseded fetches and accepts only a response for the current
draft revision. Hidden pages and empty drafts do not start checks. Assessment
errors leave the editor usable and hide old findings.

`POST /api/prompt-health` takes `prompt`, a nonnegative `revision`, and a local
`session_id`. It returns a draft hash and revision, status, four dimension
judgments, a provisional Prompt clarity score when complete, coverage, source
sentence flags with exact offsets, cache counts, and separate live-health usage.
`GET /api/prompt-health/settings` reports availability and the current rolling
allowance. `/api/settings` also includes that read-only summary. Health requests
use a separate Gateway instance, so an optimization run's usage ledger and
decision log cannot be reset by typing.

Each dimension asks a Noul applicability question and a four-level Score question.
Code averages normalized Scores for applicable dimensions with equal weights.
Uncertain applicability is reported as unknown. An incomplete assessment has no
overall score. Sentence checks cover vagueness, unresolved references,
contradictions, and evaluator-steering language. The latter three see the full
prompt and target sentence ID; vagueness sees only the sentence text. Findings
are advisory and do not block Optimize. Their labels and UI text are fixed in
code; Jev does not generate explanatory prose.
Matching #50 calibration gates take precedence over provisional 0.8 cutoffs;
stale or rank-only artifacts produce uncertain applicability or no visible flag.

SQLite stores raw answers under a canonical hash of supplied state, question,
criteria, question version, and answering snapshot. Full-prompt and relational
checks invalidate when context changes. Identical context-free sentence checks
reuse one answer while the server maps it to each current occurrence ID. Only
answers from the verified pinned Jev snapshot are cached. The display policy
version is separate from the semantic cache key.

The defaults are a 20,000-character draft cap, 100 question cap, 20 inference
refreshes per minute, at most three provider requests per refresh, and a $0.05
rolling-hour allowance shared through SQLite across tabs and reloads. A live
refresh reserves a conservative provider-priced cost before dispatch. The
isolated health Gateway disables internal retries, so its two batches cannot
exceed the physical request cap. Failed attempts keep their reservation when
their billable usage is unknown; measured one-attempt completions reconcile to
the provider's reported cost. Cache-only reads do not consume the allowance.
Missing pricing, credentials, or provider access results in a quiet unavailable
or paused status. These caps are provisional product policy, not a Jev price or
latency claim.

To exercise the strict replay benchmark with synthetic known-answer cases:

```sh
uv run --locked python scripts/benchmark_prompt_health.py
```

Pass `--recording path/to/recording.json` and `--cases path/to/labeled-cases.json`
to replay an external recording. The report names false and missed flags,
cache-hit rate, local replay p50/p95 execution time, and cost provenance.
Synthetic replay verifies the reporting path; it does not measure live Jev
accuracy or provider latency.
