# Jev quality-review pilot — 2026-09-24

This is a small integration pilot, not a code-review accuracy benchmark. Rules remain advisory.

## Candidate verification

The [seed corpus](../../tests/fixtures/quality_review/pilot.json) contains 12 hand-authored synthetic cases: five supported findings, five contradicted findings, and two cases missing essential context. Labels are hidden from the model state. This is not a sample of independently human-reviewed pull requests.

The [recording](../../tests/fixtures/quality_review/jev-pilot-recording.json) contains 12 live Gateway requests, each asking two independent Choice questions. Requested model: `typesafe/jev-1.13`; served model: `typesafe/jev-1.13-20260917`. All 24 answers were returned and parsed; source-evidence and replay tests run without inference.

The support judgment matched 11 of 12 expected labels. It identified all five seeded supported findings, contradicted four of five seeded false findings, and returned insufficient evidence for both missing-context cases. The remaining negative case (`c10`) also received insufficient evidence instead of contradicted. No seeded valid finding was classified contradicted. These counts do not establish real-world precision or recall.

Gateway-reported usage: 8,664 input tokens, 1,210 output tokens, **$0.000363888**; accumulated request time approximately 10.19 seconds. No threshold was selected from this sample.

## Repeat uncertain cases

Two lower-confidence supported findings (`c03`, `c09`) were repeated once with identical questions and evidence. Both kept their supported label; the confidence changed. The [subset](../../tests/fixtures/quality_review/repeat.json) and [second recording](../../tests/fixtures/quality_review/jev-repeat-recording.json) preserve that observation.

| Case | First supported probability / confidence | Repeat supported probability / confidence |
| --- | --- | --- |
| c03 | 0.77 / 0.66 | 0.81 / 0.71 |
| c09 | 0.66 / 0.50 | 0.60 / 0.40 |

The repeat used 1,455 input tokens and 196 output tokens; Gateway-reported cost was $0.00006111. One repeat of two cases does not establish stability. In particular, the failure-handling case remains ambiguous enough to require reviewer judgment.

## Existing linter smoke test

Pinned `jev-lint` 0.7.0 was exercised through OpenRouter on the changed lines in `src/prompt_enhancer/jev.py` from commit `0ea0a3b` versus its parent. Its dry-run estimate was 6,812 tokens; the live request used 5,630 input and 104 output tokens, took approximately 3.82 seconds, and returned verdicts for all five selected subjects with no findings or missing results. The native recording identified `typesafe/jev-1.13-20260917`. This verifies transport and report plumbing; zero findings is not proof of correctness.

The linter's fixed-price estimate was $0.00023646; unlike Gateway usage above, this is not a provider-returned bill. Real-source linter records remain in ignored `.local/quality-review/`.

A broader dry run exceeded the 50,000-token default and correctly produced `partial` without calling inference. Reducing to changed-line subjects lowered the planned work from 301 subjects to 50; selecting one file made the live smoke test fit the budget. This supports keeping explicit scope and budget controls.

## Reproduce without inference

```sh
uv run --locked python scripts/review_quality.py replay --recording tests/fixtures/quality_review/jev-pilot-recording.json --output .local/quality-review/replayed.json
uv run --locked python scripts/review_quality.py evaluate --recording tests/fixtures/quality_review/jev-pilot-recording.json --output .local/quality-review/metrics.json
uv run --locked python scripts/review_quality.py replay --corpus tests/fixtures/quality_review/repeat.json --recording tests/fixtures/quality_review/jev-repeat-recording.json --output .local/quality-review/repeated.json
```

See the [operating guide](../quality-review.md) for live runs, evidence selection, failure states, and the optional workflow. The next evidence to collect is human-labeled real review findings, including legitimate exception-handling policies and cross-function context; no automatic suppression or merge gate is justified by this pilot.
