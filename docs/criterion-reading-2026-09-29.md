# Criterion-reading measurement (2026-09-29)

The script measures whether Jev or a chat model can read a success criterion's
countable property, comparison, number or range, and whether the requirement is
partial, conditional, negated or approximate. It scores raw recorded answers
offline against `tests/fixtures/evaluation/criterion_reading_cases.json`, which
contains 50 development and 100 held-out cases. The held-out labels are not yet
reviewed by a maintainer.

## Label provenance and review status

Every case carries `label_origin` (`issue-116` for the development split,
`claude-2026-09-29` for the held-out split) and `review_status`. All 150 cases are
`unreviewed`; the maintainer review of the held-out labels is tracked in
[issue #116](https://github.com/nicolas-found42/prompt-enhancer/issues/116) and no
label is marked reviewed here.
`label_provenance_errors` in the script validates each case, and a test runs it over
the whole fixture:

- `unreviewed` may not carry `reviewed_by`, `reviewed_on`, `audit_ref` or
  `previous_expected`, so a generated label cannot look approved.
- `reviewed` needs `reviewed_by` and an ISO `reviewed_on` date.
- `corrected` also needs `audit_ref`, the link to the review record, and
  `previous_expected`, the label it replaced.

`report` prints how many scored labels are unreviewed per split and says that its
scores measure agreement with labels no maintainer has reviewed until none remain.

## Offline replay data

Real recordings stay local and git-ignored. Each is a single historical observation
of one run and says nothing about run-to-run stability; a stability measurement needs
repeated fresh runs. A report on a recording without provenance says so.

Two small synthetic recordings are committed so a fresh clone can replay `report`
without an API key or network:

```sh
uv run python scripts/measure_criterion_reading.py report \
  --recording tests/fixtures/evaluation/criterion_reading_synthetic_jev.json
uv run python scripts/measure_criterion_reading.py report \
  --recording tests/fixtures/evaluation/criterion_reading_synthetic_cheap.json
```

They are hand-authored (`model: synthetic-fixture`), not model output, and their
`provenance` block says so; the report prints a `SYNTHETIC RECORDING` banner.
They do not measure how well Jev or a chat model reads criteria, so cite the
real recordings and issue #116 for that. Their rows are chosen to land in every
outcome, an unresolved band edge, JSON negation bands, an unusable answer and a
truncated reply, and `tests/test_issue116_criterion_reading.py` pins the resulting
report tables. They reference cases by id from the public fixture and include no
prompts, keys, costs, token counts or provider identifiers. If the question
wording changes, `questions_digest` no longer matches and those tests fail until
the recordings are regenerated.

Run a live recording and score it offline:

```sh
uv run --env-file .env python scripts/measure_criterion_reading.py run \
  --reader jev --split development \
  --output .local/criterion-reading/jev-dev.json
uv run python scripts/measure_criterion_reading.py report \
  --recording .local/criterion-reading/jev-dev.json
```

Loading `.env` is optional; it is only needed for live `run` calls that need gateway credentials.

For the chat reader, set its completion-token cap with `--max-tokens` (default
`300`, maximum `4096`):

Use a new recording path when changing the cap; resuming a cheap recording with
a different cap is rejected.

Per-judgment cutoffs are selectable offline with `report --cutoff`; values must be
between 0 and 1. Reports print the chosen cutoff values, while recordings preserve
the raw judgments for rescoring.

`run --model MODEL` overrides the reader's default model (cheap: `mistralai/mistral-nemo`;
Jev: `PROMPT_ENHANCER_JEV_MODEL` or the Gateway default). The value must be non-empty
and no more than 200 characters. The selected model is stored in each recording;
resuming with a different model is rejected.

```sh
uv run --env-file .env python scripts/measure_criterion_reading.py run \
  --reader cheap --split heldout --max-tokens 300 \
  --output .local/criterion-reading/cheap-held.json
uv run python scripts/measure_criterion_reading.py report \
  --recording .local/criterion-reading/cheap-held.json \
  --cutoff partial=0.5,conditional=0.5,approximate=0.5
```

The report prints observed judgment distributions. For chat-reader (`--reader cheap`)
recordings it also flags truncation when the provider's finish reason is `length`
(`finish_reason`, `stop_reason == "max_tokens"`, or
`incomplete_details.reason == "max_output_tokens"`). Older rows without a finish
reason use the completion-token count reaching the recorded cap as a
truncation-suspect heuristic. Jev recordings carry no finish-reason evidence and are
never flagged.
The measurement results and interpretation are recorded in [issue #116](https://github.com/nicolas-found42/prompt-enhancer/issues/116).
Real recordings stay in the git-ignored `.local/criterion-reading/` directory; do not
commit them or the whole directory. Only the synthetic recordings above are committed.

## Label audit and recommended cutoffs (2026-09-30)

All 150 labels are `audited`, not human-reviewed: the maintainer asked for Jev to do the
review, so each label was compared with five blind Jev readings (`typesafe/jev-1.13-20260917`,
`questions_digest` in the record) and re-read once by the model that wrote the held-out
labels. **No label was changed.** 141 agree in every run; 15 were contested and each was
adjudicated with a stated reason in
[`criterion_label_audit_2026-09-30.json`](../tests/fixtures/evaluation/criterion_label_audit_2026-09-30.json).
The record also holds per-case probability ranges, the stability figures, and the negative
result of using Jev as a label verifier (it cannot tell whether "the slogan" is the whole
output, which is the scope question under audit).

One convention inconsistency was found: short-form named artifacts count as the whole
output for some nouns (tagline, slogan, blurb, bio, welcome message, email body) and as a
part for others (headline, title, subject line, greeting, code comment). Both are context
dependent, so consumers should abstain when scope is ambiguous.

**Recommended cutoffs:** `partial=0.40`, `conditional=0.50`, `approximate=0.50`, JSON
negation band `(0.35, 0.65)`. Per run over the 150 criteria this gave 86.6 correct checks, 55.0
correct abstentions, 8.4 missed and 0 confidently wrong, against the regex's 29, 33, 62 and 26.
The default `partial=0.5` sits inside Jev's own uncertainty: one null-labelled criterion
(`held-076`) became a confident wrong check in two of five runs from a ±0.02 wobble.
`conditional` and `approximate` separate cleanly (null-labelled >= 0.89, checkable <= 0.26
and <= 0.08); only `partial` overlaps. The cutoff was chosen after seeing the held-out split,
so those numbers are optimistic, and zero wrong in 150 bounds the wrong rate at about 2.0%
(one-sided 95%). Tightening that needs more labels.
