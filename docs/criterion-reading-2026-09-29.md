# Criterion-reading measurement (2026-09-29)

The script measures whether Jev or a chat model can read a success criterion's
countable property, comparison, number or range, and whether the requirement is
partial, conditional, negated or approximate. It scores raw recorded answers
offline against `tests/fixtures/evaluation/criterion_reading_cases.json`, which
contains 50 development and 100 held-out cases. The held-out labels are not yet
reviewed by a maintainer.

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

The report prints observed judgment distributions and flags truncation when the
provider's `finish_reason` is `length`. Older rows without a finish reason use the
completion-token count reaching the recorded cap as a truncation-suspect heuristic.
The measurement results and interpretation are recorded in [issue #116](https://github.com/nicolas-found42/prompt-enhancer/issues/116).
Recordings stay in the git-ignored `.local/criterion-reading/` directory; do not
commit them.
