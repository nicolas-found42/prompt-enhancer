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
uv run --env-file .env python scripts/measure_criterion_reading.py report \
  --recording .local/criterion-reading/jev-dev.json
```

For the chat reader, set its completion-token cap with `--max-tokens` (default
`300`):

```sh
uv run --env-file .env python scripts/measure_criterion_reading.py run \
  --reader cheap --split heldout --max-tokens 300 \
  --output .local/criterion-reading/cheap-held.json
uv run --env-file .env python scripts/measure_criterion_reading.py report \
  --recording .local/criterion-reading/cheap-held.json \
  --cutoff partial=0.5,conditional=0.5,approximate=0.5
```

The report also prints the observed judgment distributions and marks rows whose
completion-token count reaches the recording's cap as truncation-suspect. The
measurement results and interpretation are recorded in [issue #116](https://github.com/nicolas-found42/prompt-enhancer/issues/116).
Recordings stay in the git-ignored `.local/criterion-reading/` directory; do not
commit them.
