# Gap cutoff recalibration and writer check (2026-09-24)

## Why

In local use the optimizer returned almost every prompt unchanged. Of 10
completed local runs, 8 stopped at diagnosis with no confirmed gap, and none
improved. The earlier reports show the same across every cohort: 0 improved.
Jev separated clear from vague prompts well (a clear control prompt never went
above 0.41; vague prompts scored 0.55–0.90 for missing context), but the gap
cutoffs sat at the top of that range (0.87–0.90). The user chose to act more
often, accepting that the tool will sometimes ask a question it did not need.

## Step 1: the version 2 writer, live

The diagnosis-input writer fix (instruction version 2) had never been measured
because OpenCode Go refused requests. With Go active, the three affected ROPE
cases were rerun live for both writers and the six-case planted pilot for Space
Bunny, at Fast tier on `main` (62d95c7), which still used the old cutoffs.

| Run | Cases | Reached candidates | Passed fidelity | Improved | Unavailable |
| --- | ---: | ---: | ---: | ---: | --- |
| ROPE affected, Space Bunny | 3 | 3 | 0 | 0 | none |
| ROPE affected, DeepSeek V4.1 Flash | 3 | 2 | 0 | 0 | 1 (Jev HTTP 400) |
| Planted pilot, Space Bunny | 6 | 1 | 0 | 0 | 1 (writer returned invalid success tests) |

The six candidates scored 0.59–0.86 on `meaning_preserved`, `no_invention` and
`edits_confined`, against 0.80 cutoffs. Reading them shows the gate is right.
When the missing piece is context only the user has, `add_missing_context`
cannot supply it, so the writer turns the request into a request for the
details (for example "Create a travel itinerary…" became "Please provide the
number of days, the city… so I can create a travel itinerary"). That changes
the task. The fidelity cutoffs are therefore unchanged.

These recordings replay exactly under the commit that made them (62d95c7). They
do not replay under the code below, because it no longer sends the empty
`possible_gaps` field to models (see Replay compatibility).

## Steps 2 and 3: gap cutoffs

Confirmed gaps drive both clarifying questions and candidates, so one cutoff
per question covers both steps. Cutoffs were selected on the training split of
the 139-prompt user-delegated review (`public-delegated-gap-review.json`, same
participant-group split), using a rule fixed before looking at holdout: the
lowest cutoff on a 0.60–0.95 grid whose training precision is at least 0.5, with
at least one true positive.

| Question | Old | New | Training TP/FP/FN | Holdout TP/FP/FN, old → new |
| --- | ---: | ---: | --- | --- |
| `context` | 0.87 | **0.83** | 7/7/5 | 0/0/4 → 3/2/1 |
| `language` (coding) | 0.90 | **0.77** | 11/11/3 | 0/0/3 → 2/1/1 |
| `goal` | 0.90 | 0.90 | no cutoff reaches precision 0.5 | unchanged |
| `constraints`, `output_format`, `done_criteria`, `sources`, `tests`, `time_horizon` | 0.90 | 0.90 | no training positives | unchanged |

The keys without positives were not lowered. At 0.80 they would flag 13–39% of
the prompts judged gap-free (`output_format` 28/119, `tests` 24/61,
`done_criteria` 16/119, `constraints` 15/118).

`outside_reference` ("details only you know") postdates the review and has no
labels, so its recall cannot be measured. Its exact question was sent for all
139 prompts: none of the 101 judged gap-free reached 0.75. The cutoff moves from
0.90 to **0.80** on that false-flag bound alone; the near-miss hint band is now
0.75–0.80.

`context` is now high impact. Impact only decides whether an unknown gap is
asked about. Rewrites cannot supply missing context and failed fidelity in
every recorded attempt (the 6 above, plus every candidate in the earlier planted
pilot, ROPE studies and local runs), so asking is the only route that can help.

## Live check on the local prompts

The eight distinct prompts from the local history were run live at Standard
tier with clarification allowed (as in the app), using the code below.

- Before: at most 1 of 8 got a question (the Hendersons letter, in one of its
  runs); 0 improved.
- After: 4 of 8 get one specific question, for example "What are the 'new rules
  for the vans' that need to be communicated?" and "What hours did you work
  today, including your start time, end time, and any unpaid break?". The
  clear control prompt is still not flagged.
- Answering two of them and resuming returned the prompt with the answer
  appended ("Details: …", "Context: …"), which the app shows as an optimized
  prompt.

This is eight prompts from one person, not a representative sample.

## Replay compatibility

Strict replay of the recorded evaluations had broken before this change, in
36fba0c and 32f54a0, because two edits reached provider requests. Both are fixed:

- The clarifier's `outside_reference` instruction is sent only when that gap is
  asked about.
- `possible_gaps` (user-facing hints) are removed from the diagnosis sent to
  models, so hints never steer strategy, candidate or fidelity decisions.

Bundles now record `checklist_impacts`. A bundle without the field replays with
`context` at medium impact, as it was recorded. The recorded thresholds
(`rubric_thresholds`) already restore the old cutoffs. Answered `context` lines
read "Context:"; inferred lines keep the key that recordings contain.

All nine documented strict replays match their stored reports case for case
(SOAR 150 for both writers, planted pilot for both, ROPE 35 for both at both
faithfulness gates, and ClariQ 40): 492 cases.

## Remaining limits

- No prompt has been improved by a rewrite yet. The new improvements come from
  asking and adding the answer.
- `strategy_recheck` (0.80) and the fidelity questions still need their own
  labels. One local run stopped at a recheck of 0.77.
- `outside_reference` recall and the new `context`/`language` cutoffs rest on
  model-delegated labels and small holdout counts (4 and 3 positives).

## Reproduction

```sh
uv run python scripts/calibrate_delegated_gaps.py \
  --dataset .local/evaluation/public-delegated-gap-review.json \
  --decisions .local/evaluation/public-delegated-jev-decisions.json \
  --output .local/evaluation/public-delegated-gap-calibration-precision-floor.json \
  --selection precision_floor --min-precision 0.5
uv run --env-file .env python scripts/measure_outside_reference.py \
  --dataset .local/evaluation/public-delegated-gap-review.json \
  --decisions .local/evaluation/public-delegated-outside-reference-decisions.json \
  --output .local/evaluation/public-delegated-outside-reference-summary.json --measure
uv run --env-file .env python -m prompt_enhancer.evaluation .local/evaluation/rope-affected-3.json \
  --live --record .local/evaluation/rope-bunny-writer-v2-affected-replay-0924.json \
  --tier fast --output .local/evaluation/rope-bunny-writer-v2-affected-report-0924.json
uv run --env-file .env python -m prompt_enhancer.evaluation .local/evaluation/rope-affected-3.json \
  --live --record .local/evaluation/rope-deepseek-writer-v2-affected-replay-0924.json \
  --tier fast --writer-model deepseek-v4.1-flash \
  --output .local/evaluation/rope-deepseek-writer-v2-affected-report-0924.json
uv run --env-file .env python -m prompt_enhancer.evaluation evaluation/planted-pilot.json \
  --live --record .local/evaluation/planted-pilot-writer-v2-replay-0924.json \
  --tier fast --output .local/evaluation/planted-pilot-writer-v2-report-0924.json
```

`--measure` resumes and skips prompts already answered. The three live
commands ran at 62d95c7; replay them with `--replay` at that commit.
