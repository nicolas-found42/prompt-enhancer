# SOAR live evaluation, 2026-09-23

The later [delegated-review follow-up](delegated-evaluation-2026-09-23.md)
completes the two review materials, reruns the changed faithfulness paths,
and updates the acceptance matrix. Historical measurements below remain as
recorded; statements about pending review reflect the earlier stage.

## Method

The [SOAR Lab prompt knowledge gap corpus](https://github.com/SOAR-Lab/prompt-knowledge-gap)
supplied 150 real developer prompts with human source labels, selected from
distinct conversations by `scripts/prepare_soar_prompt_gaps.py`. The dataset
digest is `313d22aabb21bd7ab653961d2ce40036f408a54b37ba009b52260ecc43da9314`.
The source's **missing context** label maps directly to the optimizer's
`context` question. Its other label categories do not correspond one-to-one
with the checklist, so the accuracy figures below apply only to `context`.

The live run used Fast tier, seed 0, Jev 1.13 on OpenRouter, two OpenRouter
weak models (`meta-llama/llama-3.1-8b-instruct` and
`mistralai/mistral-nemo`), and OpenCode Go for the writer and
`glm-5.3-flash` strong check. The baseline writer was `space-bunny-free`.
Eighteen Space Bunny cases required a separate retry after writer network
failures; the successful attempts and per-case cost and latency were merged
into the strict replay. Every one of the 150 completed live case outcomes
matched its replayed outcome for status, diagnosis, final prompt, scores,
cost, and latency. Raw prompts and responses remain under ignored
`.local/evaluation/` because of the source's license and session privacy.

## Space Bunny results

| Context cutoff | Cases completed | True positives | False positives | False negatives | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.90 baseline | 150/150 | 6 | 6 | 62 | 50.0% | 8.8% | 15.0% |
| 0.87 calibrated | 150/150 | 8 | 7 | 60 | 53.3% | 11.8% | 19.3% |

For the baseline, recorded metered cost was **$0.02873** in total
($0.000192 per case), with a 35.1-second median and 83.4-second p95 case
latency. The calibrated run cost was **$0.02879** for the recorded successful
paths. These figures record the OpenRouter/Jev charges in this Fast run;
OpenCode Go subscription usage is not represented as a per-call dollar charge.
They exclude the first 18 aborted attempts, which accumulated another
**$0.00376** of recorded Jev charges and 28.1 minutes of summed case time
before the successful retries. The default Space Bunny writer is listed as free by
[OpenCode Go](https://opencode.ai/docs/go/).

None of the 150 cases produced a comparable original-versus-winner weak-panel
score. The harness correctly reports all 150 improvement outcomes as
**unavailable**. The high confidence gap and success-test gates caused the
optimizer to retain the original prompt, so these data establish diagnosis
accuracy and operational cost/latency but cannot establish improvement or
regression magnitude. A zero improvement rate would be misleading here.

## Context calibration

`scripts/calibrate_soar_context.py` reads the strict replay and the 150
human-labeled cases without making provider calls. It splits by a stable hash
of case ID: 118 training cases and 32 untouched holdout cases. It selects
the threshold with the highest training F0.5 (which favors precision) among
0.80–0.95 in 0.01 increments; ties choose the
higher threshold. The selected cutoff was **0.87**.

| Holdout cutoff | TP | FP | FN | Precision | Recall | F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.90 baseline | 2 | 1 | 13 | 66.7% | 13.3% | 0.370 |
| 0.87 selected | 3 | 1 | 12 | 75.0% | 20.0% | 0.484 |

The 0.87 cutoff is applied only to the exact default `gap:context` question.
Four cases needed new live responses because that change altered their
downstream paths; their results also matched strict replay. The holdout is
small, and the gains involve only one additional true positive in that split.
Other Jev question thresholds remain provisional until they have matching
human labels. The 150 private coding-agent session examples described in
`docs/evaluation-data.md` are ready for that review and are not counted as
labeled evidence yet.

## Writer comparison

The same 150 cases were rerun with `deepseek-v4.1-flash` as the only configured
model change. Both writers used OpenCode Go. The comparison retained the
baseline 0.90 context cutoff, Fast tier, seed 0, and the same other model IDs.
Separate live Jev calls mean this is a paired prompt comparison rather than
identical judge responses. All 150 DeepSeek cases completed on the first pass,
and their strict replay reproduced every case field checked above.

| Writer | Cases completed | Proposed success tests | Cases with ≥1 Jev-accepted test | Accepted tests | Cases with both a confirmed gap and accepted test | Median case latency | Recorded metered cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Space Bunny Free | 150/150 after 18 retries | 477 | 18 | 23 | 0 | 35.1 s | $0.02873 |
| DeepSeek V4.1 Flash | 150/150 on first pass | 728 | 31 | 49 | 0 | 20.2 s | $0.03228 |

DeepSeek produced more Jev-accepted tests and had lower median latency on this
set. Neither writer produced a scored rewrite, so the data cannot rank them by
actual prompt improvement. The metered cost column is the recorded OpenRouter
cost, mainly Jev; it does not price Go subscription consumption. The requested
default remains Space Bunny Free. DeepSeek's strict replay digest is
`c47c3e1f1b44fe1e7a30a1577e235a5460b2c2ba73658c0afcf524b530f8b342`.

## Planted-defect pilot

The checked-in [`evaluation/planted-pilot.json`](../evaluation/planted-pilot.json)
contains six controlled edits across writing, analysis, coding, and planning.
Each case records the requirement withheld from a fuller version of its prompt.
These are exploratory synthetic cases, not spontaneous user prompts or an
independent human diagnosis set. Both writer runs used the same Fast panel and
the default 0.87 `context` cutoff. Their six live outcomes each matched strict
replay on status, diagnosed gaps, final prompt, scores, cost, and latency.

| Writer | Cases completed | Planted gaps detected | Cases with comparable scores | Improved | Unchanged | Regressed | Cases without comparable scores | Median latency | Metered OpenRouter cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Space Bunny Free | 6/6 | 2/6 | 1 | 0 | 1 | 0 | 5 | 24.6 s | $0.00122 |
| DeepSeek V4.1 Flash | 6/6 | 2/6 | 1 | 0 | 1 | 0 | 5 | 22.3 s | $0.00148 |

The sole comparable case asked the agent to fix a failing repository test
without the relevant repository context. Both writers proposed a rewrite, but
the candidate failed fidelity, so its strong check was not run. Both weak-panel
scores were 0.0 and the original was retained. The five unscored cases have no
measured improvement outcome. The intentionally withheld details make some
labels less determinable from the visible prompt than a human-reviewed gap
label; use this pilot to exercise live paths and task breadth, not as a
general accuracy estimate or writer ranking.

The pilot replay digests are `4df11be61a2329b8edf5b93cc706bb661706a4e1aeb1aded7a93e1ec578ea955`
for Space Bunny and `1a228a69168c7c5bdd4e0bebf97f7b3ddee1d6816f35a38902482ae1b877a7a7`
for DeepSeek. The OpenCode Go subscription calls are not priced in the metered
cost column.

## Reproduction

```sh
uv run python scripts/prepare_soar_prompt_gaps.py --output .local/evaluation/soar-150.json
uv run python scripts/calibrate_soar_context.py \
  --dataset .local/evaluation/soar-150.json \
  --replay .local/evaluation/bunny-merged-replay.json \
  --output .local/evaluation/context-calibration-official.json
uv run python -m prompt_enhancer.evaluation .local/evaluation/soar-150.json \
  --replay .local/evaluation/bunny-calibrated-complete-replay.json \
  --tier fast --output .local/evaluation/bunny-calibrated-complete-report.json
uv run python -m prompt_enhancer.evaluation .local/evaluation/soar-150.json \
  --replay .local/evaluation/deepseek-merged-replay.json \
  --tier fast --writer-model deepseek-v4.1-flash \
  --output .local/evaluation/deepseek-full-replayed-report.json
uv run python -m prompt_enhancer.evaluation evaluation/planted-pilot.json \
  --replay .local/evaluation/planted-pilot-replay.json --tier fast \
  --output .local/evaluation/planted-pilot-replayed-report.json
uv run python -m prompt_enhancer.evaluation evaluation/planted-pilot.json \
  --replay .local/evaluation/planted-pilot-deepseek-replay.json --tier fast \
  --writer-model deepseek-v4.1-flash \
  --output .local/evaluation/planted-pilot-deepseek-replayed-report.json
```

The replay files are retained locally and ignored by Git. The calibrated
replay digest is
`094682a9a2fb01804ff6c6254746aed3a3b4c1c3f93b938c4aa3d8eb25715a74`.

## Follow-up audit: why real cases were unscored

`scripts/analyze_recorded_paths.py` now replays the public optimizer and counts
each stage without printing prompts. The original SOAR selection included 81
later turns from conversations, some of which rely on earlier exchanges that
were not included in the optimizer input. The original results above remain
reproducible single-turn measurements; their context accuracy must not be
interpreted as accuracy with the original conversation available.

| Recorded run | Cases | Valid test proposals | Accepted tests | Cases with confirmed gap | Cases with accepted test | Both | Candidate/scored cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOAR, Space Bunny | 150 | 477 | 23 | 12 | 18 | 0 | 0 |
| SOAR, DeepSeek | 150 | 728 | 49 | 11 | 31 | 0 | 0 |
| Planted pilot, Space Bunny | 6 | 17 | 2 | 2 | 2 | 1 | 1 |

All 150 SOAR cases reached success-test parsing and Jev faithfulness checking,
and all 150 completed. The Jev recordings use the documented `noul` response
shape (`{"type":"noul","noul":p}`); there was no parser or transport loss in
these completed replays. The 0.90 faithfulness cutoff accepted none of the 24
Space Bunny proposals on SOAR cases that had a confirmed gap. This is an
observed gate intersection, not proof that the rejected tests were good or
bad. A [blinded 100-test human review sheet](human-gap-review.md#success-test-faithfulness-review)
is prepared to resolve that question.

In the one pilot case that reached candidate grading, the candidate failed
fidelity: Jev returned 0.70 for meaning preservation, 0.84 for no invention,
and 0.74 for confining edits. It was **not** run through the candidate strong
check; the strong model scored the original only. The selector's old report
also said “strong check did not pass” for this unrun candidate. That false
reason is fixed, and the report now retains all three fidelity probabilities.

## Cleaner first-turn context calibration

The new `scripts/prepare_soar_first_turn_context.py` selects 80 human-labeled
missing-context and 70 human-labeled no-gap **first turns**, one per
conversation, excluding obvious placeholders in the source's processed text.
The 150-case development set reused 34 exact recorded answers and made 116
new live Jev calls. A separate 60-case source holdout (20 positive, 40
negative) was selected from the remaining conversations and measured live.
The sets are deliberately stratified, so precision here is not an estimate
at ordinary prompt prevalence.

The preexisting question was tuned on the development training split using a
0.50–0.95 grid and maximum F0.5. Its selected cutoff was **0.58**. The
separate holdout rejected that as a high-confidence production cutoff:

| Exact context question cutoff | Holdout TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.58, selected on development | 6 | 13 | 14 | 27 | 31.6% | 30.0% |
| 0.87, current default | 1 | 0 | 19 | 40 | 100% | 5.0% |

A clearer candidate wording, “Is essential background information missing
from this request such that a correct response requires asking the user or
guessing?”, selected 0.50 on its development training split. On the same
separate source holdout it produced 10 TP, 21 FP, 10 FN, and 19 TN: 32.3%
precision and 50.0% recall. Neither the lower cutoff nor revised wording was
adopted. The 0.87 default is retained to avoid a large increase in false
flags, but its very low first-turn recall means it is **provisional**, not a
validated general-purpose context detector. Source labels, balancing, and
possible surrounding GitHub issue context remain limitations. The other
question cutoffs and their required human labels are in the
[threshold inventory](evaluation-threshold-inventory.md).

The development and independent holdout dataset digests are
`7509c4658a1b4cb7bb79d60ce0ec1d30863627cf9ab389a159fc26df405cd58a` and
`23f0b2c62e8608f03fce0d20613de35727fefabdc82c65b6d86a4888c29ce9ac`.
The original-wording holdout replay digest is
`35c9e38c1a98394bb26cc1cabb6ccc8a76b44b96760829811dcd6eb0cb926067`.

## Cross-task participant prompts and paired writers

`scripts/prepare_rope_user_prompts.py` selected 12 participants by stable hash
from the [ROPE user study](https://github.com/mqo00/rope), pinned to commit
`1ada01830031e5882f2585577720b182deac6246`. Each participant wrote one
prompt for each of four fixed tasks: Connect4, TicTacToe, OutlineAssistant,
and TripAdvisor. These are **real participant-written prompts**, but the
study's assigned tasks are narrower than spontaneous requests in a chat box.
The original 48-case dataset digest is
`4ec4d0d46c23f47ee273a4900794d9a5bf2de7c09ba75e415851724b4922aec5`.
No source gap labels exist for these cases, so diagnosis accuracy is
unavailable. Raw study text and provider recordings remain in ignored
`.local/evaluation/`.

Both writers completed all 48 Fast-tier live cases, and every case field
matched strict replay. Thirteen prompts contained unresolved template
placeholders; a fixed regex removed them **after** recording, leaving 35
identical paired cases (9 Connect4, 11 TicTacToe, 10 OutlineAssistant, 5
TripAdvisor). The screened dataset digest is
`1f5ab3123b48442cddf2b26a58e8a84474772c2c785c5b1e303b6a56927ca539`.

| Screened writer | Completed | Proposed tests | Accepted tests | Confirmed-gap cases | Cases with both gap and accepted test | Candidate/comparably scored cases | Improved | Unchanged | Regressed | Unavailable scores | Median latency | Metered OpenRouter cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Space Bunny Free | 35/35 | 179 | 52 | 3 | 1 | 1 | 0 | 1 | 0 | 34 | 38.8 s | $0.00822 |
| DeepSeek V4.1 Flash | 35/35 | 220 | 111 | 3 | 2 | 2 | 0 | 2 | 0 | 33 | 16.7 s | $0.01010 |

All three reached candidates failed Jev fidelity, so no candidate strong
check ran and the original was retained. The two writers were both scored on
**one** identical case; both were unchanged there. The other 34 pairwise
outcomes are unavailable. The full unscreened 48-case runs scored 1 Space
Bunny case and 3 DeepSeek cases, all unchanged; the screen is the more usable
study cohort. A writer quality ranking or a representative improvement rate
cannot be inferred from these denominators. DeepSeek was faster in this run;
the cost columns are metered OpenRouter charges and omit unpriced OpenCode Go
subscription consumption. Space Bunny remains the configured default.

The Space Bunny replay digest is
`ef76a04f936e1cc523b40568b2ff668bbc09907244368347edb3b08a6eb74cdf`;
DeepSeek's is
`9847ff0a2f74075536e0d87cc009ca6061510d44124623f41ec90047a028bb63`.
The paired report is generated by `scripts/compare_writer_reports.py` and
compares only jointly scored case IDs.

## Real search requests with human clarification ratings

`scripts/prepare_clariq.py --per-level 10` selected 40 unique
[ClariQ](https://github.com/aliannejadi/ClariQ) initial search requests,
ten at each of the source's human clarification-need ratings 1–4. Rating 1
maps to no clarification needed; ratings 2–4 map to clarification needed.
The source does **not** label specific optimizer checklist gaps. The local
dataset digest is
`9f77a0d601eddfd5f2fbae93bd11ee645ec7acb9c8ef4777bc873c2549f03387`.

With clarification allowed, the default Fast-tier writer completed 40/40.
The optimizer requested no clarification on any case: 0/30 true positives,
0/10 false positives, and 30/30 missed source-rated needs. No case had
comparable original-versus-winner weak-panel scores; improvement and
regression are unavailable for all 40. Recorded metered OpenRouter cost for
the completed paths was **$0.00437**, and median case latency was **26.8 s**;
OpenCode Go subscription use has no per-call dollar price here. The source's
search-clarification judgment is broader than an exact optimizer gap
checklist, so this is a task-specific clarification audit, not proof that any
particular `gap:*` threshold is miscalibrated.

One first attempt failed at Space Bunny success-test response parsing after
accruing $0.000085512 in recorded Jev charges and 27.2 seconds. A single-case
live retry completed. Its recording was merged into the full strict replay;
the other 39 case reports were identical to their first live reports, and the
retry case matched its live retry. The final replay digest is
`afb2b5089a40774ce8e176061fc63689429ca24cb1bfc6aaf1b673f1a8e799f2`.
The failed attempt's charge and latency are **additional** to the completed
path totals above.

## Acceptance status

This section's earlier human-review dependency was resolved by the user's
explicit delegation of both pending judgments to Codex. The current
[acceptance matrix and provenance](delegated-evaluation-2026-09-23.md#acceptance-matrix)
distinguish user-delegated model labels from source or human annotations and
show which empirical limits remain. No further review form is required to
use the app. The result still does not support a representative improvement
rate or a writer quality ranking where comparable outcomes are scarce.

## Additional replay commands

```sh
uv run python scripts/screen_rope_cohort.py \
  --input .local/evaluation/rope-original-48.json \
  --output .local/evaluation/rope-screened.json
uv run python -m prompt_enhancer.evaluation .local/evaluation/rope-screened.json \
  --replay .local/evaluation/rope-bunny-replay.json --tier fast \
  --output .local/evaluation/rope-bunny-screened-report.json
uv run python -m prompt_enhancer.evaluation .local/evaluation/rope-screened.json \
  --replay .local/evaluation/rope-deepseek-replay.json --tier fast \
  --writer-model deepseek-v4.1-flash \
  --output .local/evaluation/rope-deepseek-screened-report.json
uv run python scripts/compare_writer_reports.py \
  .local/evaluation/rope-bunny-screened-report.json \
  .local/evaluation/rope-deepseek-screened-report.json \
  --output .local/evaluation/rope-paired-comparison.json
uv run python -m prompt_enhancer.evaluation .local/evaluation/clariq-40.json \
  --replay .local/evaluation/clariq-40-complete-replay.json \
  --tier fast --clarification-allowed \
  --output .local/evaluation/clariq-40-complete-report.json
```
