# Delegated review and cross-task acceptance follow-up

## Provenance and decision policy

The user delegated the two pending review materials to Codex and instructed
that the resulting judgments be treated as ground truth for this project. GLM
5.3 Flash on OpenCode Go supplied first-pass labels; Codex audited all 100
proposed success tests against the source prompts and corrected unsupported
labels. Artifacts record `user_delegated_model` or Codex model provenance.
These judgments fulfill the user's requested project decision process, **not**
an independent human annotation or inter-rater reliability claim. Original
source labels remain separate. The private sessions, prompts, raw provider
answers, review exports, and replay files stay under ignored
`.local/evaluation/`.

`scripts/review_public_prompt_gaps.py` reviewed 139 real prompts: 99 screened,
participant-written ROPE originals across four fixed study tasks, and 40
ClariQ initial search requests. It retained ROPE participant groups and did
not show the model ClariQ's source clarification ratings. The reviewed dataset
digest is `8a02201b1e9d2b5dbda6e719d99faf3ca1b784fe7415e59cf6d8293057bfee27`.
Of the 139, 101 were explicitly judged to have no checklist gap. The
38 other cases carried 43 gap labels: `language` 17, `context` 16, `goal` 9,
and `constraints` 1. Task strata were coding 61, general 23, chat 20,
research 17, writing 10, and planning 8. The cohort spans real requests but
is weighted toward fixed study tasks and short search queries; it is not a
sample of ordinary chat traffic.

The private 150-session Codex/Claude Code/oh-my-pi review was repaired before
labeling: 18 Claude metadata or sidechain entries were replaced with true
main-session first user turns. The delegated review found 77 usable cases
(35 with reconstructed prior conversation), and excluded 73 whose effective
context could not be recovered or whose turn was unsuitable. All 77 usable
cases were judged to have no checklist gap. That cohort is separate from the
139-case public cohort; its zero-positive result can only inform specificity,
not recall. Its dataset digest is
`b5c7e3cc09e2833239fe9ce32844e88aaf740293131a7bf926a89755efe73fed`.

## Success-test faithfulness

The second sheet contains 100 blinded proposed success tests from first-turn
SOAR prompts, balanced 50 per writer. GLM 5.3 Flash reviewed them, then Codex
audited every proposed test against the actual prompt. The corrected decision
set has 60 faithful, 39 unfaithful, and 1 uncertain. Source prompt groups
stay together: 71 definite judgments in training and 28 in holdout. The
training F0.5 optimum was a 0.73 Jev yes-probability cutoff. On holdout it
accepted 13 faithful and 4 unfaithful tests, missing 3 faithful tests. The
old 0.90 cutoff accepted 3 faithful and 1 unfaithful, missing 13 faithful.

The product now uses **0.80**, the preexisting derived-confidence floor, as
the effective faithfulness gate. On holdout at 0.80 it accepted 10 faithful
and 2 unfaithful tests, with 6 false negatives and 10 true negatives
(83.3% precision, 62.5% recall). This conservative choice was made after
inspecting holdout results and is therefore a product judgment, **not** an
untouched-holdout estimate of a newly selected threshold. The exact question
wording is preserved in `.local/evaluation/faithfulness-delegated-calibration.json`.

## Exact gap-question calibration

`scripts/calibrate_delegated_gaps.py` sent the exact current `gap:*` Jev
questions for all 139 reviewed public prompts and split by ROPE participant
or ClariQ query group (114 training, 25 holdout). The nine question results
are stored locally in `.local/evaluation/public-delegated-gap-calibration.json`.
This is direct question calibration; it does not account for the product's
task classifier suppressing questions outside the selected checklist.

| Exact gap question | Train positives | Holdout positives | Current holdout TP/FP/FN | Training-selected cutoff | Selected holdout TP/FP/FN |
| --- | ---: | ---: | --- | ---: | --- |
| `goal` | 8 | 1 | 0/0/1 | none: no training true positives at any grid cutoff | unavailable |
| `context` | 12 | 4 | 0/0/4 | 0.90 | 0/0/4 |
| `constraints` | 0 | 1 | 0/0/1 | none: no training positives | unavailable |
| `output_format`, `done_criteria`, `sources`, `tests`, `time_horizon` | 0 each | 0 each | 0/0/0 each | none: no positive labels | unavailable |
| `language` | 14 | 3 | 0/0/3 | 0.85 | 1/1/2 |

The selection grid was 0.80–0.95, maximizing training F0.5 with higher
cutoffs breaking positive-score ties. An all-zero training score yields no
selected cutoff. `context` retains its 0.87 cutoff; the other questions
retain 0.90. The `language` selected threshold yielded only 50% precision on
three positive holdout cases. Questions with no positive labels cannot have
recall calibrated by this cohort. The low prevalence, fixed tasks, and
model-derived labels limit generalization. Other Jev decision families need
their own sentence, value, candidate, or output judgments; prompt-level gap
labels cannot validate those thresholds. The full inventory remains in
[`evaluation-threshold-inventory.md`](evaluation-threshold-inventory.md).

## Measured product outcomes

The 40 ClariQ Fast-tier Space Bunny cases completed under the earlier 0.90
faithfulness gate. Against the user-delegated checklist labels, the product
found 0 of 10 asserted gap occurrences and made no false flags. All 40
original-versus-winner scores remained unavailable. Metered OpenRouter cost
was $0.00437. The source's human clarification-need rating is a different
annotation and remains reported in
[`evaluation-results-2026-09-23.md`](evaluation-results-2026-09-23.md).

The paired 35 screened ROPE cases were rerun live with the 0.80 gate and
strictly replayed. Both writers completed 35/35 with identical options except
writer model. Against delegated gap labels, each found 3 of 14 asserted gap
occurrences, with no false flags (21.4% recall on these 35). The other 11
labels were missed. The 40 ClariQ cases and 35 ROPE cases are source-different
cohorts; their costs are reported separately.

| ROPE writer | Proposed tests | Accepted tests | Confirmed-gap/candidate cases | Fidelity passes | Comparably scored | Improved | Unchanged | Regressed | Unavailable scores | Median latency | Metered OpenRouter cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Space Bunny Free | 176 | 121 | 3 | 0 | 3 | 0 | 3 | 0 | 32 | 44.0 s | $0.01215 |
| DeepSeek V4.1 Flash | 230 | 184 | 3 | 0 | 3 | 0 | 3 | 0 | 32 | 20.9 s | $0.01330 |

All three jointly scored cases were unchanged for both writers; 32 pairwise
outcomes are unavailable. Each writer generated three candidates, all of which
failed Jev fidelity. The replays matched every live case object exactly.
After the recordings gained an explicit `faithfulness_threshold: 0.8` field
(below), Space Bunny's replay digest is
`a0ce72a408fc79d6c95f5d7eb24913c55aab009a7e62162f1dad4daa197c241d`
(originally `97c7cbe75ba46577dbea32c70d4679c43b50f649563faafad5a7d6784499ffce`);
DeepSeek's is
`3d06b404c7c51070c28532a88d36b69c6017ab1372389210db4cd13cd4499027`
(originally `c13663ae2fa385d59d04a1df410fa7623a83e7ca34ae1df3261d72ee4a07320a`).
The responses are unchanged.
The only supported writer differences here are accepted-test count, latency,
and metered Jev/weak-model charge. There is no evidence to rank their
rewrite quality. OpenCode Go writer usage remains subscription consumption
with no invented per-call dollar price. No unscored case is counted as an
unchanged or improved outcome.

Inspecting the six rejected candidates found a code defect: the candidate
writer instruction said to preserve unflagged text, but the structured writer
request omitted the diagnosis and flagged sentences. Both writers added
unrequested travel details or changed more text than the chosen strategy
allowed. `CandidateBatchRequest` now carries the diagnosis to the writer,
and the instruction explicitly forbids invented facts and requirements. A
focused public-seam regression test verifies that the diagnosis reaches the
writer. The writer instruction is now versioned. Version 1 is the exact historical
request without diagnosis; version 2 is the current default. The
success-test faithfulness gate is also engine metadata, because the move from
0.90 to 0.80 accepts more tests and so changes later Jev requests: under the
new gate, three cases in the earlier 0.90 ROPE recording failed strict replay.
Live recordings now store `writer_instruction_version` and
`faithfulness_threshold`. A recording without them replays as version 1 at
0.90. The two 0.80 ROPE recordings above were annotated with
`faithfulness_threshold: 0.8`, with pre-annotation copies kept under
`.local/evaluation/backup-pre-faithfulness-metadata/`. Under the current code,
every replay command in this document and in
[`evaluation-results-2026-09-23.md`](evaluation-results-2026-09-23.md)
(SOAR 150 for both writers, the planted pilot for both writers, ROPE 35 at
both gates for both writers, and ClariQ 40) matched its recorded report case
for case. Regression tests cover an unversioned historical recording and a
version 2 recording, and verify that a version 2 recording mislabeled as
version 1 fails strict replay. `RecordingGateway` now serializes its writes,
because the weak-model panel records from worker threads and concurrent saves
raced on one temporary file.

A live rerun of the three affected cases was attempted for each writer. The
first attempts failed before candidate generation because OpenCode Go
returned HTTP 403 with Cloudflare error 1010, including a direct model-list
request. [Cloudflare's documentation](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-1xxx-errors/error-1010/)
describes 1010 as a site-owner block based on the client's signature. A later
recheck got HTTP 200 from the model list, but every chat completion, for both
Space Bunny Free and DeepSeek V4.1 Flash, returned HTTP 403: "An active
OpenCode Go subscription is required to use Go models." The version 2 reruns
failed at the first writer call for all three cases. The paired results above
therefore describe the **version 1 writer instruction**. The impact of the
diagnosis-input fix remains unmeasured until the Go subscription is active
again. The default writer has not been silently substituted.

## Acceptance matrix

| Spec requirement | Current evidence | Status |
| --- | --- | --- |
| 57, real prompts across task types | 139 labeled real ROPE/ClariQ prompts across six task strata; 35 ROPE and 40 ClariQ product runs measured | Cross-task cohort complete; fixed study and search tasks limit ordinary-chat generalization |
| 58, planted defects | Six-case live and strict-replay pilot in the earlier report | Exploratory path coverage complete; some hidden requirements are not inferable from visible prompts |
| 59, 100–200 real prompts labeled for actual gaps | 139 explicit user-delegated model judgments, plus 77 separately usable private-session judgments | Complete under the user's delegated ground-truth decision; no independent human validation claim |
| 60, accuracy, improvement, regression, and cost | Exact-question and product diagnosis accuracy, costs, and honest scored/unavailable denominators reported | Operationally measured; scored rewrites remain too rare for a representative improvement-rate estimate, and the version 2 writer awaits live validation once the Go subscription is active |
| 61, per-question Jev calibration | All nine exact `gap:*` questions measured on group-split labels; faithfulness threshold audited and changed | Partial: sparse positives prevent several gap cutoffs from being tuned, and other Jev decision families lack their own judgment sets |
| 62, recorded reproducibility | Completed live paths have strict recorded replay | Complete for measured paths |
| 63, outcome-based writer comparison | Paired 35-case ROPE study under identical options, with 3 jointly scored unchanged cases and 32 unavailable pairs | Partial: too few jointly scored rewrites to rank writer quality; version 2 writer rerun blocked by Go 403 (inactive subscription) |

The remaining limits are measured limitations, not missing review forms.
Provider-backed optimization in the app needs an active OpenCode Go
subscription for the writer. After that, more outcome evidence would require
prompts that actually reach a faithful candidate under the current safety
gates.

## Reproduction

```sh
uv run --env-file .env python scripts/review_public_prompt_gaps.py \
  --rope .local/evaluation/rope-screened-all.json \
  --clariq .local/evaluation/clariq-40.json \
  --output .local/evaluation/public-delegated-gap-review.json \
  --raw .local/evaluation/public-delegated-gap-review-raw.json
uv run --env-file .env python scripts/calibrate_delegated_gaps.py \
  --dataset .local/evaluation/public-delegated-gap-review.json \
  --decisions .local/evaluation/public-delegated-jev-decisions.json \
  --output .local/evaluation/public-delegated-gap-calibration.json --measure
uv run python scripts/calibrate_faithfulness.py \
  --review .local/evaluation/faithfulness-delegated-review.csv \
  --evidence .local/evaluation/faithfulness-review-evidence.json \
  --output .local/evaluation/faithfulness-delegated-calibration.json
uv run python scripts/human_gap_review.py import \
  --batch .local/evaluation/local-agent-review-batch.json \
  --review .local/evaluation/prompt-gap-delegated-review.json \
  --minimum 1 --output .local/evaluation/private-delegated-gap-dataset.json
uv run python -m prompt_enhancer.evaluation .local/evaluation/rope-screened.json \
  --replay .local/evaluation/rope-bunny-faithfulness-080-replay.json \
  --tier fast --output .local/evaluation/rope-bunny-faithfulness-080-strict-report.json
uv run python -m prompt_enhancer.evaluation .local/evaluation/rope-screened.json \
  --replay .local/evaluation/rope-deepseek-faithfulness-080-replay.json \
  --tier fast --writer-model deepseek-v4.1-flash \
  --output .local/evaluation/rope-deepseek-faithfulness-080-strict-report.json
uv run python scripts/compare_writer_reports.py \
  .local/evaluation/rope-bunny-faithfulness-080-report.json \
  .local/evaluation/rope-deepseek-faithfulness-080-report.json \
  --output .local/evaluation/rope-faithfulness-080-paired-comparison.json
uv run python scripts/score_delegated_reports.py \
  --dataset .local/evaluation/public-delegated-gap-review.json \
  --report rope-bunny=.local/evaluation/rope-bunny-faithfulness-080-report.json \
  --report rope-deepseek=.local/evaluation/rope-deepseek-faithfulness-080-report.json \
  --report clariq-bunny=.local/evaluation/clariq-40-complete-report.json \
  --output .local/evaluation/public-delegated-observed-metrics.json
```

The strict replay commands need no provider access. Once Go access works, the
version 2 writer check reruns the affected cases live, merges them, and
strictly replays the full 35 (shown for Space Bunny; DeepSeek adds
`--writer-model deepseek-v4.1-flash`):

```sh
uv run --env-file .env python -m prompt_enhancer.evaluation .local/evaluation/rope-affected-3.json \
  --live --record .local/evaluation/rope-bunny-writer-v2-affected-replay.json \
  --tier fast --output .local/evaluation/rope-bunny-writer-v2-affected-report.json
uv run python scripts/merge_live_case_retries.py \
  --base-record .local/evaluation/rope-bunny-faithfulness-080-replay.json \
  --retry-record .local/evaluation/rope-bunny-writer-v2-affected-replay.json \
  --retry-report .local/evaluation/rope-bunny-writer-v2-affected-report.json \
  --output .local/evaluation/rope-bunny-writer-v2-merged-replay.json
uv run python -m prompt_enhancer.evaluation .local/evaluation/rope-screened.json \
  --replay .local/evaluation/rope-bunny-writer-v2-merged-replay.json \
  --tier fast --output .local/evaluation/rope-bunny-writer-v2-merged-strict-report.json
```

The merged recording takes the retry's writer version. Unaffected base cases
never reached the writer, so strict replay of all 35 checks that the merge is
sound.

The model-review commands resume completed rows and do not resend them.
`--minimum 1` imports the private usable subset without claiming it alone
meets the 100-case target; the separate public cohort has 139 labeled cases.
