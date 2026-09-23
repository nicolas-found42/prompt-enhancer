# SOAR live evaluation, 2026-09-23

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
```

The replay files are retained locally and ignored by Git. The calibrated
replay digest is
`094682a9a2fb01804ff6c6254746aed3a3b4c1c3f93b938c4aa3d8eb25715a74`.
