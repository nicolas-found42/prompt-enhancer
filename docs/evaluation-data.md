# Evaluation data

The evaluation harness accepts real, synthetic, and hand-labeled cases. The
three-case fixture in `tests/fixtures/evaluation/` only checks the harness
contract; it is too small to calibrate Jev thresholds.

The local evaluation working directory is `.local/evaluation/`. It is ignored
by Git because source prompts, agent-session text, and provider responses can
contain private content. The live CLI can create a strict request-keyed replay:

```sh
uv run --env-file .env python -m prompt_enhancer.evaluation .local/evaluation/soar-150.json \
  --live --record .local/evaluation/soar-live-replay.json \
  --tier fast --output .local/evaluation/soar-live-report.json
uv run python -m prompt_enhancer.evaluation .local/evaluation/soar-150.json \
  --replay .local/evaluation/soar-live-replay.json \
  --tier fast --output .local/evaluation/soar-replayed-report.json
```

The recording also stores each case's measured cost and latency. Replay uses
those observations while recomputing diagnosis and improvement from recorded
model responses; it never makes provider calls.

The current model split is OpenCode Go for `space-bunny-free` (writer),
`glm-5.3-flash` (strong check), `mimo-v2.6-flash`, and
`muse-spark-1.3-contributor` (the two additional Deep weak models). OpenRouter
supplies the pinned `typesafe/jev-1.13-20260917` snapshot and the three default
Llama/Mistral weak models. New recordings store the snapshot returned by Jev
for each decision. Replay checks it against `PROMPT_ENHANCER_JEV_MODEL` (or the
default pin); use `--allow-snapshot-mismatch` only when comparing across snapshots.
Historical recordings without snapshot metadata remain replayable, with unknown
answer provenance.
The Muse default reflects the user's explicit selection; [OpenCode Go's model
notes](https://opencode.ai/docs/go/) state that Contributor prompts and
completions may be used to train Meta models.

## Offline per-question Jev calibration

The calibration mode consumes labeled raw Jev answers; it does not call a
provider. Run it on the checked-in synthetic fixture or a private event
manifest:

```sh
mkdir -p .local/evaluation
python -m prompt_enhancer.evaluation --calibrate \
  tests/fixtures/evaluation/calibration_known_answer.json \
  --output .local/evaluation/calibration-report.json \
  --artifact .local/evaluation/calibration-artifact.json
# Equivalent subcommand form:
python -m prompt_enhancer.evaluation calibrate \
  tests/fixtures/evaluation/calibration_known_answer.json \
  --output .local/evaluation/calibration-report.json \
  --artifact .local/evaluation/calibration-artifact.json
```

The report is JSON on stdout or at `--output`; the versioned artifact is written
to `--artifact` (or, when `--output` is set and `--artifact` is omitted, to the
same path with `.artifact.json` as its suffix). Both retain per-question
identity, metrics, threshold/predicate, partitions, verdict, and evidence.

A manifest has an `events` array (and may include top-level `schema_version`,
`name`, and `metadata`). Each event should explicitly record a unique `id`,
non-empty `source_group`, `example_id`, `label_provenance`, stable `question_id`,
exact `question`, `family`, `primitive`, `criteria`, `event` mapping, label,
`answering_snapshot`, `rubric_version`, `repeat_index`, and raw `answer`; unique
`request_id` and `answer_id` identify independent requests/responses. For
example, one Noul event can be:

```json
{
  "id": "gap-00-r0",
  "source_group": "gap-group-00",
  "example_id": "gap-example-00",
  "question_id": "gap:goal",
  "question": "Is the required piece 'goal' confidently missing from the request?",
  "family": "gap",
  "primitive": "noul",
  "criteria": ["no", "yes"],
  "event": {"positive_class": "yes"},
  "label": true,
  "label_provenance": "synthetic_known_answer",
  "answering_snapshot": "typesafe/jev-1.13-20260917",
  "rubric_version": "default-v1",
  "repeat_index": 0,
  "request_id": "gap-request-00-r0",
  "answer_id": "gap-answer-00-r0",
  "answer": {"type": "noul", "probability_true": 0.95}
}
```

Map the labeled event to the raw answer's primitive: Noul uses its
`probability_true` for the positive event (set `event.polarity` to `negative`
to target the complement); Choice answers carry `choice` and a `probabilities`
map, with `event.expected_class` naming the labeled option, or
`event.selected_correctness: true` when the label is the expected option and
the event is whether the selected option was correct. In that mode the binary
prediction is the probability assigned to the selected option, while the
categorical label still scores the full Choice distribution. Score answers carry
`levels` probabilities and require `event.boundary` (numeric levels at or above
it form the positive event); their full ordinal distribution is scored with
the mean cumulative Brier loss across adjacent level boundaries. See the fixture
for complete Noul, Choice, and
Score examples. Keep the full question identity—question text, primitive,
criteria/mapping, family/schema and rubric versions, policy version, and
answering snapshot—consistent within a question. These fields bind the result
to that exact question and Jev snapshot; do not combine labels or answers with
different identities. Label provenance must describe its real source, such as
source annotation, human review, delegated judgment, or synthetic fixture.
Give each independent repeat a distinct non-negative `repeat_index`; when
present, `request_id` and `answer_id` must also be distinct. A cached response
is not an independent repeat.

By default, complete `source_group`s are assigned deterministically to
disjoint fit/calibration/evaluation partitions (60/20/20, seeded; default seed
1729). An explicit `partition` may instead be `fit`, `calibration`, or
`evaluation`, but every event in a source group must use the same partition.
Optional `--fit temperature` fits only on fit groups; threshold selection uses
calibration groups, and evaluation groups are held for metrics/verdicts. The
artifact records `mode: none` with an unavailable reason when the fit partition
has no usable labels, so runtime and evaluation both use the raw probabilities.
When `--calibration-policy` changes a verdict threshold or evidence requirement,
set a distinct `policy_version` in that policy and in every matching input event.
The runtime `DecisionPolicy` must use the same version to apply the artifact;
version or snapshot mismatches abstain.
The CLI bounds inputs by default to 100 source examples, 3 repeats per
question/example/arm (`--runs`), and 5,000 question evaluations; override with
`--max-source-examples`, `--runs`, and `--max-question-evaluations` as needed.
For offline records, `--runs` limits the repeats already present; it never
duplicates one answer to manufacture stability evidence.

To capture live evidence, supply one unanswered primary event per
question/example with its complete identity, label provenance, and non-empty
state. The command makes a fresh Gateway request for every repeat and for each
matching empty-state control, then writes the raw event manifest for offline
replay. A live run requires an explicit finite dollar budget and a recording
path:

```sh
uv run --env-file .env python -m prompt_enhancer.evaluation calibrate \
  .local/evaluation/calibration-templates.json --live --runs 3 \
  --budget 2.00 --request-cost-ceiling 0.01 \
  --record .local/evaluation/calibration-recording.json \
  --output .local/evaluation/calibration-live-report.json
uv run --locked python -m prompt_enhancer.evaluation calibrate \
  .local/evaluation/calibration-recording.json \
  --output .local/evaluation/calibration-replayed-report.json
```

The CLI disables Gateway retries during capture, records zero retries and one
identity per provider attempt, and reserves `--request-cost-ceiling` USD before
each request. It stops with a partial report when the next reservation would
exceed `--budget` or a source/request limit. Because provider charges arrive
after a request, the ceiling must be conservative; an actual charge above it
stops capture immediately and is reported as a ceiling overrun. The report
keeps observed cost and reserved cost separately. Offline replay is
deterministic; each new live capture has a new request identity.

The default `issue-50-v1` verdict policy is provisional: a gate needs at least
30 evaluation groups, 5 positive and 5 negative examples, precision at least
0.90, recall at least 0.50, coverage at least 0.90, independent repeats and a
state-blind control; its Brier loss must beat the control. A ranker requires a
bootstrap AUC lower bound above 0.5. These are explicit policy defaults, not
claims of established Jev accuracy. The known-answer fixture is synthetic and
proves parsing, fitting, reporting, and artifact code paths—not empirical
accuracy. Any live mode must be explicit and have a finite, positive dollar
budget (`--budget USD`); calibration rejects `--live` without an explicit
budget and `--record`. Keep private prompts, answers, reports, and artifacts under ignored
`.local/evaluation/`.

## Local coding-agent sessions

An inspection of the local Codex, Claude Code, and oh-my-pi histories found
735, 1,589, and 1,682 user/assistant prompt/result pairs, respectively. The
oh-my-pi count uses 700 main session logs only; an initial scan also counted
`__advisor.jsonl` notes as user prompts, so that scan and its sample were
discarded. The corrected source has 973 pairs that pass the batch's length,
result, and likely-secret screens, spread across 162 sessions. A private batch
at `.local/evaluation/local-agent-review-batch.json` samples 50 pairs from
each source across 150 distinct sessions. Historical assistant results are
included for review; they are not correctness or gap labels. Its
`human_labels` fields remain null until a person reviews them. Many prompts
refer to local files, codebases, or prior conversation, so live evaluation
would also need that missing context to measure task success fairly. These
cases cannot be counted as a hand-labeled diagnosis benchmark yet.

The offline [review form and importer](human-gap-review.md) now make those
judgments auditable. They preserve reviewer identity, source-session inspection,
task type, and exact reconstructed context. The 150 prompts and historical
results remain private; generating the form does **not** turn null labels into
human gold.

For each batch item, a reviewer should inspect the prompt and historical
result, then enter checklist keys for gaps actually present in the prompt
(`goal`, `context`, `constraints`, `output_format`, `done_criteria`, plus any
task-specific keys). An empty list means the prompt has no such gap; an
uncertain judgment should be marked for adjudication rather than guessed.
Reviewers should not infer prompt quality from the historical result alone.
This batch is the route to labels for the checklist questions that SOAR does
not cover directly.

## Ready-to-import datasets

| Dataset | What its prompts represent | Human labels | Best use |
| --- | --- | --- | --- |
| [SOAR Lab prompt knowledge gaps](https://github.com/SOAR-Lab/prompt-knowledge-gap) (MSR 2025) | Real developer–ChatGPT turns shared in GitHub issues | No gap, missing context, missing specification, unclear instruction, multiple context | Primary 150-case real-prompt gap evaluation |
| [ClariQ](https://github.com/aliannejadi/ClariQ) | Real conversational search requests | Clarification need, rated 1–4 | Independent 150-case binary clarification evaluation |
| [ClarifyCodeBench](https://github.com/fangz-cs/ClarifyCodeBench) (2026) | LiveCodeBench tasks manually edited by deleting required details | 11 fine-grained underspecification types, with key questions and answers | Separate 150-case coding stress set; these are controlled edits, not spontaneous prompts |

`scripts/prepare_soar_prompt_gaps.py` selects 150 distinct conversations from
the 433-conversation SOAR Lab corpus. It takes the source's per-turn labels and
stratifies by gap type. Multi-label cases keep all their labels, so the final
per-label counts can exceed a selection quota. Prompts longer than 8,000
characters are excluded without truncating them. The source replaces some code
and error text with placeholders; the importer preserves that representation.
The four source gap labels do not all match the optimizer's current checklist
keys. Each SOAR case retains its source labels, but the harness scores only
`context` against SOAR's missing-context annotation. It counts explicit
"No gap" cases as negatives when calculating false positives.

A follow-up audit found that **81 of those 150 selected turns are not the first
turn** of their conversation. Some prompts depend on preceding exchanges that
were not supplied to the optimizer; source labels cannot automatically be
interpreted as the gap truth of that isolated turn. The initial live run is a
single-turn diagnostic measurement with that limitation. For a cleaner
context-only calibration, `scripts/prepare_soar_first_turn_context.py` selects
80 missing-context and 70 no-gap first turns without obvious processed-text
placeholders, plus a disjoint 60-case holdout from remaining conversations.
This does not solve the separate problem that surrounding GitHub issue context
may have been available to the original user and annotator.

`scripts/prepare_clariq.py` converts 150 distinct real user requests. It maps
rating 1 to no `clarification_need` label and ratings 2–4 to that single label.
It does not claim the annotators identified specific rubric gaps. The harness
scores `clarification_need` from whether the run actually needs user input;
run this dataset with `--clarification-allowed`. The selection
is stratified 25/50/50/25 at levels 1/2/3/4. ClariQ's 50 distinct development
topics remain available for a holdout.

`scripts/prepare_clarifycodebench.py` selects 150 of 419 human-annotated tasks.
The source labels are kept at their original granularity, while the harness
scores `output_format`, the type shared with the optimizer's checklist. The source includes
the clarification question and answer for each removed detail. The importer
marks these cases `synthetic` because the ambiguities were introduced by
human editing.

Run locally:

```sh
python3 scripts/prepare_soar_prompt_gaps.py --output /tmp/soar-evaluation-150.json
python3 scripts/prepare_clariq.py --output /tmp/clariq-evaluation-150.json
python3 scripts/prepare_clarifycodebench.py --output /tmp/clarifycodebench-evaluation-150.json
.venv/bin/python -m prompt_enhancer.evaluation /tmp/soar-evaluation-150.json --replay path/to/recorded-responses.json --output /tmp/soar-report.json
.venv/bin/python -m prompt_enhancer.evaluation /tmp/clariq-evaluation-150.json --clarification-allowed --replay path/to/clariq-recorded-responses.json --output /tmp/clariq-report.json
```

SOAR Lab and ClariQ declare no repository license, so their generated prompt
files stay local. ClarifyCodeBench's annotations use MIT, while the underlying
LiveCodeBench tasks have separate terms. A real evaluation replay must contain
responses for each selected request; the small fixture replay is for the smoke
dataset only. These importers create evaluation inputs, not model responses or
measured precision, recall, cost, and latency.

The 2026-09-23 full live and replayed SOAR results, context calibration, and
writer comparison are in [the evaluation report](evaluation-results-2026-09-23.md).
It also records a six-case live pilot using
[`evaluation/planted-pilot.json`](../evaluation/planted-pilot.json) across four
task types. The pilot contains controlled edits with recorded withheld
requirements, so its labels do not replace review of spontaneous prompts.

Other sources considered: [WildChat](https://huggingface.co/datasets/allenai/WildChat)
has broad real user prompts but no human gap labels;
[LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) is gated;
[HumanEvalComm](https://github.com/jie-jw-wu/human-eval-comm) has 762 human
verified ambiguity, inconsistency, and incompleteness variants, but they are
controlled coding benchmark modifications;
[MIMICS-Manual](https://arxiv.org/abs/2006.10174) has more than 2,000 real Bing
queries with human ratings of clarification questions and answers, rather than
prompt gap types; [PromptAmbiguityDataset](https://github.com/SCAlabUnical/PromptAmbiguityDataset)
includes annotated coding, analysis, and writing examples but does not establish
that they were spontaneous user prompts or declare a license. These are useful
for targeted coverage but are not interchangeable gold sets.

## Further 2026 datasets for targeted evaluation

- [MARCH](https://github.com/jeonghyunpark2002/MARCH) (ACL 2026) releases
  2,209 multi-hop questions with semantic, syntactic, or constraint ambiguity,
  clarified interpretations, and answers. Its repository is MIT licensed and
  was last pushed in April 2026. The creators report human validation on a
  sampled 60 instances, so the full set should not be treated as 2,209
  individually human-verified gap labels. Use it to stress clarification of
  multi-hop questions, outside the primary real-prompt accuracy score.
- [Clarify-Then-Search Hard518](https://clarify-then-search.github.io/)
  (KDD 2026) releases 518 paired information-seeking queries based on real
  Baidu queries. It supplies an underspecified query, hidden clear intent,
  and static answer nuggets, making it useful for downstream clarification
  utility. The underspecified queries are deliberately blurred from fused
  intents, so they are a controlled benchmark rather than spontaneous user
  prompts; its labels do not map directly to the current gap checklist.
- [MTRAG-UN](https://github.com/IBM/mt-rag-benchmark) (ACL 2026) extends the
  human-generated MTRAG conversations for multi-turn retrieval and uncertainty
  behavior. It can test whether clarification helps in context, but it does
  not provide per-prompt labels for this optimizer's gap types.

The [public-source follow-up](evaluation-source-followup.md) adds ROPE,
PELS, and other leads. ROPE supplies participant-written prompts across game,
travel, and outline-assistant tasks under Apache-2.0, but no per-prompt gap
gold. `scripts/prepare_rope_user_prompts.py` selects 48 original prompts from
12 participants; `scripts/screen_rope_cohort.py` excludes unresolved template
placeholders before reporting a 35-case cross-task outcome probe. All prompts
from one participant were initially selected together. The study tasks remain
narrow and the prompts were written for a study rather than observed in
ordinary chat use.

### User-delegated gap judgments (2026-09-23)

The user delegated both pending review sheets to Codex and instructed that
those decisions be used as this project's ground truth. The judgments retain
`user_delegated_model` provenance throughout; they are not source annotations
or human inter-rater validation. `scripts/review_public_prompt_gaps.py` labels
the 99 screened original ROPE prompts from all 30 study participants and the
40 selected ClariQ initial queries. Every row retains the source dataset and
ROPE participant or ClariQ query group. GLM 5.3 Flash on OpenCode Go made the
task and gap judgments using the actual prompt without seeing the source's
clarification-need rating. There are 139 labeled real prompts across coding,
writing, planning, general, chat, and research strata; 101 have no asserted
checklist gap. The source prompts, model responses, and resulting dataset stay
under ignored `.local/evaluation/`.

The private agent-session batch was also corrected before delegated review.
`scripts/repair_claude_review_batch.py` replaced 18 Claude metadata or
sidechain selections with main-session first user turns. The original batch
was backed up under ignored `.local/evaluation/`. Replacements are ranked
across every Claude session present when the script runs, so a later rerun
picks different sessions (a rerun on 2026-09-23 differed in 14 cases). The
stored `.local/evaluation/local-agent-review-batch.json` is the reviewed
batch of record; its digest is checked at import. The review script checks
that each selected turn occurs in its original main session, supplies up to
two preceding user turns with intervening assistant messages, and excludes a
case when its needed context cannot be reconstructed. Original results are
reviewer context, never gold answers. The full private prompts and responses
remain local. This cohort is reported separately from the public 139, because
its many context-dependent exclusions change the denominator.
