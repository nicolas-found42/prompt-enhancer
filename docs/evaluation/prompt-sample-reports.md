# Prompt sample comparison reports

Use `prompt_enhancer.evaluation.prompt_experiment_report` to turn captured
Gateway answers, per-answer sample metrics, Jev judgments, and the main Prompt
improvement run status into a reproducible JSON report. The report puts the main
run status first. It then describes the separate sample experiment.

```sh
uv run --locked python -m prompt_enhancer.evaluation.prompt_experiment_report \
  --answers .local/run/sample-answers.json \
  --metrics .local/run/sample-metrics.json \
  --jev .local/run/jev-experiments.json \
  --optimizer-status .local/run/optimizer-status.json \
  --criteria-manifest .local/run/criteria-manifest.json \
  --output .local/run/prompt-sample-report.json
```

The JSON records `prompt_improvement_run` before `sample_experiment`. A missing
final optimizer result stays missing; sample answers do not change the main run
status or establish Convergence.

The report keeps the original three-sentence requirement separate from the
experimental no-introduction check. It counts terminal punctuation for the
sentence check and applies a named, deterministic heuristic to prefatory text
before a blank line for the stricter experiment check. The stricter check does
not count as an original requirement. Use a stronger sentence parser or human
review when inputs contain abbreviations, unusual punctuation, or formatting
that the recorded method cannot handle.

An improvement claim requires an explicit criteria manifest. The manifest binds
to the exact original prompt and lists all original criteria. Mark each
criterion `measured` or `unmeasured`, and name its measurement method. A measured
criterion must match a supported method and gets per-sample pass, fail, or
unresolved outcomes from the captured evidence. Unknown measured methods are
rejected. An omitted manifest, a prompt mismatch, an incomplete declaration,
or any unmeasured criterion prevents an improvement claim.

For example, a manifest for `Explain photosynthesis to a 10-year-old in three
sentences.` can record the task topic and audience as unmeasured, sentence count
as measured by `terminal_punctuation_count`, and scientific accuracy as measured
by `jev_classify`. The manifest's `all_original_criteria_listed` field is an
explicit attestation that the list includes every original requirement. Topic
or audience criteria must not be omitted from that list. Supported measured
criteria are `three_explanatory_sentences` with method
`terminal_punctuation_count`, and `scientific_accuracy` with method
`jev_classify`.

```json
{
  "prompt": "Explain photosynthesis to a 10-year-old in three sentences.",
  "all_original_criteria_listed": true,
  "criteria": [
    {"id": "task_topic", "status": "unmeasured", "method": "not_evaluated"},
    {"id": "audience_age", "status": "unmeasured", "method": "not_evaluated"},
    {"id": "three_explanatory_sentences", "status": "measured", "method": "terminal_punctuation_count"},
    {"id": "scientific_accuracy", "status": "measured", "method": "jev_classify"}
  ]
}
```

Samples pair only when both variants have the same model and sample index. The
report fails when those pairs are missing or duplicated. Its improvement rule
requires an exhaustive prompt-bound manifest with no unmeasured criterion, all
answer-quality classifications resolved, no matched pair to regress on the
original sentence-count criterion, and more successful matched pairs for the
rewrite. A Jev `review` decision remains unresolved. Relevance
ranking is reported separately and cannot change a quality classification.
Pairs with unresolved quality evidence count as unresolved, not confirmed
regressions.
This is a conservative reporting rule for the experiment; it is not a general
statistical claim about model behavior.

Jev capability evidence has a named scope. Classification can assess prompt
faithfulness or answer quality when the question asks for that judgment.
Reranking orders candidates. Verification checks claims against supplied
evidence. Deterministic count and pair checks use arithmetic. Patch review and
completion gates assess code changes and completion evidence; they do not count
as prompt or answer quality evidence.
