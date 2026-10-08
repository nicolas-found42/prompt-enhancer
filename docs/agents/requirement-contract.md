# Compound source requirement controls

Issues #189–#195 extend the source ledger introduced by #187. Before ordinary
optimization, Qwen (`qwen3.8-flash`, through Gateway's writer role) extracts at most
32 obligations. Every proposed span must equal the unchanged source at its
Unicode codepoint offsets. Fenced, quoted and delegated data must not become
independent instructions. Deterministically recognized obligations stay in the
ledger even when extraction or interpretation is uncertain.

Jev audits each item's source, scope and protected interpretation, then performs
an independent **whole-source** completeness audit. Both probability and response
confidence must reach the existing 0.8 policy boundary. Missing, malformed,
low-confidence or disputed answers retain partial coverage. Item approvals never
substitute for whole-source completeness. The ledger includes source digest,
extractor model, oracle version and typed audit evidence; the run's existing
configuration and judgment records carry commit/profile/model identities.
The existing input character cap applies before extraction. Audit batches split
to fit the advertised Jev context with framing/answer reserve; a whole-source
request that cannot fit remains unresolved and never becomes window-only proof.

The report distinguishes audited coverage from candidate qualification. Known
mechanical failures still reject a draft regardless of positive semantic scores.
Unsupported sentence/word conventions, nested lists, repeated headings and
ambiguous protected boundaries remain untestable. Named sections support unique
Markdown headings or explicit `Name:` headings; counts bind to that section.
Punctuation, case and word-order edits support explicit source permissions, with
uncertainty for unsupported tokenization. Protected fenced regions bind to their
original region position, so a later duplicate cannot repair a changed region.

Required clarification has no preselected answer and cannot be skipped. Essential
missing meaning requires text; reusable `{date}`/`{venue}` variables stay literal.
Hard conflicts are checked in the same scope, pair by pair within a bounded audit.
An explicit choice supersedes only the competing obligation in the working copy;
the original source and answer provenance remain in the ledger. Further conflicting
requirements require another choice. Resume retains run identity and active time,
excluding human delay. API/job responses, check events and saved history retain
this evidence; the browser explains it under **View report → Requirement coverage**.

Run the offline maintainer controls from a feature branch:

```sh
uv run --locked python scripts/check_requirement_contract.py --output .local/requirement-contract/new-run
```

Use a new directory. The command writes a source/commit/profile/oracle receipt,
the actual test command and output, JUnit case results, and SQLite histories under
`histories/`. Every negative/uncertain scenario is a control with an independently
asserted result; passing these controls does not establish live model performance
or completion of #187. A timeout, failed/skipped/missing test, source change or
unavailable evidence exits nonzero and retains a failure receipt. Its default
allowance is 180 seconds; `--timeout` supports 1–600 seconds. CLI regression tests
also exercise incomplete coverage and timeout failure paths.

To gate supplied run archives for a release campaign, append one or more
`--release-report /absolute/path/run.json` arguments. Each must match its original
source digest and retain audited coverage, accepted item and whole-source audits,
no extraction gaps, and no unresolved conflict. Partial coverage cannot pass this
campaign gate. The command does not read credentials or call live providers.

Standard local/CI validation remains the deterministic completion authority; see
[validation](validation.md). Source semantic review follows
[quality review](../quality-review.md), with live inference only when authorized.
