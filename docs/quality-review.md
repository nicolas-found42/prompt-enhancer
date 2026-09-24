# Advisory semantic code review

Jev reviews relationships between code and its stated contracts. Ruff/Prettier,
ESLint, types, tests, coverage, security scanning, and dependency audits remain the
required deterministic checks. Semantic findings never block a commit or merge.

## Agent procedure

Use this procedure when changing source code or tests, reviewing a branch, or
modifying the semantic review tooling. Documentation-only changes need link and
content checks rather than model inference.

1. **Fix the comparison base before work.** For a branch review, use the PR's
   base or the user-specified ref. For a new change, record the starting commit's
   full SHA. Keep that base through fixes and pushes so a moved `origin/main`
   cannot silently make the review empty.
2. **Run the required deterministic checks.** Follow the
   [validation workflow](agents/validation.md) for local hooks and additional CI
   checks. For changes to the semantic review runner, rules, or
   recordings, also run the offline quality-review tests described below.
3. **Review the committed snapshot.** Once the task has an authorized commit,
   run `lint` without `--live` against the recorded base and reviewed head. Run
   `plan` for changed Gateway methods; use `--findings` for explicit review claims
   with their relevant tests, helpers, and trusted contracts. Inspect the selected
   evidence and budgets. If the task ends with uncommitted changes, report that
   this tool has not reviewed them; its input is Git commits, not the working tree.
4. **Use the appropriate inference mode.** Honor existing authorization for live
   inference in the task. When authorized, use `lint --live` and, for applicable
   Gateway or candidate findings, `run --live`. Otherwise use offline planning or
   an applicable recording. A dry run, replay, missing key, or partial review must
   be identified as such. Follow the limits below; narrow scope or report omitted
   work when a budget is exceeded.
5. **Triage and close the loop.** Check each finding against its source and policy.
   Fix substantiated defects within the task's scope; retain the reason for
   dismissing or deferring a candidate. After a fix changes the reviewed code,
   rerun the affected review on the new committed head with the same base. In the
   final handoff, report the reviewed SHA and scope, mode, completion status,
   actionable or unresolved findings, and report location. Jev judgments inform
   the review; a completed request or high confidence does not establish that
   code is correct.

## Install and select evidence

Use Node 24+ and the repository's Python environment:

```sh
uv sync --locked
npm ci --prefix tools/quality --ignore-scripts
```

The tools package pins `jev-lint` 0.7.0 separately from the web application.
`.jev-lint.yaml` enables three rule families for Python and TypeScript: comments
versus implementations, names versus behavior, and tests versus their claims.
The Python assertion rule is local because upstream's Python test-name rule only
checks whether the name describes the exercised operation. Every rule is advisory;
its cutoff is provisional for this repository.

Review committed changes, first without calling a model:

```sh
uv run --locked python scripts/review_quality.py lint --base origin/main --head HEAD
uv run --locked python scripts/review_quality.py plan --base origin/main --head HEAD
```

`lint` builds a temporary Git repository containing immutable before/after source
snapshots and trusted rules. It runs the pinned linter in diff mode. Reviewed code
is never imported or executed, and target-branch configuration is not loaded.
Dirty or staged changes are deliberately absent: commit them before this review.
The default limits are 12 changed files, 100 KB per file, 250 KB total source,
50,000 estimated input tokens, one concurrent request, and 180 seconds per linter
subprocess. Library retries can add billed input beyond the initial estimate.
An omitted file, degraded context, or missing verdict is visible in the report.
To narrow the scope explicitly, repeat `--path path/to/changed.py`. A path filter
is recorded in the report. `--max-cases` (files for lint) and `--max-tokens` allow
bounded, explicit increases; inspect the plan first.

## Live review

Only `--live` enables inference, using `OPENROUTER_API_KEY` and
`typesafe/jev-1.13`. A missing key reports `skipped` without a request.

```sh
uv run --locked python scripts/review_quality.py lint --base origin/main --head HEAD --live
uv run --locked python scripts/review_quality.py run --base origin/main --head HEAD --live
```

The generic linter uses OpenRouter's `/api/v1/systemone` compatibility endpoint.
Its reported dollar cost is a fixed-price estimate. The custom runner reuses the
application's Gateway and its existing Decisions API route and usage accounting.
It does not change the application's pinned model default or the Gateway's raw
answer contract. Its default is at most 12 cases, 32 KB of state each, two
independent Choice questions per case, a 20-second request timeout, and no retries.
The runner does not interpret the parser's derived Noul confidence as provider
confidence. Generic Noul rules retain their probability rather than a severity.

## Architecture and candidate review findings

`plan` and `run` automatically select changed Gateway `chat`, `decide`, and
`decide_batch` functions for ADR-0001 review. Their evidence includes the prior
function, current function, and the ADR from the trusted base commit. Other checks
are explicit candidates, supplied as a JSON array with `--findings findings.json`:

```json
[
  {
    "id": "review-1",
    "rule_id": "finding-support",
    "path": "src/prompt_enhancer/example.py",
    "line": 10,
    "end_line": 11,
    "quote": "the exact two lines at the reviewed commit",
    "claim": "A specific behavioral problem established by this code.",
    "evidence_paths": ["tests/test_example.py"]
  }
]
```

That example illustrates the schema; substitute real paths and verbatim lines.
Spans must intersect the diff. `quality/review-rules.json` supplies trusted policy
for `finding-support`, `gateway-raw-answers`, and `error-contract`. Add a rule's
`contracts` paths there to review a specific acceptance criterion against its
base-commit specification. Candidate text never supplies or overrides trusted
policy. Related evidence must be regular source files; symlinks, path traversal,
missing files, mismatched quotes, and oversized evidence are rejected before
inference. Python evidence keeps the complete enclosing function. Referenced
helpers, callers, and tests should be provided with `evidence_paths`. Files under
`fixtures/` directories are rejected as evidence. Structurally valid evidence does
not guarantee that all semantic context is present.

Each case asks whether the claim is supported, contradicted, or lacks evidence,
and independently whether the claimed behavior was introduced or preexisting.
All results remain visible, including contradicted candidates and uncertainty.
There is no confidence-based automatic suppression or approval. Jev does not
write explanations: the report retains the candidate's claim and raw judgments.

## Reports, replay, and evaluation

`--output .local/quality-review/name.json` selects an output file. The custom
runner also writes a `.plan.json` evidence record and `.md` report. Records contain
raw answers, requested/served models, rule/question digests, source commit IDs,
latency, and usage. Keep real-source reports in ignored `.local/`; do not commit
private evidence. Linter JSON includes its native `recording` object; extract that
object to a file to use the pinned CLI's `replay` command offline.

Statuses mean:

- `complete`: every selected custom case or linter subject received a usable result;
  it does not assert that the code is correct.
- `partial`: some evidence, subjects, or results are missing or exceeded budgets.
- `skipped`: live inference was not enabled, its key was absent, or tooling was absent.
- `failed`: the review could not be completed or its answers were unusable.
- `planned`: a custom evidence plan was created without inference.

Warnings exit zero; partial/failed reviews exit 2. The optional workflow records
these conditions without making semantic findings a required check. For replay:

```sh
uv run --locked python scripts/review_quality.py replay --base origin/main --head HEAD --recording .local/quality-review/name.json --output .local/quality-review/replayed.json
```

Pass the original `--findings` when replaying candidate findings. Changed commits,
evidence, rules, questions, or model identity invalidate a custom recording.
Replay reproduces report policy, not a fresh measure of model stability.

A small, hand-authored synthetic pilot includes correct and incorrect comments,
function names, test assertions, Gateway contracts, candidate findings, and missing
context. Expected labels are excluded from model state; tune/holdout labels are
reported separately. The committed model recording can be replayed without a key:

```sh
uv run --locked python scripts/review_quality.py replay --recording tests/fixtures/quality_review/jev-pilot-recording.json --output .local/quality-review/pilot-replay.json
uv run --locked python scripts/review_quality.py evaluate --recording tests/fixtures/quality_review/jev-pilot-recording.json --output .local/quality-review/metrics.json
uv run --locked pytest -q tests/test_quality_review.py
```

To obtain fresh measurements explicitly:

```sh
uv run --locked python scripts/review_quality.py pilot --live --output .local/quality-review/new-pilot.json
uv run --locked python scripts/review_quality.py evaluate --recording .local/quality-review/new-pilot.json --output .local/quality-review/new-metrics.json
```

Metrics include precision, recall with and without abstentions/missing answers,
exact relation agreement, false suppressions, latency, and Gateway-reported cost.
A small synthetic set is not representative accuracy. Grow it with human-labeled
real findings, track warnings per PR and reviewer usefulness, repeat uncertain
cases, and evaluate held-out examples before changing thresholds. Re-run after
changes to models, questions, evidence selection, or rules. Any decision to block
on an individual rule requires separate evidence and an explicit policy change.

## GitHub Actions

`Jev advisory review` runs from the PR's trusted base, with no API key on PR events.
It produces offline plans and artifacts. On the bootstrap PR it explicitly skips
because the base does not yet contain this runner. `workflow_dispatch` can perform
a live run when a maintainer selects `live` and the repository has an
`OPENROUTER_API_KEY` Actions secret. Use a trusted workflow ref and a full target
commit SHA. Only reviewed source blobs are read from the target; its code and
configuration are not executed. The workflow uses read-only repository access,
never posts comments, and never uses `pull_request_target`.

Results are retained for seven days. Do not make this optional advisory workflow a
required branch-protection check. The standard Checks workflow installs the pinned
linter to exercise its offline integration test and audits its dependencies.
