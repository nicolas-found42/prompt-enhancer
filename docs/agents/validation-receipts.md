# Validation receipts

Use these commands to retain local check evidence or prepare final completion
claims. Each run requires a fresh output directory. Keep reports in ignored
`.local/`; they can contain project source and command output. The tools do not
read an env file, invoke a model, publish GitHub checks, or cache test results.

## Configured checks

```sh
uv run --locked python scripts/run_validation.py --output .local/validation/final-1
```

The runner reads hook IDs from `.pre-commit-config.yaml`, runs all fast hooks
first, then build and test hooks, and stops at the first failure. Every command
uses pre-commit's existing all-file scope and verbose output. A formatter change
ends that run; inspect, stage, and rerun before expensive checks. Source changes
during a run invalidate its completion receipt.

Git subprocesses discard inherited hook repository/index selectors so `--repo`
and the chosen working directory remain authoritative.

Successful and failed output is saved per hook. Python tests also write
`pytest.xml` and report their ten slowest durations. CI uploads the receipt and
logs even after failure. The usual installed commit hook remains mandatory and
stops at its first failure as well.

Receipts record the starting commit and a digest of source contents and file
kinds, including nonignored untracked files. The evidence builder accepts a
pre-commit validation run after committing those exact contents: the digest
must match the reviewed committed snapshot. Changed source requires a new run.
Only CI's existing `SKIP=no-commit-to-branch` exception is supported; skipping
other checks cannot produce a final validation receipt.

## Local CodeQL

Install the [CodeQL bundle](https://docs.github.com/en/code-security/codeql-cli/getting-started-with-the-codeql-cli/setting-up-the-codeql-cli)
if it is not already available, then run:

```sh
uv run --locked python scripts/run_codeql.py --ref HEAD \
  --codeql /path/to/codeql --output .local/validation/codeql-1
```

Both workflow languages run by default. `--language python` or
`--language javascript-typescript` selects one explicitly. The runner extracts
`git archive` for the resolved commit into a separate source tree, so working
changes and ignored coverage/build files are absent. It uses build mode `none`
and each language's `*-code-scanning.qls` suite. The receipt records the CLI
version, source digest, suite, subprocess results, SARIF hashes, and finding
counts. A failed process, absent output, malformed SARIF, or an explicitly
unsuccessful SARIF invocation fails the run.

A completed analysis may have findings. Read the count and findings before
making a security claim. Local receipts do not satisfy or replace the server's
required Checks and CodeQL runs.

## Final review evidence

After committing the reviewed source:

```sh
uv run --locked python scripts/build_review_evidence.py \
  --base <starting-sha> --head HEAD \
  --receipt .local/validation/final-1/receipt.json \
  --receipt .local/validation/codeql-1/receipt.json \
  --output .local/validation/evidence-1.json
```

The builder verifies receipt schema, final phase, completed checks, source
identity, configured hook coverage, and log/SARIF hashes. CodeQL receipts must
also match the exact reviewed commit, even when two commits have identical trees. A failed, reproduction,
modified, incomplete, or stale receipt is rejected. Use `--phase reproduction`
on either runner to label red-test evidence; keep those receipts separate from
final completion claims.

The resulting JSON contains full base/head commit IDs, the patch in `diff`, and
the patch plus one grouped item per receipt in `evidence` (at most 16 items).
Supply both fields to an external
Jev gate so its review and verification parts receive the patch. The builder
never truncates evidence: a patch over 50,000 characters or a bundle exceeding
`--max-bytes` (200,000 by default) fails explicitly. Select only
relevant receipts or raise the limit deliberately. It checks local consistency,
not the authenticity of independently supplied receipt files.
