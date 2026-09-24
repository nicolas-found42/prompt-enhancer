# Validation workflow

Use this workflow before handing off changes or declaring a PR ready. The linked
configuration files own the commands, filters, and thresholds; read their current
versions when deciding which checks apply.

## Local checks and commit hooks

1. Follow the [README setup instructions](../../README.md#check-the-implementation)
   for Python, Node, dependencies, Playwright, actionlint, and Gitleaks. On branches
   containing Jev tooling, install its pinned dependencies with
   `npm ci --prefix tools/quality --ignore-scripts` so its offline integration test
   runs rather than skips.
2. Verify that this checkout has an executable pre-commit hook, accounting for
   `core.hooksPath` if configured. Install the repository hook when missing:
   `uv run --locked pre-commit install`.
3. Validate the final implementation with
   `uv run --locked pre-commit run --all-files`. A successful commit-hook run covers
   the staged-file checks and its always-run project checks; check any remaining
   unstaged implementation separately. Reuse successful results for unchanged
   scope instead of repeating the same suite. When a fixer edits files, inspect
   the edits, stage the intended changes, and rerun affected checks.
4. For documentation-only work, verify content and links; the installed commit
   hook still runs its configured checks. Validation does not itself authorize
   creating a commit when the task is meant to leave changes uncommitted.

The [pre-commit configuration](../../.pre-commit-config.yaml) currently covers:

| Area | Checks |
| --- | --- |
| File hygiene | Conflict markers, large files, filename case conflicts, YAML/TOML/JSON validity |
| Workflows and secrets | Actionlint on workflow files; Gitleaks on staged changes |
| Dependency consistency | uv lock consistency; deptry on packaged Python source |
| Python | Staged Ruff fixes/formatting, whole-repository Ruff lint, ty on source/scripts |
| Web | Staged Prettier/ESLint fixes; TypeScript checking and Vite build |
| Tests | pytest with the configured coverage floor, Vitest with coverage collection, Playwright including accessibility assertions |

The build and test hooks currently use `always_run: true`, including for
documentation commits. Jev's offline regression tests are part of pytest; live
model inference is separate from the commit hook.

## Additional CI checks

The [Checks workflow](../../.github/workflows/checks.yml) repeats the hook suite and
also runs:

- The optional CatBoost training-path tests.
- Python dependency vulnerability auditing with uv.
- npm vulnerability audits for the web app and semantic review tooling.
- GitHub Actions security analysis with zizmor.
- Gitleaks across Git history, beyond the staged-only hook.

Use the commands in that workflow when reproducing a failure locally. Run the
optional training tests after the standard checks; restore the standard
environment with `uv sync --locked` before repeating ty, since installing the
optional dependency changes whether its unresolved-import suppression is needed.

The separate [CodeQL workflow](../../.github/workflows/codeql.yml) analyzes Python
and JavaScript/TypeScript. Checks and CodeQL run on PRs and pushes to `main`;
CodeQL also runs weekly. A feature-branch push alone does not trigger them.

When a PR exists, inspect `gh pr checks <number>` and the workflow runs for its
latest head. Declare merge readiness only after Checks and both CodeQL analyses
have completed successfully for that head. Investigate failures; report pending,
skipped, unavailable, or untriggered checks explicitly. A successful local hook
run establishes local validation, not remote CI completion.

## Semantic review and handoff

For source/test changes or branch reviews, follow the
[Jev agent procedure](../quality-review.md#agent-procedure). Keep its reviewed
commit, scope, inference mode, and incomplete results explicit. Its optional
advisory workflow is separate from the deterministic completion criteria above.

In the final handoff, report local checks performed and their results, remote CI
status for the relevant commit, Jev review status when applicable, and any
remaining validation gaps. Include enough evidence to distinguish a passed check
from a check that was not run.
