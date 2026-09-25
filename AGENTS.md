## Agent skills

### Branches

When creating, naming, or checking a Git branch, use the `conventional-branch` skill.

### Issue tracker

Issues live in GitHub Issues for `nicolas-found42/prompt-enhancer`; use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five canonical labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This is a multi-context repository: start with `CONTEXT-MAP.md`, then see `docs/agents/domain.md`.

### Code quality and review

Before handing off changes or declaring a PR ready, follow [docs/agents/validation.md](docs/agents/validation.md) for pre-commit installation, local checks, CI-only checks, CodeQL, and the validation report.

When changing source code or tests, reviewing a branch, or modifying Jev review tooling, also follow the [Jev agent procedure](docs/quality-review.md#agent-procedure) for comparison bases, offline/live review, and finding triage. Semantic findings remain advisory alongside the deterministic checks.

### Merged PR cleanup

After a PR merges, follow [docs/agents/branch-cleanup.md](docs/agents/branch-cleanup.md) to sync local `main` and prune its branch and worktree.
