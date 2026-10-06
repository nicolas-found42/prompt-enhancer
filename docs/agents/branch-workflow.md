# Branch and PR workflow

`main` changes only through a squash-merged PR from a feature branch. Cleanup of
the branch is part of the work, not a follow-up.

## What enforces it

| Rule | Enforced by |
| --- | --- |
| No direct pushes, force pushes, or deletion of `main` | Repository ruleset `default` (target `~DEFAULT_BRANCH`, no bypass actors, so admins are bound too) |
| Changes arrive by PR, squash merge only | Same ruleset: `pull_request` rule with `allowed_merge_methods: [squash]` |
| `checks`, `analyze (python)`, `analyze (javascript-typescript)` pass before merge | Same ruleset: `required_status_checks` |
| No local commits on `main` | `no-commit-to-branch` pre-commit hook (CI skips it; it runs on `main` after a merge) |
| Remote branch deleted on merge | Repository setting `deleteBranchOnMerge` |
| Stale remote-tracking refs pruned on fetch | `git config remote.origin.prune true` in each clone |
| Local branch and worktree removed | `scripts/finish_merged_pr.py`, run by whoever merges |

Check the server-side rules at any time:

```sh
gh api repos/nicolas-found42/prompt-enhancer/rulesets/24130379 --jq '{enforcement, bypass_actors, rules: [.rules[].type]}'
gh repo view --json deleteBranchOnMerge
```

Expect `enforcement: active`, an empty `bypass_actors`, the rules above, and
`deleteBranchOnMerge: true`. If either drifts, restore it before merging anything.

To run the full hook suite while standing on `main` (for example to validate a
dirty checkout), use `SKIP=no-commit-to-branch uv run --locked pre-commit run --all-files`.

## Finishing a change

1. Branch from current `main` with a short kebab-case name; see
   [AGENTS.md](../../AGENTS.md#change-and-verify) for prefixes.
2. Commit, push, and open a PR from the [template](../../.github/pull_request_template.md).
   The PR body follows the `pr` skill: why, a before/after view, evidence, and
   Merge Danger. Issue branches (`issue-<n>-…`) need `Closes #<n>`.
3. Merge only after the required checks are green for the PR's latest head
   ([validation workflow](validation.md)). Capture Qodo readiness on that head:

   ```sh
   uv run --locked python scripts/review_readiness.py <pr-number> --head <full-sha> --dispositions /path/dispositions.json --output .local/review-readiness/final.json
   ```

   The command waits at most five minutes and requires ten seconds of stable
   review/comment metadata. Each captured finding needs a `fixed`, `dismissed`
   or `deferred` disposition with an evidence reference. Example JSON:
   `{"4197420107": {"status": "fixed", "evidence": "test name and fix SHA"}}`.
   A stale head, untriaged finding, unavailable API or timeout is explicit.
   Keep judgments advisory: report and assess unresolved findings; a completed
   review is not an approval. If Qodo times out, record that gap in the PR before
   proceeding on deterministic checks. Refresh the snapshot immediately before
   merge, and repeat after any head change. Reviews can arrive after a timeout.
4. After the merge, finish the cleanup and prove it:

   ```sh
   uv run --locked python scripts/finish_merged_pr.py <pr-number>
   git ls-remote --heads origin <branch>   # prints nothing
   git branch --list <branch>              # prints nothing
   git worktree list                       # branch worktree absent
   ```

   The merge is not done until all three come back empty. When the primary
   checkout has WIP, use the `--cleanup-only` mode in the
   [cleanup guide](branch-cleanup.md).

   For app-managed checkouts, carry ownership and the archive route using
   [managed-worktree handoffs](managed-worktrees.md); archive through the app
   after retaining required ignored evidence, then finish Git branch cleanup.

## Issues

Open issues from the [issue forms](../../.github/ISSUE_TEMPLATE/); blank issues
are disabled in the web UI. Fields mirror the `issue-authoring` skill, so an
agent publishing with `gh issue create` fills the same sections. The forms add
`needs-triage` and a type label; follow the [triage labels](triage-labels.md)
from there.
