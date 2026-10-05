# Merged PR cleanup

The repository setting `deleteBranchOnMerge` enables automatic deletion of a
PR's remote branch after merge; check it with
`gh repo view --json deleteBranchOnMerge` if remote branches start accumulating.
Enable it with `gh repo edit --delete-branch-on-merge` when needed.

For each local clone, set `git config remote.origin.prune true` once. Fetching then
removes remote-tracking refs for deleted branches. This setting does not remove
local branches or worktrees.

After a PR merges, run this from any worktree of the repository:

```sh
python3 scripts/finish_merged_pr.py <pr-number>
```

The helper verifies that the PR was merged into this repository's `main` and that
any remaining local or remote branch tip matches the PR head commit. It fetches
and prunes, fast-forwards the local `main` worktree, deletes the matching remote
branch if it predates automatic deletion, then removes a clean branch worktree
and local branch. It works with squash merges because it checks the PR's head
commit rather than Git ancestry. If `main` diverged or a branch worktree has
uncommitted or ignored files, the helper stops and reports what needs attention.
Move ignored data you want to retain, such as `.env`, local databases, and
`.local/` reports. If all ignored files in that worktree are disposable, rerun
with `--discard-ignored`. Untracked files in the `main` worktree are preserved
unless they conflict with the merge.

## Preserve dirty main while cleaning a merged branch

Synchronization and cleanup can run independently. When the primary checkout has
WIP, leave its branch, index, and files in place:

```sh
uv run --locked python scripts/finish_merged_pr.py <pr-number> --cleanup-only --receipt /path/outside/repo/cleanup.json
```

The receipt records synchronization as deferred and cleanup as complete. This
mode still requires a merged PR targeting this repository, exact matching branch
tips, and a clean owned worktree. The default invocation continues to stop on
dirty or diverged main. Ignored-data protection applies in either mode.

Before removing a worktree, the helper checks the effective shared pre-commit
hook, including `core.hooksPath`. A generated hook whose interpreter belongs to
the removed worktree is repaired using primary main's Python environment or the
helper's interpreter, after verifying that it can import `pre_commit`. Supply
`--hook-python /stable/environment/bin/python` to select a replacement. A custom
hook with a dependency in that worktree, an unavailable replacement, or a hook
file located inside that worktree stops cleanup before branch deletion.

## Pull main while retaining WIP

Use a new backup directory outside this checkout:

```sh
uv run --locked python scripts/pull_preserving_wip.py --backup /path/outside/repo/pull-backup
```

The helper snapshots modified tracked and untracked files, records index and
working changes, fetches the remote head, and rehearses restoration in a
throwaway worktree. Tracked conflicts or differing untracked files becoming
tracked block the pull before the primary branch or index changes. Identical
untracked files already present upstream are classified as already incorporated.

After a successful rehearsal, the helper retains a full WIP stash, fast-forwards
to the fetched commit, restores the staged and unstaged tracked changes, and
restores untracked files. `receipt.json` accounts for every original file as
restored, already incorporated, or preserved for reconciliation. A failure after
stashing is recorded as reconciliation required; the stash and snapshot remain
available. Ignored files are excluded from snapshots and remain in the checkout.
The helper requires the requested branch (main by default), refuses existing
conflicts and divergence, and detects checkout changes during rehearsal.
