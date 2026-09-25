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
