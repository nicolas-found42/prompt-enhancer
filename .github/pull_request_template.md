## Summary

<!-- Problem and why it matters; resulting behaviour for the user or caller. Use GLOSSARY.md terms. -->

Closes #<!-- issue number; delete this line if no issue. Branches named issue-<n>-… require it. -->

<!-- Smallest before/after view of each important change:
     - exact syntax or API contract: a fenced `diff` excerpt
     - behaviour or control flow: a fenced `diff` labelled **Conceptual diff**
     - wholly new feature: a usage example and its result, plus the previous limitation -->

<!-- Tradeoffs, limitations, migration steps. -->

## Evidence

<!-- One bullet per behaviour claim. Name the command, test, or scenario and record what happened.
     Before/after tests need observations from both versions. Screenshots only where appearance
     matters, and they must be hosted so reviewers can open them. -->

- **<behaviour checked>**: <command/test or scenario and conditions>
  - **Before:** <observed result>
  - **After:** <observed result>

<!-- If incomplete: what remains unverified, why, and the check still needed.
     Local `pre-commit` results do not establish remote CI; report Checks and both CodeQL
     analyses for the latest head separately. See docs/agents/validation.md. -->

## Merge Danger

**Door:** <one-way or two-way, with the reason>

**Blast Radius:** <affected users, callers, or systems and what could go wrong>

## After merge

- [ ] Remote branch is gone (automatic on merge; confirm with `git ls-remote --heads origin <branch>`).
- [ ] Local branch and worktree removed with `uv run --locked python scripts/finish_merged_pr.py <pr-number>`
      ([cleanup guide](../docs/agents/branch-cleanup.md)).
