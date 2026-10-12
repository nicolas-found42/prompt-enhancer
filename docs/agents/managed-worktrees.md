# Managed worktree handoffs and cleanup

When handing off a Codex managed worktree, inspect `list_artifacts` and retain
the exact attachment identity, checkout path, owning chat ID/title, branch and
head. Ownership that the app does not expose is unknown, not inferred from a
similar directory name. Include these fields beside remaining delivery work:

```json
{
  "attachment_identity": "exact identityKey from list_artifacts",
  "checkout": "absolute workspace path",
  "owner_chat": {"id": "known owning chat ID", "title": "verbatim title"},
  "branch": "codex/task-branch",
  "head": "full SHA",
  "evidence_destination": "absolute path outside the worktree",
  "preservation_manifest": "artifact inventory and hashes",
  "primary_wip_proof": "inventory and content hashes",
  "archive_route": "archive_worktree from the authorized owning chat",
  "messaging_authorized": false
}
```

Preserve needed ignored artifacts before archive and verify the manifest against
the external copy. For a teardown, retain failed experiments, raw judgments,
profiles, traces, fixtures, and final receipts as well as successful summaries.
Store them outside the worktree and primary checkout; record relative paths,
file sizes and SHA-256 hashes, plus symlink targets. Preserve existing artifacts
in place and use a new destination. A PR description should name the durable
location and distinguish local evidence from reviewer-accessible CI artifacts. Use `archive_worktree` for app-managed worktrees; the app's
ownership check determines which chat can do it. If another chat owns the
attachment, prepare its bounded cleanup request after preservation. Sending
that request requires explicit user authorization. Carry that authorization
state through the handoff; an archive request does not resume a paused goal.

After the app archives the checkout, run the repository's
[merged-PR cleanup](branch-cleanup.md) to remove matching branch refs and verify
their absence. Use the WIP-preserving pull procedure for a dirty primary checkout.
