# Prompt Enhancer agent guide

Prompt Enhancer evaluates prompts with Jev and bounded rewrites. The Gateway is
the engine's single route to model providers.

## Find the right guidance

- For domain changes, read [GLOSSARY-MAP.md](GLOSSARY-MAP.md), the relevant
  glossary, and applicable [ADRs](docs/adr/). Use their terms and surface any
  conflict with an accepted decision.
- For setup, local servers, model configuration, and API behavior, use the
  [README](README.md) and current code or configuration.
- For GitHub issue work, use `gh` and read the
  [issue tracker guide](docs/agents/issue-tracker.md) and
  [triage labels](docs/agents/triage-labels.md). Before publishing an issue,
  apply the `issue-authoring` skill.

- For Gateway recordings or request-hash audit claims, follow the
  [capture integrity procedure](docs/agents/capture-audits.md).

## Change and verify

- When creating a branch, use a short lowercase kebab-case name. Prefix Codex
  task branches with `codex/`; otherwise use a conventional type prefix. Preserve
  a branch name supplied by the user.
- Before handing off changes, follow the
  [validation workflow](docs/agents/validation.md).
- For source or test changes, branch reviews, or changes to Jev review tooling,
  follow the [semantic review procedure](docs/quality-review.md#agent-procedure).
  Jev findings are advisory; deterministic checks decide completion.
- Before pulling into a checkout with local changes, follow the
  [WIP-preserving pull procedure](docs/agents/branch-cleanup.md#pull-main-while-retaining-wip).
- For merged-PR cleanup or worktree removal, follow the
  [cleanup guide](docs/agents/branch-cleanup.md).

Keep credentials in the ignored `.env` and server-side. Make live model calls
only when the task has authorized them; never expose credentials in arguments,
logs, fixtures, or committed files.
Stop and ask before reading or writing real credentials or weakening a
deterministic check.
