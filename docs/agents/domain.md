# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT-MAP.md`** at the repo root: it points to the context glossaries under `docs/contexts/`. Read each glossary relevant to the topic.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in. Also check for context-scoped ADRs beside the relevant glossary.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

This repository uses a context map because prompt improvement and model access have distinct vocabularies:

```
/
├── CONTEXT-MAP.md
└── docs/
    ├── contexts/
    │   ├── prompt-improvement/CONTEXT.md
    │   └── model-access/CONTEXT.md
    └── adr/
        └── 0001-gateway-returns-raw-answers.md
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in the relevant context's `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
