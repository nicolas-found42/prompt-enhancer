# Prompt Enhancer

A local prompt workbench that diagnoses a request with Jev, checks proposed
success tests, tries bounded rewrites when a material gap is confirmed, and
shows the original or selected prompt with its evidence. Runs and feedback
are stored in a local SQLite database.

## Open the app

The current local instance is available at <http://127.0.0.1:5173/>. To start
it again from this repository, run the backend and frontend in separate
terminals:

```sh
uv sync
uv run --env-file .env uvicorn prompt_enhancer.api:app --host 127.0.0.1 --port 8000
```

```sh
cd web
npm install
npm run dev -- --host 127.0.0.1
```

The private `.env` file should contain `OPENCODE_GO_KEY` and
`OPENROUTER_API_KEY`. It is ignored by Git. The configured default writer is
Space Bunny Free on OpenCode Go; Jev 1.13 is the fixed judge. The app's
**Model choices** section lets you change eligible writer, strong-check, and
weak-panel models per run.

Go writer models require an active OpenCode Go subscription. Without one,
every writer call returns HTTP 403 and the run fails with a provider error.

## Check the implementation

```sh
uv run pytest -q
uv run ruff check .
uv run ty check src scripts
cd web && npm run build && npx playwright test
```

The [spec](docs/spec.md) defines the product and acceptance criteria. The
[evaluation report](docs/evaluation-results-2026-09-23.md) and
[delegated-review follow-up](docs/delegated-evaluation-2026-09-23.md) state
the measured results, exact denominators, and remaining evidence limits.
Source prompts and provider recordings stay in ignored `.local/evaluation/`.
