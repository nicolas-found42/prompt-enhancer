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
Space Bunny Free on OpenCode Go; Jev defaults to the pinned
`typesafe/jev-1.13-20260917` snapshot. Set `PROMPT_ENHANCER_JEV_MODEL` to use
a different Jev snapshot for new runs. The app's
**Model choices** section lets you change eligible writer, strong-check, and
weak-panel models per run.

Go writer models require an active OpenCode Go subscription. Without one,
every writer call returns HTTP 403. The app checks Go when it loads and, if Go
refuses requests, shows a banner that switches the writer and strong check to
OpenRouter models (`PROMPT_ENHANCER_FALLBACK_WRITER_MODEL` and
`PROMPT_ENHANCER_FALLBACK_STRONG_MODEL` override the choices). A run that fails
anyway shows what went wrong and what to do next.

The web app starts runs in the background (`POST /api/jobs/optimize`, then
poll `GET /api/jobs/{run_id}`), so it shows each stage and elapsed time, can
cancel a run, and reattaches after a reload. The synchronous
`POST /api/optimize` endpoint remains for API clients.

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
the measured results, exact denominators, and remaining evidence limits. The
[gap cutoff recalibration](docs/gap-cutoff-recalibration-2026-09-24.md) records
why the tool now asks more readily and what the version 2 writer measured.
Source prompts and provider recordings stay in ignored `.local/evaluation/`.
