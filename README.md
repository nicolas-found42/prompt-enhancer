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
uv sync --locked
uv run --env-file .env uvicorn prompt_enhancer.api:app --host 127.0.0.1 --port 8000
```

```sh
cd web
npm ci
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

The project pins Python 3.12 in `.python-version` and recommends Node 24 in
`web/.nvmrc`. Install both toolchains, then set up the locked dependencies and
Git hook. The hook also uses `actionlint` and `gitleaks` from your PATH
(`brew install actionlint gitleaks` on macOS):

```sh
uv sync --locked
npm --prefix web ci
uv run pre-commit install
uv run pre-commit run --all-files
```

The web toolchain uses TypeScript 7 for `tsc` through the `@typescript/native`
npm alias. The `typescript` alias points to `@typescript/typescript6` because
`typescript-eslint` still needs the TypeScript 6 compiler API. This follows
[TypeScript's side-by-side setup](https://devblogs.microsoft.com/typescript/announcing-typescript-7-0/#running-side-by-side-with-typescript-60)
and keeps `npm ci` compatible with the linter's peer dependencies.

The commit hook checks file hygiene, GitHub Actions syntax, staged secrets,
and `uv.lock`; lints and formats staged Python and web files; then checks
Python types and dependencies, builds the web app, and runs pytest with an
80% coverage floor, Vitest unit tests, and the Playwright suite (including
automated accessibility checks). Install Playwright's browser
once with `npm --prefix web exec -- playwright install chromium` if it is missing.
Run checks individually with:

```sh
uv lock --check
uv run --locked ruff format --check src scripts tests
uv run --locked ruff check .
uv run --locked ty check src scripts
uv run --locked deptry src
uv run --locked pytest -q --cov=prompt_enhancer --cov-report=term
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run build
npm --prefix web run test:unit
npm --prefix web run test:e2e
actionlint .github/workflows/*.yml
gitleaks git . --redact --no-banner
```

The CatBoost model test requires the optional training dependencies; run
`uv sync --locked --extra training` and `uv run --locked --extra training pytest -q`
when working on that path. Pull requests run the same pre-commit checks in CI.
CI also audits the locked Python and npm dependencies, checks workflow
security with zizmor, and scans Git history with Gitleaks. CodeQL analyzes
Python and TypeScript on pull requests and weekly; Dependabot proposes weekly
uv, npm, pre-commit, and Actions updates. GitHub secret scanning and push
protection are enabled for this repository. Accessibility automation covers
common detectable issues; keyboard and task-flow review still requires a
person.

The [spec](docs/spec.md) defines the product and acceptance criteria. The
[evaluation report](docs/evaluation-results-2026-09-23.md) and
[delegated-review follow-up](docs/delegated-evaluation-2026-09-23.md) state
the measured results, exact denominators, and remaining evidence limits. The
[gap cutoff recalibration](docs/gap-cutoff-recalibration-2026-09-24.md) records
why the tool now asks more readily and what the version 2 writer measured.
Source prompts and provider recordings stay in ignored `.local/evaluation/`.
