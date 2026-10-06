# Prompt Enhancer

A local prompt workbench that diagnoses a request with Jev, resolves the
requested improvement style, and evaluates up to six candidates per bounded
round. An early routing or writing decision can end a round before panel
requests; generated candidates are tested on five distinct models with three
samples per model, then fidelity, score-floor, and acceptance checks decide
which qualify. The Perfect Prompt Loop repeats rounds until its score evidence meets the quality
floors and stops gaining, or a separate run-control or provider condition
interrupts it. When those details are available, a result names its applied
style and gives a brief explanation grounded in the run evidence. The five
outcomes are converged,
improved (tested), improved (unverified), impossible, and failed (operational).
Pauses, cancellations, and a user's decision to stop are reported as control
states, separate from the outcome. Runs and feedback are stored in a local
SQLite database.

Recorded keep/reject labels can explicitly recalibrate the per-dimension
quality floors. Recalibration requires 12 linked labels (at least three of each
decision); see [quality floor feedback calibration](docs/quality-floor-feedback.md)
for the rule and `POST /api/quality/floors/recalibrate` endpoint.

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
`POST /api/optimize` endpoint remains for API clients. It blocks until a result
or budget pause and has no job cancellation handle; clients needing cancellation
should use the job endpoints. Synchronous callers can set `time_limit_s` or
`spend_limit_usd` explicitly to pause at a completed-round boundary. Limits are
optional, and a healthy run has no automatic attempt or stagnation cap.

Each Gateway operation has a separate 180-second deadline by default. Set
`PROMPT_ENHANCER_OPERATION_TIMEOUT` to change this limit in seconds. The
deadline covers model-catalog routing, the provider request, retries, and retry
delays. `PROMPT_ENHANCER_TIMEOUT` remains the per-request socket timeout; it
does not replace the operation deadline. The `time_limit_s` run option still
pauses only after a completed Round when another Round would start.

If a transport ignores its timeout and cannot abort the active request, the
Gateway returns a timeout or cancellation result by the operation deadline and
keeps the abandoned worker bounded to eight per Gateway. The built-in HTTP
transport can close a response body after it receives a response handle. The
standard library does not expose that handle while it waits for response
headers, so an abandoned worker in that phase can remain until the connection
ends or the server process stops.

## Writer reply recovery

New runs use writer instruction version 14. Success-test generation and candidate
writing each allow one additional writer call when the reply has no usable text,
unreadable JSON, or an unusable required shape. Two unusable replies end the run
as failed (operational). Transport failures retain the Gateway's existing retry
policy; the caller does not add another transport retry.

A malformed success-test item is retained in `report.test_screening.rejected`
with its raw value, original position, and rejection reason. Valid siblings keep
their position-based IDs and go through normal screening. If all items are
rejected, the Round proceeds with no usable success tests under the existing
unverified-evidence policy. Malformed items do not trigger a whole-reply retry.
The optional Choice-description repair keeps its existing fallback behavior.

`report.writer_attempts` records the operation, Round, attempt number, model,
outcome, and failure reason when applicable. Completed Round history retains its
own attempts; failed runs retain attempts from the unfinished Round as well.
Continuing a paused run retains its saved attempts, including unfinished Rounds.
Both answered requests contribute to the Gateway's usage and cost accounting,
including billed empty replies. Recordings without usage remain without measured
per-call costs.

Only the second request adds `writer_reply_retry` metadata to its state, giving
it a distinct request hash so both answers fit the existing recording format.
Writer versions 1–13 retain their previous request sequence and item handling.
Reading and validation stay in callers; the Gateway continues returning raw
answers as specified by [ADR-0001](docs/adr/0001-gateway-returns-raw-answers.md).

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

Optional [Jev semantic review](docs/quality-review.md) checks changed code against
comments, names, test claims, and explicit repository contracts. It runs separately
from commit hooks, defaults to offline planning, and reports advisory findings.

The [spec](docs/spec.md) defines the product and acceptance criteria. The
[evaluation report](docs/evaluation-results-2026-09-23.md) and
[delegated-review follow-up](docs/delegated-evaluation-2026-09-23.md) state
the measured results, exact denominators, and remaining evidence limits. The
[gap cutoff recalibration](docs/gap-cutoff-recalibration-2026-09-24.md) records
why the tool now asks more readily and what the version 2 writer measured.
Source prompts and provider recordings stay in ignored `.local/evaluation/`.
