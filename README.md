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
python3 scripts/bootstrap.py
uv run pre-commit run --all-files
```

The web toolchain uses TypeScript 7 for `tsc` through the `@typescript/native`
npm alias. The `typescript` alias points to `@typescript/typescript6` because
`typescript-eslint` still needs the TypeScript 6 compiler API. This follows
[TypeScript's side-by-side setup](https://devblogs.microsoft.com/typescript/announcing-typescript-7-0/#running-side-by-side-with-typescript-60)
and keeps `npm ci` compatible with the linter's peer dependencies.

The bootstrap installs the locked Python, web and quality-review dependencies,
Chromium and the effective Git hook. On macOS it installs missing actionlint and
Gitleaks through an available Homebrew. `python3 scripts/bootstrap.py --check`
verifies availability without installing anything. CI retains its pinned scanner
installer. Optional training dependencies remain a separate profile.

For an isolated diagnostic run, export the saved key-free model settings first:

```sh
uv run --locked python scripts/isolated_run.py --source-settings prompt_enhancer.settings.json --output .local/isolated/example
```

Preparation makes no model calls. With authorized live inference, add `--run
--prompt 'Explain photosynthesis.'` and select the judge/operation/request/wall
limits explicitly when comparing configurations. The launcher uses a fresh
database, checks effective models and exported score floors before submission,
and retains source/settings hashes, limits, final job and history. It inherits
server-side credentials from the runtime environment and excludes credentials
from provenance. New run records retain effective model/budget configuration.
`initial_configuration` preserves the initial execution settings, `configuration`
holds the latest effective settings (including approved time/spend limits), and
`configuration_history` records each
optimization or continuation before its provider calls. Historical records
without this metadata still have unknown historical configuration.

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

Runtime grading calibration can be loaded with
`PROMPT_ENHANCER_CALIBRATION=/absolute/path/to/calibration-artifact.json`.
The app uses that artifact's policy version and still requires an exact match
for the question identity and answering Jev snapshot. Inconclusive grading
remains unresolved when verification calibration is absent, incomplete or
incompatible. Generate artifacts from labeled observations with the
[offline calibration workflow](docs/evaluation-data.md#offline-per-question-jev-calibration);
synthetic test policies are not production calibration.

Current Round protocol 15 combines probability across permissible sentence
support reasons, judges style on the resulting prompt, and omits grades when
there are no success tests. Older protocols retain historical request hashes
and report contracts for replay. Weak-panel generations have a default
16,384-token output cap, configurable through
`PROMPT_ENHANCER_WEAK_MAX_OUTPUT_TOKENS`. MiMo weak-panel calls disable deep
thinking; writer and strong-check calls keep their provider behavior. Muse's
direct Gateway default reserves 8,192 tokens. Empty and explicitly incomplete
weak replies stop operationally with allowlisted finish/token metadata rather
than contributing a prompt-quality score. Operation deadlines remain separate
from output caps and Round-boundary run limits.

For the provisional #187 evaluation contract, submit
`options={"evaluation_profile": "llama-tuning-v1"}` to the optimizer or job API.
It pins Jev as judge, `qwen/qwen3.7-flash` for non-Jev roles, and
`meta-llama/llama-3.1-8b-instruct` through OpenRouter for three weak samples.
Novita is the provisional first service; `"evaluation_provider": "groq"`
selects Groq first for the matched pilot. Requests restrict routing to that
service and require parameter support. A permitted service fallback reruns the
baseline and every draft on the other service with matching requested seeds;
incomplete attempts and completed sibling samples remain in the evidence.
Reported provider/model identities remain unknown when the response omits them,
and matching requests do not establish effective sampling settings. The
4,096-token weak output cap is a pilot starting point. Neither service has been
chosen by a measured pilot, and this profile does not establish the release bar.
An injected Gateway whose actual judge differs from the pinned Jev model is
rejected before profile calls, so configuration evidence cannot misidentify it.

For request profiling, construct `HttpGateway(profile_requests=True)` and read
`profiling_report()` after a run. Its private sidecars retain each adapter attempt,
whether it dispatched, role, requested and reported identities, canonical JSON size,
output cap, requested sampling/reasoning controls, available usage, and monotonic
start/dispatch/finish times. Queue and transport durations are distinct. A new
logical run resets these sidecars; retain the report before starting another run.
Raw answers and adapter usage accounting keep their existing contracts. Request
and answer content, credentials and arbitrary provider diagnostics are omitted
from the sidecars. This nonstreaming path cannot measure headers, first byte,
first visible token or a visible generation interval: TTFT and output TPS remain
unknown. Requested controls do not establish their supported or effective values.

For opt-in streaming measurements, construct
`HttpGateway(profile_requests=True, stream_chat_for_profiling=True)`. An injected
HTTP transport must also enable `HttpTransport(profile_streams=True)`. This path
requests streamed chat answers and retains the received SSE bytes (including a
base64 copy), parsed frames, framing errors and completion marker in a raw answer
with protocol `raw-chat-sse-1`. Answer interpretation remains with callers;
incomplete, malformed or error streams cannot supply a usable completion.
Visible truncated weak-model text is retained only as failure evidence.
The sidecars distinguish headers, first bytes and the first nonempty visible
content frame; metadata, empty content and reasoning frames do not start TTFT.
Observed frame-arrival timing is separate from client rendering time. Output TPS
uses reported completion tokens minus reported reasoning tokens over the
first-to-last visible-content frame interval. It remains unknown without both
counts and a positive interval, or when the stream is incomplete or malformed.
Frames in one transport read share the receipt timestamp; parsing work cannot
create a generation interval between them.
Conflicting reported identities stay unknown and invalidate a matched comparison.
Raw stream answers and accounting frames stay available for private capture;
sidecars omit their contents. The stream reader stops at the completion marker
and rejects responses exceeding its 16 MiB retention limit. These controls do not
establish a measured provider pilot or normal completion targets. The framing and
accounting behavior follows the
[OpenRouter streaming documentation](https://openrouter.ai/docs/api_reference/streaming)
and [SSE parsing rules](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream).

Native requirement evidence checks compound JSON declarations with explicitly required
keys and types, nested object type maps, and array element types. Unless the source
says `exactly these keys`, additional keys remain permitted. Values and array lengths
remain unconstrained. Duplicate applicable keys fail these bindings; duplicate
undeclared keys, byte-order marks and unsupported schemas retain uncertainty. The
historical whole-answer grammar keeps its original duplicate-key convention.
Compound CSV declarations check their explicit header, record width and optional
`exactly N data rows` separately from the header. Comma-, semicolon- and tab-delimited
forms honor quoted fields; unsupported dialects and ambiguous blank-record
conventions remain untestable. Rewritten declarations receive separate binding
checks so compliant sampled answers cannot hide deleted or contradictory
declarations. Nested duplicate source keys remain ambiguous, and descriptions
such as `Explain how to return JSON` do not declare a JSON answer format.

Audited semantic obligations receive separate Jev judgments for the rewritten
prompt and each model/sample answer. Raw typed decisions, source spans, Round and
candidate identities stay in the report. A confident failure or unresolved judgment
cannot qualify a draft. Semantic passes never erase deterministic failures. This
records judgment uncertainty; it does not establish semantic or sentence calibration.
Rejected proposed success tests retain their source associations and screening
reasons. Only tests with a supported source binding enter grading; binding checks
use bounded batches and retain incomplete judgments. The engine may generate one replacement batch in a Round and screen it
again within the remaining active deadline. The original obligation remains active
when proposals are rejected or no usable binding is found. Losing drafts pass their
own source failure IDs, offending evidence and text to the next Round's writer;
repairs receive fresh panel and requirement checks before qualification.

The live workbench and history show obligation sources, scopes, protected values,
audit status, coverage gaps and draft findings with expandable Round chronology.
Run the controlled public-flow checks with
`uv run --locked python scripts/check_requirement_contract.py --output .local/requirement-controls/<new-name> --timeout 600`.
The command retains a receipt, source and test digests, logs, test results and SQLite
histories, and fails on timeout, skipped controls or a changed source snapshot.
These controlled checks do not constitute a live provider evaluation or completion
of the parent issue #187.

An explicit
`Do not change any character in the supplied code:` line (using `data` for data)
followed by a top-level fenced block protects its literal
contents in a rewrite. The checker permits fence presentation changes that
preserve those contents; changes or appended content in the protected block
remain known failures. Tabs with ambiguous indentation and content moved outside
the supported block scope retain uncertainty. The conservative lexer reads
fenced content as source data, including exact-reply text quoted inside it;
delegated instructions and other source scopes still need semantic coverage.
It follows the literal-content and closing-fence rules of
[CommonMark 0.31.2](https://spec.commonmark.org/0.31.2/#fenced-code-blocks).
Recognized constraints keep their exact source spans; the ledger still reports
partial coverage pending broader audited
extraction.

Native diagnosis retains the original questions, raw partial answers and errors.
Oversized single questions use contiguous source windows, with at most sixteen
windows per question and eight physical requests across bounded recovery.
Held windows preserve their source, and window answers cannot establish a
whole-prompt diagnosis. Malformed required answers make diagnosis explicitly
incomplete; unused speculative answers remain separate from required evidence.

The [spec](docs/spec.md) defines the product and acceptance criteria. The
[evaluation report](docs/evaluation-results-2026-09-23.md) and
[delegated-review follow-up](docs/delegated-evaluation-2026-09-23.md) state
the measured results, exact denominators, and remaining evidence limits. The
[gap cutoff recalibration](docs/gap-cutoff-recalibration-2026-09-24.md) records
why the tool now asks more readily and what the version 2 writer measured.
Source prompts and provider recordings stay in ignored `.local/evaluation/`.
