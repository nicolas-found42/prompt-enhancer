# Adversarial teardown — 2026-10-12

## Scope, identity, and completion boundary

Target: Prompt Enhancer 0.1.0, starting commit
`5ee37e6a54fc6f1734b6c422e045984f3e1b4cdd`. Changes are local on
`codex/adversarial-teardown`, in the attached isolated worktree. The primary
checkout's modification to `docs/agents/branch-cleanup.md` and nine untracked
research notes were preserved. No publication, deployment, PR, or message to
another person was performed.

This is an executed local teardown with real Python/FastAPI/SQLite consumers and
Chromium browser workflows, using the existing ScriptedGateway at the model
boundary. It establishes application behavior, not real provider quality or
human adoption. The persona is an API integrator who wants to submit a bounded
job, inspect a truthful result, recover from mistakes, and find saved work.
The browser walkthrough additionally follows a local user resolving conflicting
requirements. These are simulated users, not participant research.

The initial budget was 60 minutes, with half reserved for implementation and
verification. It expanded to 120 minutes after full validation exposed a separate
durable-edit race, a test barrier problem, and regressions in the first attempted
deadline fix. Investigation also expanded to
cover history truncation, large-report serialization cost, and intermittent browser failures. We stopped
unchanged retries; reruns used an isolated port, a narrowed scenario, captured
network evidence, a changed implementation, or removed competing workloads.

The user prohibited live optimizer/provider cases, then explicitly requested
configured development TypeSafe/Jev judgments again for the speed investigation.
Development Jev tools resumed; no optimizer live case was submitted, no real
credential was read, and no existing live session was used. Final review/gate
artifacts are recorded separately from deterministic execution evidence.
Completion depends on their current terminal outcomes, not this narrative.

## Requirements and coverage map

Criteria were established before editing. R5 was expanded when the 200-record
storage cap was discovered; the original history/recovery criterion is retained.

| ID | Observable acceptance criterion | Coverage and evidence |
| --- | --- | --- |
| R1 | Locked setup and first local use work with an isolated database | Tested: E01 bootstrap; E02 API consumer; E08 browser fixtures. Actual paid-provider setup/use excluded by final user constraint. |
| R2 | Invalid requests return usable 422 errors before jobs, persistence, or Gateway work | Tested: F1–F3, F9; E02, E03; valid merges, numeric limits, genuine booleans, invalid JSON numeric constants, and continuation types included. |
| R3 | Jobs can complete, pause, cancel, resume, survive reload, and report control states separately | Tested with fixtures: baseline targeted 58 tests and browser suite; complete final hook logs in E12. Live transport/cancellation effectiveness remains untested. |
| R4 | Required conflict choices preserve provenance and produce a readable copyable prompt | Tested: F6, E07, browser walkthrough; source variables and superseded obligations checked. Real writer quality untested. |
| R5 | History finds relevant old records, preserves nested evidence, and paginates without gaps/duplicates | Tested: F5, E04–E06; 260-record restarted SQLite corpus, exact reference set, metadata filters, empty search, legacy store fallback. Large databases beyond this workload remain untested. |
| R6 | JSON/CSV evidence respects explicit constraints and marks unsupported conventions uncertain | Tested: E09 independent 16-case holdout and existing requirement tests. Broader semantic completeness is not inferred from native format success. |
| R7 | Required deterministic checks pass and final patch/claims receive fresh Jev review/gate | Deterministic results: E12. Partial: J14 adverse intermediate review retained; fresh final judgment recorded in E17. Committed-snapshot semantic CLI does not cover this uncommitted patch. |
| R9 | Progress preserves evidence, ownership, events, restart recovery without full evidence serialization per event; reload preserves this tab’s saved result | Tested: F12/F13, E15/E16; focused consumers and browser regressions. Full final checks: E12. |
| R8 | No new credentialed/paid execution dependency; privacy and compatibility boundaries remain explicit | Tested/inspected: initial teardown had no dependency changes; subsequent validation acceleration adds verified MIT development-only xdist/execnet, synthetic evidence only, E10 audits, legacy history fallback and unchanged Gateway raw-answer contract. No cross-platform, screen-reader, or penetration-test claim. |

Accessibility coverage is existing axe assertions and browser interaction, plus
inspection of the history tooltip and compound-requirement screenshots. A full
page screenshot does not prove viewport fit. Screen-reader, mobile, and actual
participant testing were not performed. Existing real app/run/browser sessions
were not used for the experiments; the created disposable browser tab and
fixture servers were closed.

## Findings and dispositions

Each finding has a separate type, impact, evidence strength, and disposition.

| ID | Problem, reproduction, and expected basis | Type / impact / evidence | Counterevidence and discriminating probe | Action and acceptance / disposition |
| --- | --- | --- | --- | --- |
| F1 | `options.model_overrides=5` yields sync HTTP 500 versus job HTTP 422; adding a top-level writer override produces 500 on both. Expected: existing request-validation contract rejects malformed selections before work. E02. | Confirmed defect / recoverable client failure / reproduced HTTP execution, zero calls and runs on these crashes | Async validation already handles the scalar alone; merge happens before its exception boundary. Compare identical payloads and empty valid objects. | Validate mapping before merge and consistently raise ValueError in optimizer, including false/null/empty arrays. E03 regressions require 422 and no side effects. Fixed locally. |
| F2 | Top-level `time_limit_s=true` or `"0"` is coerced; the nested bool is rejected. Accepted cases made 24 fixture calls and saved a run. Expected: RunControl explicitly rejects booleans and non-numbers. E02. | Confirmed defect / unintended work and wrong budget meaning / observed execution | Pydantic coercion is normally useful; here it bypasses the existing stricter engine contract. Valid integer zero and fractional limits remain controls. | Strict numeric fields on optimize and continue requests. Negative and nonfinite values retain engine rejection. E03. Fixed locally; formerly coerced inputs intentionally become 422. |
| F3 | On `Help me with this.`, boolean false completes without questions, but string `"false"` yields `needs_input`, like boolean true. Expected: caller controls clarification with a boolean. E07/E03. | Confirmed defect / unwanted clarification / paired actual engine runs with scripted models | Initial malformed-answer probe established acceptance only, not a displayed question. The paired valid diagnosis fixture separated truthiness from harness failure. | Strict top-level boolean; validate nested flag before model work. Genuine false retained. E03. Fixed locally. |
| F4 | Unknown settings aliases return 200 unchanged. An initial broad suspicion that supported settings updates were broken was unsupported. | Rejected broad hypothesis / no established impact / alias probe plus contrary supported-field tests | Supported `writer_model` update persists across restart; aliases such as `writer` are not that interface. | Retain supported settings behavior. No speculative schema migration. Rejected by counterevidence; unknown-field ergonomics remain a possible future preference. |
| F5 | Store has 205 records; direct oldest lookup succeeds, but search finds none, unlimited history returns 200, and offset 200 is empty. Expected: documented history search searches saved work. E04. | Confirmed defect / inaccessible old search results / known corpus and independent direct lookup | This is truncation rather than lost storage. Native complete reader exists. Repeat after restart and compare exact IDs including old metadata matches. | Use native `all_runs` complete snapshot; preserve bounded storage listing and legacy three-method store behavior. E05/E06: complete 260 records, exact filtered matches and pages. Fixed locally. |
| F6 | Resolved scoped conflict returns a copyable prompt containing `choice:requirement:…` despite a human question label. E07 and disposable browser walkthrough. | Demonstrated friction / minor readable-output burden / actual rendered result and failing regression | The superseded clause **is removed** in the direct current-source reproduction; opaque IDs do not prove real model failure. Prior downscaled screenshot interpretation was insufficient for that broader claim. | Recognize `choice:` among labelled clarification keys. Output uses `Resolve conflicting requirements: two bullets`; original source, selected ID, variable, and superseded evidence remain stored. E07 regression. Fixed locally. |
| F7 | History deep-copies a multi-megabyte result/report before emitting a 1,365-byte summary. E06. | Demonstrated performance friction / workload-dependent latency and allocation burden / same-record comparison | Native full snapshot initially already removes extra per-record reads; small bare corpus timing was effectively unchanged. Need actual large evidence, not extrapolation from row count. | Defer deep copies only in transient snapshot normalization; emitted summaries remain JSON-normalized, public details keep independent values, and search still examines nested evidence. Exact output comparison plus E06 timings. Improved locally. |
| F8 | Intermittent browser failures: stale result/history mismatch, pending history after reload, and slow details. E08 traces preserve successful bodies and unfinished requests. | Unresolved attribution / intermittent local test obstruction / failures reproduced under competing workloads | A narrowed lifecycle test and an entire suite run alone pass. A failing trace has 5.77 MB detail response at 4.74 s with two reads pending at the 5 s assertion. Resource contention is plausible, not proven to explain every failure. | Retain all failed logs/traces. No assertions, timeouts, or worker configuration weakened. A later focused probe demonstrated a separate reload selection defect (F13). Do not attribute all earlier failures to resource contention. |
| F9 | Whitespace-only writer/strong IDs pass run override validation though saved defaults require nonblank IDs. | Confirmed input defect / avoidable invalid model dispatch / failing HTTP regressions | Nonempty opaque model IDs remain allowed; catalog eligibility and live routing are separate. | Reject blank-after-strip model IDs (including weak entries), preserving nonblank IDs. E03. Fixed locally. |
| F10 | The full suite's assumption edit returns its new prompt, but storage retains the old prompt; an isolated replay reproduces it. A deliberately delayed deadline worker overwrites a later durable edit during its final persistence. | Confirmed persistence defect / lost user correction / existing failure plus deterministic interleaving regression | Deadline publication must retain the canonical cost/timing and newly audited source evidence. Terminal job status alone does not prove the worker has finished. An initial attempted publish-once fix passed narrow tests but failed four broader controls. | Preserve the existing deadline publication path except when storage contains an explicit later `edited` report. A typed check handles malformed prior reports. E14: 56 focused passes cover the original failures, worker interleaving, all four deadline-evidence regressions, and malformed reports. Fixed locally, subject to final whole-snapshot checks. |
| F11 | The continuation concurrency test starts a writer waiting five seconds before continuation performs unrelated preparation, so traced/slower runs time out before reaching the intended read/save barrier. | Confirmed harness attribution / unreliable validation signal / full failure, isolated pass, and source inspection | The transaction property remains essential. Extending the timeout would not isolate that property. | Start the writer at the intercepted read boundary, retaining the same five-second waits, lock ownership probe, exact configuration-history assertion, and final thread/error assertions. E13. Harness corrected; no product timing promise inferred. |

| F12 | A two-run diagnosis test emits 278 progress callbacks, each reading/writing the complete result/evidence; profile counts 300 `update_run`, 304 row decodes, 302 `save_run`. | Demonstrated performance friction / expensive repeated serialization / actual offline profile | Threaded cumulative times overlap. No claim that all suite time comes from this path. Same test and exact assertions are the control. | Add nullable SQLite `job_json`, update only running progress atomically under existing locks/publication owner. Initial, final, recovery and legacy fallback retain full writes. E15: large evidence/restart, old-column migration, missing rows, stale ownership and all original concurrency assertions. Verified in focused tests; final E12. |
| F13 | Edited result reload displays another active run. A focused browser case with saved `my-result` plus running `other-run` reproduces missing own result. | Confirmed selection defect / replaces user’s current saved work / actual Chromium failure and source trace | New tabs must still discover active runs; remembered active runs still win. Existing tests cover both. | Prefer this tab’s recent saved result over unassociated active jobs. No inference needed for exact run IDs. E16: red regression then full browser checks. |

## Experiments, research, and decisions

X01 compares the same malformed bodies on both optimize interfaces and counts
Gateway calls, persisted runs, and active jobs. E02 is before; E03 is after.
The new request-boundary suite first produced **38 failures / 3 passes**. Valid
merges, original option immutability, and real bool/numeric values are controls.
The original external probe subsequently returns 422 with zero calls/runs for
all invalid optimization cases. Negative controls already rejected invalid
seed/negative limits and keep that behavior.

X02 uses monotonically dated 260-record SQLite data, closes/reopens storage, and
compares exact IDs. Sixteen records satisfy the known metadata predicate.
Before: 200/260 listed, 12 true positives, four false negatives, zero false
positives; precision 100%, recall 75%. After: 260/260, 16 true positives, zero
false positives/negatives or duplicate IDs; precision and recall 100% on this
corpus. The original three coverage tests fail before and pass after.

X03 applies the same baseline history module (read from the fixed Git revision)
and final implementation to the **same** fixture-generated 3.79 MB stored record.
Five before/after measurements preserve identical emitted rows. Median listing
time was **700.4 ms before / 60.8 ms after**. Individual samples and payload sizes
are E06; competing local workloads and a small sample limit generalization.
The simpler 260-row corpus's initial five-sample comparison was 9.23/9.07 ms,
which did not establish a meaningful speed gain. Full native search still
materializes history and scales in memory with stored data; no indexed or
production-scale performance claim is made.

X04 resolves a scoped two-versus-three-bullet conflict via the real optimizer
and fixture browser. The label regression fails before, then passes with the
human label while ledger IDs and unchanged source survive. This does not measure
real writer quality. X05 holds out 16 JSON/CSV cases (large exponent, boolean
versus integer, duplicate keys, BOM, quoted comma, width, blank record, malformed
quote, prose/fence wrapper); all match their explicit tested/failed/untestable
expectations. Supported native conservative choices were retained.

X06 re-attacks durable editing after the full validation failure. The first new
test incorrectly checked storage immediately after `close()` (which does not
wait for workers), and passed before the fix. It was rejected as an inadequate
witness. A sentinel in the single-worker queue then waits for the actual final
persistence: this test fails before and passes after F10. The original assumption
test also reproduces without coverage and passes after the fix. F11's test starts
its competing write at the actual read/save boundary rather than assuming all
prior continuation work finishes inside five seconds. No assertion was removed
and no timeout increased.

The first attempted F10 fix limited deadline replacement to initial publication.
The next full suite exposed four regressions: canonical deadline cost/timing and
source audit evidence were not durably retained. That approach was rejected.
The final patch keeps canonical deadline publication and excludes only a later
explicit assumption edit; malformed prior reports do not crash that check.
All four newly failing scenarios and the original lost-edit scenario pass in the
56-test focused follow-up. These results do not substitute for the final full
validation receipt.

D01/D02 retain the existing Gateway and use internal validation/history changes.
[FastAPI request models](https://fastapi.tiangolo.com/tutorial/body-nested-models/)
and [Pydantic strict fields](https://docs.pydantic.dev/latest/concepts/strict_mode/)
support the boundary approach. Installed Pydantic is 2.13.5; docs were inspected
on 2026-10-12 and behavior was tested against the locked environment. General
coercion is useful, but conflicts with this engine's existing limit semantics.
[Python JSON conventions](https://docs.python.org/3.11/library/json.html)
motivated raw NaN/Infinity probes; the native format reader already conservatively
rejects non-JSON constants. No reader replacement was justified.

D03 considered retaining behavior, broad TypeError-catching, a new schema
framework, and precise existing-library validation. Broad catches conceal code
faults and leave truthiness/coercion unchanged. D04 compared full native snapshot,
removing listing bounds, and a new SQL search subsystem; existing `all_runs`
provides completeness without changing the bounded storage contract. D05
eliminates unnecessary copies instead of extending browser assertions.

Alternative discovery included [Promptfoo's local Ollama
provider](https://www.promptfoo.dev/docs/providers/ollama/) and its
[red-team configuration source](https://github.com/promptfoo/promptfoo/blob/main/site/docs/red-team/configuration.md).
A local provider design is relevant to credentialless evaluation, but no measured
benefit, exact Jev Decisions compatibility, transitive accountless execution,
model license/resource requirements, or clean-home installation was established
for a proposed integration. It was **not added**. Reconsider only after an
isolated credentialless install/operation, compatible model licensing, resource
measurements, and same-workload benefit. Community reports were discovery signals,
not corroboration of fitness. No source-category quota or popularity argument
was used.

**Addition eligibility:** no new runtime/build/dev/test dependency, service, tool
requirement, or manifest/lockfile change. Bootstrap installs existing locked
dependencies; configured audit/scanner tools remain baseline checks. Paid/keyed
Go/OpenRouter/Jev remain existing constraints explicitly documented in README;
local storage does not imply local inference. No new paid access or service was
purchased and no new backend was introduced.

## Evidence ledger and reproducibility

Detailed evidence stays in ignored `.local/teardown`; it contains synthetic
prompts/records only. No credential or private prompt was sent to research or
judgment providers. Earlier Jev source excerpts were limited to the changed
public project code and synthetic logs. Reports must preserve these artifacts
before archiving the managed worktree.

| Evidence ID | Artifact under `.local/teardown/` | Meaning |
| --- | --- | --- |
| E01 | `evidence/bootstrap.log`, `evidence/environment.json` | Locked setup, Python 3.12.14, macOS 26.4.1 arm64, Node 26.9.0, npm 11.19.1; skill paths/hashes. |
| E02 | `probe_api.py`, `evidence/api-before.jsonl`, `evidence/baseline-tests.log` | Original HTTP reproductions; 58 targeted tests passed. |
| E03 | `evidence/boundaries-red.log`, `evidence/boundaries-green.log`, `evidence/api-after.jsonl`, `evidence/focused-final-3.log` | Before/after input failures; final focused 78 tests passed. |
| E04 | `evidence/history-before.json`, `evidence/history-red.log` | Oldest-record truncation and three failing coverage regressions. The initial wrong `HistoryService` import was a harness error, corrected to `RunHistory`. |
| E05 | `evidence/history-green.log`, new `tests/test_history_coverage.py` | Exact corpus, restarted SQLite, pages, native snapshot and legacy fallback checks. |
| E06 | `evidence/history-comparison.json`, `evidence/real-history-performance.json`, `evidence/history-performance-final.json` | Precision/recall and all timing samples, including unfavorable/inconclusive comparisons. |
| E07 | `evidence/scoped-clarification.json`, `evidence/labels-red.log`, new `tests/test_clarification_labels.py` | Actual opaque-ID reproduction and provenance-preserving regression. |
| E08 | `evidence/baseline-e2e.log`, `evidence/baseline-e2e-isolated.log`, `evidence/lifecycle-replay.log`, `evidence/e2e-repeat.log`, `evidence/e2e-routing.log`, `evidence/routing-trace.zip`, `evidence/routing-network-summary.jsonl`, `evidence/e2e-optimized.log`, `evidence/optimized-trace.zip`, `evidence/optimized-network-summary.jsonl`, `evidence/e2e-alone-final.log` | Initial occupied 8765 port was an environment restriction; never killed its owner. Captured failures retained. Single lifecycle passes; whole suite alone passes 29/29 in 40.6 s. |
| E09 | `evidence/format-holdout.json` | Sixteen independent deterministic format cases. |
| E10 | `evidence/python-audit.log`, `evidence/web-audit.log`, `evidence/quality-audit.log`, `evidence/zizmor.log`, `evidence/gitleaks.log` | No known Python vulnerabilities/adverse statuses in 67 packages; both npm audits zero; offline zizmor no findings, 13 suppressed; 185 Git commits scanned with no leaks. |
| E11 | `evidence/baseline-full.log`, `requirement-contract/receipt.json`, `requirement-contract-final/receipt.json` | Exploratory full pytest: 1386 passed, one optional training skip, 87.74% coverage. Source changed during this run, so it is not final snapshot evidence. First contract run hit its 180 s harness timeout; second finished tests but correctly rejected changed source identity. Neither failed receipt is a pass. |
| E12 | `validation-speed-final/receipt.json`, `requirement-controls-speed/receipt.json`, `evidence/final-status.json` | Final immutable snapshot validation/check outcomes and hashes. These receipts, not preparatory logs, establish final deterministic completion; inspect terminal status. |
| E13 | `validation-final/receipt.json`, `evidence/final-failure-isolation.log`, `evidence/deadline-edit-red.log`, `evidence/deadline-edit-red-2.log`, `evidence/deadline-edit-green.log` | First final full suite: 1446 passed, two failures, one optional skip, 1375.46 s. Its receipt is failed, not a pass. Isolated failure distinguishes the durable-edit defect; strengthened interleaving reproduction and 15 focused passes support F10/F11. The canceled standalone helper lacked pytest's fake clock and is not completion evidence. |
| E14 | `validation-final-v2/receipt.json`, `evidence/deadline-edit-green-v3.log`, `evidence/deadline-edit-green-final.log` | Second full suite: 1445 passed, four deadline-evidence failures, one optional skip, 1080.02 s. Its receipt is failed. Narrowed final fix passes 56 focused controls, including those four failures and malformed prior reports. A misplaced test patch caused a syntax error, corrected before tests; no assertion was removed. |
| E15 | `evidence/engine-profile.prof`, `evidence/engine-profile-after.prof`, `evidence/progress-red.log`, `evidence/progress-green-v2.log`, `evidence/progress-holdout.log`, `tests/test_progress_persistence.py` | Profiled existing two-run fanout regression: 11.74 s before vs 3.06 s after; full row updates 300→20, decodes 304→24, full saves 302→22; 278 progress callbacks retained. 49 focused tests passed in 6.07 s. New holdout adds old-schema migration. Four original tests initially failed because interception targeted old full-write path; moved barriers to new progress writes with every assertion/timeout retained. |
| E16 | `validation-final-v3/22-web-tests.log`, `evidence/restore-red.log`, `evidence/speed-e2e.log`, `web/e2e/app.spec.ts` | Previous full Python suite 1453 passed, 1 optional skip, 88.60% coverage in 1528.47 s; browser28/29 failed at reload. New focused restore test failed before fix. Updated browser suite contains30 tests including remembered job/new tab/own-result cases. |
| E17 | `evidence/J17.json`, `evidence/J18.json`, `evidence/J19.json`, `evidence/speed-review.json`, `evidence/final-gate.json` | Consequential speed decision, instruction reconciliation, reload choice, final actual-patch review and completion gate. Inspect current artifacts for action/distributions; missing artifacts are gaps, not passes. |
| J01–J16 | `evidence/J*.json` and earlier tool-call distributions in this chat | Narrow plan, screening, ranking, verification, classification, alternative choices, hypotheses, patch review, and attribution. |

Final commands (run from the worktree, with inference keys removed and a fresh
fixture port):

```sh
env -u OPENROUTER_API_KEY -u OPENCODE_GO_KEY PROMPT_ENHANCER_DB=:memory: E2E_API_PORT=18883 uv run --locked python scripts/run_validation.py --output .local/teardown/validation-speed-final
env -u OPENROUTER_API_KEY -u OPENCODE_GO_KEY PROMPT_ENHANCER_DB=:memory: uv run --locked python scripts/check_requirement_contract.py --timeout 600 --output .local/teardown/requirement-controls-speed
```

The first runs every configured hook sequentially: hygiene, secrets/workflows,
lock/dependencies, Ruff, direct pytest parameter audit, ty, TypeScript/build,
Vitest, coverage pytest, and Playwright/axe. The second uses the runner's supported
600-second envelope; assertions and tested controls are unchanged. Final status
is written after checking terminal receipt statuses and artifact hashes. Remote
Checks/CodeQL are not triggered, since no PR or push is authorized. Optional
CatBoost training and cross-platform runtime checks were not run; no changed
training or platform-specific implementation depends on them.

## Consequential judgments and remaining uncertainty

J04's broad combined F1 claim received a low-confidence contradiction
(0.46); J05's narrowed actual rows verified the merge 500 and asymmetric 500/422
claims at 0.92/0.97. A claim about supported settings failure was rejected, and
the narrower verification initially lacked enough payload context for F3.
The paired engine probe supplied the missing observation. J06 classification
marked several findings review: the host applied explicit rules (reproduced
violation of the existing engine contract = defect; unsupported provider/UI
claims remain unresolved), rather than silently accepting weak labels.

J08/J11/J15 selected internal boundary validation, complete native snapshot,
and deferred copying respectively, with all supplied requirement checks
supported. They are judgment signals, not correctness proofs. J10 verified the
native truncation claim at 0.97. J12 rejected extrapolating fixture output to
real provider compliance (probability 0.11). J13's combined label verification
was unstable/low-confidence (contradiction 0.52, confidence 0.28); direct JSON,
browser observation, and a red/green regression establish the narrower local
claim without relying on that judgment. J16 assigned probability 0.02 to the
claim that the later browser failure proved lost evidence, and 0.87 to retaining
resource contention as an unproven competing explanation.

J14 reviewed the intermediate raw patch and escalated: safe-to-apply 0.10,
correctness confidence 0.28, composite 0.398, with no concrete defect explanation.
It is preserved, **not an approval**. The patch subsequently changed to address
large-history copying, and deterministic tests inspect semantics, not merely
exit codes. Stronger host source review checks: no broad exception swallowing;
validation precedes merge/work; numeric zero and genuine false survive;
snapshot normalizes the same fields without mutation; exact source provenance
survives labels; details and search retain evidence. Final semantic review and gate use the actual raw diff and current execution logs.
Their raw distributions and unfavorable outcomes are retained in E17. A model
review does not replace the deterministic contract checks or establish real
provider quality.

Remaining limits: intermittent behavior under competing local workloads is not
fully attributed; no real provider quality, current provider eligibility/privacy
policy, paid cancellation behavior, very large database memory profile, actual
screen-reader usability, mobile or other-platform execution was established.
These are separate from the concrete local fixes and their recorded checks.

## Speed investigation and bounded TypeSafe scope

The user’s speed request extended the investigation beyond the original budget.
The prior full coverage attempts lasted 23, 18, and 25½ minutes; retries found
real failures but profiling should have preceded repeating such expensive work.
The new process uses a discriminating profile, focused regressions, and then one
frozen full validation, with successful unchanged audit results reused.

Current TypeSafe primary API, Python question reference, Score primitive,
speculative fan-out pattern and parallel-question cookbook were retrieved on
2026-10-12, screened, and retained under `evidence/typesafe-docs/`. The Markdown
pages embed large UI code: a truncated initial fetch was discarded, targeted
excerpts were screened before use. The cookbook’s model-specific 10×/12.2× claims
are external example results, not measurements of this project. No SDK or new
service was added. The existing Gateway already batches common diagnosis,
taxonomy, gap, pointer and existence judgments, with dependent confirmation
when evidence requires it. Preserve those raw Choice/Noul/Score questions,
distributions, calibration, no-match and replay semantics while removing their
repeated persistence overhead. New inference would not help exact run identity
or SQLite writes. Jev J17 selected the dedicated progress column (0.89,
confidence0.87, no contradicted requirements); J19 preferred saved-result restore
(0.95, confidence0.94). A predictive speed Noul was uncertain(0.55): direct
measurements, rather than that hypothesis, justify the change.

The additive SQLite migration uses existing hardware and standard library only.
Old records with progress in evidence JSON remain readable, and later full saves
move progress to its dedicated column. Public record/evidence values are retained.
Older application versions do not understand this new progress column; rollback
compatibility with an older executable is not promised. Event snapshots still grow
with event count and complete history remains linear in stored evidence size.
No raw evidence was deleted to achieve these speed gains. No live application
accuracy/token/cost/latency improvement is claimed: provider experiments remain
excluded by the no-live-cases constraint. Development tool usage is logged; no
verified price metadata was available, so dollar cost is not inferred from tokens.

X08 falsification: replacing the native detail route with `JSONResponse` was
prototyped and rejected. Five same-fixture HTTP reads returned exact JSON values;
median latency was253.08ms before versus679.17ms in the prototype, with substantial
sample variability. The API already has a typed response serializer; this
experiment did not demonstrate a gain. The original route was restored. Logs:
`evidence/detail-before.json`, `evidence/detail-after.json`. The initial new browser
restore fixture omitted its required report and crashed only once restoration
worked; the fixture was repaired without changing application parsing or assertions.

The later two-minute full local validation objective supersedes the initial
no-development-dependency statement for the combined patch. Its source research,
awesome-list discoveries, eligibility and measured experiments are in
[validation-two-minute-research-2026-10-12.md](validation-two-minute-research-2026-10-12.md).
Final all-check outcomes/timing are in `.local/teardown/two-minute/validation-final-v11/receipt.json`.
Earlier receipts establish only their recorded source identities.


## Publication follow-up

The fresh pre-commit delivery run passed all 1,460 Python tests and the existing
89 Vitest tests but reproduced the history-details browser deadline failure. Its
trace shows overlapping full-detail reads for the same run during StrictMode
restoration and user history selection. Responses took up to 9.60 seconds under
the current load. This does not invalidate the earlier two measured 92-second
runs; it demonstrates their stated external-load limitation and extra work in
the reload path. The failed receipt and trace remain in the preserved evidence.

The client now shares only overlapping reads of the exact run-detail URL. It
does not cache completed responses, gives each caller an independent JSON value,
invalidates pending reads on run/job mutations, preserves caller-specific
cancellation requests, and leaves unrelated health assessments independent.
Five API regressions exercise coalescing, ownership, failures/recovery, mutation
ordering and URL/cancellation isolation. A browser regression opens a history
row while restoration is still pending and requires exactly one detail request.
Original lifecycle assertions and deadlines remain unchanged. Final delivery
checks and CI determine the outcome; no live provider/model inference is
authorized for this follow-up, including development Jev. The earlier model
reviews do not cover this new client patch or agent-guidance edits.
