# Two-minute local validation research — 2026-10-12

The first experiment should use explicit bounded local pytest-xdist workers with existing pytest-cov collection and the unchanged 80% aggregate coverage floor. This is a hypothesis to measure, not evidence that the pipeline already takes under two minutes. Source changes and final receipts belong to the implementation task; this research did not install dependencies or modify source/tests.

## Scope and baseline

Parent-reported baseline at the current teardown worktree: 1,456 Python tests passed and one optional training test skipped, with Python coverage taking 496.32 seconds; the complete 23-hook pipeline took 552.86 seconds. Vitest ran 89 tests in roughly 7 seconds and Playwright took roughly 30 seconds. Hardware has ten logical CPUs and 16 GB RAM. These measurements are inputs from the parent task and should be linked to its final receipt before presenting them as independently verified.

The remaining serial work is 56.54 seconds. Perfect eight-worker division would give 62.04 seconds for Python and 118.58 seconds overall; perfect ten-worker division would give 49.63 and 106.17 seconds. These are arithmetic scenarios, excluding duplicated collection, startup, uneven work, memory pressure, and coverage merge. Eight workers leave little margin. They are neither expected times nor a performance guarantee.

The repository currently locks pytest 9.1.1, pytest-cov 7.1.0, and coverage.py 7.16.1. Its top-level Python hook is serial; its receipt runner executes fast checks before sequential build/test hooks and rejects missing dependencies, skipped deterministic checks, and source changes.

## Candidates and primary evidence

| Candidate                 | Mechanism and burden                                                                                                                              | Evidence and limitations                                                                                                                                                                                              | Disposition                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Retain serial validation  | No dependencies or execution changes                                                                                                              | Actual baseline exceeds 120 seconds. Optimizing measured source hot spots remains useful and may be necessary.                                                                                                        | Control; not sufficient alone     |
| Local pytest-xdist 3.8.0  | Separate local interpreter workers, bounded with an explicit integer count; dynamic scheduling                                                    | Existing pytest-cov handles distributed coverage; upstream collection equality guards against different worker inventories. Introduces execnet 2.1.2, requires process-safe external resources and stable collection. | First measured prototype          |
| Standard-library sharding | Collect exact node IDs once; assign each to one subprocess; retain per-shard XML/logs; combine unique coverage files after every shard terminates | No new library dependency, but custom completeness, crash, cancellation, scheduling, and coverage orchestration must be implemented and tested.                                                                       | Fallback if xdist is incompatible |

Official xdist supports explicit worker counts and disabling distribution with `-n 0`. Its worksteal scheduler redistributes uneven queues and is a promising comparison for varying test durations. File/scope grouping preserves fixture locality but can create a long final group. Avoid optional `psutil` by selecting an integer count rather than logical-core discovery. [Distribution documentation](https://pytest-xdist.readthedocs.io/en/stable/distribution.html)

Every worker collects all tests; the controller compares exact IDs and ordering and aborts on disagreement. Local processes isolate Python globals/monkeypatches, but filesystem paths, ports, databases, and subprocess side effects still need explicit isolation. [Execution model](https://pytest-xdist.readthedocs.io/en/stable/how-it-works.html)

Session fixtures execute independently in workers. When a truly shared expensive setup is needed, use existing fixture resource naming or an explicit synchronization mechanism rather than assuming session means once per controller. This repository uses a function-scoped autouse deterministic clock, tmp_path stores, and a loopback HTTP test server that binds port zero; these inspected examples support feasibility, not proof that every test is parallel-safe. [Fixture and worker how-tos](https://pytest-xdist.readthedocs.io/en/stable/how-to.html)

pytest-cov documents aggregation across load-distributed workers. Preserve `--cov=prompt_enhancer`, reporting, and the configured `fail_under = 80`; success must include the combined final report, not individual passing workers. This research has not measured whether a changed scheduling mode yields identical executed lines on the project. [Distributed coverage](https://pytest-cov.readthedocs.io/en/latest/xdist.html)

Custom sharding must assign independent coverage base files. coverage.py warns that independently running tools with one base filename can erase or combine each other's files. Combine only this run's completed shard artifacts, retain inputs with `--keep`, fail if any shard is missing/failed, and enforce the aggregate floor after merge. Never average shard percentages. [Coverage combination](https://coverage.readthedocs.io/en/latest/commands/cmd_combine.html)

Unordered parametrization can produce different test IDs/order across workers. Distributed stdio/PDB debugging is limited; reproduce an individual failure serially with the same seed/environment. A worker failure or collection mismatch is a failed run rather than permission to drop a test. [Known limitations](https://pytest-xdist.readthedocs.io/en/stable/known-limitations.html)

## Addition eligibility

| Proposed dependency | Retrieved version and license | Required transitive path              | Verified here                                                                         | Still required before integration                                                                              |
| ------------------- | ----------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| pytest-xdist        | 3.8.0, MIT, Python >=3.9      | execnet >=2.1; existing pytest >=7.0  | Anonymous HTTPS PyPI metadata retrieval; public wheel URL/hash; upstream license text | Inspect exact wheel metadata/source and run isolated installation/operation with clean home and no credentials |
| execnet             | 2.1.2, MIT, Python >=3.8      | No non-extra requirements in metadata | Same anonymous metadata and public license retrieval                                  | Same isolated installed execution test                                                                         |

Both publish platform-independent Python wheels; optional testing, psutil, and process-title extras are not part of the proposed integration. The metadata and MIT grants support a zero software-license-charge local path. Anonymous retrieval establishes public access to metadata/artifacts, not the complete operational eligibility test. The parent task is separately verifying clean-home install and operation. Local execution consumes CPU and memory and does not need a model API, account, signup, or paid service. Remote SSH execution, hosted services, and optional extras are excluded.

Primary sources: [pytest-xdist PyPI metadata](https://pypi.org/pypi/pytest-xdist/json), [execnet PyPI metadata](https://pypi.org/pypi/execnet/json), [pytest-xdist license](https://github.com/pytest-dev/pytest-xdist/blob/master/LICENSE), [execnet license](https://github.com/pytest-dev/execnet/blob/master/LICENSE). Retrieved metadata/URLs/hashes and upstream license excerpts are retained in `.local/teardown/validation-speed-research/`. Upstream master licenses are mutable; inspect pinned wheel license contents before approval.

## Experiment protocol and acceptance

1. Hold application source and environment fixed; run the serial control or reuse the current unchanged receipt. Record source digest, exact inventory, allowed pre-existing training skip, coverage percentage/lines, elapsed time, and peak resource costs.
2. Verify the new dependencies in the intended anonymous clean-home installation/operation path before adding them. Do not read or inherit provider credentials.
3. Compare eight-worker then ten-worker top-level coverage runs using the same tests, assertions, fixture gateways, and coverage floor. Use worksteal initially; compare default load only if a scheduling imbalance materially affects the decision. Stop unchanged retries.
4. Preserve full XML/logs, exact executed test IDs including parameter values, collection count, pass/fail/skip totals, worker count, and aggregate coverage. Any missing or duplicate scenario, unexpected skip, worker crash, or collection mismatch fails acceptance.
5. Run representative failure controls: known failing test remains failure; missing worker/dependency/coverage data cannot produce a completed receipt. Check isolation of temporary stores, ports, and inherited environment.
6. Run the actual complete 23-hook receipt on the final frozen diff. The target is measured elapsed <=120 seconds with all existing deterministic hooks complete, full test inventory and coverage unchanged in meaning. Report a miss honestly and inspect measured critical paths.

Prefer configuring workers on the top-level hook or a dedicated explicit runner. Do not inject distributed flags into global pytest addopts/PYTEST_ADDOPTS: several repository tests launch nested pytest/scripts, and implicit propagation risks recursively starting workers. Keep `UPDATE_GATEWAY_KEYS` absent; that environment flag deliberately writes the pinned fixture. For debugging, provide a serial option. Keep fixture-backed Playwright sequential from heavy coverage unless a new discriminating experiment defeats the existing contention hypothesis.

## Jev and evidence boundaries

Jev screening passed the xdist distribution/execution/limitations docs, pytest-cov distributed coverage docs, coverage combination docs, metadata, and license excerpts; injection signals were 0.02–0.04. The TypeSafe index was screened as irrelevant to validation mechanics and excluded from the recommendation; a TypeSafe citation-check fetch returned no useful content and was skipped. General Python subprocess documentation also received skip and was not used as research evidence.

Jev verified the combined-coverage claim (confidence 1.0), metadata/license/transitive claim (0.98), and concurrent coverage base conflict claim (1.0). A negative control claiming documentation proves a sub-120-second result was unsupported (1.0). The bounded experiment choice selected xdist with probability 0.56 and confidence 0.47, with investigate/none each 0.18; there was no escape or requirement contradiction. That uncertainty was sent to the parent for stronger reasoning and concrete measurement. It supports a prototype rather than rollout or completion.

Raw final verification/decision distributions and available screen results are saved in `.local/teardown/validation-speed-research/judgments.json`. Configured Jev development calls were used under the latest user instruction; no live optimizer/provider sessions or credentials were read. The installed Jev and TypeSafe skills and live matching tool schemas were read. No new project TypeSafe integration was proposed by this validation-only subtask.

Research is complete as a source-backed experiment recommendation. The two-minute implementation, performance outcome, clean-home dependency eligibility, full regression receipt, and final patch gate remain the parent task's work.

## Awesome-list expansion and implemented experiments

Curated context-awesome searches covered Python testing/performance sections,
pytest, testmon, profiling, Scalene, pytest-xdist, pytest-split, py-spy,
Pyinstrument and Memray. Generic taxonomy queries mostly returned load-testing
lists; focused queries found more relevant leads. Results were screened and
canonical URLs/hosts retained. List stars describe the hosting list, not tool
fitness. This bounded discovery is not an exhaustive corpus audit.

| Lead | Awesome host / primary evidence | Decision |
| --- | --- | --- |
| xdist | jimbrig/jimsghstars; [source](https://github.com/pytest-dev/pytest-xdist) | Implemented local workers. Complete8worker CTracer run216.51s: insufficient alone. |
| testmon | cleder/awesome-python-testing; [source](https://github.com/tarpas/pytest-testmon) | Changed-test selection is useful incrementally; it cannot establish faster complete fresh execution. Not installed. |
| Pyinstrument | msaroufim/awesome-profiling; [source](https://github.com/joerick/pyinstrument) | Wall-time profiler lead if further hotspot diagnosis is needed; installation/benefit untested. |
| py-spy | vinta/awesome-python; [source](https://github.com/benfred/py-spy) | Primary docs require root on macOS; existing cProfile/sample used, no permission changes or installation. |
| Memray | olest/awesome-performance; [source](https://github.com/bloomberg/memray) | Allocation profiler; no measured memory bottleneck justifies an addition. |
| Scalene | msaroufim/awesome-profiling; [source](https://github.com/plasma-umass/scalene) | CPU/native/allocation profiler lead; no execution eligibility or benefit established, no hosted AI service added. |
| pytest-split | Focused query found no distinct curated item | Not added; extra shard/merge machinery remains unverified. |

Jev's full-run relevance ordering was xdist0.87, sysmon0.73, Pyinstrument0.60,
sharding0.53, testmon0.33, py-spy0.28, Memray0.10. These are judgment signals.

Root verified exact wheel hashes/licenses, then installed xdist3.8.0 in a fresh
venv with empty HOME, public PyPI, disabled uv configuration/keyring, no inherited
credentials and no Python downloads. Eight cases passed on two local popen
workers. Logs/metadata are under `.local/teardown/two-minute/`: anonymous-install,
anonymous-operation, package-inspection and eligibility. Added dependencies are
development-only MIT xdist3.8.0 and execnet2.1.2; no optional extras/services.

Existing coverage7.16.1 supports line-only sysmon on existingPython3.12.14; see
[versioned core documentation](https://coverage.readthedocs.io/en/7.16.1/config.html#run-core)
and installed-source hashes in validation-speed-research/sysmon-source.json.
Branch/context limitations warn/fall back; coverage plugins need CTracer.
The representative ThreadPool/round fixture recorded exactly6061 executed lines
with both cores. Diagnostic subset probes disable the whole-corpus coverage
floor only for measurement; they are not verification. Full runs retain80%floor.

Complete1457case inventories match serial/parallel/core controls:1456passes and
one existing optional CatBoost skip. CTracer8workers216.51s/88.60%; Sysmon8workers
98.90s/88.59%. Two missing nondeterministic race-guard lines were investigated,
and a new holdout deterministically invokes obsolete expiry and a late checkpoint
callback, asserting no transition/store mutation and RunCancelled.

Fast checks still finish first. Independent Python coverage, build, Vitest and
Chromium checks now overlap under one main-thread supervisor, with individual
logs/PIDs/hashes/statuses. Failure/interruption stops unfinished owned groups.
New barrier and deliberate-failure controls prove concurrency after fast checks,
and peer termination. The initial barrier harness correctly tripped source-change
validation because it wrote untracked coordination files; those artifacts moved
under ignored.local, with assertions retained. A typing error was fixed.

The complete120second target must be established by a passing frozen final receipt.
The next complete experiment is `.local/teardown/two-minute/validation-final-v11/receipt.json`. Its wall-time field
includes every configured check; it excludes dependency installation/downloads,
remote CI and optional training. Sum of overlapping check times is not wall time.

The first all-at-once run failed three existing UI tests at their unchanged five
second timeout (40.43s Vitest, versus7.28s alone); Python/browser peers were
correctly interrupted and the failed receipt retained. The next experiment bounds
Vitest to one worker, Chromium to two local workers (CI remains one), and Python
to eight. Guard/fixers stay sequential; read-only fast hooks run in batches of four
before tests. No assertion, timeout, test inventory or coverage floor changed.
Focused runner/persistence controls passed24tests; original receipt controls also
passed18tests. Root source review checks owned process-group cleanup and receipt
failure semantics. The final complete receipt determines the120second result.

A subsequent startup failed because the aborted first browser group left its own
detached fixture servers on5174/18890. Root verified their PID creation times and
worktree cwd before stopping those owned servers. The supervisor now requests
SIGINT cleanup before SIGTERM/SIGKILL escalation, keeping its leader unreaped to
reserve the process-group identity. A synthetic detached child regression verifies
peer failure allows cleanup and leaves neither process alive. Failed receipts
and server attribution remain retained; no existing user session was touched.

The bounded run recorded23finished checks in90.945911seconds (receipt timestamp
difference),22passed and one failed Python check. Build,89unit and30browser cases
passed. Python had1459passes/onefailure/oneoptional skip,84.72seconds,88.62%coverage.
The existing process guard exposed macOS EPERM when signaling only-zombie groups
after graceful SIGINT. The original zombie-only exception policy now applies at
every escalation signal; a group with a live member still propagates EPERM.
The failed receipt is preserved, and the complete passing target still requires
v4. Jev's broad failed-run timing/cause claim returned a low-confidence contrary
signal (confidence0.46); deterministic23/22/1counts and timestamp arithmetic
retain the precise narrower observation without treating the signal as a pass.

The next bounded overlap attempt failed one browser case while History remained
loading at its unchanged5second assertion; peers were stopped and the failed
receipt retained. Caps alone do not establish response reliability. The new
experiment lowers Python-check CPU priority by POSIX nice10 when available, leaving
browser/API work at normal priority. Existing/usr/bin/nice ran an isolated
stdlib consumer reporting priority10 without credentials or elevation; no package
or service was added, and platforms lacking nice retain the same checks at normal
priority. All stages/checks remain. This is an experiment until the final v5
receipt passes and meets120seconds, not proof that priority guarantees deadlines.


The nice10 overlap experiment also failed the unchanged browser deadline; its
failed v5receipt is retained. Priority and all-heavy overlap are rejected. The
next prototype runs web build/unit/browser together, then Python separately.
New barrier controls require both web checks to start together and finish before
Python begins; the peer-failure control remains within the concurrent web phase.

The existing Python3.14.7 interpreter was inspected and tried in a separate
clean-home locked environment. Its ten-worker Python run took116.34seconds,
1458passes/oneoptional skip/two new harness failures (the scheduling policy
changed during this exploratory run), and29226warnings mostly unclosedSQLite
resources. This is not a frozen final result or evidence for adopting3.14.
Python3.12 remains the chosen stack. The interpreter is GIL-enabled, JIT-disabled,
and tail-call-interpreter-enabled; no speed claim is attributed to those flags.
Jev favored retaining3.12/isolation at0.73 (confidence0.69), with no contradicted
requirements, while the full-coverage requirement remained unknown pending checks.

A cProfile run of the slowest original recording replay control observed821
saves,539806 audited records and541311 request-key computations:9.54seconds in
recording saves,5.82seconds validating captures. Reducing these audits would
conflict with the documented every-save integrity contract. No audit was removed.
The complete pipeline keeps the original80%coverage floor and all assertions.
Detailed profiling and failed experiments are retained under
`.local/teardown/two-minute/`.


The isolatedweb/Python v7pipeline passedall23hooks in135.704seconds (Python
1460passes/oneoptional skip,88.62%,89.84seconds;89Vitest,30Chromium). The120second
requirement remains unmet. A nativeJSON serialization experiment was rejected:
1347syntheticcaptures measured stdlib3.72ms versus native+strictvalidation4.22ms;
plainnative1.05ms but incorrectlyaccepteddates/bytes thatstdlibrejects. Removing
sortkeys didnotimprove theprofile; source restored exactly. Everycaptureaudit
andcanonicalrequesthash remain intact. Primary PydanticAPI was screened
(pass,injection0.04), Jev favored boundednativeprototype0.96, but directnegative
evidence defeated integration. NoPydantic dependency/source change wasaccepted.

The next v8experiment retains web-beforePython and raises boundedcaps:Python
autoworkers maximum10 (availablelogicalCPUcap), localChromium4, Vitest2. CI
Chromium remains1. Fullinventory, assertions andtimeouts stayfixed; finalreceipt
determines benefit/reliability rather than worker count or arithmetic.


The ten-worker v8run passedall23hooks but took191.717seconds:162.80seconds
Python,1460passes/oneoptional skip,88.62%; isolatedwebphase21.221seconds,all
30browser and89unitcases. IncreasedPythonconcurrency isrejected as measured
regression. The next final v9run restores the observed eight-workercap andretains
the improvedfourChrome/twoVitestwebphase. The earlier eight-worker89.84second
Python result doesnotguarantee120seconds withchangedwebcaps or changingmachine
load; only thecomplete finalreceipt establishes its measuredoutcome.


V9fourChrome/twoVitest webphase failed oneunchangedfive-second detail assertion:
29/30passed; Python neverstarted. Savedtrace GETdetails took4.933/4.891seconds;
this is observedtiming, notproofsystemswapcausedit. Hardwareinspectionshows
fourperformance/sixefficiencycores; afterv8systemmemoryfree48%,swap18745.88MiB.
No unrelatedload modified. Nextv10prototype boundsPython tofourworkers and
overlaps onlybuild/Python/Vitest, thenrunsChromiumalone. Barrierregressionsnow
assertbothCPUtestchecksstarttogether andbrowserbeginsafterbothfinish; peer
failurestops itsCPUpeer. Alloriginal18receiptcontrols retained, noassertion
or timeout changed. Thistestsresourcecontention ratherthan promisingperfect
scaling. Earlierincreasedcaps/fullconcurrency are rejected experiments.


V10CPU-then-browser passedall23hooks in134.070seconds:Python4 took99.60seconds
(1460passes/oneoptional skip88.62%), unit89/buildpassed, browser30passed24.000s.
Thispreservescorrectness butmisses120seconds by14.070seconds. One finalmiddle
worker-count experiment uses6Pythonworkers, Vitest2, browser4alone. Thereisno
prior6workermeasurement; itisnotaprovenwin beforeitscomplete receipt. This
boundedworkercapcomparison closes theplanned investigation; further research
without a newbottleneck observation adds no evidence. Finalreport mustretain
partialstatus if120seconds remainsunmet. All earlierlogs/receiptsarepreserved.
