# PR 181 Qodo disposition

Comparison base: `41ea4726d6879fe70c6e73201814535deaf7c2b9` (merged PR 181).
The review was fetched again on 2026-10-06, including paginated inline comments,
PR comments and reviews. The inline set contained ten findings and no replies.
Qodo reviewed `e4c56805cd9614adac842b7316bffa78b11f3f79` after PR 181 merged.
The findings below were assessed against the comparison base and fresh controlled
reproducers. Suggested repairs and previous green checks were not used as proof.

## Individual dispositions

| Finding | Disposition and repair | Reproducible evidence |
| --- | --- | --- |
| [1: stale cancellation](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420107) | Confirmed and fixed. Persistence checks current job identity while holding the persistence lock used for replacement and terminal writes. The in-memory status lock is released before SQLite I/O. | `test_stale_cancel_cannot_persist_over_a_replacement_job` holds an old cancellation before its write, finishes the old job, starts a continuation with the same run ID, then releases cancellation. Before: the durable job becomes the old completed, cancelled optimization. After: it remains the running, uncancelled continuation with its operation intact. |
| [2: transient worker saturation](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420165) | Confirmed and fixed. Both workers release capacity before signalling completion. Acquisition waits in cancellation-aware intervals within the existing operation deadline. The eight-worker cap remains. | `test_gateway_waits_for_transient_worker_capacity_within_deadline` covers both routing and transport: before, `transport_busy`; after, a successful raw response. `test_cancellation_while_waiting_for_worker_capacity_starts_no_request` proves prompt cancellation without transport. `test_eight_way_panel_survives_one_lingering_worker_slot` proves eight calls finish when a temporarily occupied slot becomes available. The existing unresponsive-worker test proves permanent saturation remains bounded and starts no ninth worker. |
| [3: identical prompts](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420123) | Confirmed and fixed. Every variant needs one consistent prompt; the rewrite must differ from the original after stripping outer whitespace. Invalid experiment inputs raise `PromptExperimentReportError`. | `test_identical_variant_prompts_cannot_establish_rewrite_improvement`: before, an identical prompt plus differing sampled quality can establish improvement; after, rejection. `test_each_variant_requires_one_consistent_prompt` also covers inconsistent rewrite samples; original consistency was already enforced. |
| [4: arbitrary sample IDs](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420136) | Confirmed and fixed. String classification IDs are matched to metric IDs without a hard-coded prefix. | `test_quality_judgments_match_arbitrary_string_sample_ids`: matched `case-0`/`case-1` quality is unresolved before and correctly failed/passed after. Complete paired evidence can then establish improvement. |
| [5: conflicting classifications](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420147) | Confirmed and fixed. Duplicate classifications that disagree on classification or decision fail closed. Repeated agreeing classifications remain usable. | `test_conflicting_classifications_fail_closed_in_either_order` supplies disagreement across two classification entries and reverses their order. Before: order chooses the quality result and can establish improvement. After: both orders raise an explicit conflict error. |
| [6: handled provider failure telemetry](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420159) | Confirmed and fixed. Screening and relation provider-error handlers sit outside their operation scopes, preserving the existing fallback while the scope observes the failure. | `test_handled_provider_failure_emits_error_for_the_success_test_substage` exercises the real HTTP Gateway context manager with a scripted failing provider. Before: terminal `end`; after: terminal `error` with `ProviderError`, for both substages. Screening still rejects unusable tests; relation failures still retain accepted tests. |
| [7: body-read cancellation](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420175) | Confirmed and fixed. Request-scoped HTTP abort shuts down the socket before closing it; the worker's response context closes the response. Already-closed sockets and custom close-only wrappers remain supported. | `test_http_abort_interrupts_real_socket_body_read` uses a local HTTP server that sends headers and stalls mid-body. Before: abort leaves the makefile read blocked beyond the 500 ms assertion. After: the read finishes within that bound. Waiting for response headers still has the documented standard-library handle limitation. |
| [8: cancellation spend](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420185) | Confirmed and fixed. A successfully returned response reaches HTTP-adapter usage accounting before cancellation or deadline rejection. The cancelled answer is still withheld. An interrupted request with no completed response supplies no known usage. | `test_completed_paid_response_is_accounted_before_cancellation`: before, zero calls/tokens/cost; after, exactly one call, 18 tokens and $0.03, with `RunCancelled`. `test_cancelled_api_job_persists_completed_provider_spend` additionally proves $0.03 in the cancelled result, SQLite and history. This does not estimate unknown usage from a response that never reaches the caller. |
| [9: restart during continuation](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420196) | Confirmed and fixed. Recovery fills empty or active placeholder results, preserves canonical saved results and timing, and exposes the preserved result in the interrupted job snapshot. | `test_restart_preserves_saved_canonical_result` covers resume, skip, continue and optimize with paused or completed results. Before: all eight canonical results are overwritten. After: job state is interrupted, canonical result/timing/context survive, and `_paused_record` still accepts paused runs. The existing interrupted-placeholder test still produces a truthful operational failure. |
| [10: every event writes SQLite](https://github.com/nicolas-found42/prompt-enhancer/pull/181#discussion_r4197420207) | The write-amplification observation is supported. Optimization is deferred: a harmful production regression or violated latency requirement was not reproduced. Durable operation telemetry remains intact. Throttling would deliberately lose recent restart evidence; that tradeoff is not justified by the current measurements. | Fresh isolated real-SQLite benchmark: eight event-emitting threads, 64 events, three repetitions per condition. Four stage/lifecycle commits become 68. Median elapsed times for 0/256 KiB/2 MiB synthetic evidence are 19.404/100.629/851.789 ms versus 1.135/5.502/43.479 ms with no operation events. Evidence contents and terminal state survived every run. These are synthetic sizes and timings, not production distributions or proof of the historical stall. |

## Follow-up review on PR 182

Two additional findings were fetched and assessed before merging the follow-up.

- [Cancelled capacity wait leaves routing start unclosed](https://github.com/nicolas-found42/prompt-enhancer/pull/182#discussion_r4197681131): confirmed. The existing cancellation-capacity test was extended to assert the actual `route_model` start/error pair and failed on `6281888`. Capacity-acquisition exceptions now emit the routing terminal error before propagating to the outer operation. No provider request starts.
- [Job status polls wait behind SQLite event writes](https://github.com/nicolas-found42/prompt-enhancer/pull/182#issuecomment-6020388113): confirmed as a lock-coupling regression introduced by the initial ownership repair. `test_job_status_and_cancel_signal_do_not_wait_for_sqlite_io` controls a blocked real-store write, demonstrating that `get`, `active`, and the provider-facing cancel signal all wait before repair. A separate persistence lock now coordinates ownership/replacement/terminal writes; the in-memory lock covers only job lookup/mutation. All three checks pass while the old stale-cancel and durable-transition regressions remain green. The cancellation HTTP response can still await its own durable write, but setting the provider-facing cancellation event does not await unrelated SQLite I/O. Operation-write frequency is unchanged.

The original performance measurements above describe the synthetic workload at
`6281888`; they establish write amplification, not a final-source latency
promise. The added lock regression establishes responsiveness through controlled
scheduling rather than interpreting benchmark timings as a production SLA.

## Run the regression evidence

```sh
uv run --locked pytest -q tests/test_prompt_experiment_report.py tests/test_jobs_api.py \
  tests/test_gateway.py tests/test_success_test_screening.py tests/test_success_test_set_relations.py
```

Before/after logs, the real-SQLite benchmark script and results, fetched review
JSON and final validation/review/delivery receipts are retained outside the
checkout in the separate `2026-10-06-qodo-181` directory under the owner's
`Documents/github/prompt-enhancer-evidence`. The previous evidence archive is
unchanged. These local artifacts are not public GitHub download links; the
committed tests above let reviewers reproduce the behavioral checks.

The fix commit is the commit introducing this document and its regression tests;
the follow-up PR links this disposition map. Required deterministic validation
and exact-head CI determine delivery readiness. Advisory review results are
recorded separately, with their actual scope and completion status.
