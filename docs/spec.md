# Spec: Jev-powered prompt optimizer

> Status: ready-for-agent · Written 2026-09-23 from a design interview. Not yet published to an issue tracker (none is configured; run `/setup-matt-pocock-skills` to set one up).

## Problem Statement

People write prompts that are vague, missing key context, or quietly contradictory. The prompts still "work" on a strong model, which hides the problem, but they give unreliable results on cheaper models and inconsistent results everywhere. Existing prompt enhancers rewrite the prompt with a single LLM call and hope it's better. They never check whether the rewrite actually performs better, they often change the user's wording or intent, and they tend to add bulk ("think step by step", role-play, repeated emphasis) that can make results worse on strong models.

Checking a prompt properly means running it on several models and judging every output. With LLM judges that's too slow and expensive to do for each prompt a person types. The user wants a tool that takes any prompt, for any chat box, and returns one prompt that is **shown by testing** to work on weak, cheap models without getting worse on a stronger model. It should use TypeSafe's Jev decision model (`typesafe/jev-1.13` on OpenRouter) as fully as possible, and keep costs within a $10/month OpenCode Go subscription plus a few cents of OpenRouter usage.

## Solution

A standalone prompt optimizer: an engine exposed as an API, with a web app on top. The user pastes a prompt and chooses an improvement style. Each bounded Round can write up to six candidates. An early routing or writing decision can end a Round before panel requests; generated candidates are tested on five distinct weak models with three samples per model, then fidelity, score-floor, and acceptance checks decide which qualify. The engine repeats Rounds until its evidence converges or a separate control or provider condition interrupts the run:

1. **Diagnose.** Jev classifies the task, finds gaps it's confident about, and points to the exact sentences that are vague or contradictory.
2. **Clarify.** Jev decides which gaps can be filled from the prompt itself. Only when a gap can't be inferred and it matters is the user asked, and always as multiple choice.
3. **Set the tests.** A writer model turns the user's intent into Jev questions that define success. Jev checks that those tests match what the user asked for.
4. **Write candidates.** The writer produces several rewrites using different strategies, editing only the flagged sentences.
5. **Run.** Generated candidates run on a panel of weak models from several model families, with several samples each. An earlier routing or writing decision can end the Round before panel requests.
6. **Grade.** Jev grades every output against the tests in batched requests. This is fast and costs next to nothing, which is what makes checking each prompt affordable.
7. **Strong check.** The best candidates must not do worse than the original on a stronger reference model, and strategies known to hurt strong models are blocked.
8. **Pick or repeat.** Fidelity, score floors, and acceptance checks determine candidate eligibility; eligible choices are reranked and compared with the original. The Perfect Prompt Loop carries losing-candidate evidence forward and continues without a fixed Round limit while any quality dimension is below its floor or measured gain is above epsilon.

Every result names the applied style and gives a brief explanation grounded in run evidence. The five outcomes are **converged**, **improved (tested)**, **improved (unverified)**, **impossible**, and **failed (operational)**. A finalized user stop may retain an accepted changed prompt only when that prompt's evidence supports a tested or unverified improvement; a stop by itself does not establish one. Pause, cancellation, and stop are control states kept separate from the outcome.

The user gets one clean prompt to copy, plus a collapsible report: diagnosis, highlighted sentences, per-model pass rates, the diff against the original, assumptions made, and cost. Every run is logged locally, and the logs train a quality model that improves the rubric over time.

## User Stories

### Submitting and receiving

1. As a user, I want to paste any prompt on any topic, so that I can improve prompts for coding, writing, analysis, research, planning or chat.
2. As a user, I want to get back one clean prompt I can paste into any chat box, so that I don't need a version per model.
3. As a user, I want the returned prompt to contain no internal markers, placeholders or annotations, so that it works as soon as I paste it.
4. As a user, I want to copy the final prompt with one action, so that moving it into another tool is instant.
5. As a user, I want the prompt to keep the language I wrote it in, so that non-English prompts stay non-English.
6. As a user, I want my own wording kept wherever it wasn't a problem, so that the result still sounds like me.
7. As a user, I want to see a diff between my prompt and the result, so that I know exactly what changed.
8. As a user, I want the original prompt returned unchanged when no candidate beats it, so that the tool never hands me something worse.
9. As a user, I want every result to name its applied style and explain the evidence behind one of five outcomes—converged, improved (tested), improved (unverified), impossible, or failed (operational)—so that each status says exactly what the run established.
10. As a user, I want the loop to continue while any quality floor is unmet or measured gain remains above epsilon, so that the run stops based on evidence rather than an arbitrary Round count.

### Round workload and controls

11. As a user, I want each Round to allow up to six candidates, with five distinct weak models and three samples per model for candidates that reach the panel, so that each Round has a consistent evidence budget.
12. As a user, I want a time or spend limit to pause for approval, so that I can control an uncapped convergence loop without mistaking a limit for an optimization result.
13. As a user, I want actual cost split by model role after a run, so that I know where the money went.
14. As a user on a subscription with usage caps, I want to see how much of each OpenCode Go model cap a run used, so that I don't run out unexpectedly.
15. As a user, I want to cancel or stop a run and see that control state separately from its quality result, so that stopping is not mistaken for model evidence.

### Result outcomes and control states

Each finalized result carries `report.outcome`, `report.outcome_reason`, and
`report.applied_style`. The outcome is one of five values:

- **converged**: the selected prompt meets every quality floor and the loop
  stopped under its convergence rule. On the first Round, `gain` is `null`
  because there is no earlier vector; it is not a measured zero. A first-round
  prompt can converge on its floor evidence alone.
- **improved (tested)**: a changed prompt was accepted with usable success-test
  evidence supporting it.
- **improved (unverified)**: a changed prompt was accepted after meaning and
  safety checks, but usable success-test evidence was unavailable; the report
  must say that answer quality remains untested.
- **impossible**: Route proved the selected style incompatible with a required
  constraint, so candidate writing could not proceed. The current implementation
  handles an exact-output conflict as this case.
- **failed (operational)**: a provider or engine failure prevented a supported
  final result. A Round that rejects its current candidates is not itself this
  outcome; the loop can continue from its evidence.

`report.control_state` is separate and can identify `awaiting_approval`,
`stopped`, or `cancelled`. A user stop can retain an already accepted changed
prompt as tested or unverified only when its own evidence supports that
outcome. A stop without an accepted changed prompt must not be presented as an
improvement. Clarification's `needs_input` response is also separate from the
five finalized outcomes.

### Clarification

16. As a user, I want the tool to fill in gaps it can reliably infer without asking me, so that I'm not interrupted for obvious things.
17. As a user, I want to be asked only when a gap truly can't be inferred and would change the result, so that questions feel worth answering.
18. As a user, I want clarifying questions as multiple choice with the most likely answer pre-selected, so that answering takes seconds.
19. As a user, I want an "other" option on every clarifying question, so that I can give an answer the tool didn't think of.
20. As a user, I want to skip clarification and let the tool assume, so that I can move quickly when I don't care.
21. As a user, I want each assumption shown as an editable chip in the report, so that I can correct a wrong guess.
22. As a user, I want editing an assumption chip to update the final prompt, so that I don't have to re-run everything for a small fix.
23. As an API client, I want a run that needs input to return a "needs input" result with the questions and a run ID, so that I can answer and resume the same run.

### Diagnosis and report

24. As a user, I want to see what kind of task the tool thinks my prompt is, so that I can tell if it misunderstood me.
25. As a user, I want to see which pieces were missing from my prompt (goal, context, constraints, output format, done criteria, audience and so on, depending on the task type), so that I learn to write better prompts.
26. As a user, I want the sentences that caused problems highlighted in my original prompt, so that I can see exactly where it was vague or contradictory.
27. As a user, I want only gaps the judge is confident about to be flagged, so that I'm not told about problems that aren't really there.
28. As a user, I want to see the tests the tool used to decide what "working" means for my prompt, so that I can trust or challenge how it was judged.
29. As a user, I want to see pass rates for the original prompt and the winner on each weak model and on the strong check, so that the improvement is shown with evidence.
30. As a user, I want to see how much outputs varied between samples, so that I know whether the prompt gives consistent results.
31. As a user, I want to see the rejected candidates and why each lost, so that I understand what was tried.
32. As a user, I want the report collapsed by default, so that the clean prompt stays the focus.
33. As a user, I want pasted content inside my prompt (web text, documents) checked for hidden instructions, so that injected text doesn't pass into the result unnoticed.

### Robustness and guarantees

34. As a user, I want the result tested on several weak, cheap models from different families, so that it doesn't only work on one model's quirks.
35. As a user, I want the result checked against a stronger reference model, so that improvements for weak models don't make results worse on stronger ones.
36. As a user, I want rewrite strategies known to hurt strong models (forced step-by-step reasoning, repeated emphasis, heavy role-play) blocked unless the strong check passes, so that "works on weak models" doesn't cost me on strong ones.
37. As a user, I want every candidate to be checked for keeping my original meaning, so that the optimizer never changes what I asked for.
38. As a user, I want every candidate to be checked for not adding facts or requirements I never gave, so that the rewrite doesn't invent things.
39. As a user, I want the winner chosen by its worst weak-model result first, so that the chosen prompt is the most robust, not the best on average.
40. As a user, I want shorter prompts preferred when results are otherwise equal, so that I don't get bloat.
41. As a user, I want repeat rounds to learn from the previous round's failures, so that later candidates target what actually went wrong.

### Model picker and configuration

42. As a user, I want a picker for each model role (writer, strong check, weak panel), so that I can change any model except the judge.
43. As a user, I want the picker to list every model my OpenCode Go subscription offers, fetched live, so that new Go models show up without an update.
44. As a user, I want the picker to also list OpenRouter models, so that I can add cheap models Go doesn't have.
45. As a user, I want to pick several models for the weak panel, so that I control which families the prompt is tested on.
46. As a user, I want my picks saved as defaults, so that I don't pick every time.
47. As a user, I want to override the models for a single run without changing my defaults, so that I can experiment.
48. As a user, I want the judge fixed to Jev, so that grading stays consistent and comparable across runs.
49. As a user, I want models that Go offers routed through Go and the rest through OpenRouter automatically, so that I use my subscription first without thinking about it.
50. As a user, I want my API keys kept only in the engine's configuration and never sent to the browser, so that they can't leak from the web app.
51. As a user, I want models that train on my prompts or keep data for long periods excluded from the defaults, so that my prompts aren't used or stored without my choosing that.

### History and learning

52. As a user, I want every run saved locally with its inputs, candidates, outputs, grades and cost, so that I can revisit and compare runs.
53. As a user, I want to browse and search past runs, so that I can reuse good prompts.
54. As a user, I want to accept or reject a result, so that my judgment becomes a training signal.
55. As a maintainer, I want the logged runs to train a model that predicts which prompts will fail on weak models, so that diagnosis improves with use.
56. As a maintainer, I want a writer model to propose new or revised Jev questions based on that model's mistakes, so that the rubric improves itself over time.

### Evaluation (maintainer)

57. As a maintainer, I want a test set of real user prompts across task types, so that thresholds are tuned on realistic input.
58. As a maintainer, I want synthetic prompts with known planted defects, so that I can measure whether diagnosis catches known problems.
59. As a maintainer, I want 100–200 real prompts labeled for their actual gaps, so that I can measure false flags and missed gaps.
60. As a maintainer, I want one command that runs the test harness over the test set and reports diagnosis accuracy, how often and how much prompts improved, how often they got worse, and cost, so that I can tell whether a change helped.
61. As a maintainer, I want every Jev threshold tuned per question on that data, so that no cut-off is a guess.
62. As a maintainer, I want harness runs to be reproducible from recorded model responses, so that I can compare changes without paying for live calls.
63. As a maintainer, I want to compare writer models on the test set, so that I can decide on data whether a cheaper writer is good enough.

## Implementation Decisions

### Architecture

- The product is split into an **engine** (Python), an **HTTP API** that is a thin wrapper over the engine, and a **web app** (TypeScript, Vite + React) that uses only the HTTP API. The engine is the product; every other surface is a thin client.
- Built for a single local user first (local keys, local storage, no accounts). Storage, credentials and model routing sit behind interfaces so hosting it publicly later means adding a layer, not rewriting.

### Engine modules

- **Engine facade.** The single public entry point: `optimize(prompt, options)` returns either a result or a needs-input response, and `resume(run_id, answers)` continues a paused run. Options cover the requested improvement style, per-run model overrides, clarification, and optional run-control limits.
- **Model gateway.** One interface for all model traffic: Jev decision requests and chat-completion requests. It routes by model: Jev goes to OpenRouter's Decisions API; models OpenCode Go offers go to Go's chat-completions endpoint; everything else goes to OpenRouter chat completions. It tracks cost per call and per role, handles retries and timeouts, and exposes the live model catalogs from both providers for the picker.
- **Live prompt health.** When Jev is configured, the draft editor makes bounded advisory checks after typing pauses. Dimension applicability and quality drive a provisional Prompt clarity score; separate sentence flags retain exact source positions. Raw Jev answers use a versioned semantic cache and a persisted rolling allowance. Health usage is separate from optimization runs. See [the health workflow](prompt-health.md).
- **OpenCode Go compliance.** Requests to Go send the tool's own user agent and one stable `x-opencode-session` value per optimization run.
- **Jev client.** Builds Decisions API requests and parses answers into typed results (`noul` probability; `choice` pick, per-option probabilities and confidence; `score` level, per-level probabilities and confidence). The user's prompt and the model outputs being judged always go in `state`, never in `instructions`, so text being judged can't steer the question. Structured instructions can name state fields but cannot embed their text. The writer can propose Choice descriptions; only descriptions in a success test that passed Jev's faithfulness check can enter `criteria`.
- **Diagnoser.** One batched Jev request per prompt:
  - A task-type taxonomy, classified with tree search over `choice` probabilities (each option's description shows the subtree beneath it). Each leaf carries its own checklist of required pieces.
  - One `noul` per checklist item. A gap counts only when the answer is confidently "missing". Answers near 0.5 are treated as undecided, not as gaps.
  - The prompt is split into sentences with stable IDs, and a `choice` over those IDs picks the sentence behind each problem (pointer pattern, up to 255 options; longer prompts are searched in windows). Sentence-level `noul` questions check vagueness, unresolved references, contradictions and embedded instructions in pasted content.
- **Clarifier.** For each confirmed gap, the writer proposes candidate values. Jev picks among them with an explicit "unknown" option. A confident pick becomes an assumption. An "unknown" becomes a multiple-choice question, but only if the gap is marked high-impact. Questions come back to the caller as a needs-input response.
- **Test compiler.** The writer proposes Jev success criteria. New runs give them stable code-generated IDs and screen each complete test for faithfulness, invented requirements, evaluator-directed instructions, and assessability. Faithfulness and assessability require probability at least 0.8; invented requirements and evaluator-directed instructions require probability at most 0.2. Ambiguous or missing answers discard the test. The approved criterion stays in state, never in application instructions. Only approved Choice and Score descriptions can enter provider criteria. A snapshot-specific cache can reuse an exact approved screen; historical recordings retain their versioned faithfulness request.
  Version 13 also asks Jev for the relationship of each bounded pair in the accepted success-test set: conflict, duplicate, distinct, or unclear. The relation and full probability distribution are recorded as advisory evidence, with conflicts listed before duplicates for triage; they do not remove tests or change grading. At most six accepted tests are compared under a 48 KB serialized request cap, and the report states how many pairs were omitted. Older writer versions retain their exact request sequence for replay.
- **Strategy library.** Named rewrite strategies (for example: add missing context, specify output format, add done criteria, split into steps, add an example, remove contradictions). Each is tagged as safe or as a crutch (known to hurt strong models). Strategies are selected per prompt with rank-then-recheck: first a `choice` across the whole library, then a recheck of the shortlist with one `noul` per strategy.
- **Candidate writer.** Generates K candidates in one structured call, each using a different strategy, editing only the flagged sentences unless the strategy explicitly restructures. It includes previous-round failures when running repeat rounds.
- **Lossless restructuring.** The `restructure_lossless` strategy uses Jev Choices only to assign stable source units to Context, Task, Constraints, Output format, Examples, or Other. Code renders fixed headings and moves each exact source unit once; uncertain roles remain in Other. A deterministic proof checks IDs, source hashes, multiplicity, and the rendered candidate before fidelity. The candidate writer never receives this strategy. Whole-prompt meaning, weak-panel ranking, and the strong check still apply because moving text can change its interpretation. Reports expose the proof result, roles, outcome, and role-assignment cost; the evaluation harness counts generation, wins, and strong regressions with denominators. Missing observed evidence is reported as unavailable.
- **Runner.** Each Round has capacity for six candidates. Generated candidates run on five distinct weak-panel models with three samples per model, at a nonzero temperature so samples differ, with a fixed per-run seed for reproducibility. The original is also measured for comparison. Runs happen in parallel. An earlier routing or writing decision can end a Round before panel requests.
- **Grader.** Grades every (candidate × model × sample) output against the approved tests and computes pass rate per model, worst-model pass rate, mean pass rate and sample spread. New runs send one output and all its test questions in a shared-state Jev request, splitting only that output when question or request-size limits require it. Oversized or partial responses are ungradable and cannot verify a candidate. An unresolved grade for either the original or a candidate makes that candidate ineligible; if the finite confirmation budget leaves no eligible candidate, the Round retains the original rather than claiming an improvement. Each Noul success test uses one direct question; an expected "no" uses the complement of that answer. The reversed-form Noul self-check stays dropped because [Jev 1.13 does not guarantee that separate questions and their negations sum to 1](https://docs.typesafe.ai/model-jaggedness/jev-1.13). Choice and Score option order is governed by a versioned order-bias artifact: `single` asks once, and `mean_pair` averages the two orders after aligning them to semantic options/levels. With no compatible artifact, the grader keeps the existing `legacy_min_pair` rule. The offline `order-bias` experiment compares cross-order changes with same-order repeat variation using paired source-group bootstrap intervals. It recommends `single` only when both excess-effect upper bounds are at most 0.01, `mean_pair` when either lower bound exceeds 0.01, and `insufficient_evidence` otherwise; it requires 30 distinct prompt/output groups. Synthetic-only, incomplete, snapshot-mismatched, or inconclusive evidence cannot activate a policy. The report does not change the production default by itself.
- **Fidelity checker.** Code first checks that edits stay within diagnosed sentences, authorized gap insertions, or explicit restructuring; a confinement failure rejects the candidate without a Jev request. Jev then checks each inserted or changed sentence for support and the whole prompt for preserved meaning. Writer instruction versions 1–3 keep the historical three-question protocol.
- **Selector.** Applies hard eligibility checks before semantic reranking and decides whether the Perfect Prompt Loop continues from this Round's evidence.
- **Run store.** A SQLite-backed repository behind an interface. It logs everything: prompt, options, diagnosis, questions and answers, tests, candidates, outputs, every Jev answer with its question, costs, timings and the user's accept/reject.
- **Self-improving rubric.** An offline job trains a gradient-boosted model (CatBoost) on logged runs. Features come from Jev answers (`noul` probabilities; `score` mean and spread). The label is the original prompt's weak-panel pass rate. A writer model proposes new, revised or dropped Jev questions based on the model's biggest errors (the pattern from TypeSafe's autoresearch cookbook). Adopted questions feed back into the diagnoser. An explicit REWORD workflow proposes at most four equivalent wordings for one active question, screens their meaning with Jev, selects on training groups, recalibrates on a separate partition, and adopts automatically only after a sealed independent final set clears every quantitative gate. Missing evidence returns hold or reject with an audit record; ordinary Optimize calls never start a wording search. See [automatic rewording](reword-workflow.md).

### Selection rules

1. Reject candidates that fail edit confinement, meaning support, success-test requirements, the strong-model gate, or any applicable score-vector floor. A candidate cannot average away a breached floor with stronger scores elsewhere.
2. Apply the requested or Auto-resolved improvement style and route constraints. A proved style/constraint incompatibility produces **impossible** before candidate writing; the current implementation handles an incompatible required exact literal.
3. Rerank only eligible candidates for relevance to the user's request, then select using worst weak-model pass rate, mean pass rate, lower sample spread, and shorter length as tie-breaks. Compare the selected candidate with the original under the same applicable gates.
4. Continue the Perfect Prompt Loop while floors remain unmet or measured mean-vector gain exceeds epsilon. A first-Round gain is `null` because there is no previous vector; it is not a measured zero. A Round that rejects all current candidates may still continue, so it does not by itself finalize the run.

### Round workload and model defaults

Each Round may write up to six candidates. An early routing or writing decision
can end a Round before panel requests. Generated candidates are tested on five
distinct weak models with three samples per model; later eligibility checks
decide which candidates qualify.
There is no fixed total Round count. A per-run time or spend limit pauses at a
Round boundary for approval; approval continues the same run, while cancel or
stop is recorded as control state separately from the run outcome. Provider
failures are operational failures, not a claim about prompt quality.

| Role                           | Default model                                                                                                            | Route                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| Judge (pinned default)         | `typesafe/jev-1.13-20260917`                                                                                             | OpenRouter Decisions API |
| Writer                         | `space-bunny-free`                                                                                                       | OpenCode Go              |
| Strong check                   | `glm-5.3-flash`                                                                                                          | OpenCode Go              |
| Weak panel (five models total) | `meta-llama/llama-3.1-8b-instruct`, `mistralai/mistral-nemo`, `meta-llama/llama-3.2-3b-instruct`, `google/gemma-3-4b-it` | OpenRouter               |
| Weak panel (five models total) | `mimo-v2.6-flash`                                                                                                        | OpenCode Go              |

- Muse Spark 1.3 Contributor and GPT 5.6 Luna remain excluded from defaults.
- GLM 5.3 Flash is the strong reference by design. No frontier-model check runs per prompt or on a schedule.
- Every role except the judge can be changed in the picker. Choices are saved as defaults and can be overridden per run.

### API contract (shape, not final)

- **Optimize:** takes prompt text, a requested improvement style, optional per-role model overrides, a clarification-allowed flag, and optional time/spend limits. Returns a result with one of the five outcomes and an evidence-grounded explanation, or _needs-input_ (run ID and a list of multiple-choice questions, each with options, a pre-selected default and an "other" option).
- **Resume:** takes a run ID and answers, and returns a result.
- **Update assumption:** takes a run ID and an edited assumption, and returns an updated final prompt. It re-checks meaning with Jev but does not re-run the whole loop.
- **Catalog:** returns available models per provider for the picker.
- **Runs:** lists, fetches and searches past runs, and records accept/reject.
- **Settings:** reads and writes the default model per role.

### Configuration

- `OPENROUTER_API_KEY` and `OPENCODE_GO_KEY` come from the engine's environment. The web app never receives keys.

## Testing Decisions

- **What makes a good test:** it drives the engine only through its public entry points (`optimize`, `resume`, update assumption) and checks what comes back: the final prompt, whether the original was kept, needs-input questions, report contents, cost accounting and what the run store recorded. Gateway contract tests may also inspect outgoing requests to verify the boundary between user-controlled state and static instructions. Tests must not depend on internal module structure, prompt templates or the order of internal calls.
- **Single seam:** the engine facade, called in-process. All model traffic goes through the model gateway, which tests replace with (a) a scripted fake returning chosen Jev answers and chat completions, or (b) a replay of recorded real responses keyed by request, like the JSON cache in TypeSafe's cookbooks. No test calls live APIs by default.
- **Engine behavior to cover through the seam:**
  - Even when a prompt has no confirmed gaps, the style-selected rewrite is tested; the result states whether it passed, remained unverified, converged, proved impossible, or failed operationally.
  - Confident gaps produce a rewritten prompt; undecided answers near 0.5 produce no gaps.
  - An unknown, high-impact gap produces needs-input, and resuming with answers completes the run.
  - A candidate that fails the meaning or no-invention check is never selected.
  - A candidate that does worse than the original on the strong check is never selected.
  - A crutch strategy is selected only when it passes the strong check.
  - Ranking follows the ordered rules (worst model first, then mean, then spread, then length).
  - A result names whether the run converged, returned a tested or unverified improvement, found a style/constraint incompatibility, or failed operationally; pause, cancel, and stop remain separate control states.
  - Repeat rounds receive earlier failures.
  - Routing sends Go-listed models to Go (with the session header) and others to OpenRouter.
  - Cost is split correctly by role.
  - The user's prompt text always ends up in `state`, never in `instructions`.
- **HTTP API:** a few contract tests confirming each endpoint maps onto the engine and serializes results and needs-input responses correctly.
- **Web app:** a few end-to-end tests against the HTTP API running with the fake gateway: submit, answer a clarification, edit an assumption chip, copy the result, view the report.
- **Test harness:** a client of the same engine entry point. It runs over the test set (real prompts from WildChat / LMSYS-Chat-1M plus the user's past prompts, stratified by task type; synthetic prompts with planted defects; 100–200 hand-labeled prompts). It reports diagnosis precision and recall per gap type, how often and how much prompts improved, how often they got worse, cost and latency. It can replay recorded responses for reproducible comparisons.
- **Prior art:** none in this repository (it's new). External references: TypeSafe's cookbooks use a recorded-response cache for runnable, no-cost tests; prompt-oscilloscope tests against fake Jev and ACP implementations.

## Out of Scope

- Plugins or hooks for coding agents (Claude Code, Codex, Pi). The tool is standalone.
- A browser extension that attaches to third-party chat boxes (possible later, as another thin client of the API).
- Model-specific output versions. There is exactly one universal prompt per run.
- Accounts, billing, multi-user hosting and bring-your-own-key flows (the design leaves room for these later).
- Per-prompt or scheduled checks against frontier models. GLM 5.3 Flash is the strong reference.
- Optimizing system prompts or templates offline against a dataset. This spec covers optimizing one prompt at a time.
- Running Jev or any judge locally.

## Further Notes

- **Build order:**
  - M0: test set, gateway, Jev client, test harness
  - M1: core loop (diagnose, tests, write, run, grade, fidelity check, select)
  - M2: repeat rounds and clarification
  - M3: web app
  - M4: task taxonomy with tree search, and the strategy library with rank-then-recheck
  - M5: self-improving rubric (trained on run logs gathered from M1 onward)
- **Jev constraints the design depends on** (from TypeSafe's list of Jev 1.13 weaknesses and two independent audits):
  - Jev can't generate text.
  - It reads questions literally.
  - It loses accuracy when the input contains lots of irrelevant material.
  - Content in `state` can steer its answers.
  - Probabilities depend on exact wording, so thresholds can't be reused across differently worded questions.
  - Every `choice` needs an explicit "unknown" or "none" option, because without one the model answers confidently and wrongly.
- **Weak → strong transfer is a floor, not a guarantee.** Prompt preferences are fairly consistent across model sizes (S2LPP, arXiv 2505.20097). But strict rule-based prompting that helps smaller models can hurt the strongest ones ("prompting inversion", arXiv 2510.22251), and explicit step-by-step reasoning can reduce instruction-following (arXiv 2505.11423). This is why the strong check and the crutch rule exist.
- **OpenCode Go usage policy risk (accepted).** Go says it is designed for coding-agent traffic and monitors for abuse. This tool's traffic is not coding-agent traffic, so the account could be flagged or throttled. The user chose Go for every model it offers anyway; the gateway sends the required headers.
- **Cost:** Writer and strong check come out of Go caps; the small weak panel and Jev are paid through OpenRouter. Live benchmark reports provide the measured cost for the selected writer.
- **Prior art studied:** pi-prompt-enhancer (Jev gap checklist, fail-open, confident-gaps-only scoring), prompt-oscilloscope (Jev prompt analysis, debounce, hash caching, local secret checks), mimicry and Lossless Rewrite (LLM rewrites, Jev checks meaning in a bounded loop), and OpenRouter's Jev-verified cascade and "prompt to questions" lab.
- The research note in `docs/research/` from before this interview says the repository was empty and that "Jev" was undefined. This spec supersedes it.
