# Prompt Improvement

This context tests whether a prompt can be improved and decides which version to return.

## Language

**Round**:
One bounded evaluation with capacity for up to six candidate rewrites. An
early routing or writing decision can end a Round before panel requests.
Generated candidates run on five distinct weak models with three samples per
model; fidelity, score-floor, and acceptance checks then decide which qualify.
A Round records the selected candidate and score evidence, or why no changed
candidate qualified.
_Avoid_: iteration, attempt count

**Improvement style**:
The target the user chooses for a rewrite, such as Clearer or Shorter. Auto is a request, not a style applied to the prompt: the Understand stage resolves it to a named style from the task, with Clearer as the conservative fallback when the inference is invalid or uncertain. The report keeps both the requested and applied style.
_Avoid_: mode, tone (unless describing only tone/voice)

**Quality dimension**:
One of the six independently scored properties of a candidate: fidelity, style fit, clarity, specificity, coherence, or safety. Fidelity is also enforced by the existing meaning and edit-confinement checks. A candidate must meet every applicable dimension's floor; a strong score in one dimension cannot erase a breach in another.
_Avoid_: aggregate quality score

**Floor**:
The minimum accepted score for one Quality dimension. Floors are recorded with each round's vector and may be explicitly recalibrated from linked keep/reject feedback. A floor is a gate, not a target averaged against other dimensions.
_Avoid_: average threshold

**Convergence**:
The successful stop when an accepted, useful changed prompt's own score vector meets every dimension floor and the mean score gain over the previous Round is at or below the configured epsilon (0.01 by default). The first Round has no previous vector, so its gain is `null`, not zero or a measured plateau; if that first vector meets every floor, the current policy still permits immediate convergence on the floor evidence.
_Avoid_: stalled, exhausted attempts

**Perfect Prompt Loop**:
The sequence of bounded Rounds within one 150-second active processing allowance. It carries losing-candidate evidence into the next Round and continues while a floor is unmet or measured gain remains above epsilon. It may finish at Convergence, a proven task conflict that needs the user's choice, or an operational failure. An exact-output instruction permits faithful edits to its surrounding presentation; rejecting a strategy bundle does not prove the task impossible. A configured time or spend limit pauses at a Round boundary for approval. The active deadline is a terminal control stop, separate from quality outcomes; without a qualified changed prompt it establishes no improvement. Clarification replies resume the same identity with accumulated active time, excluding the human reply delay. Finalizing a user stop preserves a previously accepted changed prompt only when that prompt's own evidence supports an improved outcome.

**Rejection cause label**:
A Jev suggestion for why a user rejected a completed prompt result, inferred from the user's optional note and retained for evaluation.
_Avoid_: ground-truth label, Round signal

**Improvement not verified**:
An internal or historical description that the run has not established a tested improvement; it is not one of the five canonical outcomes. **Improved (unverified)** requires an actually accepted changed prompt that passed meaning and safety checks when usable success-test evidence was unavailable. A proven task incompatibility may be **impossible**; exact-output instructions alone do not establish impossibility. A technical or provider failure is **failed (operational)**. A Round that rejects every current candidate may still continue, so it does not by itself determine the final outcome.
_Avoid_: silent success

## Policy

A run reports one of five outcomes: **converged**, **improved (tested)**,
**improved (unverified)**, **impossible**, or **failed (operational)**. Each
result names the applied style when known and gives a brief explanation tied
to the run evidence. An unverified result is an improved outcome only when a
changed candidate was actually accepted after meaning and safety checks; the
explanation must say that answer quality was not tested. Exact-output instructions permit faithful surrounding instruction edits and do not by themselves establish impossibility. A failed
operational result explains why the run could not produce a final accepted
outcome.

Pause, cancellation, active deadline, and user stop are control states, not extra outcome
categories. If the user stops after a changed candidate was accepted, the run
may retain it as **improved (tested)** or **improved (unverified)** only when
its own evidence supports that outcome; a stop alone never creates an
improvement. A `no_qualified_candidate` Round may be healthy continuing
evidence, so it must not be promoted to the final outcome prematurely.

The score vector is round-local evidence. A first-round `gain` is `null` because
there is no prior vector to compare; it is never described as a measured zero.

**Success-test set**:
The accepted checks used together to judge answers in a Round. Each check addresses an observable outcome of the user's request; two checks may still overlap or conflict when considered together.
