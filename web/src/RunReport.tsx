import RequirementCoverage from "./RequirementCoverage";
import type { OptimizeResult } from "./api";
import {
  humanize,
  isCanonicalOutcome,
  items,
  outcomeOf,
  plainReason,
  record,
} from "./outcome";
import { STYLE_LABELS, type ImprovementStyle } from "./styles";

function text(value: unknown): string {
  return typeof value === "string"
    ? value
    : value === undefined || value === null
      ? ""
      : String(value);
}

function score(value: unknown): string {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "—";
}

const calibrationDispositions: Record<string, string> = {
  legacy: "Existing policy remains in use",
  gate: "Calibration permits this question to gate decisions",
  "gate-above-confidence":
    "Calibration permits gating after an additional confidence check",
  ranker:
    "Calibration marked this question as ranking-only; it does not gate decisions",
  abstain: "Calibration abstained; no calibrated decision was applied",
};

const calibrationVerdicts: Record<string, string> = {
  gate: "Supports a calibrated gate",
  "gate-above-confidence": "Supports a calibrated gate with a confidence check",
  ranker: "Supports ranking-only use",
  unusable: "Did not meet the requirements for calibrated use",
  "too-few-examples": "Too few examples to establish a policy",
};

const calibrationReasons: Record<string, string> = {
  no_calibration_artifact: "No calibration artifact was available",
  calibration_artifact:
    "The calibration verdict did not clear the requirements for applying a decision",
  calibration_identity_or_snapshot_mismatch:
    "This calibration does not match the current question or model",
  invalid_calibration_threshold:
    "The calibrated confidence threshold is invalid",
  gate_without_threshold: "No confidence threshold was available",
  calibration_event_probability_unavailable:
    "The answer's event probability could not be checked",
  probability_below_calibrated_threshold:
    "The answer's event probability was below the calibrated threshold",
  frozen_calibration_predicate_failed:
    "The answer did not meet the calibrated checks",
  invalid_calibration_predicate: "The calibrated checks could not be applied",
};

const gradingPolicyLabels: Record<string, string> = {
  single: "One option order",
  mean_pair: "Mean across both option orders",
  legacy_min_pair: "Conservative minimum across both option orders",
  noul_direct: "One direct Noul question",
};

const gradingPolicyReasons: Record<string, string> = {
  compatible_order_bias_evidence:
    "A compatible matched experiment supports this policy",
  no_order_bias_artifact: "No order-bias artifact was configured",
  incompatible_snapshot: "The artifact targets a different Jev snapshot",
  synthetic_only_evidence: "Synthetic results cannot activate a runtime policy",
  insufficient_evidence: "The experiment did not establish a policy",
  incomplete_observations: "The experiment has missing or unusable answers",
  no_matching_policy_group: "No policy matches this success test",
  runtime_snapshot_unavailable: "The current Jev snapshot is unavailable",
  outside_order_bias_experiment: "Order bias applies only to Choice and Score",
};

const attributionKinds: Record<string, string> = {
  ignored_constraint: "Ignored constraint",
  misread_instruction: "Misread instruction",
  missing_context_in_prompt: "Missing context in prompt",
  format_not_followed: "Format not followed",
  task_not_attempted: "Task not attempted",
  other: "Other prompt wording issue",
};

const attributionReasons: Record<string, string> = {
  source_backed_hypothesis: "Source backed hypothesis",
  low_confidence_or_unsupported_source: "Uncertain or unsupported source",
  missing_answering_snapshot: "Answering snapshot unavailable",
  incomplete_answer: "Incomplete answer",
  pair_budget_exhausted: "Pair limit reached",
  dollar_budget_exhausted: "Spending limit reached",
  missing_trustworthy_pricing: "Pricing unavailable",
  request_exceeds_provider_limit: "Request size limit reached",
};

function highlightedPrompt(
  prompt: string,
  problems: Record<string, unknown>[]
) {
  const spans = problems
    .map((problem) => record(problem.sentence))
    .map((sentence) => ({
      start: Number(sentence.start),
      end: Number(sentence.end),
    }))
    .filter(
      (span) =>
        Number.isInteger(span.start) &&
        Number.isInteger(span.end) &&
        span.start >= 0 &&
        span.end > span.start &&
        span.end <= prompt.length
    )
    .sort((a, b) => a.start - b.start);
  const parts = [];
  let position = 0;
  for (const span of spans) {
    if (span.start < position) continue;
    parts.push(prompt.slice(position, span.start));
    parts.push(
      <mark key={`${span.start}-${span.end}`}>
        {prompt.slice(span.start, span.end)}
      </mark>
    );
    position = span.end;
  }
  parts.push(prompt.slice(position));
  return parts;
}

const jevCapabilities = [
  "verify",
  "screen",
  "noul",
  "find",
  "rerank",
  "classify",
  "decide",
  "compare",
  "extract",
  "audit",
  "review",
  "gate",
] as const;

const comparisonLabels: Record<string, string> = {
  task_preserved: "Task preserved",
  no_invented_detail: "No invented detail",
  structure_added: "Structure added",
  verbosity_direction: "Verbosity direction",
};

function jsonText(value: unknown): string {
  if (value === undefined || value === null) return "No raw answer recorded";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

function judgmentText(value: unknown): string {
  const judgment = record(value);
  const usable =
    judgment.usable === true
      ? "Usable"
      : judgment.usable === false
        ? "Not usable"
        : "Usability not recorded";
  const probability =
    typeof judgment.probability === "number"
      ? `; probability ${score(judgment.probability)}`
      : "";
  const rawAnswer = record(judgment.raw_answer);
  const selected =
    judgment.usable === true
      ? typeof judgment.selected === "string"
        ? judgment.selected
        : typeof rawAnswer.choice === "string"
          ? rawAnswer.choice
          : null
      : null;
  return `${usable}${selected ? `; selected: ${selected}` : ""}${probability}`;
}

function judgmentList(
  value: unknown,
  labels?: Record<string, string>
): Record<string, unknown>[] {
  return Object.entries(record(value)).map(([key, judgment]) => ({
    key,
    label: labels?.[key] ?? humanize(key),
    judgment,
  }));
}

function evidenceDetails(label: string, value: unknown) {
  if (value === undefined || value === null) return null;
  return (
    <details>
      <summary>{label}</summary>
      <pre>{jsonText(value)}</pre>
    </details>
  );
}

export default function RunReport({ result }: { result: OptimizeResult }) {
  const report = result.report;
  const evaluationEvidence = record(report.evaluation_evidence);
  const evaluationCandidates = record(evaluationEvidence.candidates);
  const judgmentProvenance = items(report.judgment_provenance);
  const capabilitiesFired = record(report.capabilities_fired);
  const hasEvaluationEvidence =
    Object.keys(evaluationEvidence).length > 0 ||
    judgmentProvenance.length > 0 ||
    Object.keys(capabilitiesFired).length > 0;
  const diagnosis = record(report.diagnosis);
  const diagnosisRequests = record(diagnosis.request_evidence);
  const calibration = record(diagnosis.calibration);
  const problems = items(diagnosis.problem_sentences);
  const gaps = items(diagnosis.confirmed_gaps);
  const tests = items(report.tests);
  const testScreening = record(report.test_screening);
  const screenChecks = items(testScreening.screening_checks);
  const screenObservation = record(testScreening.screening_observation);
  const gradingObservation = record(report.grading_observation);
  const outputScreen = items(report.output_screen);
  const gradingCascade = record(report.grading_cascade);
  const cascadePairs = items(gradingCascade.pairs);
  const roundAttributions = items(report.history)
    .map((round) => record(record(round.evidence).failure_attribution))
    .filter((entry) => Object.keys(entry).length > 0);
  const attributions =
    roundAttributions.length > 0
      ? roundAttributions
      : Object.keys(record(report.failure_attribution)).length > 0
        ? [record(report.failure_attribution)]
        : [];
  const attributionPairs = attributions.flatMap((entry) => items(entry.pairs));
  const attributionCount = (name: string) =>
    attributions.reduce((total, entry) => total + Number(entry[name] ?? 0), 0);
  const gradingPolicies = items(report.grading_policy);
  const selection = record(report.selection_evidence);
  const originalScore = record(selection.original_score);
  const winnerScore = record(
    selection.winner_score ??
      (selection.original_kept || result.original_kept
        ? selection.original_score
        : null)
  );
  const originalRates = record(originalScore.per_model);
  const winnerRates = record(winnerScore.per_model);
  const strong = record(report.strong_check);
  const strongCandidates = items(strong.candidates);
  const rejected = items(selection.rejected_candidates);
  const restructuring = record(report.lossless_restructuring);
  const preservation = record(restructuring.source_preservation);
  const restructureRoles = items(restructuring.roles);
  const unknownRoles = Array.isArray(restructuring.unknowns)
    ? restructuring.unknowns.map(String)
    : [];
  const restructureCost = record(restructuring.cost);
  const costByRole = record(result.cost.cost_by_role);
  const goUsage = Object.values(result.cost.by_role ?? {})
    .flat()
    .filter(
      (entry) =>
        entry.provider === "go" &&
        typeof entry.cap === "number" &&
        entry.cap > 0
    );
  const originalPrompt = result.original_prompt ?? "";
  const diff = text(report.diff);
  const appliedStyle = text(report.applied_style);
  const appliedLabel = Object.hasOwn(STYLE_LABELS, appliedStyle)
    ? STYLE_LABELS[appliedStyle as ImprovementStyle]
    : appliedStyle;
  const canonicalOutcome = isCanonicalOutcome(report.outcome)
    ? outcomeOf(result)
    : null;

  return (
    <div className="report-content">
      <p>{text(report.summary)}</p>
      <RequirementCoverage
        history={[
          ...items(report.history),
          ...Object.values(record(report.requirement_checks)).map((value) => {
            const check = record(value);
            return {
              round_number: check.round,
              evidence: {
                selection_evidence: {
                  ranking: [
                    {
                      candidate_id: check.candidate_id,
                      text: check.draft,
                      metadata: { requirement_findings: check.findings },
                    },
                  ],
                },
              },
            };
          }),
        ]}
        ledger={record(report.requirements)}
        selection={selection}
      />
      {canonicalOutcome && (
        <section aria-label="Run outcome">
          <p>
            <strong>Outcome:</strong> {canonicalOutcome.headline}
          </p>
          {canonicalOutcome.reason && (
            <p className="outcome-reason">{canonicalOutcome.reason}</p>
          )}
          {canonicalOutcome.controlState && (
            <p>Run state: {canonicalOutcome.controlState}.</p>
          )}
        </section>
      )}
      {appliedStyle && (
        <p>
          Applied style: {appliedLabel}
          {report.improvement_style === "auto" &&
            " (inferred — your style was Auto)"}
          .
        </p>
      )}
      {originalPrompt && (
        <section>
          <h3>Original prompt</h3>
          <p className="original-prompt">
            {highlightedPrompt(originalPrompt, problems)}
          </p>
        </section>
      )}
      {hasEvaluationEvidence && (
        <section aria-labelledby="evaluation-acceptance-heading">
          <h3 id="evaluation-acceptance-heading">Evaluation and acceptance</h3>
          {Object.keys(evaluationCandidates).length > 0 && (
            <ul>
              {Object.entries(evaluationCandidates).map(([key, value]) => {
                const candidate = record(value);
                const candidateId = text(candidate.candidate_id || key);
                const acceptance = record(candidate.accept);
                const eligible =
                  candidate.eligible === true
                    ? "Eligible"
                    : candidate.eligible === false
                      ? "Not eligible"
                      : "Eligibility not recorded";
                const accepted =
                  acceptance.accepted === true
                    ? "Accepted by the acceptance check"
                    : acceptance.accepted === false
                      ? "Rejected by the acceptance check"
                      : "Acceptance check not recorded";
                const rejectionReasons = Array.isArray(
                  candidate.rejection_reasons
                )
                  ? candidate.rejection_reasons.map(String)
                  : [];
                const comparisons = judgmentList(
                  candidate.comparison,
                  comparisonLabels
                );
                const verification = judgmentList(candidate.verification);
                const audit = judgmentList(candidate.audit);

                return (
                  <li key={candidateId}>
                    <strong>{candidateId}</strong> — round{" "}
                    {typeof candidate.round_number === "number"
                      ? candidate.round_number
                      : "not recorded"}
                    . {eligible}. {accepted}
                    {typeof acceptance.probability === "number" && (
                      <>
                        : {score(acceptance.probability)} probability against a{" "}
                        {score(acceptance.threshold)} threshold
                      </>
                    )}
                    .
                    {rejectionReasons.length > 0 && (
                      <>
                        <p>Reasons: {rejectionReasons.join("; ")}</p>
                      </>
                    )}
                    {rejectionReasons.length === 0 &&
                      candidate.eligible === true && (
                        <p>No eligibility rejection reasons recorded.</p>
                      )}
                    {comparisons.length > 0 && (
                      <>
                        <h4>Comparisons</h4>
                        <ul>
                          {comparisons.map((item) => (
                            <li key={String(item.key)}>
                              {text(item.label)}: {judgmentText(item.judgment)}
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                    {verification.length > 0 && (
                      <>
                        <h4>Verification</h4>
                        <ul>
                          {verification.map((item) => (
                            <li key={String(item.key)}>
                              {text(item.label)}: {judgmentText(item.judgment)}
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                    {audit.length > 0 && (
                      <>
                        <h4>Audit</h4>
                        <ul>
                          {audit.map((item) => (
                            <li key={String(item.key)}>
                              {text(item.label)}: {judgmentText(item.judgment)}
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                    <p>
                      Rerank: {judgmentText(candidate.rerank)}. Review:{" "}
                      {judgmentText(candidate.review)}.
                    </p>
                    {evidenceDetails(
                      "Score vector evidence",
                      candidate.score_vector
                    )}
                    {evidenceDetails(
                      "Success-test grade evidence",
                      candidate.success_test_grade
                    )}
                    <details>
                      <summary>Acceptance answer</summary>
                      <p>{judgmentText(acceptance)}</p>
                      <pre>{jsonText(acceptance.raw_answer)}</pre>
                    </details>
                  </li>
                );
              })}
            </ul>
          )}
          {Object.keys(capabilitiesFired).length > 0 && (
            <>
              <h4>Jev capability calls</h4>
              <ul>
                {jevCapabilities.map((capability) => {
                  const evidence = record(capabilitiesFired[capability]);
                  if (!Object.keys(evidence).length) {
                    return (
                      <li key={capability}>{capability}: count not recorded</li>
                    );
                  }
                  const stages = Object.entries(record(evidence.stages))
                    .map(
                      ([stage, count]) => `${humanize(stage)} ${text(count)}`
                    )
                    .join(", ");
                  return (
                    <li key={capability}>
                      {capability}: {text(evidence.count)} times;{" "}
                      {evidence.ran === true
                        ? "ran"
                        : evidence.ran === false
                          ? "did not run"
                          : "run status not recorded"}
                      {stages && `; stages: ${stages}`}.
                    </li>
                  );
                })}
              </ul>
            </>
          )}
          {judgmentProvenance.length > 0 && (
            <details>
              <summary>
                Judgment provenance ({judgmentProvenance.length})
              </summary>
              <ul>
                {judgmentProvenance.map((item, index) => {
                  const capability = text(item.capability) || "not recorded";
                  const candidateId =
                    typeof item.candidate_id === "string"
                      ? item.candidate_id
                      : "run-level";
                  const round =
                    typeof item.round_number === "number"
                      ? `current round ${item.round_number}`
                      : "current round not recorded";
                  const sourceRound =
                    typeof item.source_round === "number"
                      ? `, source round ${item.source_round}`
                      : ", source round not recorded";
                  return (
                    <li key={`${text(item.question_key)}-${index}`}>
                      <strong>{capability}</strong> /{" "}
                      {text(item.stage) || "stage not recorded"}: {candidateId};{" "}
                      {round}
                      {sourceRound}; question{" "}
                      {text(item.question_key) || "not recorded"}; model{" "}
                      {text(item.model) || "not recorded"};{" "}
                      {item.usable === true
                        ? "usable"
                        : item.usable === false
                          ? "not usable"
                          : "usability not recorded"}
                      {typeof item.probability === "number" &&
                        `; probability ${score(item.probability)}`}
                      <details>
                        <summary>Raw answer</summary>
                        <pre>{jsonText(item.raw_answer)}</pre>
                      </details>
                    </li>
                  );
                })}
              </ul>
            </details>
          )}
        </section>
      )}
      <section>
        <h3>Diagnosis</h3>
        <p>
          Task type:{" "}
          {text(diagnosis.task_type_label || diagnosis.task_type) ||
            "undetermined"}
        </p>
        {Object.keys(diagnosisRequests).length > 0 && (
          <p>
            Diagnosis requests: {text(diagnosisRequests.provider_requests ?? 0)}
            .
            {diagnosisRequests.mode === "bounded_sequential_fallback" &&
              " Bounded sequential fallback was used."}
            {diagnosisRequests.complete === false &&
              " Diagnosis evidence is incomplete; the original prompt was kept."}
          </p>
        )}
        {gaps.length > 0 ? (
          <ul>
            {gaps.map((gap, index) => (
              <li key={text(gap.key) || index}>{text(gap.label || gap.key)}</li>
            ))}
          </ul>
        ) : diagnosisRequests.complete === false ? null : (
          <p>No confirmed missing pieces.</p>
        )}
        {problems.length > 0 && (
          <ul>
            {problems.map((problem, index) => (
              <li key={index}>
                {humanize(text(problem.kind))}:{" "}
                {text(record(problem.sentence).text)}
              </li>
            ))}
          </ul>
        )}
      </section>
      {Object.keys(calibration).length > 0 && (
        <section aria-labelledby="calibration-decisions-heading">
          <h3 id="calibration-decisions-heading">Calibration decisions</h3>
          <ul>
            {Object.entries(calibration).map(([questionId, value]) => {
              const evidence = record(value);
              const disposition = text(evidence.disposition);
              const verdict = text(evidence.verdict);
              const reason = text(evidence.reason);
              const threshold =
                typeof evidence.threshold === "number"
                  ? score(evidence.threshold)
                  : "";

              return (
                <li key={questionId}>
                  <strong>Question {questionId}:</strong>{" "}
                  {calibrationDispositions[disposition] ??
                    (disposition
                      ? humanize(disposition)
                      : "Status unavailable")}
                  {verdict && (
                    <>
                      . Verdict:{" "}
                      {calibrationVerdicts[verdict] ?? humanize(verdict)}
                    </>
                  )}
                  {threshold && <>. Minimum event probability: {threshold}</>}
                  {reason &&
                    (disposition === "abstain" || disposition === "legacy") && (
                      <>
                        . Reason:{" "}
                        {calibrationReasons[reason] ?? humanize(reason)}
                      </>
                    )}
                  .
                </li>
              );
            })}
          </ul>
        </section>
      )}
      <section>
        <h3>Success tests</h3>
        {tests.length > 0 ? (
          <ul>
            {tests.map((test, index) => (
              <li key={text(test.id) || index}>{text(test.question)}</li>
            ))}
          </ul>
        ) : (
          <p>No faithful success tests were available for this run.</p>
        )}
      </section>
      {Object.keys(testScreening).length > 0 && (
        <section aria-labelledby="test-screening-heading">
          <h3 id="test-screening-heading">Success test screening</h3>
          <p>
            Approved: {text(screenObservation.approved_count ?? tests.length)}.
            Discarded: {text(screenObservation.discarded_count ?? 0)}.
          </p>
          {screenChecks.some((check) => check.accepted === false) && (
            <ul>
              {screenChecks
                .filter((check) => check.accepted === false)
                .map((check, index) => (
                  <li key={text(check.test_id) || index}>
                    {text(check.test_id)}: {humanize(text(check.reason))}
                  </li>
                ))}
            </ul>
          )}
          <p>
            Screening requests:{" "}
            {text(screenObservation.gateway_batch_calls ?? 0)}. Measured judge
            cost:{" "}
            {typeof screenObservation.judge_cost_usd_measured === "number"
              ? `$${screenObservation.judge_cost_usd_measured.toFixed(4)}`
              : "unavailable"}
            .
          </p>
        </section>
      )}
      {Object.keys(gradingObservation).length > 0 && (
        <section aria-labelledby="grading-observation-heading">
          <h3 id="grading-observation-heading">Grading requests</h3>
          <p>
            {text(gradingObservation.gateway_batch_calls ?? 0)} requests for{" "}
            {text(gradingObservation.graded_output_count ?? 0)} outputs.
            Estimated serialized input:{" "}
            {text(gradingObservation.serialized_input_bytes_estimate ?? 0)}{" "}
            bytes.
          </p>
          {Number(gradingObservation.ungradable_output_count ?? 0) > 0 && (
            <p>
              Ungradable outputs:{" "}
              {text(gradingObservation.ungradable_output_count)}. No verified
              improvement was selected from incomplete grading.
            </p>
          )}
          <p>
            Measured judge cost:{" "}
            {typeof gradingObservation.judge_cost_usd_measured === "number"
              ? `$${gradingObservation.judge_cost_usd_measured.toFixed(4)}`
              : "unavailable"}
            .
          </p>
        </section>
      )}
      {outputScreen.length > 0 && (
        <section aria-labelledby="output-screen-heading">
          <h3 id="output-screen-heading">Output screen</h3>
          <p>
            Each weak-panel output was checked for instructions aimed at the
            evaluator. Detected outputs score zero; unresolved screens keep
            their ordinary score for audit but cannot verify an improvement.
          </p>
          <ul>
            {outputScreen.map((item, index) => {
              const entry = record(item);
              return (
                <li
                  key={`${text(entry.candidate_id)}-${text(entry.model)}-${text(entry.sample)}-${index}`}
                >
                  {text(entry.candidate_id)} on {text(entry.model)} sample{" "}
                  {text(entry.sample)}:{" "}
                  {humanize(text(entry.reason || entry.status))}
                  {entry.status === "screen_unresolved" &&
                    " (screen unresolved)"}
                  .
                </li>
              );
            })}
          </ul>
        </section>
      )}
      {outputScreen.length === 0 &&
        Object.keys(gradingObservation).length > 0 &&
        gradingObservation.protocol !== "single_output_screened_v2" && (
          <p>Output screen unavailable for this historical run.</p>
        )}
      {Object.keys(gradingCascade).length > 0 && (
        <section aria-labelledby="grade-confirmation-heading">
          <h3 id="grade-confirmation-heading">Grade confirmation</h3>
          <p>
            {text(gradingCascade.confirmation_count ?? 0)} confirmation
            requests, {text(gradingCascade.escalation_count ?? 0)} strong-model
            evidence calls, {text(gradingCascade.verification_count ?? 0)} Jev
            verification requests; {text(gradingCascade.unresolved_count ?? 0)}{" "}
            unresolved pairs.
          </p>
          <p>
            Reserved cascade cost:{" "}
            {typeof gradingCascade.reserved_cost_usd === "number" &&
            Number.isFinite(gradingCascade.reserved_cost_usd)
              ? `$${gradingCascade.reserved_cost_usd.toFixed(4)}`
              : "unavailable"}{" "}
            of{" "}
            {typeof gradingCascade.dollar_cap === "number" &&
            Number.isFinite(gradingCascade.dollar_cap)
              ? `$${gradingCascade.dollar_cap.toFixed(4)}`
              : "unavailable"}
            .
          </p>
          {cascadePairs.length > 0 && (
            <ul>
              {cascadePairs.map((item, index) => {
                const pair = record(item);
                return (
                  <li key={`${text(pair.pair_id)}-${index}`}>
                    {text(pair.candidate_id)} test {text(pair.test_id)}:{" "}
                    {humanize(text(pair.reason || pair.status))}.
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      )}
      {attributions.length > 0 && (
        <section aria-labelledby="failure-attribution-heading">
          <h3 id="failure-attribution-heading">Failure attribution</h3>
          <p>
            {attributionCount("attributed_count")} supported hypotheses,{" "}
            {attributionCount("unresolved_count")} unresolved pairs,{" "}
            {attributionCount("skipped_count")} skipped pairs.
          </p>
          {attributionPairs.length > 0 && (
            <ul>
              {attributionPairs.map((item, index) => {
                const pair = record(item);
                const supported = pair.status === "supported";
                const reason = text(pair.reason);
                return (
                  <li key={`${text(pair.pair_id)}-${index}`}>
                    {text(pair.candidate_id)} on {text(pair.model)} sample{" "}
                    {text(pair.sample)}, test {text(pair.test_id)}:{" "}
                    {supported
                      ? "Hypothesis"
                      : pair.status === "unresolved"
                        ? "Unresolved"
                        : "Skipped"}
                    {supported && (
                      <>
                        {" "}
                        — {text(pair.sentence_id)} “{text(pair.sentence_text)}”
                        (
                        {attributionKinds[text(pair.kind)] ??
                          "Other prompt wording issue"}
                        )
                      </>
                    )}
                    {!supported && reason && (
                      <>
                        {" "}
                        —{" "}
                        {attributionReasons[reason] ??
                          (reason.startsWith("provider_")
                            ? "Provider unavailable"
                            : "Uncertain attribution")}
                      </>
                    )}
                    .
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      )}
      {gradingPolicies.length > 0 && (
        <section aria-labelledby="grading-policy-heading">
          <h3 id="grading-policy-heading">Grading policy</h3>
          <ul>
            {gradingPolicies.map((policy, index) => {
              const policyName = text(policy.policy);
              const reason = text(policy.reason);
              const question = text(policy.question);
              const snapshot = text(policy.snapshot);
              return (
                <li key={`${text(policy.test_id) || index}-${policyName}`}>
                  {question && <strong>{question}: </strong>}
                  {gradingPolicyLabels[policyName] ??
                    (policyName ? humanize(policyName) : "Status unavailable")}
                  {reason && (
                    <>. {gradingPolicyReasons[reason] ?? humanize(reason)}</>
                  )}
                  {snapshot && <>. Jev snapshot: {snapshot}</>}.
                </li>
              );
            })}
          </ul>
        </section>
      )}
      {Object.keys(restructuring).length > 0 && (
        <section aria-labelledby="lossless-restructuring-heading">
          <h3 id="lossless-restructuring-heading">Lossless restructuring</h3>
          <p>
            Source preservation:{" "}
            {humanize(text(preservation.status) || "unavailable")}. Outcome:{" "}
            {humanize(
              text(restructuring.selection_outcome || restructuring.outcome)
            )}
            .
          </p>
          {Boolean(restructuring.decline_reason) && (
            <p>Reason: {text(restructuring.decline_reason)}</p>
          )}
          {restructureRoles.length > 0 && (
            <>
              <h4>Source unit roles</h4>
              <ul>
                {restructureRoles.map((role, index) => (
                  <li key={text(role.unit_id) || index}>
                    {text(role.unit_id)}: {humanize(text(role.role))}
                    {typeof role.confidence === "number" &&
                      ` (${score(role.confidence)} confidence)`}
                  </li>
                ))}
              </ul>
            </>
          )}
          {unknownRoles.length > 0 && (
            <p>
              Uncertain source units retained in Other:{" "}
              {unknownRoles.join(", ")}.
            </p>
          )}
          <p>
            Role assignment requests:{" "}
            {text(restructuring.role_assignment_requests || 0)}. Reported role
            assignment cost:{" "}
            {restructureCost.status === "reported"
              ? `$${Object.values(record(restructureCost.cost_by_role))
                  .reduce<number>(
                    (total, value) =>
                      total + (typeof value === "number" ? value : 0),
                    0
                  )
                  .toFixed(4)}`
              : "unavailable"}
            .
          </p>
        </section>
      )}
      {Object.keys(originalRates).length > 0 && (
        <section>
          <h3>Pass rates on test models</h3>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Original</th>
                <th>Selected</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(originalRates).map((model) => (
                <tr key={model}>
                  <td>{model}</td>
                  <td>{score(originalRates[model])}</td>
                  <td>{score(winnerRates[model])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>
            Variation between samples: original {score(originalScore.spread)},
            selected {score(winnerScore.spread)}
          </p>
        </section>
      )}
      {Object.keys(strong).length > 0 && (
        <section>
          <h3>Strong check</h3>
          <p>Original: {score(strong.original_score)}</p>
          <ul>
            {strongCandidates.map((candidate, index) => (
              <li key={text(candidate.candidate_id) || index}>
                {humanize(text(candidate.candidate_id))}:{" "}
                {score(candidate.candidate_score)} — {text(candidate.reason)}
              </li>
            ))}
          </ul>
        </section>
      )}
      {rejected.length > 0 && (
        <section>
          <h3>Rewrites that were not used</h3>
          <ul>
            {rejected.map((candidate, index) => (
              <li key={text(candidate.candidate_id) || index}>
                {humanize(text(candidate.strategy || candidate.candidate_id))}:{" "}
                {Array.isArray(candidate.rejection_reasons) &&
                candidate.rejection_reasons.length > 0
                  ? candidate.rejection_reasons
                      .map((reason) => plainReason(String(reason)))
                      .join("; ")
                  : "scored below the chosen prompt"}
              </li>
            ))}
          </ul>
        </section>
      )}
      {diff && (
        <section>
          <h3>Changes from original</h3>
          <pre>{diff}</pre>
        </section>
      )}
      <section>
        <h3>Cost</h3>
        <p>Total: ${result.cost.total.toFixed(4)}</p>
        {Object.keys(costByRole).length > 0 && (
          <ul>
            {Object.entries(costByRole).map(([role, value]) => (
              <li key={role}>
                {humanize(role)}: ${Number(value).toFixed(4)}
              </li>
            ))}
          </ul>
        )}
        {goUsage.length > 0 && (
          <>
            <h4>OpenCode Go model caps used by this run</h4>
            <ul>
              {goUsage.map((entry, index) => (
                <li key={`${entry.model ?? "go"}-${index}`}>
                  {entry.model}: $
                  {(entry.cap_used ?? entry.cost ?? 0).toFixed(4)} of $
                  {entry.cap!.toFixed(2)} cap (
                  {(
                    ((entry.cap_used ?? entry.cost ?? 0) / entry.cap!) *
                    100
                  ).toFixed(2)}
                  %)
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
      {Array.isArray(report.history) && report.history.length > 0 && (
        <section>
          <h3>Rounds</h3>
          <p>
            {report.history.length} completed round
            {report.history.length === 1 ? "" : "s"}
          </p>
        </section>
      )}
    </div>
  );
}
