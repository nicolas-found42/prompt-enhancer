import { humanize, items, record } from "./outcome";

function text(value: unknown): string {
  return typeof value === "string"
    ? value
    : value === undefined || value === null
      ? ""
      : String(value);
}

function requirementCheckSummary(findings: Record<string, unknown>[]): string {
  const count = (status: string) =>
    findings.filter((item) => item.status === status).length;
  const label = (amount: number, type: string) =>
    `${amount} ${type} check${amount === 1 ? "" : "s"}`;
  return [
    label(count("tested"), "passed"),
    label(count("failed"), "failed"),
    label(count("untestable"), "untestable"),
    label(count("unresolved"), "unresolved"),
  ].join(", ");
}

export function coverageHistory(
  history: Record<string, unknown>[],
  checkpoints: Record<string, unknown>
): Record<string, unknown>[] {
  const rounds = new Map<number, Record<string, unknown>>();
  for (const round of history) {
    rounds.set(Number(round.round_number), round);
  }
  for (const value of Object.values(checkpoints)) {
    const check = record(value);
    const number = Number(check.round);
    if (!Number.isFinite(number)) continue;
    const existing = rounds.get(number);
    // A completed round already owns its final findings and selection outcome.
    if (existing && existing.unfinished !== true) continue;
    const ranking = items(
      record(record(existing?.evidence).selection_evidence).ranking
    );
    ranking.push({
      candidate_id: check.candidate_id,
      text: check.draft,
      status: "checks unfinished",
      metadata: { requirement_findings: check.findings },
    });
    rounds.set(number, {
      round_number: number,
      unfinished: true,
      evidence: { selection_evidence: { ranking } },
    });
  }
  return [...rounds.entries()]
    .sort(([a], [b]) => a - b)
    .map(([, round]) => round);
}

export default function RequirementCoverage({
  ledger,
  selection,
  history = [],
  live = false,
}: {
  ledger: Record<string, unknown>;
  selection: Record<string, unknown>;
  history?: Record<string, unknown>[];
  live?: boolean;
}) {
  const requirements = items(ledger.requirements);
  if (Object.keys(ledger).length === 0) return null;
  const selected = record(selection.selected_candidate);
  const findings = items(record(selected.metadata).requirement_findings);
  const rejected = items(selection.ranking).filter(
    (item) =>
      item.candidate_id !== selected.candidate_id ||
      item.round !== selected.round
  );
  return (
    <section aria-label="Requirement coverage">
      <h3>Requirement coverage</h3>
      <p>
        {ledger.coverage === "partial"
          ? "Coverage is partial. "
          : ledger.coverage === "audited"
            ? "Coverage was audited. "
            : ""}
        {text(ledger.reason)}
      </p>
      {items(ledger.contradictions).map((conflict, index) => (
        <p key={index}>
          {conflict.status === "resolved_by_user"
            ? conflict.selected_count
              ? `Your answer resolved the conflicting counts: use exactly ${text(conflict.selected_count)}.`
              : "Your answer resolved these conflicting requirements."
            : "Conflicting requirements still need your answer."}{" "}
          {text(conflict.reason)}
        </p>
      ))}
      {record(ledger.whole_source_audit).status === "unresolved" && (
        <p>
          The complete request has not yet passed its separate coverage audit.
        </p>
      )}
      {items(ledger.gaps).map((gap, index) => (
        <p key={`gap-${index}`}>{text(gap.reason)}</p>
      ))}
      <ul>
        {requirements.map((requirement, index) => {
          const own = findings.filter(
            (item) => item.requirement_id === requirement.id
          );
          const reasons = [
            ...new Set(own.map((item) => text(item.reason)).filter(Boolean)),
          ];
          return (
            <li key={text(requirement.id) || index}>
              <p>{text(requirement.source)}</p>
              <p>
                {requirement.source_kind === "user_answer"
                  ? "From your answer. "
                  : "From your original prompt. "}
                {requirement.scope === "candidate_prompt"
                  ? "Applies to the rewritten prompt."
                  : text(requirement.scope).startsWith("section:")
                    ? `Applies to the ${text(requirement.scope).slice(8)} section.`
                    : "Applies to the complete answer."}
              </p>
              {Array.isArray(requirement.protected_values) &&
                requirement.protected_values.length > 0 && (
                  <p>
                    Protected values:{" "}
                    {requirement.protected_values.map(text).join(", ")}
                  </p>
                )}
              {record(requirement.audit).status === "accepted" && (
                <p>Source interpretation audit passed.</p>
              )}
              {record(requirement.audit).status === "unresolved" && (
                <p>
                  This source interpretation remains uncertain; recognized
                  requirements are retained.
                </p>
              )}
              {requirement.superseded_by != null && (
                <p>
                  Superseded by your explicit choice; the original requirement
                  remains in this history.
                </p>
              )}
              {Boolean(record(requirement.user_answer).value) && (
                <p>
                  Your answer: {text(record(requirement.user_answer).value)}
                </p>
              )}
              {own.length > 0 ? (
                <>
                  <p>Selected draft: {requirementCheckSummary(own)}.</p>
                  {reasons.map((reason) => (
                    <p key={reason}>{reason}</p>
                  ))}
                </>
              ) : (
                <p>
                  {live
                    ? "Check updates for this obligation appear in Run activity."
                    : "No check result for the selected draft was retained for this source requirement."}
                </p>
              )}
            </li>
          );
        })}
      </ul>
      {history.map((round, index) => {
        const ranking = items(
          record(record(round.evidence).selection_evidence).ranking
        );
        if (
          !ranking.some(
            (candidate) =>
              items(record(candidate.metadata).requirement_findings).length > 0
          )
        )
          return null;
        return (
          <details key={index}>
            <summary>
              Round {text(round.round_number) || index + 1}: source requirement
              findings
            </summary>
            {ranking.map((candidate) => (
              <details key={text(candidate.candidate_id)}>
                <summary>
                  Draft {text(candidate.candidate_id)} —{" "}
                  {candidate.selected
                    ? "qualified"
                    : text(candidate.status) || "did not qualify"}
                </summary>
                <p className="preserve-lines">{text(candidate.text)}</p>
                <ul>
                  {items(record(candidate.metadata).requirement_findings).map(
                    (finding, findingIndex) => (
                      <li key={findingIndex}>
                        <p>
                          {text(finding.source)} —{" "}
                          {finding.status === "tested"
                            ? "Passed"
                            : humanize(text(finding.status))}
                          . {text(finding.reason)}
                        </p>
                        <p>
                          Check: {humanize(text(finding.check))}
                          {finding.model
                            ? ` · model ${text(finding.model)} · sample ${text(finding.sample)}`
                            : " · rewritten prompt"}
                          .
                        </p>
                      </li>
                    )
                  )}
                </ul>
              </details>
            ))}
          </details>
        );
      })}
      {rejected.some(
        (candidate) =>
          items(record(candidate.metadata).requirement_findings).length > 0
      ) && (
        <details>
          <summary>
            {live
              ? "Draft checks so far"
              : "Checks for drafts that did not qualify"}
          </summary>
          {rejected.map((candidate, index) => {
            const checks = items(
              record(candidate.metadata).requirement_findings
            );
            if (checks.length === 0) return null;
            return (
              <div key={text(candidate.candidate_id) || index}>
                <p>
                  Draft {index + 1}: {requirementCheckSummary(checks)}.
                </p>
                <p className="preserve-lines">{text(candidate.text)}</p>
                {[
                  ...new Set(
                    checks
                      .filter((item) => item.status !== "tested")
                      .map((item) => text(item.reason))
                  ),
                ].map((reason) => (
                  <p key={reason}>{reason}</p>
                ))}
              </div>
            );
          })}
        </details>
      )}
    </section>
  );
}
