import type { OptimizeResult } from "./api";
import { humanize, items, plainReason, record } from "./outcome";

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

export default function RunReport({ result }: { result: OptimizeResult }) {
  const report = result.report;
  const diagnosis = record(report.diagnosis);
  const problems = items(diagnosis.problem_sentences);
  const gaps = items(diagnosis.confirmed_gaps);
  const tests = items(report.tests);
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

  return (
    <div className="report-content">
      <p>{text(report.summary)}</p>
      {originalPrompt && (
        <section>
          <h3>Original prompt</h3>
          <p className="original-prompt">
            {highlightedPrompt(originalPrompt, problems)}
          </p>
        </section>
      )}
      <section>
        <h3>Diagnosis</h3>
        <p>
          Task type:{" "}
          {text(diagnosis.task_type_label || diagnosis.task_type) ||
            "undetermined"}
        </p>
        {gaps.length > 0 ? (
          <ul>
            {gaps.map((gap, index) => (
              <li key={text(gap.key) || index}>{text(gap.label || gap.key)}</li>
            ))}
          </ul>
        ) : (
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
