import RequirementCoverage from "../RequirementCoverage";
import { record } from "../outcome";
import type { Job } from "../api";
import { elapsedText, stageLabels } from "../outcome";

type Props = {
  job: Job;
  onCancel: () => void;
};

const kindTitles: Record<Job["kind"], string> = {
  optimize: "Improving your prompt",
  resume: "Continuing with your answers",
  skip: "Continuing without answers",
  continue: "Continuing after your approval",
};

function spentText(costTotal: number | undefined): string | null {
  if (typeof costTotal !== "number" || !Number.isFinite(costTotal)) return null;
  return `$${costTotal.toFixed(4)} spent`;
}

export default function RunProgress({ job, onCancel }: Props) {
  const round = job.round.round ? ` · round ${job.round.round}` : "";
  const spent = spentText(job.cost_total);
  const coverageEvent = [...(job.events ?? [])]
    .reverse()
    .find((event) => event.requirements);
  const candidates = new Map<string, Record<string, unknown>>();
  let selected: Record<string, unknown> | undefined;
  for (const event of job.events ?? []) {
    if (!event.candidate_id) continue;
    const key = `${event.round}:${event.candidate_id}`;
    const candidate = candidates.get(key) ?? {
      candidate_id: event.candidate_id,
      round: event.round,
      status: "checks pending",
    };
    if (event.draft) candidate.text = event.draft;
    if (event.checks?.length) {
      candidate.metadata = {
        requirement_findings: event.checks.flatMap(
          (check) => check.evidence ?? []
        ),
      };
    }
    if (event.kind === "qualified") {
      candidate.selected = true;
      candidate.status = "qualified";
      selected = candidate;
    } else if (event.kind === "blocked") {
      candidate.selected = false;
      candidate.status = "did not qualify";
    }
    candidates.set(key, candidate);
  }
  const selection = {
    selected_candidate: selected,
    ranking: [...candidates.values()].filter(
      (candidate) => candidate !== selected
    ),
  };
  const rounds = new Map<number, Record<string, unknown>[]>();
  for (const candidate of candidates.values()) {
    const number = Number(candidate.round);
    const ranking = rounds.get(number) ?? [];
    ranking.push(candidate);
    rounds.set(number, ranking);
  }
  const history = [...rounds.entries()]
    .sort(([a], [b]) => a - b)
    .map(([number, ranking]) => ({
      round_number: number,
      evidence: { selection_evidence: { ranking } },
    }));
  const cancellationPending = job.cancellation_pending ?? job.cancel_requested;

  return (
    <section
      id="run-progress"
      className="result progress"
      aria-live="polite"
      aria-labelledby="progress-heading"
    >
      <div className="result-heading">
        <div>
          <p className="eyebrow">WORKING</p>
          <h2 id="progress-heading">{kindTitles[job.kind]}</h2>
          {job.prompt && <p className="progress-prompt">{job.prompt}</p>}
          <p className="progress-meta">
            <span className="elapsed">
              {elapsedText(job.elapsed_ms)} elapsed
            </span>
            {job.remaining_active_ms !== undefined && (
              <span>
                {" "}
                · {elapsedText(job.remaining_active_ms)} active time remaining
              </span>
            )}
            {spent && <span className="spent"> · {spent}</span>}
            {round}
          </p>
        </div>
        <button
          className="secondary"
          type="button"
          onClick={onCancel}
          disabled={cancellationPending}
        >
          {cancellationPending ? "Cancelling…" : "Cancel"}
        </button>
      </div>
      <ol className="stages" aria-label="Run activity">
        {job.events?.length ? (
          job.events.map((event) => (
            <li key={event.cursor} className="stage">
              <span>
                {event.stage
                  ? `${stageLabels[event.stage] ?? event.summary} — ${event.kind}`
                  : event.summary}
              </span>
              <small>
                {" "}
                · {elapsedText(event.elapsed_ms)}
                {event.round ? ` · round ${event.round}` : ""}
              </small>
              {event.candidate_id && <p>Draft {event.candidate_id}</p>}
              {event.draft && (
                <details>
                  <summary>
                    {event.kind === "qualified"
                      ? "Qualified rewrite"
                      : "Draft preview — awaiting checks"}
                  </summary>
                  <pre>{event.draft}</pre>
                </details>
              )}
              {event.diff && (
                <details>
                  <summary>Changes from your prompt</summary>
                  <pre>{event.diff}</pre>
                </details>
              )}
              {event.reasons?.length ? (
                <ul>
                  {event.reasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              ) : null}
              {event.checks?.length ? (
                <ul aria-label="Requirement checks">
                  {event.checks.map((check) => (
                    <li key={check.requirement_id}>
                      <p>{check.source}</p>
                      <p>
                        {check.tested} passed check
                        {check.tested === 1 ? "" : "s"}, {check.failed} failed
                        check{check.failed === 1 ? "" : "s"}, {check.untestable}{" "}
                        untestable check{check.untestable === 1 ? "" : "s"},{" "}
                        {check.unresolved ?? 0} unresolved checks.
                      </p>
                      {check.reasons.map((reason) => (
                        <p key={reason}>{reason}</p>
                      ))}
                    </li>
                  ))}
                </ul>
              ) : null}
            </li>
          ))
        ) : (
          <li className="stage stage-current" aria-current="step">
            {job.state === "queued"
              ? "Waiting for processing to begin"
              : job.stage
                ? (stageLabels[job.stage] ?? "Processing your prompt")
                : "Processing your prompt"}
          </li>
        )}
      </ol>
      {coverageEvent && (
        <RequirementCoverage
          live
          ledger={record(coverageEvent.requirements)}
          selection={selection}
          history={history}
        />
      )}
      <p className="progress-note">
        You can leave this page open or come back later. The run keeps going and
        will show up in History.
      </p>
    </section>
  );
}
