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
      <p className="progress-note">
        You can leave this page open or come back later. The run keeps going and
        will show up in History.
      </p>
    </section>
  );
}
