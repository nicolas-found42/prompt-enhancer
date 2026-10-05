import type { Job } from "../api";
import { elapsedText, stageLabels, stageOrder } from "../outcome";

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
  const current = job.stage ? stageOrder.indexOf(job.stage) : -1;
  const round = job.round.round ? ` · round ${job.round.round}` : "";
  const spent = spentText(job.cost_total);

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
            {spent && <span className="spent"> · {spent}</span>}
            {round}
          </p>
        </div>
        <button
          className="secondary"
          type="button"
          onClick={onCancel}
          disabled={job.cancel_requested}
        >
          {job.cancel_requested ? "Cancelling…" : "Cancel"}
        </button>
      </div>
      <ol className="stages">
        {stageOrder.map((stage, index) => {
          const state =
            job.state === "queued"
              ? "pending"
              : index < current
                ? "done"
                : index === current
                  ? "current"
                  : "pending";
          return (
            <li
              key={stage}
              className={`stage stage-${state}`}
              aria-current={state === "current" ? "step" : undefined}
            >
              {stageLabels[stage]}
            </li>
          );
        })}
      </ol>
      <p className="progress-note">
        You can leave this page open or come back later. The run keeps going and
        will show up in History.
      </p>
    </section>
  );
}
