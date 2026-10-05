import type { OptimizeResult } from "../api";
import { failureOf, items } from "../outcome";

type Props = {
  result: OptimizeResult;
  onRetry?: () => void;
  onOpenModels?: () => void;
  busy?: boolean;
};

export default function FailureCard({
  result,
  onRetry,
  onOpenModels,
  busy = false,
}: Props) {
  const failure = failureOf(result);
  const cancelled = failure.kind === "cancelled";
  const modelProblem =
    !cancelled && failure.kind !== "internal" && Boolean(failure.provider);
  const keptRounds = items(result.report.history);

  return (
    <section
      id="run-outcome"
      className={`result failure${cancelled ? " cancelled" : ""}`}
      role="alert"
      aria-labelledby="failure-heading"
      tabIndex={-1}
    >
      <p className="eyebrow">{cancelled ? "CANCELLED" : "RUN FAILED"}</p>
      <h2 id="failure-heading">{failure.headline}</h2>
      <p>{failure.hint}</p>
      {!cancelled && result.original_kept === true && (
        <p className="failure-kept">Your prompt was not changed.</p>
      )}
      {keptRounds.length > 0 && (
        <div className="failure-rounds">
          <p>
            {keptRounds.length} completed round
            {keptRounds.length === 1 ? "" : "s"} kept:
          </p>
          <ul>
            {keptRounds.map((entry, index) => (
              <li key={String(entry.round_number ?? index)}>
                Round {String(entry.round_number ?? index + 1)} —{" "}
                {String(entry.status ?? "completed")}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="failure-actions">
        {modelProblem && onOpenModels && (
          <button className="primary" type="button" onClick={onOpenModels}>
            Change models
          </button>
        )}
        {onRetry && (
          <button
            className="secondary"
            type="button"
            onClick={onRetry}
            disabled={busy}
          >
            Try again
          </button>
        )}
      </div>
      {failure.message && (
        <details className="failure-details">
          <summary>Technical details</summary>
          <pre>{failure.message}</pre>
        </details>
      )}
    </section>
  );
}
