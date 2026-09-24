import type { OptimizeResult } from "../api";
import { failureOf } from "../outcome";

type Props = {
  result: OptimizeResult;
  onRetry?: () => void;
  onOpenModels?: () => void;
  busy?: boolean;
};

export default function FailureCard({ result, onRetry, onOpenModels, busy = false }: Props) {
  const failure = failureOf(result);
  const cancelled = failure.kind === "cancelled";
  const modelProblem = !cancelled && failure.kind !== "internal" && Boolean(failure.provider);

  return (
    <section className={`result failure${cancelled ? " cancelled" : ""}`} role="alert" aria-labelledby="failure-heading">
      <p className="eyebrow">{cancelled ? "CANCELLED" : "RUN FAILED"}</p>
      <h2 id="failure-heading">{failure.headline}</h2>
      <p>{failure.hint}</p>
      {!cancelled && <p className="failure-kept">Your prompt was not changed.</p>}
      <div className="failure-actions">
        {modelProblem && onOpenModels && (
          <button className="primary" type="button" onClick={onOpenModels}>
            Change models
          </button>
        )}
        {onRetry && (
          <button className="secondary" type="button" onClick={onRetry} disabled={busy}>
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
