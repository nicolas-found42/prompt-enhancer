import type { PromptHealthResult, PromptHealthSettings } from "./api";

const flagLabels: Record<string, string> = {
  vagueness: "Vague wording",
  unresolved_reference: "Unresolved reference",
  contradiction: "Possible contradiction",
  embedded_instruction: "Possible evaluator instruction",
};

type Props = {
  enabled: boolean;
  settings: PromptHealthSettings | null;
  assessment: PromptHealthResult | null;
  checking: boolean;
  onToggle: (enabled: boolean) => void;
  onSelectSpan: (start: number, end: number) => void;
};

export default function PromptHealthPanel({
  enabled,
  settings,
  assessment,
  checking,
  onToggle,
  onSelectSpan,
}: Props) {
  if (!settings?.available) return null;
  const complete = assessment?.status === "complete";
  const score = complete ? assessment?.composite : null;

  return (
    <section className="prompt-health" aria-label="Live prompt health">
      <div className="prompt-health-heading">
        <label className="prompt-health-toggle">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(event) => onToggle(event.target.checked)}
          />
          Live prompt checks
        </label>
        <span className="prompt-health-usage">
          $
          {(
            assessment?.usage.rolling_hour_usd ??
            settings.usage.rolling_hour_usd
          ).toFixed(4)}{" "}
          / ${settings.hourly_allowance_usd.toFixed(2)} this hour
        </span>
      </div>
      {enabled && (
        <div aria-live="polite">
          {checking && (
            <p className="prompt-health-muted">Checking this draft…</p>
          )}
          {!checking && assessment && (
            <>
              {score !== null && score !== undefined && (
                <p className="prompt-health-score">
                  Prompt clarity: {Math.round(score * 100)}%{" "}
                  <small>estimate</small>
                </p>
              )}
              {assessment.status === "paused" && (
                <p className="prompt-health-muted">
                  Live checks paused: {assessment.reason}
                </p>
              )}
              {assessment.status === "partial" && (
                <p className="prompt-health-muted">
                  Assessment incomplete; no overall score is shown.
                </p>
              )}
              {assessment.status === "complete" && score === null && (
                <p className="prompt-health-muted">
                  No applicable dimension has a numeric score.
                </p>
              )}
              {assessment.dimensions.length > 0 && (
                <>
                  <p className="prompt-health-muted">
                    Coverage: {assessment.coverage.assessed} applicable
                    dimensions assessed; {assessment.coverage.unknown}{" "}
                    uncertain.
                  </p>
                  <ul className="prompt-health-dimensions">
                    {assessment.dimensions.map((item) => (
                      <li key={item.id}>
                        <span>{item.label}</span>
                        <strong>
                          {item.applicable === false
                            ? "Not needed"
                            : item.applicable === null
                              ? "Uncertain"
                              : item.score === null
                                ? "Unavailable"
                                : `${item.score.toFixed(1)} / 3`}
                        </strong>
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {assessment.flags.length > 0 && (
                <div className="prompt-health-flags">
                  <p>Check these sentences:</p>
                  <ul>
                    {assessment.flags.map((flag) => (
                      <li key={`${flag.sentence_id}:${flag.kind}`}>
                        <button
                          type="button"
                          onClick={() => onSelectSpan(flag.start, flag.end)}
                        >
                          {flagLabels[flag.kind] ?? flag.kind}: “{flag.text}”
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}
