import { useEffect, useRef, useState, type FormEvent } from "react";
import { requestJson, type OptimizeResult } from "./api";
import FailureCard from "./components/FailureCard";
import {
  controlLabel,
  isCanonicalOutcome,
  OUTCOME_LABELS,
  record,
} from "./outcome";
import { STYLE_LABELS, type ImprovementStyle } from "./styles";

export type RunSummary = {
  run_id: string;
  created_at?: string;
  status?: string;
  outcome?: string;
  outcome_reason?: string | null;
  applied_style?: string | null;
  control_state?: string | null;
  prompt: string;
  final_prompt?: string | null;
  original_kept?: boolean | null;
  legacy_metadata?: Record<string, unknown> | null;
  feedback?: "accept" | "reject" | null;
  feedback_labels?: {
    status?: "linked" | "unavailable";
    weak_dimensions?: string[];
  } | null;
  cost?: Record<string, unknown>;
  escalated_from?: string | null;
};

export type RunDetail = RunSummary & {
  original_prompt?: string;
  original_kept?: boolean | null;
  models?: Record<string, unknown>;
  diagnosis?: unknown;
  tests?: unknown;
  candidates?: unknown;
  outputs?: unknown;
  grades?: unknown;
  timing?: Record<string, unknown>;
  timings?: Record<string, unknown>;
  report?: unknown;
  metadata?: Record<string, unknown>;
  feedback_at?: string;
  result?: OptimizeResult;
};

type HistoryProps = {
  onOpen?: (result: OptimizeResult) => void;
  refreshKey?: string;
};

/** The status labels `badgeFor` can return. */
export type BadgeLabel =
  | "Converged"
  | "Improved (tested)"
  | "Improved (unverified)"
  | "Impossible"
  | "Failed (operational)"
  | "Outcome not established"
  | "Legacy run";

type Badge = {
  label: BadgeLabel;
  tone: "good" | "neutral" | "warn" | "bad";
};

/**
 * Plain-language meaning of every pill, keyed by the exact labels
 * `badgeFor` returns.  Typing this as a total record of `BadgeLabel` means a
 * new label cannot ship without an explanation.  Each entry is a worded
 * sentence, so the status never reads from colour alone.
 */
export const BADGE_EXPLANATIONS: Record<BadgeLabel, string> = {
  Converged:
    "Every quality dimension met its floor and further rounds stopped buying improvement, so the run stopped on purpose.",
  "Improved (tested)":
    "A changed prompt was accepted with usable success-test evidence supporting it.",
  "Improved (unverified)":
    "A changed prompt passed meaning and safety checks, but usable success-test evidence was unavailable.",
  Impossible:
    "The selected improvement style conflicts with a required prompt constraint, so candidate writing could not proceed.",
  "Failed (operational)":
    "A provider or engine failure prevented the run from producing a supported final result.",
  "Outcome not established":
    "This run has no accepted quality outcome yet; its separate control state explains why it paused or stopped.",
  "Legacy run":
    "This saved run predates canonical outcomes, so its result is shown without inferring an outcome or applied style.",
};

export function badgeFor(run: RunSummary | RunDetail): Badge {
  const detail = run as RunDetail;
  const report = record(detail.report ?? record(detail.result).report);
  const value = run.outcome ?? report.outcome;
  if (!isCanonicalOutcome(value)) {
    return {
      label:
        controlLabelFor(run) || !run.legacy_metadata
          ? "Outcome not established"
          : "Legacy run",
      tone: "neutral",
    };
  }
  const label = OUTCOME_LABELS[value] as BadgeLabel;
  const tone =
    value === "converged" || value === "improved_tested"
      ? "good"
      : value === "failed_operational"
        ? "bad"
        : value === "impossible"
          ? "warn"
          : "neutral";
  return { label, tone };
}

export function controlLabelFor(run: RunSummary | RunDetail): string | null {
  const detail = run as RunDetail;
  const report = record(detail.report ?? record(detail.result).report);
  return controlLabel(run.control_state ?? report.control_state);
}

function runReason(run: RunSummary | RunDetail): string | null {
  const detail = run as RunDetail;
  const report = record(detail.report ?? record(detail.result).report);
  const reason = run.outcome_reason ?? report.outcome_reason;
  return typeof reason === "string" && reason.trim() ? reason : null;
}

function appliedStyle(run: RunSummary | RunDetail): string | null {
  const detail = run as RunDetail;
  const report = record(detail.report ?? record(detail.result).report);
  const style = run.applied_style ?? report.applied_style;
  return typeof style === "string" && Object.hasOwn(STYLE_LABELS, style)
    ? STYLE_LABELS[style as ImprovementStyle]
    : null;
}

const feedbackText = { accept: "helpful", reject: "not helpful" } as const;

/**
 * The short reason shown on a Failed row without opening it, or a pointer to
 * where the reason appears.  List summaries carry no `report`/`result` in the
 * real API, so a row with no stored failure data falls back to a pointer.
 */
export function failedRowReason(run: RunSummary | RunDetail): string {
  const detail = run as RunDetail;
  const report = record(detail.report ?? record(detail.result).report);
  const failure = record(report.failure);
  const headline =
    typeof failure.headline === "string" ? failure.headline.trim() : "";
  if (headline) return headline;
  const message =
    typeof failure.message === "string" ? failure.message.trim() : "";
  if (message) return message;
  const error = typeof report.error === "string" ? report.error.trim() : "";
  if (error) return error;
  return "Open this row to see why it failed.";
}

const detailsExplanationId = "selected-run-status-explanation";

function rowExplanationId(runId: string): string {
  return `history-status-explanation-${runId.replace(/[^A-Za-z0-9_-]+/g, "-")}`;
}

/**
 * A status pill with its explanation attached twice: `title` for the pointer
 * (a native tooltip) and `aria-describedby` for keyboard and assistive
 * technology.  The description is always worded text, so the status never
 * depends on the pill's colour.  The visible label is untouched.
 */
function StatusBadge({
  run,
  describedBy,
  focusable = false,
}: {
  run: RunSummary | RunDetail;
  describedBy?: string;
  focusable?: boolean;
}) {
  const badge = badgeFor(run);
  const title = BADGE_EXPLANATIONS[badge.label];
  return (
    <span
      className={`badge badge-${badge.tone}`}
      title={title}
      aria-describedby={describedBy}
      tabIndex={focusable ? 0 : undefined}
    >
      {badge.label}
    </span>
  );
}

function ControlBadge({ run }: { run: RunSummary | RunDetail }) {
  const label = controlLabelFor(run);
  if (!label) return null;
  return <span className="badge badge-warn badge-control-state">{label}</span>;
}

/** The explanation text a pill points at; hidden until hover or focus. */
function StatusExplanation({
  id,
  run,
}: {
  id: string;
  run: RunSummary | RunDetail;
}) {
  const badge = badgeFor(run);
  return (
    <span id={id} className="status-explanation">
      {BADGE_EXPLANATIONS[badge.label]}
    </span>
  );
}

function when(value?: string): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? ""
    : date.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
}

function money(cost?: Record<string, unknown>): string {
  const total = cost?.total;
  return typeof total === "number" ? `$${total.toFixed(4)}` : "—";
}

function duration(run: RunDetail): string {
  const timing = run.timings ?? run.timing;
  const ms = timing?.total_ms;
  if (typeof ms !== "number") return "—";
  return ms < 60000
    ? `${Math.round(ms / 1000)} s`
    : `${(ms / 60000).toFixed(1)} min`;
}

function highlightedFinalPrompt(original: string, finalPrompt: string) {
  const originalCharacters = Array.from(original);
  const finalCharacters = Array.from(finalPrompt);
  let prefix = 0;
  while (
    prefix < originalCharacters.length &&
    prefix < finalCharacters.length &&
    originalCharacters[prefix] === finalCharacters[prefix]
  ) {
    prefix += 1;
  }

  let originalSuffix = originalCharacters.length;
  let finalSuffix = finalCharacters.length;
  while (
    originalSuffix > prefix &&
    finalSuffix > prefix &&
    originalCharacters[originalSuffix - 1] === finalCharacters[finalSuffix - 1]
  ) {
    originalSuffix -= 1;
    finalSuffix -= 1;
  }

  const changedText = finalCharacters.slice(prefix, finalSuffix).join("");
  return (
    <>
      {finalCharacters.slice(0, prefix).join("")}
      {changedText ? (
        <mark className="history-prompt-change">{changedText}</mark>
      ) : null}
      {finalCharacters.slice(finalSuffix).join("")}
    </>
  );
}

/** A small history browser backed only by the local HTTP API. */
export function History({ onOpen, refreshKey }: HistoryProps) {
  const [query, setQuery] = useState("");
  const [appliedQuery, setAppliedQuery] = useState("");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedRunId, setCopiedRunId] = useState<string | null>(null);
  const [copyError, setCopyError] = useState<string | null>(null);
  const runsRequestId = useRef(0);
  const details = useRef<HTMLElement | null>(null);

  // Keep the opened details on screen; they appear under the clicked row.
  useEffect(() => {
    details.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selected?.run_id]);

  async function loadRuns(search = query) {
    const requestId = ++runsRequestId.current;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (search.trim()) params.set("search", search.trim());
      const body = await requestJson<RunSummary[]>(
        `/api/runs?${params.toString()}`
      );
      if (requestId === runsRequestId.current) setRuns(body);
    } catch (cause) {
      if (requestId === runsRequestId.current)
        setError(
          cause instanceof Error ? cause.message : "Could not load history"
        );
    } finally {
      if (requestId === runsRequestId.current) setLoading(false);
    }
  }

  async function openRun(runId: string) {
    setError(null);
    setCopiedRunId(null);
    setCopyError(null);
    if (selected?.run_id === runId) {
      setSelected(null);
      return;
    }
    try {
      const run = await requestJson<RunDetail>(
        `/api/runs/${encodeURIComponent(runId)}`
      );
      setSelected(run);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not load run details"
      );
    }
  }

  async function copyFinalPrompt(run: RunDetail, finalPrompt: string) {
    setCopyError(null);
    try {
      await navigator.clipboard.writeText(finalPrompt);
      setCopiedRunId(run.run_id);
    } catch {
      setCopiedRunId(null);
      setCopyError("Copy failed. Select the prompt and copy it manually.");
    }
  }

  async function saveFeedback(decision: "accept" | "reject") {
    if (!selected) return;
    setError(null);
    try {
      const run = await requestJson<RunDetail>(
        `/api/runs/${encodeURIComponent(selected.run_id)}/feedback`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ decision }),
        }
      );
      setSelected(run);
      setRuns((current) =>
        current.map((item) =>
          item.run_id === run.run_id
            ? {
                ...item,
                feedback: run.feedback,
                feedback_labels: run.feedback_labels,
              }
            : item
        )
      );
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not save feedback"
      );
    }
  }

  useEffect(() => {
    void loadRuns(appliedQuery);
    // Refresh when the active optimization changes, including a resumed run.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  function renderDetails(run: RunDetail) {
    const waitingQuestionCount = run.result?.questions?.length ?? 0;
    const originalPrompt = run.original_prompt ?? run.prompt;
    const isCompleted =
      run.status === "completed" || run.result?.status === "completed";
    const finalPrompt =
      run.final_prompt ??
      run.result?.final_prompt ??
      (isCompleted ? originalPrompt : null);
    const originalKept = run.original_kept ?? run.result?.original_kept;
    const hasDistinctFinalPrompt =
      typeof finalPrompt === "string" && finalPrompt !== originalPrompt;
    const canCopyFinalPrompt = isCompleted && typeof finalPrompt === "string";
    const copyButton =
      isCompleted && typeof finalPrompt === "string" ? (
        <button
          type="button"
          className="secondary"
          onClick={() => void copyFinalPrompt(run, finalPrompt)}
        >
          {copiedRunId === run.run_id ? "Copied" : "Copy prompt"}
        </button>
      ) : null;

    return (
      <article
        id="selected-run"
        aria-labelledby="selected-run-heading"
        ref={details}
      >
        <div className="run-details-heading">
          <h3 id="selected-run-heading">Run details</h3>
          <StatusBadge run={run} describedBy={detailsExplanationId} focusable />
          <ControlBadge run={run} />
          <StatusExplanation id={detailsExplanationId} run={run} />
        </div>
        <p className="history-meta">
          {[
            when(run.created_at),
            appliedStyle(run) ? `Applied style: ${appliedStyle(run)}` : "",
            duration(run),
            money(run.cost),
          ]
            .filter(Boolean)
            .join(" · ")}
        </p>
        {runReason(run) && (
          <p className="history-outcome-reason">{runReason(run)}</p>
        )}
        {run.status === "failed" && run.result ? (
          <FailureCard
            result={{ ...run.result, report: record(run.result.report) }}
          />
        ) : (
          <>
            {run.status === "needs_input" && (
              <p className="history-note">
                {waitingQuestionCount > 0
                  ? `${waitingQuestionCount} ${waitingQuestionCount === 1 ? "question is" : "questions are"} waiting in the clarification panel above History.`
                  : "Your answers are needed in the clarification panel above History."}
              </p>
            )}
            {originalKept === true && hasDistinctFinalPrompt ? (
              <>
                <p className="history-note">
                  Your original prompt was kept. Highlighted text was added to
                  it.
                </p>
                <div className="history-copy-heading">
                  <h4>Final prompt</h4>
                  {copyButton}
                </div>
                <pre className="history-text">
                  {highlightedFinalPrompt(originalPrompt, finalPrompt)}
                </pre>
              </>
            ) : (
              <>
                <div className="history-copy-heading">
                  <h4>Your prompt</h4>
                  {canCopyFinalPrompt && !hasDistinctFinalPrompt
                    ? copyButton
                    : null}
                </div>
                <pre className="history-text">{originalPrompt}</pre>
                {hasDistinctFinalPrompt && typeof finalPrompt === "string" ? (
                  <>
                    <div className="history-copy-heading">
                      <h4>Final prompt</h4>
                      {copyButton}
                    </div>
                    <pre className="history-text">
                      {highlightedFinalPrompt(originalPrompt, finalPrompt)}
                    </pre>
                    <p className="history-note">
                      Highlighted text was added or changed in the final prompt.
                    </p>
                  </>
                ) : null}
              </>
            )}
            {canCopyFinalPrompt && copyError ? (
              <p role="alert" className="history-copy-error">
                {copyError}
              </p>
            ) : null}
          </>
        )}
        <div className="history-actions">
          {run.result && run.status !== "failed" && onOpen && (
            <button
              type="button"
              className="secondary"
              onClick={() => {
                const result = run.result as OptimizeResult;
                onOpen(
                  run.legacy_metadata
                    ? {
                        ...result,
                        report: {
                          ...record(result.report),
                          legacy_metadata: run.legacy_metadata,
                        },
                      }
                    : result
                );
              }}
            >
              {run.status === "needs_input"
                ? "Answer the questions"
                : "Open this result"}
            </button>
          )}
          {run.status === "completed" && (
            <div
              role="group"
              aria-label="Was this helpful?"
              className="feedback-group"
            >
              <span>Was this helpful?</span>
              <span className="feedback-outcome-context">
                {badgeFor(run).label}
                {appliedStyle(run) ? ` · ${appliedStyle(run)}` : ""}
                {runReason(run) ? ` — ${runReason(run)}` : ""}
              </span>
              <button
                type="button"
                className="secondary"
                onClick={() => void saveFeedback("accept")}
                aria-pressed={run.feedback === "accept"}
              >
                Yes
              </button>
              <button
                type="button"
                className="secondary"
                onClick={() => void saveFeedback("reject")}
                aria-pressed={run.feedback === "reject"}
              >
                No
              </button>
              {run.feedback ? (
                <>
                  <span role="status">
                    Thanks, saved as {feedbackText[run.feedback]}.
                  </span>
                  {run.feedback_labels?.status === "linked" ? (
                    <span>
                      {run.feedback_labels.weak_dimensions?.length
                        ? `Weak dimensions: ${run.feedback_labels.weak_dimensions.join(", ")}.`
                        : "Feedback linked to the selected candidate's score vector."}
                    </span>
                  ) : run.feedback_labels?.status === "unavailable" ? (
                    <span>
                      This run has no selected candidate score vector for
                      calibration.
                    </span>
                  ) : null}
                </>
              ) : null}
            </div>
          )}
        </div>
      </article>
    );
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAppliedQuery(query);
    void loadRuns(query);
  }

  function clearSearch() {
    setQuery("");
    setAppliedQuery("");
    void loadRuns("");
  }

  return (
    <section aria-labelledby="history-heading" className="history-browser">
      <h2 id="history-heading">History</h2>
      <form onSubmit={submit} role="search">
        <label htmlFor="history-search">Search prompts and metadata</label>
        <input
          id="history-search"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search past runs"
        />
        <button type="submit">Search</button>
      </form>
      {error ? <p role="alert">{error}</p> : null}
      {loading ? <p aria-live="polite">Loading history…</p> : null}
      {!loading && runs.length === 0 && appliedQuery.trim() ? (
        <div>
          <p>No runs match this search.</p>
          <button type="button" onClick={clearSearch}>
            Clear search
          </button>
        </div>
      ) : null}
      {!loading && runs.length === 0 && !appliedQuery.trim() ? (
        <>
          <p>No saved runs yet.</p>
          <p>Start a run to see it here.</p>
        </>
      ) : null}
      {runs.length > 0 ? (
        <ul aria-label="Saved optimization runs">
          {runs.map((run) => {
            const open = selected?.run_id === run.run_id;
            const explanationId = rowExplanationId(run.run_id);
            const failed =
              run.status === "failed" ||
              badgeFor(run).label === "Failed (operational)";
            return (
              <li key={run.run_id}>
                <button
                  type="button"
                  className="history-row"
                  onClick={() => void openRun(run.run_id)}
                  aria-expanded={open}
                  aria-controls={open ? "selected-run" : undefined}
                  aria-describedby={explanationId}
                >
                  <StatusBadge run={run} />
                  <ControlBadge run={run} />
                  <span className="history-prompt">{run.prompt}</span>
                  <span className="history-meta">
                    {[
                      when(run.created_at),
                      appliedStyle(run)
                        ? `Applied style: ${appliedStyle(run)}`
                        : "",
                      run.feedback
                        ? `you found it ${feedbackText[run.feedback]}`
                        : "",
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                  {runReason(run) ? (
                    <span className="history-row-reason">{runReason(run)}</span>
                  ) : null}
                  {failed && !runReason(run) ? (
                    <span className="history-failure-reason">
                      <span className="history-failure-label">
                        Why it failed:
                      </span>{" "}
                      {failedRowReason(run)}
                    </span>
                  ) : null}
                </button>
                <StatusExplanation id={explanationId} run={run} />
                {open && selected ? renderDetails(selected) : null}
              </li>
            );
          })}
        </ul>
      ) : null}
    </section>
  );
}

export default History;
