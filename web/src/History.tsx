import { useEffect, useRef, useState, type FormEvent } from "react";
import { requestJson, type OptimizeResult } from "./api";
import FailureCard from "./components/FailureCard";
import { record } from "./outcome";

export type RunSummary = {
  run_id: string;
  created_at?: string;
  status?: string;
  outcome?: string;
  prompt: string;
  final_prompt?: string | null;
  original_kept?: boolean | null;
  tier?: string | null;
  feedback?: "accept" | "reject" | null;
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

type Badge = { label: string; tone: "good" | "neutral" | "warn" | "bad" };

function badgeFor(run: RunSummary | RunDetail): Badge {
  const detail = run as RunDetail;
  const reportStatus = String(
    record(detail.report).status ??
      record(record(detail.result).report).status ??
      ""
  );
  if (run.outcome === "cancelled" || reportStatus === "cancelled")
    return { label: "Cancelled", tone: "neutral" };
  if (run.status === "failed") return { label: "Failed", tone: "bad" };
  if (run.status === "needs_input")
    return { label: "Waiting for answers", tone: "warn" };
  if (run.original_kept === false) {
    const untested =
      run.outcome === "unverified" || reportStatus === "unverified";
    return untested
      ? { label: "Not tested", tone: "neutral" }
      : { label: "Improved", tone: "good" };
  }
  return { label: "Unchanged", tone: "neutral" };
}

const feedbackText = { accept: "helpful", reject: "not helpful" } as const;

/** "standard" or, after a Deep pass, "standard, then deep". */
function tierText(run: RunSummary): string {
  if (!run.tier) return "";
  return run.escalated_from && run.escalated_from !== run.tier
    ? `${run.escalated_from}, then ${run.tier}`
    : run.tier;
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
            ? { ...item, feedback: run.feedback }
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
          <span className={`badge badge-${badgeFor(run).tone}`}>
            {badgeFor(run).label}
          </span>
        </div>
        <p className="history-meta">
          {[when(run.created_at), tierText(run), duration(run), money(run.cost)]
            .filter(Boolean)
            .join(" · ")}
        </p>
        {run.escalated_from && (
          <p className="history-note">
            This run started on {run.escalated_from} and then had a Deep pass;
            this shows the latest result.
          </p>
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
                {originalKept === true ? (
                  <p className="history-outcome">Your prompt was kept as-is.</p>
                ) : null}
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
              onClick={() => onOpen(run.result as OptimizeResult)}
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
                <span role="status">
                  Thanks, saved as {feedbackText[run.feedback]}.
                </span>
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
            return (
              <li key={run.run_id}>
                <button
                  type="button"
                  className="history-row"
                  onClick={() => void openRun(run.run_id)}
                  aria-expanded={open}
                  aria-controls={open ? "selected-run" : undefined}
                >
                  <span className={`badge badge-${badgeFor(run).tone}`}>
                    {badgeFor(run).label}
                  </span>
                  <span className="history-prompt">{run.prompt}</span>
                  <span className="history-meta">
                    {[
                      when(run.created_at),
                      tierText(run),
                      run.feedback
                        ? `you found it ${feedbackText[run.feedback]}`
                        : "",
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </button>
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
