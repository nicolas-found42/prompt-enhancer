import { useEffect, useState, type FormEvent } from "react";
import { requestJson } from "./api";

export type RunSummary = {
  run_id: string;
  created_at?: string;
  status?: string;
  prompt: string;
  final_prompt?: string | null;
  tier?: string | null;
  feedback?: "accept" | "reject" | null;
  cost?: Record<string, unknown>;
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
};

type HistoryProps = {
  onSelect?: (run: RunDetail) => void;
  refreshKey?: string;
};

function valueText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === undefined || value === null) return "—";
  return JSON.stringify(value, null, 2);
}

/** A small history browser backed only by the local HTTP API. */
export function History({ onSelect, refreshKey }: HistoryProps) {
  const [query, setQuery] = useState("");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadRuns(search = query) {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (search.trim()) params.set("search", search.trim());
      const body = await requestJson<RunSummary[]>(`/api/runs?${params.toString()}`);
      setRuns(body);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load history");
    } finally {
      setLoading(false);
    }
  }

  async function openRun(runId: string) {
    setError(null);
    try {
      const run = await requestJson<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);
      setSelected(run);
      onSelect?.(run);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load run details");
    }
  }

  async function saveFeedback(decision: "accept" | "reject") {
    if (!selected) return;
    setError(null);
    try {
      const run = await requestJson<RunDetail>(`/api/runs/${encodeURIComponent(selected.run_id)}/feedback`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ decision }),
      });
      setSelected(run);
      setRuns((current) => current.map((item) => (item.run_id === run.run_id ? { ...item, feedback: run.feedback } : item)));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save feedback");
    }
  }

  useEffect(() => {
    void loadRuns("");
    // Refresh when the active optimization changes, including a resumed run.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void loadRuns();
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
      {!loading && runs.length === 0 ? <p>No saved runs yet.</p> : null}
      {runs.length > 0 ? (
        <ul aria-label="Saved optimization runs">
          {runs.map((run) => (
            <li key={run.run_id}>
              <button type="button" onClick={() => void openRun(run.run_id)} aria-pressed={selected?.run_id === run.run_id}>
                <strong>{run.tier || "Run"}</strong> — {run.prompt}
                {run.feedback ? ` (${run.feedback})` : ""}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      {selected ? (
        <article aria-labelledby="selected-run-heading">
          <h3 id="selected-run-heading">Run details</h3>
          <dl>
            <dt>Tier</dt><dd>{selected.tier || "—"}</dd>
            <dt>Original prompt</dt><dd>{valueText(selected.original_prompt ?? selected.prompt)}</dd>
            <dt>Final prompt</dt><dd>{valueText(selected.final_prompt)}</dd>
            <dt>Diagnosis</dt><dd>{valueText(selected.diagnosis)}</dd>
            <dt>Tests</dt><dd>{valueText(selected.tests)}</dd>
            <dt>Candidates</dt><dd>{valueText(selected.candidates)}</dd>
            <dt>Outputs</dt><dd>{valueText(selected.outputs)}</dd>
            <dt>Grades</dt><dd>{valueText(selected.grades)}</dd>
            <dt>Cost</dt><dd>{valueText(selected.cost)}</dd>
            <dt>Timings</dt><dd>{valueText(selected.timings ?? selected.timing)}</dd>
          </dl>
          <div role="group" aria-label="Result feedback">
            <button type="button" onClick={() => void saveFeedback("accept")} aria-pressed={selected.feedback === "accept"}>
              Accept result
            </button>
            <button type="button" onClick={() => void saveFeedback("reject")} aria-pressed={selected.feedback === "reject"}>
              Reject result
            </button>
            {selected.feedback ? <span> Saved feedback: {selected.feedback}</span> : null}
          </div>
        </article>
      ) : null}
    </section>
  );
}

export default History;
