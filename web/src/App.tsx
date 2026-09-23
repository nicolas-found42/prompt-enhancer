import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  optimizePrompt,
  getCatalog,
  getSettings,
  resumeRun,
  saveSettings,
  skipClarification,
  startDeepPass,
  updateAssumption,
  type Assumption,
  type OptimizeResult,
  type ModelCatalog,
  type ModelSelection,
  type Tier,
} from "./api";
import ClarificationPanel, { type ClarificationQuestion } from "./components/ClarificationPanel";
import History from "./History";
import ModelPicker from "./ModelPicker";
import RunReport from "./RunReport";

const defaultPrompt = "Summarize this article in three concise bullets for a busy reader.";
const tierEstimates: Record<Tier, string> = {
  fast: "About 1–2 min · ~$0.001–$0.004 · up to 1 round",
  standard: "About 2–5 min · ~$0.006–$0.015 · up to 2 rounds",
  deep: "About 5–15 min · ~$0.03–$0.08 · up to 3 rounds",
};

type AssumptionEditorProps = {
  assumptions: Assumption[];
  busy: boolean;
  onSave: (assumption: Assumption) => void | Promise<void>;
};

function AssumptionEditor({ assumptions, busy, onSave }: AssumptionEditorProps) {
  const [values, setValues] = useState<Record<string, string>>({});
  const assumptionSignature = JSON.stringify(assumptions);

  useEffect(() => {
    setValues(Object.fromEntries(assumptions.map((item) => [item.key, item.value])));
  }, [assumptionSignature]);

  if (assumptions.length === 0) return null;
  return (
    <section className="assumption-editor" aria-labelledby="assumptions-heading">
      <h3 id="assumptions-heading">Review assumptions</h3>
      <p>Edit an assumption to refine the final prompt. The meaning check runs before it is accepted.</p>
      {assumptions.map((assumption) => (
        <div className="assumption-row" key={assumption.key}>
          <label htmlFor={`assumption-${assumption.key}`}>{assumption.key}</label>
          <input
            id={`assumption-${assumption.key}`}
            value={values[assumption.key] ?? assumption.value}
            onChange={(event) =>
              setValues((current) => ({ ...current, [assumption.key]: event.target.value }))
            }
          />
          <button
            className="secondary"
            type="button"
            disabled={busy || !(values[assumption.key] ?? assumption.value).trim()}
            onClick={() =>
              void onSave({
                ...assumption,
                value: (values[assumption.key] ?? assumption.value).trim(),
              })
            }
          >
            Save
          </button>
        </div>
      ))}
    </section>
  );
}

function isAssumption(value: unknown): value is Assumption {
  if (typeof value !== "object" || value === null) return false;
  return "key" in value && typeof value.key === "string" && "value" in value && typeof value.value === "string";
}

function reportAssumptions(result: OptimizeResult): Assumption[] {
  const values = result.report.assumptions;
  if (!Array.isArray(values)) return [];
  return values.filter(isAssumption);
}

export default function App() {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [tier, setTier] = useState<Tier>("standard");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [selection, setSelection] = useState<ModelSelection | null>(null);

  useEffect(() => {
    void Promise.all([getCatalog(), getSettings()])
      .then(([available, defaults]) => {
        setCatalog(available);
        setSelection({ writer: defaults.writer_model, strong: defaults.strong_check_model, weak: defaults.weak_models });
      })
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Unable to load model choices."));
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setCopied(false);
    try {
      setResult(await optimizePrompt(prompt, tier, selection ?? undefined));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to optimize this prompt.");
    } finally {
      setBusy(false);
    }
  }

  async function saveModelDefaults() {
    if (!selection) return;
    setBusy(true);
    setError(null);
    try {
      await saveSettings(selection);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to save model defaults.");
    } finally {
      setBusy(false);
    }
  }

  async function resume(answers: Record<string, unknown>) {
    if (!result) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await resumeRun(result.run_id, answers));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to continue this run.");
    } finally {
      setBusy(false);
    }
  }

  async function skip() {
    if (!result) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await skipClarification(result.run_id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to continue this run.");
    } finally {
      setBusy(false);
    }
  }

  async function saveAssumption(assumption: Assumption) {
    if (!result) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await updateAssumption(result.run_id, assumption));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to update this assumption.");
    } finally {
      setBusy(false);
    }
  }

  async function deepPass() {
    if (!result) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await startDeepPass(result.run_id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to start Deep pass.");
    } finally {
      setBusy(false);
    }
  }

  async function copyPrompt() {
    if (!result?.final_prompt) return;
    try {
      await navigator.clipboard.writeText(result.final_prompt);
      setCopied(true);
    } catch {
      setCopied(false);
      setError("Copy failed. Select the prompt and copy it manually.");
    }
  }

  const questions: ClarificationQuestion[] = result?.status === "needs_input" ? (result.questions ?? []) : [];
  const assumptions = useMemo(() => (result ? reportAssumptions(result) : []), [result]);

  return (
    <main className="shell">
      <header className="hero">
        <p className="eyebrow">LOCAL PROMPT WORKBENCH</p>
        <h1>Make your prompt clearer.</h1>
        <p className="lede">
          Paste one prompt, choose how much effort to spend, and get a clean result you can copy.
        </p>
      </header>

      <form className="composer" onSubmit={submit}>
        <label htmlFor="prompt">Your prompt</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="What would you like help with?"
          rows={8}
          required
        />
        <div className="form-actions">
          <label className="tier-label" htmlFor="tier">
            Effort
            <select id="tier" value={tier} onChange={(event) => setTier(event.target.value as Tier)}>
              <option value="fast">Fast</option>
              <option value="standard">Standard</option>
              <option value="deep">Deep</option>
            </select>
          </label>
          <button className="primary" type="submit" disabled={busy || !prompt.trim()}>
            {busy ? "Optimizing…" : "Optimize prompt"}
          </button>
        </div>
        <p className="effort-estimate">{tierEstimates[tier]} (rough estimate; model choice and prompt length change time and cost).</p>
        {selection && <ModelPicker catalog={catalog} selection={selection} onChange={setSelection} onSave={() => void saveModelDefaults()} busy={busy} />}
      </form>

      {error && <p className="error" role="alert">{error}</p>}

      {questions.length > 0 ? (
        <ClarificationPanel
          questions={questions}
          onSubmit={resume}
          onSkip={skip}
          busy={busy}
          error={error}
        />
      ) : null}

      {result?.status === "completed" ? (
        <section className="result" aria-live="polite">
          <div className="result-heading">
            <div>
              <p className="eyebrow">RESULT</p>
              <h2>{result.original_kept ? "No change needed" : "Optimized prompt"}</h2>
            </div>
            {result.final_prompt && (
              <button className="secondary" type="button" onClick={copyPrompt}>
                {copied ? "Copied" : "Copy prompt"}
              </button>
            )}
          </div>
          {result.final_prompt && <pre className="final-prompt">{result.final_prompt}</pre>}
          {Boolean(result.report.offer_deep) && (
            <div className="deep-offer">
              <p>Deep pass: {typeof result.report.offer_deep === "object" && result.report.offer_deep !== null
                ? String((result.report.offer_deep as Record<string, unknown>).expected_effort ?? tierEstimates.deep)
                : tierEstimates.deep}</p>
              <p>{typeof result.report.offer_deep === "object" && result.report.offer_deep !== null
                ? String((result.report.offer_deep as Record<string, unknown>).expected_cost_change ?? "Higher model cost")
                : "Higher model cost"}</p>
              <button className="secondary" type="button" disabled={busy} onClick={() => void deepPass()}>
                Try a Deep pass
              </button>
            </div>
          )}
          <details className="report">
            <summary>View report</summary>
            <AssumptionEditor assumptions={assumptions} busy={busy} onSave={saveAssumption} />
            <p className="run-meta">
              Run {result.run_id} · {result.timing.total_ms} ms · ${result.cost.total.toFixed(4)}
            </p>
            <RunReport result={result} />
          </details>
        </section>
      ) : null}

      <History refreshKey={`${result?.run_id ?? ""}:${result?.status ?? ""}:${result?.final_prompt ?? ""}`} />
    </main>
  );
}
