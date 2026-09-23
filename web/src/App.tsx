import { FormEvent, useState } from "react";
import { optimizePrompt, OptimizeResult, Tier } from "./api";

const defaultPrompt = "Summarize this article in three concise bullets for a busy reader.";


export default function App() {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [tier, setTier] = useState<Tier>("standard");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setCopied(false);
    try {
      setResult(await optimizePrompt(prompt, tier));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to optimize this prompt.");
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
      </form>

      {error && <p className="error" role="alert">{error}</p>}

      {result && (
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
          <details className="report">
            <summary>View report</summary>
            <p className="run-meta">
              Run {result.run_id} · {result.timing.total_ms} ms · ${result.cost.total.toFixed(4)}
            </p>
            <pre>{JSON.stringify(result.report, null, 2)}</pre>
          </details>
        </section>
      )}
    </main>
  );
}
