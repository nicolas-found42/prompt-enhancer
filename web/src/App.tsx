import {
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ApiError,
  cancelJob,
  checkPromptHealth,
  getActiveJobs,
  getCatalog,
  getEstimates,
  getJob,
  getProviders,
  getPromptHealthSettings,
  getRunResult,
  getSettings,
  saveSettings,
  startDeep,
  startOptimize,
  startResume,
  startSkip,
  updateAssumption,
  type Assumption,
  type Job,
  type ModelCatalog,
  type ModelSelection,
  type OptimizeResult,
  type ProviderReport,
  type PromptHealthResult,
  type PromptHealthSettings,
  type Tier,
  type TierEstimate,
} from "./api";
import ClarificationPanel, {
  type ClarificationQuestion,
} from "./components/ClarificationPanel";
import FailureCard from "./components/FailureCard";
import RunProgress from "./components/RunProgress";
import History from "./History";
import ModelPicker from "./ModelPicker";
import PromptHealthPanel from "./PromptHealthPanel";
import {
  confirmedGaps,
  estimateText,
  humanize,
  outcomeOf,
  possibleGapHints,
  record,
  roughCost,
  tierDescriptions,
} from "./outcome";
import RunReport from "./RunReport";

const ACTIVE_RUN_KEY = "prompt-enhancer.active-run";
const LAST_RESULT_KEY = "prompt-enhancer.last-result";
const DRAFT_KEY = "prompt-enhancer.draft";
const HEALTH_ENABLED_KEY = "prompt-enhancer.live-health";
const HEALTH_SESSION_KEY = "prompt-enhancer.health-session";
const POLL_MS = 1000;
// A result shown this recently comes back after a reload instead of vanishing.
const RESTORE_RESULT_MS = 30 * 60 * 1000;

type RememberedRun = { runId: string; prompt: string };
type RememberedResult = { runId: string; at: number };
type StoredDraft = { present: boolean; prompt: string };

function readDraft(): StoredDraft {
  try {
    const prompt = localStorage.getItem(DRAFT_KEY);
    return { present: prompt !== null, prompt: prompt ?? "" };
  } catch {
    return { present: false, prompt: "" };
  }
}

function saveDraft(prompt: string) {
  try {
    localStorage.setItem(DRAFT_KEY, prompt);
  } catch {
    // Ignore: the draft remains available for this page session.
  }
}

function healthPreference(): string | null {
  try {
    return localStorage.getItem(HEALTH_ENABLED_KEY);
  } catch {
    return null;
  }
}

function healthSession(): string {
  try {
    const existing = sessionStorage.getItem(HEALTH_SESSION_KEY);
    if (existing) return existing;
    const created =
      typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `health-${Date.now()}-${Math.random()}`;
    sessionStorage.setItem(HEALTH_SESSION_KEY, created);
    return created;
  } catch {
    return `health-${Date.now()}-${Math.random()}`;
  }
}

// Storage can be unavailable; reattaching then falls back to /api/jobs.
function store(key: string, value: unknown) {
  try {
    if (value) localStorage.setItem(key, JSON.stringify(value));
    else localStorage.removeItem(key);
  } catch {
    // Ignore: these are conveniences only.
  }
}

function stored<T extends { runId: string }>(key: string): T | null {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) ?? "null") as T | null;
    return parsed && typeof parsed.runId === "string" ? parsed : null;
  } catch {
    return null;
  }
}

function rememberRun(run: RememberedRun | null) {
  store(ACTIVE_RUN_KEY, run);
}

function rememberedRun(): RememberedRun | null {
  return stored<RememberedRun>(ACTIVE_RUN_KEY);
}

type AssumptionEditorProps = {
  assumptions: Assumption[];
  busy: boolean;
  onSave: (assumption: Assumption) => void | Promise<void>;
};

const unknownValue = "Not specified";

function AssumptionEditor({
  assumptions,
  busy,
  onSave,
}: AssumptionEditorProps) {
  const [values, setValues] = useState<Record<string, string>>({});
  const assumptionSignature = JSON.stringify(assumptions);

  useEffect(() => {
    setValues(
      Object.fromEntries(
        assumptions.map((item) => [
          item.key,
          item.value === unknownValue ? "" : item.value,
        ])
      )
    );
  }, [assumptionSignature]);

  if (assumptions.length === 0) return null;
  return (
    <section
      className="assumption-editor"
      aria-labelledby="assumptions-heading"
    >
      <h3 id="assumptions-heading">Review assumptions</h3>
      <p>
        These are the details the optimizer filled in or left out. Change one to
        update the final prompt; it is checked to make sure your meaning is
        kept.
      </p>
      {assumptions.map((assumption) => {
        const unknown = assumption.value === unknownValue;
        const value =
          values[assumption.key] ?? (unknown ? "" : assumption.value);
        const label = assumption.label
          ? humanize(assumption.label)
          : humanize(assumption.key);
        return (
          <div className="assumption-row" key={assumption.key}>
            <label htmlFor={`assumption-${assumption.key}`}>{label}</label>
            <input
              id={`assumption-${assumption.key}`}
              value={value}
              placeholder={
                unknown
                  ? "Not given yet. Type it here if you know it."
                  : undefined
              }
              onChange={(event) =>
                setValues((current) => ({
                  ...current,
                  [assumption.key]: event.target.value,
                }))
              }
            />
            <button
              className="secondary"
              type="button"
              disabled={
                busy || !value.trim() || value.trim() === assumption.value
              }
              onClick={() =>
                void onSave({ ...assumption, value: value.trim() })
              }
            >
              Save
            </button>
            {unknown && (
              <p className="assumption-hint">
                The optimizer didn't know this, so the prompt leaves it out.
              </p>
            )}
          </div>
        );
      })}
    </section>
  );
}

function isAssumption(value: unknown): value is Assumption {
  if (typeof value !== "object" || value === null) return false;
  return (
    "key" in value &&
    typeof value.key === "string" &&
    "value" in value &&
    typeof value.value === "string"
  );
}

function reportAssumptions(result: OptimizeResult): Assumption[] {
  const values = result.report.assumptions;
  if (!Array.isArray(values)) return [];
  return values.filter(isAssumption);
}

function originalPromptOf(result: OptimizeResult): string | undefined {
  return (
    result.original_prompt ??
    (result.original_kept ? result.final_prompt : undefined)
  );
}

type ClarificationValidationError = { questionId: string; message: string };

function clarificationValidationError(
  caught: unknown
): ClarificationValidationError | null {
  if (!(caught instanceof ApiError) || caught.status !== 422) return null;
  const detail = record(caught.detail);
  if (
    detail.code !== "invalid_answer" ||
    typeof detail.question_id !== "string" ||
    typeof detail.message !== "string"
  ) {
    return null;
  }
  return { questionId: detail.question_id, message: detail.message };
}

function deepOfferText(result: OptimizeResult): string {
  const offer = record(result.report.offer_deep);
  const multiplier =
    typeof offer.expected_evaluation_multiplier === "number"
      ? offer.expected_evaluation_multiplier
      : null;
  if (multiplier === null)
    return "Deep tries more rewrites on more test models. It takes longer and costs more.";
  const cost = result.cost.total * multiplier;
  const costText = cost > 0 ? `, ${roughCost(cost)}` : "";
  return `Deep tries more rewrites on more test models. Expect about ${multiplier.toFixed(1)}× the work of this run${costText}.`;
}

export default function App() {
  const [initialDraft] = useState(readDraft);
  const [prompt, setPrompt] = useState(initialDraft.prompt);
  const [draftRevision, setDraftRevision] = useState(0);
  const [healthSettings, setHealthSettings] =
    useState<PromptHealthSettings | null>(null);
  const [healthEnabled, setHealthEnabled] = useState(false);
  const [healthResult, setHealthResult] = useState<PromptHealthResult | null>(
    null
  );
  const [healthChecking, setHealthChecking] = useState(false);
  const [pageVisible, setPageVisible] = useState(!document.hidden);
  const [healthSessionId] = useState(healthSession);
  const promptRef = useRef(prompt);
  const revisionRef = useRef(0);
  const lastAssessed = useRef<string | null>(null);
  const promptField = useRef<HTMLTextAreaElement | null>(null);
  const [tier, setTier] = useState<Tier>("standard");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [viewingHistoryResult, setViewingHistoryResult] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clarificationError, setClarificationError] =
    useState<ClarificationValidationError | null>(null);
  const [saving, setSaving] = useState(false);
  const [copied, setCopied] = useState(false);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [selection, setSelection] = useState<ModelSelection | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [providers, setProviders] = useState<ProviderReport | null>(null);
  const [estimates, setEstimates] = useState<
    Partial<Record<Tier, TierEstimate>>
  >({});
  const lastRequest = useRef<{ prompt: string; tier: Tier } | null>(null);
  const composer = useRef<HTMLFormElement | null>(null);
  const draftWasSet = useRef(initialDraft.present);
  const busy = job !== null || saving;

  const changeDraft = useCallback((next: string) => {
    draftWasSet.current = true;
    saveDraft(next);
    promptRef.current = next;
    revisionRef.current += 1;
    setDraftRevision(revisionRef.current);
    lastAssessed.current = null;
    setHealthResult(null);
    setPrompt(next);
  }, []);

  const restoreDraft = useCallback((next: string) => {
    if (draftWasSet.current) return;
    draftWasSet.current = true;
    saveDraft(next);
    promptRef.current = next;
    revisionRef.current += 1;
    setDraftRevision(revisionRef.current);
    lastAssessed.current = null;
    setHealthResult(null);
    setPrompt(next);
  }, []);

  useEffect(() => {
    void getPromptHealthSettings()
      .then((settings) => {
        setHealthSettings(settings);
        setHealthEnabled(settings.available && healthPreference() !== "off");
      })
      .catch(() => setHealthSettings(null));
    const onVisibility = () => setPageVisible(!document.hidden);
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, []);

  useEffect(() => {
    if (
      !healthSettings?.available ||
      !healthEnabled ||
      !pageVisible ||
      !prompt.trim()
    ) {
      setHealthChecking(false);
      return;
    }
    if (lastAssessed.current === prompt) return;
    const controller = new AbortController();
    const currentPrompt = prompt;
    const currentRevision = draftRevision;
    const timer = window.setTimeout(() => {
      setHealthChecking(true);
      void checkPromptHealth(
        currentPrompt,
        currentRevision,
        healthSessionId,
        controller.signal
      )
        .then((assessment) => {
          if (
            controller.signal.aborted ||
            promptRef.current !== currentPrompt ||
            revisionRef.current !== assessment.draft.revision ||
            assessment.draft.revision !== currentRevision
          )
            return;
          lastAssessed.current = currentPrompt;
          setHealthResult(assessment);
          setHealthChecking(false);
        })
        .catch(() => {
          if (controller.signal.aborted) return;
          lastAssessed.current = currentPrompt;
          setHealthResult(null);
          setHealthChecking(false);
        });
    }, healthSettings.debounce_ms);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [
    prompt,
    draftRevision,
    healthSettings,
    healthEnabled,
    pageVisible,
    healthSessionId,
  ]);

  useEffect(() => {
    void Promise.all([getCatalog(), getSettings()])
      .then(([available, defaults]) => {
        setCatalog(available);
        setSelection({
          writer: defaults.writer_model,
          strong: defaults.strong_check_model,
          weak: defaults.weak_models,
        });
      })
      .catch((caught) =>
        setError(
          caught instanceof Error
            ? caught.message
            : "Unable to load model choices."
        )
      );
    void getProviders(true)
      .then(setProviders)
      .catch(() => setProviders(null));
    void getEstimates()
      .then(setEstimates)
      .catch(() => setEstimates({}));
  }, []);

  const finish = useCallback(
    (finished: Job) => {
      setJob(null);
      rememberRun(null);
      const finishedResult = finished.result;
      if (finishedResult) {
        setResult(finishedResult);
        const original = originalPromptOf(finishedResult);
        if (original) restoreDraft(original);
      }
      void getEstimates()
        .then(setEstimates)
        .catch(() => undefined);
      void getProviders(false)
        .then(setProviders)
        .catch(() => undefined);
    },
    [restoreDraft]
  );

  // Bring back a result shown shortly before a reload.
  const restoreLastResult = useCallback(() => {
    const last = stored<RememberedResult>(LAST_RESULT_KEY);
    if (!last || Date.now() - last.at > RESTORE_RESULT_MS) return;
    void getRunResult(last.runId)
      .then((found) => {
        const restored = found.result;
        if (!restored) return;
        setResult((current) => current ?? restored);
        const original = originalPromptOf(restored);
        if (original) restoreDraft(original);
      })
      .catch(() => store(LAST_RESULT_KEY, null));
  }, [restoreDraft]);

  // Reattach to a run that was in progress before a reload or dropped connection.
  useEffect(() => {
    const remembered = rememberedRun();
    if (remembered?.prompt) restoreDraft(remembered.prompt);
    const lookup = remembered
      ? getJob(remembered.runId).then((found) => [found])
      : getActiveJobs();
    void lookup
      .then((found) => {
        const current = found[0];
        if (!current) {
          restoreLastResult();
          return;
        }
        if (current.prompt) restoreDraft(current.prompt);
        if (current.state === "done") finish(current);
        else setJob(current);
      })
      .catch(() => {
        rememberRun(null);
        restoreLastResult();
      });
  }, [finish, restoreDraft, restoreLastResult]);

  useEffect(() => {
    if (!result) return;
    if (stored<RememberedResult>(LAST_RESULT_KEY)?.runId !== result.run_id) {
      store(LAST_RESULT_KEY, { runId: result.run_id, at: Date.now() });
    }
  }, [result?.run_id]);

  useEffect(() => {
    if (!job || job.state === "done") return;
    const runId = job.run_id;
    const timer = window.setInterval(() => {
      void getJob(runId)
        .then((latest) => {
          if (latest.state === "done") finish(latest);
          else setJob(latest);
        })
        .catch((caught) => {
          if (caught instanceof ApiError && caught.status === 404) {
            setJob(null);
            rememberRun(null);
            setError(
              "The local engine restarted, so this run was lost. Your prompt is still in the box; press Optimize to try again."
            );
          } else {
            setError(
              caught instanceof Error
                ? caught.message
                : "Lost track of the run."
            );
          }
        });
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [job?.run_id, job?.state, finish]);

  async function begin(
    start: () => Promise<Job>,
    onFailure?: (caught: unknown) => void
  ) {
    setError(null);
    setClarificationError(null);
    setCopied(false);
    setViewingHistoryResult(false);
    try {
      const started = await start();
      rememberRun({ runId: started.run_id, prompt });
      store(LAST_RESULT_KEY, null);
      setResult(null);
      setJob(started);
    } catch (caught) {
      if (onFailure) onFailure(caught);
      else
        setError(
          caught instanceof Error ? caught.message : "Unable to start the run."
        );
    }
  }

  function submit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    lastRequest.current = { prompt, tier };
    void begin(() => startOptimize(prompt, tier, selection ?? undefined));
  }

  function retry() {
    const previous = lastRequest.current;
    if (previous) {
      changeDraft(previous.prompt);
      setTier(previous.tier);
      void begin(() =>
        startOptimize(previous.prompt, previous.tier, selection ?? undefined)
      );
    } else {
      submit();
    }
  }

  async function cancel() {
    if (!job) return;
    try {
      setJob(await cancelJob(job.run_id));
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Unable to cancel the run."
      );
    }
  }

  async function saveModelDefaults() {
    if (!selection) return;
    setSaving(true);
    setError(null);
    try {
      await saveSettings(selection);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Unable to save model defaults."
      );
    } finally {
      setSaving(false);
    }
  }

  async function saveAssumption(assumption: Assumption) {
    if (!result) return;
    setSaving(true);
    setError(null);
    try {
      setResult(await updateAssumption(result.run_id, assumption));
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Unable to update this assumption."
      );
    } finally {
      setSaving(false);
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

  function openModels() {
    setPickerOpen(true);
    composer.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function openFromHistory(opened: OptimizeResult) {
    setResult(opened);
    setCopied(false);
    setClarificationError(null);
    changeDraft(prompt);
    setViewingHistoryResult(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const goIds = useMemo(
    () => new Set((catalog?.providers.go ?? []).map((model) => model.id)),
    [catalog]
  );
  const goUnavailable = providers?.providers.go?.status === "unavailable";
  const goRoles = selection
    ? [
        goIds.has(selection.writer) ? `writer (${selection.writer})` : null,
        goIds.has(selection.strong)
          ? `strong check (${selection.strong})`
          : null,
        selection.weak.some((id) => goIds.has(id))
          ? "some weak-panel models"
          : null,
      ].filter((item): item is string => item !== null)
    : [];

  function switchToFallbackModels() {
    if (!selection || !providers) return;
    const openRouterIds = new Set(
      (catalog?.providers.openrouter ?? []).map((model) => model.id)
    );
    setSelection({
      writer: goIds.has(selection.writer)
        ? providers.fallback.writer
        : selection.writer,
      strong: goIds.has(selection.strong)
        ? providers.fallback.strong
        : selection.strong,
      weak: selection.weak.filter(
        (id) => !goIds.has(id) || openRouterIds.has(id)
      ),
    });
  }

  const questions: ClarificationQuestion[] =
    !job && result?.status === "needs_input" ? (result.questions ?? []) : [];
  const assumptions = useMemo(
    () => (result ? reportAssumptions(result) : []),
    [result]
  );
  const estimate = estimateText(tier, estimates[tier]);
  const progressEstimate =
    job?.kind === "deep" ? estimateText("deep", estimates.deep) : estimate;
  const outcome = result?.status === "completed" ? outcomeOf(result) : null;
  const gaps =
    result?.status === "completed" && result.original_kept
      ? confirmedGaps(result)
      : [];
  const hints =
    result?.status === "completed" && result.original_kept
      ? possibleGapHints(result)
      : [];
  const resultPrompt = result ? originalPromptOf(result) : undefined;
  const resultForEarlierPrompt =
    resultPrompt !== undefined && resultPrompt !== prompt;
  // Deep only rewrites against a confirmed gap; without one it cannot do more.
  const offerDeep = Boolean(result?.report.offer_deep) && gaps.length > 0;

  return (
    <main className="shell">
      <header className="hero">
        <p className="eyebrow">LOCAL PROMPT WORKBENCH</p>
        <h1>Make your prompt clearer.</h1>
        <p className="lede">
          Paste one prompt, choose how much effort to spend, and get a clean
          result you can copy.
        </p>
      </header>

      {goUnavailable && goRoles.length > 0 && (
        <div className="banner" role="status">
          <p>
            <strong>OpenCode Go isn't active</strong> (HTTP{" "}
            {providers?.providers.go?.http_status ?? "403"}). Your{" "}
            {goRoles.join(" and ")} use it, so runs will fail.
          </p>
          <button
            className="primary"
            type="button"
            onClick={switchToFallbackModels}
          >
            Switch to OpenRouter models
          </button>
        </div>
      )}

      <form className="composer" onSubmit={submit} ref={composer}>
        <label htmlFor="prompt">Your prompt</label>
        <textarea
          id="prompt"
          ref={promptField}
          value={prompt}
          onChange={(event) => changeDraft(event.target.value)}
          placeholder="Paste your prompt here. For example: Write a friendly reply to a customer whose repair was delayed."
          rows={8}
          required
        />
        <PromptHealthPanel
          enabled={healthEnabled}
          settings={healthSettings}
          assessment={healthResult}
          checking={healthChecking}
          onToggle={(enabled) => {
            try {
              localStorage.setItem(HEALTH_ENABLED_KEY, enabled ? "on" : "off");
            } catch {
              // The current page still honors the toggle.
            }
            lastAssessed.current = null;
            setHealthEnabled(enabled);
            setHealthResult(null);
          }}
          onSelectSpan={(start, end) => {
            promptField.current?.focus();
            promptField.current?.setSelectionRange(start, end);
          }}
        />
        <div className="form-actions">
          <label className="tier-label" htmlFor="tier">
            Effort
            <select
              id="tier"
              value={tier}
              onChange={(event) => setTier(event.target.value as Tier)}
            >
              <option value="fast">Fast</option>
              <option value="standard">Standard</option>
              <option value="deep">Deep</option>
            </select>
          </label>
          <button
            className="primary"
            type="submit"
            disabled={busy || !prompt.trim()}
          >
            {job ? "Optimizing…" : "Optimize prompt"}
          </button>
        </div>
        <p className="effort-estimate">
          {tierDescriptions[tier]} {estimate}
        </p>
        {selection && (
          <ModelPicker
            catalog={catalog}
            selection={selection}
            onChange={setSelection}
            onSave={() => void saveModelDefaults()}
            busy={busy}
            open={pickerOpen}
            onToggle={setPickerOpen}
            providers={providers?.providers ?? {}}
          />
        )}
      </form>

      {(viewingHistoryResult || resultForEarlierPrompt) && (
        <p className="history-result-context" role="status">
          {viewingHistoryResult
            ? "A saved result is open below."
            : "A result for an earlier prompt is open below."}{" "}
          Your current draft remains in Your prompt.
        </p>
      )}

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {job && (
        <RunProgress
          job={job}
          estimate={progressEstimate}
          onCancel={() => void cancel()}
        />
      )}

      {questions.length > 0 ? (
        <ClarificationPanel
          key={`${result?.run_id}:${JSON.stringify(questions)}`}
          questions={questions}
          onSubmit={(answers) =>
            begin(
              () => startResume(result!.run_id, answers),
              (caught) => {
                const validationError = clarificationValidationError(caught);
                if (validationError) {
                  setClarificationError(validationError);
                } else {
                  setError(
                    caught instanceof Error
                      ? caught.message
                      : "Unable to continue with these answers."
                  );
                }
              }
            )
          }
          onSkip={() => begin(() => startSkip(result!.run_id))}
          busy={busy}
          error={error}
          validationError={clarificationError}
          onValidationErrorDismiss={() => setClarificationError(null)}
        />
      ) : null}

      {!job && result?.status === "failed" && (
        <FailureCard
          result={result}
          onRetry={retry}
          onOpenModels={openModels}
          busy={busy}
        />
      )}

      {!job && result?.status === "completed" && outcome ? (
        <section className="result" aria-live="polite">
          <div className="result-heading">
            <div>
              <p className="eyebrow">RESULT</p>
              <h2>{outcome.headline}</h2>
              {outcome.reason && (
                <p className="outcome-reason">{outcome.reason}</p>
              )}
            </div>
            {result.final_prompt && (
              <button className="secondary" type="button" onClick={copyPrompt}>
                {copied ? "Copied" : "Copy prompt"}
              </button>
            )}
          </div>
          {gaps.length > 0 && (
            <div className="gap-list">
              <p>What would help:</p>
              <ul>
                {gaps.map((gap) => (
                  <li key={gap.key}>{humanize(gap.label)}</li>
                ))}
              </ul>
            </div>
          )}
          {hints.length > 0 && (
            <div className="gap-list hint-list">
              <p>Worth checking:</p>
              <ul>
                {hints.map((hint) => (
                  <li key={hint}>{hint}</li>
                ))}
              </ul>
            </div>
          )}
          {result.final_prompt && (
            <pre className="final-prompt">{result.final_prompt}</pre>
          )}
          {offerDeep && (
            <div className="deep-offer">
              <p>{deepOfferText(result)}</p>
              <button
                className="secondary"
                type="button"
                disabled={busy}
                onClick={() => void begin(() => startDeep(result.run_id))}
              >
                Try a Deep pass
              </button>
            </div>
          )}
          <details className="report">
            <summary>View report</summary>
            <AssumptionEditor
              assumptions={assumptions}
              busy={busy}
              onSave={saveAssumption}
            />
            <p className="run-meta">
              Run {result.run_id} · {(result.timing.total_ms / 1000).toFixed(1)}{" "}
              s · ${result.cost.total.toFixed(4)}
            </p>
            <RunReport result={result} />
          </details>
        </section>
      ) : null}

      <History
        onOpen={openFromHistory}
        refreshKey={`${job?.run_id ?? ""}:${job?.state ?? ""}:${result?.run_id ?? ""}:${result?.status ?? ""}:${result?.final_prompt ?? ""}`}
      />
    </main>
  );
}
