import { mergeJobEvents } from "./jobEvents";
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
  getJob,
  getProviders,
  getPromptHealthSettings,
  getRunResult,
  getSettings,
  saveSettings,
  startContinue,
  startOptimize,
  startResume,
  startSkip,
  stopRun,
  updateAssumption,
  type Assumption,
  type Job,
  type ModelCatalog,
  type ModelSelection,
  type OptimizeResult,
  type ProviderReport,
  type PromptHealthResult,
  type PromptHealthSettings,
  type RunLimits,
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
  humanize,
  failureOf,
  outcomeOf,
  pauseOf,
  pauseText,
  possibleGapHints,
  record,
} from "./outcome";
import RunReport from "./RunReport";
import {
  COMMON_STYLES,
  MORE_STYLES,
  STYLE_LABELS,
  type ImprovementStyle,
} from "./styles";

const ACTIVE_RUN_KEY = "prompt-enhancer.active-run";
const LAST_RESULT_KEY = "prompt-enhancer.last-result";
const DRAFT_KEY = "prompt-enhancer.draft";
const HEALTH_ENABLED_KEY = "prompt-enhancer.live-health";
const HEALTH_SESSION_KEY = "prompt-enhancer.health-session";
const HEALTH_RETRY_INITIAL_MS = 3000;
const HEALTH_RETRY_MAX_MS = 60000;
const POLL_MS = 1000;
// A result shown this recently comes back after a reload instead of vanishing.
const RESTORE_RESULT_MS = 30 * 60 * 1000;

function isTerminalJob(job: Job): boolean {
  return job.state === "done" || job.state === "interrupted";
}

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

/** Scroll a run's panel into view, moving focus to it when it holds the outcome. */
function bringIntoView(panel: HTMLElement | null, focus: boolean) {
  if (!panel) return;
  if (focus) panel.focus({ preventScroll: true });
  const reduceMotion = window.matchMedia?.(
    "(prefers-reduced-motion: reduce)"
  ).matches;
  panel.scrollIntoView({
    behavior: reduceMotion ? "auto" : "smooth",
    block: "start",
  });
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
  const [healthRetry, setHealthRetry] = useState(0);
  const [pageVisible, setPageVisible] = useState(!document.hidden);
  const [healthSessionId] = useState(healthSession);
  const promptRef = useRef(prompt);
  const revisionRef = useRef(0);
  const lastAssessed = useRef<string | null>(null);
  const pausedDelay = useRef(HEALTH_RETRY_INITIAL_MS);
  const promptField = useRef<HTMLTextAreaElement | null>(null);
  const [style, setStyle] = useState<ImprovementStyle>("auto");
  const [spendLimit, setSpendLimit] = useState("");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [viewingHistoryResult, setViewingHistoryResult] = useState(false);
  const [historyNavigation, setHistoryNavigation] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clarificationError, setClarificationError] =
    useState<ClarificationValidationError | null>(null);
  const [saving, setSaving] = useState(false);
  const [copied, setCopied] = useState(false);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [selection, setSelection] = useState<ModelSelection | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [modelSwitch, setModelSwitch] = useState<{
    writer: string;
    strong: string;
  } | null>(null);
  const [providers, setProviders] = useState<ProviderReport | null>(null);
  const lastRequest = useRef<{
    prompt: string;
    style: ImprovementStyle;
    spendLimit: string;
  } | null>(null);
  const composer = useRef<HTMLFormElement | null>(null);
  const draftWasSet = useRef(initialDraft.present);
  const busy = job !== null || saving;
  // Scroll only for runs the user started here, never for a reattached one.
  const followRun = useRef(false);

  const changeDraft = useCallback((next: string) => {
    draftWasSet.current = true;
    saveDraft(next);
    promptRef.current = next;
    revisionRef.current += 1;
    setDraftRevision(revisionRef.current);
    lastAssessed.current = null;
    pausedDelay.current = HEALTH_RETRY_INITIAL_MS;
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
    pausedDelay.current = HEALTH_RETRY_INITIAL_MS;
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
    let retryTimer: number | undefined;
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
          if (assessment.status === "paused") {
            const delay = pausedDelay.current;
            pausedDelay.current = Math.min(delay * 2, HEALTH_RETRY_MAX_MS);
            retryTimer = window.setTimeout(() => {
              setHealthRetry((value) => value + 1);
            }, delay);
          } else {
            pausedDelay.current = HEALTH_RETRY_INITIAL_MS;
            lastAssessed.current = currentPrompt;
          }
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
      window.clearTimeout(retryTimer);
      controller.abort();
    };
  }, [
    prompt,
    draftRevision,
    healthSettings,
    healthEnabled,
    pageVisible,
    healthSessionId,
    healthRetry,
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
        if (isTerminalJob(current)) finish(current);
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
    if (!job || isTerminalJob(job)) return;
    const runId = job.run_id;
    let current = job;
    let pending = false;
    let active = true;
    const timer = window.setInterval(() => {
      if (pending) return;
      pending = true;
      void getJob(runId, current.event_cursor ?? 0)
        .then((latest) => {
          if (!active) return;
          current = mergeJobEvents(current, latest);
          if (isTerminalJob(current)) finish(current);
          else setJob(current);
        })
        .catch((caught) => {
          if (caught instanceof ApiError && caught.status === 404) {
            // The job is gone, but a budget-paused run survives a restart
            // in history: offer its Continue/Stop choice instead of loss.
            void getRunResult(runId)
              .then((found) => {
                const resumed = found.result ?? null;
                setJob(null);
                rememberRun(null);
                if (resumed && pauseOf(resumed)) setResult(resumed);
                else
                  setError(
                    "The local engine restarted, so this run was lost. Your prompt is still in the box; press Optimize to try again."
                  );
              })
              .catch(() => {
                setJob(null);
                rememberRun(null);
                setError(
                  "The local engine restarted, so this run was lost. Your prompt is still in the box; press Optimize to try again."
                );
              });
          } else {
            setError(
              caught instanceof Error
                ? caught.message
                : "Lost track of the run."
            );
          }
        })
        .finally(() => {
          pending = false;
        });
    }, POLL_MS);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [job?.run_id, job?.state, finish]);

  async function begin(
    start: () => Promise<Job>,
    onFailure?: (caught: unknown) => void
  ) {
    setError(null);
    setClarificationError(null);
    setCopied(false);
    setViewingHistoryResult(false);
    setModelSwitch(null);
    try {
      const started = await start();
      rememberRun({ runId: started.run_id, prompt });
      store(LAST_RESULT_KEY, null);
      setResult(null);
      followRun.current = true;
      setJob(started);
    } catch (caught) {
      if (onFailure) onFailure(caught);
      else
        setError(
          caught instanceof Error ? caught.message : "Unable to start the run."
        );
    }
  }

  function runLimits(raw: string): RunLimits | undefined {
    const trimmed = raw.trim();
    if (!trimmed) return undefined;
    const value = Number(trimmed);
    if (!Number.isFinite(value) || value < 0) return undefined;
    return { spend_limit_usd: value };
  }

  function submit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (spendLimit.trim() && !runLimits(spendLimit)) {
      setError("Spend limit must be a number of dollars, at least 0.");
      return;
    }
    lastRequest.current = { prompt, style, spendLimit };
    const limits = runLimits(spendLimit);
    void begin(() =>
      startOptimize(prompt, style, selection ?? undefined, limits)
    );
  }

  function retry() {
    const previous = lastRequest.current;
    if (previous) {
      changeDraft(previous.prompt);
      setStyle(previous.style);
      setSpendLimit(previous.spendLimit);
      const limits = runLimits(previous.spendLimit);
      void begin(() =>
        startOptimize(
          previous.prompt,
          previous.style,
          selection ?? undefined,
          limits
        )
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

  async function stopPaused() {
    if (!result) return;
    setSaving(true);
    setError(null);
    try {
      setResult(await stopRun(result.run_id));
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Unable to stop the run."
      );
    } finally {
      setSaving(false);
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
    followRun.current = false;
    setResult(opened);
    setCopied(false);
    setClarificationError(null);
    changeDraft(prompt);
    setViewingHistoryResult(true);
    setHistoryNavigation((current) => current + 1);
    if (opened.status !== "needs_input")
      window.scrollTo({ top: 0, behavior: "smooth" });
  }

  useEffect(() => {
    if (!viewingHistoryResult || result?.status !== "needs_input") return;
    const panel = document.getElementById("clarification-panel");
    if (!panel) return;
    panel.focus({ preventScroll: true });
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [historyNavigation, result?.status, viewingHistoryResult]);

  const showingProgress = job !== null;
  useEffect(() => {
    if (!showingProgress || !followRun.current) return;
    bringIntoView(document.getElementById("run-progress"), false);
  }, [showingProgress]);

  const outcomeId = !job && result ? result.run_id : null;
  useEffect(() => {
    if (!outcomeId || !followRun.current) return;
    followRun.current = false;
    bringIntoView(
      document.getElementById("run-outcome") ??
        document.getElementById("clarification-panel"),
      true
    );
  }, [outcomeId]);

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
  const fallbackProviderUnavailable =
    providers?.providers.openrouter?.status === "unavailable";
  const canSwitchToFallbackModels = Boolean(
    selection &&
    providers &&
    goRoles.length > 0 &&
    !fallbackProviderUnavailable &&
    (!goIds.has(selection.writer) ||
      (providers.fallback.writer.trim() &&
        !goIds.has(providers.fallback.writer))) &&
    (!goIds.has(selection.strong) ||
      (providers.fallback.strong.trim() &&
        !goIds.has(providers.fallback.strong)))
  );

  function switchToFallbackModels() {
    if (!selection || !providers) return;
    const openRouterIds = new Set(
      (catalog?.providers.openrouter ?? []).map((model) => model.id)
    );
    const next = {
      writer: goIds.has(selection.writer)
        ? providers.fallback.writer
        : selection.writer,
      strong: goIds.has(selection.strong)
        ? providers.fallback.strong
        : selection.strong,
      weak: selection.weak.filter(
        (id) => !goIds.has(id) || openRouterIds.has(id)
      ),
    };
    setSelection(next);
    setModelSwitch({ writer: next.writer, strong: next.strong });
    // A refusal from the models just replaced is out of date; other failures stay.
    if (
      !job &&
      result?.status === "failed" &&
      failureOf(result).provider === "go"
    ) {
      setResult(null);
    }
  }

  const questions: ClarificationQuestion[] =
    !job && result?.status === "needs_input" ? (result.questions ?? []) : [];
  const paused = !job && result ? pauseOf(result) : null;
  const assumptions = useMemo(
    () => (result ? reportAssumptions(result) : []),
    [result]
  );
  const outcome =
    result &&
    (result.status === "completed" || typeof result.report.outcome === "string")
      ? outcomeOf(result)
      : null;
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

  return (
    <main className="shell">
      <header className="hero">
        <p className="eyebrow">LOCAL PROMPT WORKBENCH</p>
        <h1>Make your prompt clearer.</h1>
        <p className="lede">
          Paste one prompt, choose how to improve it, and get a clean result you
          can copy.
        </p>
      </header>

      {modelSwitch && (
        <div className="switch-confirmation" role="status">
          <p>
            Switched to different models (writer {modelSwitch.writer}, strong
            check {modelSwitch.strong}). Your prompt is unchanged; press{" "}
            <strong>Optimize prompt</strong> to run it again.
          </p>
        </div>
      )}

      {goUnavailable && goRoles.length > 0 && (
        <div className="banner" role="status">
          <div>
            <p>
              Some models selected for this app can&apos;t be reached, so your
              prompt can&apos;t run.{" "}
              {canSwitchToFallbackModels
                ? "Use the button to try a different set of models."
                : "Open Model choices below to choose different models."}
            </p>
            <details className="provider-warning-details">
              <summary>Show troubleshooting details</summary>
              <p>Provider: OpenCode Go</p>
              {providers?.providers.go?.http_status != null && (
                <p>
                  Provider response: HTTP {providers.providers.go.http_status}
                </p>
              )}
              <p>Selected models: {goRoles.join(" and ")}</p>
            </details>
          </div>
          {canSwitchToFallbackModels && (
            <button
              className="primary"
              type="button"
              onClick={switchToFallbackModels}
            >
              Try different models
            </button>
          )}
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
            pausedDelay.current = HEALTH_RETRY_INITIAL_MS;
            setHealthEnabled(enabled);
            setHealthResult(null);
          }}
          onSelectSpan={(start, end) => {
            promptField.current?.focus();
            promptField.current?.setSelectionRange(start, end);
          }}
        />
        <div className="form-actions">
          <label className="style-label" htmlFor="improvement-style">
            Improvement style
            <select
              id="improvement-style"
              value={style}
              onChange={(event) =>
                setStyle(event.target.value as ImprovementStyle)
              }
            >
              {COMMON_STYLES.map((value) => (
                <option key={value} value={value}>
                  {STYLE_LABELS[value]}
                </option>
              ))}
              <optgroup label="More styles">
                {MORE_STYLES.map((value) => (
                  <option key={value} value={value}>
                    {STYLE_LABELS[value]}
                  </option>
                ))}
              </optgroup>
            </select>
          </label>
          <button
            className="primary"
            type="submit"
            disabled={busy || !prompt.trim()}
            aria-describedby={!prompt.trim() ? "optimize-hint" : undefined}
          >
            {job ? "Optimizing…" : "Optimize prompt"}
          </button>
        </div>
        <div className="form-actions">
          <label className="limit-label" htmlFor="spend-limit">
            Spend limit (USD, optional)
            <input
              id="spend-limit"
              type="number"
              min={0}
              step={0.01}
              inputMode="decimal"
              value={spendLimit}
              onChange={(event) => setSpendLimit(event.target.value)}
              placeholder="No limit"
            />
          </label>
        </div>
        {!prompt.trim() && (
          <p className="composer-hint" id="optimize-hint">
            Enter a prompt to enable Optimize prompt.
          </p>
        )}
        <p className="loop-estimate">
          Each run has 150 seconds of active time to find a useful improvement.
          Answering clarification questions pauses that clock. Your optional
          limits pause the run for approval; the active deadline ends it.
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

      {job && <RunProgress job={job} onCancel={() => void cancel()} />}

      {paused ? (
        <section
          id="paused-panel"
          className="result paused"
          aria-live="polite"
          tabIndex={-1}
        >
          <p className="eyebrow">PAUSED</p>
          <h2>Paused at your limit</h2>
          <p>{pauseText(paused)}</p>
          <div className="paused-actions">
            <button
              className="primary"
              type="button"
              disabled={busy}
              onClick={() => void begin(() => startContinue(result!.run_id))}
            >
              Continue
            </button>
            <button
              className="secondary"
              type="button"
              disabled={busy}
              onClick={() => void stopPaused()}
            >
              Stop
            </button>
          </div>
        </section>
      ) : null}

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

      {!job && outcome && result ? (
        <section
          id="run-outcome"
          className="result"
          aria-live="polite"
          tabIndex={-1}
        >
          <div className="result-heading">
            <div>
              <p className="eyebrow">RESULT</p>
              <h2>{outcome.headline}</h2>
              {outcome.appliedStyle && (
                <p>Applied style: {outcome.appliedStyle}.</p>
              )}
              {outcome.reason && (
                <p className="outcome-reason">{outcome.reason}</p>
              )}
              {outcome.controlState && (
                <p className="control-state">
                  Run state: {outcome.controlState}.
                </p>
              )}
            </div>
            {result.final_prompt && (
              <div className="copy-guidance">
                <button
                  className="secondary"
                  type="button"
                  onClick={copyPrompt}
                >
                  {copied ? "Copied" : "Copy prompt"}
                </button>
                <p>
                  After copying, paste this prompt into an AI chat or another
                  tool that accepts prompts.
                </p>
              </div>
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
