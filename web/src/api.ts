export type ClarificationQuestion = {
  id: string;
  prompt: string;
  options: {
    value: string;
    label: string;
    preselected?: boolean;
    other?: boolean;
  }[];
  default?: string;
  default_answer?: string;
  required_answer?: boolean;
  allow_other?: boolean;
  other_value?: string;
};

export type Assumption = {
  key: string;
  value: string;
  label?: string;
  source?: string;
  confidence?: number;
};

export type Failure = {
  kind: string;
  headline: string;
  hint: string;
  message: string;
  provider?: string;
  model?: string;
  role?: string | null;
  http_status?: number | null;
};

export type ModelInfo = {
  id: string;
  name?: string;
  provider: string;
  safe_for_default?: boolean;
};
export type ModelCatalog = {
  judge: ModelInfo;
  providers: { go: ModelInfo[]; openrouter: ModelInfo[] };
};
export type ModelSettings = {
  judge_model: string;
  writer_model: string;
  strong_check_model: string;
  weak_models: string[];
};

export type PromptHealthSettings = {
  available: boolean;
  default_enabled: boolean;
  debounce_ms: number;
  hourly_allowance_usd: number;
  usage: { rolling_hour_usd: number; refreshes_last_minute: number };
};

export type PromptHealthResult = {
  draft: { revision: number; hash: string };
  status: "complete" | "partial" | "unavailable" | "paused" | "empty";
  reason?: string;
  composite: number | null;
  provisional: boolean;
  dimensions: {
    id: string;
    label: string;
    applicable: boolean | null;
    applicability_probability: number;
    score: number | null;
    normalized: number | null;
  }[];
  coverage: { assessed: number; applicable: number; unknown: number };
  flags: {
    sentence_id: string;
    kind: string;
    start: number;
    end: number;
    text: string;
    probability: number;
    threshold: number;
  }[];
  cache: { hits: number; misses: number };
  usage: {
    rolling_hour_usd: number;
    request_usd: number;
    provider_requests: number;
  };
};
export type ModelSelection = { writer: string; strong: string; weak: string[] };

export type RunOutcome =
  | "converged"
  | "improved_tested"
  | "improved_unverified"
  | "impossible"
  | "failed_operational";

export type RunControlState = "awaiting_approval" | "stopped" | "cancelled";

export type OptimizeResult = {
  status: "completed" | "needs_input" | "failed";
  run_id: string;
  original_prompt?: string;
  final_prompt?: string;
  original_kept?: boolean;
  questions?: ClarificationQuestion[];
  report: Record<string, unknown> & {
    outcome?: RunOutcome | null;
    outcome_reason?: string | null;
    applied_style?: string | null;
    control_state?: RunControlState | null;
    evaluation_evidence?: EvaluationEvidence;
    judgment_provenance?: JudgmentProvenance[];
    capabilities_fired?: CapabilitySummary;
  };
  cost: {
    total: number;
    by_role?: Record<
      string,
      {
        provider?: string;
        model?: string;
        cost?: number;
        cap?: number;
        cap_used?: number;
        cap_remaining?: number;
      }[]
    >;
    cost_by_role?: Record<string, number>;
  };
  timing: { total_ms: number };
};

export type JobRound = { round?: number };
export type RunEvent = {
  cursor: number;
  kind: string;
  summary: string;
  elapsed_ms: number;
  round: number | null;
  stage?: string;
  candidate_id?: string;
  draft?: string;
  diff?: string;
  reasons?: string[];
  checks?: {
    requirement_id: string;
    source: string;
    tested: number;
    failed: number;
    untestable: number;
    reasons: string[];
  }[];
};
export type Job = {
  run_id: string;
  kind: "optimize" | "resume" | "skip" | "continue";
  prompt?: string;
  state: "queued" | "running" | "done" | "interrupted";
  stage: string | null;
  round: JobRound;
  stages_seen: string[];
  elapsed_ms: number;
  remaining_active_ms?: number;
  events?: RunEvent[];
  event_cursor?: number;
  /** Accumulated provider cost in USD from the run's progress events. */
  cost_total?: number;
  cancel_requested: boolean;
  /** True while a requested cancellation is still waiting for active work to stop. */
  cancellation_pending?: boolean;
  result: OptimizeResult | null;
};

export type ProviderState = {
  status: "ok" | "unavailable" | "unknown";
  http_status?: number | null;
  model?: string;
};
export type ProviderReport = {
  providers: Record<string, ProviderState>;
  fallback: { writer: string; strong: string };
};

export type JudgmentCapability =
  | "verify"
  | "screen"
  | "noul"
  | "find"
  | "rerank"
  | "classify"
  | "decide"
  | "compare"
  | "extract"
  | "audit"
  | "review"
  | "gate";

export type JudgmentProvenance = {
  capability: JudgmentCapability | null;
  stage: string;
  candidate_id: string | null;
  round_number: number | null;
  source_round: number | null;
  question_key: string;
  model: string | null;
  raw_answer: unknown;
  usable: boolean;
  probability?: number;
  selected?: string;
};

export type CapabilitySummary = Record<
  JudgmentCapability,
  { count: number; ran: boolean; stages: Record<string, number> }
>;

export type JudgmentEvidence = {
  raw_answer: unknown;
  usable: boolean;
  probability?: number;
  selected?: string;
};

export type EvaluationCandidateEvidence = {
  candidate_id: string;
  round_number: number;
  comparison: Record<string, JudgmentEvidence>;
  verification: Record<string, JudgmentEvidence>;
  audit: Record<string, JudgmentEvidence>;
  rerank: JudgmentEvidence;
  review: JudgmentEvidence;
  score_vector: unknown;
  fidelity: unknown;
  strong_check: unknown;
  downstream_verification: "verified" | "unverified";
  success_tests: unknown[];
  success_test_outputs: unknown[];
  success_test_grade: unknown;
  accept: JudgmentEvidence & {
    threshold: number;
    accepted: boolean;
  };
  eligible: boolean;
  rejection_reasons: string[];
};

export type EvaluationEvidence = {
  candidates: Record<string, EvaluationCandidateEvidence>;
};

/** An HTTP error from the local API, with the server's `detail` when present. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail: unknown = message
  ) {
    super(message);
  }
}

export const connectionLostMessage =
  "Lost connection to the local engine. Check that it is still running. A run in progress may still finish; it will show up in History.";

export async function requestJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(input, init);
  } catch {
    throw new Error(connectionLostMessage);
  }
  if (!response.ok) {
    const body = await response.text();
    let detail = body;
    let detailValue: unknown = body;
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (typeof parsed.detail === "string") {
        detail = parsed.detail;
        detailValue = parsed.detail;
      } else if (parsed.detail !== undefined) {
        detailValue = parsed.detail;
        detail = JSON.stringify(parsed.detail);
      }
    } catch {
      // Not JSON; keep the raw text.
    }
    if (response.status >= 502 && response.status <= 504 && !detail)
      detail = connectionLostMessage;
    throw new ApiError(
      detail || `Request failed (${response.status})`,
      response.status,
      detailValue
    );
  }
  return (await response.json()) as T;
}

function postJson<T>(url: string, body?: unknown): Promise<T> {
  return requestJson<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export type RunLimits = {
  time_limit_s?: number;
  spend_limit_usd?: number;
};

export function startOptimize(
  prompt: string,
  improvementStyle: string,
  modelOverrides?: ModelSelection,
  limits?: RunLimits
): Promise<Job> {
  return postJson<Job>("/api/jobs/optimize", {
    prompt,
    improvement_style: improvementStyle,
    model_overrides: modelOverrides,
    ...(limits?.time_limit_s != null
      ? { time_limit_s: limits.time_limit_s }
      : {}),
    ...(limits?.spend_limit_usd != null
      ? { spend_limit_usd: limits.spend_limit_usd }
      : {}),
  });
}

export function startResume(
  runId: string,
  answers: Record<string, unknown>
): Promise<Job> {
  return postJson<Job>(`/api/jobs/${encodeURIComponent(runId)}/resume`, {
    answers,
  });
}

export function startSkip(runId: string): Promise<Job> {
  return postJson<Job>(`/api/jobs/${encodeURIComponent(runId)}/skip`);
}

export function startContinue(runId: string, limits?: RunLimits): Promise<Job> {
  return postJson<Job>(`/api/jobs/${encodeURIComponent(runId)}/continue`, {
    ...(limits?.time_limit_s != null
      ? { time_limit_s: limits.time_limit_s }
      : {}),
    ...(limits?.spend_limit_usd != null
      ? { spend_limit_usd: limits.spend_limit_usd }
      : {}),
  });
}

export function stopRun(runId: string): Promise<OptimizeResult> {
  return postJson<OptimizeResult>(
    `/api/runs/${encodeURIComponent(runId)}/stop`
  );
}

export function getJob(runId: string, afterCursor = 0): Promise<Job> {
  const cursor = afterCursor > 0 ? `?after_cursor=${afterCursor}` : "";
  return requestJson<Job>(`/api/jobs/${encodeURIComponent(runId)}${cursor}`);
}

export function getActiveJobs(): Promise<Job[]> {
  return requestJson<Job[]>("/api/jobs");
}

export function getRunResult(
  runId: string
): Promise<{ result?: OptimizeResult | null }> {
  return requestJson<{ result?: OptimizeResult | null }>(
    `/api/runs/${encodeURIComponent(runId)}`
  );
}

export function cancelJob(runId: string): Promise<Job> {
  return postJson<Job>(`/api/jobs/${encodeURIComponent(runId)}/cancel`);
}

export function getProviders(probe: boolean): Promise<ProviderReport> {
  return requestJson<ProviderReport>(
    `/api/providers?probe=${probe ? "true" : "false"}`
  );
}

export function getCatalog(): Promise<ModelCatalog> {
  return requestJson<ModelCatalog>("/api/catalog");
}

export function getSettings(): Promise<ModelSettings> {
  return requestJson<ModelSettings>("/api/settings");
}

export function getPromptHealthSettings(): Promise<PromptHealthSettings> {
  return requestJson<PromptHealthSettings>("/api/prompt-health/settings");
}

export function checkPromptHealth(
  prompt: string,
  revision: number,
  sessionId: string,
  signal: AbortSignal
): Promise<PromptHealthResult> {
  return requestJson<PromptHealthResult>("/api/prompt-health", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, revision, session_id: sessionId }),
    signal,
  });
}

export function saveSettings(
  selection: ModelSelection
): Promise<ModelSettings> {
  return requestJson<ModelSettings>("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      writer_model: selection.writer,
      strong_check_model: selection.strong,
      weak_models: selection.weak,
    }),
  });
}

export function updateAssumption(
  runId: string,
  assumption: Assumption
): Promise<OptimizeResult> {
  return requestJson<OptimizeResult>(
    `/api/runs/${encodeURIComponent(runId)}/assumption`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ assumption }),
    }
  );
}
