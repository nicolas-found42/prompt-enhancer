export type Tier = "fast" | "standard" | "deep";

export type ClarificationQuestion = {
  id: string;
  prompt: string;
  options: { value: string; label: string; preselected?: boolean; other?: boolean }[];
  default?: string;
  default_answer?: string;
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

export type ModelInfo = { id: string; name?: string; provider: string; safe_for_default?: boolean };
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
export type ModelSelection = { writer: string; strong: string; weak: string[] };

export type OptimizeResult = {
  status: "completed" | "needs_input" | "failed";
  run_id: string;
  original_prompt?: string;
  final_prompt?: string;
  original_kept?: boolean;
  questions?: ClarificationQuestion[];
  report: Record<string, unknown>;
  cost: {
    total: number;
    by_role?: Record<string, { provider?: string; model?: string; cost?: number; cap?: number; cap_used?: number; cap_remaining?: number }[]>;
    cost_by_role?: Record<string, number>;
  };
  timing: { total_ms: number };
};

export type JobRound = { round?: number; max_rounds?: number };
export type Job = {
  run_id: string;
  kind: "optimize" | "resume" | "skip" | "deep";
  state: "queued" | "running" | "done";
  stage: string | null;
  round: JobRound;
  stages_seen: string[];
  elapsed_ms: number;
  cancel_requested: boolean;
  result: OptimizeResult | null;
};

export type ProviderState = { status: "ok" | "unavailable" | "unknown"; http_status?: number | null; model?: string };
export type ProviderReport = {
  providers: Record<string, ProviderState>;
  fallback: { writer: string; strong: string };
};

export type TierEstimate = { runs: number; minutes: [number, number]; cost: [number, number] };

/** An HTTP error from the local API, with the server's `detail` when present. */
export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export const connectionLostMessage =
  "Lost connection to the local engine. Check that it is still running. A run in progress may still finish; it will show up in History.";

export async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(input, init);
  } catch {
    throw new Error(connectionLostMessage);
  }
  if (!response.ok) {
    const body = await response.text();
    let detail = body;
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      // Not JSON; keep the raw text.
    }
    if (response.status >= 502 && response.status <= 504 && !detail) detail = connectionLostMessage;
    throw new ApiError(detail || `Request failed (${response.status})`, response.status);
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

export function startOptimize(prompt: string, tier: Tier, modelOverrides?: ModelSelection): Promise<Job> {
  return postJson<Job>("/api/jobs/optimize", { prompt, tier, model_overrides: modelOverrides });
}

export function startResume(runId: string, answers: Record<string, unknown>): Promise<Job> {
  return postJson<Job>(`/api/jobs/${encodeURIComponent(runId)}/resume`, { answers });
}

export function startSkip(runId: string): Promise<Job> {
  return postJson<Job>(`/api/jobs/${encodeURIComponent(runId)}/skip`);
}

export function startDeep(runId: string): Promise<Job> {
  return postJson<Job>(`/api/jobs/${encodeURIComponent(runId)}/deep`);
}

export function getJob(runId: string): Promise<Job> {
  return requestJson<Job>(`/api/jobs/${encodeURIComponent(runId)}`);
}

export function getActiveJobs(): Promise<Job[]> {
  return requestJson<Job[]>("/api/jobs");
}

export function cancelJob(runId: string): Promise<Job> {
  return postJson<Job>(`/api/jobs/${encodeURIComponent(runId)}/cancel`);
}

export function getProviders(probe: boolean): Promise<ProviderReport> {
  return requestJson<ProviderReport>(`/api/providers?probe=${probe ? "true" : "false"}`);
}

export function getEstimates(): Promise<Partial<Record<Tier, TierEstimate>>> {
  return requestJson<Partial<Record<Tier, TierEstimate>>>("/api/estimates");
}

export function getCatalog(): Promise<ModelCatalog> {
  return requestJson<ModelCatalog>("/api/catalog");
}

export function getSettings(): Promise<ModelSettings> {
  return requestJson<ModelSettings>("/api/settings");
}

export function saveSettings(selection: ModelSelection): Promise<ModelSettings> {
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

export function updateAssumption(runId: string, assumption: Assumption): Promise<OptimizeResult> {
  return requestJson<OptimizeResult>(`/api/runs/${encodeURIComponent(runId)}/assumption`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ assumption }),
  });
}
