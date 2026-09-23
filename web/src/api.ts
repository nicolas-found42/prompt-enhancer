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
  source?: string;
  confidence?: number;
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
  status: "completed" | "needs_input";
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

export async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed (${response.status})`);
  }
  return (await response.json()) as T;
}

export function startDeepPass(runId: string): Promise<OptimizeResult> {
  return requestJson<OptimizeResult>(`/api/runs/${encodeURIComponent(runId)}/deep`, { method: "POST" });
}

export function optimizePrompt(prompt: string, tier: Tier, modelOverrides?: ModelSelection): Promise<OptimizeResult> {
  return requestJson<OptimizeResult>("/api/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, tier, model_overrides: modelOverrides }),
  });
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

export function resumeRun(runId: string, answers: Record<string, unknown>): Promise<OptimizeResult> {
  return requestJson<OptimizeResult>(`/api/optimize/resume/${encodeURIComponent(runId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answers }),
  });
}

export function skipClarification(runId: string): Promise<OptimizeResult> {
  return requestJson<OptimizeResult>(`/api/optimize/skip/${encodeURIComponent(runId)}`, {
    method: "POST",
  });
}

export function updateAssumption(runId: string, assumption: Assumption): Promise<OptimizeResult> {
  return requestJson<OptimizeResult>(`/api/runs/${encodeURIComponent(runId)}/assumption`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ assumption }),
  });
}
