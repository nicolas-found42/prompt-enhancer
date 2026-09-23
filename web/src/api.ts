export type Tier = "fast" | "standard" | "deep";

export type OptimizeResult = {
  status: "completed" | "needs_input";
  run_id: string;
  final_prompt?: string;
  original_kept?: boolean;
  report: Record<string, unknown>;
  cost: { total: number; by_role?: Record<string, number> };
  timing: { total_ms: number };
};

export async function optimizePrompt(prompt: string, tier: Tier): Promise<OptimizeResult> {
  const response = await fetch("/api/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, tier }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return (await response.json()) as OptimizeResult;
}
