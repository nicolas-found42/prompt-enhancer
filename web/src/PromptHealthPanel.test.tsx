import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { PromptHealthResult, PromptHealthSettings } from "./api";
import PromptHealthPanel from "./PromptHealthPanel";

const settings: PromptHealthSettings = {
  available: true,
  default_enabled: true,
  debounce_ms: 600,
  hourly_allowance_usd: 0.05,
  usage: { rolling_hour_usd: 0, refreshes_last_minute: 0 },
};

const assessment: PromptHealthResult = {
  draft: { revision: 1, hash: "draft" },
  status: "unavailable",
  reason: "too many sentence checks for one refresh",
  composite: null,
  provisional: false,
  dimensions: [],
  coverage: { assessed: 0, applicable: 0, unknown: 0 },
  flags: [],
  cache: { hits: 0, misses: 0 },
  usage: { rolling_hour_usd: 0, request_usd: 0, provider_requests: 0 },
};

it("explains why live prompt health is unavailable", () => {
  render(
    <PromptHealthPanel
      enabled
      settings={settings}
      assessment={assessment}
      checking={false}
      onToggle={() => {}}
      onSelectSpan={() => {}}
    />
  );

  expect(
    screen.getByText(
      "Live checks unavailable: too many sentence checks for one refresh"
    )
  ).toBeInTheDocument();
});
