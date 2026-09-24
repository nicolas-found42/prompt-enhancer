import { defineConfig, devices } from "@playwright/test";

// Override when 8765 is taken by another local service.
const apiPort = process.env.E2E_API_PORT ?? "8765";
const apiTarget = `http://127.0.0.1:${apiPort}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  workers: process.env.CI ? 1 : undefined,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:5174",
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: [
    {
      command: "uv run --project .. python ../tests/e2e_server.py",
      url: `http://127.0.0.1:${apiPort}/health`,
      reuseExistingServer: false,
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5174",
      env: { PROMPT_ENHANCER_API_TARGET: apiTarget },
      url: "http://127.0.0.1:5174",
      reuseExistingServer: false,
    },
  ],
});
