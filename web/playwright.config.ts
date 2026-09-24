import { defineConfig, devices } from "@playwright/test";

// Override when 8765 is taken by another local service.
const apiPort = process.env.E2E_API_PORT ?? "8765";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:5174",
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: [
    {
      command: "../.venv/bin/python ../tests/e2e_server.py",
      url: `http://127.0.0.1:${apiPort}/health`,
      reuseExistingServer: false,
    },
    {
      command: `PROMPT_ENHANCER_API_TARGET=http://127.0.0.1:${apiPort} npm run dev -- --host 127.0.0.1 --port 5174`,
      url: "http://127.0.0.1:5174",
      reuseExistingServer: false,
    },
  ],
});
