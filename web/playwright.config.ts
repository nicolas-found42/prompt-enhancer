import { defineConfig, devices } from "@playwright/test";

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
      url: "http://127.0.0.1:8765/health",
      reuseExistingServer: false,
    },
    {
      command: "PROMPT_ENHANCER_API_TARGET=http://127.0.0.1:8765 npm run dev -- --host 127.0.0.1 --port 5174",
      url: "http://127.0.0.1:5174",
      reuseExistingServer: false,
    },
  ],
});
