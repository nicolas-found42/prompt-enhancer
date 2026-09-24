import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { ClarificationPanel } from "./ClarificationPanel";

it("submits the user's custom answer for a missing goal", async () => {
  const onSubmit = vi.fn();
  const user = userEvent.setup();
  render(
    <ClarificationPanel
      questions={[
        {
          id: "goal",
          prompt: "What should the assistant do?",
          options: [{ value: "summarize", label: "Summarize" }],
        },
      ]}
      onSubmit={onSubmit}
      onSkip={vi.fn()}
    />
  );

  await user.click(screen.getByRole("radio", { name: "Other" }));
  await user.type(
    screen.getByRole("textbox", {
      name: "Other answer for What should the assistant do?",
    }),
    "Compare the two drafts"
  );
  await user.click(screen.getByRole("button", { name: "Continue" }));

  expect(onSubmit).toHaveBeenCalledWith({
    goal: { value: "other", text: "Compare the two drafts" },
  });
});
