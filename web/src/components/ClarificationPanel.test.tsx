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

it.each(["", "   \t\n"])(
  "keeps a blank Other answer in the form and focuses it for correction (%j)",
  async (text) => {
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
    const other = screen.getByRole("textbox", {
      name: "Other answer for What should the assistant do?",
    });
    if (text) await user.type(other, text);
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(other).toHaveFocus();
    expect(other).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Enter an answer for this question."
    );
  }
);

it("keeps a server error until its own question changes", async () => {
  const onValidationErrorDismiss = vi.fn();
  const user = userEvent.setup();
  render(
    <ClarificationPanel
      questions={[
        {
          id: "goal",
          prompt: "What should the assistant do?",
          options: [{ value: "summarize", label: "Summarize" }],
          default: "other",
        },
        {
          id: "audience",
          prompt: "Who is this for?",
          options: [{ value: "team", label: "The team" }],
          default: "other",
        },
      ]}
      onSubmit={vi.fn()}
      onSkip={vi.fn()}
      validationError={{ questionId: "goal", message: "Fix the goal answer." }}
      onValidationErrorDismiss={onValidationErrorDismiss}
    />
  );

  await user.type(
    screen.getByRole("textbox", { name: "Other answer for Who is this for?" }),
    "Project leads"
  );
  await user.click(screen.getByRole("radio", { name: "The team" }));

  expect(onValidationErrorDismiss).not.toHaveBeenCalled();
  expect(screen.getByText("Fix the goal answer.")).toBeVisible();

  await user.type(
    screen.getByRole("textbox", {
      name: "Other answer for What should the assistant do?",
    }),
    "Compare drafts"
  );
  expect(onValidationErrorDismiss).toHaveBeenCalled();
});
