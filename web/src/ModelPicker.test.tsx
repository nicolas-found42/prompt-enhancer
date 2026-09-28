import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import type { ModelCatalog, ModelSelection, ProviderState } from "./api";
import ModelPicker from "./ModelPicker";

const catalog: ModelCatalog = {
  judge: { id: "typesafe/jev-1.13", provider: "openrouter" },
  providers: {
    go: [
      { id: "go-writer", name: "Go writer", provider: "go" },
      { id: "go-weak", name: "Go test", provider: "go" },
    ],
    openrouter: [
      { id: "or-writer", name: "Router writer", provider: "openrouter" },
      { id: "or-weak", name: "Router test", provider: "openrouter" },
    ],
  },
};

const selection: ModelSelection = {
  writer: "go-writer",
  strong: "go-writer",
  weak: ["or-weak"],
};

const providers: Record<string, ProviderState> = {
  go: { status: "unavailable", http_status: 403 },
  openrouter: { status: "ok" },
};

function renderPicker(open: boolean, onChange = vi.fn()) {
  const onToggle = vi.fn();
  const result = render(
    <ModelPicker
      catalog={catalog}
      selection={selection}
      onChange={onChange}
      onSave={vi.fn()}
      busy={false}
      open={open}
      onToggle={onToggle}
      providers={providers}
    />
  );
  return { ...result, onChange, onToggle };
}

it("says model choices are optional and keeps controls closed on first view", () => {
  const { container } = renderPicker(false);

  expect(screen.getByText("Optional · you can ignore this")).toBeVisible();
  expect(container.querySelector("details.model-picker")).toHaveProperty(
    "open",
    false
  );
});

it("explains each role in plain language when the optional picker is opened", () => {
  renderPicker(true);

  expect(
    screen.getByText(
      "Checks what your prompt needs and whether a rewrite keeps your meaning."
    )
  ).toBeVisible();
  expect(
    screen.getByText("Writes clearer versions of your prompt.")
  ).toBeVisible();
  expect(
    screen.getByText(
      "Checks that a rewrite still works at least as well as your original."
    )
  ).toBeVisible();
  expect(
    screen.getByText("Check whether a rewrite helps on less expensive models.")
  ).toBeVisible();
});

it("groups unavailable choices while keeping them manually selectable", async () => {
  const user = userEvent.setup();
  const onChange = vi.fn();
  const { container } = renderPicker(true, onChange);

  const writer = screen.getByRole("combobox", { name: "Writer" });
  expect(
    within(writer).getByRole("group", {
      name: "Provider reports unavailable",
    })
  ).toContainElement(
    within(writer).getByRole("option", {
      name: "Go writer · go-writer (OpenCode Go)",
    })
  );
  expect(
    within(writer).getByRole("group", { name: "Other model choices" })
  ).toContainElement(
    within(writer).getByRole("option", {
      name: "Router writer · or-writer (OpenRouter)",
    })
  );

  await user.selectOptions(writer, "or-writer");
  expect(onChange).toHaveBeenCalledWith({ ...selection, writer: "or-writer" });

  const unavailableSummary = container.querySelector(
    ".unavailable-model-choices summary"
  );
  if (!unavailableSummary)
    throw new Error("Unavailable model group is missing");
  fireEvent.click(unavailableSummary);
  const unavailableWeakChoice = screen.getByRole("checkbox", {
    name: "Go test, model go-weak, OpenCode Go, unavailable",
  });
  await user.click(unavailableWeakChoice);
  expect(onChange).toHaveBeenLastCalledWith({
    ...selection,
    weak: ["or-weak", "go-weak"],
  });
});
