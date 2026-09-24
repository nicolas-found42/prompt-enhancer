import { useState } from "react";
import type { ModelCatalog, ModelInfo, ModelSelection, ProviderState } from "./api";

type Props = {
  catalog: ModelCatalog | null;
  selection: ModelSelection;
  onChange: (selection: ModelSelection) => void;
  onSave: () => void;
  busy: boolean;
  open: boolean;
  onToggle: (open: boolean) => void;
  providers: Record<string, ProviderState>;
};

const MATCH_LIMIT = 30;

function optionLabel(model: ModelInfo, providers: Record<string, ProviderState>): string {
  const unavailable = providers[model.provider]?.status === "unavailable";
  return `${model.name ?? model.id} (${model.provider}${unavailable ? " · unavailable" : ""})`;
}

export default function ModelPicker({ catalog, selection, onChange, onSave, busy, open, onToggle, providers }: Props) {
  const [query, setQuery] = useState("");
  const available = [
    ...(catalog?.providers.go ?? []),
    ...(catalog?.providers.openrouter ?? []),
  ];
  const models = [...available];
  for (const id of [selection.writer, selection.strong, ...selection.weak]) {
    if (id && !models.some((model) => model.id === id)) {
      models.push({ id, provider: "configured" });
    }
  }
  const byId = new Map(models.map((model) => [model.id, model]));
  const chosenWeak = selection.weak.map((id) => byId.get(id) ?? { id, provider: "configured" });
  const needle = query.trim().toLowerCase();
  const unchosen = models.filter((model) => !selection.weak.includes(model.id));
  const matches = needle
    ? unchosen.filter((model) => `${model.name ?? ""} ${model.id} ${model.provider}`.toLowerCase().includes(needle))
    : unchosen;
  const shown = matches.slice(0, MATCH_LIMIT);

  function toggleWeak(id: string, checked: boolean) {
    onChange({
      ...selection,
      weak: checked ? [...selection.weak, id] : selection.weak.filter((item) => item !== id),
    });
  }

  return (
    <details className="model-picker" open={open} onToggle={(event) => onToggle(event.currentTarget.open)}>
      <summary>Model choices</summary>
      <div className="model-picker-intro">
        <p>Change models for this run, or save these choices as your defaults. The judge model is fixed.</p>
        <button className="secondary" type="button" disabled={busy || selection.weak.length === 0} onClick={onSave}>
          Save model defaults
        </button>
      </div>
      <p>Judge: {catalog?.judge.id ?? "typesafe/jev-1.13"} (fixed)</p>
      <label htmlFor="writer-model">Writer</label>
      <select id="writer-model" value={selection.writer} onChange={(event) => onChange({ ...selection, writer: event.target.value })}>
        {models.map((model) => <option key={model.id} value={model.id}>{optionLabel(model, providers)}</option>)}
      </select>
      <label htmlFor="strong-model">Strong check</label>
      <select id="strong-model" value={selection.strong} onChange={(event) => onChange({ ...selection, strong: event.target.value })}>
        {models.map((model) => <option key={model.id} value={model.id}>{optionLabel(model, providers)}</option>)}
      </select>
      <fieldset>
        <legend>Weak panel ({selection.weak.length} selected)</legend>
        {chosenWeak.map((model) => (
          <label key={model.id} className="chosen">
            <input type="checkbox" checked onChange={(event) => toggleWeak(model.id, event.target.checked)} />
            {optionLabel(model, providers)}
          </label>
        ))}
        <label htmlFor="weak-search" className="weak-search-label">Add models</label>
        <input
          id="weak-search"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search models, e.g. llama or openrouter"
        />
        {shown.map((model) => (
          <label key={model.id}>
            <input type="checkbox" checked={false} onChange={(event) => toggleWeak(model.id, event.target.checked)} />
            {optionLabel(model, providers)}
          </label>
        ))}
        <p className="match-count">
          {matches.length === 0
            ? "No models match."
            : matches.length > shown.length
              ? `Showing ${shown.length} of ${matches.length}. Type to narrow the list.`
              : `Showing all ${matches.length}.`}
        </p>
      </fieldset>
    </details>
  );
}
