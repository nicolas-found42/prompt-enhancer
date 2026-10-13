import { useState } from "react";
import type {
  ModelCatalog,
  ModelInfo,
  ModelSelection,
  ProviderState,
} from "./api";

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
// The engine runs a five-entry weak panel; a model may fill several entries.
export const WEAK_PANEL_SIZE = 5;
const providerNames: Record<string, string> = {
  go: "OpenCode Go",
  openrouter: "OpenRouter",
};

function providerName(provider: string): string {
  return providerNames[provider] ?? provider;
}

function isUnavailable(
  model: ModelInfo,
  providers: Record<string, ProviderState>
): boolean {
  return providers[model.provider]?.status === "unavailable";
}

function optionLabel(model: ModelInfo): string {
  const name = model.name ?? model.id;
  const id = name === model.id ? "" : ` · ${model.id}`;
  return `${name}${id} (${providerName(model.provider)})`;
}

function ModelOptions({
  models,
  providers,
}: {
  models: ModelInfo[];
  providers: Record<string, ProviderState>;
}) {
  const otherModels = models.filter(
    (model) => !isUnavailable(model, providers)
  );
  const unavailableModels = models.filter((model) =>
    isUnavailable(model, providers)
  );

  return (
    <>
      {otherModels.length > 0 && (
        <optgroup label="Other model choices">
          {otherModels.map((model) => (
            <option key={model.id} value={model.id}>
              {optionLabel(model)}
            </option>
          ))}
        </optgroup>
      )}
      {unavailableModels.length > 0 && (
        <optgroup label="Provider reports unavailable">
          {unavailableModels.map((model) => (
            <option key={model.id} value={model.id}>
              {optionLabel(model)}
            </option>
          ))}
        </optgroup>
      )}
    </>
  );
}

function ModelChoice({
  model,
  checked,
  unavailable,
  again = false,
  onToggle,
}: {
  model: ModelInfo;
  checked: boolean;
  unavailable: boolean;
  again?: boolean;
  onToggle: (id: string, checked: boolean) => void;
}) {
  const name = model.name ?? model.id;
  const accessibleName =
    name === model.id ? name : `${name}, model ${model.id}`;

  return (
    <label
      className={`model-choice${checked ? " chosen" : ""}${unavailable ? " is-unavailable" : ""}`}
    >
      <input
        type="checkbox"
        checked={checked}
        aria-label={`${accessibleName}, ${providerName(model.provider)}${unavailable ? ", unavailable" : ""}${again ? ", add again" : ""}`}
        onChange={(event) => onToggle(model.id, event.target.checked)}
      />
      <span className="model-choice-name">{name}</span>
      {name !== model.id && <span className="model-choice-id">{model.id}</span>}
      <span className="model-choice-provider">
        {providerName(model.provider)}
      </span>
      {unavailable && (
        <span className="model-unavailable-label">Unavailable</span>
      )}
      {again && <span className="model-choice-provider">Add again</span>}
    </label>
  );
}

export default function ModelPicker({
  catalog,
  selection,
  onChange,
  onSave,
  busy,
  open,
  onToggle,
  providers,
}: Props) {
  const [query, setQuery] = useState("");
  const listedModels = [
    ...(catalog?.providers.go ?? []),
    ...(catalog?.providers.openrouter ?? []),
  ];
  const models = [...listedModels];
  for (const id of [selection.writer, selection.strong, ...selection.weak]) {
    if (id && !models.some((model) => model.id === id)) {
      models.push({ id, provider: "configured" });
    }
  }
  const byId = new Map(models.map((model) => [model.id, model]));
  const chosenWeak = selection.weak.map(
    (id) => byId.get(id) ?? { id, provider: "configured" }
  );
  const needle = query.trim().toLowerCase();
  // Chosen models stay addable: each repeat adds samples for that model.
  const addable = models;
  const matchesQuery = (model: ModelInfo) =>
    !needle ||
    `${model.name ?? ""} ${model.id} ${providerName(model.provider)}`
      .toLowerCase()
      .includes(needle);
  const otherMatches = addable.filter(
    (model) => !isUnavailable(model, providers) && matchesQuery(model)
  );
  const unavailableMatches = addable.filter(
    (model) => isUnavailable(model, providers) && matchesQuery(model)
  );
  const shownOther = otherMatches.slice(0, MATCH_LIMIT);
  const shownUnavailable = unavailableMatches.slice(0, MATCH_LIMIT);

  const panelShort = selection.weak.length < WEAK_PANEL_SIZE;

  function addWeak(id: string) {
    onChange({ ...selection, weak: [...selection.weak, id] });
  }

  function removeWeakAt(index: number) {
    onChange({
      ...selection,
      weak: selection.weak.filter((_, position) => position !== index),
    });
  }

  return (
    <details
      className="model-picker"
      open={open}
      onToggle={(event) => onToggle(event.currentTarget.open)}
    >
      <summary>
        <span>Model choices</span>
        <span className="model-picker-optional">
          Optional · you can ignore this
        </span>
      </summary>
      <div className="model-picker-intro">
        <p>
          You don&apos;t need to choose models here to use the app. Open this
          section only if you want to customize it.
        </p>
        <button
          className="secondary"
          type="button"
          disabled={busy || panelShort}
          aria-describedby={panelShort ? "weak-panel-short" : undefined}
          onClick={onSave}
        >
          Save model defaults
        </button>
        {panelShort && (
          <p id="weak-panel-short" className="role-hint">
            Choose {WEAK_PANEL_SIZE} lower-cost test model entries to save; a
            model can be chosen more than once.
          </p>
        )}
      </div>
      <section className="model-role" aria-labelledby="judge-model-heading">
        <h3 id="judge-model-heading">Judge · fixed</h3>
        <p className="role-hint">
          Checks what your prompt needs and whether a rewrite keeps your
          meaning.
        </p>
        <p className="configured-model">
          Model: {catalog?.judge.id ?? "typesafe/jev-1.13"}
        </p>
      </section>
      <section className="model-role">
        <label id="writer-model-heading" htmlFor="writer-model">
          Writer
        </label>
        <select
          id="writer-model"
          value={selection.writer}
          aria-describedby="writer-hint"
          onChange={(event) =>
            onChange({ ...selection, writer: event.target.value })
          }
        >
          <ModelOptions models={models} providers={providers} />
        </select>
        <p id="writer-hint" className="role-hint">
          Writes clearer versions of your prompt.
        </p>
      </section>
      <section className="model-role">
        <label id="strong-model-heading" htmlFor="strong-model">
          Strong check
        </label>
        <select
          id="strong-model"
          value={selection.strong}
          aria-describedby="strong-hint"
          onChange={(event) =>
            onChange({ ...selection, strong: event.target.value })
          }
        >
          <ModelOptions models={models} providers={providers} />
        </select>
        <p id="strong-hint" className="role-hint">
          Checks that a rewrite still works at least as well as your original.
        </p>
      </section>
      <fieldset>
        <legend>
          Lower-cost test models ({selection.weak.length} selected)
        </legend>
        <p className="role-hint">
          Check whether a rewrite helps on less expensive models.
        </p>
        {chosenWeak.map((model, index) => (
          <ModelChoice
            key={`${model.id}-${index}`}
            model={model}
            checked
            unavailable={isUnavailable(model, providers)}
            onToggle={() => removeWeakAt(index)}
          />
        ))}
        <label htmlFor="weak-search" className="weak-search-label">
          Find models to add
        </label>
        <input
          id="weak-search"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by model name or provider"
        />
        {otherMatches.length > 0 ? (
          <section
            className="model-choice-group"
            aria-label="Other model choices"
          >
            <h4>Other model choices</h4>
            {shownOther.map((model) => (
              <ModelChoice
                key={model.id}
                model={model}
                checked={false}
                unavailable={false}
                again={selection.weak.includes(model.id)}
                onToggle={addWeak}
              />
            ))}
            {otherMatches.length > shownOther.length && (
              <p className="match-count">
                Showing {shownOther.length} of {otherMatches.length} other model
                choices. Search to narrow the list.
              </p>
            )}
          </section>
        ) : unavailableMatches.length === 0 ? (
          <p className="match-count">No model choices match.</p>
        ) : (
          <p className="match-count">No other model choices match.</p>
        )}
        {unavailableMatches.length > 0 && (
          <details className="unavailable-model-choices">
            <summary>
              Unavailable model choices ({unavailableMatches.length})
            </summary>
            <p className="role-hint">
              These models were reported unavailable by their provider. They
              remain listed here for manual selection.
            </p>
            {shownUnavailable.map((model) => (
              <ModelChoice
                key={model.id}
                model={model}
                checked={false}
                unavailable
                again={selection.weak.includes(model.id)}
                onToggle={addWeak}
              />
            ))}
            {unavailableMatches.length > shownUnavailable.length && (
              <p className="match-count">
                Showing {shownUnavailable.length} of {unavailableMatches.length}
                . Search above to narrow the list.
              </p>
            )}
          </details>
        )}
      </fieldset>
    </details>
  );
}
