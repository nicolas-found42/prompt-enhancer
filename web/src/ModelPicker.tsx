import type { ModelCatalog, ModelSelection } from "./api";

type Props = {
  catalog: ModelCatalog | null;
  selection: ModelSelection;
  onChange: (selection: ModelSelection) => void;
  onSave: () => void;
  busy: boolean;
};

export default function ModelPicker({ catalog, selection, onChange, onSave, busy }: Props) {
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

  return (
    <details className="model-picker">
      <summary>Model choices</summary>
      <p>Change models for this run, or save these choices as your defaults. Jev remains the judge.</p>
      <p>Judge: {catalog?.judge.id ?? "typesafe/jev-1.13"} (fixed)</p>
      <label htmlFor="writer-model">Writer</label>
      <select id="writer-model" value={selection.writer} onChange={(event) => onChange({ ...selection, writer: event.target.value })}>
        {models.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id} ({model.provider})</option>)}
      </select>
      <label htmlFor="strong-model">Strong check</label>
      <select id="strong-model" value={selection.strong} onChange={(event) => onChange({ ...selection, strong: event.target.value })}>
        {models.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id} ({model.provider})</option>)}
      </select>
      <fieldset>
        <legend>Weak panel</legend>
        {models.map((model) => (
          <label key={model.id}>
            <input
              type="checkbox"
              checked={selection.weak.includes(model.id)}
              onChange={(event) => onChange({
                ...selection,
                weak: event.target.checked
                  ? [...selection.weak, model.id]
                  : selection.weak.filter((id) => id !== model.id),
              })}
            />
            {model.name ?? model.id} ({model.provider})
          </label>
        ))}
      </fieldset>
      <button className="secondary" type="button" disabled={busy || selection.weak.length === 0} onClick={onSave}>
        Save model defaults
      </button>
    </details>
  );
}
