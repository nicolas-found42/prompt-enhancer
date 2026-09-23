import { FormEvent, useMemo, useState } from "react";

export type ClarificationOption = {
  value: string;
  label: string;
  preselected?: boolean;
  other?: boolean;
};

export type ClarificationQuestion = {
  id: string;
  prompt: string;
  options: ClarificationOption[];
  default?: string;
  default_answer?: string;
  allow_other?: boolean;
  other_value?: string;
};

export type ClarificationAnswers = Record<string, string | { value: string; text?: string }>;

type Props = {
  questions: ClarificationQuestion[];
  onSubmit: (answers: ClarificationAnswers) => void | Promise<void>;
  onSkip: () => void | Promise<void>;
  busy?: boolean;
  error?: string | null;
};

export function ClarificationPanel({ questions, onSubmit, onSkip, busy = false, error }: Props) {
  const defaults = useMemo(
    () =>
      Object.fromEntries(
        questions.map((question) => [
          question.id,
          question.default ?? question.default_answer ?? question.options[0]?.value ?? "",
        ]),
      ),
    [questions],
  );
  const [answers, setAnswers] = useState<ClarificationAnswers>(defaults);
  const [otherText, setOtherText] = useState<Record<string, string>>({});

  const setAnswer = (id: string, value: string) => {
    setAnswers((current) => ({ ...current, [id]: value }));
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const payload: ClarificationAnswers = { ...answers };
    for (const question of questions) {
      if (payload[question.id] === (question.other_value ?? "other")) {
        payload[question.id] = {
          value: question.other_value ?? "other",
          text: otherText[question.id] ?? "",
        };
      }
    }
    await onSubmit(payload);
  };

  const skip = async () => {
    await onSkip();
  };

  if (questions.length === 0) return null;

  return (
    <section className="clarification-panel" aria-labelledby="clarification-heading">
      <h2 id="clarification-heading">A few details will improve the result</h2>
      <p>Choose the closest answer. You can skip these and continue with assumptions.</p>
      <form onSubmit={submit}>
        {questions.map((question) => {
          const otherValue = question.other_value ?? "other";
          const selected = answers[question.id];
          const selectedValue = typeof selected === "string" ? selected : selected?.value;
          return (
            <fieldset key={question.id}>
              <legend>{question.prompt}</legend>
              {question.options.filter((option) => !option.other).map((option) => (
                <label key={option.value}>
                  <input
                    type="radio"
                    name={question.id}
                    value={option.value}
                    checked={selectedValue === option.value}
                    onChange={() => setAnswer(question.id, option.value)}
                  />
                  {option.label}
                  {option.preselected && <small> (recommended)</small>}
                </label>
              ))}
              {(question.allow_other ?? true) && (
                <div>
                  <label>
                    <input
                      type="radio"
                      name={question.id}
                      value={otherValue}
                      checked={selectedValue === otherValue}
                      onChange={() => setAnswer(question.id, otherValue)}
                    />
                    Other
                  </label>
                  {selectedValue === otherValue && (
                    <input
                      aria-label={`Other answer for ${question.prompt}`}
                      value={otherText[question.id] ?? ""}
                      onChange={(event) =>
                        setOtherText((current) => ({ ...current, [question.id]: event.target.value }))
                      }
                      placeholder="Enter your own answer"
                    />
                  )}
                </div>
              )}
            </fieldset>
          );
        })}
        {error && <p role="alert">{error}</p>}
        <button type="submit" disabled={busy}>
          {busy ? "Continuing…" : "Continue"}
        </button>
        <button type="button" onClick={skip} disabled={busy}>
          Skip and continue
        </button>
      </form>
    </section>
  );
}

export default ClarificationPanel;
