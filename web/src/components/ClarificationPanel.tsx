import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

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

export type ClarificationAnswers = Record<
  string,
  string | { value: string; text?: string }
>;

type Props = {
  questions: ClarificationQuestion[];
  onSubmit: (answers: ClarificationAnswers) => void | Promise<void>;
  onSkip: () => void | Promise<void>;
  busy?: boolean;
  error?: string | null;
  validationError?: { questionId: string; message: string } | null;
  onValidationErrorDismiss?: () => void;
};

export function ClarificationPanel({
  questions,
  onSubmit,
  onSkip,
  busy = false,
  error,
  validationError = null,
  onValidationErrorDismiss,
}: Props) {
  const defaults = useMemo(
    () =>
      Object.fromEntries(
        questions.map((question) => [
          question.id,
          question.default ??
            question.default_answer ??
            question.options[0]?.value ??
            "",
        ])
      ),
    [questions]
  );
  const [answers, setAnswers] = useState<ClarificationAnswers>(defaults);
  const [otherText, setOtherText] = useState<Record<string, string>>({});
  const [localErrors, setLocalErrors] = useState<Record<string, string>>({});
  const otherInputs = useRef<Record<string, HTMLInputElement | null>>({});

  useEffect(() => {
    if (validationError) {
      otherInputs.current[validationError.questionId]?.focus();
    }
  }, [validationError]);

  const setAnswer = (id: string, value: string) => {
    setAnswers((current) => ({ ...current, [id]: value }));
    setLocalErrors((current) => {
      if (!(id in current)) return current;
      const next = { ...current };
      delete next[id];
      return next;
    });
    onValidationErrorDismiss?.();
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const invalidOtherQuestions = questions.filter((question) => {
      const answer = answers[question.id];
      const value = typeof answer === "string" ? answer : answer?.value;
      return (
        value === (question.other_value ?? "other") &&
        !(otherText[question.id] ?? "").trim()
      );
    });
    if (invalidOtherQuestions.length > 0) {
      setLocalErrors(
        Object.fromEntries(
          invalidOtherQuestions.map((question) => [
            question.id,
            "Enter an answer for this question.",
          ])
        )
      );
      const firstInvalidQuestion = invalidOtherQuestions[0];
      if (firstInvalidQuestion) {
        otherInputs.current[firstInvalidQuestion.id]?.focus();
      }
      return;
    }
    setLocalErrors({});
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
    <section
      className="result clarification-panel"
      aria-labelledby="clarification-heading"
    >
      <h2 id="clarification-heading">A few details will improve the result</h2>
      <p>
        Choose the closest answer. You can skip these and continue with
        assumptions.
      </p>
      <form onSubmit={submit}>
        {questions.map((question) => {
          const otherValue = question.other_value ?? "other";
          const selected = answers[question.id];
          const selectedValue =
            typeof selected === "string" ? selected : selected?.value;
          return (
            <fieldset key={question.id}>
              <legend>{question.prompt}</legend>
              {question.options
                .filter((option) => !option.other)
                .map((option) => (
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
                      aria-invalid={Boolean(
                        localErrors[question.id] ||
                        validationError?.questionId === question.id
                      )}
                      aria-describedby={
                        localErrors[question.id] ||
                        validationError?.questionId === question.id
                          ? `clarification-error-${question.id}`
                          : undefined
                      }
                      ref={(input) => {
                        otherInputs.current[question.id] = input;
                      }}
                      onChange={(event) => {
                        setOtherText((current) => ({
                          ...current,
                          [question.id]: event.target.value,
                        }));
                        setLocalErrors((current) => {
                          if (!(question.id in current)) return current;
                          const next = { ...current };
                          delete next[question.id];
                          return next;
                        });
                        onValidationErrorDismiss?.();
                      }}
                      placeholder="Enter your own answer"
                    />
                  )}
                  {(localErrors[question.id] ||
                    (validationError?.questionId === question.id &&
                      validationError.message)) && (
                    <p
                      className="field-error"
                      id={`clarification-error-${question.id}`}
                      role="alert"
                    >
                      {localErrors[question.id] ?? validationError?.message}
                    </p>
                  )}
                </div>
              )}
            </fieldset>
          );
        })}
        {error && <p role="alert">{error}</p>}
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "Continuing…" : "Continue"}
        </button>
        <button
          className="secondary"
          type="button"
          onClick={skip}
          disabled={busy}
        >
          Skip and continue
        </button>
      </form>
    </section>
  );
}

export default ClarificationPanel;
