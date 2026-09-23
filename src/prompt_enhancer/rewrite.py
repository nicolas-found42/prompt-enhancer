"""First-pass candidate generation, verification, and selection.

This module deliberately keeps the model seam small.  The engine owns the
workflow; this module accepts a gateway (or explicit callbacks in tests),
runs the original and one candidate through the same weak model, grades both
against the compiled tests, and only returns a rewrite after all fidelity
checks pass.  The data returned by :meth:`VerifiedRewrite.evaluate` is also
the report payload used by the HTTP and web surfaces.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import unified_diff
from typing import Any, Protocol


class ModelGateway(Protocol):
    """The subset of the model gateway used by this pass."""

    def chat(
        self, model: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> Any: ...

    def decide(self, payload: Mapping[str, Any], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Candidate:
    """A writer proposal and the exact edits it claims to make."""

    text: str
    strategy: str = "clarify"
    edits: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "strategy": self.strategy,
            "edits": list(self.edits),
        }


@dataclass(frozen=True)
class FidelityResult:
    """Outcome of the three non-negotiable fidelity checks."""

    meaning_preserved: bool
    no_invention: bool
    edits_confined: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.meaning_preserved and self.no_invention and self.edits_confined

    def to_dict(self) -> dict[str, Any]:
        return {
            "meaning_preserved": self.meaning_preserved,
            "no_invention": self.no_invention,
            "edits_confined": self.edits_confined,
            "meaning": self.meaning_preserved,
            "invention": self.no_invention,
            "confined": self.edits_confined,
            "passed": self.passed,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class GradedOutput:
    """A weak-model output and its test evidence."""

    output: str
    scores: tuple[float, ...]
    grades: tuple[Mapping[str, Any], ...]

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def pass_rate(self) -> float:
        return (
            sum(score >= 0.8 for score in self.scores) / len(self.scores)
            if self.scores
            else 0.0
        )

    @property
    def passed(self) -> bool:
        return bool(self.scores) and all(score >= 0.8 for score in self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "scores": list(self.scores),
            "grades": [dict(grade) for grade in self.grades],
            "mean_score": self.mean_score,
            "pass_rate": self.pass_rate,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RewriteResult:
    """Public result consumed by the engine, API, and report UI."""

    final_prompt: str
    original_kept: bool
    candidate: Candidate
    diagnosis: Sequence[Mapping[str, Any]]
    tests: Sequence[Mapping[str, Any]]
    per_model: Mapping[str, Any]
    diff: str
    selection_evidence: Mapping[str, Any]
    fidelity: FidelityResult

    def report(self) -> dict[str, Any]:
        """Return a JSON-friendly report with the stable bootstrap fields."""

        evidence = dict(self.selection_evidence)
        return {
            "status": "no_change" if self.original_kept else "improved",
            "diagnosis": [dict(item) for item in self.diagnosis],
            "tests": [dict(item) for item in self.tests],
            "candidates": [
                {
                    **self.candidate.to_dict(),
                    "fidelity": self.fidelity.to_dict(),
                    "selection": evidence,
                }
            ],
            "per_model": dict(self.per_model),
            "assumptions": list(evidence.get("assumptions", [])),
            "diff": self.diff,
            "selection_evidence": evidence,
            "fidelity": self.fidelity.to_dict(),
            "final_prompt": self.final_prompt,
            "original_kept": self.original_kept,
        }


class CandidateWriter:
    """Generate one candidate while keeping all user text in Jev ``state``.

    The writer receives the original prompt, diagnosis, and compiled tests as
    state data.  The instructions contain no user text, preventing a prompt
    from steering a decision request or leaking into the wrong field.
    """

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        writer_model: str = "deepseek-v4.1-flash",
        chat: Callable[..., Any] | None = None,
    ) -> None:
        self.gateway = gateway
        self.writer_model = writer_model
        self._chat = chat

    def _complete(
        self,
        *,
        model: str,
        prompt: str,
        state: Mapping[str, Any],
        temperature: float,
        seed: int,
    ) -> Any:
        if self._chat is not None:
            return _invoke(
                self._chat,
                model=model,
                prompt=prompt,
                state=state,
                temperature=temperature,
                seed=seed,
            )
        return _chat(
            self.gateway,
            model=model,
            prompt=prompt,
            state=state,
            temperature=temperature,
            seed=seed,
            role="writer",
        )

    def write(
        self,
        prompt: str,
        diagnosis: Sequence[Mapping[str, Any]],
        tests: Sequence[Mapping[str, Any]],
    ) -> Candidate:
        state = {
            "prompt": prompt,
            "diagnosis": [dict(item) for item in diagnosis],
            "tests": [dict(item) for item in tests],
        }
        response = self._complete(
            model=self.writer_model,
            prompt=(
                "Write exactly one improved prompt candidate. Preserve the input language and "
                "wording outside the diagnosed problem sentences. Return only the candidate text."
            ),
            state=state,
            temperature=0.2,
            seed=0,
        )
        text, strategy, edits = _parse_candidate(response)
        if not text:
            text = prompt
        return Candidate(text=text, strategy=strategy, edits=edits)


class VerifiedRewrite:
    """Verify one candidate and select it only when it improves the original.

    ``run_chat`` and ``grade_output``/``check_fidelity`` callbacks are useful
    for deterministic public-seam tests and for engines that already own a
    richer runner.  When omitted, the methods call the injected gateway.
    """

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        weak_model: str = "meta-llama/llama-3.1-8b-instruct",
        judge_model: str = "typesafe/jev-1.13",
        run_chat: Callable[..., Any] | None = None,
        grade_output: Callable[..., Any] | None = None,
        check_fidelity: Callable[..., Any] | None = None,
    ) -> None:
        self.gateway = gateway
        self.weak_model = weak_model
        self.judge_model = judge_model
        self._run_chat = run_chat
        self._grade_output = grade_output
        self._check_fidelity = check_fidelity

    def evaluate(
        self,
        prompt: str,
        candidate: Candidate | str,
        *,
        diagnosis: Sequence[Mapping[str, Any]] = (),
        tests: Sequence[Mapping[str, Any]] = (),
        assumptions: Sequence[Mapping[str, Any]] = (),
        sample_seed: int = 0,
    ) -> RewriteResult:
        proposal = (
            candidate if isinstance(candidate, Candidate) else Candidate(str(candidate))
        )
        if not tests:
            return self._retained(
                prompt,
                proposal,
                diagnosis,
                tests,
                assumptions,
                reason="no_compiled_tests",
            )

        original_output = self._run(prompt, sample_seed)
        candidate_output = self._run(proposal.text, sample_seed)
        original_grade = self._grade(prompt, original_output, tests)
        candidate_grade = self._grade(proposal.text, candidate_output, tests)
        fidelity = self._fidelity(prompt, proposal, diagnosis)

        candidate_passes = candidate_grade.passed
        beats_original = candidate_grade.mean_score > original_grade.mean_score
        accepted = fidelity.passed and candidate_passes and beats_original

        evidence: dict[str, Any] = {
            "accepted": accepted,
            "candidate": candidate_grade.to_dict(),
            "original": original_grade.to_dict(),
            "fidelity": fidelity.to_dict(),
            "weak_model": self.weak_model,
            "tests_compiled": len(tests),
            "assumptions": [dict(item) for item in assumptions],
            "rule": "candidate must pass fidelity, all tests, and beat the original mean score",
        }
        if not fidelity.passed:
            evidence["reason"] = "failed_fidelity"
        elif not beats_original:
            evidence["reason"] = "did_not_beat_original"
        elif not candidate_passes:
            evidence["reason"] = "failed_success_tests"
        else:
            evidence["reason"] = "verified_improvement"

        per_model = {
            self.weak_model: {
                "original": original_grade.to_dict(),
                "candidate": candidate_grade.to_dict(),
                "beats_original": beats_original,
            }
        }
        if accepted:
            return RewriteResult(
                final_prompt=proposal.text,
                original_kept=False,
                candidate=proposal,
                diagnosis=diagnosis,
                tests=tests,
                per_model=per_model,
                diff=_diff(prompt, proposal.text),
                selection_evidence=evidence,
                fidelity=fidelity,
            )
        return RewriteResult(
            final_prompt=prompt,
            original_kept=True,
            candidate=proposal,
            diagnosis=diagnosis,
            tests=tests,
            per_model=per_model,
            diff=_diff(prompt, prompt),
            selection_evidence=evidence,
            fidelity=fidelity,
        )

    def _retained(
        self,
        prompt: str,
        candidate: Candidate,
        diagnosis: Sequence[Mapping[str, Any]],
        tests: Sequence[Mapping[str, Any]],
        assumptions: Sequence[Mapping[str, Any]],
        *,
        reason: str,
    ) -> RewriteResult:
        fidelity = FidelityResult(True, True, True, {"skipped": True})
        evidence = {
            "accepted": False,
            "reason": reason,
            "candidate": {"text": candidate.text},
            "original": {},
            "fidelity": fidelity.to_dict(),
            "weak_model": self.weak_model,
            "assumptions": [dict(item) for item in assumptions],
        }
        return RewriteResult(
            final_prompt=prompt,
            original_kept=True,
            candidate=candidate,
            diagnosis=diagnosis,
            tests=tests,
            per_model={},
            diff=_diff(prompt, prompt),
            selection_evidence=evidence,
            fidelity=fidelity,
        )

    def _run(self, prompt: str, seed: int) -> str:
        if self._run_chat is not None:
            value = _invoke(
                self._run_chat,
                model=self.weak_model,
                prompt=prompt,
                temperature=0.7,
                seed=seed,
            )
        else:
            value = _chat(
                self.gateway,
                model=self.weak_model,
                prompt=prompt,
                temperature=0.7,
                seed=seed,
                role="weak",
            )
        return _text(value)

    def _grade(
        self,
        prompt: str,
        output: str,
        tests: Sequence[Mapping[str, Any]],
    ) -> GradedOutput:
        if self._grade_output is not None:
            value = _invoke(
                self._grade_output,
                prompt=prompt,
                output=output,
                tests=tests,
                judge_model=self.judge_model,
            )
            return _coerce_grade(value, output)
        scores: list[float] = []
        grades: list[Mapping[str, Any]] = []
        for test in tests:
            question = _test_question(test)
            state = {"prompt": prompt, "output": output, "test": dict(test)}
            response = _judge(
                self.gateway,
                model=self.judge_model,
                state=state,
                question=question,
            )
            score = _score(response)
            scores.append(score)
            grades.append(
                {"test_id": test.get("id"), "question": question, "response": response}
            )
        return GradedOutput(output=output, scores=tuple(scores), grades=tuple(grades))

    def _fidelity(
        self,
        prompt: str,
        candidate: Candidate,
        diagnosis: Sequence[Mapping[str, Any]],
    ) -> FidelityResult:
        if self._check_fidelity is not None:
            value = _invoke(
                self._check_fidelity,
                prompt=prompt,
                candidate=candidate,
                diagnosis=diagnosis,
            )
            return _coerce_fidelity(value)
        state = {
            "original_prompt": prompt,
            "candidate_prompt": candidate.text,
            "diagnosis": [dict(item) for item in diagnosis],
            "flagged_sentences": _flagged_sentences(diagnosis),
            "claimed_edits": list(candidate.edits),
        }
        checks = (
            (
                "meaning_preserved",
                "Does the candidate preserve the original request and all stated constraints?",
            ),
            (
                "no_invention",
                "Does the candidate avoid adding facts, requirements, or promises not present in the original?",
            ),
            (
                "edits_confined",
                "Are all candidate edits confined to the sentences flagged by diagnosis, unless explicitly required?",
            ),
        )
        evidence: dict[str, Any] = {}
        outcomes: dict[str, bool] = {}
        for name, question in checks:
            response = _judge(
                self.gateway, model=self.judge_model, state=state, question=question
            )
            outcomes[name] = _yes(response)
            evidence[name] = response
        return FidelityResult(
            meaning_preserved=outcomes["meaning_preserved"],
            no_invention=outcomes["no_invention"],
            edits_confined=outcomes["edits_confined"],
            evidence=evidence,
        )


def _diff(original: str, candidate: str) -> str:
    if original == candidate:
        return ""
    return "".join(
        unified_diff(
            original.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile="original",
            tofile="candidate",
        )
    )


def _flagged_sentences(diagnosis: Iterable[Mapping[str, Any]]) -> list[Any]:
    values: list[Any] = []
    for item in diagnosis:
        for key in ("flagged_sentences", "problem_sentences", "sentences"):
            value = item.get(key)
            if isinstance(value, list):
                values.extend(value)
    return values


def _test_question(test: Mapping[str, Any]) -> str:
    for key in ("question", "prompt", "text", "statement"):
        value = test.get(key)
        if isinstance(value, str) and value:
            return value
    return str(test)


def _invoke(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Invoke an injected seam, tolerating simple positional fakes."""

    try:
        return fn(**kwargs)
    except TypeError as first_error:
        try:
            return fn(
                kwargs.get("model"),
                kwargs.get("prompt"),
                kwargs.get("temperature", 0.0),
                kwargs.get("seed", 0),
            )
        except TypeError:
            raise first_error


def _chat(gateway: Any, **kwargs: Any) -> Any:
    method = getattr(gateway, "chat", None) or gateway.complete
    model = kwargs["model"]
    prompt = kwargs["prompt"]
    state = kwargs.get("state")
    if state is not None:
        content = json.dumps(dict(state), ensure_ascii=False, sort_keys=True)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    if method.__name__ == "chat" or "messages" in getattr(
        method, "__annotations__", {}
    ):
        try:
            return method(
                model,
                messages,
                role=kwargs.get("role", "chat"),
                temperature=kwargs.get("temperature", 0.0),
                seed=kwargs.get("seed", 0),
            )
        except TypeError:
            return method(model, messages)
    return _invoke(
        method,
        model=model,
        prompt=prompt,
        temperature=kwargs.get("temperature", 0.0),
        seed=kwargs.get("seed", 0),
    )


def _judge(gateway: Any, **kwargs: Any) -> Any:
    method = getattr(gateway, "judge", None) or getattr(gateway, "jev", None)
    if method is not None:
        return _invoke(method, **kwargs)
    method = gateway.decide
    payload = {
        "model": kwargs["model"],
        "state": dict(kwargs["state"]),
        "question": kwargs["question"],
    }
    try:
        return method(payload, role="judge")
    except TypeError:
        return method(payload)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("content", "text", "output", "completion", "answer"):
            if isinstance(value.get(key), str):
                return value[key]
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            return _text(choices[0])
    for key in ("content", "text", "output", "completion", "answer"):
        found = getattr(value, key, None)
        if isinstance(found, str):
            return found
    raise ValueError("model gateway returned no text completion")


def _parse_candidate(value: Any) -> tuple[str, str, tuple[str, ...]]:
    if isinstance(value, Mapping):
        text = value.get("text") or value.get("prompt") or value.get("candidate")
        if not isinstance(text, str):
            text = _text(value)
        strategy = value.get("strategy", "clarify")
        edits = value.get("edits", ())
    else:
        text = _text(value)
        strategy = "clarify"
        edits = ()
    if isinstance(strategy, Mapping):
        strategy = strategy.get("name", "clarify")
    if not isinstance(edits, (list, tuple)):
        edits = ()
    return text.strip(), str(strategy), tuple(str(edit) for edit in edits)


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, Mapping):
        for key in ("score", "probability", "pass_probability", "yes_probability"):
            number = value.get(key)
            if isinstance(number, (int, float)):
                return max(0.0, min(1.0, float(number)))
        for key in ("passed", "pass", "correct", "yes"):
            if isinstance(value.get(key), bool):
                return 1.0 if value[key] else 0.0
        choice = value.get("choice")
        if isinstance(choice, str) and choice.lower() in {
            "yes",
            "pass",
            "passes",
            "true",
        }:
            return 1.0
        probabilities = value.get("probabilities")
        if isinstance(probabilities, Mapping):
            return _score(probabilities.get("yes", probabilities.get("pass", 0.0)))
    if isinstance(value, str):
        if value.lower() in {"yes", "true", "pass", "passed", "correct"}:
            return 1.0
        if value.lower() in {"no", "false", "fail", "failed", "incorrect"}:
            return 0.0
    return 0.0


def _yes(value: Any) -> bool:
    return _score(value) >= 0.8


def _coerce_grade(value: Any, output: str) -> GradedOutput:
    if isinstance(value, GradedOutput):
        return value
    if isinstance(value, Mapping):
        if "scores" in value:
            scores = value.get("scores", ())
            if not isinstance(scores, (list, tuple)):
                scores = ()
            grades = value.get("grades", ())
            if not isinstance(grades, (list, tuple)):
                grades = ()
            return GradedOutput(
                str(value.get("output", output)),
                tuple(_score(item) for item in scores),
                tuple(grades),
            )
        value = value.get("score", value.get("passed", value))
    if isinstance(value, (list, tuple)):
        scores = tuple(_score(item) for item in value)
    else:
        scores = (_score(value),)
    return GradedOutput(output=output, scores=scores, grades=({"score": scores[0]},))


def _coerce_fidelity(value: Any) -> FidelityResult:
    if isinstance(value, FidelityResult):
        return value
    if isinstance(value, Mapping):
        def first(*names: str, default: bool = False) -> bool:
            for name in names:
                if name in value:
                    return bool(value[name])
            return default

        meaning = first("meaning", "meaning_preserved", "passed")
        no_invention = first("invention", "invention_free", "no_invention", "passed")
        confined = first("confined", "confined_edits", "edits_confined", "passed")
        evidence = value.get("evidence", {})
        return FidelityResult(
            meaning,
            no_invention,
            confined,
            evidence if isinstance(evidence, Mapping) else {},
        )
    raise ValueError("fidelity checker must return FidelityResult or a mapping")


__all__ = [
    "Candidate",
    "CandidateWriter",
    "FidelityResult",
    "GradedOutput",
    "ModelGateway",
    "RewriteResult",
    "VerifiedRewrite",
]
