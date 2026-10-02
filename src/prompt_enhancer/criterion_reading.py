"""Read success criteria with a batched Jev request instead of regexes.

The model selects the constraint a criterion sets (which count, which
comparison, which number); code owns everything else. Candidate numbers are
found by regex, the selected span becomes an integer here, counting words and
parsing JSON happen here, and any yes/no judgment at or above its audited
cutoff leaves the criterion unresolved.

``scripts/measure_criterion_reading.py`` imports this module, so the question
set, the candidate-number step and the resolution policy cannot drift between
measurement and production; ``questions_digest()`` stays comparable.

Production reads one criterion through :class:`CriterionReader` (one batched
request per distinct criterion, cached per run) and gets back a
:class:`CriterionCheck`, so the grading cascade's ``exact``/``unsupported``
contract is unchanged. The requests are gated behind writer instruction
version :data:`CRITERION_READING_MIN_VERSION`, so recordings and replays made
before it reproduce exactly.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

from .criterion_checks import CriterionCheck, _word_check, count_words, parses_as_json
from .gateway import ProviderError
from .jev import ChoiceDecision, NoulDecision, parse_decision

# Version 12 of the writer instructions is the first that reads criteria with
# Jev. The gate lives here so production and the cascade agree on the number.
CRITERION_READING_MIN_VERSION = 12

NONE = "none"
NOUL_KEYS = ("partial", "conditional", "negated", "approximate")
OPERATORS = ("at most", "under", "at least", "more than", "exactly", "between")
JUDGMENTS = ("partial", "conditional", "approximate")
CHOICE_KEYS = ("kind", "op", "bound", "low", "high")
READABLE_KINDS = ("word_count", "sentence_count", "valid_json")

_READING_ROLE = "judge_criterion_reading"

_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_WORD_VALUES: dict[str, int] = {
    **{word: value for value, word in enumerate(_UNITS)},
    **{word: 20 + 10 * value for value, word in enumerate(_TENS)},
}
_SMALL = rf"(?:(?:{'|'.join(_TENS)})(?:[- ](?:{'|'.join(_UNITS[1:10])}))?|{'|'.join(_UNITS)})"
_NUMBER_CANDIDATE = re.compile(
    r"\d{1,3}(?:,\d{3})+|\d+"
    rf"|\b{_SMALL}(?:\s+(?:hundred|thousand)"
    rf"(?:\s+(?:and\s+)?{_SMALL}(?!\s+(?:hundred|thousand)))?)?\b"
)


def number_candidates(criterion: str) -> dict[str, int]:
    """Numbers in the criterion, by the span as written, valued in code."""
    found: dict[str, int] = {}
    for match in _NUMBER_CANDIDATE.finditer(criterion.casefold()):
        span = match.group()
        found.setdefault(span, _value(span))
    return found


def _value(span: str) -> int:
    if span[0].isdigit():
        return int(span.replace(",", ""))
    total = current = 0
    for word in re.split(r"[-\s]+", span):
        if word == "hundred":
            current = max(current, 1) * 100
        elif word == "thousand":
            total, current = total + max(current, 1) * 1000, 0
        elif word != "and":
            current += _WORD_VALUES[word]
    return total + current


def reading_questions(candidates: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The questions asked about one criterion, keyed by what code reads back."""
    numbers: dict[str, Any] = {span: None for span in candidates} | {
        NONE: "The criterion gives no such number."
    }
    return {
        "kind": {
            "type": "choice",
            "instructions": (
                "This is a success criterion that a piece of writing will be "
                "graded against. Which countable property of the whole output "
                "does it set a limit or target on?"
            ),
            "criteria": {
                "word_count": "A limit or target on the number of words, such as "
                "'under 100 words', '50 words or fewer', 'at least 500 words', "
                "'between 100 and 150 words'.",
                "sentence_count": "A limit or target on the number of sentences, "
                "such as 'at most 3 sentences' or 'exactly two sentences'.",
                "valid_json": "The output has to be JSON that parses, such as "
                "'valid JSON' or 'parseable as JSON'.",
                "other": "Anything else: tone, content, format, paragraphs, "
                "bullets, characters, sources, punctuation, a field inside JSON, "
                "or no countable property.",
            },
        },
        "op": {
            "type": "choice",
            "instructions": (
                "If the criterion sets a limit or target on a count of words or "
                "sentences, which comparison must the count satisfy? Read "
                "negations into the answer: 'not under 100' means at least 100 "
                "and 'never more than 60' means at most 60."
            ),
            "criteria": {
                "at most": "No more than the number, e.g. 'at most 50', '50 or "
                "fewer', 'no longer than 50', 'within 50', 'a 50-word limit'.",
                "under": "Strictly fewer than the number, e.g. 'under 50', "
                "'fewer than 50', 'shorter than 50'.",
                "at least": "No fewer than the number, e.g. 'at least 50', "
                "'50 or more', 'a minimum of 50', or a plus sign after the "
                "number as in '50+'.",
                "more than": "Strictly more than the number, e.g. 'over 50', "
                "'longer than 50', 'more than 50'.",
                "exactly": "Equal to the number.",
                "between": "Within a range of two numbers, e.g. 'between 3 and "
                "5', '100 to 150', '100-150'.",
                NONE: "No comparison on a count is set.",
            },
        },
        "bound": {
            "type": "choice",
            "instructions": (
                "Which of these numbers is the limit or target that the count of "
                "words or sentences is compared against? Ignore numbers that "
                "count something else, such as paragraphs, questions or ages."
            ),
            "criteria": numbers,
        },
        "low": {
            "type": "choice",
            "instructions": (
                "If the criterion gives a range for the count of words or "
                "sentences, such as 'between 3 and 5' or '100 to 150', which "
                "number is the lower end of the range? Otherwise choose none."
            ),
            "criteria": numbers,
        },
        "high": {
            "type": "choice",
            "instructions": (
                "If the criterion gives a range for the count of words or "
                "sentences, such as 'between 3 and 5' or '100 to 150', which "
                "number is the upper end of the range? Otherwise choose none."
            ),
            "criteria": numbers,
        },
        "partial": {
            "type": "noul",
            "instructions": (
                "Is the requirement in the criterion about only part of the "
                "output, such as a title, an intro, each section, each bullet, "
                "every sentence, or the JSON in a reply that also contains other "
                "text, rather than about the output as a whole?"
            ),
        },
        "conditional": {
            "type": "noul",
            "instructions": (
                "Does the criterion apply only under a condition, or offer an "
                "alternative that would also satisfy it, such as 'unless', 'if', "
                "'when' or 'or a table if longer'?"
            ),
        },
        "negated": {
            "type": "noul",
            "instructions": (
                "Does the criterion require the property it names to be absent "
                "or to fail, such as 'is not valid JSON' or 'must not be JSON'?"
            ),
        },
        "approximate": {
            "type": "noul",
            "instructions": (
                "Does the criterion itself say its target is only approximate, "
                "with a word such as 'roughly', 'about', 'around', "
                "'approximately' or 'give or take'?"
            ),
        },
    }


@lru_cache(maxsize=1)
def questions_digest() -> str:
    blob = json.dumps(reading_questions(["100"]), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Reading:
    """The model's answers about one criterion, before any policy is applied."""

    kind: str
    op: str
    bound: str
    low: str
    high: str
    noul: Mapping[str, float]


def read_jev(answers: Mapping[str, Any]) -> Reading:
    choices: dict[str, str] = {}
    for key in CHOICE_KEYS:
        decision = parse_decision(answers[key])
        if not isinstance(decision, ChoiceDecision):
            raise TypeError(f"expected choice decision for {key}")
        choices[key] = decision.selected
    nouls: dict[str, float] = {}
    for key in NOUL_KEYS:
        decision = parse_decision(answers[key])
        if not isinstance(decision, NoulDecision):
            raise TypeError(f"expected Noul decision for {key}")
        nouls[key] = decision.probability
    return Reading(noul=nouls, **choices)


@dataclass(frozen=True, slots=True)
class ReadingPolicy:
    """The audited cutoffs, named so the run report can pin them."""

    version: str = "issue-116-v1"
    partial: float = 0.40
    conditional: float = 0.50
    approximate: float = 0.50
    json_negation_low: float = 0.35
    json_negation_high: float = 0.65

    def cutoffs(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in JUDGMENTS}

    def band(self) -> tuple[float, float]:
        return (self.json_negation_low, self.json_negation_high)


CURRENT_READING_POLICY = ReadingPolicy()


def policy_from_metadata(metadata: Any) -> ReadingPolicy:
    """Rebuild a recorded :class:`ReadingPolicy`, falling back to the default.

    A replay reads the policy a recording was made with so its cutoffs decide
    the same way; anything malformed falls back to ``CURRENT_READING_POLICY``
    rather than letting a hand-edited bundle change a grade.
    """
    if not isinstance(metadata, Mapping):
        return CURRENT_READING_POLICY
    cutoffs = metadata.get("cutoffs")
    band = metadata.get("json_negation_band")
    if not isinstance(cutoffs, Mapping) or not isinstance(band, (list, tuple)):
        return CURRENT_READING_POLICY
    if len(band) != 2:
        return CURRENT_READING_POLICY
    values = dict(CURRENT_READING_POLICY.cutoffs())
    for name in JUDGMENTS:
        value = cutoffs.get(name, values[name])
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return CURRENT_READING_POLICY
        if not 0 <= float(value) <= 1:
            return CURRENT_READING_POLICY
        values[name] = float(value)
    low, high = band
    if (
        not isinstance(low, (int, float))
        or isinstance(low, bool)
        or not isinstance(high, (int, float))
        or isinstance(high, bool)
        or not 0 <= low <= high <= 1
    ):
        return CURRENT_READING_POLICY
    version = metadata.get("policy_version")
    return ReadingPolicy(
        version=version if isinstance(version, str) and version else "issue-116-v1",
        partial=values["partial"],
        conditional=values["conditional"],
        approximate=values["approximate"],
        json_negation_low=float(low),
        json_negation_high=float(high),
    )


def policy_for_gateway(gateway: Any) -> ReadingPolicy:
    """The policy a gateway was built with, or the current default."""
    return policy_from_metadata(getattr(gateway, "criterion_reading", None))


def resolve(
    reading: Reading,
    candidates: Mapping[str, int],
    band: tuple[float, float],
    *,
    cutoffs: Mapping[str, float] | None = None,
) -> dict[str, Any] | None:
    """The check a reading supports, or None when the criterion stays unresolved.

    A yes/no judgment at or above its cutoff that the criterion is partial,
    conditional or approximate leaves it unresolved. Without per-judgment
    cutoffs, all three use the band's lower edge. For JSON, a negation at or
    above ``high`` becomes a negated check and one between the band edges is
    uncertain.
    """
    low, high = band
    thresholds = {
        key: cutoffs.get(key, low) if cutoffs is not None else low for key in JUDGMENTS
    }
    if reading.kind not in READABLE_KINDS:
        return None
    if any(reading.noul[key] >= thresholds[key] for key in JUDGMENTS):
        return None
    if reading.kind == "valid_json":
        negation = reading.noul["negated"]
        if low < negation < high:
            return None
        return {"kind": "valid_json", "negated": negation >= high}
    if reading.op == "between":
        ends = {
            candidates[end] for end in (reading.low, reading.high) if end in candidates
        }
        if len(ends) != 2:
            return None
        bound, upper = sorted(ends)
        return {
            "kind": reading.kind,
            "operator": "between",
            "bound": bound,
            "upper_bound": upper,
        }
    if reading.op not in OPERATORS or reading.bound not in candidates:
        return None
    return {
        "kind": reading.kind,
        "operator": reading.op,
        "bound": candidates[reading.bound],
        "upper_bound": None,
    }


def resolve_with_policy(
    reading: Reading,
    candidates: Mapping[str, int],
    policy: ReadingPolicy = CURRENT_READING_POLICY,
) -> dict[str, Any] | None:
    """``resolve`` with the named production policy instead of a raw band."""
    return resolve(reading, candidates, policy.band(), cutoffs=policy.cutoffs())


def recording_metadata(
    policy: ReadingPolicy = CURRENT_READING_POLICY,
) -> dict[str, Any]:
    """The reading policy a version-12 recording carries, for exact replay."""
    return {
        "min_writer_instruction_version": CRITERION_READING_MIN_VERSION,
        "policy_version": policy.version,
        "cutoffs": policy.cutoffs(),
        "json_negation_band": list(policy.band()),
        "questions_digest": questions_digest(),
    }


@dataclass(frozen=True, slots=True)
class CriterionReading:
    """One criterion's reading, resolved bound and counted check."""

    criterion: str
    candidates: Mapping[str, int]
    reading: Reading | None
    resolved: Mapping[str, Any] | None
    exact: dict[str, Any] | None
    unsupported: tuple[str, ...]
    reason: str
    policy_version: str
    questions_digest: str
    answered_by: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    judgment_probabilities: Mapping[str, float] = field(default_factory=dict)
    selected_probabilities: Mapping[str, float] = field(default_factory=dict)
    request_count: int = 0

    @property
    def resolved_reading(self) -> bool:
        return self.resolved is not None

    def as_check(self, *, regex: CriterionCheck | None = None) -> CriterionCheck:
        """The ``exact``/``unsupported`` contract the grading cascade consumes.

        Without ``regex`` the reading alone decides. With it, the reading's
        check replaces the regex's ``exact`` when it resolved one, and its
        abstention vetoes a regex check; the regex's other ``unsupported``
        names (a negated or part-scoped bound, a unit code cannot count) are
        kept, so #111's handling is unchanged. The conflict rule itself lives
        in the cascade: it compares a resolved check to Jev's judgment.
        """
        if regex is None:
            return CriterionCheck(exact=self.exact, unsupported=self.unsupported)
        if self.exact is not None:
            return CriterionCheck(exact=self.exact, unsupported=regex.unsupported)
        if self.reading is None:
            # Reading unavailable or unusable: keep the pre-version-12 answer,
            # and the reason on this CriterionReading says what happened.
            return regex
        names = [
            *regex.unsupported,
            *(
                ()
                if self.reason == "criterion_reading_sentence_count_unsupported"
                else (self.reason,)
            ),
        ]
        return CriterionCheck(exact=None, unsupported=tuple(dict.fromkeys(names)))

    def to_evidence(self) -> dict[str, Any]:
        """Per-criterion evidence for the run report: text, reading, probabilities."""
        reading = self.reading
        return {
            "criterion": self.criterion,
            "policy_version": self.policy_version,
            "questions_digest": self.questions_digest,
            "candidates": dict(self.candidates),
            "reading": (
                {
                    "kind": reading.kind,
                    "op": reading.op,
                    "bound": reading.bound,
                    "low": reading.low,
                    "high": reading.high,
                }
                if reading is not None
                else None
            ),
            "judgment_probabilities": dict(self.judgment_probabilities),
            "selected_probabilities": dict(self.selected_probabilities),
            "resolved": dict(self.resolved) if self.resolved is not None else None,
            "check": self.exact,
            "reason": self.reason,
            "answered_by": self.answered_by,
            "usage": dict(self.usage),
            "request_count": self.request_count,
        }


class CriterionReader:
    """Read criteria with one batched Jev request each, cached per criterion."""

    def __init__(
        self,
        gateway: Any,
        *,
        model: str | None = None,
        policy: ReadingPolicy = CURRENT_READING_POLICY,
        role: str = _READING_ROLE,
        run_id: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.model = model or gateway.jev_model
        self.policy = policy
        self.role = role
        self.run_id = run_id
        self.request_count = 0
        self._cache: dict[str, CriterionReading] = {}

    @property
    def read_cache(self) -> Mapping[str, CriterionReading]:
        """The per-criterion base readings, read once per distinct criterion."""
        return self._cache

    def check(
        self, criterion: str, output: str, *, run_id: str | None = None
    ) -> CriterionReading:
        """The counted check for one criterion against one output."""
        base = self._cache.get(criterion)
        if base is None:
            base = self._read(
                criterion, run_id=run_id if run_id is not None else self.run_id
            )
            self._cache[criterion] = base
        return _counted(base, output)

    def _read(self, criterion: str, *, run_id: str | None) -> CriterionReading:
        candidates = number_candidates(criterion)
        questions = reading_questions(list(candidates))
        requests = [
            _request(self.model, key, question, criterion)
            for key, question in questions.items()
        ]
        log_start = len(self.gateway.decision_log)
        try:
            answers = self.gateway.decide_batch(requests, role=self.role, run_id=run_id)
        except ProviderError as exc:
            return self._unread(
                criterion,
                candidates,
                f"criterion_reading_provider_{exc.kind}",
                request_count=0,
            )
        self.request_count += 1
        entry = (
            self.gateway.decision_log[log_start]
            if len(self.gateway.decision_log) > log_start
            else {}
        )
        answered_by = entry.get("answered_by")
        usage = entry.get("usage")
        if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
            return self._unread(
                criterion, candidates, "criterion_reading_incomplete", request_count=1
            )
        if len(answers) != len(requests):
            return self._unread(
                criterion,
                candidates,
                "criterion_reading_incomplete",
                answered_by=answered_by,
                usage=usage,
                request_count=1,
            )
        try:
            reading = read_jev(dict(zip(questions, answers, strict=True)))
        except (TypeError, ValueError, KeyError):
            return self._unread(
                criterion,
                candidates,
                "criterion_reading_unusable",
                answered_by=answered_by,
                usage=usage,
                request_count=1,
            )
        resolved = resolve_with_policy(reading, candidates, self.policy)
        return CriterionReading(
            criterion=criterion,
            candidates=candidates,
            reading=reading,
            resolved=resolved,
            exact=None,
            unsupported=(),
            reason=(
                "criterion_reading_resolved"
                if resolved is not None
                else "criterion_reading_unresolved"
            ),
            policy_version=self.policy.version,
            questions_digest=questions_digest(),
            answered_by=answered_by if isinstance(answered_by, str) else None,
            usage=usage if isinstance(usage, Mapping) else {},
            judgment_probabilities={key: reading.noul[key] for key in NOUL_KEYS},
            selected_probabilities=_selected_probabilities(questions, answers),
            request_count=len(questions),
        )

    def _unread(
        self,
        criterion: str,
        candidates: Mapping[str, int],
        reason: str,
        *,
        answered_by: Any = None,
        usage: Any = None,
        request_count: int,
    ) -> CriterionReading:
        return CriterionReading(
            criterion=criterion,
            candidates=candidates,
            reading=None,
            resolved=None,
            exact=None,
            unsupported=(),
            reason=reason,
            policy_version=self.policy.version,
            questions_digest=questions_digest(),
            answered_by=answered_by if isinstance(answered_by, str) else None,
            usage=usage if isinstance(usage, Mapping) else {},
            request_count=request_count,
        )


def _request(
    model: str, key: str, question: Mapping[str, Any], criterion: str
) -> dict[str, Any]:
    return {
        "key": f"criterion-reading:{key}",
        "model": model,
        "type": str(question.get("type", "noul")),
        "state": criterion,
        "question": question["instructions"],
        **(
            {"criteria": dict(question["criteria"])}
            if question.get("type") == "choice"
            else {}
        ),
    }


def _selected_probabilities(
    questions: Mapping[str, Any], answers: Sequence[Any]
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for name, answer in zip(questions, answers, strict=True):
        try:
            decision = parse_decision(answer)
        except (ValueError, TypeError):
            continue
        if isinstance(decision, ChoiceDecision):
            selected[name] = decision.probabilities.get(decision.selected, 0.0)
        elif isinstance(decision, NoulDecision):
            selected[name] = decision.probability
    return selected


def _counted(base: CriterionReading, output: str) -> CriterionReading:
    """Count words or parse JSON in code, keeping the reading's evidence."""
    resolved = base.resolved
    if resolved is None:
        return base
    if resolved["kind"] == "word_count":
        exact = _word_check(
            str(resolved["operator"]),
            int(resolved["bound"]),
            count_words(output),
            upper=(
                int(resolved["upper_bound"])
                if resolved.get("upper_bound") is not None
                else None
            ),
        )
        # The reading supplied the operator and bound, so code can count words;
        # the regex's "words" name no longer describes anything unsupported.
        unsupported = tuple(name for name in base.unsupported if name != "words")
        return replace(base, exact=exact, unsupported=unsupported)
    if resolved["kind"] == "valid_json":
        valid = parses_as_json(output)
        passed = (not valid) if resolved.get("negated") else valid
        return replace(base, exact={"kind": "valid_json", "passed": passed})
    # Sentence counting is not implemented in code, so a sentence criterion
    # stays unresolved instead of being counted the wrong way.
    return replace(
        base,
        resolved=None,
        exact=None,
        reason="criterion_reading_sentence_count_unsupported",
    )


__all__ = [
    "CHOICE_KEYS",
    "CRITERION_READING_MIN_VERSION",
    "CURRENT_READING_POLICY",
    "JUDGMENTS",
    "NONE",
    "NOUL_KEYS",
    "OPERATORS",
    "READABLE_KINDS",
    "CriterionReader",
    "CriterionReading",
    "Reading",
    "ReadingPolicy",
    "number_candidates",
    "questions_digest",
    "read_jev",
    "reading_questions",
    "resolve",
    "resolve_with_policy",
]
