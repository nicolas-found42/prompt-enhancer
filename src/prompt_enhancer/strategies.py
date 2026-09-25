"""Named prompt-rewrite strategies and deterministic strategy search.

The strategy library deliberately keeps the policy separate from a model gateway.
A writer can be supplied for production rewrites, while the deterministic local
rewriter keeps the engine and its tests useful without provider calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import Tier, TierBudget

StrategyKind = Literal["safe", "crutch"]


@dataclass(frozen=True)
class RewriteStrategy:
    """A named rewrite policy and its safety classification."""

    name: str
    kind: StrategyKind
    description: str
    applies_to: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    priority: int = 0
    gap_fill_keys: tuple[str, ...] = ()
    restructures: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "applies_to": list(self.applies_to),
            "keywords": list(self.keywords),
            "priority": self.priority,
            "gap_fill_keys": list(self.gap_fill_keys),
            "restructures": self.restructures,
        }


# The ordering is the default policy order.  A prompt can still change the
# order through diagnosis matches and previous-round failures.
STRATEGY_LIBRARY: tuple[RewriteStrategy, ...] = (
    RewriteStrategy(
        name="add_missing_context",
        kind="safe",
        description="Add the context needed to make the request unambiguous.",
        applies_to=("context", "background", "missing_context"),
        keywords=("context", "background", "audience"),
        priority=100,
        gap_fill_keys=(
            "context",
            "language",
            "sources",
            "time_horizon",
            "outside_reference",
        ),
    ),
    RewriteStrategy(
        name="specify_output_format",
        kind="safe",
        description="State the expected shape and order of the answer.",
        applies_to=("output_format", "format", "deliverable"),
        keywords=("format", "output", "table", "json", "list"),
        priority=95,
        gap_fill_keys=("output_format", "format"),
    ),
    RewriteStrategy(
        name="add_done_criteria",
        kind="safe",
        description="Define observable completion criteria for the answer.",
        applies_to=("done", "success", "acceptance", "criteria"),
        keywords=("done", "success", "complete", "criteria"),
        priority=90,
        gap_fill_keys=("done_criteria", "done", "success", "acceptance", "criteria"),
    ),
    RewriteStrategy(
        name="remove_contradictions",
        kind="safe",
        description="Resolve conflicting instructions without inventing facts.",
        applies_to=("contradiction", "conflict", "vagueness"),
        keywords=("but", "however", "instead", "never", "always"),
        priority=85,
    ),
    RewriteStrategy(
        name="split_into_steps",
        kind="crutch",
        description="Break the request into explicit steps for weak models.",
        applies_to=("planning", "reasoning", "workflow"),
        keywords=("plan", "steps", "workflow", "procedure"),
        priority=40,
    ),
    RewriteStrategy(
        name="add_example",
        kind="crutch",
        description="Provide a concrete pattern or example for the response.",
        applies_to=("example", "style", "ambiguity"),
        keywords=("example", "like", "similar", "format"),
        priority=35,
    ),
    RewriteStrategy(
        name="role_play",
        kind="crutch",
        description="Assign a persona to make the expected response clearer.",
        applies_to=("audience", "tone", "persona"),
        keywords=("act as", "expert", "professional", "tone"),
        priority=30,
    ),
    RewriteStrategy(
        name="repeated_emphasis",
        kind="crutch",
        description="Repeat a key instruction to increase adherence.",
        applies_to=("emphasis", "constraint", "adherence"),
        keywords=("important", "must", "required", "ensure"),
        priority=25,
    ),
)

# A short alias is useful to callers and keeps the public API discoverable.
STRATEGIES = STRATEGY_LIBRARY


@dataclass(frozen=True)
class CandidateBatchRequest:
    """The one structured request sent to a candidate writer."""

    prompt: str
    strategies: tuple[RewriteStrategy, ...]
    previous_failures: tuple[str, ...] = ()
    diagnosis: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "strategies": [strategy.to_dict() for strategy in self.strategies],
            "previous_failures": list(self.previous_failures),
            "diagnosis": dict(self.diagnosis),
        }


@dataclass(frozen=True)
class CandidateDraft:
    """A generated candidate before it is run and graded."""

    candidate_id: str
    text: str
    strategy: RewriteStrategy
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.text

    @property
    def strategy_name(self) -> str:
        return self.strategy.name

    @property
    def strategy_kind(self) -> str:
        return self.strategy.kind

    @property
    def is_crutch(self) -> bool:
        return self.strategy.kind == "crutch"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "text": self.text,
            "prompt": self.text,
            "strategy": self.strategy_name,
            "strategy_kind": self.strategy_kind,
            "is_crutch": self.is_crutch,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StrategyRejection:
    strategy: str
    reason: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "reason": self.reason, "score": self.score}


@dataclass(frozen=True)
class StrategySearchResult:
    """Selected strategies, generated candidates, and audit information."""

    candidates: tuple[CandidateDraft, ...]
    selected_strategies: tuple[RewriteStrategy, ...]
    rejections: tuple[StrategyRejection, ...]
    budget: TierBudget
    previous_failures: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "strategies": [strategy.to_dict() for strategy in self.selected_strategies],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
            "previous_failures": list(self.previous_failures),
        }


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _diagnosis_terms(diagnosis: Any) -> set[str]:
    """Extract stable, lower-case labels from diagnoses and prior runs."""

    if diagnosis is None:
        return set()
    raw: list[Any] = []
    if isinstance(diagnosis, str):
        raw.append(diagnosis)
    elif isinstance(diagnosis, Mapping):
        for key in ("gaps", "issues", "diagnosis", "labels", "tags"):
            value = diagnosis.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                raw.extend(value)
        raw.extend(diagnosis.get(key) for key in ("task_type", "task", "summary"))
    else:
        for key in (
            "gaps",
            "issues",
            "diagnosis",
            "labels",
            "tags",
            "task_type",
            "task",
            "summary",
        ):
            value = getattr(diagnosis, key, None)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                raw.extend(value)
            elif value is not None:
                raw.append(value)
    terms: set[str] = set()
    for item in raw:
        if isinstance(item, Mapping):
            item = _get(item, "id") or _get(item, "name") or _get(item, "type")
        if item is None:
            continue
        text = str(item).lower()
        terms.update(
            part.strip() for part in text.replace("-", "_").split() if part.strip()
        )
    return terms


def _strategy_score(
    strategy: RewriteStrategy,
    prompt: str,
    diagnosis: Any,
    previous_failures: Sequence[str],
) -> int:
    terms = _diagnosis_terms(diagnosis)
    prompt_lower = prompt.lower()
    failures_lower = " ".join(str(item).lower() for item in previous_failures)
    score = strategy.priority
    score += 40 * len(set(strategy.applies_to) & terms)
    score += 8 * sum(1 for keyword in strategy.keywords if keyword in prompt_lower)
    score += 30 * sum(
        1
        for keyword in (strategy.name, *strategy.keywords)
        if keyword.replace("_", " ") in failures_lower or keyword in failures_lower
    )
    # Safe strategies are searched first unless a previous failure explicitly
    # targets a crutch.  This prevents weak-model scaffolding from becoming
    # the default while still allowing Deep to test it.
    if strategy.kind == "safe":
        score += 2
    return score


def _default_candidate_text(prompt: str, strategy: RewriteStrategy) -> str:
    """Produce a deterministic, test-friendly local rewrite.

    Production callers should pass a writer.  These changes are deliberately
    ordinary instructions rather than internal markers or placeholders.
    """

    text = prompt.rstrip()
    if strategy.name == "add_missing_context":
        return f"{text}\n\nContext: Use the relevant background supplied by the user."
    if strategy.name == "specify_output_format":
        return (
            f"{text}\n\nOutput: Present the answer in a clear, directly usable format."
        )
    if strategy.name == "add_done_criteria":
        return f"{text}\n\nDone means the response fully answers the request and can be used as requested."
    if strategy.name == "remove_contradictions":
        return f"{text}\n\nIf instructions conflict, follow the latest specific instruction and state the conflict."
    if strategy.name == "split_into_steps":
        return f"{text}\n\nWork through the request in clear, ordered steps."
    if strategy.name == "add_example":
        return f"{text}\n\nInclude a short concrete example when it makes the expected result clearer."
    if strategy.name == "role_play":
        return f"{text}\n\nRespond in the voice of a careful domain expert."
    if strategy.name == "repeated_emphasis":
        return f"{text}\n\nThis instruction is important: follow it precisely."
    return text


def _call_writer(
    writer: Any,
    request: CandidateBatchRequest,
) -> Any:
    """Call either a model-gateway writer or a simple callable.

    A writer receives one :class:`CandidateBatchRequest`, ensuring the K
    strategies are generated in one structured call.  A mapping keyed by
    strategy name or a sequence of strings/dicts is accepted for small fakes.
    """

    if hasattr(writer, "generate_candidates"):
        return writer.generate_candidates(request)
    if hasattr(writer, "write_candidates"):
        return writer.write_candidates(request)
    if hasattr(writer, "write"):
        return tuple(
            writer.write(request.prompt, (), ()).text for _ in request.strategies
        )
    if callable(writer):
        return writer(request)
    raise TypeError(
        "writer must be callable or expose generate_candidates/write_candidates/write"
    )


def _writer_texts(
    writer_result: Any, strategies: Sequence[RewriteStrategy]
) -> list[str] | None:
    if writer_result is None:
        return None
    if isinstance(writer_result, Mapping):
        texts: list[str] = []
        for strategy in strategies:
            value = writer_result.get(strategy.name)
            if value is None:
                value = writer_result.get(strategy.name.replace("_", " "))
            if isinstance(value, Mapping):
                value = value.get("text", value.get("prompt"))
            if value is not None:
                texts.append(str(value))
        return texts if texts else None
    if isinstance(writer_result, Sequence) and not isinstance(
        writer_result, (str, bytes)
    ):
        texts = []
        for value in writer_result:
            if isinstance(value, Mapping):
                value = value.get("text", value.get("prompt", value.get("candidate")))
            texts.append(str(value))
        return texts
    return None


def search_strategies(
    prompt: str,
    diagnosis: Any = None,
    tier: Tier | str = "standard",
    *,
    budget: TierBudget | None = None,
    strategies: Iterable[RewriteStrategy] = STRATEGY_LIBRARY,
    writer: Any = None,
    previous_failures: Sequence[str] | None = None,
    previous_round_failures: Sequence[str] | None = None,
    include_crutch: bool = True,
    recheck: Callable[[RewriteStrategy], Any] | None = None,
    priority_strategy: str | None = None,
) -> StrategySearchResult:
    """Rank strategies, apply the budget, and generate one candidate each.

    ``previous_round_failures`` is an alias accepted for repeat-round callers.
    Rejected strategies are retained in the result for the report rather than
    disappearing silently.
    """

    selected_budget = budget or Tier.parse(tier).budget
    failures = tuple(
        str(item)
        for item in (
            previous_failures
            if previous_failures is not None
            else previous_round_failures or ()
        )
    )
    available = list(strategies)
    if not include_crutch:
        available = [strategy for strategy in available if strategy.kind != "crutch"]

    ranked = sorted(
        available,
        key=lambda strategy: (
            strategy.name != priority_strategy,
            -_strategy_score(strategy, prompt, diagnosis, failures),
            available.index(strategy),
        ),
    )
    rejected: list[StrategyRejection] = []
    eligible: list[RewriteStrategy] = []
    for strategy in ranked:
        score = _strategy_score(strategy, prompt, diagnosis, failures)
        if recheck is not None:
            try:
                decision = recheck(strategy)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                rejected.append(
                    StrategyRejection(strategy.name, f"recheck failed: {exc}", score)
                )
                continue
            passed = _get(decision, "eligible", _get(decision, "pass", decision))
            if passed is False:
                rejected.append(
                    StrategyRejection(
                        strategy.name, "rejected by strategy recheck", score
                    )
                )
                continue
        if len(eligible) >= selected_budget.candidates:
            rejected.append(
                StrategyRejection(strategy.name, "outside tier candidate budget", score)
            )
            continue
        eligible.append(strategy)

    if not eligible:
        return StrategySearchResult((), (), tuple(rejected), selected_budget, failures)

    request = CandidateBatchRequest(
        prompt,
        tuple(eligible),
        failures,
        diagnosis=diagnosis if isinstance(diagnosis, Mapping) else {},
    )
    generated: list[str] | None = None
    if writer is not None:
        generated = _writer_texts(_call_writer(writer, request), eligible)
    if writer is not None and (generated is None or len(generated) < len(eligible)):
        return StrategySearchResult(
            (),
            (),
            tuple(rejected)
            + tuple(
                StrategyRejection(
                    strategy.name,
                    "writer did not return a complete candidate",
                    _strategy_score(strategy, prompt, diagnosis, failures),
                )
                for strategy in eligible
            ),
            selected_budget,
            failures,
        )
    if generated is None or len(generated) < len(eligible):
        generated = [text or "" for text in (generated or [])]
        generated.extend(
            _default_candidate_text(prompt, strategy)
            for strategy in eligible[len(generated) :]
        )

    candidates = tuple(
        CandidateDraft(
            candidate_id=f"candidate-{index + 1}-{strategy.name}",
            text=text,
            strategy=strategy,
            metadata={
                "rank_score": _strategy_score(strategy, prompt, diagnosis, failures)
            },
        )
        for index, (strategy, text) in enumerate(zip(eligible, generated, strict=False))
    )
    return StrategySearchResult(
        candidates=candidates,
        selected_strategies=tuple(eligible),
        rejections=tuple(rejected),
        budget=selected_budget,
        previous_failures=failures,
    )


__all__ = [
    "STRATEGIES",
    "STRATEGY_LIBRARY",
    "CandidateBatchRequest",
    "CandidateDraft",
    "RewriteStrategy",
    "StrategyKind",
    "StrategyRejection",
    "StrategySearchResult",
    "search_strategies",
]
