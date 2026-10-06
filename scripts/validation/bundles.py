"""Plan complete review patches and lossless verification evidence separately."""

import math

from validation.receipts import digest


def estimated_tokens(text: str) -> int:
    # Conservative planning estimate, not a claim to reproduce a tokenizer.
    return math.ceil(len(text.encode()) / 2)


def partition(
    patches: list[dict],
    evidence: list[dict],
    *,
    context_tokens: int,
    reserve_tokens: int,
    dependencies: dict[str, list[dict]] | None = None,
) -> dict:
    budget = context_tokens - reserve_tokens
    if budget <= 0:
        raise ValueError("context must exceed the reserved request/output tokens")
    dependencies = dependencies or {}
    bundles = []
    coverage = []
    for patch in patches:
        related = dependencies.get(patch["path"], [])
        text = patch["diff"] + "".join(item["text"] for item in related)
        if len(patch["diff"]) > 50_000 or estimated_tokens(text) > budget:
            raise ValueError(
                f"Complete patch/dependencies for {patch['path']} exceed the budget"
            )
        bundles.append(
            {
                "kind": "patch_review",
                **patch,
                "evidence": related,
                "estimated_input_tokens": estimated_tokens(text),
            }
        )
    for item in evidence:
        text = item["text"]
        # Split by Unicode characters so byte bounds cannot corrupt UTF-8.
        chunks = []
        current = []
        size = 0
        for char in text:
            char_size = len(char.encode())
            if current and size + char_size > budget * 2:
                chunks.append("".join(current))
                current = []
                size = 0
            if char_size > budget * 2:
                raise ValueError("Evidence character cannot fit the budget")
            current.append(char)
            size += char_size
        if current:
            chunks.append("".join(current))
        indices = []
        for ordinal, chunk in enumerate(chunks):
            indices.append(len(bundles))
            bundles.append(
                {
                    "kind": "claim_verification",
                    "evidence": [{"id": f"{item['id']}:{ordinal}", "text": chunk}],
                    "estimated_input_tokens": estimated_tokens(chunk),
                }
            )
        coverage.append(
            {
                "id": item["id"],
                "sha256": digest(text.encode()),
                "bundle_indices": indices,
            }
        )
    return {
        "context_tokens": context_tokens,
        "reserve_tokens": reserve_tokens,
        "estimator": "ceil(UTF-8 bytes / 2); actual provider tokenization may differ",
        "patch_paths": [p["path"] for p in patches],
        "patch_hashes": {p["path"]: digest(p["diff"].encode()) for p in patches},
        "evidence_coverage": coverage,
        "bundles": bundles,
    }
