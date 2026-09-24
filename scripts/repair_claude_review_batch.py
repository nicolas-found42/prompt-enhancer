"""Replace Claude metadata/sidechain entries with main-session user turns.

This reads private local session logs and writes only to an ignored local path.
It does not produce human gap labels or submit prompts to a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation_review_common import SECRET


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    return ""


def _main_messages(path: Path) -> list[tuple[int, str, str]]:
    messages: list[tuple[int, str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("type") not in {"user", "assistant"}
                or event.get("isMeta")
                or event.get("isSidechain")
            ):
                continue
            message = event.get("message")
            if not isinstance(message, dict) or message.get("role") != event["type"]:
                continue
            text = _text(message.get("content")).strip()
            if text:
                messages.append((line_number, event["type"], text))
    return messages


def _normalized(value: str) -> str:
    return " ".join(value.split())


def repair(
    batch: dict[str, Any], session_root: Path
) -> tuple[dict[str, Any], list[str]]:
    cases = batch["cases"]
    used_sessions = {(case["source"], case["session"]) for case in cases}
    invalid: list[str] = []
    for case in cases:
        if case["source"] != "claude":
            continue
        messages = _main_messages(session_root / case["session"])
        prompt = _normalized(case["prompt"])
        if not any(
            role == "user" and prompt in _normalized(text) for _, role, text in messages
        ):
            invalid.append(case["id"])

    candidates: list[tuple[str, dict[str, Any]]] = []
    for path in session_root.glob("**/*.jsonl"):
        relative = str(path.relative_to(session_root))
        if "subagents" in path.parts or ("claude", relative) in used_sessions:
            continue
        messages = _main_messages(path)
        if not messages:
            continue
        first_user = next(
            (
                (index, line, text)
                for index, (line, role, text) in enumerate(messages)
                if role == "user"
            ),
            None,
        )
        if first_user is None:
            continue
        index, line, prompt = first_user
        if not 30 <= len(prompt) <= 4000 or SECRET.search(prompt):
            continue
        result_parts: list[str] = []
        for _, role, text in messages[index + 1 :]:
            if role == "user":
                break
            if role == "assistant" and text not in result_parts:
                result_parts.append(text)
        result = "\n".join(result_parts).strip()
        if len(result) < 30 or SECRET.search(result):
            continue
        candidate = {
            "prompt": prompt,
            "result": result[:8000],
            "session": relative,
            "source": "claude",
            "human_labels": None,
            "source_event_line": line,
        }
        rank = hashlib.sha256(relative.encode()).hexdigest()
        candidates.append((rank, candidate))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < len(invalid):
        raise ValueError(
            f"need {len(invalid)} replacement sessions; found {len(candidates)}"
        )
    replacement_by_id = {
        case_id: {"id": case_id, **candidate}
        for case_id, (_, candidate) in zip(invalid, candidates, strict=False)
    }
    revised = {
        **batch,
        "cases": [replacement_by_id.get(case["id"], case) for case in cases],
    }
    revised["notes"] = (
        str(batch.get("notes", ""))
        + " Claude metadata and sidechain entries were replaced with verified main-session first user turns."
    )
    if len({(case["source"], case["session"]) for case in revised["cases"]}) != len(
        revised["cases"]
    ):
        raise ValueError("repaired batch contains duplicate source sessions")
    return revised, invalid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--session-root", type=Path, default=Path.home() / ".claude" / "projects"
    )
    args = parser.parse_args()
    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    revised, invalid = repair(batch, args.session_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(revised, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Replaced {len(invalid)} non-main Claude entries; retained {len(revised['cases'])} cases"
    )


if __name__ == "__main__":
    main()
