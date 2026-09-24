"""Record only the exact default context question, reusing matching answers.

Run with ``uv run --env-file .env python``. Existing request-keyed answers are
copied without provider calls; every missing answer is measured live.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.diagnosis import default_gap_question
from prompt_enhancer.evaluation.datasets import load_dataset
from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.gateway import GatewayConfig, ModelGateway, ReplayGateway
from prompt_enhancer.jev import NoulDecision, parse_decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--question", default=default_gap_question("context"))
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    reused = json.loads(args.reuse.read_text(encoding="utf-8")).get("responses", {}) if args.reuse else {}
    if not isinstance(reused, dict):
        raise TypeError("reuse replay requires response object")
    recording = RecordingGateway(ModelGateway(config=GatewayConfig.from_env()), args.record)
    if args.record.exists():
        prior = json.loads(args.record.read_text(encoding="utf-8"))
        recording.responses.update(prior.get("responses", {}))
    copied = live = 0
    for case in dataset.cases:
        question = {
            "model": JEV_MODEL, "query": args.question,
            "state": {"prompt": case.prompt}, "type": "noul", "key": "gap:context",
        }
        key = ReplayGateway.request_key("decide", JEV_MODEL, question, "judge")
        if key in recording.responses:
            answer = recording.responses[key]
        elif key in reused:
            answer = reused[key]
            recording.responses[key] = answer
            recording.save()
            copied += 1
        else:
            answer = recording.decide(question)
            live += 1
        if not isinstance(parse_decision(answer), NoulDecision):
            raise TypeError(f"context response was not noul for {case.id}")
    print(f"Validated {len(dataset.cases)} context answers: {copied} reused, {live} new live")


if __name__ == "__main__":
    main()
