"""Check the miss-decision grader with fake LLMs, before spending on real ones.

Runs every case through run_miss_eval.run_case with three fake replies:
- oracle: the right answer, with the category's true description for a new option. Should score
  about 1.0, and its new_fit is roughly the best a description can do.
- null: always reuse the first current option. Should score about 0.
- garbage: an unparseable reply. Should score 0 with format_ok 0, not raise.

    uv run evals/sanity_check.py   # needs TYPESAFE_API_KEY for the option check
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from statistics import mean
from types import SimpleNamespace
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient

import run_miss_eval as runner

INTENTS = json.loads((runner.ROOT / "evals" / "banking77" / "intents.json").read_text())


def oracle(case: dict[str, Any]) -> Any:
    gold = case["gold"]
    if gold["decision"] == "create":
        return {"decision": "create", "key": gold["intent"], "description": INTENTS[gold["intent"]]}
    return {"decision": "reuse", "key": gold["intent"], "description": ""}


def null(case: dict[str, Any]) -> Any:
    return {"decision": "reuse", "key": next(iter(case["options"])), "description": ""}


def garbage(case: dict[str, Any]) -> Any:
    return "I'm not sure."


def fake_llm(policy: Callable[[dict[str, Any]], Any], cases: dict[str, dict[str, Any]]) -> Any:
    async def acompletion(**kwargs: Any) -> Any:
        state = kwargs["messages"][0]["content"].split("Input:\n", 1)[1].split("\n\n", 1)[0]
        reply = policy(cases[state])
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(
            model=kwargs["model"].rsplit("/", 1)[-1],
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            _hidden_params={"response_cost": 0.0},
        )

    return acompletion


async def main() -> None:
    review = json.loads(runner.REVIEW.read_text())
    cases = [json.loads(line) for line in runner.CASES.read_text().splitlines()]
    cases = [case for case in cases if case["id"] not in review["drop"]]
    for case in cases:
        case["gold"]["also"] = review["also"].get(case["id"], [])
    by_input = {case["input"]: case for case in cases}
    jev_limit = asyncio.Semaphore(16)
    async with AsyncTypeSafeClient() as client:
        for name, policy in [("oracle", oracle), ("null", null), ("garbage", garbage)]:
            runner._acompletion = fake_llm(policy, by_input)
            rows = [
                row
                for row, _ in await asyncio.gather(
                    *(runner.run_case(c, "fake/model", client, jev_limit) for c in cases)
                )
            ]
            grades = [row["grade"] for row in rows]
            fits = [g["new_fit"] for g in grades if "new_fit" in g]
            leaks = [g["new_leak"] for g in grades if "new_leak" in g]
            line = f"{name:8} correct {mean(g['correct'] for g in grades):.2f}  format_ok {mean(g['format_ok'] for g in grades):.2f}"
            if fits:
                line += f"  new_fit {mean(fits):.2f}  new_leak {mean(leaks):.2f} (over {len(fits)})"
            print(line)


if __name__ == "__main__":
    asyncio.run(main())
