"""Run the miss-decision eval: how well OpenChoice's LLM handles a miss.

Each case goes through the same code as a real miss: OpenChoice._propose asks the LLM and
OpenChoice._apply turns its reply into an answer. Grades per case and rep:

- correct: reused or restored the right option, or created one when the category is absent.
- format_ok: the first reply parsed, so the retry wasn't needed.
- new_fit: for a correctly created option, jev's mean probability on it for five other messages
  from the same category (higher is better).
- new_leak: jev's mean probability on it for five messages from categories among the options
  (lower is better).

    uv run evals/run_miss_eval.py --model anthropic/claude-opus-5 [--variant baseline] [--reps 2]

Writes to .claude/hillclimb/<flow>/<variant>/: results.jsonl, traces/, errors.jsonl, summary.json.
The flow defaults to miss-<model name>. A rerun resumes, skipping (case, rep) pairs already
done. The runner refuses to start when this file or the cases changed since the last run with
--approve-harness, which only the user should pass.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import litellm
from typesafe_sdk import (
    AsyncTypeSafeClient,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from categorize import OpenChoice

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "banking77" / "cases.jsonl"
REVIEW = ROOT / "evals" / "banking77" / "review.json"  # dropped cases, extra acceptable answers
HARNESS_PATHS = ["evals/run_miss_eval.py", "evals/banking77/cases.jsonl", "evals/banking77/review.json"]
METRICS = [
    {"id": "correct", "kind": "binary", "label": "Correct"},
    {"id": "format_ok", "kind": "binary", "label": "Format OK"},
    {"id": "new_fit", "kind": "float", "scale": 1, "label": "New fit"},
    {"id": "new_leak", "kind": "float", "scale": 1, "label": "New leak", "better": "lower"},
]
PERF_FIELDS = [
    {"id": "latency_s", "label": "Latency", "unit": "s"},
    {"id": "cost_usd", "label": "Cost", "unit": "$"},
    {"id": "in_tokens", "label": "In tokens"},
    {"id": "out_tokens", "label": "Out tokens"},
]
TRANSIENT = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)
ATTEMPTS = 4

# The LLM calls made by the current case, recorded by the wrapper around litellm.acompletion.
_calls: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("calls", default=None)
_acompletion = litellm.acompletion


async def _recording_acompletion(**kwargs: Any) -> Any:
    started = time.perf_counter()
    response = await _acompletion(**kwargs)
    calls = _calls.get()
    if calls is not None:
        calls.append(
            {"messages": list(kwargs["messages"]), "response": response, "seconds": time.perf_counter() - started}
        )
    return response


litellm.acompletion = _recording_acompletion


class CaseStore:
    """Hands OpenChoice a case's options and retired options."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.state = {
            "instructions": case["question"],
            "options": [{"key": k, "description": d} for k, d in case["options"].items()],
            "retired": [{"key": k, "description": d} for k, d in case["retired"].items()],
            "stats": {},
            "clock": 0,
        }

    def load(self) -> dict[str, Any]:
        return self.state

    def save(self, state: dict[str, Any]) -> None:
        pass


class ServedModelMismatch(Exception):
    pass


def describe_expected(case: dict[str, Any]) -> str:
    gold = case["gold"]
    also = "".join(f" or pick {key}" for key in gold.get("also", []))
    if gold["decision"] == "create":
        return f"create a new option (category: {gold['intent']}){also}"
    return f"{gold['decision']} {gold['intent']}{also}"


async def run_case(
    case: dict[str, Any], model: str, client: AsyncTypeSafeClient, jev_limit: asyncio.Semaphore
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One attempt at one case: the row for results.jsonl and the trace."""
    calls: list[dict[str, Any]] = []
    _calls.set(calls)
    choice = OpenChoice(case["question"], store=CaseStore(case), max_choices=254, llm=model)
    invalid = None
    try:
        proposal = await choice._propose(case["input"])
    except ValueError as error:  # the reply was invalid twice
        proposal, invalid = None, str(error)
    answer = choice._apply(proposal, None) if proposal else None
    if not calls:
        raise RuntimeError("no LLM call was recorded")
    served = calls[-1]["response"].model or ""
    if not served.rsplit("/", 1)[-1].startswith(model.rsplit("/", 1)[-1]):
        raise ServedModelMismatch(f"asked for {model}, served by {served}")

    got = {"outcome": answer.outcome, "key": answer.choice, "description": answer.description} if answer else None
    gold = case["gold"]
    if answer is None:
        correct = False
    elif answer.outcome == "created":
        correct = gold["decision"] == "create"
    else:  # reused or restored: right if it is the labeled option or an accepted alternative
        correct = answer.choice in ([] if gold["decision"] == "create" else [gold["intent"]]) + gold.get("also", [])
    grade: dict[str, float] = {"correct": int(correct), "format_ok": int(answer is not None and len(calls) == 1)}
    trace = [{"role": m["role"], "content": m["content"]} for m in calls[-1]["messages"]]
    trace.append({"role": "assistant", "content": calls[-1]["response"].choices[0].message.content or ""})
    check = None
    if correct and answer.outcome == "created":
        # Would jev now send the rest of this category to the new option, and not the neighbors?
        question = choice._question()

        async def p_new(text: str) -> float:
            async with jev_limit:
                response = await client.system_one(text, {"q": question})
            return response.answers["q"].probabilities.get(answer.choice, 0.0)

        texts = case["siblings"] + [n["text"] for n in case["neighbors"]]
        ps = await asyncio.gather(*(p_new(text) for text in texts))
        fit, leak = ps[: len(case["siblings"])], ps[len(case["siblings"]) :]
        grade |= {"new_fit": mean(fit), "new_leak": mean(leak)}
        check = {
            "same_category": [{"text": t, "p": round(p, 3)} for t, p in zip(case["siblings"], fit)],
            "neighbors": [{**n, "p": round(p, 3)} for n, p in zip(case["neighbors"], leak)],
        }
        trace.append({"role": "tool_call", "name": "jev_option_check", "content": json.dumps(got, indent=2)})
        trace.append({"role": "tool_result", "name": "jev_option_check", "content": json.dumps(check, indent=2)})

    usage = [c["response"].usage for c in calls]
    costs = [c["response"]._hidden_params.get("response_cost") for c in calls]
    finish = calls[-1]["response"].choices[0].finish_reason
    jev_top = max((p for k, p in case["jev"].items() if k != "none_of_these"), default=0.0)
    row = {
        "prompt_id": case["id"],
        "prompt": f"{case['input']}\n\nExpected: {describe_expected(case)}\n\n{calls[0]['messages'][0]['content']}",
        "tags": [case["kind"], "jev<0.5" if jev_top < 0.5 else "jev 0.5-0.8"],
        "grade": grade,
        "model": served,
        "latency_s": round(sum(c["seconds"] for c in calls), 2),
        "usage": {
            "input_tokens": sum(u.prompt_tokens or 0 for u in usage),
            "output_tokens": sum(u.completion_tokens or 0 for u in usage),
        },
        "cost_usd": round(sum(costs), 6) if all(isinstance(c, (int, float)) for c in costs) else None,
        "stop_reason": finish,
        "status": "truncated" if finish == "length" else "ok",
        "meta": {
            "expected": describe_expected(case),
            "got": got,
            "invalid_reply": invalid,
            "llm_calls": len(calls),
            "option_check": check,
        },
    }
    return row, trace


async def run(args: argparse.Namespace, cases: list[dict[str, Any]], out: Path) -> None:
    done = set()
    results = out / "results.jsonl"
    if results.exists():
        done = {(r["prompt_id"], r["rep"]) for r in map(json.loads, results.read_text().splitlines())}
    todo = [(case, rep) for rep in range(args.reps) for case in cases if (case["id"], rep) not in done]
    print(f"{len(todo)} to run ({len(done)} already done) on {args.model}, writing to {out.relative_to(ROOT)}")
    (out / "traces").mkdir(parents=True, exist_ok=True)
    limit, jev_limit = asyncio.Semaphore(args.concurrency), asyncio.Semaphore(16)
    finished, started = 0, time.monotonic()

    async with AsyncTypeSafeClient() as client:

        async def one(case: dict[str, Any], rep: int) -> None:
            nonlocal finished
            async with limit:
                for attempt in range(1, ATTEMPTS + 1):
                    failure = None
                    try:
                        row, trace = await asyncio.wait_for(
                            run_case(case, args.model, client, jev_limit), args.timeout_s
                        )
                    except asyncio.TimeoutError:
                        failure = ("timeout", f"no result after {args.timeout_s}s")
                    except ServedModelMismatch as error:
                        failure = ("model_mismatch", str(error))
                    except TRANSIENT as error:
                        if attempt < ATTEMPTS:
                            await asyncio.sleep(min(60, 2**attempt) * (0.5 + random.random()))
                            continue
                        failure = ("serving_error", f"{type(error).__name__}: {error}"[:500])
                    if failure:
                        with (out / "errors.jsonl").open("a") as file:
                            file.write(
                                json.dumps(
                                    {
                                        "prompt_id": case["id"],
                                        "rep": rep,
                                        "class": failure[0],
                                        "error": failure[1],
                                        "attempts": attempt,
                                    }
                                )
                                + "\n"
                            )
                        break
                    row = {"prompt_id": row.pop("prompt_id"), "rep": rep, **row}
                    row["meta"]["attempts"] = attempt
                    (out / "traces" / f"{case['id']}_rep{rep}.json").write_text(json.dumps(trace, indent=2))
                    with results.open("a") as file:
                        file.write(json.dumps(row) + "\n")
                    break
            finished += 1
            if finished % 20 == 0 or finished == len(todo):
                rate = (time.monotonic() - started) / finished
                print(f"  {finished}/{len(todo)} done, ~{rate * (len(todo) - finished):.0f}s left")

        await asyncio.gather(*(one(case, rep) for case, rep in todo))


def summarize(out: Path, ids: set[str]) -> None:
    rows = [r for r in map(json.loads, (out / "results.jsonl").read_text().splitlines()) if r["prompt_id"] in ids]
    ok = [r for r in rows if r.get("status", "ok") == "ok"]
    errors = (out / "errors.jsonl").read_text().splitlines() if (out / "errors.jsonl").exists() else []
    n = len(ok)
    if not n:
        print(f"\n{out.relative_to(ROOT)}: nothing graded ({len(errors)} failed attempts)")
        return
    acc = mean(r["grade"]["correct"] for r in ok)
    half = 1.96 * math.sqrt(acc * (1 - acc) / n) if n else 0.0
    by_kind: dict[str, list[int]] = defaultdict(list)
    for r in ok:
        by_kind[r["tags"][0]].append(r["grade"]["correct"])
    fits = [r["grade"]["new_fit"] for r in ok if "new_fit" in r["grade"]]
    leaks = [r["grade"]["new_leak"] for r in ok if "new_leak" in r["grade"]]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    print(f"\n{out.relative_to(ROOT)}: {n} graded ({len(rows) - n} truncated, {len(errors)} failed attempts)")
    print(
        f"  correct    {acc:.2f} ± {half:.2f}   " + "  ".join(f"{k} {mean(v):.2f}" for k, v in sorted(by_kind.items()))
    )
    print(f"  format_ok  {mean(r['grade']['format_ok'] for r in ok):.2f}")
    if fits:
        print(f"  new_fit    {mean(fits):.2f}   new_leak {mean(leaks):.2f}   (over {len(fits)} created options)")
    lat = sorted(r["latency_s"] for r in ok)
    print(
        f"  latency    median {lat[len(lat) // 2]:.1f}s   cost ${sum(costs):.2f}"
        + ("" if len(costs) == len(rows) else f" ({len(rows) - len(costs)} rows unpriced)")
    )


def harness_sha() -> str:
    digest = hashlib.sha256()
    for path in HARNESS_PATHS:
        digest.update(path.encode() + b"\0" + (ROOT / path).read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="LiteLLM model name")
    parser.add_argument("--flow", help="default: miss-<model name>")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--per-kind", type=int, help="only the first N cases of each kind (for a pilot)")
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--approve-harness", action="store_true", help="record the current harness as approved")
    args = parser.parse_args()
    if not re.fullmatch(r"baseline|v\d+", args.variant):
        parser.error("--variant must be baseline or v<N>")

    flow = ROOT / ".claude" / "hillclimb" / (args.flow or "miss-" + args.model.rsplit("/", 1)[-1])
    state_path = flow / "_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"reps": args.reps, "harness_paths": HARNESS_PATHS}
    )
    state |= {"metrics": METRICS, "perf_fields": PERF_FIELDS}
    if args.approve_harness:
        state["harness_sha"] = harness_sha()
    elif state.get("harness_sha") != harness_sha():
        sys.exit(
            "The eval harness changed since it was last approved (or was never approved). Review it, then rerun with --approve-harness."
        )
    flow.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    review = json.loads(REVIEW.read_text())
    cases = [json.loads(line) for line in CASES.read_text().splitlines()]
    cases = [case for case in cases if case["id"] not in review["drop"]]
    for case in cases:
        case["gold"]["also"] = review["also"].get(case["id"], [])
    if args.per_kind:
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            by_kind[case["kind"]].append(case)
        cases = [case for group in by_kind.values() for case in group[: args.per_kind]]
    out = flow / args.variant
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args, cases, out))
    summarize(out, {c["id"] for c in cases})


if __name__ == "__main__":
    main()
