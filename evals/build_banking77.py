"""Build the BANKING77 cases for the miss-decision eval.

Each case is a moment where OpenChoice asks its LLM: some current (and retired) options, and a
customer message that jev could not place confidently. The expected answer comes from BANKING77's
labels: reuse the message's category if it is a current option, restore it if it is retired,
otherwise create a new option.

    uv run evals/build_banking77.py

Writes to evals/banking77/: intents.json (a description per category), absorbers.json (for each
category, the others jev sends its messages to when it is missing), cases.jsonl, and cases.html
for review. A step whose file exists is skipped; delete the file to redo it. To grow the set
without disturbing the cases already reviewed, append more with a fresh seed:

    uv run evals/build_banking77.py --add reuse=12 --seed 78
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import litellm
from typesafe_sdk import AsyncTypeSafeClient, Choice

from categorize import NONE_OF_THESE, OpenChoice

HERE = Path(__file__).parent / "banking77"
DATA_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"
QUESTION = "What does the customer need help with?"
DRAFT_LLM = "anthropic/claude-opus-5"
MISS_THRESHOLD = 0.8  # jev must put at least this on an option for OpenChoice to skip the LLM
QUOTAS = {"reuse": 30, "create_near": 25, "create_far": 15, "restore": 10}
SIBLINGS = 5  # other messages from the case's category, for checking a new option
NEIGHBORS = 5  # messages from categories among the options, for checking it doesn't overreach
SEED = 77


class DictStore:
    """Hands OpenChoice a fixed set of options and retired options."""

    def __init__(self, options: dict[str, str], retired: dict[str, str]) -> None:
        self.state = {
            "instructions": QUESTION,
            "options": [{"key": k, "description": d} for k, d in options.items()],
            "retired": [{"key": k, "description": d} for k, d in retired.items()],
            "stats": {},
            "clock": 0,
        }

    def load(self) -> dict[str, Any]:
        return self.state

    def save(self, state: dict[str, Any]) -> None:
        pass


def load_split(split: str) -> dict[str, list[str]]:
    path = HERE / "data" / f"{split}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL.format(split=split), path)
    by_intent: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            by_intent[row["category"]].append(" ".join(row["text"].split()))
    return dict(sorted(by_intent.items()))


async def draft_intents(train: dict[str, list[str]]) -> dict[str, str]:
    examples = "\n\n".join(f"{name}:\n" + "\n".join(f"- {t}" for t in texts[:8]) for name, texts in train.items())
    prompt = (
        f"These are the {len(train)} categories a bank uses to sort customer support messages, each "
        f"with example messages.\n\n{examples}\n\nFor each category, write a one-sentence "
        "description saying which messages it covers, written so a classifier that sees only the "
        "category names and descriptions can tell it apart from the other categories. Reply with "
        "only a JSON object mapping each category name to its description."
    )
    response = await litellm.acompletion(
        model=DRAFT_LLM, messages=[{"role": "user", "content": prompt}], max_tokens=16000
    )
    content = response.choices[0].message.content
    drafted = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
    missing = set(train) - set(drafted)
    assert not missing, f"no description for {missing}"
    return {name: drafted[name] for name in train}


async def find_absorbers(
    client: AsyncTypeSafeClient, intents: dict[str, str], train: dict[str, list[str]]
) -> dict[str, list[str]]:
    """For each category, the others that jev sends its messages to when it is missing."""
    limit = asyncio.Semaphore(16)

    async def absorbers(name: str) -> tuple[str, list[str]]:
        question = Choice(instructions=QUESTION, criteria={k: v for k, v in intents.items() if k != name})
        mass: Counter[str] = Counter()
        for text in train[name][8:11]:  # not the messages the descriptions were drafted from
            async with limit:
                response = await client.system_one(text, {"q": question})
            mass.update(response.answers["q"].probabilities)
        return name, [k for k, _ in mass.most_common(5)]

    return dict(await asyncio.gather(*(absorbers(name) for name in intents)))


def candidate(
    rng: random.Random,
    kind: str,
    name: str,
    text: str,
    intents: dict[str, str],
    absorbers: dict[str, list[str]],
    test: dict[str, list[str]],
) -> dict[str, Any]:
    near = absorbers[name][:2]
    others = [k for k in intents if k != name]
    size = rng.randint(8, 40)
    if kind == "reuse":
        chosen = {name, *(near if rng.random() < 0.5 else [])}
    elif kind == "create_near":
        chosen = set(near)
    else:
        chosen = set()
        if kind == "create_far":
            others = [k for k in others if k not in absorbers[name]]
    chosen |= set(rng.sample([k for k in others if k not in chosen], size - len(chosen)))
    options = list(chosen)
    rng.shuffle(options)
    spare = [k for k in others if k not in chosen]
    retired = rng.sample(spare, rng.randint(1, 5)) if rng.random() < 0.5 else []
    if kind == "restore":
        retired = [*retired, name]
        rng.shuffle(retired)
    decision = {"create_near": "create", "create_far": "create"}.get(kind, kind)
    in_options = [k for k in absorbers[name] if k in chosen and k != name]
    in_options += [k for k in options if k not in in_options and k != name]
    return {
        "kind": kind,
        "question": QUESTION,
        "input": text,
        "options": {k: intents[k] for k in options},
        "retired": {k: intents[k] for k in retired},
        "gold": {"decision": decision, "intent": name},
        "siblings": rng.sample([t for t in test[name] if t != text], SIBLINGS),
        "neighbors": [{"intent": k, "text": rng.choice(test[k])} for k in in_options[:NEIGHBORS]],
    }


async def jev_view(client: AsyncTypeSafeClient, case: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    """Whether jev misses this case at MISS_THRESHOLD, and its top probabilities."""
    choice = OpenChoice(
        QUESTION, store=DictStore(case["options"], case["retired"]), hit_threshold=MISS_THRESHOLD, max_choices=254
    )
    response = await client.system_one(case["input"], {"q": choice._question()})
    answer = response.answers["q"]
    top = dict(sorted(answer.probabilities.items(), key=lambda kv: -kv[1])[:3])
    top.setdefault(NONE_OF_THESE, answer.probabilities.get(NONE_OF_THESE, 0.0))
    return choice._hit(answer) is None, top


async def build_cases(
    client: AsyncTypeSafeClient,
    intents: dict[str, str],
    absorbers: dict[str, list[str]],
    test: dict[str, list[str]],
    quotas: dict[str, int],
    seed: int,
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """New cases, `quotas[kind]` of each kind, numbered after the existing ones."""
    rng = random.Random(seed)
    cases: dict[str, list[dict[str, Any]]] = {kind: [] for kind in quotas}
    used_texts = {case["input"] for case in existing}
    uses = Counter(case["gold"]["intent"] for case in existing)
    limit = asyncio.Semaphore(16)

    async def view(case: dict[str, Any]) -> tuple[bool, dict[str, float]]:
        async with limit:
            return await jev_view(client, case)

    for _ in range(40):  # rounds of candidates; each round tops up every kind still short
        batch = []
        for kind, quota in quotas.items():
            for _ in range(max(0, quota - len(cases[kind])) * 3):
                name = rng.choice([k for k in intents if uses[k] < 2] or list(intents))
                text = rng.choice(test[name])
                if text not in used_texts:
                    batch.append(candidate(rng, kind, name, text, intents, absorbers, test))
        if not batch:
            break
        for case, (missed, top) in zip(batch, await asyncio.gather(*(view(c) for c in batch))):
            kind, name = case["kind"], case["gold"]["intent"]
            if missed and len(cases[kind]) < quotas[kind] and case["input"] not in used_texts and uses[name] < 2:
                case["jev"] = top
                cases[kind].append(case)
                used_texts.add(case["input"])
                uses[name] += 1
        print("  " + ", ".join(f"{kind} {len(found)}/{quotas[kind]}" for kind, found in cases.items()))
    ordered = []
    for kind, found in cases.items():
        start = sum(case["kind"] == kind for case in existing)
        for i, case in enumerate(found, start + 1):
            ordered.append({"id": f"{kind}-{i:02d}", **case})
    return ordered


def review_page(cases: list[dict[str, Any]], review: dict[str, Any]) -> str:
    def options_list(options: dict[str, str], gold: str) -> str:
        items = "".join(
            f"<li{' class=gold' if key == gold else ''}><b>{html.escape(key)}</b>: {html.escape(desc)}</li>"
            for key, desc in options.items()
        )
        return f"<ul>{items}</ul>"

    sections = []
    for case in cases:
        gold = case["gold"]
        jev = ", ".join(f"{html.escape(k)} {p:.2f}" for k, p in case["jev"].items())
        expected = (
            f"create a new option (the message is <b>{html.escape(gold['intent'])}</b>)"
            if gold["decision"] == "create"
            else f"{gold['decision']} <b>{html.escape(gold['intent'])}</b>"
        )
        retired = (
            f"<details><summary>{len(case['retired'])} retired options</summary>"
            f"{options_list(case['retired'], gold['intent'])}</details>"
            if case["retired"]
            else ""
        )
        siblings = "".join(f"<li>{html.escape(t)}</li>" for t in case["siblings"])
        neighbors = "".join(
            f"<li><b>{html.escape(n['intent'])}</b>: {html.escape(n['text'])}</li>" for n in case["neighbors"]
        )
        if case["id"] in review.get("also", {}):
            expected += f" (also accepted: {html.escape(', '.join(review['also'][case['id']]))})"
        dropped = review.get("drop", {}).get(case["id"])
        sections.append(
            f"<section{' class=dropped' if dropped else ''}><h2>{case['id']}"
            f"{' - dropped: ' + html.escape(dropped) if dropped else ''}</h2>"
            f"<p class=input>{html.escape(case['input'])}</p>"
            f"<p><b>Expected:</b> {expected}<br><b>jev:</b> {jev}</p>"
            f"<details><summary>{len(case['options'])} current options</summary>"
            f"{options_list(case['options'], gold['intent'])}</details>{retired}"
            f"<details><summary>Option check inputs</summary><p>Same category:</p><ul>{siblings}</ul>"
            f"<p>Categories among the options:</p><ul>{neighbors}</ul></details></section>"
        )
    counts = Counter(case["kind"] for case in cases)
    summary = ", ".join(f"{n} {kind}" for kind, n in counts.items())
    return f"""<!doctype html><meta charset=utf-8><title>Miss-decision cases</title>
<style>
body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; max-width: 860px; margin: 2em auto; padding: 0 16px;
       color: #1d1d1f; background: #fff; }}
@media (prefers-color-scheme: dark) {{ body {{ color: #e8e8ea; background: #161618; }} section {{ border-color: #333; }} }}
section {{ border-top: 1px solid #ddd; padding: .5em 0 1em; }}
h2 {{ font-size: 15px; margin: .6em 0 .2em; font-family: ui-monospace, monospace; }}
.input {{ font-size: 18px; margin: .2em 0; }}
li.gold {{ background: #ffe58f55; }}
.dropped {{ opacity: .45; }}
summary {{ cursor: pointer; }}
</style>
<h1>Miss-decision eval cases</h1>
<p>{len(cases)} cases ({summary}). Question: <i>{html.escape(QUESTION)}</i><br>
Every input is one jev could not place with at least {MISS_THRESHOLD} probability, so OpenChoice would ask
the LLM. The message's category is highlighted wherever it appears among the options.</p>
{"".join(sections)}"""


async def main(add: dict[str, int], seed: int) -> None:
    train, test = load_split("train"), load_split("test")
    print(
        f"BANKING77: {len(train)} categories, {sum(map(len, train.values()))} train / {sum(map(len, test.values()))} test messages"
    )
    paths = {name: HERE / f"{name}.json" for name in ("intents", "absorbers")}
    if not paths["intents"].exists():
        print(f"Drafting category descriptions with {DRAFT_LLM}...")
        paths["intents"].write_text(json.dumps(await draft_intents(train), indent=2) + "\n")
    intents = json.loads(paths["intents"].read_text())
    async with AsyncTypeSafeClient() as client:
        if not paths["absorbers"].exists():
            print("Finding where jev sends each category's messages when it is missing...")
            paths["absorbers"].write_text(json.dumps(await find_absorbers(client, intents, train), indent=2) + "\n")
        absorbers = json.loads(paths["absorbers"].read_text())
        cases_path = HERE / "cases.jsonl"
        if not cases_path.exists():
            print("Sampling cases jev misses...")
            cases = await build_cases(client, intents, absorbers, test, QUOTAS, SEED, [])
            cases_path.write_text("".join(json.dumps(case) + "\n" for case in cases))
        if add:
            print(f"Sampling more cases jev misses (seed {seed})...")
            existing = [json.loads(line) for line in cases_path.read_text().splitlines()]
            added = await build_cases(client, intents, absorbers, test, add, seed, existing)
            with cases_path.open("a") as file:
                file.write("".join(json.dumps(case) + "\n" for case in added))
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    review_path = HERE / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    (HERE / "cases.html").write_text(review_page(cases, review))
    print(f"{len(cases)} cases in {cases_path}; review page: {HERE / 'cases.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--add", nargs="*", default=[], metavar="KIND=N", help="append N more cases of KIND")
    parser.add_argument("--seed", type=int, default=SEED + 1, help="seed for --add; use a new one each time")
    args = parser.parse_args()
    asyncio.run(main({kind: int(n) for kind, n in (item.split("=") for item in args.add)}, args.seed))
