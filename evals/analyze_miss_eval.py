"""Break down miss-decision results by kind of failure.

    uv run evals/analyze_miss_eval.py miss-claude-opus-5 [miss-deepseek-v4.1-flash ...] [--variant v1 --against baseline]

For each flow: accuracy by case kind; failures by type (false merge: reused or restored when
it should have created; duplicate: created when it should have reused or restored; wrong pick:
reused or restored the wrong option; invalid reply); how often the two reps agree; and the new
options jev can't use (new_fit under 0.5) or that pull in other categories (new_leak over 0.2).
With --against, also the paired change in `correct` per kind: each case's mean over reps in
--variant minus its mean in the other variant, with a 95% interval over cases.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent


def failure(row: dict) -> str | None:
    if row["grade"]["correct"]:
        return None
    got, expected = row["meta"]["got"], row["meta"]["expected"]
    if got is None:
        return "invalid reply"
    if expected.startswith("create"):
        return "false merge"
    return "duplicate" if got["outcome"] == "created" else "wrong pick"


def per_case(path: Path) -> dict[str, tuple[str, float]]:
    """Each case's kind and mean `correct` over its reps."""
    reps: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        reps[row["prompt_id"]].append(row)
    return {pid: (rows[0]["tags"][0], mean(r["grade"]["correct"] for r in rows)) for pid, rows in reps.items()}


def paired(deltas: list[float]) -> str:
    if len(deltas) < 2:
        return f"{mean(deltas):+.2f}" if deltas else "n/a"
    half = 1.96 * stdev(deltas) / math.sqrt(len(deltas))
    return f"{mean(deltas):+.2f} [{mean(deltas) - half:+.2f}, {mean(deltas) + half:+.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flows", nargs="+")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--against", help="another variant to compare with, case by case")
    parser.add_argument("--list", action="store_true", help="list every failure")
    args = parser.parse_args()
    for flow in args.flows:
        path = ROOT / ".claude" / "hillclimb" / flow / args.variant / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        by_kind: dict[str, list[int]] = defaultdict(list)
        by_case: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_kind[row["tags"][0]].append(row["grade"]["correct"])
            by_case[row["prompt_id"]].append(row)
        failures = Counter(f for f in map(failure, rows) if f)
        agree = [
            len({(r["meta"]["got"] or {}).get("key") for r in reps}) == 1 for reps in by_case.values() if len(reps) > 1
        ]
        print(f"== {flow}/{args.variant}: {len(rows)} rows, correct {mean(r['grade']['correct'] for r in rows):.2f}")
        print("   by kind:   " + "  ".join(f"{k} {mean(v):.2f} (n={len(v)})" for k, v in sorted(by_kind.items())))
        print("   failures:  " + ", ".join(f"{k} {n}" for k, n in failures.most_common()) or "none")
        if agree:
            print(f"   reps pick the same key in {mean(agree):.0%} of cases")
        if args.against:
            now, before = per_case(path), per_case(path.parent.parent / args.against / "results.jsonl")
            shared = [pid for pid in now if pid in before]
            deltas: dict[str, list[float]] = defaultdict(list)
            for pid in shared:
                deltas[now[pid][0]].append(now[pid][1] - before[pid][1])
            print(
                f"   vs {args.against}: correct {paired([d for ds in deltas.values() for d in ds])} over {len(shared)} cases"
            )
            print("   " + "  ".join(f"{kind} {paired(ds)}" for kind, ds in sorted(deltas.items())))
        weak = [r for r in rows if r["grade"].get("new_fit", 1) < 0.5 or r["grade"].get("new_leak", 0) > 0.2]
        for row in weak:
            got = row["meta"]["got"]
            print(
                f"   weak new option in {row['prompt_id']} rep {row['rep']}: {got['key']} "
                f"(fit {row['grade']['new_fit']:.2f}, leak {row['grade']['new_leak']:.2f}): {got['description']}"
            )
        if args.list:
            for row in rows:
                kind = failure(row)
                if kind:
                    got = row["meta"]["got"] or {}
                    print(
                        f"   {kind:13} {row['prompt_id']} rep {row['rep']}: expected {row['meta']['expected']}; "
                        f"got {got.get('outcome')} {got.get('key')}  | {row['prompt'].splitlines()[0][:90]}"
                    )


if __name__ == "__main__":
    main()
