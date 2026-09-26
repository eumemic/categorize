# categorize

A [jev](https://docs.typesafe.ai/) Choice question that grows its own options.

You ask a question without listing the answers. jev picks from the options seen so far, plus
`none_of_these`. When no option fits, an LLM (any [LiteLLM](https://docs.litellm.ai/) model)
writes a new one, which is saved and offered to jev from then on. LLM calls grow with the number
of distinct answers, not the number of questions asked, and the same kind of input keeps getting
the same label.

```
state ──► jev: Choice over options + none_of_these
            │
            ├─ an option gets ≥ hit_threshold (and beats none_of_these) ──► that option   (hit)
            │
            └─ otherwise ──► LLM sees question, state, current and retired options
                               ├─ picks a current option                             (reused)
                               ├─ brings back an evicted option                      (restored)
                               └─ writes a new option, evicting the LRU one if full  (created)
```

## Usage

```python
import asyncio
from typesafe_sdk import AsyncTypeSafeClient, Noul
from categorize import OpenChoice, system_one

problem = OpenChoice(
    "What is the customer's underlying problem?",
    store="problems.json",            # options persist here between runs
    max_choices=64,                   # evict the least recently used option past this
    llm="anthropic/claude-haiku-4-5", # any LiteLLM model; default anthropic/claude-opus-5
    guidance="Name problem types many tickets could share, such as double_charge.",
)

async def main():
    async with AsyncTypeSafeClient() as client:
        answer = await problem.ask(client, "I was charged twice for order A-104.")
        print(answer.choice, answer.outcome, answer.description)

        # Mixed with ordinary jev questions: still one jev request.
        result = await system_one(client, "My parcel is a week late!", {
            "problem": problem,
            "urgent": Noul(instructions="Does the message convey urgency?"),
        })
        print(result.answers["problem"].choice, result.answers["urgent"].noul)

asyncio.run(main())
```

`examples/support_tickets.py` runs a dozen tickets through one question.

## The answer

| Field | Meaning |
| --- | --- |
| `choice` | The option's key. |
| `description` | The option's description, which is what jev matches against. |
| `outcome` | `hit` (jev chose it), or `reused` / `restored` / `created` (the LLM decided). `is_new` is `outcome == "created"`. |
| `probabilities`, `confidence` | jev's answer, including `none_of_these`. Empty/None when there were no options yet. |
| `evicted` | Key of the option evicted to make room, if any. |

`problem.options` lists the current options; `problem.stats` counts outcomes and evictions.

## Options

| Argument | Default | |
| --- | --- | --- |
| `store` | None (memory only) | JSON file path, or any object with `load()` and `save(state)`. |
| `max_choices` | 64 | 1–254. jev takes 255 options per Choice; one is `none_of_these`. |
| `hit_threshold` | 0.5 | Least probability jev must give an option to use it without the LLM. |
| `llm`, `llm_kwargs` | `anthropic/claude-opus-5` | Passed to `litellm.acompletion`. |
| `guidance` | None | Instructions for the LLM only, such as how broad options should be. |
| `seed` | None | Starting options, key to description. Never evicted; the seed in code overrides the store. |

## Behavior worth knowing

- **Misses are one at a time per question.** Concurrent calls that hit the same new kind of
  answer wait for the first to create the option, then check it with jev, so they reuse it
  instead of each creating a duplicate.
- **The LLM can reuse an option jev missed.** A new option whose normalized key matches an existing
  one also counts as a reuse. Many `reused` in `stats` means jev is missing matches; lowering
  `hit_threshold` or sharpening descriptions helps.
- **A false hit is silent**: the caller gets a plausible wrong option and no LLM sees it. That is
  why a hit needs `hit_threshold`, not just a top spot. The default of 0.5 is untuned; measure
  on your own data.
- **Evicted options are kept** (up to `max_choices` of them) and shown to the LLM, so a returning
  kind of answer gets its old key back.
- **No `response_format`.** The LLM is asked for JSON in the prompt instead. With structured
  outputs, claude-opus-5 sometimes corrupted text around dashes inside the JSON strings.
  Replies are parsed leniently and retried once.
- Async only. One process per store file.
- The options form one jev question, so their descriptions count toward jev's 32k-token budget
  for state plus the longest question.

## Development

```sh
uv run pytest            # offline: fake jev endpoint and fake LLM
uv run pytest -m live    # real APIs; costs money; needs ANTHROPIC_API_KEY (and TYPESAFE_API_KEY for end to end)
```
