"""The miss path: ask an LLM, through LiteLLM, to reuse an option or write a new one."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import JSONContent

DEFAULT_LLM = "anthropic/claude-opus-5"
NONE_OF_THESE = "none_of_these"


@dataclass(frozen=True)
class Proposal:
    key: str
    """Normalized with `normalize_key`."""
    description: str
    """Empty when the LLM reused an option."""
    examples: tuple[str, ...] = ()
    """Other inputs that should get a new option; empty when the LLM reused one."""


def normalize_key(key: str) -> str:
    """Lowercase snake_case, so near-identical names compare equal."""
    return re.sub(r"[\W_]+", "_", key.strip().lower()).strip("_")[:64]


async def propose(
    llm: str,
    llm_kwargs: Mapping[str, Any],
    instructions: JSONContent,
    state: JSONContent,
    options: Mapping[str, JSONContent | None],
    retired: Mapping[str, JSONContent | None],
    guidance: str | None,
) -> Proposal:
    """Ask `llm` which option answers `instructions` for `state`, or for a new one.

    A malformed reply, or one that reuses a key that doesn't exist, is retried once with the
    error shown to the LLM; a second failure raises ValueError.
    """
    known = {normalize_key(key) for key in [*options, *retired]}
    messages = [{"role": "user", "content": _prompt(instructions, state, options, retired, guidance)}]
    reply = await _complete(llm, llm_kwargs, messages)
    try:
        return _parse(reply, known)
    except ValueError as error:
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"That reply is invalid: {error}. Reply again."},
        ]
    return _parse(await _complete(llm, llm_kwargs, messages), known)


async def _complete(llm: str, llm_kwargs: Mapping[str, Any], messages: list[dict[str, str]]) -> str:
    import litellm  # slow to import, and only needed on a miss

    # No response_format: constrained decoding sometimes corrupts the text around dashes (seen
    # with claude-opus-5, through LiteLLM and through the Anthropic SDK alike), while a JSON reply
    # asked for in the prompt parses reliably and works with every LiteLLM model.
    response = await litellm.acompletion(model=llm, messages=messages, **llm_kwargs)
    return response.choices[0].message.content or ""


def _parse(reply: str, known: Collection[str]) -> Proposal:
    found = re.search(r"\{.*\}", reply, re.DOTALL)  # tolerates code fences or prose around the object
    if found is None:
        raise ValueError("the reply contains no JSON object")
    data = json.loads(found.group())
    key = normalize_key(str(data.get("key", "")))
    if not key or key == NONE_OF_THESE:
        raise ValueError(f"{data.get('key')!r} is not a usable key")
    if data.get("decision") == "reuse" and key not in known:
        raise ValueError(f"{key!r} is not one of the current or retired answers")
    examples = data.get("examples") if isinstance(data.get("examples"), list) else []
    examples = tuple(" ".join(str(e).split()) for e in examples if str(e).strip())[:3]
    return Proposal(key, " ".join(str(data.get("description") or "").split()), examples)


def _prompt(
    instructions: JSONContent,
    state: JSONContent,
    options: Mapping[str, JSONContent | None],
    retired: Mapping[str, JSONContent | None],
    guidance: str | None,
) -> str:
    sections = [
        (
            "You keep the list of answers to a recurring question. For each new input, a fast "
            "classifier picks an answer from the list, seeing only each answer's key and "
            "description. You are asked when it is not sure any answer fits."
        ),
        f"Question:\n{_render(instructions)}",
    ]
    if guidance:
        sections.append(f"Guidance:\n{guidance}")
    sections.append(f"Current answers:\n{_listing(options) or '(none yet)'}")
    if retired:
        sections.append(f"Retired answers, which you can bring back:\n{_listing(retired)}")
    sections.append(f"Input:\n{_render(state)}")
    sections.append(
        "If a current or retired answer is the right answer to the question for this input, reply "
        'with decision "reuse" and its key, even if the input is worded differently, is more general '
        "or more specific than the answer's description, or mentions details the description doesn't. "
        "If the right answer is something else, create a new answer, even when the input shares a "
        'subject with an existing one. To create one, reply with decision "create", a new key, and a '
        "description. The key is a short snake_case name for the general kind of answer: one that "
        "other inputs "
        "could share, not the details of this input, and at the same level of generality as "
        "the current answers. The description is one sentence saying which inputs the answer "
        "covers, written so the classifier can tell it apart from the other answers. Also give "
        "two examples: short inputs, different from this one, that should get the same answer. "
        "Reply with only a JSON object with the fields decision, key, description, and examples "
        "(a list, empty when reusing)."
    )
    return "\n\n".join(sections)


def _render(value: JSONContent, indent: int | None = 2) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=indent)


def _listing(options: Mapping[str, JSONContent | None]) -> str:
    return "\n".join(
        f"- {key}" if description is None else f"- {key}: {_render(description, indent=None)}"
        for key, description in options.items()
    )
