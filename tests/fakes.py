"""Stand-ins for the jev endpoint and litellm.acompletion."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from categorize import NONE_OF_THESE


class FakeJev:
    """Answers Choice questions from `fits`, which maps a state to the option that fits it,
    or to explicit probabilities. States not in `fits` get all probability on none_of_these.
    Noul questions always get 0.9."""

    def __init__(self, fits: dict[str, str | dict[str, float]]) -> None:
        self.fits = fits
        self.requests: list[dict[str, Any]] = []

    def client(self) -> AsyncTypeSafeClient:
        return AsyncTypeSafeClient(
            api_key="test", transport=httpx2.MockTransport(self._handle), retry=RetryPolicy(max_retries=0)
        )

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        answers = {key: self._answer(body["state"], question) for key, question in body["questions"].items()}
        return httpx2.Response(
            200, json={"model": "jev-fake", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}
        )

    def _answer(self, state: Any, question: dict[str, Any]) -> dict[str, Any]:
        if question["type"] == "noul":
            return {"type": "noul", "noul": 0.9}
        fit = self.fits.get(json.dumps(state) if not isinstance(state, str) else state)
        probabilities = dict.fromkeys(question["criteria"], 0.0)
        if isinstance(fit, dict):
            probabilities.update(fit)
        elif fit in probabilities:
            probabilities.update({fit: 0.95, NONE_OF_THESE: 0.05})
        else:
            probabilities[NONE_OF_THESE] = 1.0
        choice = max(probabilities, key=probabilities.__getitem__)
        return {"type": "choice", "choice": choice, "confidence": probabilities[choice], "probabilities": probabilities}


class FakeLLM:
    """Replaces litellm.acompletion. `reply` gets the input state from the prompt and returns
    the reply: a dict (sent as JSON) or a raw string."""

    def __init__(self, reply: Callable[[str], dict[str, str] | str], delay: float = 0.0) -> None:
        self.reply = reply
        self.delay = delay
        self.prompts: list[str] = []

    async def acompletion(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self.prompts.append(messages[-1]["content"])
        await asyncio.sleep(self.delay)
        state = messages[0]["content"].split("Input:\n", 1)[1].split("\n\n", 1)[0]
        content = self.reply(state)
        message = SimpleNamespace(content=content if isinstance(content, str) else json.dumps(content))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def create(key: str, description: str = "d") -> dict[str, str]:
    return {"decision": "create", "key": key, "description": description}


def reuse(key: str) -> dict[str, str]:
    return {"decision": "reuse", "key": key, "description": ""}
