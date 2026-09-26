from __future__ import annotations

import litellm
import pytest

from fakes import FakeJev, FakeLLM


@pytest.fixture
async def jev():
    """Returns a function taking FakeJev's `fits` and returning (fake, client)."""
    clients = []

    def make(fits):
        fake = FakeJev(fits)
        clients.append(fake.client())
        return fake, clients[-1]

    yield make
    for client in clients:
        await client.aclose()


@pytest.fixture
def llm(monkeypatch):
    """Returns a function taking FakeLLM's arguments and installing it as litellm.acompletion."""

    def install(reply, delay=0.0):
        fake = FakeLLM(reply, delay)
        monkeypatch.setattr(litellm, "acompletion", fake.acompletion)
        return fake

    return install
