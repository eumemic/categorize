"""Tests against real APIs. They cost money, so they run only with `pytest -m live`."""

from __future__ import annotations

import os

import pytest
from typesafe_sdk import AsyncTypeSafeClient

from categorize import OpenChoice

pytestmark = pytest.mark.live
needs_llm = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="the default LLM needs ANTHROPIC_API_KEY"
)
needs_jev = pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="needs TYPESAFE_API_KEY")

QUESTION = "What is the customer's underlying problem?"


@needs_llm
async def test_llm_names_the_kind_of_answer_not_the_case(jev):
    _, client = jev({})

    answer = await OpenChoice(QUESTION).ask(client, "I was charged twice for order A-104. Please refund one.")

    assert answer.is_new
    assert "104" not in answer.choice
    assert isinstance(answer.description, str) and answer.description


@needs_llm
async def test_llm_reuses_an_option_that_fits(jev):
    _, client = jev({})  # the fake jev never finds a fit, so the LLM decides
    problem = OpenChoice(
        QUESTION, seed={"duplicate_charge": "The customer was billed more than once for one purchase."}
    )

    answer = await problem.ask(client, "You took the money for my subscription two times this month!")

    assert (answer.choice, answer.outcome) == ("duplicate_charge", "reused")


@needs_llm
@needs_jev
async def test_end_to_end():
    problem = OpenChoice(QUESTION)
    async with AsyncTypeSafeClient() as client:
        first = await problem.ask(client, "I was charged twice for my order last week. Please refund one of them.")
        same = await problem.ask(client, "My card got billed two times for a single purchase. I want my money back.")
        other = await problem.ask(client, "I can't log in. The password reset email never arrives.")

    assert first.is_new
    assert same.choice == first.choice
    assert other.choice != first.choice
