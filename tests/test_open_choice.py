from __future__ import annotations

import asyncio
import json

import pytest
from typesafe_sdk import Noul, NoulAnswer

from categorize import NONE_OF_THESE, JsonFileStore, OpenChoice, system_one

from fakes import FakeJev, create, reuse

QUESTION = "What is the customer's underlying problem?"


def never(state):
    pytest.fail(f"the LLM was asked about {state!r}")


async def test_first_answer_comes_from_the_llm_without_asking_jev(jev, llm):
    fake_jev, client = jev({})
    fake_llm = llm(lambda state: create("billing", "Charges and refunds."))
    problem = OpenChoice(QUESTION)

    answer = await problem.ask(client, "I was charged twice.")

    assert (answer.choice, answer.outcome, answer.is_new) == ("billing", "created", True)
    assert answer.description == "Charges and refunds."
    assert (answer.probabilities, answer.confidence) == ({}, None)
    assert problem.options == {"billing": "Charges and refunds."}
    assert fake_jev.requests == []
    assert len(fake_llm.prompts) == 1


async def test_confident_jev_answer_skips_the_llm(jev, llm):
    fake_jev, client = jev({"I was charged twice.": "billing"})
    llm(never)
    problem = OpenChoice(QUESTION, seed={"billing": "Charges and refunds."})

    answer = await problem.ask(client, "I was charged twice.")

    assert (answer.choice, answer.outcome, answer.probabilities["billing"]) == ("billing", "hit", 0.95)
    question = fake_jev.requests[0]["questions"]["answer"]
    assert question["instructions"] == QUESTION
    assert list(question["criteria"]) == ["billing", NONE_OF_THESE]


async def test_uncertain_jev_answer_lets_the_llm_reuse_an_option(jev, llm):
    _, client = jev({"Refund please": {"billing": 0.45, NONE_OF_THESE: 0.55}})
    llm(lambda state: reuse("billing"))
    problem = OpenChoice(QUESTION, seed={"billing": "Charges and refunds."})

    answer = await problem.ask(client, "Refund please")

    assert (answer.choice, answer.outcome) == ("billing", "reused")
    assert answer.probabilities == {"billing": 0.45, NONE_OF_THESE: 0.55}
    assert list(problem.options) == ["billing"]


@pytest.mark.parametrize(("hit_threshold", "outcome"), [(0.4, "hit"), (0.5, "reused")])
async def test_top_option_must_reach_the_hit_threshold(jev, llm, hit_threshold, outcome):
    _, client = jev({"Refund please": {"billing": 0.45, "shipping": 0.25, NONE_OF_THESE: 0.3}})
    llm(lambda state: reuse("billing"))
    problem = OpenChoice(QUESTION, hit_threshold=hit_threshold, seed={"billing": None, "shipping": None})

    assert (await problem.ask(client, "Refund please")).outcome == outcome


async def test_new_option_named_like_an_existing_one_reuses_it(jev, llm):
    _, client = jev({})
    llm(lambda state: create("Billing ", "Something else."))
    problem = OpenChoice(QUESTION, seed={"billing": "Charges and refunds."})

    answer = await problem.ask(client, "Refund please")

    assert (answer.choice, answer.outcome, answer.description) == ("billing", "reused", "Charges and refunds.")
    assert problem.options == {"billing": "Charges and refunds."}


async def test_least_recently_used_option_is_evicted_and_can_be_restored(jev, llm):
    _, client = jev({"a again": "a"})
    replies = {"new a": create("a"), "new b": create("b"), "new c": create("c"), "b again": reuse("b")}
    fake_llm = llm(replies.__getitem__)
    problem = OpenChoice(QUESTION, max_choices=2)

    await problem.ask(client, "new a")
    await problem.ask(client, "new b")
    assert (await problem.ask(client, "a again")).outcome == "hit"  # a is now more recent than b
    answer = await problem.ask(client, "new c")
    assert (answer.choice, answer.evicted) == ("c", "b")
    assert list(problem.options) == ["a", "c"]

    answer = await problem.ask(client, "b again")
    assert "Retired answers, which you can bring back:\n- b: d" in fake_llm.prompts[-1]
    assert (answer.choice, answer.outcome, answer.evicted) == ("b", "restored", "a")
    assert list(problem.options) == ["c", "b"]
    assert problem.stats == {"hit": 1, "reused": 0, "restored": 1, "created": 3, "evicted": 2}


async def test_seed_options_are_never_evicted(jev, llm):
    _, client = jev({})
    llm(lambda state: create(state))
    problem = OpenChoice(QUESTION, max_choices=2, seed={"billing": None})

    await problem.ask(client, "x")
    answer = await problem.ask(client, "y")

    assert answer.evicted == "x"
    assert list(problem.options) == ["billing", "y"]


async def test_options_and_stats_persist_in_the_store(tmp_path, jev, llm):
    path = tmp_path / "problem.json"
    _, client = jev({"charged twice again": "billing"})
    llm(lambda state: create("billing", "Charges and refunds."))
    await OpenChoice(QUESTION, store=path).ask(client, "charged twice")

    reloaded = OpenChoice(QUESTION, store=path)
    assert reloaded.options == {"billing": "Charges and refunds."}
    assert reloaded.stats["created"] == 1
    assert (await reloaded.ask(client, "charged twice again")).outcome == "hit"
    assert json.loads(path.read_text())["options"][0]["uses"] == 2


async def test_seed_overrides_the_store(tmp_path, jev, llm):
    path = tmp_path / "problem.json"
    _, client = jev({})
    llm(lambda state: create("shipping"))
    await OpenChoice(QUESTION, store=path, seed={"billing": "Old."}).ask(client, "late parcel")

    reloaded = OpenChoice(QUESTION, store=path, seed={"billing": "New.", "refunds": None})

    assert reloaded.options == {"billing": "New.", "shipping": "d", "refunds": None}


def test_store_saved_for_a_different_question_warns(tmp_path):
    path = tmp_path / "problem.json"
    JsonFileStore(path).save({"instructions": "Other?", "options": [], "retired": [], "stats": {}, "clock": 0})

    with pytest.warns(UserWarning, match="different question"):
        OpenChoice(QUESTION, store=path)


async def test_concurrent_misses_on_one_new_answer_make_one_option(jev, llm):
    states = [f"parcel lost #{i}" for i in range(5)]
    _, client = jev(dict.fromkeys(states, "lost_parcel"))
    fake_llm = llm(lambda state: create("lost_parcel"), delay=0.05)
    problem = OpenChoice(QUESTION)

    answers = await asyncio.gather(*(problem.ask(client, state) for state in states))

    assert {answer.choice for answer in answers} == {"lost_parcel"}
    assert sorted(answer.outcome for answer in answers) == ["created", "hit", "hit", "hit", "hit"]
    assert len(fake_llm.prompts) == 1


async def test_open_and_plain_questions_share_one_jev_request(jev, llm):
    fake_jev, client = jev({"charged twice": "billing"})
    llm(never)
    problem = OpenChoice(QUESTION, seed={"billing": None})

    result = await system_one(
        client, "charged twice", {"problem": problem, "urgent": Noul(instructions="Is it urgent?")}
    )

    assert len(fake_jev.requests) == 1
    assert set(fake_jev.requests[0]["questions"]) == {"problem", "urgent"}
    assert result.answers["problem"].choice == "billing"
    assert isinstance(result.answers["urgent"], NoulAnswer)
    assert result.response.model == "jev-fake"


async def test_malformed_llm_reply_is_retried_once(jev, llm):
    _, client = jev({})
    replies = iter(["not json", create("billing")])
    fake_llm = llm(lambda state: next(replies))

    answer = await OpenChoice(QUESTION).ask(client, "charged twice")

    assert answer.choice == "billing"
    assert fake_llm.prompts[1].startswith("That reply is invalid: the reply contains no JSON object")


async def test_llm_reusing_an_unknown_key_twice_raises(jev, llm):
    _, client = jev({})
    llm(lambda state: reuse("nonexistent"))

    with pytest.raises(ValueError, match="not one of the current or retired answers"):
        await OpenChoice(QUESTION).ask(client, "charged twice")


async def test_prompt_shows_question_guidance_options_and_state(jev, llm):
    _, client = jev({})
    fake_llm = llm(lambda state: create("greeting"))
    problem = OpenChoice(
        {"question": "What is the problem?"}, guidance="Use broad categories.", seed={"billing": "Charges."}
    )

    await problem.ask(client, {"message": "hello"})

    prompt = fake_llm.prompts[0]
    assert '"question": "What is the problem?"' in prompt
    assert "Guidance:\nUse broad categories." in prompt
    assert "Current answers:\n- billing: Charges." in prompt
    assert '"message": "hello"' in prompt


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_choices": 0},
        {"max_choices": 255},
        {"hit_threshold": 1.5},
        {"max_choices": 1, "seed": {"billing": None}},
        {"seed": {NONE_OF_THESE: None}},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        OpenChoice(QUESTION, **kwargs)


def test_works_across_event_loops(llm):
    llm(lambda state: create("lost_parcel"), delay=0.01)
    problem = OpenChoice(QUESTION)

    async def two_misses_at_once():
        async with FakeJev({}).client() as client:
            await asyncio.gather(problem.ask(client, "a"), problem.ask(client, "b"))

    asyncio.run(two_misses_at_once())
    asyncio.run(two_misses_at_once())
    assert problem.stats == {"hit": 0, "reused": 3, "restored": 0, "created": 1, "evicted": 0}
