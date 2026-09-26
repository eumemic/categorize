"""OpenChoice: a jev Choice question whose options an LLM writes as they are needed."""

from __future__ import annotations

import asyncio
import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

from typesafe_sdk import Answer, AsyncTypeSafeClient, Choice, ChoiceAnswer, JSONContent, Question, SystemOneResponse

from categorize.propose import DEFAULT_LLM, NONE_OF_THESE, Proposal, normalize_key, propose
from categorize.store import JsonFileStore, Store

MAX_CHOICES = 254  # jev accepts up to 255 options per Choice, and one is none_of_these
_NONE_DESCRIPTION = "None of the other options correctly answers the question."

Outcome = Literal["hit", "reused", "restored", "created"]


@dataclass
class _Option:
    key: str
    description: JSONContent | None = None
    pinned: bool = False
    uses: int = 0
    last_used: int = 0


@dataclass(frozen=True)
class OpenAnswer:
    """The answer to an OpenChoice question."""

    choice: str
    """The chosen option's key."""
    description: JSONContent | None
    """The chosen option's description, which is what jev matches against."""
    outcome: Outcome
    """`hit`: jev chose an existing option. Otherwise the LLM was asked, and it `reused` an
    existing option, `restored` an evicted one, or `created` a new one."""
    probabilities: dict[str, float] = field(default_factory=dict)
    """jev's probabilities over the options and `none_of_these`; empty when jev was not asked
    because there were no options yet."""
    confidence: float | None = None
    """jev's confidence, or None when jev was not asked."""
    evicted: str | None = None
    """The key of the option evicted to make room for this one, if any."""

    @property
    def is_new(self) -> bool:
        return self.outcome == "created"


@dataclass(frozen=True)
class Result:
    """The answers to a `system_one` request."""

    answers: dict[str, Answer | OpenAnswer]
    """One answer per question, under the question's key."""
    response: SystemOneResponse | None
    """jev's response to the combined request, or None when no question needed jev."""


class OpenChoice:
    """A jev Choice question that starts without options and grows them.

    Each call shows jev the options so far, plus `none_of_these`. If jev puts at least
    `hit_threshold` probability on an option, and more than on `none_of_these`, that option is
    the answer. Otherwise the LLM sees the question, the state, and the options, and either picks
    one of them or writes a new one, which jev is shown from then on. Past `max_choices`, the
    least recently used option outside `seed` is evicted; the LLM can bring it back later.

    Misses are handled one at a time per OpenChoice, so concurrent calls that meet the same new
    kind of answer create one option, not one each. Not safe to share across threads or
    processes.
    """

    def __init__(
        self,
        instructions: JSONContent,
        *,
        store: Store | str | os.PathLike[str] | None = None,
        max_choices: int = 64,
        hit_threshold: float = 0.5,
        llm: str = DEFAULT_LLM,
        llm_kwargs: Mapping[str, Any] | None = None,
        guidance: str | None = None,
        seed: Mapping[str, JSONContent | None] | None = None,
    ) -> None:
        """
        Args:
            instructions: The question, as for a jev Choice.
            store: Where options persist: a JSON file path or a `Store`. None keeps them in
                memory only.
            max_choices: Most options shown to jev at once, from 1 to 254.
            hit_threshold: Least probability jev must put on an option for it to be used without
                asking the LLM.
            llm: LiteLLM model that writes options, such as "anthropic/claude-haiku-4-5".
            llm_kwargs: Extra arguments for `litellm.acompletion`.
            guidance: Instructions for the LLM only, such as how broad options should be.
            seed: Options to start with, key to description. They are never evicted.
        """
        seed = dict(seed or {})
        if not 1 <= max_choices <= MAX_CHOICES:
            raise ValueError(f"max_choices must be between 1 and {MAX_CHOICES}")
        if not 0 <= hit_threshold <= 1:
            raise ValueError("hit_threshold must be between 0 and 1")
        if len(seed) >= max_choices:
            raise ValueError("max_choices must be larger than the number of seed options")
        if NONE_OF_THESE in seed:
            raise ValueError(f"{NONE_OF_THESE!r} is a reserved key")
        self.instructions = instructions
        self.max_choices = max_choices
        self.hit_threshold = hit_threshold
        self.llm = llm
        self.llm_kwargs = dict(llm_kwargs or {})
        self.guidance = guidance
        self._store = JsonFileStore(store) if isinstance(store, (str, os.PathLike)) else store
        self._options: dict[str, _Option] = {}  # oldest first
        self._retired: dict[str, _Option] = {}  # evicted options, oldest first
        self._stats = {"hit": 0, "reused": 0, "restored": 0, "created": 0, "evicted": 0}
        self._clock = 0  # counts uses; an option's last_used is the clock at its latest use
        self._version = 0  # counts changes to the current options
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._load(seed)

    @property
    def options(self) -> dict[str, JSONContent | None]:
        """The current options, key to description, oldest first."""
        return {key: option.description for key, option in self._options.items()}

    @property
    def stats(self) -> dict[str, int]:
        """How many answers had each outcome, and how many options were evicted."""
        return dict(self._stats)

    def __repr__(self) -> str:
        return f"OpenChoice({self.instructions!r}, options={len(self._options)})"

    async def ask(self, client: AsyncTypeSafeClient, state: JSONContent, **kwargs: Any) -> OpenAnswer:
        """Answer the question for `state`. Keyword arguments go to `client.system_one`."""
        result = await system_one(client, state, {"answer": self}, **kwargs)
        return cast(OpenAnswer, result.answers["answer"])

    def _question(self) -> Choice | None:
        """The jev question for the current options, or None when there are none yet."""
        if not self._options:
            return None
        criteria = {key: option.description for key, option in self._options.items()}
        return Choice(instructions=self.instructions, criteria={**criteria, NONE_OF_THESE: _NONE_DESCRIPTION})

    async def _resolve(
        self,
        client: AsyncTypeSafeClient,
        state: JSONContent,
        jev_kwargs: Mapping[str, Any],
        version: int,
        jev: ChoiceAnswer | None,
    ) -> OpenAnswer:
        """Turn jev's answer, given over the options as of `version`, into an OpenAnswer."""
        key = self._hit(jev)
        if key is not None:
            return self._use(key, "hit", jev)
        async with self._get_lock():
            question = self._question()
            if self._version != version and question is not None:
                # The options changed while this call waited, and a new one may fit.
                response = await client.system_one(state, {"answer": question}, **jev_kwargs)
                jev = cast(ChoiceAnswer, response.answers["answer"])
                key = self._hit(jev)
                if key is not None:
                    return self._use(key, "hit", jev)
            return self._apply(await self._propose(state), jev)

    async def _propose(self, state: JSONContent) -> Proposal:
        """Ask the LLM to pick an option for `state` or write a new one."""
        retired = {key: option.description for key, option in self._retired.items()}
        return await propose(self.llm, self.llm_kwargs, self.instructions, state, self.options, retired, self.guidance)

    def _hit(self, jev: ChoiceAnswer | None) -> str | None:
        """The current option jev favors enough to use without the LLM, if any."""
        if jev is None:
            return None
        # Options evicted since the request was built no longer count.
        current = [(p, key) for key, p in jev.probabilities.items() if key in self._options]
        if not current:
            return None
        p, key = max(current)
        if p >= self.hit_threshold and p > jev.probabilities.get(NONE_OF_THESE, 0.0):
            return key
        return None

    def _apply(self, proposal: Proposal, jev: ChoiceAnswer | None) -> OpenAnswer:
        # Keys compare normalized, so a "new" option that repeats an existing name reuses it.
        current = {normalize_key(key): key for key in self._options}
        retired = {normalize_key(key): key for key in self._retired}
        if proposal.key in current:
            return self._use(current[proposal.key], "reused", jev)
        if proposal.key in retired:
            option = self._retired.pop(retired[proposal.key])
            return self._use(option.key, "restored", jev, evicted=self._add(option))
        # Examples go in the description jev sees, since jev matches better with example inputs.
        description: JSONContent | None = proposal.description or None
        if proposal.examples:
            description = {"what": proposal.description, "examples": list(proposal.examples)}
        evicted = self._add(_Option(proposal.key, description))
        return self._use(proposal.key, "created", jev, evicted=evicted)

    def _use(self, key: str, outcome: Outcome, jev: ChoiceAnswer | None, evicted: str | None = None) -> OpenAnswer:
        option = self._options[key]
        self._clock += 1
        option.uses += 1
        option.last_used = self._clock
        self._stats[outcome] += 1
        self._save()
        return OpenAnswer(
            choice=key,
            description=option.description,
            outcome=outcome,
            probabilities=dict(jev.probabilities) if jev else {},
            confidence=jev.confidence if jev else None,
            evicted=evicted,
        )

    def _add(self, option: _Option) -> str | None:
        """Make `option` current, evicting the least recently used option if full."""
        evicted = self._evict() if len(self._options) >= self.max_choices else None
        self._options[option.key] = option
        self._version += 1
        return evicted

    def _evict(self) -> str:
        option = min((o for o in self._options.values() if not o.pinned), key=lambda o: o.last_used)
        del self._options[option.key]
        self._retired[option.key] = option
        if len(self._retired) > self.max_choices:
            del self._retired[next(iter(self._retired))]
        self._stats["evicted"] += 1
        self._version += 1
        return option.key

    def _get_lock(self) -> asyncio.Lock:
        # An asyncio.Lock that has had a waiter belongs to that event loop, and an OpenChoice can
        # outlive one asyncio.run(), so keep one lock per loop.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    def _load(self, seed: dict[str, JSONContent | None]) -> None:
        saved = self._store.load() if self._store is not None else None
        if saved is not None:
            if saved["instructions"] != json.loads(json.dumps(self.instructions)):
                warnings.warn(
                    f"{self._store!r} holds options for a different question: {saved['instructions']!r}",
                    stacklevel=3,
                )
            self._options = {o["key"]: _Option(**o) for o in saved["options"]}
            self._retired = {o["key"]: _Option(**o) for o in saved["retired"]}
            self._stats.update(saved["stats"])
            self._clock = saved["clock"]
        # The seed in code overrides the store: seed options are pinned, with the descriptions
        # given here, and saved options that have left the seed become ordinary ones.
        for option in self._options.values():
            option.pinned = option.key in seed
        for key, description in seed.items():
            option = self._options.get(key) or self._retired.pop(key, None) or _Option(key)
            option.description, option.pinned = description, True
            self._options[key] = option
        while len(self._options) > self.max_choices:
            self._evict()

    def _save(self) -> None:
        if self._store is not None:
            self._store.save(
                {
                    "instructions": self.instructions,
                    "options": [asdict(o) for o in self._options.values()],
                    "retired": [asdict(o) for o in self._retired.values()],
                    "stats": self._stats,
                    "clock": self._clock,
                }
            )


async def system_one(
    client: AsyncTypeSafeClient,
    state: JSONContent,
    questions: Mapping[str, Question | OpenChoice],
    **kwargs: Any,
) -> Result:
    """`client.system_one`, except that `questions` may include OpenChoice questions.

    Every question goes to jev in one request, then each OpenChoice that missed asks its LLM,
    concurrently. Keyword arguments (model, timeout, ...) go to `client.system_one`.
    """
    if not questions:
        raise ValueError("At least one question is required.")
    open_choices = {key: q for key, q in questions.items() if isinstance(q, OpenChoice)}
    jev_questions = {key: q for key, q in questions.items() if not isinstance(q, OpenChoice)}
    versions = {}
    for key, open_choice in open_choices.items():
        versions[key] = open_choice._version
        question = open_choice._question()
        if question is not None:
            jev_questions[key] = question
    response = await client.system_one(state, jev_questions, **kwargs) if jev_questions else None
    answers: dict[str, Answer | OpenAnswer] = dict(response.answers) if response else {}
    resolved = await asyncio.gather(
        *(
            open_choice._resolve(client, state, kwargs, versions[key], cast("ChoiceAnswer | None", answers.get(key)))
            for key, open_choice in open_choices.items()
        )
    )
    answers.update(zip(open_choices, resolved))
    return Result(answers, response)
