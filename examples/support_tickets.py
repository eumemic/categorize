"""Sort support tickets by problem, with no list of problems written in advance.

Needs TYPESAFE_API_KEY, plus the key for the LLM (ANTHROPIC_API_KEY for the default model).
Options are saved next to this file, so a second run starts with the options from the first.

    uv run examples/support_tickets.py
"""

import asyncio
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient

from categorize import OpenChoice

TICKETS = [
    "I was charged twice for order A-104. Please refund one.",
    "My parcel was supposed to arrive Monday and it's now Thursday. Where is it?",
    "You billed my card two times for the same subscription this month!",
    "I can't log in, the password reset email never arrives.",
    "Tracking says delivered but there's nothing at my door.",
    "Still no sign of my order, it's a week late now.",
    "The app keeps logging me out every few minutes.",
    "How do I change the shipping address on an order I already placed?",
    "Why is there a $4.99 fee on my invoice I didn't agree to?",
    "Reset link expired before I could use it, and now I'm locked out.",
    "Same charge appears twice on my statement for one purchase.",
    "Can I update the delivery address? I just moved.",
]

problem = OpenChoice(
    "What is the customer's underlying problem?",
    store=Path(__file__).with_name("support_problems.json"),
    guidance="Name problem types that many tickets could share, such as double_charge or login_failure.",
)


async def main() -> None:
    async with AsyncTypeSafeClient() as client:
        for ticket in TICKETS:
            answer = await problem.ask(client, ticket)
            p = answer.probabilities.get(answer.choice)
            jev = f"{p:.2f}" if p is not None else "  - "
            print(f"{answer.outcome:8} {jev}  {answer.choice:28} {ticket}")
    print("\nOptions:")
    for key, description in problem.options.items():
        print(f"  {key}: {description}")
    print("\nStats:", problem.stats)


asyncio.run(main())
