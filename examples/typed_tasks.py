"""Typed judgments with Pydantic: support triage, then a policy check that depends on it.

    python examples/typed_tasks.py                 # hosted Nimble (public endpoint)
    TYPESAFE_API_KEY=... python examples/typed_tasks.py --jev
"""

import argparse
from typing import Literal

from pydantic import BaseModel

from nimble.typed import Judgments, ask, configure, jev, nimble, task


# State: plain Pydantic models. Field names become the JSON keys the questions can reference.
class Ticket(BaseModel):
    customer: str
    message: str
    days_since_purchase: int


# Judgments: each field is one independent question; its type is the answer type.
class Triage(Judgments):
    wants_refund: bool
    """Does the customer explicitly ask for their money back in `ticket.message`?"""

    team: Literal["billing", "technical", "account"] = ask(
        "Which team should handle `ticket.message`?",
        criteria={"billing": "Charges, invoices, refunds",
                  "technical": "Bugs, errors, outages",
                  "account": "Login, profile, plan changes"})

    urgency: float = ask(
        "How operationally urgent is `ticket.message`?",
        levels=["Routine question; nothing is broken",
                "Something is degraded but there is a workaround",
                "The customer is fully blocked"])


class RefundDecision(Judgments):
    eligible: bool = ask("Under `policy`, is the refund in `request` eligible?",
                         yes="The policy allows this refund", no="The policy forbids or does not cover it")


POLICY = "Refunds are allowed within 30 days of purchase, or at any time for a duplicate charge."


@task
def triage(ticket: Ticket) -> Triage:
    """Route an incoming support ticket."""


@task
def refund_check(ticket: Ticket) -> RefundDecision:
    """Apply the refund policy. The body shapes the state explicitly."""
    return {"policy": POLICY, "request": ticket}


def handle(ticket: Ticket):
    """Code owns the workflow; judgments supply the semantic parts."""
    t = triage(ticket)
    action = {"team": t.team, "urgency": round(t.urgency, 2), "needs_human": t.uncertain(0.75)}
    if t.wants_refund:
        decision = refund_check(ticket)
        action["refund"] = "approve" if decision.probability("eligible") > 0.8 else "review"
    return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jev", action="store_true")
    configure(jev() if parser.parse_args().jev else nimble())

    tickets = [
        Ticket(customer="Ada", message="I was charged twice this month. Please refund one of them.",
               days_since_purchase=45),
        Ticket(customer="Lin", message="The export button throws a 500 error and I can't send reports.",
               days_since_purchase=3),
    ]
    print(triage.request(tickets[0]))
    for ticket in tickets:
        print(ticket.customer, handle(ticket))
    # Batch: one request per ticket, run concurrently, results in input order.
    for ticket, result in zip(tickets, triage.map(tickets)):
        print(ticket.customer, result.team, result.distribution("team"))
