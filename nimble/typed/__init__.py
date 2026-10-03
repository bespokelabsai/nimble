"""Typed judgments from plain Pydantic models.

    from typing import Annotated, Literal
    from pydantic import BaseModel, Field
    from nimble.typed import Criteria, Levels, configure, evidence, nimble, task

    class Triage(BaseModel):
        refund: bool = Field(description="Does the customer explicitly ask for their money back?")
        team: Annotated[Literal["billing", "technical"],
                        Criteria(billing="Charges and refunds", technical="Bugs and outages")] = Field(
            description="Which team should handle this ticket?")
        urgency: Annotated[float, Levels("Routine", "Degraded", "Blocked")] = Field(
            description="How urgent is this ticket?")

    @task
    def triage(ticket: str) -> Triage:
        '''Route an incoming support ticket.'''

    configure(nimble())                   # or jev(), or LocalScorer(scorer)
    result = triage("I was charged twice, please refund me.")   # a plain Triage
    result.team, evidence(result).probability("refund")
"""

from .backends import LocalScorer, Scripted, SystemOneHTTP, jev, nimble
from .core import Judgment, Task, ajudge, configure, evidence, from_answers, judge, questions, task
from .questions import Criteria, Evidence, Levels

__all__ = ["task", "Task", "judge", "ajudge", "questions", "from_answers", "evidence", "Judgment",
           "Criteria", "Levels", "Evidence", "configure",
           "jev", "nimble", "LocalScorer", "SystemOneHTTP", "Scripted"]
