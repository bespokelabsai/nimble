"""Typed judgments with Pydantic: define the answer type, get calibrated typed answers.

    from typing import Literal
    from nimble.typed import Judgments, ask, task, configure, nimble

    class Triage(Judgments):
        refund: bool
        '''Does the customer explicitly ask for their money back?'''
        team: Literal["billing", "technical"] = ask("Which team should handle this ticket?")

    @task
    def triage(ticket: str) -> Triage:
        '''Route an incoming support ticket.'''

    configure(nimble())                   # or jev(), or LocalScorer(scorer)
    result = triage("I was charged twice, please refund me.")
    result.team, result.probability("refund")
"""

from .backends import LocalScorer, Scripted, SystemOneHTTP, jev, nimble
from .core import Judgments, Task, configure, task
from .questions import Evidence, ask

__all__ = ["Judgments", "Task", "task", "ask", "configure", "Evidence",
           "jev", "nimble", "LocalScorer", "SystemOneHTTP", "Scripted"]
