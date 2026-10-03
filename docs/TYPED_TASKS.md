# Typed tasks with Pydantic (prototype)

`nimble.typed` lets you write judgments as Pydantic models and tasks as typed
Python functions. The same code runs on hosted Nimble, TypeSafe's Jev, or a
local Nimble scorer, because all three use the `/v1/systemone` question format.

Instructor-style libraries ask an LLM to generate JSON and then validate it.
System One models already return typed answers, so this layer doesn't parse
or retry anything. It turns field types into questions and turns the answer
probabilities back into typed values. The distribution behind each value is
kept, so your code can still use it.

## Judgments: the type is the question

```python
from typing import Literal
from nimble.typed import Judgments, ask

class Triage(Judgments):
    wants_refund: bool
    """Does the customer explicitly ask for their money back in `ticket.message`?"""

    team: Literal["billing", "technical", "account"] = ask(
        "Which team should handle `ticket.message`?",
        criteria={"billing": "Charges, invoices, refunds", "technical": "Bugs, errors, outages",
                  "account": "Login, profile, plan changes"})

    urgency: float = ask("How operationally urgent is `ticket.message`?",
                         levels=["Routine question", "Degraded with a workaround", "Fully blocked"])
```

| Annotation | Primitive | Value you get |
| --- | --- | --- |
| `bool` | Noul | `True` when P(yes) > 0.5 |
| `Literal[...]` or `Enum` | Choice | The most probable option (Enum member for an Enum) |
| `float` with `levels=` | Score | Expected level, from 0 to n-1 |
| `int` with `levels=` | Score | Index of the most probable level |
| `Literal[...]` with `levels=` | Score | The option at the most probable level (an ordered label) |

You can give the instructions in `ask("...")`, in `Field(description=...)`, or
in an attribute docstring. A field with no instructions raises an error when
the class is defined. So does an unsupported type such as `Optional`: add an
explicit "none" choice instead.

## Tasks: the arguments are the state

```python
from pydantic import BaseModel
from nimble.typed import task, configure, nimble

class Ticket(BaseModel):
    customer: str
    message: str

@task
def triage(ticket: Ticket) -> Triage:
    """Route an incoming support ticket."""

configure(nimble())          # or jev(), or LocalScorer(scorer)
t = triage(Ticket(customer="Ada", message="I was charged twice"))
t.team                       # "billing", typed as the Literal
t.probability("wants_refund")
t.distribution("team")       # {"billing": 0.91, "technical": 0.06, "account": 0.03}
t.uncertain(0.75)            # fields to escalate
```

- The arguments are sent as named JSON state, such as `{"ticket": {...}}`.
  Questions can refer to parts of it with backticked paths such as
  `ticket.message`. A single `str` argument is sent as the raw text.
- To control the state yourself, return it from the function body, for
  example `return {"policy": POLICY, "request": ticket}`.
- `triage.request(...)` shows the exact payload without making a call.
  `triage.map(items, concurrency=8)` runs a batch, and `await triage.acall(...)`
  runs one call asynchronously.
- `triage.using(jev())` runs the same task on another backend, which makes it
  easy to compare Jev and Nimble.
- All the questions in one model are sent in one request and answered
  independently. When a judgment depends on an earlier answer, write a second
  task and connect the two in ordinary Python. See
  [examples/typed_tasks.py](../examples/typed_tasks.py).

## Backends

| Backend | Use |
| --- | --- |
| `nimble(url=None)` | Hosted Nimble (`NIMBLE_URL` or the public Modal endpoint) |
| `jev(model="jev-1.13.0")` | TypeSafe's API, with `TYPESAFE_API_KEY` |
| `LocalScorer(scorer)` | A loaded `ParallelScorer` or `CudaCandidateScorer`. Score levels become the enum codes `"0"`..`"n-1"`, as in `nimble.serving.compiler` |
| `Scripted(fn)` | Fixed answers for tests |

```sh
python3 -m unittest tests.test_typed
python3 -m examples.typed_tasks          # needs network access to the endpoint
```
