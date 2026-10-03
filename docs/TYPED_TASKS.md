# Typed tasks with Pydantic (prototype)

`nimble.typed` lets you write judgments as Pydantic models and tasks as typed
Python functions. The same code runs on hosted Nimble, TypeSafe's Jev, or a
local Nimble scorer, because all three use the `/v1/systemone` question format.

Instructor-style libraries ask an LLM to generate JSON and then validate it.
System One models already return typed answers, so this layer doesn't parse
or retry anything. It turns field types into questions and turns the answer
probabilities back into typed values. The distribution behind each value is
kept, so your code can still use it.

## Judgments: an ordinary Pydantic model

There is no base class to inherit from. Any `BaseModel` works: each field is one
question, the field type is the answer type, and the field description is the
instruction. `Annotated` metadata says what each answer means.

```python
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict
from nimble.typed import Criteria, Levels

class Triage(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    wants_refund: bool
    """Does the customer explicitly ask for their money back in `ticket.message`?"""

    team: Annotated[Literal["billing", "technical", "account"], Criteria(
        billing="Charges, invoices, refunds",
        technical="Bugs, errors, outages",
        account="Login, profile, plan changes")]
    """Which team should handle `ticket.message`?"""

    urgency: Annotated[float, Levels("Routine question", "Degraded with a workaround", "Fully blocked")]
    """How operationally urgent is `ticket.message`?"""
```

| Annotation | Primitive | Value you get |
| --- | --- | --- |
| `bool` | Noul | `True` when P(yes) > 0.5 |
| `Literal[...]` or `Enum` | Choice | The most probable option (Enum member for an Enum) |
| `Annotated[float, Levels(...)]` | Score | Expected level, from 0 to n-1 |
| `Annotated[int, Levels(...)]` | Score | Index of the most probable level |
| `Annotated[Literal[...], Levels(...)]` | Score | The option at the most probable level (an ordered label) |

- The instruction comes from `Field(description=...)` or from an attribute
  docstring when the model sets `use_attribute_docstrings=True`.
- `Criteria(...)` describes the options of a Choice. On a `bool`, use
  `Criteria(yes=..., no=...)`. Options without a description are sent as
  their names.
- If you'd rather keep the type plain, you can write
  `Field(json_schema_extra={"criteria": {...}})` or `{"levels": [...]}`
  instead of the `Annotated` metadata.
- A model with an unsupported field fails when a task is defined, for
  example a field with no description, an `Optional` field (add an explicit
  "none" choice instead), or a number without `Levels`.
- The answers go through `model_validate`, so the model's own validators
  still run.

## Tasks: the arguments are the state

```python
from pydantic import BaseModel
from nimble.typed import configure, evidence, nimble, task

class Ticket(BaseModel):
    customer: str
    message: str

@task
def triage(ticket: Ticket) -> Triage:
    """Route an incoming support ticket."""

configure(nimble())          # or jev(), or LocalScorer(scorer)
t = triage(Ticket(customer="Ada", message="I was charged twice"))
t                            # a plain Triage instance
t.team                       # "billing", typed as the Literal
e = evidence(t)              # the probabilities behind it
e.probability("wants_refund")
e.distribution("team")       # {"billing": 0.91, "technical": 0.06, "account": 0.03}
e.uncertain(0.75)            # fields to escalate
```

- The arguments are sent as named JSON state, such as `{"ticket": {...}}`.
  Questions can refer to parts of it with backticked paths such as
  `ticket.message`. A single `str` argument is sent as the raw text.
- To control the state yourself, return it from the function body, for
  example `return {"policy": POLICY, "request": ticket}`.
- Outside a task, `judge(Triage, state)` asks the same questions about any
  state, and `questions(Triage)` shows the questions that will be sent.
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
