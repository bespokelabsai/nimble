"""Pydantic models as judgments, Python functions as tasks."""

import asyncio
import functools
import inspect
import os
import threading
import typing

import httpx
from pydantic import BaseModel, ConfigDict, PrivateAttr
from pydantic_core import to_jsonable_python

from . import backends as _backends
from .questions import compile_field, decode

_default_backend = None


def configure(backend):
    """Set the backend used by tasks and judgments that do not name one."""
    global _default_backend
    _default_backend = backend
    return backend


def default_backend():
    if _default_backend is not None:
        return _default_backend
    if os.environ.get("NIMBLE_URL"):
        return configure(_backends.nimble())
    if os.environ.get("TYPESAFE_API_KEY"):
        return configure(_backends.jev())
    raise RuntimeError("No backend: call nimble.typed.configure(nimble.typed.nimble()) or "
                       "configure(nimble.typed.jev()), or set NIMBLE_URL / TYPESAFE_API_KEY")


def run_sync(coroutine):
    """asyncio.run that also works inside a running loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    box = {}
    def target():
        try:
            box["value"] = asyncio.run(coroutine)
        except BaseException as exc:  # Re-raised in the caller's thread.
            box["error"] = exc
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class Judgments(BaseModel):
    """Subclass with typed fields; each field is one independent question.

        class Triage(Judgments):
            refund: bool
            '''Does the customer explicitly ask for their money back?'''

            team: Literal["billing", "technical"] = ask(
                "Which team should handle this ticket?",
                criteria={"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"})

            urgency: float = ask("How urgent is this ticket operationally?",
                                 levels=["Routine; nothing is broken", "Degraded", "Complete outage"])

    The instance holds the typed answers; `.probability()`, `.confidence()` and
    `.evidence()` expose the distribution behind each one.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)
    __questions__: typing.ClassVar[dict] = {}
    _evidence: dict = PrivateAttr(default_factory=dict)
    _state: object = PrivateAttr(default=None)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        cls.__questions__ = {name: compile_field(name, info) for name, info in cls.model_fields.items()}

    @classmethod
    def questions(cls, only=None):
        """The wire questions, ready for any `/v1/systemone` request."""
        names = only or cls.__questions__
        return {name: cls.__questions__[name].wire() for name in names}

    @classmethod
    def from_answers(cls, answers, state=None):
        values, evidence = {}, {}
        for name, question in cls.__questions__.items():
            values[name], evidence[name] = decode(question, answers[name])
        instance = cls(**values)
        instance._evidence, instance._state = evidence, state
        return instance

    @classmethod
    async def ajudge(cls, state, backend=None, client=None):
        state = to_jsonable_python(state)
        answers = await (backend or default_backend()).evaluate(state, cls.questions(), client=client)
        return cls.from_answers(answers, state)

    @classmethod
    def judge(cls, state, backend=None):
        """Ask every question about `state` (a string, dict, or Pydantic model) in one request."""
        return run_sync(cls.ajudge(state, backend))

    def evidence(self, field):
        return self._evidence[field]

    def probability(self, field, value=True):
        """Probability of one allowed answer; for a bool field, the probability of True."""
        return self._evidence[field].probability(value)

    def distribution(self, field):
        e = self._evidence[field]
        return {e.values[k]: p for k, p in e.probabilities.items()}

    def confidence(self, field):
        """Distribution concentration in [0, 1]; not a calibrated accuracy."""
        return self._evidence[field].confidence

    def uncertain(self, threshold=0.8):
        """Fields whose most probable answer is below `threshold`, for escalation."""
        return [name for name, e in self._evidence.items() if e.top[1] < threshold]

    @property
    def state(self):
        return self._state


Output = typing.TypeVar("Output", bound=Judgments)


class Task(typing.Generic[Output]):
    """A function whose arguments are the state and whose return type is the judgments.

    Built with `@task`. The body is optional: return a value to shape the state
    explicitly; return nothing to send the arguments as named JSON fields.
    """

    def __init__(self, fn, output: type[Output], backend=None):
        self.fn = fn
        self.output = output
        self.backend = backend
        self.signature = inspect.signature(fn)
        functools.update_wrapper(self, fn)

    def __repr__(self):
        return f"<task {self.fn.__name__} -> {self.output.__name__}>"

    def using(self, backend):
        """The same task on another backend, e.g. to compare Jev and Nimble."""
        return Task(self.fn, self.output, backend)

    def state(self, *args, **kwargs):
        """The exact JSON state that will be sent; useful for debugging prompts."""
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        shaped = self.fn(*bound.args, **bound.kwargs)
        if shaped is not None:
            return to_jsonable_python(shaped)
        if len(bound.arguments) == 1:
            (only,) = bound.arguments.values()
            if isinstance(only, str):
                return only
        return to_jsonable_python(dict(bound.arguments))

    def request(self, *args, **kwargs):
        return {"state": self.state(*args, **kwargs), "questions": self.output.questions()}

    async def acall(self, *args, client=None, **kwargs) -> Output:
        return await self.output.ajudge(self.state(*args, **kwargs), self.backend, client=client)

    def __call__(self, *args, **kwargs) -> Output:
        return run_sync(self.acall(*args, **kwargs))

    async def amap(self, items, concurrency=8, return_exceptions=False):
        """Run over many inputs with bounded concurrency; results keep input order.

        Each item is one argument, a tuple of arguments, or a dict of keyword arguments.
        """
        limit = asyncio.Semaphore(concurrency)
        single = len(self.signature.parameters) == 1
        async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=concurrency)) as client:
            async def one(item):
                async with limit:
                    if isinstance(item, dict) and not single:
                        return await self.acall(client=client, **item)
                    if isinstance(item, tuple) and not single:
                        return await self.acall(*item, client=client)
                    return await self.acall(item, client=client)
            return await asyncio.gather(*(one(i) for i in items), return_exceptions=return_exceptions)

    def map(self, items, concurrency=8, return_exceptions=False) -> list[Output]:
        return run_sync(self.amap(list(items), concurrency, return_exceptions))


def task(fn=None, *, backend=None):
    """Turn a typed function into a judgment task.

        @task
        def triage(ticket: Ticket) -> Triage:
            '''Route an incoming support ticket.'''

        result = triage(Ticket(...))   # -> Triage, fully typed
    """
    def wrap(fn):
        output = typing.get_type_hints(fn).get("return")
        if not (isinstance(output, type) and issubclass(output, Judgments)):
            raise TypeError(f"{fn.__name__} must be annotated to return a Judgments subclass")
        return Task(fn, output, backend)
    return wrap(fn) if fn is not None else wrap
