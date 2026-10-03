"""Compile Pydantic fields into System One questions, and answers back into values."""

import enum
import math
import types
import typing
from dataclasses import dataclass, field as dataclass_field

from pydantic import Field
from pydantic.fields import FieldInfo

_ASK = "nimble_ask"


def ask(instructions=None, *, criteria=None, levels=None, yes=None, no=None, **field_kwargs):
    """Declare a judgment on a `Judgments` field.

    The annotation picks the primitive: `bool` is a Noul, a `Literal` or `Enum`
    is a Choice, and an `int` or `float` with `levels` is a Score. Instructions
    may also come from `Field(description=...)` or the attribute docstring.
    """
    if levels is not None and criteria is not None:
        raise ValueError("Use criteria for a Choice or levels for a Score, not both")
    spec = {"instructions": instructions, "criteria": criteria, "levels": levels, "yes": yes, "no": no}
    extra = field_kwargs.pop("json_schema_extra", None) or {}
    return Field(json_schema_extra={**extra, _ASK: spec}, **field_kwargs)


@dataclass
class Question:
    """One field's wire question and how to turn its answer into a Python value."""
    name: str
    kind: str  # "choice", "noul", or "score"
    instructions: object
    criteria: object = None
    values: dict = dataclass_field(default_factory=dict)  # wire key -> Python value
    expected: bool = False  # Score fields typed float return the expected level

    def wire(self):
        question = {"type": self.kind, "instructions": self.instructions}
        if self.criteria is not None:
            question["criteria"] = self.criteria
        return question


@dataclass
class Evidence:
    """The full distribution behind one typed value."""
    kind: str
    probabilities: dict  # wire key -> probability, in criteria order
    values: dict

    @property
    def top(self):
        key = max(self.probabilities, key=self.probabilities.get)
        return self.values.get(key, key), self.probabilities[key]

    @property
    def confidence(self):
        """1 - normalized entropy: concentration of the distribution, not accuracy."""
        ps = [p for p in self.probabilities.values() if p > 0]
        if len(self.probabilities) < 2:
            return 1.0
        return 1 - (-sum(p * math.log(p) for p in ps)) / math.log(len(self.probabilities))

    def probability(self, value):
        for key, candidate in self.values.items():
            if candidate == value or key == value:
                return self.probabilities[key]
        raise KeyError(f"{value!r} is not an allowed answer")


def _spec(info: FieldInfo):
    extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
    return extra.get(_ASK) or {}


def _strip(annotation):
    """Unwrap Annotated[...] so the base type decides the primitive."""
    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def compile_field(name, info: FieldInfo):
    spec = _spec(info)
    instructions = spec.get("instructions") or info.description
    if not instructions:
        raise TypeError(f"{name}: add instructions with ask(...), Field(description=...), or a docstring")
    annotation = _strip(info.annotation)
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        raise TypeError(f"{name}: Optional/Union fields are not judgments; add an explicit 'none' choice")

    if annotation is bool:
        criteria = {k: spec[k] for k in ("yes", "no") if spec.get(k)}
        return Question(name, "noul", instructions, criteria or None, {"false": False, "true": True})

    if annotation in (int, float):
        levels = spec.get("levels")
        if not levels or len(levels) < 2:
            raise TypeError(f"{name}: a Score needs ask(..., levels=[...]) with at least two levels")
        return Question(name, "score", instructions, list(levels),
                        {str(i): i for i in range(len(levels))}, expected=annotation is float)

    if origin is typing.Literal:
        options = typing.get_args(annotation)
        values = {str(o): o for o in options}
    elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = {str(m.value): m for m in annotation}
    else:
        raise TypeError(f"{name}: use bool, Literal[...], an Enum, or int/float with levels; got {annotation!r}")
    descriptions = spec.get("criteria") or {}
    unknown = set(descriptions) - set(values)
    if unknown:
        raise TypeError(f"{name}: criteria for unknown options {sorted(unknown)}")
    if spec.get("levels"):
        # An ordered Literal/Enum: ask a Score and map the most probable level back to the option.
        if len(spec["levels"]) != len(values):
            raise TypeError(f"{name}: levels must describe each option, in order")
        return Question(name, "score", instructions, list(spec["levels"]),
                        {str(i): v for i, v in enumerate(values.values())})
    return Question(name, "choice", instructions,
                    {k: descriptions.get(k) or k for k in values}, values)


def decode(question: Question, answer):
    """Turn one wire answer into (python value, Evidence)."""
    if answer.get("type") not in (None, question.kind):
        raise ValueError(f"{question.name}: expected a {question.kind} answer, got {answer.get('type')}")
    if question.kind == "noul":
        p = float(answer["noul"])
        probabilities = {"false": 1 - p, "true": p}
    else:
        raw = {str(k): float(v) for k, v in answer["probabilities"].items()}
        missing = set(question.values) - set(raw)
        if missing:
            raise ValueError(f"{question.name}: no probability for {sorted(missing)}")
        total = sum(raw[k] for k in question.values)
        probabilities = {k: raw[k] / total for k in question.values}
    evidence = Evidence(question.kind, probabilities, question.values)
    if question.kind == "score" and question.expected:
        value = answer.get("score")
        value = float(value) if value is not None else sum(int(k) * p for k, p in probabilities.items())
    elif question.kind == "noul":
        value = probabilities["true"] > 0.5
    elif question.kind == "choice" and answer.get("choice") in question.values:
        value = question.values[answer["choice"]]
    else:
        value = evidence.top[0]
    return value, evidence
