"""Compile plain Pydantic fields into System One questions, and answers back into values."""

import enum
import math
import types
import typing
from dataclasses import dataclass, field as dataclass_field

from pydantic import BaseModel
from pydantic.fields import FieldInfo


class Criteria(dict):
    """`Annotated` metadata describing what each answer means.

        team: Annotated[Literal["billing", "technical"],
                        Criteria(billing="Charges and refunds", technical="Bugs and outages")]
        refund: Annotated[bool, Criteria(yes="They ask for money back", no="They do not")]
    """

    def __init__(self, mapping=None, /, **descriptions):
        super().__init__({**(mapping or {}), **descriptions})


class Levels(tuple):
    """`Annotated` metadata for an ordered rubric, lowest level first; makes the field a Score.

        urgency: Annotated[float, Levels("Routine", "Degraded", "Blocked")]
    """

    def __new__(cls, *levels):
        if len(levels) < 2:
            raise ValueError("A Score needs at least two levels")
        return super().__new__(cls, levels)


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
        if len(self.probabilities) < 2:
            return 1.0
        entropy = -sum(p * math.log(p) for p in self.probabilities.values() if p > 0)
        return 1 - entropy / math.log(len(self.probabilities))

    def probability(self, value):
        for key, candidate in self.values.items():
            if candidate == value or key == value:
                return self.probabilities[key]
        raise KeyError(f"{value!r} is not an allowed answer")


def _metadata(info: FieldInfo):
    """Criteria/Levels from Annotated metadata, or from json_schema_extra as a fallback."""
    criteria = next((m for m in info.metadata if isinstance(m, Criteria)), None)
    levels = next((m for m in info.metadata if isinstance(m, Levels)), None)
    extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
    criteria = criteria if criteria is not None else extra.get("criteria")
    levels = levels if levels is not None else extra.get("levels")
    return dict(criteria or {}), list(levels) if levels else None


def compile_field(name, info: FieldInfo):
    criteria, levels = _metadata(info)
    instructions = info.description
    if not instructions:
        raise TypeError(f"{name}: add Field(description=...) or an attribute docstring "
                        "with model_config = ConfigDict(use_attribute_docstrings=True)")
    annotation = info.annotation
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        raise TypeError(f"{name}: Optional/Union fields are not judgments; add an explicit 'none' choice")

    if annotation is bool:
        unknown = set(criteria) - {"yes", "no"}
        if unknown:
            raise TypeError(f"{name}: a bool takes Criteria(yes=..., no=...), not {sorted(unknown)}")
        return Question(name, "noul", instructions, criteria or None, {"false": False, "true": True})

    if annotation in (int, float):
        if not levels:
            raise TypeError(f"{name}: a numeric judgment is a Score; annotate it with Levels(...)")
        return Question(name, "score", instructions, levels,
                        {str(i): i for i in range(len(levels))}, expected=annotation is float)

    if origin is typing.Literal:
        values = {str(o): o for o in typing.get_args(annotation)}
    elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = {str(m.value): m for m in annotation}
    else:
        raise TypeError(f"{name}: use bool, Literal[...], an Enum, or int/float with Levels; got {annotation!r}")
    unknown = set(criteria) - set(values)
    if unknown:
        raise TypeError(f"{name}: criteria for unknown options {sorted(unknown)}")
    if levels:
        # An ordered Literal/Enum: ask a Score and map the most probable level back to the option.
        if len(levels) != len(values):
            raise TypeError(f"{name}: Levels must describe each option, in order")
        return Question(name, "score", instructions, levels,
                        {str(i): v for i, v in enumerate(values.values())})
    return Question(name, "choice", instructions, {k: criteria.get(k) or k for k in values}, values)


_compiled = {}


def questions_for(model: type[BaseModel]):
    """Every field of `model` as a Question; compiled once per class."""
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError(f"Expected a Pydantic model class, got {model!r}")
    if model not in _compiled:
        if not model.model_fields:
            raise TypeError(f"{model.__name__} has no fields to judge")
        _compiled[model] = {name: compile_field(name, info) for name, info in model.model_fields.items()}
    return _compiled[model]


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
