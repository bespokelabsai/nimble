"""The Pydantic task interface, without a model or network."""

import asyncio
import enum
import json
import unittest
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from nimble.typed import Judgments, LocalScorer, Scripted, SystemOneHTTP, ask, task


class Ticket(BaseModel):
    subject: str
    body: str


class Team(str, enum.Enum):
    billing = "billing"
    technical = "technical"


class Triage(Judgments):
    refund: bool
    """Does the customer explicitly ask for their money back?"""

    team: Team = ask("Which team should handle this ticket?",
                     criteria={"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"})

    urgency: float = ask("How urgent is this ticket operationally?",
                         levels=["Routine; nothing is broken", "Degraded", "Complete outage"])

    tone: Literal["calm", "angry"] = Field(description="What is the customer's tone?")

    severity: Literal["low", "high"] = ask("How severe is the impact?",
                                           levels=["Cosmetic or optional", "Customers are blocked"])


ANSWERS = {
    "refund": {"type": "noul", "noul": 0.93},
    "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "technical": 0.1}},
    "urgency": {"type": "score", "score": 0.4, "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1}},
    "tone": {"type": "choice", "choice": "angry", "probabilities": {"calm": 0.45, "angry": 0.55}},
    "severity": {"type": "score", "score": 0.8, "probabilities": {"0": 0.2, "1": 0.8}},
}


class QuestionCompilation(unittest.TestCase):
    def test_annotations_choose_primitives(self):
        q = Triage.questions()
        self.assertEqual(q["refund"], {"type": "noul",
                                       "instructions": "Does the customer explicitly ask for their money back?"})
        self.assertEqual(q["team"]["type"], "choice")
        self.assertEqual(q["team"]["criteria"]["billing"], "Charges, invoices, refunds")
        self.assertEqual(q["urgency"], {"type": "score", "instructions": "How urgent is this ticket operationally?",
                                        "criteria": ["Routine; nothing is broken", "Degraded", "Complete outage"]})
        self.assertEqual(q["tone"]["criteria"], {"calm": "calm", "angry": "angry"})
        self.assertEqual(q["severity"]["type"], "score")

    def test_missing_instructions_fail_at_definition(self):
        with self.assertRaisesRegex(TypeError, "instructions"):
            class Bad(Judgments):
                flag: bool

    def test_unsupported_types_fail_at_definition(self):
        with self.assertRaisesRegex(TypeError, "Score needs"):
            class Bad(Judgments):
                level: float = ask("How much?")
        with self.assertRaisesRegex(TypeError, "unknown options"):
            class Worse(Judgments):
                pick: Literal["a", "b"] = ask("Which?", criteria={"c": "nope"})
        with self.assertRaisesRegex(TypeError, "Optional"):
            class Worst(Judgments):
                pick: Literal["a", "b"] | None = ask("Which?")


class Decoding(unittest.TestCase):
    def test_typed_values_and_evidence(self):
        result = Triage.from_answers(ANSWERS)
        self.assertIs(result.refund, True)
        self.assertIs(result.team, Team.billing)
        self.assertAlmostEqual(result.urgency, 0.4)
        self.assertEqual(result.tone, "angry")
        self.assertEqual(result.severity, "high")
        self.assertAlmostEqual(result.probability("refund"), 0.93)
        self.assertAlmostEqual(result.probability("team", Team.technical), 0.1)
        self.assertEqual(result.distribution("severity"), {"low": 0.2, "high": 0.8})
        self.assertEqual(result.uncertain(0.8), ["urgency", "tone"])
        self.assertLess(result.confidence("tone"), 0.01)
        with self.assertRaises(ValidationError):
            result.refund = False  # Judgments are frozen.

    def test_probabilities_renormalize_over_criteria(self):
        answers = {**ANSWERS, "team": {"type": "choice", "choice": "billing",
                                       "probabilities": {"billing": 0.49, "technical": 0.49}}}
        self.assertAlmostEqual(Triage.from_answers(answers).probability("team", "billing"), 0.5)

    def test_wrong_answer_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected a noul"):
            Triage.from_answers({**ANSWERS, "refund": ANSWERS["team"]})


class Tasks(unittest.TestCase):
    def test_arguments_become_named_state(self):
        backend = Scripted(lambda state, questions: ANSWERS)

        @task(backend=backend)
        def triage(ticket: Ticket, customer_tier: str = "free") -> Triage:
            """Route an incoming ticket."""

        result = triage(Ticket(subject="Double charge", body="Refund me"))
        self.assertIsInstance(result, Triage)
        self.assertEqual(backend.calls[0]["state"], {
            "ticket": {"subject": "Double charge", "body": "Refund me"}, "customer_tier": "free"})
        self.assertEqual(set(backend.calls[0]["questions"]), set(Triage.model_fields))
        self.assertEqual(triage.__doc__, "Route an incoming ticket.")

    def test_body_shapes_state_and_single_string_is_raw(self):
        backend = Scripted(lambda state, questions: ANSWERS)

        @task(backend=backend)
        def shaped(ticket: Ticket) -> Triage:
            return {"message": ticket.body, "policy": "Refunds within 30 days."}

        @task(backend=backend)
        def raw(text: str) -> Triage: ...

        shaped(Ticket(subject="s", body="b"))
        raw("plain text")
        self.assertEqual(backend.calls[0]["state"], {"message": "b", "policy": "Refunds within 30 days."})
        self.assertEqual(backend.calls[1]["state"], "plain text")

    def test_map_keeps_order(self):
        def answers(state, questions):
            return {**ANSWERS, "refund": {"type": "noul", "noul": 0.9 if "refund" in state else 0.1}}

        @task(backend=Scripted(answers))
        def triage(text: str) -> Triage: ...

        results = triage.map(["refund please", "app crashed", "refund now"], concurrency=2)
        self.assertEqual([r.refund for r in results], [True, False, True])

    def test_return_type_is_required(self):
        with self.assertRaises(TypeError):
            @task
            def nope(text: str) -> dict: ...


class Backends(unittest.TestCase):
    def test_http_payload_and_retry(self):
        seen = []
        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"model": "nimble-latest", "answers": ANSWERS})

        backend = SystemOneHTTP("https://example.test/", "nimble-latest", api_key="secret")

        @task(backend=backend)
        def triage(text: str) -> Triage: ...

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await triage.acall("refund", client=client)

        result = asyncio.run(run())
        self.assertIs(result.refund, True)
        self.assertEqual(len(seen), 2)
        body = json.loads(seen[-1].content)
        self.assertEqual(seen[-1].url, "https://example.test/v1/systemone")
        self.assertEqual(seen[-1].headers["authorization"], "Bearer secret")
        self.assertEqual(body["model"], "nimble-latest")
        self.assertEqual(body["state"], "refund")
        self.assertNotIn("secret", repr(backend))

    def test_local_scorer_translation(self):
        class FakeScorer:
            def score(self, context, schema):
                self.context, self.schema = context, schema
                return {"fields": {
                    "refund": {"scores": {"false": 0.2, "true": 0.8}},
                    "team": {"scores": {"billing": 0.6, "technical": 0.4}},
                    "urgency": {"scores": {"0": 0.5, "1": 0.25, "2": 0.25}},
                    "tone": {"scores": {"calm": 0.9, "angry": 0.1}},
                    "severity": {"scores": {"0": 0.3, "1": 0.7}},
                }}

        scorer = FakeScorer()
        result = Triage.judge({"text": "x"}, backend=LocalScorer(scorer))
        self.assertEqual(scorer.context, '{"text": "x"}')
        self.assertEqual(scorer.schema["refund"], {"description": Triage.questions()["refund"]["instructions"],
                                                   "type": "boolean", "choices": [False, True]})
        self.assertEqual(scorer.schema["urgency"]["choices"], ["0", "1", "2"])
        self.assertEqual(scorer.schema["urgency"]["choice_descriptions"]["2"], "Complete outage")
        self.assertIs(result.team, Team.billing)
        self.assertAlmostEqual(result.urgency, 0.75)
        self.assertEqual(result.severity, "high")


if __name__ == "__main__":
    unittest.main()
