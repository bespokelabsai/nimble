"""Training-prompt parity and API behavior without a GPU."""

import asyncio
import math
import os
from pathlib import Path

import pytest

pytest.importorskip("openjev")
from fastapi.testclient import TestClient
from transformers import AutoTokenizer
from openjev.backend import Generation
from openjev.config import Settings
from openjev.models import SystemOneRequest
from openjev.service import EvaluationService
from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.scoring.calibration import fitted_temperature
from nimble.scoring.parallel_schema import prepare_prompts
from nimble.serving.compiler import NimbleCompiler
from nimble.serving.server import MAX_PROMPT_TOKENS, MODEL, make_app


@pytest.fixture(scope="module")
def tokenizer():
    path = Path(os.getenv("NIMBLE_TEST_TOKENIZER", ".cache/adapters/openjevon-mixed9b-v1"))
    if not (path / "tokenizer.json").exists():
        pytest.skip("Set NIMBLE_TEST_TOKENIZER to the published tokenizer directory")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.fixture
def payload():
    return {"model": "nimble-latest", "state": {"ticket": "I was charged twice. Please refund me."},
            "questions": {
                "refund": {"type": "noul", "instructions": "Is a refund requested?",
                           "criteria": {"false": "No refund request", "true": "Refund requested"}},
                "department": {"type": "choice", "instructions": "Where should this go?",
                               "criteria": {"billing": "Payments", "technical": "Software bugs"}},
                "urgency": {"type": "score", "instructions": "How urgent is this?",
                            "criteria": ["Routine", "Urgent", "Emergency"]}}}


def test_exact_training_prompt_and_boolean_order(tokenizer, payload):
    actual = NimbleCompiler(tokenizer).prepare(SystemOneRequest.model_validate(payload))
    context, schema = adapt_input(payload)
    expected = prepare_prompts(tokenizer, context, schema, 2048)
    assert actual.prefix_ids == expected.prefix_ids
    assert [b.input_ids for b in actual.branches] == expected.full_ids
    assert [b.label_ids for b in actual.branches] == expected.candidate_ids
    assert actual.branches[0].option_keys == ["false", "true"]


def test_candidate_and_length_limits(tokenizer, payload):
    payload["questions"]["department"]["criteria"] = {str(i): str(i) for i in range(27)}
    with pytest.raises(ValueError, match="1–26"):
        NimbleCompiler(tokenizer).prepare(SystemOneRequest.model_validate(payload))
    del payload["questions"]["department"]
    payload["state"] = "Long context " * 1500
    with pytest.raises(ValueError, match="Nothing was truncated"):  # over the trained 2,048 default
        NimbleCompiler(tokenizer).prepare(SystemOneRequest.model_validate(payload))
    longer = NimbleCompiler(tokenizer, max_prompt_tokens=8192).prepare(SystemOneRequest.model_validate(payload))
    assert 2048 < len(longer.branches[0].input_ids) <= 8192
    payload["state"] = "Long context " * 5000
    with pytest.raises(ValueError, match="Nothing was truncated"):
        NimbleCompiler(tokenizer, max_prompt_tokens=8192).prepare(SystemOneRequest.model_validate(payload))


class Backend:
    def __init__(self):
        self.warmed = False
        self.active = self.peak = self.calls = 0

    async def health(self):
        return True

    async def generate(self, ids, label_ids=None):
        self.calls += 1
        if label_ids is None:
            await asyncio.sleep(0.001)
            self.warmed = True
            return Generation([], len(ids), 1, 0)
        assert self.warmed
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        probs = [0.2, 0.8] if len(label_ids) == 2 else [0.2, 0.3, 0.5]
        return Generation([math.log(p) for p in probs], len(ids), 1, 10)


def test_parallel_scoring_auth_and_catalogue(tokenizer, payload):
    settings = Settings(model="test", served_model_name="test", model_alias="nimble-latest",
                        api_key="local-test-secret", max_input_tokens=MAX_PROMPT_TOKENS + 1)
    backend = Backend()
    service = EvaluationService(settings, NimbleCompiler(tokenizer, max_prompt_tokens=MAX_PROMPT_TOKENS), backend)
    with TestClient(make_app(settings, service)) as client:
        assert client.post("/v1/systemone", json=payload).status_code == 401
        assert backend.calls == 0
        headers = {"Authorization": "Bearer local-test-secret"}
        response = client.post("/v1/systemone", headers=headers, json=payload)
        assert response.status_code == 200, response.text
        answers = response.json()["answers"]
        assert answers["refund"]["noul"] == pytest.approx(0.8)
        assert answers["urgency"]["score"] == pytest.approx(1.3)
        assert answers["department"]["choice"] == "technical"
        assert response.json()["usage"]["output_tokens"] == 4
        assert backend.calls == 4 and backend.peak == 3
        limits = client.get("/v1/limits", headers=headers).json()
        assert limits["max_answers_per_question"] == 26
        assert limits["max_input_tokens"] == MAX_PROMPT_TOKENS + 1
        assert (limits["max_prompt_tokens"], limits["trained_prompt_tokens"]) == (MAX_PROMPT_TOKENS, 2048)
        assert client.get("/openapi.json", headers=headers).status_code == 200


def test_fitted_temperature_scales_served_probabilities(tokenizer, payload):
    # Settings ignores unknown fields, so a renamed upstream field would silently serve T=1.
    temperature = fitted_temperature(MODEL, "93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c")
    settings = Settings(model="test", served_model_name="test", model_alias="nimble-latest",
                        max_input_tokens=MAX_PROMPT_TOKENS + 1, temperature=temperature)
    service = EvaluationService(settings, NimbleCompiler(tokenizer, max_prompt_tokens=MAX_PROMPT_TOKENS), Backend())
    app = make_app(settings, service)
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    true, false = 0.8 ** (1 / temperature), 0.2 ** (1 / temperature)
    assert response.json()["answers"]["refund"]["noul"] == pytest.approx(true / (true + false))
    assert "temperature 2.179" in app.description
