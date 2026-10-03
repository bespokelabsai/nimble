"""Where judgments run: the TypeSafe Jev API, a hosted Nimble endpoint, or a local scorer.

Every backend takes JSON state and wire questions and returns wire answers in
the `/v1/systemone` response shape, so tasks never depend on the backend.
"""

import asyncio
import json
import os

import httpx

NIMBLE_URL = "https://bespokelabs--nimble-sglang-nimble.us-west.modal.direct"
TYPESAFE_URL = "https://api.typesafe.ai"
RETRYABLE = {429, 500, 502, 503, 504, 529}


class BackendError(RuntimeError):
    def __init__(self, status, message=""):
        super().__init__(f"HTTP {status} {message}".strip())
        self.status = status


class SystemOneHTTP:
    """Any service that speaks `POST /v1/systemone`: Jev or Nimble on openjev-sglang."""

    def __init__(self, url, model, api_key=None, timeout=90, attempts=4, headers=None):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.timeout = timeout
        self.attempts = attempts
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def __repr__(self):  # Never show the key.
        return f"{type(self).__name__}(url={self.url!r}, model={self.model!r})"

    async def evaluate(self, state, questions, client=None):
        payload = {"model": self.model, "state": state, "questions": questions}
        own = client is None
        client = client or httpx.AsyncClient(timeout=self.timeout)
        try:
            for attempt in range(self.attempts):
                try:
                    response = await client.post(self.url, json=payload, headers=self._headers)
                except httpx.TransportError:
                    if attempt == self.attempts - 1:
                        raise BackendError(0, "connection failed") from None
                else:
                    if response.status_code == 200:
                        return response.json()["answers"]
                    # Provider bodies may echo inputs; surface only the status.
                    if response.status_code not in RETRYABLE or attempt == self.attempts - 1:
                        raise BackendError(response.status_code)
                await asyncio.sleep(0.5 * 2 ** attempt)
        finally:
            if own:
                await client.aclose()


def jev(model="jev-1.13.0", api_key=None, url=TYPESAFE_URL, **kwargs):
    """TypeSafe's hosted Jev; reads TYPESAFE_API_KEY when no key is given."""
    api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise ValueError("Set TYPESAFE_API_KEY or pass api_key=")
    return SystemOneHTTP(url, model, api_key=api_key, **kwargs)


def nimble(url=None, model="nimble-latest", api_key=None, **kwargs):
    """A Nimble endpoint served with `deploy/modal_app.py` (public by default)."""
    return SystemOneHTTP(url or os.environ.get("NIMBLE_URL", NIMBLE_URL), model,
                         api_key=api_key or os.environ.get("NIMBLE_API_KEY"), **kwargs)


class LocalScorer:
    """Run on an already-loaded `ParallelScorer` (MLX) or `CudaCandidateScorer`.

    Mirrors `nimble.serving.compiler`: Score levels become enum choices "0".."n-1".
    """

    def __init__(self, scorer, **score_kwargs):
        self.scorer = scorer
        self.score_kwargs = score_kwargs
        self._lock = asyncio.Lock()  # One forward pass at a time on one device.

    @staticmethod
    def schema(questions):
        schema = {}
        for name, q in questions.items():
            instructions = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])
            field = {"description": instructions}
            if q["type"] == "noul":
                field.update(type="boolean", choices=[False, True])
                criteria = q.get("criteria") or {}
                descriptions = {k: criteria[c] for k, c in (("false", "no"), ("true", "yes")) if criteria.get(c)}
                if descriptions:
                    field["choice_descriptions"] = descriptions
            elif q["type"] == "choice":
                field.update(type="enum", choices=list(q["criteria"]),
                             choice_descriptions={k: v or k for k, v in q["criteria"].items()})
            else:
                keys = [str(i) for i in range(len(q["criteria"]))]
                field.update(type="enum", choices=keys, choice_descriptions=dict(zip(keys, q["criteria"])))
            schema[name] = field
        return schema

    @staticmethod
    def answers(questions, result):
        answers = {}
        for name, q in questions.items():
            scores = result["fields"][name]["scores"]
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": scores["true"]}
            elif q["type"] == "choice":
                answers[name] = {"type": "choice", "choice": max(scores, key=scores.get), "probabilities": scores}
            else:
                answers[name] = {"type": "score", "probabilities": scores,
                                 "score": sum(int(k) * p for k, p in scores.items())}
        return answers

    async def evaluate(self, state, questions, client=None):
        context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, allow_nan=False)
        async with self._lock:
            result = await asyncio.to_thread(self.scorer.score, context, self.schema(questions),
                                             **self.score_kwargs)
        return self.answers(questions, result)


class Scripted:
    """A deterministic backend for tests: `fn(state, questions) -> answers`."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    async def evaluate(self, state, questions, client=None):
        self.calls.append({"state": state, "questions": questions})
        return self.fn(state, questions)
