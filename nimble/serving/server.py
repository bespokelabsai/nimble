"""Launch the GPU backend and the lightweight Nimble API in separate runtimes."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import httpx
import uvicorn
from transformers import AutoTokenizer
from starlette.responses import RedirectResponse
from openjev.api import create_app
from openjev.backend import SGLangClient
from openjev.config import Settings
from openjev.runtime import stop_process, wait_ready
from openjev.service import EvaluationService

from nimble.scoring import parallel_schema
from nimble.scoring.calibration import ADAPTER_REVISIONS, fitted_temperature
from .compiler import NimbleCompiler

MODEL = "bespokelabs/Bespoke-Nimble-9B"
# Keep the trained prompt budget visible while allowing longer inference inputs.
# Override the serving budget with NIMBLE_MAX_PROMPT_TOKENS.
TRAINED_PROMPT_TOKENS = 2048
MAX_PROMPT_TOKENS = int(os.environ.get("NIMBLE_MAX_PROMPT_TOKENS", "8192"))


def served_temperature(ready):
    """Return the fitted temperature for the checkpoint that READY.json describes, else 1.0.

    deploy/modal_app.py records the Hub revision. merge_local_adapter.py records only the
    adapter's SHA-256. In both files, base_revision is the Qwen base and never selects one.
    """
    revision = ready.get("revision") or ADAPTER_REVISIONS.get(ready.get("adapter_sha256"))
    return fitted_temperature(MODEL, revision) or 1.0


def make_app(settings, service):
    app = create_app(settings, service=service)
    app.title = "Nimble"
    app.description = (
        "Nimble's trained candidate scoring on SGLang. Choice, Noul, and Score; "
        f"26 candidates, up to {MAX_PROMPT_TOKENS} prompt tokens (trained at {TRAINED_PROMPT_TOKENS}). "
        "Question IDs and option keys are included "
        f"in the trained prompt. Probabilities use temperature {settings.temperature:.4g}. "
        "Confidence is entropy concentration, not calibrated accuracy."
    )
    app.router.routes[:] = [r for r in app.router.routes if r.path not in {"/", "/v1/models", "/v1/limits"}]
    @app.get("/", include_in_schema=False)
    async def home():
        return RedirectResponse("/docs")

    for route in app.routes:
        if route.path == "/v1/systemone":
            route.description = app.description

    @app.get("/v1/models")
    async def models():
        names = ["nimble-latest", MODEL]
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "nimble"} for n in names],
                "models": [{"name": n, "description": MODEL, "release_date": "2026-09-17"} for n in names]}

    @app.get("/v1/limits")
    async def limits():
        return {"max_answers_per_question": 26, "max_questions": 64,
                "max_prompt_tokens": MAX_PROMPT_TOKENS, "trained_prompt_tokens": TRAINED_PROMPT_TOKENS,
                "max_input_tokens": settings.max_input_tokens,
                "max_body_bytes": settings.max_body_bytes,
                "max_total_input_tokens": settings.max_total_input_tokens,
                "max_concurrent_requests": settings.max_concurrent_requests}

    original_openapi = app.openapi
    def openapi():
        document = original_openapi()
        schemas = document["components"]["schemas"]
        for name in ("SystemOneRequest", "SystemOneResponse"):
            if "example" in schemas[name]:
                schemas[name]["example"]["model"] = "nimble-latest"
        schemas["SystemOneRequest"]["properties"]["model"]["description"] = (
            "Published checkpoint ID or nimble-latest."
        )
        for name in ("ChoiceQuestion", "ScoreQuestion"):
            criteria = schemas[name]["properties"]["criteria"]
            criteria["maxProperties" if name == "ChoiceQuestion" else "maxItems"] = 26
            criteria["description"] = "2–26 candidates, in the supplied order."
        return document
    app.openapi = openapi
    return app


async def main():
    path = Path(os.environ["NIMBLE_MODEL_PATH"])
    contract = json.loads((path / "schema_config.json").read_text())
    if hashlib.sha256(Path(parallel_schema.__file__).read_bytes()).hexdigest() != contract["prompt_code_sha256"]:
        raise RuntimeError("Local prompt compiler differs from the published training contract")
    settings = Settings(model=str(path), served_model_name=MODEL, model_alias="nimble-latest",
                        max_input_tokens=MAX_PROMPT_TOKENS + 1,
                        max_total_input_tokens=32 * (MAX_PROMPT_TOKENS + 1),
                        max_concurrent_requests=4, max_concurrent_branches=32,
                        temperature=served_temperature(json.loads((path / "READY.json").read_text())))
    command = ["/opt/sglang/bin/python", "-m", "sglang.launch_server",
               "--model-path", str(path), "--tokenizer-path", str(path),
               "--host", "127.0.0.1", "--port", "30000",
               # Qwen honors the config field; 0.5.19's CLI flag has a narrower allowlist.
               "--json-model-override-args", '{"language_model_only": true}',
               "--context-length", str(MAX_PROMPT_TOKENS + 1), "--dtype", "bfloat16",
               "--mem-fraction-static", "0.80", "--attention-backend", "flashinfer",
               "--mamba-radix-cache-strategy", "extra_buffer",
               "--cuda-graph-backend-prefill", "breakable", "--cuda-graph-max-bs-decode", "32",
               "--max-running-requests", "32", "--enable-cache-report", "--enable-metrics"]
    started = time.monotonic()
    process = subprocess.Popen(command, env={**os.environ, "SGLANG_RUST_SERVER": "1"}, start_new_session=True)
    try:
        async with httpx.AsyncClient(base_url=settings.backend_url, timeout=120,
                                    limits=httpx.Limits(max_connections=40)) as client:
            backend = SGLangClient(settings, client)
            await wait_ready(backend, process, 1100)
            compiler = NimbleCompiler(AutoTokenizer.from_pretrained(path, local_files_only=True),
                                      max_prompt_tokens=MAX_PROMPT_TOKENS)
            app = make_app(settings, EvaluationService(settings, compiler, backend))
            app.state.startup_seconds = round(time.monotonic() - started, 2)
            server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, access_log=False))
            async def monitor():
                while process.poll() is None:
                    await asyncio.sleep(1)
                server.should_exit = True
            task = asyncio.create_task(monitor())
            try:
                await server.serve()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    finally:
        stop_process(process)


if __name__ == "__main__":
    asyncio.run(main())
