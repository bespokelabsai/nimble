# Published Nimble on Modal

The endpoint is now public: no API key or Modal account is needed to call it.
Share the [quickstart instructions](TRY_NIMBLE.md) with people who want to try it.
The [public verification](assets/modal-public-smoke.json) confirms that the docs,
OpenAPI schema, and the quickstart inference request succeed without credentials.
The root URL redirects to the interactive docs.

The deployment serves `bespokelabs/Bespoke-Nimble-9B` at revision
`93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c`. This is the published 2,676-example
adapter, not the newer local checkpoint. Source code lives under `nimble/serving`. The pinned revision predates the rename;
its weights are unchanged. The existing `openjeff-models` Volume and
`openjeff-huggingface` Secret retain their original resource names.

It reuses [openjev-sglang](https://github.com/ekzhang/openjev-sglang) at commit
`7f84bedc169439f03379c2fa8d00ada220af2295`: its SGLang client, shared-prefix warmup,
parallel branch scheduling, cancellation, candidate probability normalization,
and typed responses. The prompt compiler is replaced with this checkpoint's exact
training prompt. In particular, boolean codes mean `A=false`, `B=true`.

## Deploy

Use Python 3.12 in a separate environment:

```sh
python3.12 -m venv .cache/venvs/modal
.cache/venvs/modal/bin/python -m pip install -r requirements/modal.txt
.cache/venvs/modal/bin/python deploy/setup_credentials.py
.cache/venvs/modal/bin/python deploy/modal_cli.py run deploy/modal_app.py
.cache/venvs/modal/bin/python deploy/modal_cli.py deploy deploy/modal_app.py
.cache/venvs/modal/bin/python deploy/smoke.py
```

The wrapper reads the existing project `.modal.toml` (or `.modal.tomlf`) and accepts
both `token-id`/`token-secret` and Modal's underscore spelling. The setup helper
reads `HF_TOKEN` or `.env`'s `HF_API_KEY`, creates a named Modal download secret,
and saves a proxy token to `.cache/modal/proxy-token.json` with owner-only access.
Existing named secrets and the local proxy token are reused. No credentials are
printed or baked into images. The proxy token is a workspace credential; keep it
private and revoke it through Modal when no longer needed.

The first run downloads the pinned base revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, merges the LoRA with PEFT in BF16 on
CPU, and saves it to the `openjeff-models` Modal Volume. Subsequent runs reuse that
artifact. The serving image receives only the prompt and serving source files;
datasets, local credentials, and the rest of the workspace are not uploaded.

The Qwen backend uses `--json-model-override-args '{"language_model_only": true}'`.
In the pinned SGLang 0.5.19 release, `--language-only` still initializes a vision
processor, and the separate `--language-model-only` CLI flag excludes Qwen from its
allowlist. The model configuration field is honored by the Qwen loader and keeps
this endpoint text-only without changing the checkpoint's language weights.

## API

Deployed endpoint: <https://bespokelabs--nimble-sglang-nimble.us-west.modal.direct>

Manage it in the [Modal dashboard](https://modal.com/apps/bespokelabs/main/deployed/nimble-sglang).

`POST /v1/systemone` accepts TypeSafe-shaped requests:

```json
{
  "model": "nimble-latest",
  "state": "The customer was charged twice and requests a refund.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this?",
      "criteria": {"billing": "Charges and refunds", "technical": "Software bugs"}
    }
  }
}
```

No authentication headers are required. Opening the base URL redirects to `/docs`,
where Swagger's **Try it out** button can send requests from the browser.
`/health`, `/v1/models`, `/v1/limits`, and `/openapi.json` are also public.
`deploy/smoke.py` tests all three question types without loading credentials and
saves its report to `.cache/modal/smoke-result.json`.

To restore private access later, set `unauthenticated=False` in `deploy/modal_app.py`
and redeploy. The existing proxy credentials remain available locally; pass
`--private` to the smoke or benchmark client for an authenticated deployment.
Deployment and model-download credentials are still required for administration;
they are not needed by people trying the public API.

Choice returns a winning key and candidate probabilities. Noul returns the
probability of true. Score returns the expected zero-based rubric index and the
distribution over levels. Choice/Score confidence is one minus normalized entropy,
not a calibrated probability of correctness or an exact reproduction of Jev's
confidence statistic. Probabilities are conditional on the supplied candidates.

## Limits and operation

- One H100 GPU container maximum; zero minimum; scale down after 120 idle seconds.
  Modal's `H100!` setting prevents automatic H200 substitution for benchmarking.
- Model loading and kernel compilation make the first request after idling slower.
- 2–26 Choice candidates or Score levels, at most 64 questions, and 8,192 prompt
  tokens per branch by default (`NIMBLE_MAX_PROMPT_TOKENS` overrides it). The adapter
  was trained on prompts of up to 2,048 tokens, so longer prompts are accepted but less
  tested. The backend reserves one additional token. Oversized prompts are rejected,
  never truncated. `/v1/limits` reports both numbers.
- Four active evaluations per container, at most 32 concurrent branches, and a
  262,176-token total submitted-prompt budget per evaluation by default
  (`32 × (NIMBLE_MAX_PROMPT_TOKENS + 1)`).
- Shared prefix warmup followed by concurrent field scoring. SGLang currently
  emits one discarded warmup token and one token per field to expose the selected
  next-token log probabilities: `N + 1` output tokens for `N` fields. This is not a
  zero-generation backend or a claim of zero cache allocation.
- Original field IDs, option keys, and schema descriptions remain in the prompt
  to match training. Unlike the TypeSafe service, question IDs can therefore
  affect predictions. Text/JSON states are serialized as data; there are no image
  inputs or arbitrary free-form generation endpoints.
- The merged BF16 weights and SGLang kernels may produce small numerical
  differences from unmerged PyTorch/MLX inference. A smoke test is not a new
  accuracy evaluation of the hosted runtime.

Inspect and stop the app through the Modal dashboard, or use
`deploy/modal_cli.py app list --json` and `deploy/modal_cli.py app stop <app-id>`.
Stopping this app does not delete its cached model Volume or access tokens.

## Local verification

```sh
.cache/venvs/modal/bin/python -m pip install pytest
.cache/venvs/modal/bin/python -m pytest tests/test_modal_serving.py -q
```

Set `NIMBLE_TEST_TOKENIZER` to the published tokenizer directory if the original
local adapter cache is absent. These tests check token-for-token parity with the
reference prompt, boolean ordering, limits, normalization, parallel execution,
authentication, and API metadata without loading model weights.

## Deployment verification

The initial L40S [saved live check](assets/modal-serving-smoke.json) used one three-question
billing request, followed by an identical request. The first ready inference took
1,062 ms end-to-end; the repeat took 310 ms, including about 106 ms of API-side
preparation and inference. These two observations are a smoke test, not a latency
benchmark. Successful backend initialization took 144 seconds, excluding earlier
deployment troubleshooting and scheduling.

The model returned refund probability 0.99945, selected `billing` with probability
0.99867, and placed urgency near level zero. Both requests returned normalized
distributions and four accounted output tokens. Scheduler logs showed three field
branches in a single batch: 768 cached tokens on the first field batch and 960 on
the repeat. The Rust response omits its cache counter, so the HTTP report leaves
that value null rather than claiming zero cache hits.

During the initial private deployment, unauthenticated requests returned 401; an unsupported 27-choice question returned
422, and model metadata, limits, and OpenAPI routes returned 200. No language-weight
loading warnings were found; only the unused vision weights were skipped. Three
local integration tests and 55 upstream API/backend tests passed.

## H100 deployment and repeated-request latency

The current endpoint runs on one H100, verified in the container as
`NVIDIA H100 80GB HBM3`. The URL, model revision, and configured inference parameters
remain the same as the original L40S deployment. The benchmark below was measured
while authentication was enabled; the endpoint was subsequently made public.

The [comparison measurements](assets/modal-gpu-latency.json) use the upstream
repository's three-question sample. Each GPU received one excluded warmup and
20 sequential identical requests over a persistent HTTP connection, with cache
reuse. Percentiles use linear interpolation.

| Measurement | L40S | H100 |
| --- | ---: | ---: |
| End-to-end median | 200.95 ms | 299.45 ms |
| End-to-end mean | 205.93 ms | 330.28 ms |
| End-to-end p95 | 237.77 ms | 371.09 ms |
| End-to-end minimum | 192.40 ms | 287.56 ms |
| End-to-end maximum | 257.71 ms | 679.81 ms |
| Server processing median | 102.38 ms | 67.56 ms |
| Readiness wait, excluded | 158.23 s | 163.70 s |

H100 lowered median server processing time by 34.0%, while observed end-to-end
latency increased by 49.0%. Median time outside the instrumented server stages
increased from 97.86 ms to 232.06 ms. These runs used separate container allocations;
network/routing overhead was not controlled or isolated, so the end-to-end change
cannot be attributed to the GPU alone. All 20 H100 requests succeeded.

Repeat this benchmark with a request JSON file:

```sh
.cache/venvs/modal/bin/python deploy/benchmark.py \
  --request .cache/modal/latency-request.json \
  --output .cache/modal/upstream-sample-latency-h100.json \
  --gpu-label H100 --count 20
```

The request file used in this run is also embedded in the comparison report.
