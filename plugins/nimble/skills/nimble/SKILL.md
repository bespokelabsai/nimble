---
name: nimble
license: MIT
description: >
  Build with Nimble, Bespoke Labs' hosted API for typed judgments about text
  or JSON. You send a state and up to 64 typed questions (noul for true or
  false, choice for one of several options, score for a point on an ordered
  scale) to api.bespokelabs.ai, and Nimble returns a probability for every
  allowed answer. Use when a feature needs a judgment that code can act on,
  such as routing, triage, classification, moderation, or checking text
  against a rule; when an LLM prompt-and-parse step could become a structured
  decision; or when code uses Nimble, nimble-v4, BESPOKE_API_KEY, or
  api.bespokelabs.ai.
---

# Build with Nimble

Nimble is a family of models from Bespoke Labs that turn language into decisions your
code can use. You send a **state**, the text or JSON to judge, and a set of **typed
questions** about it. Nimble answers all of them in one response, with a probability for
each allowed answer. It writes no explanations, so the workflow stays in your code: act on
confident answers, and send uncertain ones to a fallback such as a person.

This skill covers the hosted API at `https://api.bespokelabs.ai`.

## Read the live docs

The [Nimble docs](https://docs.bespokelabs.ai) are the source of truth for models, limits,
prices, and SDK usage. When this skill and the docs disagree, follow the docs.

- Start from the [docs index](https://docs.bespokelabs.ai/llms.txt). Every page is also
  served as Markdown if you add `.md` to its path.
- Before you write an integration, read the pages for the task:

| Task | Read |
| --- | --- |
| Make a first request | [Quickstart](https://docs.bespokelabs.ai/nimble/quickstart.md) |
| Choose a model | [Overview](https://docs.bespokelabs.ai/nimble/overview.md) |
| Write questions and read answers | [Question types](https://docs.bespokelabs.ai/nimble/questions.md) |
| Call Nimble from Python | [TypeSafe SDK](https://docs.bespokelabs.ai/nimble/typesafe-sdk.md) |
| Handle limits, errors, and retries | [Limits and errors](https://docs.bespokelabs.ai/nimble/limits-and-errors.md) |
| Estimate cost | [Pricing](https://docs.bespokelabs.ai/nimble/pricing.md) |
| Check the exact schema | [OpenAPI spec](https://docs.bespokelabs.ai/api-reference/openapi.yaml) |

If the docs are unreachable, use the reference below and tell the user that you could not
check it against the live docs.

## The API at a glance

Send `POST https://api.bespokelabs.ai/v1/systemone` with the header
`Authorization: Bearer $BESPOKE_API_KEY`. Keys are created in the
[console](https://console.bespokelabs.ai/home/keys). Read the key from the environment or a
secret store, and never put it in client-side code.

```json
{
  "model": "nimble-v4",
  "state": "The customer was charged twice and asks for a refund. The service still works.",
  "questions": {
    "requests_refund": {
      "type": "noul",
      "instructions": "Does the customer ask for a refund?"
    },
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "Charges, payments and refunds",
        "technical": "Software bugs and outages"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": [
        "Routine request; the service works",
        "Some functionality is unavailable",
        "Complete service outage"
      ]
    }
  }
}
```

The response uses your question IDs as keys. The numbers here are only an example:

```json
{
  "model": "nimble-v4",
  "answers": {
    "requests_refund": { "type": "noul", "noul": 0.99 },
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.98, "technical": 0.02 },
      "confidence": 0.86
    },
    "urgency": {
      "type": "score",
      "score": 0.07,
      "legend": {
        "0": "Routine request; the service works",
        "1": "Some functionality is unavailable",
        "2": "Complete service outage"
      },
      "probabilities": { "0": 0.94, "1": 0.05, "2": 0.01 },
      "confidence": 0.77
    }
  },
  "usage": { "input_tokens": 338, "output_tokens": 4 }
}
```

| Type | Asks | `criteria` | Answer fields |
| --- | --- | --- | --- |
| `noul` | Whether a statement is true | Optional: `{"true": "...", "false": "..."}` | `noul`: the probability that the statement is true |
| `choice` | Which of 2 to 255 options fits best | Option keys mapped to descriptions; `null` means "use the key" | `choice` (the most likely key), `probabilities` (sum to 1), `confidence` |
| `score` | Where the state falls on 2 to 255 ordered levels | The levels as a list, lowest first | `score` (the expected level from 0; can fall between levels), `legend`, `probabilities`, `confidence` |

`confidence` is 1 when one answer has all the probability and 0 when every answer is
equally likely. It is not the probability that the answer is right.

The `state` can be a string, or a JSON object or list. Nimble reads JSON as JSON text, so
you can pass structured records as they are.

### Models

| Model | Use it |
| --- | --- |
| `nimble-v4` | By default. Fast and low cost. `nimble-latest` currently points to it. |
| `nimble-v4-max` | When accuracy matters more than cost or speed. It is the most accurate model, and it is slower and costs more per token. |
| `nimble-v3` | Not in new code. It is deprecated. |

A request without `model` uses `nimble-latest`, which moves to newer models as they are
released. Name a specific model in production: a new model returns different
probabilities, which can shift the thresholds you tuned.

### Limits

- 1 to 64 questions per request, and 2 to 255 options or levels per question.
- Up to 32,768 prompt tokens per question, including the state. Nimble never truncates a
  prompt, so a longer one fails with `422`.
- Request bodies up to 1 MiB. A larger body gets `413`.
- Each organization can run 4 requests at once until it first buys credit, and 100 after.
  Requests over the limit get `429`.

### Errors

| Status | Meaning | What to do |
| --- | --- | --- |
| `401` | The key is missing or invalid | Fix the key. Do not retry. |
| `402` | Not enough credit | Buy credit in the console. Do not retry until then. |
| `413`, `422` | The body is too large, or the request is invalid or over a limit (`detail` says why) | Fix the request. Never retry it unchanged. |
| `429` | The organization's concurrency limit is reached | Retry after the `Retry-After` seconds. |
| `502` | A model returned an error | Retry with exponential backoff. |
| `503` | A model is starting or briefly unavailable | Retry after `Retry-After`, or back off if the header is missing. |
| `504` | The request took too long | Send fewer questions or a shorter state. |
| `529` | The model is busy | Retry after about a second, with backoff. |

A `422` whose `detail` is "Nimble request failed" comes from the model, usually because a
prompt is over the token limit or the model name is unknown. Failed requests are not
charged. Every response, including errors, has an `x-request-id` header. Log it, because
Bespoke Labs support needs it to trace a request.

### Cost

Nimble charges for input tokens only. Each question's prompt repeats the shared text, which
is the state and all the question definitions, but you pay for the shared text once per
request. So one request with many questions costs much less than many requests with one
question each. `usage.input_tokens` in the response is the billed count.

## Call Nimble from Python

Nimble accepts TypeSafe's System One request format, so use TypeSafe's Python SDK and point
it at Bespoke Labs.

```sh
pip install typesafe-sdk
```

```python
import os

from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient

# Example thresholds. Tune them on labeled examples from your own data.
REFUND_THRESHOLD = 0.9
ROUTING_THRESHOLD = 0.8

client = TypeSafeClient(
    api_key=os.environ["BESPOKE_API_KEY"],
    base_url="https://api.bespokelabs.ai",
    model="nimble-v4",
    # The SDK retries failed calls by itself. Limit it to Nimble's retryable statuses.
    retry=RetryPolicy(http_statuses={429, 502, 503, 529}),
)

ticket_text = "I was charged twice for my subscription this month. Please refund one of the charges."

response = client.system_one(
    ticket_text,
    {
        "requests_refund": Noul(instructions="Does the customer ask for a refund?"),
        "department": Choice(
            instructions="Which department should handle this request?",
            criteria={
                "billing": "Charges, payments and refunds",
                "technical": "Software bugs and outages",
            },
        ),
    },
)

refund = response.nouls["requests_refund"]
if refund.noul >= REFUND_THRESHOLD:
    print(f"Refund requested (p={refund.noul:.2f})")
else:
    print(f"No clear refund request (p={refund.noul:.2f}), so a person should check")

department = response.choices["department"]
p = department.probabilities[department.choice]
if p >= ROUTING_THRESHOLD:
    print(f"Route to {department.choice} (p={p:.2f})")
else:
    print(f"Unsure of the department, so send it to human review: {department.probabilities}")
```

- `response.nouls`, `response.choices`, and `response.scores` hold the answers of each type,
  typed for your type checker. `response.answers` holds all of them.
- `Score(instructions=..., criteria=[...])` asks a score question. Read `.score` from
  `response.scores[...]`. Its `.probabilities` are keyed by level number.
- Pass `model="nimble-v4-max"` to `system_one` to use another model for one call.
- `AsyncTypeSafeClient` takes the same arguments. Use it with `async with`.
- The SDK retries failed calls on its own: by default up to 2 retries within 30 seconds,
  honoring `Retry-After`. Its default statuses also include `504` and every other `5xx`, so
  pass `RetryPolicy(http_statuses={429, 502, 503, 529})` as above. Do not wrap SDK calls in a
  retry loop of your own. Change the `RetryPolicy` instead.
- The SDK can also read `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, and
  `TYPESAFE_DEFAULT_MODEL`. Either way, set the base URL to `https://api.bespokelabs.ai`,
  or the SDK will not call Nimble, and use a Bespoke Labs key, not a TypeSafe key.

From other languages, send the JSON above with any HTTP client.

## Design the questions

Start from the decision your code has to make, and pick the type that matches it:

- A property that is true or false: `noul`. Set a threshold on its probability.
- Exactly one of a fixed set of options: `choice`. Cover every real case, and add an
  explicit option such as `none_of_these` when an input can fit none of them.
- A point on an ordered scale, such as urgency, severity, or quality: `score`. Describe each
  level concretely, lowest first.

Then write the questions:

- Put the material to judge in `state`, the decision in `instructions`, and the meaning of
  each answer in `criteria`.
- Ask every question about the same state in one request. Nimble reads the state once for
  all of them, which is faster and cheaper than separate requests.
- Use descriptive question IDs and option keys, such as `requests_refund` rather than `q1`.
  They are part of the prompt, so renaming one can change its answer.
- Split a compound decision into separate questions instead of combining conditions in one
  question.
- When two options are easy to confuse, say in their descriptions what separates them.

## Act on the answers

- Decide what happens at each level of certainty. Act automatically when a probability
  clears its threshold, and send the rest to a fallback, such as human review or a safe
  default.
- Choose thresholds from labeled examples, not by guessing, and measure accuracy on a small
  labeled set at the threshold you pick.
- Before you act on a `choice` answer, check `probabilities[choice]`, the probability of the
  chosen option, against a threshold, as the Python example does. Send low values to review.
- Use `probabilities` when answers near a boundary matter, and the expected `score` to rank
  items.
- Start with `nimble-v4`. Move a question to `nimble-v4-max` only if labeled examples show
  that it needs more accuracy, and tune its threshold again when you do.

## Before you finish

- Run a few representative inputs, including edge cases, and check every answer field your
  code reads.
- Handle every status in the error table. Retry only `429`, `502`, `503`, and `529`, with
  backoff, and honor `Retry-After`. With the Python SDK, set this with `RetryPolicy` instead
  of writing your own retry loop.
- Keep `BESPOKE_API_KEY` out of source control and client bundles.
