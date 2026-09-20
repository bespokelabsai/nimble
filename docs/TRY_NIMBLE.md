# Try Bespoke-Nimble-9B

Nimble turns text into typed decisions and probabilities. This public demo runs on
an H100. No API key, account, or installation is needed for the browser demo.

## In your browser

1. Open <https://bespokelabs--nimble-sglang-nimble.us-west.modal.direct/docs>.
2. Expand **POST /v1/systemone**, then click **Try it out**.
3. Keep `"model": "nimble-latest"`. Try the prefilled example, or edit `state`
   and the questions to describe your own classification task.
4. Click **Execute** and read the JSON response below the request.

If the page shows **503 Service Unavailable**, the GPU is waking from idle. Wait
about three minutes and refresh. The service sleeps after two minutes without
requests; later requests may need the same startup wait.

## From a terminal

This example checks a refund request, routes it to a department, and scores urgency.
The retry options handle the temporary 503 responses during startup.

```sh
curl --fail-with-body --retry 60 --retry-delay 5 --retry-max-time 600 \
  --max-time 180 \
  'https://bespokelabs--nimble-sglang-nimble.us-west.modal.direct/v1/systemone' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nimble-latest",
    "state": "I was charged twice. Please refund the duplicate.",
    "questions": {
      "refund": {
        "type": "noul",
        "instructions": "Does the user request a refund?"
      },
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["Routine", "Urgent", "Emergency"]
      }
    }
  }'
```

## Reading the results

- **Noul:** `noul` is the probability that the answer is true, from 0 to 1.
- **Choice:** `choice` is the selected option; `probabilities` scores every option.
- **Score:** `score` is the probability-weighted rubric position. In this example,
  Routine = 0, Urgent = 1, and Emergency = 2.

The probabilities compare the supplied options. `confidence` summarizes how
concentrated the distribution is; it is not a probability that the model is correct.

The complete context and schema must fit within 8,192 tokens per field. The model was
trained on prompts of up to 2,048 tokens, so shorter prompts are the better-tested
range. Choice and Score support 2–26 options. The demo has one GPU and can
return a busy response (529); wait briefly and retry.
