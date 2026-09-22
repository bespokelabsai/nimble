# Parallel constrained scoring in MLX

Run the commands below from the project root. See the [folder layout](../README.md#folder-layout).

Implemented for the existing unquantized Qwen3.5-4B text backbone on this Mac.
One context plus a schema produces a selected value and scores for every enum
or boolean field. Model weights come from the existing pinned local checkpoint;
MLX loads the language weights and omits the unused vision encoder.

## Run

```sh
.venv-mlx/bin/python -m nimble.scoring.parallel_scorer \
  --schema-file examples/parallel_schema.json \
  --context "Our production payment service is completely down for every customer, and no one can complete a purchase."
```

The earlier priority schema also works:

```sh
.venv-mlx/bin/python -m nimble.scoring.parallel_scorer \
  --schema-file examples/priority_defined_schema.json \
  --context "Please add a filter when you have time; there is no business impact."
```

Python API, using `.venv-mlx/bin/python`:

```python
import json
from nimble.scoring.parallel_scorer import ParallelScorer

schema = json.load(open("examples/parallel_schema.json"))
scorer = ParallelScorer()  # Load once and reuse.
result = scorer.score("The payment service is down for all customers.", schema)
print(result["output"])  # Programmatically assembled, typed values.
print(result["fields"]["risk_level"]["scores"])
```

`--schema` accepts inline JSON. `--output` also saves results to a file.
`--temperature` defaults to 1. In Python, `ParallelScorer` uses a fitted
temperature only when the `model_id` and `revision` that you pass exactly match
a checkpoint that has one. Otherwise it uses 1. The only such checkpoint is
`model_id="bespokelabs/Bespoke-Nimble-9B"` with
`revision="93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c"`, and its temperature is
2.179. Pass the full 40-character revision hash. If you pass a short hash such
as `93ec5d6`, the scorer uses a temperature of 1. The README quickstart saves
the full hash for you. See [Probability temperature](../README.md#probability-temperature).
`--max-input-tokens` defaults to 4096 and includes context, schema, and suffix.
Overlength prompts are rejected without truncation.

Each enum accepts 1–26 distinct string choices. Boolean fields use:

```json
{
  "requires_review": {
    "type": "boolean",
    "description": "Whether the incident requires immediate human review."
  }
}
```

Booleans default to `[false, true]`; explicit boolean `choices` may reverse that
order. Selected booleans remain native JSON booleans. Score-object keys are
`"false"` and `"true"`, as JSON object keys must be strings. Optional
`choice_descriptions` works for enums and booleans. Nested fields and other
types are rejected in this version.

## Execution

1. Render instructions, context, all schema descriptions, and code-to-choice maps
   as one shared chat prefix. The short field selector comes after that prefix.
2. Find the exact common token prefix, accounting for token merges at text
   boundaries. Recombining prefix and suffix exactly reproduces each full prompt.
3. Prefill the text backbone once, retaining all 32 layer states: ordinary
   attention KV state and the convolution/recurrent states in linear-attention
   layers. No vocabulary projection is performed during prefill.
4. Fork new cache objects and broadcast their initial arrays across field rows.
   Evaluate the suffixes in one batched backbone call. Right padding occurs only
   after valid suffix tokens; scores use each row's last valid position. Branch
   caches are discarded, and the source prefix is never updated by a branch.
5. Gather only the relevant tied-embedding/output-head rows and project final
   hidden states into the union of allowed code tokens. Each field then selects
   its own subset. The full 248,320-token vocabulary head is never computed.
6. Compute `softmax(candidate_logits / temperature)` and select the top choice.
   Code tokens A/B/C/... map deterministically to enum strings or boolean values.
   The output JSON is assembled in Python; no model-generated JSON is parsed.

All fields are batched by default. `--field-batch-size N` divides them into
smaller groups while still prefilling once; this can limit branch memory use.
`--mode cached_serial` reuses the prefix but runs one field suffix at a time.
`--mode independent` recomputes each complete prompt and is the reference path.
All three modes use exactly the same prompt tokens and candidate projection.

This is a batched GPU computation, not separate concurrent model processes.
Each field is classified independently given the same context and schema.
Cross-field consistency rules are not enforced automatically.

## Correctness and limits

- Single-token codes avoid multi-token enum-prefix ambiguity. Literal enum token
  trees and allocation-free tree continuations are not implemented.
- Cache broadcasting saves repeated prefill computation. Standard MLX cache
  appends can allocate/materialize per-branch arrays; this is not a guarantee of
  zero allocation or constant cache memory as field count grows.
- Scores are normalized preferences over allowed choices, not calibrated
  confidence. Label/position bias and prompt sensitivity remain possible.
- Candidate projection accumulates in FP32 using the original BF16 weights.
  Raw logits therefore need not be bitwise equal to a BF16 full-vocabulary head.
- Metal kernel selection and BF16 arithmetic also cause small differences between
  batched, cached sequential, and full independent execution. Close decisions
  can change under such differences; no universal agreement is promised.
- The previous PyTorch enum scorer used one field definition per prompt. This
  version puts every definition into the shared prefix. Comparing their scores
  directly would confound prompt changes with cache changes.
- Full-vocabulary probabilities are intentionally omitted because they require
  computing the entire output head. Returned logits and probabilities refer to
  the internal single-token codes, not literal enum-string sequence probabilities.

## Measured performance

Apple M5 Pro, 64 GB unified memory; original BF16 language weights; MLX 0.32.2,
MLX-LM 0.31.3. Three repetitions per mode after warming kernels. Timings exclude
model loading and tokenization, and include prefill plus field evaluation.
Long contexts add about 1,000 context tokens. Eight-field schemas repeat the
three example definitions under distinct names; these are synthetic benchmarks.

| Fields | Shared-prefix tokens | Parallel | Cached sequential | Independent | Speedup vs independent |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 239 | 0.139 s | 0.138 s | 0.102 s | 0.74× |
| 3 | 433 | 0.232 s | 0.284 s | 0.514 s | 2.22× |
| 8 | 950 | 0.438 s | 0.696 s | 2.905 s | 6.63× |
| 3 | 1,436 | 0.598 s | 0.649 s | 1.602 s | 2.68× |
| 8 | 1,953 | 0.853 s | 1.100 s | 6.180 s | 7.25× |

For the eight-field longer-context case, parallelism added about 1.29× speedup
over already sharing the cache sequentially. Most of the overall 7.25× gain is
from avoiding repeated prefills. A single field is faster with `--mode independent`.

Measured peak MLX active allocation was 8.21–9.40 GiB across the parallel cases,
including model weights. These are MLX allocator measurements, not total system
memory. Shared prefill activations dominated the peak in these runs; branch
memory need not remain flat for more fields or much longer contexts.

All 23 field selections across the five scenarios matched both reference modes.
The largest absolute probability difference was 0.018008 (about 1.8 percentage
points), measured against full independent inference; against cached sequential
it was 0.012336. The benchmark explicitly requires matching labels and probability
differences at most 0.03. These are test tolerances, not calibration guarantees.

Full measurements: `examples/parallel_benchmark.json`.
An actual three-field output is saved in `examples/parallel_results.json`; it is
a model response, not a claim that its classification is correct.

## Reproduce checks and benchmark

```sh
.venv-mlx/bin/python -m unittest -v tests.test_parallel_scorer
.venv-mlx/bin/python -m nimble.evaluation.benchmark_parallel --repeats 3
```

The small hybrid model tests check exact cache isolation, unequal suffix lengths,
single fields, smaller field batches, candidate-only projection, boolean output,
temperature, and both CPU reference and Metal execution. CPU FP32 comparisons
agree to about 1e-6; Metal has explicit numerical tolerances. Tokenizer checks
verify exact prefix reconstruction, chat-token handling, and overlength errors.

To recreate the isolated runtime after the original weights are downloaded:

```sh
python3 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements/mlx.txt
```

Use a normal native macOS terminal with Metal GPU access. The scorer uses only
the local checkpoint and does not run a hosted inference request.
