# Separate models for training and evaluation data

> Dataset cleanup: only `data/train.jsonl` (2,826) and `data/eval.jsonl` (324)
> remain, with their manifest. See [the current dataset guide](DATASET.md).
> Old release paths and commands below describe historical experiments and
> require separately supplied source data; those releases are no longer stored.

New training contrasts default to **GPT-5.6 Terra**. Evaluation generation stays on
**GPT-5.6 Sol**. These defaults live in `nimble/datasets/curation_profiles.py`.
The original Sol-generated training release remains training data; changing model
roles does not make those examples suitable for a held-out evaluation.

| Stage | Model / execution |
|---|---|
| New training contexts and small factual edits | Terra, low reasoning |
| Training fact and consistency checks | Terra, medium reasoning, separate calls |
| Previously audited training rules | Reused from the completed Sol release; provenance recorded |
| Applying edits, checking spans, deriving labels, deduplication | Python |
| Fresh evaluation generation | Sol, existing evaluation pipeline |

## Faster training path

```sh
# Validate defaults, sources and audited plans without calling the API.
.venv-curator/bin/python -m nimble.datasets.fast_training_dataset --dry-run

# A separate training release; generation incurs API charges.
.venv-curator/bin/python -m nimble.datasets.fast_training_dataset \
  --target 1000 --output data/contrastive_training_terra

# Same command resumes content-addressed request caches.
# Replay a completed release without API calls:
.venv-curator/bin/python -m nimble.datasets.fast_training_dataset \
  --target 1000 --output data/contrastive_training_terra --offline

.venv-mlx/bin/python -m nimble.datasets.export_candidate_training \
  --data data/contrastive_training_terra/train_scoring.jsonl
```

The existing `create_eval_dataset` command remains the Sol evaluation path. Its
Jev annotation behavior is unchanged. This change does not regenerate evaluation
data or claim that model-generated labels are human ground truth.

### Luna and Claude Sonnet 5

The benchmark runner supports `gpt-5.6-luna` through OpenAI and
`claude-sonnet-5` through Anthropic, or explicitly through OpenRouter or Requesty. It uses the same curation prompts and acceptance
checks, with each selected model serving as both generator and verifier. Native Claude
uses JSON-schema outputs; the OpenRouter and Requesty Bedrock routes use a strict result-tool
schema with the same fields. All use the requested reasoning effort.
OpenAI and Anthropic effort names do not imply equal reasoning compute.
Dependencies are pinned in `requirements/curator.txt` (OpenAI 2.30.0 and
Anthropic 0.84.0 in the tested environment).

```sh
# Two accepted training rows per model; writes a timing report and separate datasets.
.venv-curator/bin/python -m nimble.datasets.benchmark_curation_models \
  --output data/curation_benchmark_new

# Replay those saved runs, preserving their original online timings.
.venv-curator/bin/python -m nimble.datasets.benchmark_curation_models \
  --output data/curation_benchmark_new --offline

# Explicit OpenRouter transport for Claude; exact Sonnet 5 model, Bedrock global
# endpoint, zero data retention required, no provider or model fallback.
.venv-curator/bin/python -m nimble.datasets.benchmark_curation_models \
  --models claude-sonnet-5 --claude-provider openrouter \
  --output data/curation_benchmark_sonnet_new

# Explicit Requesty transport for Claude; exact Sonnet 5 model pinned to the
# bedrock/claude-sonnet-5 route, no fallback policy.
.venv-curator/bin/python -m nimble.datasets.benchmark_curation_models \
  --models claude-sonnet-5 --claude-provider requesty \
  --output data/curation_benchmark_sonnet_requesty

# The same native-provider runner can create a larger release for either model.
.venv-curator/bin/python -m nimble.datasets.benchmark_curation_models \
  --models claude-sonnet-5 --target 1000 --output data/contrastive_training_sonnet5
```

Credentials are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, or `REQUESTY_API_KEY`, according to the explicitly selected transport, and sent
only to that service's API endpoint. OpenRouter forwards the synthetic prompts
to Amazon Bedrock; the requested model remains `anthropic/claude-sonnet-5`.
The adapter requires the returned model and provider to match, exactly one result
tool with the expected name, and a complete, non-refused response. No tool is executed.
Requesty forwards the synthetic prompts to Amazon Bedrock through the
`bedrock/claude-sonnet-5` model id. Requesty does not return a provider field, so
its adapter checks the returned model and applies the same result-tool checks.
Responses are
validated against the same Pydantic schemas before the existing acceptance gates.
Refusals and truncated responses cannot become training examples. Model/provider
provenance, response checksums, usage and timing are retained.

The default comparison uses sequential runs, seed 29, low effort for drafting,
medium for checks, an eight-word edit limit, and up to 32 candidates per required
pair. It excludes reused rule-plan creation and subsequent token export from the
timer; rejected candidate generation and verification are included. Fresh-run
reports should show zero application-cache hits. Do not interpret a resumed run's
time as a fresh-run latency.

The benchmark records an implementation fingerprint. Replay requires the same
code and settings; after implementation changes, use a fresh output directory
for new measurements. Existing release metadata and timings are preserved.

### Haiku 4.5

The native Anthropic adapter also supports the pinned `claude-haiku-4-5-20251001`
model. Haiku 4.5 uses manual extended thinking and does not accept Sonnet's adaptive
thinking or effort parameter. The runner's curation presets map `low` to 1,024
thinking tokens, `medium` to 2,048, `high` to 4,096, and `none` to disabled thinking.
These are explicit project defaults, not equivalent amounts of reasoning across
models. Native JSON-schema responses, prompts and semantic acceptance gates remain
unchanged. The default run uses low for drafting and medium for verification.

```sh
.venv-curator/bin/python -m nimble.datasets.benchmark_curation_models \
  --models claude-haiku-4-5-20251001 --claude-provider anthropic \
  --output data/curation_benchmark_haiku_new
```

Reference: [Haiku 4.5 capabilities](https://platform.claude.com/docs/en/models/haiku-4-5/overview).

The training runner processes category queues independently: a completed example
advances immediately rather than waiting for a global batch barrier. Up to 16
groups and 64 API requests can be in progress, subject to rolling request and
token limits. Defaults are 240 requests and 1,000,000 conservatively reserved
tokens per minute. Tune with `--group-concurrency`, `--concurrency`,
`--requests-per-minute`, and `--tokens-per-minute`.

Other savings come from reusing audited training-rule plans, shorter reviewer
explanations, low reasoning for drafts, and exact request caching. The prompt,
model, reasoning setting and response schema all enter each cache fingerprint.
Changed model settings require a new output directory. `--generator-model` and
`--verifier-model` allow separate models for a later controlled comparison.

## Small edits and MiniCheck-inspired checks

Each proposed pair changes one sentence and at most eight whitespace-separated
words, measured by a token diff; `--max-edit-words` adjusts this explicit limit.
The count is a mechanical locality check, not proof that only one fact changed.
Separate semantic checks still require exactly the intended factual change and a
different correct schema answer. Python applies the edit so surrounding context,
question and candidate labels stay fixed.

The new path retains the completed release's rule audits, individual-sentence and
joint-pair checks, full-context consistency checks, and deletion tests. Deleting
either required sentence must leave the focus proposition unknown. Missing
information is never automatically converted to false. As in the prior typed
release, deletions are verification cases, not additional labeled training rows.
This is an adaptation of MiniCheck's construction ideas, not a reproduction of
its complete C2D/D2C training mix or subclaim augmentation.

Only training-source rules are eligible. Source fingerprints must match the rule
release; evaluation and validation families are excluded. Every contrast pair
stays intact, and future splits should group by `source_family`. Related new
contexts do not constitute independent tasks or evaluation data.

## Measurement and limitations

The superseded two-example API trials and Terra smoke dataset have been removed.
Use the completed Luna/Sonnet releases below for data and recorded timing.

`performance.json` records wall time, completed API calls, cache hits and token
usage for that invocation. A resume overwrites this per-invocation report; each
cached response also preserves its own model, token usage and latency.
`progress.json` and `review_queue.jsonl` retain partial results. A bounded category
shortfall raises an error; no final release is advertised until its quotas pass.

Local tests cover independent progress, concurrency limits, full acceptance gates,
offline replay, cache tampering, model-specific caches, certificate reconstruction,
small edits and held-out-source rejection. These checks establish implementation
behavior, not Terra's label accuracy. A small live smoke test only establishes API
compatibility and end-to-end execution. Compare accepted pairs/minute, cost per
accepted pair and human-reviewed label errors before claiming a speedup at equal
quality. Different generation models alone do not establish evaluation independence.

## Parallel 1,000-row releases

The historical parallel run used separate 1,000-row training releases for Luna (seed 41) and native Sonnet 5
(seed 43). Each targets 500 complete contrast pairs and 100 rows per domain.
The run plan records earlier training files for an exact-overlap audit; these are
additional training data, with final export validation required before release.

The production runner reserves executor capacity for API connection setup as well
as group work. Incomplete, refused and schema-invalid responses are persisted as
candidate rejections; the same rejected request is never retried for a different
label. Sibling requests are drained in deterministic order before the next
candidate, allowing offline reconstruction. Successful caches survive interruption.
These changes affect execution and failure handling, not semantic acceptance gates.
Their implementation snapshots and configuration migrations are retained with the
parallel run records.

The fast runner also handles one unambiguous serialization error deterministically:
if the original context is a string and a model returns a plain paragraph with two
unique, root-level evidence spans, it JSON-encodes that paragraph without changing
any characters. It does not repair malformed objects/arrays, infer paths, rewrite
facts, or bypass downstream checks. Raw API responses remain in the cache. This
prevents serialization alone from discarding a candidate before semantic review.

The parallel runs enable `--guidance-from-variation 40` after observing repeated
construction failures. Beginning at that per-source attempt, generator prompts
add reminders about identity/record joins, isolated negative-sentence uncertainty,
and writing actual context rather than copying a fact-assignment table. Earlier
request specifications remain byte-identical, and verifier prompts and acceptance
gates do not change. The activation threshold, preserved checkpoints, and code
snapshot are recorded in `generation_guidance_01/migration.json` in the run report.

The subsequent `--resume-policy` files record exact per-category and per-source
checkpoint thresholds. After those thresholds, four out of five attempts favor
the source plan with the highest smoothed acceptance rate; every fifth explores
the least-attempted sources. Existing generation reminders also activate after
each source checkpoint. This reduces repeated spending on unproductive plans
without changing category quotas or acceptance criteria. Earlier request specs
were checked for equality before resuming; raw data and caches are retained.

`construction_examples_01` extends those policies for the final underfilled
categories. Future drafting prompts can include one accepted pair and context
from the same source and model in the current training run. They request a new
case with fresh observations and wording, not identifier-only substitutions.
Reference labels are excluded, and verification prompts never receive these
examples. Each exemplar's row hash and activation threshold are recorded;
earlier request specifications and accepted rows are preserved. This is a
training augmentation step, not an independent measure of label accuracy.

Sonnet's `object_container_recovery_01` additionally handles a valid JSON-encoded
paragraph returned for an object input: after the recorded source threshold,
it puts the unchanged paragraph under `context` and shifts two unique root
evidence paths to that key. Ambiguous JSON, non-root paths, and missing/repeated
spans are not repaired. All context and factual audits still run. Luna retains
its earlier implementation snapshot.

Replay a completed release against its recorded runner source with
`python -m nimble.datasets.replay_curation_snapshot DATA_DIRECTORY SNAPSHOT_JSON`.
The replay checks the snapshot hash, source and rule lineage, and unchanged gate
dependencies, then reconstructs the release from cached responses with no API
calls. This supports the two recorded implementations without relabeling a release
as if it had been generated with newer code.

After Sonnet stalled at 974 accepted rows, `cross_model_examples_01` added five
verified Luna illustrations for source plans in the remaining categories that
lacked a Sonnet exemplar. Their source plans were checked for exact equality,
and activation thresholds preserve all earlier requests. Sonnet still generates
and checks every new pair; reference labels are not supplied and verifiers never
see the illustrations. The final releases should not be treated as independent
generation samples. The donor dataset hash and row hashes are recorded.

Sonnet subsequently exhausted the original commerce-score attempt cap at 978
accepted rows. `acceptance_windows_01` raises the attempt multiplier from 32 to
64 and, after explicit checkpoint ordinals, ranks newly illustrated sources by
their attempts and successes since that prompt change. Earlier failures and
successes no longer distort those new estimates. Every-fifth exploration remains,
and prior request specifications, selection history, and semantic gates are
preserved. The completed release's manifest records the final policy.

Model reference: [GPT-5.6 Terra documentation](https://developers.openai.com/api/docs/models/gpt-5.6-terra).
Additional model/transport references: [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[Claude Sonnet 5](https://platform.claude.com/docs/en/models/sonnet-5/whats-new-sonnet-5),
[Claude thinking](https://platform.claude.com/docs/en/about-claude/models/extended-thinking-models),
[OpenRouter zero data retention](https://openrouter.ai/docs/guides/features/zdr),
and [Requesty](https://docs.requesty.ai).
Method reference: [MiniCheck sections 3.1–3.2 and Appendix H.1](https://arxiv.org/html/2404.10774v2).
