# Public benchmarks

The retained holdout in `data/eval.jsonl` is 324 synthetic, model-checked examples drawn
from six source families. No person reviewed its labels. This guide adds external
benchmarks whose labels were produced by people, so the scorer can be measured against
annotations it had no part in creating, on the kinds of decisions people build with Jev.

These converters measure **agreement with human annotation**, not truth. Annotators
disagree, guidelines differ between projects, and a disagreement with a reference label
is not by itself a model error.

Bespoke-Nimble-9B and Jev 1.13.0 have been run on every subset below, 3,880 records in
total. The [results](#results) section holds the numbers; the exact selected ids and
checksums of every subset are committed under
[`docs/assets/public-benchmarks/subsets/`](assets/public-benchmarks/subsets/), so each
subset can be rebuilt byte-for-byte from the upstream files.

## What the suite covers

TypeSafe's [use-case map](https://docs.typesafe.ai/concepts/use-case-map.md) groups Jev work
into retrieval, routing, guardrails, verification, moderation, and feature extraction, and its
cookbooks fix the question shapes: a Noul per query–passage pair for re-ranking and RAG
passage classification, a supports / contradicts / not-addressed Choice for citation checks,
a Noul per policy for guardrails, a Choice over handlers for intent routing, and combined
Scores for composite scoring. The [Jev 1.13 jaggedness notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)
list where the vendor expects weakness: literal reading, numbers, dates, indirection, long
distracting state, and adversarial content.

| Jev-style task | Subsets |
| --- | --- |
| Intent and domain routing | `massive-en-US` |
| Multilingual routing | `massive-de-DE`, on the same ids |
| Yes/no over a passage | `boolq` |
| RAG answerability, evidence presence | `squad2` |
| Paraphrase and deduplication | `paws` |
| Entailment | `multinli` |
| Moderation | `civil_comments` |
| Guardrails, prompt safety | `aegis2` |
| Rubric rating | `helpsteer2` |
| Summary quality | `summeval-relevance`, `summeval-consistency` |
| Domain-specific, medical | `pubmedqa` |
| Contrastive verification | `vitaminc-dev` |

Five Choice subsets, five Noul subsets, and three Score subsets. Two, `multinli` and
`civil_comments`, carry a human label distribution, so agreement with human uncertainty
can be measured as well as agreement with the majority label.

**No vendor-published public-dataset result for Jev exists.** TypeSafe's own numbers come
from a proprietary four-workflow benchmark whose reference labels are the average of two
frontier models, not human ground truth. The only independent measurement before this
suite is the [sanand0 BANKING77 pilot](https://sanand0.github.io/llmevals/jev/), 77
requests with one per intent, which this repository reproduces generatively in
`nimble/evaluation/evaluate_banking77.py`.

## The suite

Every subset is a seeded, label-blind draw of whole families from the upstream split.
Token lengths are the exact serving prompt, including the schema and criteria, measured
with the pinned Qwen3.5-9B tokenizer over the selected records; no record exceeds the
2,048-token budget.

| Subset | Upstream | Year | License | Primitive | Selected / converted / families | State fields | Tokens median / p95 / max |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `vitaminc-dev` | [VitaminC](https://github.com/TalSchuster/VitaminC) dev | 2021 | CC BY-SA 3.0 | Choice, 3 | 599 / 63,054 / 171 | `evidence`, `claim` | 318 / 370 / 421 |
| `massive-en-US` | [MASSIVE 1.1](https://huggingface.co/datasets/AmazonScience/massive) test | 2022 | CC BY 4.0 | Choice, 18 | 350 / 2,974 / 350 | `utterance`, `locale` | 651 / 657 / 662 |
| `massive-de-DE` | MASSIVE 1.1 test, same ids | 2022 | CC BY 4.0 | Choice, 18 | 350 / 2,974 / 350 | `utterance`, `locale` | 654 / 661 / 670 |
| `boolq` | [BoolQ](https://huggingface.co/datasets/google/boolq) validation | 2019 | CC BY-SA 3.0 | Noul | 300 / 3,270 / 274 | `passage`, `question` | 326 / 480 / 582 |
| `squad2` | [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) validation | 2018 | CC BY-SA 4.0 | Noul | 299 / 11,873 / 31 | `paragraph`, `question` | 414 / 775 / 1,035 |
| `paws` | [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) labeled_final test | 2019 | Google PAWS, free for any purpose | Noul | 250 / 8,000 / 250 | `sentence_1`, `sentence_2` | 281 / 313 / 326 |
| `multinli` | [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) dev matched, original zip | 2018 | Mixed per genre, permissive | Choice, 3 | 299 / 9,815 / 103 | `premise`, `hypothesis` | 270 / 312 / 374 |
| `civil_comments` | [Civil Comments](https://huggingface.co/datasets/google/civil_comments) test | 2019 | CC0 1.0 | Noul | 300 / 97,320 / 300 | comment text | 261 / 408 / 502 |
| `aegis2` | [Nemotron Content Safety V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) test | 2025 | CC BY 4.0 | Noul | 250 / 1,927 / 250 | `user_message` | 298 / 573 / 1,356 |
| `helpsteer2` | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) validation | 2024 | CC BY 4.0 | Score, 5 | 249 / 1,034 / 125 | `prompt`, `response` | 782 / 1,383 / 1,933 |
| `summeval-relevance` | [SummEval](https://huggingface.co/datasets/mteb/summeval), unpacked | 2020 | MIT | Score, 5 | 240 / 1,600 / 15 | `article`, `summary` | 761 / 1,071 / 1,104 |
| `summeval-consistency` | SummEval, unpacked | 2020 | MIT | Score, 5 | 144 / 1,600 / 9 | `article`, `summary` | 721 / 1,069 / 1,093 |
| `pubmedqa` | [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) pqa_labeled | 2019 | MIT | Choice, 3 | 250 / 1,000 / 250 | `question`, `abstract_context` | 563 / 734 / 873 |

How each subset was formed:

- **Families are never split.** A family is the natural group in the upstream data: the
  `case_id` of a VitaminC revision, the MASSIVE `id` shared across locales, the passage in
  BoolQ, the paragraph in SQuAD 2.0, the `promptID` in MultiNLI, the prompt in HelpSteer2,
  and the article in SummEval. PAWS, Civil Comments, Aegis, and PubMedQA rows are
  independent, so each is its own family.
- **`massive-de-DE` is built with `--ids-from`** against the en-US manifest, so it holds
  exactly the same 350 MASSIVE ids in German. Record ids do not embed the locale, which is
  what lets `compare_public` join the two runs as a paired multilingual comparison.
- **SQuAD 2.0 families are paragraphs of about ten questions**, every one of which mixes
  answerable and unanswerable questions. Family integrity was kept over topic spread, so
  the 299 records cover 31 paragraphs.
- **SummEval families are articles of 16 machine summaries**, so 240 relevance records
  come from 15 articles and 144 consistency records from 9. Consistency is deliberately
  smaller: 84% of its records sit at the top level, which makes it a weak discriminator.
- **Civil Comments is about 11% positive**, close to the 8.0% rate of the test split.
  Label-blind selection preserves the skew.
- **Aegis** keeps only rows whose prompt label came from a human annotator, which is all
  1,964 test rows, and skips the 36 whose prompt text was redacted. **MultiNLI** drops the
  185 dev pairs with no majority label. **HelpSteer2** and **Aegis** were the only subsets
  where the tokenizer filter removed anything: 4 and 1 rows.
- **VitaminC** is contrastive by construction: one `case_id` groups the siblings of one
  Wikipedia revision, typically four rows, with near-identical evidence and different
  labels. On the claim `Dragon Con had less than 1000 guests .`, the evidence "Among the
  more than **512** guests …" is `NOT ENOUGH INFO` and "… more than **6000** guests …" is
  `REFUTES`. That is the external, human-labeled analogue of this repository's own
  contrastive curation.

## Get the data

`data/` is gitignored. Raw downloads and converted sets stay local and are never
committed; only ids, checksums, and aggregates are.

Hugging Face datasets are the parquet export of each repository, written to JSON Lines with
the original field names; the converters skip what they do not need.

| Dataset | Source and split | Local file |
| --- | --- | --- |
| VitaminC | `https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip`, dev | `data/public/raw/vitaminc/dev.jsonl`, 63,054 rows |
| MASSIVE 1.1 | `https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz` | `data/public/raw/massive/amazon-massive-dataset-1.1.tar.gz`; read in place, test partition, 2,974 rows per locale |
| BoolQ | `google/boolq`, validation | `data/public/raw/boolq/validation.jsonl`, 3,270 rows |
| SQuAD 2.0 | `rajpurkar/squad_v2`, validation | `data/public/raw/squad_v2/validation.jsonl`, 11,873 rows |
| PAWS | `google-research-datasets/paws`, labeled_final test | `data/public/raw/paws/test.jsonl`, 8,000 rows |
| MultiNLI | `https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip`, dev matched | `data/public/raw/multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl`, 10,000 rows |
| Civil Comments | `google/civil_comments`, test | `data/public/raw/civil_comments/test.jsonl`, 97,320 rows |
| Nemotron Content Safety V2 | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`, test | `data/public/raw/aegis2/test.jsonl`, 1,964 rows |
| HelpSteer2 | `nvidia/HelpSteer2`, validation | `data/public/raw/helpsteer2/validation.jsonl`, 1,038 rows |
| SummEval | `mteb/summeval`, test | `data/public/raw/summeval/test.jsonl`, 100 articles with 16 summaries each |
| PubMedQA | `qiaojin/PubMedQA`, pqa_labeled | `data/public/raw/pubmedqa/train.jsonl`, 1,000 rows |

MultiNLI comes from the original NYU zip because the Hugging Face copy drops
`annotator_labels`, the five votes that make the label distribution. MASSIVE is a script
dataset on Hugging Face with no parquet export, so it comes from Amazon's archive. The
VitaminC zip is JSON Lines already. The manifest written at conversion time stores the
checksum of the exact upstream file it read.

## Convert

Each dataset is one module under `nimble/datasets/public_sources/`, discovered by
`nimble/datasets/public_benchmarks.py` at run time. A module defines `NAME`,
`SOURCE_URL`, `LICENSE`, `NOTE`, `RELEASE_YEAR`, and `SUBSETS`, a `rows(path, subset)`
reader over the downloaded file, and a `record(raw, subset)` converter that
builds one record with `public_records.record_for` or returns `None` to skip a row. A
module missing any part of that contract fails every build with an error naming the
module and the missing piece. Adding a dataset is adding a module; no registry is edited.

The subsets were built with exactly these commands, with `$V` a Python that has
`transformers` (needed only for `--tokenizer`), `$TOK` the merged checkpoint directory,
and `$R` `data/public/raw`:

```sh
$V -m nimble.datasets.public_benchmarks --dataset vitaminc --source $R/vitaminc/dev.jsonl --output-dir data/public/vitaminc-dev --limit 600
$V -m nimble.datasets.public_benchmarks --dataset massive --source $R/massive/amazon-massive-dataset-1.1.tar.gz --output-dir data/public/massive-en-US --subset en-US --limit 350
$V -m nimble.datasets.public_benchmarks --dataset massive --source $R/massive/amazon-massive-dataset-1.1.tar.gz --output-dir data/public/massive-de-DE --subset de-DE --ids-from data/public/massive-en-US/manifest.json
$V -m nimble.datasets.public_benchmarks --dataset boolq --source $R/boolq/validation.jsonl --output-dir data/public/boolq --limit 300
$V -m nimble.datasets.public_benchmarks --dataset squad2 --source $R/squad_v2/validation.jsonl --output-dir data/public/squad2 --limit 300
$V -m nimble.datasets.public_benchmarks --dataset paws --source $R/paws/test.jsonl --output-dir data/public/paws --limit 250
$V -m nimble.datasets.public_benchmarks --dataset multinli --source $R/multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl --output-dir data/public/multinli --limit 300
$V -m nimble.datasets.public_benchmarks --dataset civil_comments --source $R/civil_comments/test.jsonl --output-dir data/public/civil_comments --limit 300
$V -m nimble.datasets.public_benchmarks --dataset aegis2 --source $R/aegis2/test.jsonl --output-dir data/public/aegis2 --limit 250 --tokenizer $TOK
$V -m nimble.datasets.public_benchmarks --dataset helpsteer2 --source $R/helpsteer2/validation.jsonl --output-dir data/public/helpsteer2 --limit 250 --tokenizer $TOK
$V -m nimble.datasets.public_benchmarks --dataset summeval --source $R/summeval/test.jsonl --output-dir data/public/summeval-relevance --subset relevance --limit 250 --tokenizer $TOK
$V -m nimble.datasets.public_benchmarks --dataset summeval --source $R/summeval/test.jsonl --output-dir data/public/summeval-consistency --subset consistency --limit 150 --tokenizer $TOK
$V -m nimble.datasets.public_benchmarks --dataset pubmedqa --source $R/pubmedqa/train.jsonl --output-dir data/public/pubmedqa --limit 250
```

Rebuilding with identical parameters is byte-identical; the VitaminC subset, first built
before the converters were split into modules, rebuilds to the same 599 records and the
same `dataset_sha256`, `6144a345cb8846b6594f07a0037b1221b2d9633b20d8a9bfd14f9111d15965bc`.

| Flag | Meaning |
| --- | --- |
| `--dataset` | A discovered module name |
| `--source` | The downloaded upstream file (for MASSIVE, the tarball), as the module expects |
| `--output-dir` | Receives `all.jsonl` and `manifest.json`; a directory holding different content is refused |
| `--subset` | Required when the module defines `SUBSETS`: MASSIVE locales, SummEval dimensions; refused otherwise |
| `--seed` | Selection seed; defaults to `20260918` |
| `--limit` | Approximate record cap. Whole families only, so the count lands at or below it |
| `--ids-from` | Another build's `manifest.json`; select exactly its ids instead of sampling. Refused with `--limit`, and refused if any listed family would be only partially covered |
| `--tokenizer`, `--max-input-tokens` | Length filter on the exact serving prompt, default budget 2048. Without a tokenizer no filter is applied |

The converter emits this repository's existing evaluation record contract: `id`, `split`,
`domain`, `family`, `source_family`, an `input` holding only `state` and `questions`, and a
`reference` holding the human label and, for `multinli` and `civil_comments`, a
`distribution` over the same option keys. Reference material never reaches the model
input. The check is structural rather than textual, because a target string such as
`SUPPORTS` is legitimately one of the offered option keys in every prompt: `record_for`
walks the assembled input and rejects it if any reference-bearing key is reachable at any
depth. For SQuAD 2.0 the answer text and offsets are kept under `reference.answer_texts`
only; for PubMedQA the `long_answer`, which states the decision outright, never enters
the record.

The manifest records `source_url`, `license`, `release_year`, and `note`; `source_sha256`
of the raw upstream and `dataset_sha256` of the written payload; `seed`, `limit`, and
`ids_from`; the `converted`, `skipped`, `dropped_over_length`, `count`, and `families`
tallies; `types`, `labels`, and `domains` of the selection; the selection rule in prose;
and every selected `id`. Written lines escape U+2028, U+2029, and U+0085, and the readers
split on newline only, because real abstracts and prompts contain those characters.

## Prompt design

The `instructions` and `criteria` are the fixed free parameter of the benchmark. They
were written once, from TypeSafe's guidance on stating the exact condition and describing
concrete situations, before any model was run, and have not been tuned against any
split. Both models see identical wording; changing it invalidates comparison with
earlier runs. The exact criteria are the `CRITERIA` or `LEVELS` constants in each
`nimble/datasets/public_sources/<name>.py`, which are the source of truth. Noul criteria
are keyed `false` and `true` because `adapt_input` forwards them as candidate descriptions.

| Subset | State fields | Instructions |
| --- | --- | --- |
| `vitaminc` | `evidence`, `claim` | Decide how the evidence bears on the claim. Judge only from the evidence text, and not from outside knowledge about the subject. |
| `massive` | `utterance`, `locale` | Which assistant domain should handle this utterance? Judge from the request itself, not from how it is phrased. |
| `boolq` | `passage`, `question` | Does the passage answer the question with yes? Use only what the passage states or directly implies. |
| `squad2` | `paragraph`, `question` | Does the paragraph contain the information needed to answer the question? Treat a question as unanswerable when the paragraph discusses the topic but does not state the specific fact asked for. |
| `paws` | `sentence_1`, `sentence_2` | Do the two sentences have the same meaning? Sentences that reuse the same words in a different order can still mean different things. |
| `multinli` | `premise`, `hypothesis` | Assume the premise is true. How does the hypothesis relate to it? |
| `civil_comments` | the comment text | Is this comment toxic, meaning rude, disrespectful, or unreasonable enough that a reader would likely leave the discussion? |
| `aegis2` | `user_message` | Would a general-purpose assistant be unsafe to comply with this user message as written? Judge the request, not the assistant's likely reply. |
| `helpsteer2` | `prompt`, `response` | How helpful is the response to the prompt, considering whether it addresses what was asked, is correct, and is complete? |
| `summeval`, consistency | `article`, `summary` | How factually consistent is the summary with the article? Every statement in the summary should be supported by the article. |
| `summeval`, relevance | `article`, `summary` | How well does the summary capture the important content of the article, without unimportant or redundant material? |
| `pubmedqa` | `question`, `abstract_context` | Based only on the abstract, what is the answer to the research question? |

Three details of the targets: Civil Comments' target is `toxicity >= 0.5`, the Jigsaw
convention, with `reference.distribution` set to `{"true": toxicity, "false": 1 - toxicity}`;
Aegis' `true` criterion spells out the dataset card's twelve unsafe categories; SummEval's
target is the three-expert mean rounded half up to a level, which never falls exactly
halfway, with the raw mean kept as `reference.expert_mean` and no distribution because the
individual votes are not published.

## Evaluate

```sh
python -m nimble.evaluation.evaluate_public \
  --data data/public/<subset>/all.jsonl \
  --output-dir evaluations/public/<subset>/nimble-9b \
  --model-path <merged checkpoint directory> \
  --model-id bespokelabs/Bespoke-Nimble-9B \
  --revision <resolved revision> \
  --backend cuda
```

| Flag | Meaning |
| --- | --- |
| `--backend` | `mlx` for Apple Silicon, `cuda` for a PyTorch GPU, which includes ROCm |
| `--device-map`, `--max-gpu-memory` | CPU offload for a checkpoint larger than the GPU; `cuda` only |
| `--dtype`, `--attention` | Override BF16 and `sdpa`; `cuda` only. Recorded in the manifest |
| `--resume` | Continue an interrupted run after checking its manifest |
| `--shuffle-seed` | Permutes candidate order per record, to measure A-Z position bias |
| `--temperature` | Defaults to 1.0. No temperature has been fitted |
| `--max-input-tokens` | Defaults to 2048, the model contract's limit |

**No Jev annotation is required.** Every pre-existing runner in this repository demands a
`teacher` field on each record and raises without one, which is what prevented public
human-labeled datasets from being evaluated at all. This runner consults no teacher and
refuses records that carry one. Scorer backends are imported lazily, so the module loads
for offline analysis without MLX or PyTorch installed.

Outputs are `rows.jsonl` with one scored record each, `summary.json` with the aggregate
report, and `manifest.json` with the dataset and script checksums, model identity,
backend, seed, and version record. Writing into a directory that already holds results is
refused unless `--resume` is passed. A resumed run checks the dataset checksum, model,
revision, backend, and every scorer option against the saved manifest, refuses on any
difference, and scores only the records not yet in `rows.jsonl`. The runner raises on an
overlength prompt rather than truncating.

### Running the 9B checkpoint on a 16 GiB GPU

The BF16 checkpoint is about 18 GB, so it does not fit a 16 GiB card. `--device-map auto`
with `--max-gpu-memory` loads through `accelerate`, keeps as many layers on the GPU as
fit, and streams the rest from system memory on every forward pass. `--dtype` and
`--attention` override the default BF16 and `sdpa` settings when a GPU needs them. All
four options are recorded in `manifest.json`, together with the resulting layer placement,
the device of the output head, and the torch, CUDA or HIP, and GPU identity. The results
below were produced this way on an AMD Radeon RX 6800 XT (`gfx1030`, 16 GiB) with torch
2.13.0+rocm7.1: `--max-gpu-memory 13GiB` placed 22 of the 37 layers on the GPU with the
output head on the CPU, peak GPU allocation was 10.96 GiB, and BF16 matrix products are
emulated on that GPU at about half its FP32 rate. BF16 was kept so that the logits match
the H100 reference rather than introducing a precision confound.

The merge that produces the checkpoint directory accepts a local base snapshot:

```sh
python -m nimble.scoring.merge_local_adapter \
  --adapter <downloaded adapter directory> \
  --base <downloaded base snapshot> \
  --output <merged checkpoint directory>
```

Without `--base`, the merge expects the Hugging Face hub cache layout. It checks the
adapter's `schema_config.json` against this checkout's prompt code before it starts.

The suite runs one subset at a time with the same flags:

```sh
for d in vitaminc-dev massive-en-US massive-de-DE boolq squad2 paws multinli \
         civil_comments aegis2 helpsteer2 summeval-relevance summeval-consistency pubmedqa; do
  python -m nimble.evaluation.evaluate_public \
    --backend cuda \
    --model-path <merged checkpoint directory> \
    --model-id bespokelabs/Bespoke-Nimble-9B \
    --revision 594dfdcfb6f94e3d0c0db7535180d3c71689169a \
    --device-map auto --max-gpu-memory 13GiB \
    --data data/public/$d/all.jsonl \
    --output-dir evaluations/public/$d/nimble-9b
done
```

### Position bias

Candidate answers are encoded as single-token letter codes `A`, `B`, `C`, and so on, so
the order in which options are presented is part of the prompt. `--shuffle-seed` permutes
each record's candidates reproducibly, using the same `random.Random(f"{seed}:{row_id}")`
pattern as `training/schema_data.py`. Correctness is invariant under shuffling by
construction, because probabilities are keyed by candidate value rather than by position;
what changes is `selected_code_counts` in the summary. No shuffled run has been made yet,
so position bias is unmeasured.

## Metrics

`summary.json` reports, split by primitive type, by `domain`, and by `split`, everything
`evaluate_pilot.assess` already computes: accuracy, negative log likelihood of the
reference, multiclass Brier, binary Brier for Noul, and the expected level and absolute
level error for Score. Two additions:

**Expected calibration error.** Ten equal-width bins over the top candidate probability
against correctness, following the binning convention in
`nimble/evaluation/evaluate_banking77.py`; the test suite asserts equality against that
implementation on shared data.

**Distributional agreement.** Applied only when a record carries
`reference.distribution`: `jensen_shannon_bits` (base 2, bounded in [0, 1]),
`total_variation` (bounded in [0, 1]), and `human_entropy_bits`. `multinli` carries a
five-vote distribution and `civil_comments` a rater fraction; read them with the
[caveats](#interpreting-these-numbers) on what each distribution is.

## Jev

```sh
python -m nimble.evaluation.evaluate_public_jev \
  --data data/public/<subset>/all.jsonl \
  --output-dir evaluations/public/<subset>/jev-1.13.0 \
  --concurrency 4
```

The runner reads `TYPESAFE_API_KEY` from the environment or, via `python-dotenv`, from a gitignored `.env`
file. Each request carries exactly the record's `state` and `questions`; the reference
label is joined after the response arrives. Responses are validated with
`dataset_io.validate_teacher`. Invalid responses become error rows that count as
incorrect and are excluded from the probabilistic metrics. Runs append as they go and
resume by skipping completed IDs. Rejected credentials abort the run.

## Compare

```sh
python -m nimble.evaluation.compare_public \
  --runs nimble=evaluations/public/<subset>/nimble-9b/rows.jsonl \
         jev=evaluations/public/<subset>/jev-1.13.0/rows.jsonl \
  --output-dir evaluations/public/<subset>/comparison
python -m nimble.evaluation.summarize_public_suite \
  --root evaluations/public --runs nimble-9b jev-1.13.0
```

The comparison joins runs by ID, refuses runs whose ID sets or reference labels differ,
and writes `comparison.json` and `REPORT.md` with accuracy and Wilson intervals, ECE,
family-complete accuracy, agreement, and an exact McNemar test on discordant records.
Because `massive-en-US` and `massive-de-DE` share ids and labels, the same join pairs one
model's English run with its German run. The suite summary walks every subset directory
under the root and writes one suite-level table, `suite.json`, plus a Markdown rendering.

## Results

Both models answered identical records with identical human reference labels, the
criteria wording above, and no candidate shuffling. The tables below are the report;
the per-subset `comparison.json` and suite `suite.json` they were read from are run
outputs under the gitignored `evaluations/public/` and are regenerated by the commands
above.

### Accuracy

| Subset | Type | n | Families | nimble-9b accuracy (95% CI) | jev-1.13.0 accuracy (95% CI) | Agreement | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|
| aegis2 | noul | 250 | 250 | 81.2% (75.9%–85.6%) | 80.4% (75.0%–84.8%) | 85.6% | 0.8679 |
| boolq | noul | 300 | 274 | 86.0% (81.6%–89.5%) | 89.7% (85.7%–92.6%) | 89.7% | 0.0708 |
| civil_comments | noul | 300 | 300 | 70.3% (64.9%–75.2%) | 81.0% (76.2%–85.0%) | 85.3% | 0.0000 |
| helpsteer2 | score | 249 | 125 | 39.0% (33.1%–45.1%) | 34.1% (28.5%–40.2%) | 26.1% | 0.3232 |
| massive-de-DE | choice | 350 | 350 | 83.4% (79.2%–87.0%) | 86.9% (82.9%–90.0%) | 91.7% | 0.0169 |
| massive-en-US | choice | 350 | 350 | 86.9% (82.9%–90.0%) | 87.4% (83.5%–90.5%) | 93.7% | 0.7905 |
| multinli | choice | 299 | 103 | 85.3% (80.8%–88.9%) | 82.9% (78.3%–86.8%) | 81.3% | 0.4101 |
| paws | noul | 250 | 250 | 82.8% (77.6%–87.0%) | 89.2% (84.7%–92.5%) | 89.6% | 0.0025 |
| pubmedqa | choice | 250 | 250 | 75.6% (69.9%–80.5%) | 77.2% (71.6%–82.0%) | 83.2% | 0.6177 |
| squad2 | noul | 299 | 31 | 80.6% (75.7%–84.7%) | 82.9% (78.3%–86.8%) | 90.3% | 0.2649 |
| summeval-consistency | score | 144 | 9 | 75.7% (68.1%–82.0%) | 81.2% (74.1%–86.8%) | 75.0% | 0.1849 |
| summeval-relevance | score | 240 | 15 | 49.2% (42.9%–55.5%) | 35.0% (29.2%–41.2%) | 47.9% | 0.0004 |
| vitaminc-dev | choice | 599 | 171 | 76.6% (73.1%–79.8%) | 80.1% (76.8%–83.1%) | 87.5% | 0.0125 |

### Probabilities and families

| Subset | nimble-9b families complete | jev-1.13.0 families complete | nimble-9b ECE | jev-1.13.0 ECE | nimble-9b Brier | jev-1.13.0 Brier | nimble-9b score MAE | jev-1.13.0 score MAE | nimble-9b JSD / TVD | jev-1.13.0 JSD / TVD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aegis2 | 203/250 | 201/250 | 0.102 | 0.048 | 0.289 | 0.267 | — | — | — | — |
| boolq | 233/274 | 244/274 | 0.109 | 0.038 | 0.245 | 0.170 | — | — | — | — |
| civil_comments | 211/300 | 243/300 | 0.133 | 0.056 | 0.413 | 0.282 | — | — | 0.176 / 0.292 | 0.126 / 0.240 |
| helpsteer2 | 22/125 | 19/125 | 0.343 | 0.261 | 0.894 | 0.844 | 0.962 | 0.967 | — | — |
| massive-de-DE | 292/350 | 304/350 | 0.079 | 0.071 | 0.256 | 0.215 | — | — | — | — |
| massive-en-US | 304/350 | 306/350 | 0.061 | 0.075 | 0.216 | 0.204 | — | — | — | — |
| multinli | 65/103 | 59/103 | 0.083 | 0.061 | 0.211 | 0.228 | — | — | 0.104 / 0.182 | 0.100 / 0.186 |
| paws | 207/250 | 223/250 | 0.100 | 0.041 | 0.280 | 0.174 | — | — | — | — |
| pubmedqa | 189/250 | 193/250 | 0.173 | 0.129 | 0.407 | 0.344 | — | — | — | — |
| squad2 | 7/31 | 11/31 | 0.139 | 0.042 | 0.314 | 0.224 | — | — | — | — |
| summeval-consistency | 1/9 | 0/9 | 0.177 | 0.079 | 0.421 | 0.267 | 0.569 | 0.468 | — | — |
| summeval-relevance | 0/15 | 0/15 | 0.102 | 0.232 | 0.678 | 0.774 | 0.698 | 0.770 | — | — |
| vitaminc-dev | 89/171 | 100/171 | 0.172 | 0.104 | 0.391 | 0.313 | — | — | — | — |

### Averages

| Group | Subsets | nimble-9b macro | nimble-9b micro | jev-1.13.0 macro | jev-1.13.0 micro |
|---|---:|---:|---:|---:|---:|
| all | 13 | 74.8% | 75.9% | 76.0% | 77.3% |
| choice | 5 | 81.6% | 81.1% | 82.9% | 82.8% |
| noul | 5 | 80.2% | 80.1% | 84.6% | 84.6% |
| score | 3 | 54.6% | 51.2% | 50.1% | 45.2% |

### Paired multilingual: MASSIVE en-US and de-DE on identical ids

| Run | massive-en-US | massive-de-DE | Locale agreement | Right in both | Right only massive-en-US | Right only massive-de-DE | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| nimble-9b | 86.9% (82.9%–90.0%) | 83.4% (79.2%–87.0%) | 91.7% | 286 | 18 | 6 | 0.0227 |
| jev-1.13.0 | 87.4% (83.5%–90.5%) | 86.9% (82.9%–90.0%) | 93.7% | 295 | 11 | 9 | 0.8238 |

### Reading the suite

Over the thirteen subsets, Bespoke-Nimble-9B matched 74.8% of human labels as a macro
average and 75.9% pooled over records; Jev 1.13.0 matched 76.0% and 77.3%. That is a gap
of 1.2 macro points in Jev's favour across tasks that are all outside Nimble's training
categories. By task type, Jev leads on the five Noul subsets and slightly on the five
Choice subsets; Nimble leads on the three Score subsets, the rubric-rating tasks.

Five per-subset gaps have an exact McNemar p below 0.05. Four favour Jev:
`civil_comments`, `paws`, `massive-de-DE`, and `vitaminc-dev`. One favours Nimble:
`summeval-relevance`. The other eight subsets do not separate the models on this evidence.

The paired multilingual block holds the utterances fixed and changes only the language.
Switching the same 350 MASSIVE utterances from English to German costs Nimble 3.5 points:
it is right in English but wrong in German on 18 utterances and the reverse on 6,
p = 0.023. Jev loses 0.5 points, 11 against 9, p = 0.82.

On probabilities, Jev has the lower expected calibration error on 11 of the 13 subsets,
with Nimble lower on `massive-en-US` and `summeval-relevance`, and the lower Brier score
on 10 of 13. On the two subsets that carry a human label distribution the two models are
close. No temperature has been fitted for either model, so these describe each model as
shipped.

Exact-level accuracy on a five-level rubric is low for both models and they agree on only
a quarter to a half of those records. Read the score MAE column instead, the absolute
error of the probability-weighted level, which does not depend on hitting the exact
level: it is at parity on `helpsteer2`, better for Nimble on `summeval-relevance`, and
better for Jev on `summeval-consistency`.

**Jev's NLL is not comparable.** Jev's API returns each probability rounded to two
decimals, so an answer the model considered unlikely arrives as exactly `0.00`. On
VitaminC the human label received `0.00` on 19 of 599 records; `evaluate_pilot.assess`
clips a zero to 1e-15 before taking the log, which adds about 34.5 to each such record.
Nimble's probabilities are unrounded. The metric is left as computed in the comparison
files; Brier and ECE are bounded and are the fairer probabilistic comparison.

**Latencies are not compared.** Nimble ran on a consumer GPU with 15 layers streamed from
system memory and reference kernels for Qwen3.5's linear-attention layers, at about 1.1 s
per VitaminC record; Jev ran through a hosted API at a median of 0.66 s per request
including transport. Neither is a serving benchmark; for Nimble's latency on suitable
hardware see the README's [observed inference latency](../README.md#observed-inference-latency).

Two caveats specific to this suite come on top of the ones below. The Score subsets draw
on very few families, 15 articles for `summeval-relevance` and 9 for
`summeval-consistency`, so the SummEval gaps are fragile despite their p-values, which
treat records as independent when the records of one article are not. And
`civil_comments` is 89% negative, so its accuracy is dominated by how each model handles
the toxic minority: 6 records are right only for Nimble and 38 only for Jev.

## Rejected datasets

Every candidate was checked for a reachable download, a stated license, and the actual
field names before it was accepted. What did not pass:

| Dataset | Reason |
| --- | --- |
| ChaosNLI | The only distribution, a Dropbox archive, has been deleted and is not mirrored; CC BY-NC 4.0 |
| XNLI, ANLI, Financial PhraseBank, WiC, BeaverTails, ToxicChat | CC BY-NC licenses |
| Yelp review full, AG News | Academic-only terms; card states non-commercial |
| CLERC, MS MARCO, IMDb, SST-5, TREC | No license stated on the card or repository. CLERC is the dataset in TypeSafe's own re-ranking cookbook |
| dair-ai/emotion, DBpedia-14 | Labels derived from hashtags or an ontology, not from annotators |
| Nemotron Content Safety V2 response labels | Mix human and `llm_jury` sources; only the prompt labels, all human, are used |
| HelpSteer3 preference, RewardBench 2 | Pairwise shape; HelpSteer3 also far over the prompt budget |
| MMLU, MMLU-Pro, ARC | Knowledge multiple choice, not a Jev use case, and heavily contaminated |
| LLM-AggreFact, WildGuardMix | Gated behind a Hugging Face token; LLM-AggreFact is also CC BY-ND. Both are candidates once a token is configured |
| CLINC150 with out-of-scope | The ten-domain grouping exists only in the paper, so the 15-way-per-domain framing needs a hand-transcribed map |
| CaseHOLD, TrustAIRLab jailbreak prompts, STS-B, WinoGrande | Verified and usable; held as optional extensions |

BANKING77 and CLINC150 have 77 and 150 intents, more than the 26 single-token codes the
scorer supports; MASSIVE's 18 scenarios are the routing level used instead, and its 59
intents are kept under `reference` for analysis.

## Interpreting these numbers

- **These are transfer results.** Nimble was trained on ten subject categories with
  Choice, Noul, and Score questions. None of the tasks above is among them, and every
  criteria wording was written for this benchmark, not seen in training.
- **Contamination.** Every dataset except Nemotron Content Safety V2 (2025) and
  HelpSteer2 (2024) predates Qwen3.5 pretraining and is plausibly in it. Whether any is
  present in Jev's training is unknown. Absolute accuracy may therefore be inflated for
  either model or both. Weight the 2024–2025 subsets and the adversarial ones, PAWS,
  SQuAD 2.0's unanswerables, and VitaminC, when arguing about generalization.
- **Label distributions are not all the same thing.** MultiNLI's distribution is five
  votes, and 58.7% of dev pairs are unanimous, so the disagreement signal lives in the
  other 41%. Civil Comments' distribution is the published rater fraction from an
  unpublished number of raters. SummEval publishes only the three-expert mean, so no
  distribution is emitted. The other subsets are single-gold.
- **Few families means wide uncertainty.** Records within a family are not independent,
  so the Wilson intervals, which treat records as independent, are narrower than the
  family structure warrants on SummEval and SQuAD 2.0.
- **Class skew.** Label-blind selection reproduces each dataset's priors: Civil Comments
  is 11% toxic, SummEval consistency is 84% top-level, PubMedQA is 53/32/15. Read
  per-class figures and Brier alongside accuracy, and do not rebalance after seeing labels.
- **One subset, one seed.** Each subset is one seeded draw. A different seed gives a
  different subset, and no second seed has been run.
- **Criteria wording is a free parameter.** Results move with how the instructions and
  option descriptions are phrased. They were not tuned against any split.
- **Synthetic revisions are author-written.** VitaminC's `vitaminc-synthetic` domain holds
  perturbations written for the dataset, not naturally occurring Wikipedia edits; on this
  subset Nimble matched 198 of 259 real and 261 of 340 synthetic labels, Jev 212 and 268.
## Tests

All suites are offline and need no model or network. The first three run without
third-party packages; `test_cuda_scorer` needs torch and transformers, as it did before.

```sh
python3 -m unittest tests.test_public_sources tests.test_public_benchmarks tests.test_evaluate_public
python -m unittest tests.test_cuda_scorer   # torch environment
```

`test_public_sources` checks every source module on a tiny fixture: a valid record that
round-trips the schema validator, no reference-bearing key reachable in the input, family
keys, the skip rules, identical MASSIVE ids across locales, MultiNLI and Civil Comments
distributions, SummEval unpacking and rounding, and the SQuAD and MASSIVE readers.
`test_public_benchmarks` covers discovery and the contract error, label-blind
family-whole selection, `--ids-from`, the overwrite guard, and the newline-only readers.
`test_evaluate_public` covers the divergence and ECE arithmetic, teacher-free evaluation,
shuffle invariance, resume, the Jev request allowlist, retry, and error paths, the
comparison statistics, and the suite aggregation. `test_cuda_scorer` covers the candidate
projection and the offloaded output head.
