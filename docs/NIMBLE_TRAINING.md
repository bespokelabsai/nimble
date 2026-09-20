# Training Bespoke-Nimble

The committed dataset is **2,676 training examples** in `data/train.jsonl` and the
unchanged **324-example holdout** in `data/eval.jsonl`. Keep `data/manifest.json`
with them. [Dataset details](DATASET.md) explain the provenance and checksums.

## Validate the data

```sh
.venv-curator/bin/python -m nimble.training.verify_dataset
```

The loader verifies evidence certificates, frozen file hashes,
and separation of training and evaluation families. It rebuilds scoring inputs
and token IDs in memory, so old source folders and token-export files are not
needed. For the pinned 9B tokenizer, it also checks the regenerated export against
the exact fingerprint used in the prior training run.

## Train on an NVIDIA GPU

Use the CUDA/PyTorch environment described in the main README, then run:

```sh
python -m nimble.training.schema_train train \
  --model Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --data-dir data \
  --validation data/eval.jsonl \
  --output-dir .cache/runs/nimble-9b-new \
  --learning-rate 5e-5 --lora-rank 16 --seed 17 \
  --batch-size 2 --gradient-accumulation 4 --max-length 2048 \
  --max-steps 1005 --warmup-steps 101 --save-steps 335 --stop-after-epochs 1
```

This fits one epoch (335 optimizer updates) with effective batch size 8, using
the previously selected three-epoch linear learning-rate schedule. Training uses
BF16 LoRA and cross-entropy over the allowed candidate logits. It does not train
on generated reasoning or teacher probabilities. Each example's gold label stays
outside its input prompt.

The holdout is for final reporting, not checkpoint or hyperparameter selection.
For new tuning, partition only training examples by source family. Changing the
base model requires an explicit pinned `--revision`; tokenization is regenerated
for that checkpoint. Use a fresh output directory for a new run.

Outputs include adapter weights, a tokenizer, the pinned model/prompt contract,
a data audit, and before/after evaluation metrics. Upload the three canonical data
files plus the `nimble/` source tree when preparing a remote training machine.

## Score with a saved adapter

```sh
python -m nimble.training.schema_train score \
  --adapter .cache/adapters/openjeff-diverse9b-v2 \
  --context 'Our production payment service is down for every customer.' \
  --schema-file examples/parallel_schema.json
```

The saved adapter directory retains its historical name. The published
Bespoke-Nimble-9B checkpoint used the committed 2,676-example training set; the separate
local v2 checkpoint used 2,826 examples, including 150 additional cases not in this release.
Publishing the datasets does not change
these trained models. See the [published comparison](../README.md#evaluation-on-324-held-out-examples)
for the released checkpoint's results.
