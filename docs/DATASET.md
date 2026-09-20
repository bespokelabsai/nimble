# Dataset

The repository includes the published model's training set and its frozen final
holdout. These are the canonical dataset files tracked in `data/`.

| File | Records | Purpose |
| --- | ---: | --- |
| `data/train.jsonl` | 2,676 | Training set used by the published model |
| `data/eval.jsonl` | 324 | Frozen final evaluation |
| `data/manifest.json` | Metadata | Provenance and integrity checks |

Training contains 2,676 contrastive records (1,338 base/counterfactual pairs).
Labels are synthetic and model-checked, not human-reviewed. All 34 training
source families are disjoint from the holdout's families. Contrast siblings stay
together. Do not tune hyperparameters or select checkpoints on this holdout;
derive any future tuning split from the training families only.

## Integrity

Both JSONL files retain their exact bytes, order, IDs, labels, and embedded
verification records from the original 2,676-training / 324-holdout release.

- Training SHA-256: `beadbb9b81837f7c339e090cd210ce91f65a55f2a3ada1ce9b623765b8d6fe2e`
- Holdout SHA-256: `8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896`

```sh
.venv-curator/bin/python -m nimble.training.verify_dataset
```

This offline check reconstructs evidence certificates, checks
both file hashes and counts, and rejects overlapping source families, sources,
contexts, or IDs. It makes no model or API calls.

The manifest embeds the original seed records solely as certificate provenance.
They are never passed to training or evaluation as extra examples. Historical
paths in the manifest identify lineage; no removed directory is required to
validate or train.

The trainer regenerates scoring inputs and token IDs in memory. For the pinned
Qwen3.5-9B tokenizer, the regenerated fingerprints match the previous verified
exports exactly. There are no separate scoring/token dataset copies to manage.

## Training

Use `--data-dir data --validation data/eval.jsonl` with the schema trainer.
See [training instructions](NIMBLE_TRAINING.md) for the pinned checkpoint and recipe.

These three files are committed to Git and available in a fresh checkout.
Other data directories and generated exports remain ignored. Historical experiment
reports may describe additional data that is not part of this published release.
