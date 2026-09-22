"""Probability temperatures fitted for exact published checkpoints."""

# Scorers compute softmax(candidate_logits / T). One scalar T never changes the top
# answer, only how sharp the probabilities are. Any other checkpoint needs its own fit.
FITTED_TEMPERATURES = {
    # Fit by minimizing log loss on 300 examples outside the training data, then accepted on
    # 300 from other sources: ECE 0.128 -> 0.066, log loss 0.692 -> 0.555, accuracy unchanged.
    ("bespokelabs/Bespoke-Nimble-9B", "93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c"): 2.179078721266035,
}

# Hub revision of each fitted adapter, keyed by the SHA-256 of adapter_model.safetensors.
# merge_local_adapter.py records this hash, not the revision; its base_revision is the Qwen base.
ADAPTER_REVISIONS = {
    "ba7e28acb97f973e80fa51f3aa6fc6f75ea4081b89632ed45d8e5f3a1d7bfa6b": "93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c",
}


def fitted_temperature(model_id, revision):
    """Return the fitted temperature for this exact checkpoint, or None if it has none."""
    return FITTED_TEMPERATURES.get((model_id, revision))
