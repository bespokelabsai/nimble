"""Probability temperatures fitted for exact published checkpoints."""

# Scorers compute softmax(candidate_logits / T). One scalar T never changes the top
# answer, only how sharp the probabilities are. Any other checkpoint needs its own fit.
FITTED_TEMPERATURES = {
    # Fit by minimizing log loss on 300 examples outside the training data, then accepted on
    # 300 from other sources: ECE 0.128 -> 0.066, log loss 0.692 -> 0.555, accuracy unchanged.
    ("bespokelabs/Bespoke-Nimble-9B", "93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c"): 2.179078721266035,
}


def fitted_temperature(model_id, revision):
    """Return the fitted temperature for this exact checkpoint, or None if it has none."""
    return FITTED_TEMPERATURES.get((model_id, revision))
