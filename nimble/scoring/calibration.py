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


LATEST_ADAPTER_SHA256 = "29ef39b072dee97287947455337879c1e916705c2f727287922a2d81f5e2f20a"


V2_MODEL = "bespokelabs/Bespoke-Nimble-9B-v2"
V2_ADAPTER_SHA256 = "1bd126be997be6d9a0c25ce483ccf858c31b3422d480c33f02ba47b614be68ae"
V2_TEMPERATURE = 2.179078721266035


def resolve_temperature(model_id, revision, temperature=None, *, model_path=None,
                        adapter_sha256=None, allow_uncalibrated=False):
    """Resolve the release default; require opt-in to bypass v2's temperature.

    v2 transfers v1's temperature; it is not in FITTED_TEMPERATURES. Match the
    v2 release ID across documentation revisions, or its adapter hash recorded
    by merge_local_adapter. Never confuse base_revision with adapter identity.
    """
    import json
    import math
    from pathlib import Path

    if model_path is not None:
        ready = Path(model_path) / "READY.json"
        if ready.is_file():
            saved = json.loads(ready.read_text())
            adapter_sha256 = saved.get("adapter_sha256", adapter_sha256)
    # Weight identity takes precedence over an alias or stale caller revision.
    if adapter_sha256 == LATEST_ADAPTER_SHA256:
        model_id, revision = "bespokelabs/Bespoke-Nimble-9B", "latest-12026-unfitted"
    elif adapter_sha256 in ADAPTER_REVISIONS:
        model_id, revision = "bespokelabs/Bespoke-Nimble-9B", ADAPTER_REVISIONS[adapter_sha256]
    is_v2 = model_id == V2_MODEL or adapter_sha256 == V2_ADAPTER_SHA256
    default = V2_TEMPERATURE if is_v2 else (fitted_temperature(model_id, revision) or 1.0)
    value = default if temperature is None else temperature
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("temperature must be a positive finite number")
    if is_v2 and value == 1.0 and not allow_uncalibrated:
        raise ValueError("Bespoke-Nimble-9B-v2 defaults to T=2.179078721266035. "
                         "Raw T=1.0 requires allow_uncalibrated=True.")
    return float(value)


def served_temperature(ready):
    """Return the release temperature for the checkpoint described by READY.json.

    deploy/modal_app.py records the Hub revision. merge_local_adapter.py records only the
    adapter's SHA-256. In both files, base_revision is the Qwen base and never selects one.
    """
    revision = ready.get("revision") or ADAPTER_REVISIONS.get(ready.get("adapter_sha256"))
    model_id = ready.get("model", "bespokelabs/Bespoke-Nimble-9B")
    # Preserve v1 serving behavior for READY files written before its rename.
    if revision in ADAPTER_REVISIONS.values():
        model_id = "bespokelabs/Bespoke-Nimble-9B"
    return resolve_temperature(model_id, revision,
                               adapter_sha256=ready.get("adapter_sha256"))
