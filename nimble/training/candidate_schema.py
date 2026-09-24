"""Versioned training/scoring contract for up to 255 one-token candidates.

The legacy prompt module stays unchanged so existing adapters retain their hash
and their exact prompts. Wide questions use the existing extended encoder.
"""

import hashlib
from pathlib import Path

from nimble.scoring import extended_schema, parallel_schema
from nimble.scoring.parallel_schema import MODEL_ID, REVISION, SYSTEM_PROMPT, choice_key

MAX_CHOICES = 255
TASK = "schema_candidate_classification_v2"
validate_schema = extended_schema.validate_schema


def codes_for(count, tokenizer=None):
    if type(count) is not int or not 1 <= count <= MAX_CHOICES:
        raise ValueError(f"Candidate count must be between 1 and {MAX_CHOICES}")
    if count > 26 and tokenizer is None:
        raise ValueError("Wide candidates require the checkpoint tokenizer")
    return extended_schema.codes_for(count, tokenizer)


def prepare_prompts(tokenizer, context, schema, max_input_tokens, system_role=True):
    validate_schema(schema)
    width = max(len(extended_schema.choices_for(field)) for field in schema.values())
    module = parallel_schema if width <= 26 else extended_schema
    return module.prepare_prompts(tokenizer, context, schema, max_input_tokens, system_role)


def source_hashes():
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in {
        "candidate_schema.py": __file__,
        "parallel_schema.py": parallel_schema.__file__,
        "extended_schema.py": extended_schema.__file__,
    }.items()}


def encoding_contract(tokenizer):
    # Validate the full codebook at the real assistant boundary, not just alone.
    codes = codes_for(MAX_CHOICES, tokenizer)
    schema = {"decision": {"type": "enum", "description": "Select one candidate.", "choices": codes}}
    prepared = prepare_prompts(tokenizer, "Codebook validation.", schema, 32768)
    return {"task": TASK, "max_choices": MAX_CHOICES,
            "candidate_encoding": "uppercase_single_token_v1",
            "candidate_codes": codes, "candidate_token_ids": prepared.candidate_ids[0],
            "prompt_source_sha256": source_hashes(),
            "wide_system_prompt": SYSTEM_PROMPT.replace("one-letter", "short")}


def validate_contract(contract, tokenizer):
    """Reject mismatched encoders and return the checkpoint's prompt builder."""
    if contract.get("task") == "schema_candidate_classification_v1":
        if contract.get("prompt_code_sha256") != source_hashes()["parallel_schema.py"]:
            raise ValueError("Saved adapter prompt implementation differs")
        return parallel_schema.prepare_prompts
    if contract.get("task") != TASK:
        raise ValueError("Unsupported adapter candidate task")
    expected = encoding_contract(tokenizer)
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved adapter candidate encoding or prompt implementation differs")
    return prepare_prompts


def prepare_for_contract(contract, tokenizer, context, schema, max_input_tokens):
    prepare = validate_contract(contract, tokenizer)
    return prepare(tokenizer, context, schema, max_input_tokens)
