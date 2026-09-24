"""Select and verify the prompt contract saved with a model release."""
import json
from pathlib import Path

from nimble.scoring import parallel_schema


def prompt_builder(model_path, tokenizer):
    path = Path(model_path) / "schema_config.json"
    if not path.is_file():
        return parallel_schema.prepare_prompts
    from nimble.training.candidate_schema import validate_contract
    return validate_contract(json.loads(path.read_text()), tokenizer)
