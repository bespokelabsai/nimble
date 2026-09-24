"""Serving extension for 255 candidates without changing legacy training prompts.

Also shipped as a standalone module in the v2 Hugging Face package.
"""
if __package__:
    from . import extended_schema, parallel_schema
else:
    import extended_schema
    import parallel_schema

MAX_CHOICES = 255
choice_key = parallel_schema.choice_key
validate_schema = extended_schema.validate_schema


def codes_for(count, tokenizer):
    if type(count) is not int or not 1 <= count <= MAX_CHOICES:
        raise ValueError(f"Candidate count must be between 1 and {MAX_CHOICES}")
    return extended_schema.codes_for(count, tokenizer)


parse_schema = extended_schema.parse_schema


def prepare_prompts(tokenizer, context, schema, max_input_tokens, system_role=True):
    validate_schema(schema)
    width = max(len(extended_schema.choices_for(field)) for field in schema.values())
    module = parallel_schema if width <= 26 else extended_schema
    return module.prepare_prompts(tokenizer, context, schema, max_input_tokens, system_role)
