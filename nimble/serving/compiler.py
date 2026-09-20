"""Use the training prompt with openjev-sglang's parallel scoring service."""

import json

from openjev.models import ChoiceQuestion, NoulQuestion
from openjev.prompts import Branch, PreparedRequest

from nimble.scoring.parallel_schema import choice_key, prepare_prompts


def serialize(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


class NimbleCompiler:
    def __init__(self, tokenizer, max_prompt_tokens=2048):
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens

    def prepare(self, request):
        schema = {}
        for name, question in request.questions.items():
            field = {"description": serialize(question.instructions)}
            if isinstance(question, NoulQuestion):
                field.update(type="boolean", choices=[False, True], choice_descriptions={
                    "false": question.criteria.no, "true": question.criteria.yes,
                })
            elif isinstance(question, ChoiceQuestion):
                field.update(type="enum", choices=list(question.criteria), choice_descriptions={
                    key: description if description is not None else key
                    for key, description in question.criteria.items()
                })
            else:
                field.update(type="enum", choices=[str(i) for i in range(len(question.criteria))],
                             choice_descriptions={str(i): text for i, text in enumerate(question.criteria)})
            schema[name] = field
        prepared = prepare_prompts(self.tokenizer, serialize(request.state), schema, self.max_prompt_tokens)
        return PreparedRequest(prepared.prefix_ids, [
            Branch(name, request.questions[name], ids, labels, [choice_key(v) for v in choices])
            for name, ids, labels, choices in zip(
                prepared.names, prepared.full_ids, prepared.candidate_ids, prepared.choices, strict=True
            )
        ])
