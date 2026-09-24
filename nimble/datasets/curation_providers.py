"""Explicit API transports for the same curation prompts and schemas."""
import os
from types import SimpleNamespace


def provider_for(model, claude_provider='anthropic'):
    if claude_provider not in ('anthropic', 'openrouter', 'requesty'):
        raise ValueError('Unsupported Claude provider: ' + claude_provider)
    if model.startswith('gpt-'):
        return 'openai'
    if model.startswith('claude-'):
        return claude_provider
    raise ValueError('Unsupported curation model: ' + model)


def anthropic_request(spec):
    effort = spec['reasoning_effort']
    if effort not in ('none', 'low', 'medium', 'high'):
        raise ValueError('Unsupported Claude effort: ' + effort)
    output = {'format': {'type': 'json_schema',
                         'schema': spec['response_format']['json_schema']['schema']}}
    if spec['model'] in ('claude-haiku-4-5', 'claude-haiku-4-5-20251001'):
        # Haiku 4.5 supports manual thinking, not adaptive thinking/effort.
        # These are our explicit curation presets, not Anthropic effort equivalents.
        budget = {'none': 0, 'low': 1024, 'medium': 2048, 'high': 4096}[effort]
        if budget and budget >= spec['max_completion_tokens']:
            raise ValueError('Haiku thinking budget must be smaller than max completion tokens')
        return {'model': spec['model'], 'messages': spec['messages'],
                'max_tokens': spec['max_completion_tokens'], 'output_config': output,
                'thinking': {'type': 'enabled', 'budget_tokens': budget} if budget else {'type': 'disabled'}}
    if effort != 'none':
        output['effort'] = effort
    return {'model': spec['model'], 'messages': spec['messages'],
            'max_tokens': spec['max_completion_tokens'], 'output_config': output,
            'thinking': {'type': 'disabled' if effort == 'none' else 'adaptive'}}


def normalize_anthropic(result):
    usage = result.usage.model_dump()
    inputs = sum(usage.get(k, 0) or 0 for k in ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'))
    outputs = usage.get('output_tokens', 0) or 0
    normalized_usage = {'prompt_tokens': inputs, 'completion_tokens': outputs,
                        'total_tokens': inputs + outputs, 'anthropic_usage': usage}
    return SimpleNamespace(model=result.model,
        usage=SimpleNamespace(model_dump=lambda: normalized_usage),
        choices=[SimpleNamespace(
            finish_reason='stop' if result.stop_reason == 'end_turn' else result.stop_reason,
            message=SimpleNamespace(refusal='refused' if result.stop_reason == 'refusal' else None,
                content=''.join(block.text for block in result.content if block.type == 'text')))])


def openrouter_request(spec):
    """Pin the requested Claude model to a zero-retention Bedrock endpoint."""
    effort = spec['reasoning_effort']
    if effort not in ('none', 'low', 'medium', 'high'):
        raise ValueError('Unsupported Claude effort: ' + effort)
    schema = spec['response_format']['json_schema']
    # Bedrock's OpenRouter endpoint supports tool schemas, but does not advertise
    # the response_format=json_schema capability. The tool only carries a result;
    # it never executes code or performs an external action.
    return {'model': 'anthropic/' + spec['model'], 'messages': spec['messages'],
            'max_tokens': spec['max_completion_tokens'],
            'tools': [{'type': 'function', 'function': {'name': schema['name'],
                       'description': 'Return the requested curation result.',
                       'parameters': schema['schema'], 'strict': True}}],
            'tool_choice': {'type': 'function', 'function': {'name': schema['name']}},
            'extra_body': {'reasoning': {'effort': effort},
                           'provider': {'only': ['amazon-bedrock/global'], 'allow_fallbacks': False, 'zdr': True,
                                        'require_parameters': True}}}


def normalize_openrouter(result, spec):
    if getattr(result, 'provider', None) != 'Amazon Bedrock':
        raise RuntimeError('OpenRouter returned an unexpected provider')
    if result.model != 'anthropic/' + spec['model']:
        raise RuntimeError('OpenRouter returned an unexpected model')
    choice = result.choices[0]
    calls = choice.message.tool_calls or []
    if (choice.finish_reason != 'tool_calls' or choice.message.refusal or len(calls) != 1
            or calls[0].function.name != spec['response_format']['json_schema']['name']):
        raise RuntimeError('Incomplete/refused/unexpected result tool; no training label was accepted')
    usage = result.usage.model_dump() if result.usage else {}
    usage['provider'] = result.provider
    return SimpleNamespace(model=result.model, usage=SimpleNamespace(model_dump=lambda:usage),
        choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(refusal=None, content=calls[0].function.arguments))])


def requesty_request(spec):
    """Pin the requested Claude model to Requesty's Bedrock route."""
    effort = spec['reasoning_effort']
    if effort not in ('none', 'low', 'medium', 'high'):
        raise ValueError('Unsupported Claude effort: ' + effort)
    schema = spec['response_format']['json_schema']
    # Same strict result-tool schema as the OpenRouter Bedrock route. The bedrock/
    # model prefix selects the provider; no fallback policy is requested.
    request = {'model': 'bedrock/' + spec['model'], 'messages': spec['messages'],
               'max_tokens': spec['max_completion_tokens'],
               'tools': [{'type': 'function', 'function': {'name': schema['name'],
                          'description': 'Return the requested curation result.',
                          'parameters': schema['schema'], 'strict': True}}],
               'tool_choice': {'type': 'function', 'function': {'name': schema['name']}}}
    if effort != 'none':
        request['reasoning_effort'] = effort
    return request


def normalize_requesty(result, spec):
    if result.model != spec['model']:
        raise RuntimeError('Requesty returned an unexpected model')
    choice = result.choices[0]
    calls = choice.message.tool_calls or []
    if (choice.finish_reason != 'tool_calls' or choice.message.refusal or len(calls) != 1
            or calls[0].function.name != spec['response_format']['json_schema']['name']):
        raise RuntimeError('Incomplete/refused/unexpected result tool; no training label was accepted')
    usage = result.usage.model_dump() if result.usage else {}
    usage['requesty_model'] = 'bedrock/' + spec['model']
    return SimpleNamespace(model=result.model, usage=SimpleNamespace(model_dump=lambda:usage),
        choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(refusal=None, content=calls[0].function.arguments))])


class CurationClient:
    """Expose the narrow completion interface consumed by AsyncStages."""
    def __init__(self, models, clients=None, claude_provider='anthropic'):
        self.claude_provider = claude_provider
        self.clients = dict(clients or {})
        self.chat = SimpleNamespace(completions=self)
        for provider in {provider_for(m, claude_provider) for m in models}:
            if provider in self.clients:
                continue
            if provider == 'openai':
                from openai import AsyncOpenAI
                self.clients[provider] = AsyncOpenAI(base_url='https://api.openai.com/v1', max_retries=2, timeout=180)
            elif provider == 'openrouter':
                from openai import AsyncOpenAI
                self.clients[provider] = AsyncOpenAI(base_url='https://openrouter.ai/api/v1',
                    api_key=os.environ['OPENROUTER_API_KEY'], max_retries=2, timeout=180)
            elif provider == 'requesty':
                from openai import AsyncOpenAI
                self.clients[provider] = AsyncOpenAI(base_url='https://router.requesty.ai/v1',
                    api_key=os.environ['REQUESTY_API_KEY'], max_retries=2, timeout=180)
            else:
                from anthropic import AsyncAnthropic
                self.clients[provider] = AsyncAnthropic(base_url='https://api.anthropic.com', max_retries=2, timeout=180)

    async def create(self, **spec):
        provider = provider_for(spec['model'], self.claude_provider)
        client = self.clients[provider]
        if provider == 'openai':
            return await client.chat.completions.create(**spec)
        if provider == 'openrouter':
            result = await client.chat.completions.create(**openrouter_request(spec))
            return normalize_openrouter(result, spec)
        if provider == 'requesty':
            result = await client.chat.completions.create(**requesty_request(spec))
            return normalize_requesty(result, spec)
        return normalize_anthropic(await client.messages.create(**anthropic_request(spec)))

    async def close(self):
        for client in self.clients.values():
            await client.close()
