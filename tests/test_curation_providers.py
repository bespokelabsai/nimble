import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nimble.datasets.curation_providers import (
    CurationClient, anthropic_request, normalize_anthropic, provider_for, openrouter_request, requesty_request,
)


def spec(model='claude-sonnet-5', effort='low'):
    return {'model':model,'reasoning_effort':effort,'max_completion_tokens':8192,
        'messages':[{'role':'user','content':'Same curation prompt.'}],
        'response_format':{'type':'json_schema','json_schema':{'name':'Result','strict':True,
            'schema':{'type':'object','properties':{'valid':{'type':'boolean'}},
                      'required':['valid'],'additionalProperties':False}}}}


def response(reason='end_turn'):
    return SimpleNamespace(model='claude-sonnet-5',stop_reason=reason,
        content=[SimpleNamespace(type='thinking',thinking='internal'),
                 SimpleNamespace(type='text',text='{"valid":true}')],
        usage=SimpleNamespace(model_dump=lambda:{'input_tokens':10,'output_tokens':4,
            'cache_creation_input_tokens':3,'cache_read_input_tokens':2}))


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_haiku_uses_manual_thinking_without_effort_or_adaptive(self):
        for model in ('claude-haiku-4-5', 'claude-haiku-4-5-20251001'):
            for effort, budget in [('none',0),('low',1024),('medium',2048),('high',4096)]:
                original = spec(model,effort)
                actual = anthropic_request(original)
                self.assertEqual(actual['model'],model)
                self.assertEqual(actual['messages'],original['messages'])
                self.assertEqual(actual['output_config']['format']['schema'],
                                 original['response_format']['json_schema']['schema'])
                self.assertNotIn('effort',actual['output_config'])
                self.assertEqual(actual['thinking'], {'type':'enabled','budget_tokens':budget}
                                 if budget else {'type':'disabled'})
        with self.assertRaisesRegex(ValueError,'smaller'):
            anthropic_request({**spec('claude-haiku-4-5'),'max_completion_tokens':1024})

    def test_anthropic_preserves_prompt_schema_and_effort(self):
        original=spec()
        actual=anthropic_request(original)
        self.assertEqual(actual['messages'],original['messages'])
        self.assertEqual(actual['output_config']['format']['schema'],original['response_format']['json_schema']['schema'])
        self.assertEqual(actual['thinking'],{'type':'adaptive'})
        self.assertEqual(actual['output_config']['effort'],'low')
        self.assertNotIn('temperature',actual)
        disabled=anthropic_request(spec(effort='none'))
        self.assertEqual(disabled['thinking'],{'type':'disabled'})
        self.assertNotIn('effort',disabled['output_config'])

    def test_response_normalization_does_not_accept_refusal_or_truncation(self):
        normal=normalize_anthropic(response())
        self.assertEqual(normal.choices[0].message.content,'{"valid":true}')
        self.assertEqual(normal.usage.model_dump()['total_tokens'],19)
        for reason in ('refusal','max_tokens','pause_turn'):
            rejected=normalize_anthropic(response(reason))
            self.assertNotEqual(rejected.choices[0].finish_reason,'stop')
        self.assertTrue(normalize_anthropic(response('refusal')).choices[0].message.refusal)

    async def test_routes_exact_models_to_native_providers(self):
        openai_create=AsyncMock(return_value='openai-result')
        anthropic_create=AsyncMock(return_value=response())
        clients={'openai':SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=openai_create)),close=AsyncMock()),
                 'anthropic':SimpleNamespace(messages=SimpleNamespace(create=anthropic_create),close=AsyncMock())}
        client=CurationClient(['gpt-5.6-luna','claude-sonnet-5'],clients)
        luna=spec('gpt-5.6-luna')
        self.assertEqual(await client.create(**luna),'openai-result')
        openai_create.assert_awaited_once_with(**luna)
        await client.create(**spec())
        anthropic_create.assert_awaited_once_with(**anthropic_request(spec()))
        await client.close()
        for c in clients.values():c.close.assert_awaited_once()

    def test_unknown_provider_is_not_silently_routed(self):
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            provider_for('another/model')

    async def test_openrouter_pins_model_provider_and_schema(self):
        original = spec()
        request = openrouter_request(original)
        self.assertEqual(request['model'], 'anthropic/claude-sonnet-5')
        self.assertEqual(request['messages'], original['messages'])
        self.assertEqual(request['tools'][0]['function']['parameters'],
                         original['response_format']['json_schema']['schema'])
        self.assertTrue(request['tools'][0]['function']['strict'])
        self.assertEqual(request['tool_choice']['function']['name'], 'Result')
        self.assertEqual(request['extra_body']['provider'],
                         {'only':['amazon-bedrock/global'], 'allow_fallbacks':False, 'require_parameters':True, 'zdr':True})
        self.assertNotIn('temperature', request)
        call = SimpleNamespace(function=SimpleNamespace(name='Result',arguments='{"valid":true}'))
        choice = SimpleNamespace(finish_reason='tool_calls',
                                 message=SimpleNamespace(refusal=None,tool_calls=[call]))
        result = SimpleNamespace(provider='Amazon Bedrock',model='anthropic/claude-sonnet-5',
                                 choices=[choice],usage=None)
        create = AsyncMock(return_value=result)
        transport = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client = CurationClient(['claude-sonnet-5'], {'openrouter':transport}, claude_provider='openrouter')
        normalized = await client.create(**original)
        self.assertEqual(normalized.choices[0].message.content, '{"valid":true}')
        self.assertEqual(normalized.usage.model_dump()['provider'], 'Amazon Bedrock')
        create.assert_awaited_once_with(**request)
        result.provider = 'Other'
        with self.assertRaisesRegex(RuntimeError,'unexpected provider'):
            await client.create(**original)
        result.provider = 'Amazon Bedrock'
        result.model = 'different-model'
        with self.assertRaisesRegex(RuntimeError,'unexpected model'):
            await client.create(**original)
        result.model = 'anthropic/claude-sonnet-5'
        for reason, refusal, calls in [('length',None,[call]), ('tool_calls','refused',[call]),
                                       ('tool_calls',None,[]), ('tool_calls',None,[call,call])]:
            choice.finish_reason = reason
            choice.message.refusal = refusal
            choice.message.tool_calls = calls
            with self.assertRaisesRegex(RuntimeError,'no training label was accepted'):
                await client.create(**original)

    async def test_requesty_pins_bedrock_model_and_schema(self):
        original = spec()
        request = requesty_request(original)
        self.assertEqual(request['model'], 'bedrock/claude-sonnet-5')
        self.assertEqual(request['messages'], original['messages'])
        self.assertEqual(request['tools'][0]['function']['parameters'],
                         original['response_format']['json_schema']['schema'])
        self.assertTrue(request['tools'][0]['function']['strict'])
        self.assertEqual(request['tool_choice']['function']['name'], 'Result')
        self.assertEqual(request['reasoning_effort'], 'low')
        self.assertNotIn('reasoning_effort', requesty_request(spec(effort='none')))
        self.assertNotIn('temperature', request)
        call = SimpleNamespace(function=SimpleNamespace(name='Result',arguments='{"valid":true}'))
        choice = SimpleNamespace(finish_reason='tool_calls',
                                 message=SimpleNamespace(refusal=None,tool_calls=[call]))
        result = SimpleNamespace(model='claude-sonnet-5',choices=[choice],usage=None)
        create = AsyncMock(return_value=result)
        transport = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client = CurationClient(['claude-sonnet-5'], {'requesty':transport}, claude_provider='requesty')
        normalized = await client.create(**original)
        self.assertEqual(normalized.choices[0].message.content, '{"valid":true}')
        self.assertEqual(normalized.usage.model_dump()['requesty_model'], 'bedrock/claude-sonnet-5')
        create.assert_awaited_once_with(**request)
        result.model = 'different-model'
        with self.assertRaisesRegex(RuntimeError,'unexpected model'):
            await client.create(**original)
        result.model = 'claude-sonnet-5'
        for reason, refusal, calls in [('length',None,[call]), ('tool_calls','refused',[call]),
                                       ('tool_calls',None,[]), ('tool_calls',None,[call,call])]:
            choice.finish_reason = reason
            choice.message.refusal = refusal
            choice.message.tool_calls = calls
            with self.assertRaisesRegex(RuntimeError,'no training label was accepted'):
                await client.create(**original)


if __name__=='__main__':unittest.main()
