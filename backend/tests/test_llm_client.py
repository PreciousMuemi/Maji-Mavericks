import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field
from ai.config import Settings
from ai.llm_client import AnthropicClient, GeminiClient, OpenAIClient, OpenRouterClient, UnconfiguredClient, _anthropic_schema, _provider_schema, create_llm_client
from ai.schemas import DocumentExtraction
from ai.insurance_schemas import InsuranceFacts, InsuranceFinancialTerms


def test_config_reads_env_and_masks_secret(monkeypatch):
    monkeypatch.setenv('NZOIA_LLM_PROVIDER', 'gemini')
    monkeypatch.setenv('NZOIA_GEMINI_API_KEY', 'private-key')
    settings = Settings(_env_file=None)
    assert settings.llm_provider == 'gemini'
    assert 'private-key' not in repr(settings)
    assert settings.gemini_api_key.get_secret_value() == 'private-key'


def test_factory_without_credentials(monkeypatch):
    monkeypatch.delenv('NZOIA_OPENAI_API_KEY', raising=False)
    assert isinstance(create_llm_client(Settings(_env_file=None, openai_api_key=None)), UnconfiguredClient)


@pytest.mark.asyncio
async def test_openai_structured_output_and_tool_translation():
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = 'configured-model'
    message = SimpleNamespace(parsed=DocumentExtraction(), content='', tool_calls=[])
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create, parse=create)))
    assert await client.structured('system', 'document', DocumentExtraction) == DocumentExtraction()
    assert create.call_args.kwargs['response_format'] is DocumentExtraction
    message.tool_calls = [SimpleNamespace(id='call1', function=SimpleNamespace(name='get_exposures', arguments='{}'))]
    result = await client.chat([{'role': 'user', 'content': 'hello'}], [])
    assert result.tool_calls[0].name == 'get_exposures'
    assert result.tool_calls[0].arguments == {}


@pytest.mark.asyncio
async def test_openrouter_requires_capability_routing_and_translates_tools():
    client = OpenRouterClient.__new__(OpenRouterClient)
    client.model = 'openrouter/free'
    message = SimpleNamespace(content=DocumentExtraction().model_dump_json(), tool_calls=[])
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create, parse=create)))
    assert await client.structured('system', 'document', DocumentExtraction) == DocumentExtraction()
    assert create.call_args.kwargs['extra_body']['provider']['require_parameters'] is True
    assert create.call_args.kwargs['response_format']['type'] == 'json_schema'
    assert create.call_args.kwargs['response_format']['json_schema']['strict'] is True
    message.tool_calls = [SimpleNamespace(id='call1', function=SimpleNamespace(name='get_assessment', arguments='{}'))]
    result = await client.chat([{'role': 'user', 'content': 'hello'}], [{'type': 'function'}])
    assert result.tool_calls[0].name == 'get_assessment'
    assert create.call_args.kwargs['extra_body']['provider']['require_parameters'] is True


@pytest.mark.asyncio
async def test_openrouter_falls_back_when_strict_schema_returns_no_choices():
    client = OpenRouterClient.__new__(OpenRouterClient)
    client.model = 'configured-model'
    valid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=DocumentExtraction().model_dump_json()))])
    create = AsyncMock(side_effect=[SimpleNamespace(choices=None), valid])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    assert await client.structured('system', 'document', DocumentExtraction) == DocumentExtraction()
    assert create.await_count == 2
    assert create.call_args_list[0].kwargs['response_format']['type'] == 'json_schema'
    assert create.call_args_list[1].kwargs['response_format']['type'] == 'json_object'


@pytest.mark.asyncio
async def test_openrouter_rejects_invalid_fallback_output():
    client = OpenRouterClient.__new__(OpenRouterClient)
    client.model = 'configured-model'
    invalid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
    create = AsyncMock(side_effect=[SimpleNamespace(choices=None), invalid])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    # A required field proves that the fallback is still locally schema validated.
    class RequiredOutput(DocumentExtraction):
        required_value: str

    with pytest.raises(ValueError, match='schema-valid output'):
        await client.structured('system', 'document', RequiredOutput)
    assert create.await_count == 2


def test_openrouter_factory_uses_free_router(monkeypatch):
    settings = Settings(_env_file=None, llm_provider='openrouter', openrouter_api_key='test-key', llm_model=None)
    client = create_llm_client(settings)
    assert isinstance(client, OpenRouterClient)
    assert client.model == 'openrouter/free'


def test_openrouter_schema_removes_unsupported_decimal_lookaround_only():
    schema = _provider_schema(InsuranceFacts)
    money_amount = schema['$defs']['Money']['properties']['amount']['anyOf']
    assert all('(?!' not in item.get('pattern', '') for item in money_amount)
    assert schema['$defs']['Money']['properties']['currency']['anyOf'][0]['pattern'] == '^[A-Z]{3}$'


@pytest.mark.asyncio
async def test_anthropic_structured_output_and_tool_translation():
    client = AnthropicClient.__new__(AnthropicClient)
    client.model = 'claude-test'
    client.max_output_tokens = 32768
    create = AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(
        type='text', text=DocumentExtraction().model_dump_json())]))
    client.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    assert await client.structured('system', 'document', DocumentExtraction) == DocumentExtraction()
    call = create.call_args.kwargs
    assert call['system'] == 'system'
    assert call['output_config']['format']['type'] == 'json_schema'

    create.return_value = SimpleNamespace(content=[
        SimpleNamespace(type='text', text='Checking the assessment.'),
        SimpleNamespace(type='tool_use', id='tool1', name='get_assessment',
                        input={'assessment_id': 'assessment-1'}),
    ])
    result = await client.chat(
        [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'review'}],
        [{'type': 'function', 'function': {'name': 'get_assessment',
          'description': 'Get assessment', 'parameters': {'type': 'object'}}}],
    )
    assert result.text == 'Checking the assessment.'
    assert result.tool_calls[0].arguments == {'assessment_id': 'assessment-1'}
    assert create.call_args.kwargs['tools'][0]['input_schema'] == {'type': 'object'}


@pytest.mark.asyncio
async def test_anthropic_falls_back_when_compiled_grammar_is_too_large():
    client = AnthropicClient.__new__(AnthropicClient)
    client.model = 'claude-test'
    client.max_output_tokens = 32768

    class GrammarTooLarge(Exception):
        status_code = 400

    create = AsyncMock(side_effect=[
        GrammarTooLarge('The compiled grammar is too large'),
        SimpleNamespace(content=[SimpleNamespace(
            type='text', text=f'```json\n{DocumentExtraction().model_dump_json()}\n```')]),
    ])
    client.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await client.structured('system', 'document', DocumentExtraction)

    assert result == DocumentExtraction()
    assert create.await_count == 2
    assert 'output_config' not in create.call_args_list[1].kwargs


@pytest.mark.asyncio
async def test_anthropic_repairs_invalid_unconstrained_json():
    class NumericResult(BaseModel):
        value: int = Field(ge=0)

    client = AnthropicClient.__new__(AnthropicClient)
    client.model = 'claude-test'
    client.max_output_tokens = 32768

    class GrammarTooLarge(Exception):
        status_code = 400

    create = AsyncMock(side_effect=[
        GrammarTooLarge('The compiled grammar is too large'),
        SimpleNamespace(content=[SimpleNamespace(type='text', text='{"value":"approximately 5"}')]),
        SimpleNamespace(content=[SimpleNamespace(type='text', text='{"value":5}')]),
    ])
    client.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await client.structured('system', 'document', NumericResult)

    assert result.value == 5
    assert create.await_count == 3


@pytest.mark.asyncio
async def test_anthropic_repairs_cross_field_validation_after_native_output():
    class PairedResult(BaseModel):
        value: int

    client = AnthropicClient.__new__(AnthropicClient)
    client.model = 'claude-test'
    client.max_output_tokens = 32768
    create = AsyncMock(side_effect=[
        SimpleNamespace(content=[SimpleNamespace(type='text', text='{"value":"five"}')]),
        SimpleNamespace(content=[SimpleNamespace(type='text', text='{"value":5}')]),
    ])
    client.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await client.structured('system', 'document', PairedResult)

    assert result.value == 5
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_anthropic_discards_fact_that_remains_unverified_after_repairs():
    client = AnthropicClient.__new__(AnthropicClient)
    client.model = 'claude-test'
    client.max_output_tokens = 32768
    invalid = InsuranceFinancialTerms().model_dump(mode='json')
    invalid['insurance_terms']['premiums'] = {
        'value': {'amount': '100', 'currency': 'KES'}, 'status': 'provided',
        'confidence': None, 'sources': [], 'alternatives': [], 'review_reason': None,
    }
    response = SimpleNamespace(content=[SimpleNamespace(type='text', text=json.dumps(invalid))])
    create = AsyncMock(side_effect=[response, response, response])
    client.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await client.structured('system', 'document', InsuranceFinancialTerms)

    assert result.insurance_terms.premiums.status == 'not_provided'
    assert result.insurance_terms.premiums.value is None
    assert create.await_count == 3


def test_anthropic_replays_tool_results_in_native_message_blocks():
    system, messages = AnthropicClient._messages([
        {'role': 'system', 'content': 'policy'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'tool1', 'function': {'name': 'get_assessment', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'tool1', 'name': 'get_assessment',
         'content': '{"status":"success"}'},
    ])
    assert system == 'policy'
    assert messages[0]['content'][0]['type'] == 'tool_use'
    assert messages[1]['content'][0]['type'] == 'tool_result'
    assert messages[1]['content'][0]['tool_use_id'] == 'tool1'


def test_anthropic_schema_removes_unsupported_constraints_but_local_model_keeps_them():
    schema = _anthropic_schema(DocumentExtraction)
    encoded = str(schema)
    assert 'minLength' not in encoded
    assert 'minimum' not in encoded
    assert DocumentExtraction.model_json_schema() != schema


def test_anthropic_factory_uses_current_haiku_by_default():
    settings = Settings(_env_file=None, llm_provider='anthropic',
                        anthropic_api_key='test-key', llm_model=None)
    client = create_llm_client(settings)
    assert isinstance(client, AnthropicClient)
    assert client.model == 'claude-haiku-5-5'

@pytest.mark.asyncio
async def test_gemini_replays_native_signed_content():
    from google.genai import types
    client = GeminiClient.__new__(GeminiClient)
    client.model = 'configured-model'
    native = types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(name='get_exposures', args={}), thought_signature=b'signature')])
    response = SimpleNamespace(candidates=[SimpleNamespace(content=native)], function_calls=[types.FunctionCall(name='get_exposures', args={})])
    generate = AsyncMock(return_value=response)
    client.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    result = await client.chat([{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'hello'}], [])
    assert result.provider_metadata['gemini_parts'] == native.parts
    await client.chat([{'role': 'system', 'content': 'system'}, {'role': 'assistant', '_provider_metadata': result.provider_metadata}, {'role': 'tool', 'name': 'get_exposures', 'content': '{"status":"success","data":{}}'}], [])
    assert generate.call_args.kwargs['contents'][0].parts[0].thought_signature == b'signature'
    assert generate.call_args.kwargs['config'].automatic_function_calling.disable
def test_provider_name_is_case_insensitive():
    settings = Settings(_env_file=None, llm_provider="Anthropic")

    assert settings.llm_provider == "anthropic"
