"""Provider adapters validate structured output locally, regardless of provider."""
import json
from typing import Any, Protocol, TypeVar
from pydantic import BaseModel
from .config import Settings
from .schemas import LLMResponse, ToolCall

M = TypeVar("M", bound=BaseModel)


def _provider_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Remove regex features unsupported by some provider JSON grammars.

    Pydantic still validates the unmodified model after generation, so this only
    affects provider-side token constraints and never relaxes accepted output.
    """
    schema = model.model_json_schema()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            pattern = value.get('pattern')
            if isinstance(pattern, str) and any(token in pattern for token in ('(?=', '(?!', '(?<=', '(?<!')):
                value.pop('pattern')
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema


def _anthropic_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert Pydantic constraints to Anthropic's supported JSON Schema subset.

    Claude constrains the response shape. The original Pydantic model validates
    every constraint after generation, so removing unsupported grammar keywords
    here cannot cause an invalid result to be accepted.
    """
    schema = _provider_schema(model)
    unsupported = {
        'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf',
        'minLength', 'maxLength', 'minItems', 'maxItems', 'uniqueItems',
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in unsupported:
                value.pop(key, None)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema


class LLMUnavailable(RuntimeError):
    pass


class LLMClient(Protocol):
    async def structured(self, system: str, user: str, schema: type[M]) -> M: ...
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class OpenAIClient:
    def __init__(self, settings: Settings):
        from openai import AsyncOpenAI
        if not settings.openai_api_key:
            raise LLMUnavailable("OpenAI credentials are not configured")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value(), timeout=settings.llm_timeout_seconds, max_retries=2)
        self.model = settings.llm_model or "gpt-4.1-mini"

    async def structured(self, system, user, schema):
        result = await self.client.chat.completions.parse(
            model=self.model, response_format=schema,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        parsed = result.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Provider did not return structured output")
        return schema.model_validate(parsed)

    async def aclose(self):
        await self.client.close()

    async def chat(self, messages, tools):
        result = await self.client.chat.completions.create(model=self.model, messages=[{k: v for k, v in m.items() if not k.startswith("_")} for m in messages], **({"tools": tools} if tools else {}))
        message = result.choices[0].message
        return LLMResponse(text=message.content or "", tool_calls=[ToolCall(id=t.id, name=t.function.name, arguments=json.loads(t.function.arguments)) for t in message.tool_calls or []])


class OpenRouterClient(OpenAIClient):
    """OpenRouter adapter using its OpenAI-compatible API and free-model router."""
    def __init__(self, settings: Settings):
        from openai import AsyncOpenAI
        if not settings.openrouter_api_key:
            raise LLMUnavailable("OpenRouter credentials are not configured")
        self.client = AsyncOpenAI(api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.openrouter_base_url, timeout=settings.llm_timeout_seconds,
            max_retries=2, default_headers={'X-Title': 'Nzoia Risk Intelligence'})
        self.model = settings.llm_model or 'openrouter/free'

    async def structured(self, system, user, schema):
        schema_json = _provider_schema(schema)
        formats_and_prompts = [
            ({'type': 'json_schema', 'json_schema': {
                'name': schema.__name__, 'strict': True, 'schema': schema_json}}, user),
            ({'type': 'json_object'},
             f"{user}\n\nReturn one JSON object matching this JSON Schema exactly:\n"
             f"{json.dumps(schema_json, ensure_ascii=False)}"),
        ]
        last_error: Exception | None = None
        for response_format, request in formats_and_prompts:
            try:
                result = await self.client.chat.completions.create(model=self.model,
                    response_format=response_format,
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': request}],
                    extra_body={'provider': {'require_parameters': True}})
                choices = getattr(result, 'choices', None)
                if not choices:
                    last_error = ValueError('provider returned no choices')
                    continue
                content = choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    last_error = ValueError('provider returned empty content')
                    continue
                return schema.model_validate_json(content)
            except Exception as exc:
                last_error = exc
        raise ValueError('OpenRouter did not return schema-valid output') from last_error

    async def chat(self, messages, tools):
        result = await self.client.chat.completions.create(model=self.model,
            messages=[{k: v for k, v in m.items() if not k.startswith('_')} for m in messages],
            **({'tools': tools} if tools else {}),
            extra_body={'provider': {'require_parameters': bool(tools)}})
        message = result.choices[0].message
        return LLMResponse(text=message.content or '', tool_calls=[ToolCall(
            id=t.id, name=t.function.name, arguments=json.loads(t.function.arguments))
            for t in message.tool_calls or []])


class AnthropicClient:
    """Direct Claude adapter with native structured output and client tools."""
    requires_split_insurance_schema = True
    def __init__(self, settings: Settings):
        from anthropic import AsyncAnthropic
        if not settings.anthropic_api_key:
            raise LLMUnavailable("Anthropic credentials are not configured")
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
        self.model = settings.llm_model or 'claude-haiku-5-5'
        self.max_output_tokens = settings.llm_max_output_tokens

    async def structured(self, system, user, schema):
        provider_schema = _anthropic_schema(schema)
        try:
            result = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=system,
                messages=[{'role': 'user', 'content': user}],
                output_config={'format': {
                    'type': 'json_schema',
                    'schema': provider_schema,
                }},
            )
        except Exception as exc:
            # Anthropic refuses compiled grammars above an internal size limit.
            # Large extraction models still undergo the original strict local
            # validation and downstream evidence validation before acceptance.
            message = str(exc).casefold()
            if getattr(exc, 'status_code', None) != 400 or 'grammar is too large' not in message:
                raise
            result = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=(f'{system}\n\nReturn only one JSON object. Do not use Markdown fences. '
                        'The application will reject any value that does not match the supplied schema.'),
                messages=[{'role': 'user', 'content': (
                    f'{user}\n\nJSON Schema:\n{json.dumps(provider_schema, ensure_ascii=False)}')}],
            )
        content = ''.join(block.text for block in result.content
                          if getattr(block, 'type', None) == 'text')
        if not content.strip():
            raise ValueError('Anthropic did not return schema-valid output')
        def validate(raw: str):
            # Be tolerant only of a surrounding Markdown fence; the decoded
            # object is still checked by the exact Pydantic model below.
            stripped = raw.strip()
            if stripped.startswith('```') and stripped.endswith('```'):
                stripped = stripped[3:-3].strip()
                if stripped.casefold().startswith('json'):
                    stripped = stripped[4:].lstrip()
            return schema.model_validate_json(stripped)

        for repair_attempt in range(3):
            validation_error = None
            try:
                return validate(content)
            except Exception as exc:
                if repair_attempt == 2:
                    raise
                validation_error = exc
            raw_errors = getattr(validation_error, 'errors', lambda **_: [])(include_input=False)
            if not raw_errors:
                raise
            errors = [{key: item.get(key) for key in ('type', 'loc', 'msg')}
                      for item in raw_errors]
            repair = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=(f'{system}\n\nReturn only one corrected JSON object without Markdown. '
                        'Use JSON numbers for numeric fields. Do not add unsupported facts.'),
                messages=[
                    {'role': 'user', 'content': (
                        f'{user}\n\nJSON Schema:\n{json.dumps(provider_schema, ensure_ascii=False)}')},
                    {'role': 'assistant', 'content': content},
                    {'role': 'user', 'content': (
                        'Correct the preceding JSON in place. Preserve its source-backed facts, '
                        'but fix every listed type or schema violation. Qualifiers such as '
                        '"approximately" belong in uncertainty metadata; numeric value fields '
                        'must contain only JSON numbers. Return the complete corrected object.\n\n'
                        f'Validation errors:\n{json.dumps(errors, ensure_ascii=False)}')},
                ],
            )
            content = ''.join(block.text for block in repair.content
                              if getattr(block, 'type', None) == 'text')

    @staticmethod
    def _messages(messages):
        converted, system_parts = [], []
        for message in messages:
            role = message['role']
            if role == 'system':
                system_parts.append(message.get('content') or '')
                continue
            if role == 'tool':
                converted.append({'role': 'user', 'content': [{
                    'type': 'tool_result',
                    'tool_use_id': message['tool_call_id'],
                    'content': message.get('content') or '',
                }]})
                continue
            if role == 'assistant' and message.get('tool_calls'):
                blocks = []
                if message.get('content'):
                    blocks.append({'type': 'text', 'text': message['content']})
                blocks.extend({
                    'type': 'tool_use',
                    'id': call['id'],
                    'name': call['function']['name'],
                    'input': json.loads(call['function']['arguments']),
                } for call in message['tool_calls'])
                converted.append({'role': 'assistant', 'content': blocks})
            else:
                converted.append({'role': role, 'content': message.get('content') or ' '})
        return '\n\n'.join(part for part in system_parts if part), converted

    async def chat(self, messages, tools):
        system, converted = self._messages(messages)
        anthropic_tools = [{
            'name': tool['function']['name'],
            'description': tool['function'].get('description', ''),
            'input_schema': tool['function']['parameters'],
        } for tool in tools]
        result = await self.client.messages.create(
            model=self.model,
            max_tokens=min(self.max_output_tokens, 8_192),
            system=system,
            messages=converted,
            **({'tools': anthropic_tools} if anthropic_tools else {}),
        )
        text = ''.join(block.text for block in result.content
                       if getattr(block, 'type', None) == 'text')
        calls = [ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                 for block in result.content if getattr(block, 'type', None) == 'tool_use']
        return LLMResponse(text=text, tool_calls=calls)

    async def aclose(self):
        await self.client.close()


class GeminiClient:
    def __init__(self, settings: Settings):
        from google import genai
        from google.genai import types
        if not settings.gemini_api_key:
            raise LLMUnavailable("Gemini credentials are not configured")
        self.client = genai.Client(api_key=settings.gemini_api_key.get_secret_value(), http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000)))
        self.model = settings.llm_model or "gemini-2.5-flash"

    async def structured(self, system, user, schema):
        result = await self.client.aio.models.generate_content(model=self.model, contents=user,
            config={"system_instruction": system, "response_mime_type": "application/json", "response_schema": schema})
        return schema.model_validate_json(result.text or "")

    async def chat(self, messages, tools):
        from google.genai import types
        contents = []
        for message in messages:
            if message["role"] == "system":
                continue
            if message["role"] == "tool":
                parts = [types.Part(function_response=types.FunctionResponse(name=message["name"], response=json.loads(message["content"])))]
            elif message.get("_provider_metadata", {}).get("gemini_parts"):
                # Preserve native parts, including thought signatures required on later turns.
                parts = message["_provider_metadata"]["gemini_parts"]
            elif message.get("tool_calls"):
                parts = [types.Part(function_call=types.FunctionCall(name=t["function"]["name"], args=json.loads(t["function"]["arguments"]))) for t in message["tool_calls"]]
            else:
                parts = [types.Part(text=message["content"] or " ")]
            contents.append(types.Content(role="model" if message["role"] == "assistant" else "user", parts=parts))
        declarations = [types.FunctionDeclaration(name=t["function"]["name"], description=t["function"]["description"], parameters_json_schema=t["function"]["parameters"]) for t in tools]
        result = await self.client.aio.models.generate_content(model=self.model, contents=contents,
            config=types.GenerateContentConfig(system_instruction=messages[0]["content"], tools=[types.Tool(function_declarations=declarations)] if declarations else None,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
        if not result.candidates or not result.candidates[0].content:
            raise LLMUnavailable("Provider returned no usable response")
        return LLMResponse(provider_metadata={"gemini_parts": result.candidates[0].content.parts}, text="".join(p.text for p in result.candidates[0].content.parts if p.text),
            tool_calls=[ToolCall(id=f"gemini-{i}", name=c.name, arguments=dict(c.args or {})) for i, c in enumerate(result.function_calls or [])])

    async def aclose(self):
        await self.client.aio.aclose()
        self.client.close()


class UnconfiguredClient:
    async def structured(self, system, user, schema):
        raise LLMUnavailable("LLM credentials are not configured")

    async def chat(self, messages, tools):
        raise LLMUnavailable("LLM credentials are not configured")


def create_llm_client(settings: Settings) -> LLMClient:
    keys = {'openai': settings.openai_api_key, 'gemini': settings.gemini_api_key,
            'openrouter': settings.openrouter_api_key, 'anthropic': settings.anthropic_api_key}
    key = keys[settings.llm_provider]
    if not key:
        return UnconfiguredClient()
    clients = {'openai': OpenAIClient, 'gemini': GeminiClient,
               'openrouter': OpenRouterClient, 'anthropic': AnthropicClient}
    return clients[settings.llm_provider](settings)
