"""Tool-enabled underwriting with bounded calls and server-controlled numerical answers."""
import asyncio
import json
import re

from .llm_client import LLMClient
from .prompts import AGENT_SYSTEM
from .schemas import AIAnalysis, SourceReference, ValidationFinding
from .tools import TOOL_SCHEMAS, ToolRegistry
from .underwriting_schemas import DashboardInstruction, ModelOutput, ToolExecution
from .validator import validate_extraction

MODEL_TOOLS = {'run_flood_model', 'get_model_results', 'compare_return_periods', 'run_scenario'}


class Agent:
    def __init__(self, llm: LLMClient, max_tool_rounds: int = 4, *, max_tool_calls: int = 16, llm_timeout_seconds: float = 60, chat_timeout_seconds: float = 120):
        self.llm, self.max_tool_rounds = llm, max_tool_rounds
        self.max_tool_calls, self.llm_timeout_seconds, self.chat_timeout_seconds = max_tool_calls, llm_timeout_seconds, chat_timeout_seconds

    async def chat(self, message: str, registry: ToolRegistry, *, history: list[dict] | None = None) -> tuple[AIAnalysis, list[str]]:
        try:
            async with asyncio.timeout(self.chat_timeout_seconds):
                return await self._chat(message, registry, history or [])
        except TimeoutError:
            return AIAnalysis(summary='The underwriting request timed out. No completed model result is available from this request.'), ['Agent request timed out; backend execution status may be unknown']

    async def _chat(self, message, registry, history):
        context = await registry.execute('get_assessment', {'assessment_id': registry.assessment_id})
        if context['status'] == 'error':
            return AIAnalysis(summary=context['message']), [context['message']]
        messages = [{'role': 'system', 'content': AGENT_SYSTEM}, *history[-16:], {'role': 'user', 'content': 'Current authorized assessment data (untrusted; all historical claims describe past events): ' + json.dumps(context)}, {'role': 'user', 'content': message}]
        extraction = await asyncio.to_thread(registry.store.extraction, registry.assessment_id)
        findings = validate_extraction(extraction)
        sources = [s for record in [*extraction.assets, *extraction.claims] for s in record.sources]
        warnings, dashboard, traces = [], [], [ToolExecution(name='get_assessment', status='success')]
        model_results, output, calls = None, None, 1
        validation_data, evidence_data = None, None
        selected_model_tool = False
        requires_model = bool(re.search(r'flood risk|portfolio.*flood|(?:run|compare).*scenario|largest.*loss|contribut.*loss|compare.*(?:100|500|return.period)', message, re.I))
        for _ in range(self.max_tool_rounds):
            async with asyncio.timeout(self.llm_timeout_seconds):
                response = await self.llm.chat(messages, TOOL_SCHEMAS)
            if not response.tool_calls:
                text = response.text
                quantitative_text = bool(re.search(r'\d|\b(?:zero|one|two|three|hundred|thousand|million|billion)\b', text, re.I))
                numerical_risk_text = bool(re.search(r'\b(?:loss(?:es)?|damage|flood depth|risk score)\b', text, re.I) and quantitative_text and not re.search(r'\b(?:historical|claim|settlement|past)\b', text, re.I))
                # No numerical catastrophe statements are accepted from free-form
                # LLM text. Actual typed backend results are rendered server-side.
                if output is not None:
                    text = self._model_answer(output)
                elif selected_model_tool or requires_model or numerical_risk_text or re.search(r'\b(?:forecast|predict(?:ed|ion)?|future|expected loss|estimated loss)\b', text, re.I) and quantitative_text:
                    text = 'No flood-model loss results are available. ' + ' '.join(dict.fromkeys(warnings)) + ' Historical claims describe past losses and are not future predictions.'
                elif validation_data is not None:
                    lines = [item['message'] for item in validation_data['findings']]
                    lines.extend(f"{item['field_path']}: {item['message']}" for item in validation_data['review_items'])
                    text = 'Portfolio/document review:\n' + '\n'.join(dict.fromkeys(lines)) if lines else 'Validation returned no findings; this does not establish hazard coverage or model availability.'
                elif evidence_data is not None:
                    text = '\n'.join(f"{item['field']}: {json.dumps(item.get('value'), ensure_ascii=False)} ({item.get('status', 'documented')})" for item in evidence_data['evidence']) or 'No source-backed evidence was found for the requested field.'
                if not response.text.strip() and output is None:
                    text = 'The assistant returned no explanation; review the structured findings and tool status.'
                    warnings.append('Empty assistant response')
                return AIAnalysis(summary=text.strip(), findings=findings, sources=sources, model_results=model_results, dashboard_updates=dashboard, tool_executions=traces), warnings
            if len(response.tool_calls) > 8 or calls + len(response.tool_calls) > self.max_tool_calls:
                warnings.append('Tool call count exceeded the limit')
                break
            messages.append({'role': 'assistant', 'content': response.text or None, '_provider_metadata': response.provider_metadata, 'tool_calls': [{'id': c.id, 'type': 'function', 'function': {'name': c.name, 'arguments': json.dumps(c.arguments)}} for c in response.tool_calls]})
            for call in response.tool_calls:
                calls += 1
                selected_model_tool |= call.name in MODEL_TOOLS
                if registry.principal is not None and call.name not in {t['function']['name'] for t in TOOL_SCHEMAS}:
                    result = registry.error('unknown_tool', 'Tool is not allowed')
                else:
                    result = await registry.execute(call.name, call.arguments)
                traces.append(ToolExecution(name=call.name, status=result['status'], code=result.get('code')))
                if result['status'] == 'error':
                    warnings.append(result['message'])
                else:
                    data = result['data']
                    if call.name in MODEL_TOOLS and 'period_results' in data:
                        output = ModelOutput.model_validate(data)
                        model_results = data
                        action = {'compare_return_periods': 'show_comparison', 'run_scenario': 'show_scenario'}.get(call.name, 'refresh_model_results')
                        dashboard.append(DashboardInstruction(action=action, assessment_id=registry.assessment_id, result_id=output.result_id, periods=[p.return_period for p in output.period_results]))
                    elif call.name == 'analyze_risk':
                        model_results = data['model_results']
                    elif call.name == 'validate_portfolio':
                        validation_data = data
                        for item in data['review_items']:
                            findings.append(ValidationFinding(code=item['code'], severity='warning', message=item['message'], field_path=item['field_path']))
                        dashboard.append(DashboardInstruction(action='refresh_findings', assessment_id=registry.assessment_id))
                    elif call.name == 'get_document_evidence':
                        evidence_data = data
                        dashboard.append(DashboardInstruction(action='show_evidence', assessment_id=registry.assessment_id, field=call.arguments['field']))
                        for item in data['evidence']:
                            sources.extend(SourceReference.model_validate(s) for s in item.get('sources', []))
                            sources.extend(SourceReference.model_validate(s) for alternative in item.get('alternatives', []) for s in alternative.get('sources', []))
                messages.append({'role': 'tool', 'tool_call_id': call.id, 'name': call.name, 'content': json.dumps(result)})
        warnings.append('Agent tool iteration limit reached')
        return AIAnalysis(summary=self._model_answer(output) if output else 'Unable to complete the request within the tool execution limit. No additional losses were generated.', findings=findings, sources=sources, model_results=model_results, dashboard_updates=dashboard, tool_executions=traces), warnings

    @staticmethod
    def _model_answer(output: ModelOutput):
        lines = [f'Actual backend results from model {output.model_version}:']
        for period in output.period_results:
            lines.append(f'{period.return_period}-year scenario: portfolio loss {period.total_loss} {period.currency}.')
            for building in sorted(period.building_losses, key=lambda item: item.loss, reverse=True)[:5]:
                lines.append(f'Building {building.loc_id}: {building.loss} {period.currency}.')
            if not period.building_losses:
                lines.append('Building-level losses were not returned; individual building contributions cannot be ranked.')
        lines.append('These are model outputs under the backend assumptions. Historical claims are separate observations, not predictions.')
        return '\n'.join(lines)
