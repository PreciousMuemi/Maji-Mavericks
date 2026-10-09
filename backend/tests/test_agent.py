import pytest
from ai.agent import Agent
from ai.exposure_mapper import CanonicalExposureMapper
from ai.risk_analyzer import RiskAnalyzer, UnavailableFloodModel
from ai.schemas import Asset, DocumentExtraction, FinancialValue, LLMResponse, ParsedDocument, ToolCall
from ai.store import AssessmentStore
from ai.tools import ToolRegistry


@pytest.fixture
def registry(tmp_path):
    store = AssessmentStore(str(tmp_path))
    assessment_id = store.create()
    store.add_document(assessment_id, ParsedDocument(document_id='doc', segments=[]))
    store.save_extractions(assessment_id, {'doc': DocumentExtraction(assets=[Asset(asset_id='a', name='A', latitude=0, longitude=34, insured_value=FinancialValue(amount=100, currency='KES'))])})
    yield ToolRegistry(assessment_id, store, RiskAnalyzer(UnavailableFloodModel(), CanonicalExposureMapper()))
    store.close()


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    async def chat(self, messages, tools):
        self.messages = list(messages)
        return next(self.responses)


@pytest.mark.asyncio
async def test_agent_reports_unavailable_model(registry):
    llm = ScriptedLLM([LLMResponse(tool_calls=[ToolCall(id='1', name='analyze_risk', arguments={})]), LLMResponse(text='Flood model is unavailable.')])
    analysis, warnings = await Agent(llm).chat('Analyze risk', registry)
    assert analysis.model_results is None
    assert warnings == ['Flood model integration is unavailable or required inputs are incomplete']
    assert 'integration_unavailable' in llm.messages[-1]['content']
    assert registry.store.analysis(registry.assessment_id) is None


@pytest.mark.asyncio
async def test_tool_allowlist_and_assessment_scope(registry):
    assert (await registry.execute('shell', {}))['code'] == 'unknown_tool'
    assert (await registry.execute('get_exposures', {'assessment_id': 'other'}))['code'] == 'invalid_arguments'
    assert (await registry.execute('get_exposures', {}))['data']['assets'][0]['asset_id'] == 'a'


@pytest.mark.asyncio
async def test_iteration_limit(registry):
    call = LLMResponse(tool_calls=[ToolCall(id='1', name='get_exposures', arguments={})])
    analysis, warnings = await Agent(ScriptedLLM([call]), max_tool_rounds=1).chat('Loop', registry)
    assert 'iteration limit reached' in warnings[-1]
    assert analysis.model_results is None


@pytest.mark.asyncio
async def test_injected_model_receives_canonical_exposures(registry):
    class RecordingModel:
        async def analyze(self, assessment_id, exposures):
            self.received = (assessment_id, exposures)
            # Verify forwarding independently of a genuine result. Empty output
            # must now fail validation rather than count as a calculation.
            return {}

        async def scenario(self, assessment_id, exposures, parameters):
            self.scenario_received = (assessment_id, exposures, parameters)
            return {}

    model = RecordingModel()
    registry.analyzer.model = model
    result = await registry.execute('analyze_risk', {})
    assert result['status'] == 'error'
    assert result['code'] == 'integration_unavailable'
    assert model.received[1][0]['insured_value'] == {'amount': '100', 'currency': 'KES'}
    assert registry.store.analysis(registry.assessment_id) is None
    assert (await registry.execute('run_scenario', {'name': 'test', 'parameters': {'event_id': 'provided'}}))['status'] == 'error'
    assert model.scenario_received[2] == {'event_id': 'provided'}


@pytest.mark.asyncio
async def test_integration_error_does_not_leak(registry):
    from ai.risk_analyzer import IntegrationUnavailable
    class BrokenModel:
        async def analyze(self, *args):
            raise IntegrationUnavailable('CONFIDENTIAL MODEL DETAIL')
    registry.analyzer.model = BrokenModel()
    result = await registry.execute('analyze_risk', {})
    assert result['code'] == 'integration_unavailable'
    assert 'CONFIDENTIAL' not in result['message']


@pytest.mark.asyncio
async def test_excessive_tool_batch_is_not_executed(registry):
    response = LLMResponse(tool_calls=[ToolCall(id=str(i), name='analyze_risk', arguments={}) for i in range(9)])
    _, warnings = await Agent(ScriptedLLM([response])).chat('Analyze', registry)
    assert 'Tool call count exceeded the limit' in warnings
    assert registry.store.analysis(registry.assessment_id) is None


@pytest.mark.asyncio
async def test_agent_rejects_empty_injected_model_payload(registry):
    class ModelAdapter:
        async def analyze(self, *args):
            return {}
    registry.analyzer.model = ModelAdapter()
    llm = ScriptedLLM([LLMResponse(tool_calls=[ToolCall(id='1', name='analyze_risk', arguments={})]), LLMResponse(text='The configured adapter returned an empty payload.')])
    analysis, warnings = await Agent(llm).chat('Analyze', registry)
    assert analysis.model_results is None
    assert warnings == ['Flood model integration is unavailable or required inputs are incomplete']
