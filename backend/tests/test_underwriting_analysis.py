"""Synthetic evidence and model-response fixtures; no genuine model losses are fabricated."""
import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from ai.config import Settings
from ai.intelligence_schemas import AnalysisNarrative, UnderwritingAnalysis
from ai.insurance_schemas import DocumentClassification, InsuranceDocumentResult, InsuranceFacts
from ai.schemas import ParsedDocument, DocumentSegment
from ai.underwriting_analysis import UnderwritingAnalysisService, generate_underwriting_analysis
from ai.underwriting_schemas import ModelOutput, Principal
from api.ai_routes import build_services, get_principal
from app.main import create_app


class ExplainingLLM:
    async def structured(self, system, user, schema):
        assert schema is AnalysisNarrative
        facts = json.loads(user)['findings']
        preferred = [f for f in facts if 'condition' in f['id'] or 'drainage' in f['id'] or 'inventory' in f['id']]
        preferred = preferred[:3] or facts[:1]
        return schema(summary=[{'finding_id': item['id'], 'explanation': 'The evidence warrants protection and engineering review'} for item in preferred], drivers=[{'finding_id': item['id'], 'explanation': 'This can affect asset resilience and continuity of operations'} for item in preferred], actions=[{'finding_id': item['id'], 'action': 'Arrange a targeted engineering inspection and verify mitigation completion', 'rationale': 'Confirm the documented condition and suitability of proposed protections'} for item in preferred])


def build_document(store, count=4, duplicate_dates=False, currencies=None):
    assessment = store.create('alice')
    document_id = 'synthetic-risk-document'
    text = ['Insurance offer', 'Building Delta condition: poor', 'Drainage: channels are inadequate', 'Defenses: works are incomplete', 'Inventory: KES 2000', 'Broker recommends drainage improvement']
    fact = lambda value, excerpt: {'value': value, 'status': 'provided', 'confidence': 0.99, 'sources': [{'document_id': document_id, 'locator': 'page:1', 'page_number': 1, 'excerpt': excerpt}]}
    events = []
    for index in range(count):
        year = str(2010 if duplicate_dates else 2010 + index)
        currency = currencies[index] if currencies else 'KES'
        amount = (index + 1) * 100
        excerpt = f'Flood event {year}; claimed {currency} {amount}; settled {currency} {amount // 2}; flooded switchgear'
        text.append(excerpt)
        events.append({'event_date': fact(year, excerpt), 'historical_claim_value': fact({'amount': str(amount), 'currency': currency}, excerpt), 'settlement_value': fact({'amount': str(amount // 2), 'currency': currency}, excerpt), 'description': fact('flooded switchgear', excerpt)})
    store.add_document(assessment, ParsedDocument(document_id=document_id, segments=[DocumentSegment(locator='page:1', page_number=1, text='\n'.join(text))]))
    facts = InsuranceFacts.model_validate({'assets': [{'name': fact('Building Delta', 'Building Delta condition: poor'), 'condition': fact('poor', 'Building Delta condition: poor')}], 'risk_factors': {'drainage_weaknesses': [fact('channels are inadequate', 'Drainage: channels are inadequate')], 'flood_defenses': [fact('works are incomplete', 'Defenses: works are incomplete')]}, 'financial_exposure': {'inventory_values': fact({'amount': '2000', 'currency': 'KES'}, 'Inventory: KES 2000')}, 'flood_history': events, 'document_recommendations': [{'author_role': 'broker', 'recommendation': fact('Improve drainage', 'Broker recommends drainage improvement')}]})
    store.save_insurance_extraction(assessment, InsuranceDocumentResult(document_id=document_id, document_type=DocumentClassification(kind='insurance_offer', confidence=1, sources=facts.assets[0].name.sources, reason='Offer'), facts=facts, review_status='requires_review'))
    return assessment


@pytest.fixture
def services(tmp_path):
    settings = Settings(_env_file=None, storage_directory=str(tmp_path))
    value = build_services(settings, llm=ExplainingLLM())
    yield value
    value.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [2, 4, 5])
async def test_dynamic_historical_counts_totals_rankings_and_audit(services, count):
    assessment = build_document(services.store, count=count)
    result = await UnderwritingAnalysisService(services.store, services.extraction.llm).generate(assessment)
    metrics = {n.metric: n.value for f in result.historical_claims_findings for n in f.numerical_findings}
    assert metrics['documented_historical_flood_events'] == count
    assert metrics['historical_claimed_total'] == sum((i + 1) * 100 for i in range(count))
    assert metrics['historical_settled_total'] == sum((i + 1) * 50 for i in range(count))
    assert metrics['largest_historical_claimed_amount'] == count * 100
    assert metrics['events_with_repeated_description'] == count
    assert result.risk_level.value is None
    assert result.model_findings == []
    assert len(result.top_risk_drivers) == 3
    assert len(result.risk_summary.split('. ')) <= 3
    assert {a.origin for a in result.recommended_actions} >= {'broker_proposal', 'independent_analysis'}
    citations = {c.id: c for c in result.citations}
    for finding in [*result.historical_claims_findings, *result.model_findings]:
        for number in finding.numerical_findings:
            assert citations[number.citation_id].kind in {'document', 'model', 'calculation'}
    assert all(c.source.page_number == 1 for c in result.citations if c.kind == 'document')
    assert any(l.code == 'industrial_parameters_unknown' for l in result.limitations)
    UnderwritingAnalysis.model_validate_json(result.model_dump_json())


@pytest.mark.asyncio
async def test_ambiguous_event_identity_prevents_totals(services):
    assessment = build_document(services.store, duplicate_dates=True)
    result = await UnderwritingAnalysisService(services.store, services.extraction.llm).generate(assessment)
    metrics = {n.metric for finding in result.historical_claims_findings for n in finding.numerical_findings}
    assert 'historical_claimed_total' not in metrics
    assert 'documented_historical_flood_events' not in metrics
    assert any(l.code == 'historical_identity_ambiguous' for l in result.limitations)


@pytest.mark.asyncio
async def test_different_currencies_and_claim_settlement_categories_not_added(services):
    assessment = build_document(services.store, count=2, currencies=['KES', 'USD'])
    result = await UnderwritingAnalysisService(services.store, services.extraction.llm).generate(assessment)
    totals = [(n.value, n.unit, n.metric) for f in result.historical_claims_findings for n in f.numerical_findings if n.metric.endswith('_total')]
    assert (Decimal('100'), 'KES', 'historical_claimed_total') in totals
    assert (Decimal('200'), 'USD', 'historical_claimed_total') in totals
    assert not any(value == 300 for value, _, _ in totals)


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_text', ['Annual loss is 999999 KES', 'Probability is 47%', 'AAL is one million', 'A second sentence. With an unsupported conclusion'])
async def test_llm_cannot_introduce_statistics(services, bad_text):
    assessment = build_document(services.store)
    class UnsupportedLLM(ExplainingLLM):
        async def structured(self, system, user, schema):
            finding = json.loads(user)['findings'][0]['id']
            return schema(summary=[{'finding_id': finding, 'explanation': bad_text}], drivers=[], actions=[])
    result = await UnderwritingAnalysisService(services.store, UnsupportedLLM()).generate(assessment)
    assert bad_text not in result.risk_summary
    assert '999999' not in result.model_dump_json()
    assert result.risk_level.value is None
    assert any(l.code == 'ai_narrative_unavailable' for l in result.limitations)


@pytest.mark.asyncio
async def test_unknown_finding_refs_rejected_without_fabrication(services):
    assessment = build_document(services.store)
    class BadReference:
        async def structured(self, system, user, schema):
            return schema(summary=[{'finding_id': 'nonexistent', 'explanation': 'Unsupported claim'}], drivers=[], actions=[])
    result = await UnderwritingAnalysisService(services.store, BadReference()).generate(assessment)
    assert 'Unsupported claim' not in result.risk_summary
    assert result.model_findings == []


@pytest.mark.asyncio
async def test_actual_backend_fixture_concentration_is_deterministic(services):
    assessment = build_document(services.store)
    # Static model adapter response fixture; no model calculation exists in this test.
    backend = type('Backend', (), {})()
    backend.get_model_results = AsyncMock(return_value=ModelOutput(assessment_id=assessment, status='ready', model_version='TEST_ONLY', result_id='fixture-model-run', period_results=[{'return_period': 100, 'currency': 'KES', 'total_loss': '100', 'building_losses': [{'loc_id': 'asset-a', 'loss': '80', 'housing_class': 'TEST_CLASS'}, {'loc_id': 'asset-b', 'loss': '20', 'housing_class': 'TEST_CLASS'}]}, {'return_period': 500, 'currency': 'KES', 'total_loss': '150', 'building_losses': []}]))
    result = await UnderwritingAnalysisService(services.store, services.extraction.llm, model_backend=backend).generate(assessment)
    shares = [n.value for f in result.model_findings for n in f.numerical_findings if n.metric == 'share_of_reported_portfolio_loss']
    assert shares == [Decimal('80'), Decimal('20')]
    differences = [n.value for f in result.model_findings for n in f.numerical_findings if n.metric == 'difference_between_reported_scenario_totals']
    assert differences == [50]
    assert any(c.model_result_id == 'fixture-model-run' for c in result.citations)
    assert result.risk_level.value is None


@pytest.mark.asyncio
async def test_missing_hazard_and_zero_model_loss_do_not_create_percentages(services):
    assessment = build_document(services.store)
    backend = type('Backend', (), {})()
    backend.get_model_results = AsyncMock(return_value=ModelOutput(assessment_id=assessment, status='missing_hazard_coverage', missing_return_periods=[100]))
    service = UnderwritingAnalysisService(services.store, services.extraction.llm, model_backend=backend)
    missing = await service.generate(assessment)
    assert missing.model_findings == []
    assert any(l.code == 'missing_hazard_coverage' for l in missing.limitations)
    backend.get_model_results = AsyncMock(return_value=ModelOutput(assessment_id=assessment, status='ready', model_version='TEST_ONLY', result_id='zero', period_results=[{'return_period': 100, 'currency': 'KES', 'total_loss': '0', 'building_losses': [{'loc_id': 'a', 'loss': '0'}]}]))
    zero = await service.generate(assessment)
    assert not any(n.metric == 'share_of_reported_portfolio_loss' for f in zero.model_findings for n in f.numerical_findings)
    assert any(n.value == 0 for f in zero.model_findings for n in f.numerical_findings)


@pytest.mark.asyncio
async def test_empty_assessment_and_unavailable_llm_return_strict_partial_json(services):
    assessment = services.store.create('alice')
    class Unavailable:
        async def structured(self, *args):
            raise RuntimeError('CONFIDENTIAL PROVIDER MESSAGE')
    result = await UnderwritingAnalysisService(services.store, Unavailable()).generate(assessment)
    assert result.top_risk_drivers == []
    assert 'CONFIDENTIAL' not in result.model_dump_json()
    assert result.risk_level.value is None
    assert any(l.code == 'missing_document_facts' for l in result.limitations)


def test_sync_entry_point_and_authorized_api(services):
    assessment = build_document(services.store)
    result = generate_underwriting_analysis(assessment, store=services.store, llm=services.extraction.llm)
    assert set(result) == {'risk_summary', 'risk_level', 'underwriting_decision', 'top_risk_drivers', 'historical_claims_findings', 'model_findings', 'recommended_actions', 'limitations', 'citations'}
    assert result['underwriting_decision']['recommendation'] == 'refer'
    assert result['underwriting_decision']['binding_status'] == 'not_ready_to_quote'
    application = create_app(services.settings, services)
    principal = {'subject': 'alice'}
    application.dependency_overrides[get_principal] = lambda: Principal(subject=principal['subject'])
    with TestClient(application) as client:
        route = f'/api/ai/assessments/{assessment}/underwriting-analysis'
        response = client.post(route)
        assert response.status_code == 200
        UnderwritingAnalysis.model_validate(response.json()['data'])
        principal['subject'] = 'bob'
        assert client.post(route).status_code == 404


@pytest.mark.asyncio
async def test_document_amount_not_in_source_is_excluded(services):
    assessment = build_document(services.store)
    document = services.store.insurance_results(assessment)[0]
    document.facts.flood_history[0].historical_claim_value.value.amount = Decimal('999999')
    services.store.save_insurance_extraction(assessment, document)
    result = await UnderwritingAnalysisService(services.store, services.extraction.llm).generate(assessment)
    assert not any(n.value == 999999 for f in result.historical_claims_findings for n in f.numerical_findings)
    assert any(l.code == 'numerical_evidence_unverified' for l in result.limitations)


@pytest.mark.asyncio
async def test_fire_claims_do_not_become_flood_statistics(services):
    from ai.schemas import Claim, DocumentExtraction, FinancialValue, SourceReference
    assessment = services.store.create('alice')
    text = 'Fire claim FIRE1 paid KES 9000'
    services.store.add_document(assessment, ParsedDocument(document_id='doc', segments=[DocumentSegment(locator='page:1', page_number=1, text=text)]))
    services.store.save_extractions(assessment, {'doc': DocumentExtraction(claims=[Claim(claim_id='FIRE1', cause='fire', paid_amount=FinancialValue(amount=9000, currency='KES'), sources=[SourceReference(document_id='doc', locator='page:1', page_number=1, excerpt=text)])])})
    result = await UnderwritingAnalysisService(services.store, services.extraction.llm).generate(assessment)
    assert result.historical_claims_findings == []


@pytest.mark.asyncio
async def test_duplicate_model_rows_never_create_concentration_statistics(services):
    assessment = build_document(services.store)
    backend = type('Backend', (), {})()
    backend.get_model_results = AsyncMock(return_value=ModelOutput(assessment_id=assessment, status='ready', model_version='TEST_ONLY', result_id='duplicate-fixture', period_results=[{'return_period': 100, 'currency': 'KES', 'total_loss': '100', 'building_losses': [{'loc_id': 'duplicate', 'loss': '80'}, {'loc_id': 'duplicate', 'loss': '20'}]}]))
    result = await UnderwritingAnalysisService(services.store, services.extraction.llm, model_backend=backend).generate(assessment)
    assert not any(n.metric == 'share_of_reported_portfolio_loss' for f in result.model_findings for n in f.numerical_findings)
    assert any('repeated building IDs' in l.message for l in result.limitations)


def test_supplied_grain_offer_risk_analysis_when_available(tmp_path):
    import os
    from ai.insurance_extraction import InsuranceDocumentExtractor
    from ai.document_parser import DocumentParser
    from tests.test_sample_offer import sample_path
    path = sample_path()
    if os.getenv('NZOIA_RUN_LIVE_EXTRACTION') != '1':
        pytest.skip('Set NZOIA_RUN_LIVE_EXTRACTION=1 and configure provider credentials for the actual offer')
    settings = Settings(storage_directory=str(tmp_path))
    live = build_services(settings)
    async def run():
        assessment = live.store.create('integration-test')
        parsed = DocumentParser(settings).parse_file(path, 'actual-offer')
        live.store.add_document(assessment, parsed)
        extraction = await InsuranceDocumentExtractor(live.extraction.llm, settings).extract_document(parsed)
        live.store.save_insurance_extraction(assessment, extraction)
        result = await UnderwritingAnalysisService(live.store, live.extraction.llm).generate(assessment)
        assert result.historical_claims_findings
        assert all(c.source.page_number for c in result.citations if c.kind == 'document')
        assert result.model_findings == []
        if getattr(live.extraction.llm, 'aclose', None):
            await live.extraction.llm.aclose()
    try:
        asyncio.run(run())
    finally:
        live.store.close()


def test_partial_date_overlap_does_not_create_distinct_flood_counts():
    from ai.risk_calculations import event_identities_disjoint, historical_totals
    assert not event_identities_disjoint(['2022', 'May 2022'])
    assert event_identities_disjoint(['May 2022', 'June 2022'])
    rows = [{'stream': 'document_history', 'kind': 'claimed', 'event_key': key, 'amount': Decimal('100'), 'currency': 'KES', 'citation_ids': ['source']} for key in ['2022', '2022-05-01']]
    totals, _, notes = historical_totals('test-assessment', rows)
    assert totals == [] and notes


@pytest.mark.asyncio
async def test_calculation_audit_references_actual_functions(services):
    import importlib
    assessment = build_document(services.store)
    result = await UnderwritingAnalysisService(services.store, services.extraction.llm).generate(assessment)
    for citation in result.citations:
        if citation.kind == 'calculation':
            module_name, function_name = citation.function.rsplit('.', 1)
            assert callable(getattr(importlib.import_module(module_name), function_name))
            assert citation.formula and citation.inputs
