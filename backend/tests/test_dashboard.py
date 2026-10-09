"""Integration tests compare API read models with persisted facts and adapter responses."""
import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from ai.config import Settings
from ai.insurance_schemas import Fact, Interpretation, Money
from ai.schemas import Claim, DocumentExtraction, DocumentSegment, FinancialValue, ParsedDocument, SourceReference
from ai.underwriting_schemas import Principal
from api.ai_routes import build_services, get_principal
from app.main import create_app
from app.schemas.dashboard import DashboardResponse
from tests.dashboard_fixtures import assessment_fixture, backend_fixture
from tests.test_underwriting_analysis import ExplainingLLM


@pytest.fixture
def setup(tmp_path):
    services = build_services(Settings(_env_file=None, storage_directory=str(tmp_path)), llm=ExplainingLLM())
    yield services
    services.store.close()


def get_dashboard(services, assessment, *, subject='alice', headers=None):
    app = create_app(services=services)
    app.dependency_overrides[get_principal] = lambda: Principal(subject=subject)
    with TestClient(app) as client:
        return client.get(f'/api/assessments/{assessment}/dashboard', headers=headers)


def cards(data):
    return {k['id']: k for k in data['kpis']}


def charts(data):
    return {c['id']: c for c in data['charts']}


def replace_result(services, assessment, result):
    services.store.save_insurance_extraction(assessment, result)


def test_complete_dashboard_matches_validated_backend_records(setup):
    assessment, extraction = assessment_fixture(setup.store)
    setup.model_backend = backend_fixture(assessment)
    response = get_dashboard(setup, assessment)
    assert response.status_code == 200, response.text
    envelope = response.json()
    data = envelope['data']
    DashboardResponse.model_validate(data)
    assert envelope['status'] == 'success'
    assert data['status'] == 'calculation_completed'
    assert {'document_extracted', 'model_ready', 'calculation_completed'} <= set(data['statuses'])
    assert 'validation_required' not in data['statuses']
    k = cards(data)
    assert Decimal(k['property_sum_insured']['value']) == extraction.facts.financial_exposure.property_sum_insured.value.amount
    assert k['property_sum_insured']['display_value'] == 'KES 10,000.00'
    assert k['requested_flood_limit']['value'] == '2000'
    assert k['historical_flood_claims']['value'] == '400'  # excludes settled 200
    assert k['number_of_assets']['value'] == '2'
    output = setup.model_backend.get_model_results.return_value
    assert Decimal(k['rp100_modelled_loss']['value']) == output.period_results[0].total_loss
    c = charts(data)
    assert [p['value'] for p in c['historical_flood_claims_by_year']['points']] == ['100', '300']
    assert c['exposure_by_construction_class']['points'][0]['value'] == '3000'  # never property total 10000
    assert [p['value'] for p in c['inventory_breakdown']['points']] == ['1500', '2500']
    assert [p['value'] for p in c['return_period_loss_curve']['points']] == ['500', '1000']
    assert [p['value'] for p in c['scenario_comparison']['points']] == ['500', '1000']
    ranking = c['building_loss_ranking_rp100']['points']
    assert [(p['label'], p['value']) for p in ranking] == [('beta', '400'), ('alpha', '100')]
    assert len(data['ai_insights']['risk_drivers']) <= 3
    assert data['ai_insights']['origin'] == 'deterministic_evidence'
    sources = {s['id']: s for s in data['source_references']}
    assert all(sources[s]['source']['page_number'] == 1 for s in k['property_sum_insured']['citation_ids'])
    assert any(s['model_result_id'] == output.result_id for s in data['source_references'])
    assert any(s['formula'] for s in data['source_references'] if s['kind'] == 'calculation')
    setup.model_backend.get_model_results.assert_awaited_once_with(assessment)
    setup.model_backend.run_flood_model.assert_not_awaited()
    setup.model_backend.run_scenario.assert_not_awaited()


def test_partial_dashboard_preserves_historical_data_without_model(setup):
    assessment, _ = assessment_fixture(setup.store, complete=False)
    response = get_dashboard(setup, assessment)
    assert response.status_code == 200, response.text
    data = response.json()['data']
    assert data['status'] == 'partial_assessment'
    assert cards(data)['rp100_modelled_loss']['value'] is None
    assert cards(data)['requested_flood_limit']['value'] is None
    assert cards(data)['number_of_assets']['value'] == '2'
    assert cards(data)['historical_flood_claims']['value'] == '400'
    assert set(charts(data)) == {'historical_flood_claims_by_year'}
    assert data['map']['coverage_status'] == 'unverified'
    assert [(p['coordinate_scope'], p['lat'], p['lon']) for p in data['map']['locations']] == [('site', 0.1, 34.2)]
    assert any(l['code'] == 'missing_building_model_inputs' for l in data['ai_insights']['missing_data'])
    assert 'partial' == response.json()['status']
    assert 'hazard_coverage_unavailable' not in data['statuses']  # absent integration does not prove hazard absence


@pytest.mark.parametrize('mode', ['no_results', 'missing_hazard_coverage', 'timeout', 'failure', 'wrong_assessment', 'invalid_output'])
def test_model_failure_does_not_invent_loss_or_lose_document_data(setup, mode):
    assessment, _ = assessment_fixture(setup.store, complete=False)
    backend = backend_fixture(assessment, status=mode if mode in {'no_results', 'missing_hazard_coverage'} else 'no_results')
    if mode == 'failure':
        backend.get_model_results.side_effect = RuntimeError('CONFIDENTIAL BACKEND DETAIL')
    elif mode == 'timeout':
        setup.settings.agent_tool_timeout_seconds = 0.01
        async def slow(_):
            await asyncio.sleep(1)
        backend.get_model_results.side_effect = slow
    elif mode == 'wrong_assessment':
        backend = backend_fixture('SECRET-OTHER-ASSESSMENT')
    elif mode == 'invalid_output':
        backend.get_model_results.return_value = {'assessment_id': assessment, 'status': 'ready', 'period_results': [{'return_period': 100, 'total_loss': 'NaN', 'currency': 'KES'}]}
    setup.model_backend = backend
    response = get_dashboard(setup, assessment)
    assert response.status_code == 200
    data = response.json()['data']
    assert cards(data)['property_sum_insured']['value'] == '10000'
    assert cards(data)['rp100_modelled_loss']['value'] is None
    assert all(c['basis'] != 'modelled' for c in data['charts'])
    assert 'CONFIDENTIAL' not in response.text
    assert 'SECRET-OTHER' not in response.text
    if mode == 'missing_hazard_coverage':
        assert data['status'] == 'hazard_coverage_unavailable'
        assert data['map']['coverage_status'] == 'unavailable'


def test_missing_zero_and_generic_limits_are_distinct(setup):
    assessment, extraction = assessment_fixture(setup.store)
    # Source-backed zero is valid; an arbitrary overall limit is not a flood limit.
    parsed = setup.store.documents(assessment)[extraction.document_id]
    parsed.segments[0].text += '\nOverall property sum insured: KES 0\nOverall limit: KES 2000'
    setup.store.add_document(assessment, parsed)
    fact = extraction.facts.financial_exposure.property_sum_insured
    fact.value.amount = Decimal('0')
    fact.sources[0].excerpt = 'Overall property sum insured: KES 0'
    extraction.facts.insurance_terms.requested_flood_limit = Fact[Money]()
    extraction.facts.insurance_terms.limits = Fact[str](value='Overall limit: KES 2000', status='provided', confidence=1,
        sources=[SourceReference(document_id=extraction.document_id, locator='page:1', page_number=1, excerpt='Overall limit: KES 2000')])
    replace_result(setup, assessment, extraction)
    setup.model_backend = backend_fixture(assessment)
    p = setup.model_backend.get_model_results.return_value.period_results[0]
    p.total_loss = Decimal('0')
    p.building_losses = []
    data = get_dashboard(setup, assessment).json()['data']
    assert cards(data)['property_sum_insured']['value'] == '0'
    assert cards(data)['property_sum_insured']['availability'] == 'available'
    assert cards(data)['property_sum_insured']['display_value'] == 'KES 0.00'
    assert cards(data)['requested_flood_limit']['value'] is None
    assert cards(data)['rp100_modelled_loss']['value'] == '0'
    assert not any('%' in p['display_value'] for c in data['charts'] for p in c['points'])


@pytest.mark.parametrize('field', ['property_sum_insured', 'requested_flood_limit'])
def test_uncertain_and_conflicting_financial_values_require_review(setup, field):
    assessment, result = assessment_fixture(setup.store)
    parent = result.facts.financial_exposure if field == 'property_sum_insured' else result.facts.insurance_terms
    fact = getattr(parent, field)
    fact.status = 'uncertain'
    fact.review_reason = 'Unclear coverage scope'
    replace_result(setup, assessment, result)
    data = get_dashboard(setup, assessment).json()['data']
    assert cards(data)[field]['availability'] == 'review_required'
    assert cards(data)[field]['value'] is None
    assert 'validation_required' in data['statuses']


def test_conflicting_property_amounts_and_duplicate_names_are_withheld(setup):
    assessment, result = assessment_fixture(setup.store, complete=False)
    fact = result.facts.financial_exposure.property_sum_insured
    parsed = setup.store.documents(assessment)[result.document_id]
    parsed.segments[0].text += '\nProperty sum insured correction: KES 12000'
    setup.store.add_document(assessment, parsed)
    second = fact.sources[0].model_copy(update={'excerpt': 'Property sum insured correction: KES 12000'})
    result.facts.financial_exposure.property_sum_insured = Fact[Money](status='contradicted', alternatives=[Interpretation(value=fact.value, sources=fact.sources), Interpretation(value=Money(amount=12000, currency='KES'), sources=[second])], review_reason='Contradictory reported totals')
    result.facts.assets.append(result.facts.assets[0].model_copy(deep=True))
    replace_result(setup, assessment, result)
    data = get_dashboard(setup, assessment).json()['data']
    assert cards(data)['property_sum_insured']['value'] is None
    assert cards(data)['number_of_assets']['value'] is None


def test_foreign_amounts_require_explicit_conversion(setup):
    assessment, result = assessment_fixture(setup.store)
    # Make source evidence agree with the foreign currency; do not implicitly convert.
    parsed = setup.store.documents(assessment)[result.document_id]
    parsed.segments[0].text = parsed.segments[0].text.replace('Overall property sum insured: KES 10000', 'Overall property sum insured: USD 10000')
    setup.store.add_document(assessment, parsed)
    value = result.facts.financial_exposure.property_sum_insured
    value.value.currency = 'USD'
    value.sources[0].excerpt = 'Overall property sum insured: USD 10000'
    replace_result(setup, assessment, result)
    setup.model_backend = backend_fixture(assessment)
    for period in setup.model_backend.get_model_results.return_value.period_results:
        period.currency = 'USD'
    data = get_dashboard(setup, assessment).json()['data']
    assert cards(data)['property_sum_insured']['value'] is None
    assert cards(data)['rp100_modelled_loss']['value'] is None
    assert not any(c['basis'] == 'modelled' for c in data['charts'])
    assert any(l['code'] == 'currency_conversion_required' for l in data['ai_insights']['limitations'])


def test_duplicate_or_overlapping_historical_events_never_double_count(setup):
    assessment, result = assessment_fixture(setup.store, complete=False)
    result.facts.flood_history.append(result.facts.flood_history[0].model_copy(deep=True))
    replace_result(setup, assessment, result)
    data = get_dashboard(setup, assessment).json()['data']
    assert cards(data)['historical_flood_claims']['value'] is None
    assert 'historical_flood_claims_by_year' not in charts(data)


def test_overlapping_inventory_and_duplicate_model_buildings_omit_charts(setup):
    assessment, result = assessment_fixture(setup.store)
    result.facts.financial_exposure.inventory_breakdown_disjoint = Fact[bool]()
    replace_result(setup, assessment, result)
    setup.model_backend = backend_fixture(assessment)
    p = setup.model_backend.get_model_results.return_value.period_results[0]
    p.building_losses.append(p.building_losses[0].model_copy())
    data = get_dashboard(setup, assessment).json()['data']
    assert 'inventory_breakdown' not in charts(data)
    assert 'building_loss_ranking_rp100' not in charts(data)
    assert cards(data)['rp100_modelled_loss']['value'] == '500'


def test_invalid_provenance_excluded_from_dashboard(setup):
    assessment, result = assessment_fixture(setup.store, complete=False)
    result.facts.financial_exposure.property_sum_insured.sources[0].page_number = 99
    replace_result(setup, assessment, result)
    data = get_dashboard(setup, assessment).json()['data']
    assert cards(data)['property_sum_insured']['value'] is None
    assert all(c['kind'] != 'document' or c['source']['page_number'] != 99 for c in data['source_references'])
    assert any(l['code'] == 'document_evidence_unverified' for l in data['ai_insights']['limitations'])


def test_stored_ai_analysis_and_source_model_revision_invalidation(setup):
    assessment, result = assessment_fixture(setup.store)
    setup.model_backend = backend_fixture(assessment)
    app = create_app(services=setup)
    app.dependency_overrides[get_principal] = lambda: Principal(subject='alice')
    with TestClient(app) as client:
        analysis = client.post(f'/api/ai/assessments/{assessment}/underwriting-analysis')
        assert analysis.status_code == 200, analysis.text
        setup.extraction.llm.structured = AsyncMock(side_effect=AssertionError('GET must never call the LLM'))
        response = client.get(f'/api/assessments/{assessment}/dashboard')
        assert response.status_code == 200, response.text
        data = response.json()['data']
        assert data['ai_insights']['origin'] == 'stored_ai_analysis'
        assert data['ai_insights']['risk_drivers'][0]['id'] == analysis.json()['data']['top_risk_drivers'][0]['id']
        setup.model_backend.get_model_results.return_value.result_id = 'new-model-run'
        data = client.get(f'/api/assessments/{assessment}/dashboard').json()['data']
        assert data['ai_insights']['origin'] == 'deterministic_evidence'
        assert any(l['code'] == 'briefing_stale' for l in data['ai_insights']['limitations'])
        setup.model_backend.get_model_results.return_value.result_id = 'synthetic-adapter-response'
        result.review_status = 'requires_review'
        replace_result(setup, assessment, result)
        data = client.get(f'/api/assessments/{assessment}/dashboard').json()['data']
        assert data['ai_insights']['origin'] == 'deterministic_evidence'
        assert 'exposure_by_construction_class' not in charts(data)  # old confirmed mapping must be revalidated against changed facts
        setup.extraction.llm.structured.assert_not_awaited()


def test_revision_race_and_cross_assessment_intelligence_rejected(setup):
    assessment, _ = assessment_fixture(setup.store)
    snapshot = setup.store.dashboard_snapshot(assessment)
    other = setup.store.create('alice')
    setup.store.add_document(assessment, ParsedDocument(document_id='new-doc', segments=[]))
    with pytest.raises(ValueError, match='changed'):
        setup.store.save_exposure_mapping(assessment, snapshot['mapping'], source_hash=snapshot['source_hash'])
    async def analysis():
        from ai.underwriting_analysis import UnderwritingAnalysisService
        return await UnderwritingAnalysisService(setup.store, setup.extraction.llm).generate(assessment)
    result = asyncio.run(analysis())
    assert not setup.store.save_intelligence(assessment, result, fingerprint=snapshot['fingerprint'])
    with pytest.raises(ValueError, match='another assessment'):
        setup.store.save_intelligence(other, result, fingerprint=setup.store.dashboard_snapshot(other)['fingerprint'])


def test_authorization_and_authentication_apply_to_dashboard(setup):
    assessment, _ = assessment_fixture(setup.store, complete=False)
    setup.model_backend = backend_fixture(assessment)
    denied = get_dashboard(setup, assessment, subject='bob')
    unknown = get_dashboard(setup, 'unknown', subject='bob')
    assert denied.status_code == unknown.status_code == 404
    assert denied.json()['errors'] == unknown.json()['errors']
    setup.model_backend.get_model_results.assert_not_awaited()
    from pydantic import SecretStr
    setup.settings.api_token = SecretStr('TEST_SECRET')
    assert get_dashboard(setup, assessment).status_code == 401
    assert get_dashboard(setup, assessment, headers={'Authorization': 'Bearer TEST_SECRET'}).status_code == 200


def test_empty_assessment_is_insufficient_not_zero(setup):
    assessment = setup.store.create('alice')
    data = get_dashboard(setup, assessment).json()['data']
    assert data['status'] == 'insufficient_data'
    assert data['charts'] == []
    assert data['map']['locations'] == []
    assert all(k['value'] is None for k in data['kpis'] if k['basis'] != 'workflow')


def test_paid_register_chart_uses_evidenced_event_dates_and_separate_stream(setup):
    assessment = setup.store.create('alice')
    source = SourceReference(document_id='register', locator='row:1', excerpt='claim-1 flood 2019-01-02 paid KES 25')
    setup.store.add_document(assessment, ParsedDocument(document_id='register', segments=[DocumentSegment(locator='row:1', text=source.excerpt)]))
    setup.store.save_extractions(assessment, {'register': DocumentExtraction(claims=[Claim(claim_id='claim-1', cause='flood', event_date='2019-01-02', paid_amount=FinancialValue(amount=25, currency='KES'), sources=[source])])})
    data = get_dashboard(setup, assessment).json()['data']
    assert cards(data)['historical_flood_claims']['value'] == '25'
    assert charts(data)['historical_flood_claims_by_year']['points'][0]['label'] == '2019'
    assert charts(data)['historical_flood_claims_by_year']['points'][0]['value'] == '25'


def test_supplied_csv_partial_dashboard_keeps_asset_count_and_locations(setup):
    from ai.exposure_mapping import inspect_exposure_csv, map_to_exposure_schema
    from tests.test_supplied_exposure_inputs import actual_csv
    path = actual_csv()
    observation = inspect_exposure_csv(path)
    batch = map_to_exposure_schema(path)
    assessment = setup.store.create('alice')
    source_id = batch.review_required_records[0].sources['loc_id'][0].document_id
    parsed = setup.parser.parse_file(path, source_id)
    setup.store.add_document(assessment, parsed)
    setup.store.save_exposure_mapping(assessment, batch, source_hash=setup.store.dashboard_snapshot(assessment)['source_hash'])
    data = get_dashboard(setup, assessment).json()['data']
    assert Decimal(cards(data)['number_of_assets']['value']) == observation.row_count
    assert cards(data)['property_sum_insured']['value'] is None
    assert cards(data)['rp100_modelled_loss']['value'] is None
    assert 'model_ready' not in data['statuses']
    assert data['status'] == 'partial_assessment'
    assert 'exposure_by_construction_class' not in charts(data)  # Observed class names do not verify engine applicability.
    assert data['map']['locations']
    source_points = {(float(r.normalized_values['lat']), float(r.normalized_values['lon'])) for r in batch.review_required_records}
    assert all((p['lat'], p['lon']) in source_points for p in data['map']['locations'])
    missing_groups = [l for l in data['ai_insights']['missing_data'] if l['code'] == 'missing_model_inputs']
    assert len(missing_groups) < observation.row_count
    assert all(l['citation_ids'] for l in missing_groups)


def test_supplied_pdf_without_extraction_or_llm_keeps_financials_unavailable(setup):
    from tests.test_sample_offer import sample_path
    assessment = setup.store.create('alice')
    parsed = setup.parser.parse_file(sample_path(), 'provided-offer')
    setup.store.add_document(assessment, parsed)
    setup.extraction.llm.structured = AsyncMock(side_effect=AssertionError('GET must never extract via an LLM'))
    data = get_dashboard(setup, assessment).json()['data']
    assert data['status'] == 'insufficient_data'
    assert data['charts'] == []
    assert all(k['value'] is None for k in data['kpis'] if k['basis'] != 'workflow')
    setup.extraction.llm.structured.assert_not_awaited()


def test_published_examples_and_json_schemas_match_current_contract():
    from pathlib import Path
    from ai.schemas import ResponseEnvelope
    root = Path(__file__).parents[1]
    for name in ('dashboard-success.synthetic.json', 'dashboard-partial.synthetic.json'):
        payload = json.loads((root / 'examples' / name).read_text())
        ResponseEnvelope[DashboardResponse].model_validate(payload)
        assert payload['errors'] == []
    for name, schema in [('dashboard.schema.json', ResponseEnvelope[DashboardResponse].model_json_schema(mode='serialization')),
                         ('dashboard-data.schema.json', DashboardResponse.model_json_schema(mode='serialization'))]:
        published = json.loads((root / 'docs' / name).read_text())
        assert published.pop('$schema') == 'https://json-schema.org/draft/2020-12/schema'
        assert published == schema
