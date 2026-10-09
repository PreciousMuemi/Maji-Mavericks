"""QA safety regressions with explicit test doubles; these are NOT live AI tests."""
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from ai.config import Settings
from ai.extraction_service import ExtractionError, ExtractionService
from ai.exposure_mapper import CanonicalExposureMapper
from ai.risk_analyzer import IntegrationUnavailable, RiskAnalyzer
from ai.schemas import AIAnalysis, Asset, Claim, DocumentExtraction, DocumentSegment, FinancialValue, ParsedDocument, SourceReference
from ai.store import AssessmentStore
from api.ai_routes import build_services, get_principal
from app.main import create_app
from ai.underwriting_schemas import Principal
from tests.dashboard_fixtures import assessment_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['insured_amount', 'coordinates', 'paid_amount', 'event_date', 'asset_name', 'site_as_building'])
async def test_canonical_llm_values_must_match_field_evidence(field):
    text = 'Warehouse A insured value: KES 100; building GPS 0.2, 34.1\nClaim C1: flood on 2020-02-01; paid KES 20\nFacility GPS: 0.5, 34.5'
    source = SourceReference(document_id='qa-doc', locator='page:1', page_number=1, excerpt=text)
    document = ParsedDocument(document_id='qa-doc', segments=[DocumentSegment(locator='page:1', page_number=1, text=text)])
    asset = Asset(asset_id='A', name='Warehouse A', asset_type='building', sources=[source])
    claim = Claim(claim_id='C1', sources=[source])
    if field == 'insured_amount':
        asset.insured_value = FinancialValue(amount=999999, currency='KES')
    elif field == 'coordinates':
        asset.latitude, asset.longitude = 10, 70
    elif field == 'site_as_building':
        asset.latitude, asset.longitude = 0.5, 34.5
    elif field == 'paid_amount':
        claim.paid_amount = FinancialValue(amount=999999, currency='KES')
    elif field == 'event_date':
        claim.event_date = date(2020, 3, 1)
    else:
        asset.name = 'Invented Building'
    class ProviderDouble:
        async def structured(self, *args):
            return DocumentExtraction(assets=[asset], claims=[claim])
    with pytest.raises(ExtractionError):
        await ExtractionService(ProviderDouble()).extract(document)


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_output', [{}, {'status': 'no_results'}, {'status': 'missing_hazard_coverage'}, {'loss': 12}])
async def test_legacy_model_empty_or_untyped_outputs_are_not_success(bad_output):
    class ModelDouble:
        async def analyze(self, *args):
            return bad_output
    extraction = DocumentExtraction(assets=[Asset(asset_id='A', name='QA Building', latitude=0.2, longitude=34.1, insured_value=FinancialValue(amount=100, currency='KES'))])
    with pytest.raises(IntegrationUnavailable):
        await RiskAnalyzer(ModelDouble(), CanonicalExposureMapper()).analyze('qa-assessment', extraction)


def test_new_document_invalidates_old_findings_and_overwrite_invalidates_extraction(tmp_path):
    store = AssessmentStore(str(tmp_path))
    try:
        assessment, rich = assessment_fixture(store, complete=False)
        store.save_analysis(assessment, AIAnalysis(summary='OLD CALCULATION'))
        store.add_document(assessment, ParsedDocument(document_id='new-doc', segments=[DocumentSegment(locator='row:1', text='New schedule')]))
        assert store.analysis(assessment) is None
        canonical = DocumentExtraction(assets=[Asset(asset_id='A', name='Old building')])
        store.save_extractions(assessment, {rich.document_id: canonical})
        parsed = store.documents(assessment)[rich.document_id]
        parsed.segments[0].text = 'Replacement offer with different facts'
        store.add_document(assessment, parsed)
        assert store.insurance_extraction(assessment, rich.document_id) is None
        assert store.extraction(assessment).assets == []
    finally:
        store.close()


def test_scenario_http_parameters_cannot_override_assessment_identity(tmp_path):
    services = build_services(Settings(_env_file=None, storage_directory=str(tmp_path)))
    try:
        app = create_app(services=services)
        with TestClient(app) as client:
            assessment = client.post('/api/ai/assessments').json()['assessment_id']
            response = client.post(f'/api/ai/assessments/{assessment}/scenarios', json={'name': 'QA scenario', 'parameters': {'nested': [{'assessment_id': 'OTHER-ASSESSMENT'}]}})
            assert response.status_code == 422
            assert 'OTHER-ASSESSMENT' not in response.text
    finally:
        services.store.close()


def test_findings_include_persisted_rich_document_review_items(tmp_path):
    from ai.insurance_schemas import ReviewItem
    services = build_services(Settings(_env_file=None, storage_directory=str(tmp_path)))
    try:
        assessment, rich = assessment_fixture(services.store, complete=False)
        rich.review_items.append(ReviewItem(code='contradiction', field_path='facts.insured.coordinates', message='Coordinate interpretations conflict'))
        services.store.save_insurance_extraction(assessment, rich)
        app = create_app(services=services)
        app.dependency_overrides[get_principal] = lambda: Principal(subject='alice')
        with TestClient(app) as client:
            response = client.get(f'/api/ai/assessments/{assessment}/findings')
            assert response.status_code == 200
            assert any(f['code'] == 'contradiction' for f in response.json()['data']['findings'])
    finally:
        services.store.close()


@pytest.mark.asyncio
async def test_canonical_grounded_numeric_fields_and_source_page_are_preserved():
    from tests.test_extraction import StubLLM
    asset_text = 'Warehouse A insured value: KES 100; building GPS 0.2, 34.1'
    claim_text = 'Claim C1: flood on 2020-02-01; paid KES 20'
    sources = lambda text: [SourceReference(document_id='qa', locator='page:2', excerpt=text)]
    document = ParsedDocument(document_id='qa', segments=[DocumentSegment(locator='page:2', page_number=2, text=asset_text + '\n' + claim_text)])
    output = DocumentExtraction(assets=[Asset(asset_id='A', name='Warehouse A', asset_type='building', latitude=0.2, longitude=34.1, insured_value=FinancialValue(amount=100, currency='KES'), sources=sources(asset_text))], claims=[Claim(claim_id='C1', event_date='2020-02-01', paid_amount=FinancialValue(amount=20, currency='KES'), sources=sources(claim_text))])
    result = await ExtractionService(StubLLM(output)).extract(document)
    assert result.assets[0].insured_value.amount == 100
    assert result.claims[0].paid_amount.amount == 20
    assert all(s.page_number == 2 for a in [*result.assets, *result.claims] for s in a.sources)
    output.assets[0].sources[0].page_number = 99
    with pytest.raises(ExtractionError):
        await ExtractionService(StubLLM(output)).extract(document)


@pytest.mark.asyncio
async def test_legacy_output_accepts_actual_typed_zero_and_rejects_wrong_assessment():
    from ai.risk_analyzer import validate_model_output
    output = {'assessment_id': 'qa', 'status': 'ready', 'model_version': 'TEST_ONLY', 'result_id': 'static-unit-response', 'period_results': [{'return_period': 100, 'currency': 'KES', 'total_loss': '0'}]}
    assert validate_model_output(output, 'qa')['period_results'][0]['total_loss'] == '0'
    with pytest.raises(IntegrationUnavailable):
        validate_model_output(output, 'another-assessment')


def test_empty_canonical_extraction_is_partial_and_reviews_stay_partial(tmp_path):
    class EmptyProviderDouble:
        async def structured(self, *args):
            return DocumentExtraction()
    services = build_services(Settings(_env_file=None, storage_directory=str(tmp_path)), llm=EmptyProviderDouble())
    try:
        with TestClient(create_app(services=services)) as client:
            assessment = client.post('/api/ai/assessments').json()['assessment_id']
            base = f'/api/ai/assessments/{assessment}'
            client.post(base + '/documents', files={'file': ('test.csv', b'name\nUnknown\n')})
            result = client.post(base + '/extract').json()
            assert result['status'] == 'partial'
            assert result['warnings']
        assessment, rich = assessment_fixture(services.store, complete=False)
        assert rich.review_status == 'requires_review' and not rich.review_items
        app = create_app(services=services)
        app.dependency_overrides[get_principal] = lambda: Principal(subject='alice')
        with TestClient(app) as client:
            result = client.get(f'/api/ai/assessments/{assessment}/documents/{rich.document_id}/insurance-extraction').json()
            assert result['status'] == 'partial'
    finally:
        services.store.close()


@pytest.mark.asyncio
async def test_direct_analysis_and_scenario_timeouts_are_bounded():
    import asyncio
    from ai.scenario_service import ScenarioService
    from ai.schemas import ScenarioRequest
    class SlowModelDouble:
        async def analyze(self, *args):
            await asyncio.sleep(1)
        async def scenario(self, *args):
            await asyncio.sleep(1)
    analyzer = RiskAnalyzer(SlowModelDouble(), CanonicalExposureMapper(), timeout_seconds=0.01)
    extraction = DocumentExtraction(assets=[Asset(asset_id='A', name='QA Building', latitude=0.2, longitude=34.1, insured_value=FinancialValue(amount=100, currency='KES'))])
    with pytest.raises(IntegrationUnavailable, match='timed out'):
        await analyzer.analyze('qa', extraction)
    with pytest.raises(IntegrationUnavailable, match='timed out'):
        await ScenarioService(analyzer).run('qa', extraction, ScenarioRequest(name='test-only'))


@pytest.mark.parametrize('payload', [b'loc_id,name\n001,Building A,extra\n', b'loc_id,loc_id\n001,002\n'])
def test_csv_never_silently_drops_or_renames_source_columns(payload):
    from ai.document_parser import DocumentError, DocumentParser
    with pytest.raises(DocumentError):
        DocumentParser(Settings(_env_file=None)).parse(payload, 'bad_schedule.csv', 'qa-csv')


def test_flood_year_cannot_be_extracted_as_a_depth_value():
    from ai.insurance_extraction import _validate_numeric
    from ai.insurance_schemas import Measurement
    source = SourceReference(document_id='qa', locator='page:1', excerpt='Flood event 2019; observed depth 2 m')
    with pytest.raises(ExtractionError):
        _validate_numeric(Measurement(value=2019, unit='m'), [source])
    _validate_numeric(Measurement(value=2, unit='m'), [source])


def test_spreadsheet_measurement_uses_same_column_header_unit():
    from ai.insurance_extraction import _validate_numeric
    from ai.insurance_schemas import Measurement
    header = SourceReference(document_id='qa', locator='sheet:Schedule:row:1', excerpt='["Year", "Floor area m2"]')
    row = SourceReference(document_id='qa', locator='sheet:Schedule:row:2', excerpt='["2019", "120"]')
    _validate_numeric(Measurement(value=120, unit='m2'), [header, row])
    with pytest.raises(ExtractionError):
        _validate_numeric(Measurement(value=2019, unit='m2'), [header, row])


def test_inflight_legacy_calculation_cannot_save_results_after_source_change(tmp_path):
    services = build_services(Settings(_env_file=None, storage_directory=str(tmp_path)))
    try:
        assessment = services.store.create('local-development')
        services.store.add_document(assessment, ParsedDocument(document_id='source', segments=[]))
        services.store.save_extractions(assessment, {'source': DocumentExtraction(assets=[Asset(asset_id='A', name='QA Building', latitude=0.2, longitude=34.1, insured_value=FinancialValue(amount=100, currency='KES'))])})
        class ChangingSourceModelDouble:
            async def analyze(self, assessment_id, exposures):
                services.store.add_document(assessment_id, ParsedDocument(document_id='new-source', segments=[]))
                return {'assessment_id': assessment_id, 'status': 'ready', 'model_version': 'TEST_ONLY', 'result_id': 'outdated-unit-response', 'period_results': [{'return_period': 100, 'currency': 'KES', 'total_loss': '10'}]}
        services.analyzer.model = ChangingSourceModelDouble()
        with TestClient(create_app(services=services)) as client:
            response = client.post(f'/api/ai/assessments/{assessment}/analyze')
            assert response.status_code == 503
            assert response.json()['data'] is None
            assert services.store.analysis(assessment) is None
    finally:
        services.store.close()
