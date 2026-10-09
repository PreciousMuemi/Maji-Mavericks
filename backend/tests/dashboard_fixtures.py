"""Explicit synthetic documents and static model adapter fixtures, never live estimates."""
import json
from unittest.mock import AsyncMock

from ai.exposure_mapping import map_to_exposure_schema
from ai.exposure_schemas import ExposureContract, VulnerabilityClass
from ai.insurance_schemas import DocumentClassification, InsuranceDocumentResult, InsuranceFacts
from ai.schemas import DocumentSegment, ParsedDocument
from ai.underwriting_schemas import ModelOutput


def assessment_fixture(store, *, complete=True, owner='alice'):
    assessment_id = store.create(owner)
    doc_id = 'synthetic-dashboard-offer'
    lines = ['Synthetic insurance offer for API testing, not a real Nzoia assessment']

    def fact(value, excerpt):
        lines.append(excerpt)
        return {'status': 'provided', 'value': value, 'confidence': 0.99,
                'sources': [{'document_id': doc_id, 'locator': 'page:1', 'page_number': 1, 'excerpt': excerpt}]}

    def money(amount, excerpt):
        return fact({'amount': str(amount), 'currency': 'KES'}, excerpt)

    assets = []
    for index, name in enumerate(['Building Alpha', 'Building Beta']):
        asset = {'name': fact(name, name + ' condition: poor'), 'condition': fact('poor', name + ' condition: poor')}
        if complete:
            asset.update(construction_type=fact('reinforced masonry', name + ' construction: reinforced masonry'),
                         stated_value=money((index + 1) * 1000, name + f' insured value: KES {(index + 1) * 1000}'),
                         floor_area=fact({'value': '200', 'unit': 'm2'}, name + ' floor area: 200 m2'),
                         coordinates=fact({'latitude': 0.1 + index / 100, 'longitude': 34.2 + index / 100}, name + f' GPS: {0.1 + index / 100}, {34.2 + index / 100}'))
        assets.append(asset)
    financial = {'property_sum_insured': money(10000, 'Overall property sum insured: KES 10000'),
                 'inventory_values': money(4000, 'Inventory total: KES 4000')}
    if complete:
        financial.update(inventory_breakdown=[{'name': fact('Raw materials', 'Raw materials: KES 1500'), 'stated_value': money(1500, 'Raw materials: KES 1500')},
                                              {'name': fact('Finished goods', 'Finished goods: KES 2500'), 'stated_value': money(2500, 'Finished goods: KES 2500')}],
                         inventory_breakdown_disjoint=fact(True, 'Raw materials and finished goods are mutually exclusive, non-overlapping inventory categories'))
    facts = InsuranceFacts.model_validate({'insured': {'facility_location': fact('Synthetic facility', 'Facility: Synthetic facility'),
                'coordinates': fact({'latitude': 0.1, 'longitude': 34.2}, 'Facility GPS: 0.1, 34.2')},
            'assets': assets, 'financial_exposure': financial,
            'insurance_terms': {'requested_flood_limit': money(2000, 'Requested flood limit: KES 2000')} if complete else {},
            'risk_factors': {'drainage_weaknesses': [fact('Inadequate drainage', 'Drainage inspection: Inadequate drainage')]},
            'flood_history': [{'event_date': fact(str(year), f'Flood in {year}: claimed KES {amount}; settled KES {amount // 2}'),
                              'historical_claim_value': money(amount, f'Flood in {year}: claimed KES {amount}; settled KES {amount // 2}'),
                              'settlement_value': money(amount // 2, f'Flood in {year}: claimed KES {amount}; settled KES {amount // 2}')}
                             for year, amount in [(2019, 100), (2020, 300)]]})
    store.add_document(assessment_id, ParsedDocument(document_id=doc_id, segments=[DocumentSegment(locator='page:1', page_number=1, text='\n'.join(lines))]))
    result = InsuranceDocumentResult(document_id=doc_id, document_type=DocumentClassification(kind='insurance_offer', confidence=1, sources=facts.assets[0].name.sources, reason='Synthetic offer fixture'), facts=facts, review_status='ready' if complete else 'requires_review')
    store.save_insurance_extraction(assessment_id, result)
    if complete:
        contract = ExposureContract(version='TEST_ONLY_V1', verified=True, verified_by='test-only modelling catalog', vulnerability_classes=[VulnerabilityClass(name='TEST_ONLY_CLASS', occupancies=['industrial'], asset_categories=['buildings'], description='Synthetic accepted class; not a Nzoia class')])
        rows = [{'loc_id': name, 'lat': str(0.1 + index / 100), 'lon': str(34.2 + index / 100), 'housing_class': 'TEST_ONLY_CLASS', 'floor_area_m2': '200', 'tiv_kes': str((index + 1) * 1000), 'coordinate_scope': 'building', 'occupancy': 'industrial'} for index, name in enumerate(['alpha', 'beta'])]
        mapping = map_to_exposure_schema(rows, contract=contract)
        records = mapping.valid_records
        assert len(records) == 2
        source_id = records[0].sources['loc_id'][0].document_id
        store.add_document(assessment_id, ParsedDocument(document_id=source_id, segments=[DocumentSegment(locator=f'row:{index}', text=json.dumps(row)) for index, row in enumerate(rows, 1)]))
        store.save_exposure_mapping(assessment_id, mapping, source_hash=store.dashboard_snapshot(assessment_id)['source_hash'])
    return assessment_id, result


def backend_fixture(assessment_id, *, status='ready'):
    value = ModelOutput(assessment_id=assessment_id, status=status,
            model_version='TEST_ONLY_MODEL_V1' if status == 'ready' else None,
            result_id='synthetic-adapter-response' if status == 'ready' else None,
            assumptions=['Synthetic test assumption: building loss results only; industrial loss parameters are excluded'] if status == 'ready' else [],
            period_results=[{'return_period': 100, 'currency': 'KES', 'total_loss': '500', 'building_losses': [{'loc_id': 'alpha', 'loss': '100', 'housing_class': 'TEST_ONLY_CLASS'}, {'loc_id': 'beta', 'loss': '400', 'housing_class': 'TEST_ONLY_CLASS'}]},
                            {'return_period': 500, 'currency': 'KES', 'total_loss': '1000', 'building_losses': [{'loc_id': 'alpha', 'loss': '300', 'housing_class': 'TEST_ONLY_CLASS'}, {'loc_id': 'beta', 'loss': '700', 'housing_class': 'TEST_ONLY_CLASS'}]}] if status == 'ready' else [])
    backend = type('StaticModelFixture', (), {})()
    backend.get_model_results = AsyncMock(return_value=value)
    backend.run_flood_model = AsyncMock(side_effect=AssertionError('Dashboard GET must never simulate'))
    backend.run_scenario = AsyncMock(side_effect=AssertionError('Dashboard GET must never simulate'))
    return backend
