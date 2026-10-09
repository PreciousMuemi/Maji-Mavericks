import io
from decimal import Decimal

import pandas as pd
import pytest

from ai.exposure_mapping import confirm_exposure_mapping, get_missing_model_inputs, inspect_exposure_csv, map_property_description, map_to_exposure_schema
from ai.exposure_schemas import Confirmation, ConstructionRule, ExposureContract, ModelField, VulnerabilityClass
from ai.insurance_schemas import InsuranceFacts


def contract():
    # Test-only contract; these are NOT asserted to be Nzoia housing classes.
    result = ExposureContract(version='test-contract-v1', verified=True, verified_by='test modelling team', vulnerability_classes=[VulnerabilityClass(name='TEST_BUILDING', occupancies=['unknown', 'industrial', 'residential'], asset_categories=['buildings'], description='Test-only building class')])
    return result


def row(**overrides):
    return {'loc_id': '001', 'lat': '0.5', 'lon': '34.2', 'housing_class': 'TEST_BUILDING', 'floor_area_m2': '120', 'tiv_kes': '100000', **overrides}


def decision(batch, record, field, value, **kwargs):
    return Confirmation(record_id=record.record_id, revision=batch.revision, field=field, value=value, approved=True, underwriter_id='test-underwriter', reason='Explicit test confirmation', evidence='Underwriter supplied supporting evidence', **kwargs)


def test_csv_actual_headers_observed_classes_and_leading_zero_ids(tmp_path):
    path = tmp_path / 'test_only_exposures.csv'
    pd.DataFrame([row()]).to_csv(path, index=False)
    observed = inspect_exposure_csv(path)
    assert observed.columns == list(row())
    assert observed.observed_housing_classes == ['TEST_BUILDING']
    result = map_to_exposure_schema(path, contract=contract())
    assert result.model_readiness == 'ready'
    assert result.valid_records[0].normalized_values['loc_id'] == '001'
    assert result.valid_records[0].sources['tiv_kes'][0].locator == 'sheet:csv:row:2'
    assert list(result.model_records[0]) == list(contract().fields)


def test_no_verified_schema_or_guessed_class_enum():
    result = map_to_exposure_schema([row()])
    assert not result.valid_records and not result.model_records
    assert result.model_readiness == 'unavailable'
    assert 'model_contract_unverified' in get_missing_model_inputs(result)[0].blockers
    assert result.source_analysis is None


def test_excel_alias_mapping_unit_conversion_and_confirmation(tmp_path):
    path = tmp_path / 'test_only_schedule.xlsx'
    rows = [['Narrative heading', '', '', '', '', '', '', '', ''], ['building_id', 'latitude', 'longitude', 'vulnerability_class', 'area_sqft', 'structure_value_kes', 'asset_category', 'coordinate_scope', 'occupancy'], ['001', '0.5', '34.2', 'TEST_BUILDING', '100', '0', 'buildings', 'building', 'industrial']]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    result = map_to_exposure_schema(path, contract=contract())
    assert len(result.valid_records) == 1
    record = result.valid_records[0]
    assert record.normalized_values['floor_area_m2'] == Decimal('9.29030400')
    assert record.normalized_values['tiv_kes'] == 0
    assert record.original_values['floor_area_m2'] == '100'
    assert any(d.method == 'unit_conversion' for d in record.derived_fields)


@pytest.mark.parametrize('field,value', [('lat', 'nan'), ('lat', '91'), ('lon', '-181'), ('floor_area_m2', '0'), ('tiv_kes', '-2'), ('tiv_kes', '1,23')])
def test_invalid_numerics_are_reviewable(field, value):
    result = map_to_exposure_schema([row(**{field: value})], contract=contract())
    assert not result.valid_records
    assert result.review_required_records[0].original_values[field] == value
    assert get_missing_model_inputs(result)[0].blockers


def test_missing_zero_duplicates_and_unsupported_classes():
    result = map_to_exposure_schema([row(), row(), row(loc_id='002', housing_class='UNSUPPORTED', tiv_kes='')], contract=contract())
    assert len(result.review_required_records) == 3
    blockers = [set(item.blockers) for item in get_missing_model_inputs(result)]
    assert {'duplicate_loc_id', 'duplicate_record'} <= blockers[0]
    assert 'unsupported_housing_class' in blockers[2]
    assert 'tiv_kes' in get_missing_model_inputs(result)[2].fields
    zero = map_to_exposure_schema([row(tiv_kes='0')], contract=contract())
    assert zero.valid_records[0].normalized_values['tiv_kes'] == 0


def test_site_point_not_individual_positions_and_no_value_allocation():
    source = lambda excerpt: [{'document_id': 'doc', 'locator': 'page:1', 'page_number': 1, 'excerpt': excerpt}]
    fact = lambda value, excerpt: {'value': value, 'status': 'provided', 'confidence': 1, 'sources': source(excerpt)}
    facts = InsuranceFacts.model_validate({'insured': {'coordinates': fact({'latitude': 0.5, 'longitude': 34.2}, 'Facility GPS 0.5,34.2')}, 'assets': [{'name': fact('Warehouse A', 'Warehouse A'), 'construction_type': fact('masonry', 'Warehouse A masonry'), 'floor_area': fact({'value': '120', 'unit': 'm2'}, 'Warehouse A 120 m2')}, {'name': fact('Warehouse B', 'Warehouse B')}], 'financial_exposure': {'property_sum_insured': fact({'amount': '1000000', 'currency': 'KES'}, 'Property sum KES 1 million'), 'machinery_values': fact({'amount': '200000', 'currency': 'KES'}, 'Machinery KES 200000'), 'inventory_values': fact({'amount': '0', 'currency': 'KES'}, 'Inventory KES 0'), 'contents_values': fact({'amount': '5000', 'currency': 'KES'}, 'Contents KES 5000'), 'business_interruption_limit': fact({'amount': '30000', 'currency': 'KES'}, 'Interruption KES 30000')}})
    result = map_to_exposure_schema(facts, contract=contract())
    buildings = [r for r in result.review_required_records if r.asset_category == 'buildings']
    assert len(buildings) == 2
    assert all(r.coordinate_scope == 'site' and r.normalized_values['lat'] is None and r.normalized_values['lon'] is None for r in buildings)
    assert all(r.normalized_values['tiv_kes'] is None for r in buildings)
    assert all(r.site_coordinates == {'lat': 0.5, 'lon': 34.2} for r in buildings)
    assert {r.asset_category for r in result.review_required_records} == {'buildings', 'machinery', 'inventory', 'contents', 'business_interruption'}
    assert result.source_analysis['financial_exposure']['property_sum_insured']['value']['amount'] == '1000000'
    assert result.model_records == []


def test_transparent_construction_mapping_requires_confirmation():
    model_contract = contract()
    model_contract.construction_rules = [ConstructionRule(rule_id='test-rule', construction='masonry', housing_class='TEST_BUILDING', rationale='Test-only approved construction correspondence')]
    source = row(housing_class=None, construction='masonry')
    result = map_to_exposure_schema([source], contract=model_contract)
    record = result.review_required_records[0]
    assert record.normalized_values['housing_class'] == 'TEST_BUILDING'
    assert any(d.method == 'construction_rule' and 'test-rule' in d.explanation for d in record.derived_fields)
    confirmed = confirm_exposure_mapping(result, [decision(result, record, 'housing_class', 'TEST_BUILDING', basis='construction_mapping')])
    assert confirmed.model_readiness == 'ready'
    assert confirmed.valid_records[0].original_values['housing_class'] is None
    assert confirmed.revision != result.revision
    with pytest.raises(ValueError):
        confirm_exposure_mapping(confirmed, [decision(result, record, 'housing_class', 'TEST_BUILDING')])


def test_missing_asset_values_need_explicit_assumption_and_specialized_assets_never_export():
    batch = map_to_exposure_schema([row(tiv_kes=None)], contract=contract())
    record = batch.review_required_records[0]
    with pytest.raises(ValueError, match='assumption'):
        confirm_exposure_mapping(batch, [decision(batch, record, 'tiv_kes', '50000')])
    accepted = confirm_exposure_mapping(batch, [decision(batch, record, 'tiv_kes', '50000', basis='explicit_asset_value_assumption')])
    assert accepted.valid_records[0].normalized_values['tiv_kes'] == 50000
    assert accepted.valid_records[0].original_values['tiv_kes'] is None
    inventory = map_to_exposure_schema([row(asset_category='inventory')], contract=contract())
    assert not inventory.model_records
    with pytest.raises(ValueError):
        confirm_exposure_mapping(inventory, [decision(inventory, inventory.review_required_records[0], 'asset_category', 'buildings')])


def test_extra_engine_fields_are_required_without_dropping_source_data():
    catalog = contract()
    catalog.fields['storeys'] = ModelField(kind='integer', minimum=1)
    missing = map_to_exposure_schema([row()], contract=catalog)
    assert 'storeys' in get_missing_model_inputs(missing)[0].fields
    complete = map_to_exposure_schema([row(storeys='2')], contract=catalog)
    assert complete.model_records[0]['storeys'] == '2'


@pytest.mark.asyncio
async def test_llm_description_keeps_historical_analysis_when_model_blocked():
    from tests.test_insurance_extraction import FixtureLLM, SYNTHETIC_PAGES
    result = await map_property_description('\n'.join(SYNTHETIC_PAGES), llm=FixtureLLM())
    assert result.model_readiness == 'unavailable'
    assert result.source_analysis['facts']['flood_history']
    assert not result.model_records


def test_currency_conversion_requires_explicit_approval_and_retains_original():
    from ai.exposure_schemas import FXApproval
    foreign = row()
    foreign.pop('tiv_kes')
    foreign.update(insured_value='100', currency='USD', value_scope='per_asset', coordinate_scope='building')
    unapproved = map_to_exposure_schema([foreign], contract=contract())
    assert unapproved.review_required_records[0].normalized_values['tiv_kes'] is None
    approval = FXApproval(currency='USD', kes_per_unit='130', effective_date='2026-10-08', reference='test-only rate input, not an actual market rate', approved=True, approved_by='test-underwriter', reason='Test conversion contract')
    approved = map_to_exposure_schema([foreign], contract=contract(), fx_approval=approval)
    assert approved.valid_records[0].normalized_values['tiv_kes'] == 13000
    assert approved.valid_records[0].original_values['tiv_kes'] == '100'
    assert any(d.method == 'currency_conversion' and d.approved_by for d in approved.valid_records[0].derived_fields)
    conflict = map_to_exposure_schema([row(currency='USD')], contract=contract(), fx_approval=approval)
    assert 'currency_header_conflict' in get_missing_model_inputs(conflict)[0].blockers


def test_site_proxy_is_explicit_and_never_claimed_as_building_gps():
    result = map_to_exposure_schema([row(coordinate_scope='site')], contract=contract())
    record = result.review_required_records[0]
    assert record.normalized_values['lat'] is None
    approved = confirm_exposure_mapping(result, [decision(result, record, 'coordinates', {'lat': '0.5', 'lon': '34.2'}, basis='site_coordinate_assumption')])
    assert approved.valid_records[0].coordinate_scope == 'assumed_site'
    assert approved.valid_records[0].derived_fields[-1].method == 'site_coordinate_assumption'
    assert approved.valid_records[0].original_values['coordinate_scope'] == 'site'


def test_residential_only_class_does_not_cover_industrial_buildings_or_machinery():
    catalog = contract()
    catalog.vulnerability_classes[0].occupancies = ['residential']
    result = map_to_exposure_schema([row(occupancy='industrial')], contract=catalog)
    assert 'curve_occupancy_unverified' in get_missing_model_inputs(result)[0].blockers
    machinery = map_to_exposure_schema([row(asset_category='machinery', occupancy='residential')], contract=catalog)
    assert 'specialized_model_required' in get_missing_model_inputs(machinery)[0].blockers
    assert not machinery.model_records


def test_equivalent_coordinate_formatting_is_detected():
    result = map_to_exposure_schema([row(), row(loc_id='002', lat='0.50', lon='34.20')], contract=contract())
    assert all('repeated_coordinates' in item.blockers for item in get_missing_model_inputs(result))


def test_confirmation_does_not_bypass_validation_or_unverified_contract():
    result = map_to_exposure_schema([row()])
    record = result.review_required_records[0]
    approved = confirm_exposure_mapping(result, [decision(result, record, 'housing_class', 'TEST_BUILDING')])
    assert not approved.model_records
    rejected = decision(result, record, 'housing_class', 'TEST_BUILDING')
    rejected.approved = False
    with pytest.raises(ValueError):
        confirm_exposure_mapping(result, [rejected])


def test_low_confidence_and_site_evidence_do_not_become_building_gps():
    source = lambda excerpt: [{'document_id': 'doc', 'locator': 'page:1', 'excerpt': excerpt}]
    fact = lambda value, excerpt, confidence=1: {'value': value, 'status': 'provided', 'confidence': confidence, 'sources': source(excerpt)}
    facts = InsuranceFacts.model_validate({'insured': {'coordinates': fact({'latitude': 0.5, 'longitude': 34.2}, 'Facility GPS 0.5,34.2')}, 'assets': [{'name': fact('Building A', 'Building A'), 'coordinates': fact({'latitude': 0.5, 'longitude': 34.2}, 'Facility GPS 0.5,34.2'), 'floor_area': fact({'value': '120', 'unit': 'm2'}, 'Building A 120 m2', confidence=0.4)}]})
    result = map_to_exposure_schema(facts, contract=contract())
    record = result.review_required_records[0]
    assert record.coordinate_scope == 'site'
    assert record.normalized_values['lat'] is None
    assert record.normalized_values['floor_area_m2'] is None
    assert record.original_values['_source_row']['_extracted_asset']['floor_area']['value']['value'] == '120'
    assert 'uncertain_document_field' in get_missing_model_inputs(result)[0].blockers
