"""Real-input integration hooks; missing samples are never replaced by test fixtures."""
import os
from pathlib import Path

import pytest

from ai.exposure_mapping import inspect_exposure_csv, map_to_exposure_schema
from ai.exposure_schemas import ExposureContract
from ai.insurance_extraction import extract_insurance_document
from ai.insurance_schemas import InsuranceDocumentResult
from tests.test_sample_offer import sample_path


def actual_csv():
    configured = os.getenv('NZOIA_EXPOSURE_CSV_PATH')
    root = Path(__file__).parents[2]
    paths = [Path(configured)] if configured else [*list((root / 'data' / 'exposure').glob('*.csv')), *list(root.glob('exposure_nzoia_synthetic*.csv'))]
    if len(paths) != 1 or not paths[0].is_file():
        pytest.skip('Actual Nzoia exposure CSV is absent or ambiguous; set NZOIA_EXPOSURE_CSV_PATH')
    return paths[0]


def test_actual_nzoia_csv_schema_and_observed_classes():
    observation = inspect_exposure_csv(actual_csv())
    assert {'loc_id', 'lat', 'lon', 'housing_class', 'floor_area_m2', 'tiv_kes'} <= set(observation.columns)
    assert observation.row_count > 0
    assert observation.observed_housing_classes
    result = map_to_exposure_schema(actual_csv())
    assert len(result.review_required_records) == observation.row_count
    assert not result.model_records  # Observed CSV classes do not establish a catalog.


def test_actual_nzoia_csv_verified_model_contract():
    path = actual_csv()
    catalog_path = os.getenv('NZOIA_EXPOSURE_CONTRACT_PATH')
    if not catalog_path:
        pytest.skip('A verified modelling-team contract is required; set NZOIA_EXPOSURE_CONTRACT_PATH')
    contract = ExposureContract.model_validate_json(Path(catalog_path).read_text())
    assert contract.verified
    observed = inspect_exposure_csv(path)
    assert set(observed.columns) >= {name for name, field in contract.fields.items() if field.required}
    assert set(observed.observed_housing_classes) <= {item.name for item in contract.vulnerability_classes}
    result = map_to_exposure_schema(path, contract=contract)
    assert len(result.valid_records) + len(result.review_required_records) == observed.row_count


def test_grain_offer_exposure_mapping_preserves_original_analysis():
    path = sample_path()
    if os.getenv('NZOIA_RUN_LIVE_EXTRACTION') != '1':
        pytest.skip('Set NZOIA_RUN_LIVE_EXTRACTION=1 and configure credentials to run the real PDF extraction')
    extraction = InsuranceDocumentResult.model_validate(extract_insurance_document(path))
    result = map_to_exposure_schema(extraction)
    assert result.source_analysis == extraction.model_dump(mode='json')
    assert not result.model_records
    assert len([r for r in result.review_required_records if r.asset_category == 'buildings']) == len(extraction.facts.assets)
