from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai.config import Settings
from ai.exposure_mapping import map_to_exposure_schema
from ai.nzoia_model import (FinancialTerms, NzoiaCatModelBackend,
                            apply_financial_terms, damage_ratio,
                            nzoia_exposure_contract)
from ai.store import AssessmentStore
from app.main import create_app


def test_jrc_curve_interpolation_and_financial_order():
    assert damage_ratio(0) == 0
    assert damage_ratio(.5) == Decimal('.22')
    assert damage_ratio(.75) == Decimal('.3')
    result = apply_financial_terms([Decimal('100'), Decimal('300')], FinancialTerms(
        per_risk_deductible_kes=Decimal('50'), per_risk_limit_kes=Decimal('200'),
        quota_share=Decimal('.25'), cat_xol_attachment_kes=Decimal('100'),
        cat_xol_limit_kes=Decimal('100')))
    assert result.ground_up == 400
    assert result.gross == 250
    assert result.quota_share_ceded == Decimal('62.5')
    assert result.cat_xol_ceded == Decimal('87.5')
    assert result.net == 100


@pytest.mark.asyncio
async def test_supplied_nzoia_portfolio_runs_real_rasters(tmp_path):
    root = Path(__file__).parents[2]
    settings = Settings(_env_file=None, storage_directory=str(tmp_path),
                        dataset_directory=str(root / 'datasets'))
    store = AssessmentStore(str(tmp_path))
    assessment = store.create('test')
    mapping = map_to_exposure_schema(
        root / 'datasets/data/team_b_nzoia/exposure_nzoia_synthetic.csv',
        contract=nzoia_exposure_contract(), settings=settings)
    assert mapping.model_readiness == 'ready'
    snapshot = store.dashboard_snapshot(assessment)
    store.save_exposure_mapping(assessment, mapping, source_hash=snapshot['source_hash'])
    model = NzoiaCatModelBackend(store, settings)

    output = await model.run_flood_model(assessment, [100, 500])

    assert output.status == 'ready'
    assert [p.return_period for p in output.period_results] == [100, 500]
    assert all(len(p.building_losses) == 500 for p in output.period_results)
    assert output.period_results[1].total_loss >= output.period_results[0].total_loss
    assert output.period_results[0].currency == 'KES'
    assert output.model_version
    store.close()


def test_upload_pipeline_maps_samples_and_returns_rankable_results(tmp_path):
    root = Path(__file__).parents[2]
    settings = Settings(_env_file=None, storage_directory=str(tmp_path),
                        dataset_directory=str(root / 'datasets'))
    source = root / 'datasets/data/team_b_nzoia/exposure_nzoia_synthetic.csv'
    with TestClient(create_app(settings)) as client:
        assessment = client.post('/api/ai/assessments').json()['data']['assessment_id']
        with source.open('rb') as stream:
            uploaded = client.post(f'/api/ai/assessments/{assessment}/documents',
                                   files={'file': ('exposure.csv', stream, 'text/csv')})
        assert uploaded.status_code == 201

        prepared = client.post(f'/api/ai/assessments/{assessment}/prepare-model')

    assert prepared.status_code == 200
    payload = prepared.json()['data']
    assert payload['model_readiness'] == 'ready'
    assert payload['valid_records'] == 500
    assert payload['review_required_records'] == 0
    assert payload['missing_model_inputs'] == []
    assert len(payload['model_output']['period_results']) == 6
