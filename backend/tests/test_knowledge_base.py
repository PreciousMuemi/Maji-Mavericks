from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai.config import Settings
from ai.knowledge_base import KnowledgeBase
from app.main import create_app


DATASETS = Path(__file__).resolve().parents[2] / 'datasets'


def test_real_dataset_summary_is_source_derived():
    result = KnowledgeBase(DATASETS).summary()
    assert result.portfolio.record_count == 500
    assert result.portfolio.synthetic is True
    assert result.portfolio.missing_cells == 0
    assert result.portfolio.total_tiv_kes == 22_737_765_000
    assert result.supported_return_periods == [10, 20, 50, 100, 200, 500]
    assert result.supported_housing_classes == [
        'concrete_rcc', 'informal_iron_sheet', 'permanent_masonry', 'semi_permanent'
    ]
    assert all(r.crs == 'EPSG:4326' and r.depth_unit == 'metres' for r in result.rasters)
    assert [p.flooded_locations for p in result.hazard_exposure] == sorted(
        p.flooded_locations for p in result.hazard_exposure
    )
    assert all(p.classification == 'hazard intersection, not modelled loss' for p in result.hazard_exposure)
    assert result.blockers and 'depth-damage' in result.blockers[0]


def test_knowledge_search_returns_source_attribution():
    hits = KnowledgeBase(DATASETS).search('flood depth vulnerability', limit=3)
    assert hits and len(hits) <= 3
    assert all(hit.source_id and hit.title and hit.section > 0 for hit in hits)


def test_site_hazard_sampling_distinguishes_coverage_from_loss():
    knowledge = KnowledgeBase(DATASETS)
    covered = knowledge.sample_site(0.756568, 33.971985)
    assert covered.coverage_status == 'covered'
    assert [item.return_period for item in covered.depths] == [10, 20, 50, 100, 200, 500]
    assert 'not damage or insured loss' in covered.interpretation
    outside = knowledge.sample_site(-1.2921, 36.8219)
    assert outside.coverage_status == 'outside_raster_extent'
    assert all(item.depth_m is None for item in outside.depths)


def test_knowledge_chat_uses_calculated_facts_and_refuses_loss():
    knowledge = KnowledgeBase(DATASETS)
    exposure = knowledge.answer('What is the total portfolio exposure?')
    assert 'KES 22,737,765,000' in exposure.answer
    assert 'not a modelled loss' in exposure.answer
    assert exposure.citations
    loss = knowledge.answer('What is the RP100 modelled loss?')
    assert loss.status == 'requires_model_input'
    assert 'unavailable' in loss.answer
    assert 'will not treat exposure as damage or loss' in loss.answer


@pytest.mark.asyncio
async def test_gemini_narrative_is_grounded_to_known_numbers_and_sources():
    knowledge = KnowledgeBase(DATASETS)
    source_id = next(s.source_id for s in knowledge.sources() if s.kind == 'exposure')

    class GroundedLLM:
        async def structured(self, system, user, schema):
            assert 'Use only facts and numbers in the evidence JSON' in system
            return schema(answer='The synthetic portfolio contains 500 buildings.',
                key_findings=['Total structure TIV is KES 22,737,765,000.'],
                underwriting_implications=['The value is exposure, not modelled loss.'],
                recommended_actions=['Configure approved vulnerability curves.'],
                limitations=['Loss is unavailable.'], cited_source_ids=[source_id])

    answer = await knowledge.answer_with_llm('Summarize the portfolio', GroundedLLM())
    assert answer.llm_used is True
    assert answer.status == 'answered_with_llm'
    assert answer.citations[0].source_id == source_id

    class InventingLLM(GroundedLLM):
        async def structured(self, system, user, schema):
            return schema(answer='The loss is KES 999999999999.', cited_source_ids=[source_id])

    with pytest.raises(ValueError, match='introduced a number'):
        await knowledge.answer_with_llm('What is the loss?', InventingLLM())


def test_knowledge_api_and_underwriter_app(tmp_path):
    settings = Settings(_env_file=None, storage_directory=str(tmp_path), dataset_directory=str(DATASETS),
                        api_token='secret', llm_provider='openrouter', openrouter_api_key=None)
    with TestClient(create_app(settings=settings)) as client:
        assert client.get('/api/knowledge/summary').status_code == 401
        headers = {'Authorization': 'Bearer secret'}
        response = client.get('/api/knowledge/summary', headers=headers)
        assert response.status_code == 200
        assert response.json()['status'] == 'partial'
        assert response.json()['data']['portfolio']['record_count'] == 500
        search = client.get('/api/knowledge/search', params={'q': 'return period'}, headers=headers)
        assert search.status_code == 200 and search.json()['data']
        site = client.get('/api/knowledge/site-hazard', params={'latitude': 0.756568, 'longitude': 33.971985}, headers=headers)
        assert site.status_code == 200
        assert site.json()['data']['coverage_status'] == 'covered'
        chat = client.post('/api/knowledge/chat', json={'message': 'Which return periods are available?'}, headers=headers)
        assert chat.status_code == 200
        assert 'RP500' in chat.json()['data']['answer']
        assert chat.json()['data']['citations']
        app = client.get('/underwriter')
        assert app.status_code == 200
        assert 'Upload a report' in app.text
        assert 'Processing starts automatically' in app.text
        assert 'Assessment copilot' in app.text
        assert 'Process report' not in app.text
