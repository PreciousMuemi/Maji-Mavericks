from fastapi.testclient import TestClient

from ai.config import Settings
from api.ai_routes import build_services
from app.main import create_app
from tests.test_insurance_extraction import FixtureLLM, SYNTHETIC_PAGES, pdf_bytes


def test_rich_extraction_route_persistence_and_assessment_scope(tmp_path):
    settings = Settings(_env_file=None, storage_directory=str(tmp_path), api_token='integration-token')
    services = build_services(settings, llm=FixtureLLM())
    with TestClient(create_app(settings, services), headers={'Authorization': 'Bearer integration-token'}) as client:
        assessment_id = client.post('/api/ai/assessments').json()['assessment_id']
        base = f'/api/ai/assessments/{assessment_id}'
        uploaded = client.post(base + '/documents', files={'file': ('synthetic.pdf', pdf_bytes(SYNTHETIC_PAGES), 'application/pdf')})
        assert uploaded.status_code == 201
        document_id = uploaded.json()['data']['document_id']
        path = base + f'/documents/{document_id}'
        assert client.get(path + '/insurance-extraction').status_code == 404
        response = client.post(path + '/insurance-extract')
        assert response.status_code == 200
        result = response.json()
        assert result['status'] == 'partial'
        assert result['data']['facts']['financial_exposure']['inventory_values']['value']['amount'] == '0'
        assert result['data']['document_id'] == document_id
        assert client.get(path + '/insurance-extraction').json()['data'] == result['data']
        other_id = client.post('/api/ai/assessments').json()['assessment_id']
        other = f'/api/ai/assessments/{other_id}/documents/{document_id}'
        assert client.post(other + '/insurance-extract').status_code == 404
        assert client.get(other + '/insurance-extraction').status_code == 404
    services.store.close()
    restarted = build_services(settings, llm=FixtureLLM())
    try:
        assert restarted.store.insurance_extraction(assessment_id, document_id).facts.insured.client_name.value == 'Synthetic Test Mill'
    finally:
        restarted.store.close()
