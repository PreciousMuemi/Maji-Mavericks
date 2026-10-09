import json

import pytest
from fastapi.testclient import TestClient

from ai.config import Settings
from ai.schemas import Asset, DocumentExtraction, LLMResponse, SourceReference
from api.ai_routes import build_services
from app.main import create_app


class DocumentLLM:
    async def structured(self, system, user, schema):
        document = json.loads(user)
        segment = document['segments'][0]
        return DocumentExtraction(assets=[Asset(asset_id='001', name='Warehouse', sources=[SourceReference(document_id=document['document_id'], locator=segment['locator'], excerpt=segment['text'])])])

    async def chat(self, messages, tools):
        return LLMResponse(text='Coordinates and insured value are missing.')


@pytest.fixture
def client(tmp_path):
    settings = Settings(_env_file=None, storage_directory=str(tmp_path), api_token='test-token')
    services = build_services(settings, llm=DocumentLLM())
    with TestClient(create_app(settings, services), headers={'Authorization': 'Bearer test-token'}) as value:
        yield value, services
    services.store.close()


def test_document_extraction_chat_and_findings(client):
    api, services = client
    response = api.post('/api/ai/assessments')
    assert response.status_code == 201
    assessment_id = response.json()['assessment_id']
    base = f'/api/ai/assessments/{assessment_id}'
    assert api.post(base + '/documents', files={'file': ('assets.csv', b'id,name\n001,Warehouse\n', 'text/csv')}).status_code == 201
    result = api.post(base + '/extract').json()
    assert result['data']['assets'][0]['asset_id'] == '001'
    assert result['data']['assets'][0]['sources'][0]['document_id']
    findings = api.get(base + '/findings').json()
    assert findings['status'] == 'partial'
    assert findings['data']['model_results'] is None
    assert {f['code'] for f in findings['data']['findings']} == {'missing_coordinates', 'missing_insured_value'}
    response = api.post(base + '/chat', json={'message': 'What is missing?'})
    assert response.status_code == 200
    assert len(response.json()['data']['findings']) == 2
    assert api.post(base + '/analyze').status_code == 503
    assert api.post(base + '/scenarios', json={'name': 'test'}).status_code == 503


def test_errors_use_envelopes_and_do_not_echo_inputs(client, caplog):
    api, _ = client
    secret = 'CONFIDENTIAL-CUSTOMER-NAME'
    assert api.post('/api/ai/assessments', headers={'Authorization': 'Bearer wrong'}).status_code == 401
    response = api.get('/api/ai/assessments/missing/findings')
    assert response.status_code == 404
    assert response.json()['errors'][0]['code'] == 'assessment_not_found'
    assessment_id = api.post('/api/ai/assessments').json()['assessment_id']
    base = f'/api/ai/assessments/{assessment_id}'
    caplog.set_level('INFO', logger='nzoia.ai')
    response = api.post(base + '/chat', json={'message': {'secret': secret}})
    assert response.status_code == 422
    assert secret not in response.text
    response = api.post(base + '/documents', files={'file': ('private.pdf', secret.encode(), 'application/pdf')})
    assert response.status_code == 422
    assert secret not in response.text
    assert secret not in caplog.text
    assert api.post(base + '/extract').status_code == 422


def test_missing_credentials_returns_503(tmp_path, monkeypatch):
    monkeypatch.delenv('NZOIA_OPENAI_API_KEY', raising=False)
    settings = Settings(_env_file=None, storage_directory=str(tmp_path), openai_api_key=None)
    with TestClient(create_app(settings)) as api:
        assessment_id = api.post('/api/ai/assessments').json()['assessment_id']
        result = api.post(f'/api/ai/assessments/{assessment_id}/chat', json={'message': 'hello'})
        assert result.status_code == 503
        assert result.json()['errors'][0]['code'] == 'llm_unavailable'


def test_provider_failure_is_sanitized(client):
    api, services = client
    async def fail(*args):
        raise RuntimeError('PRIVATE PROVIDER DETAIL')
    services.agent.llm.chat = fail
    assessment_id = api.post('/api/ai/assessments').json()['assessment_id']
    api.raise_server_exceptions = False
    response = api.post(f'/api/ai/assessments/{assessment_id}/chat', json={'message': 'hello'})
    assert response.status_code == 502
    assert 'PRIVATE PROVIDER DETAIL' not in response.text
