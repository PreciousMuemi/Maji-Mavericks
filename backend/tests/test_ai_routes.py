import json
from fastapi.testclient import TestClient
from ai.config import Settings
from ai.schemas import Asset, SourceReference, LLMResponse
from api.ai_routes import build_services
from app.main import create_app


class TestLLM:
    __test__ = False
    async def structured(self, system, user, schema):
        document = json.loads(user)
        return schema(assets=[Asset(asset_id="001", name="Warehouse", sources=[SourceReference(document_id=document["document_id"], locator=document["segments"][0]["locator"], excerpt="Warehouse")])])
    async def chat(self, messages, tools):
        return LLMResponse(text="The portfolio needs coordinates and insured values.")


def test_upload_extract_chat_findings_and_restart(tmp_path):
    settings = Settings(_env_file=None, storage_directory=str(tmp_path), api_token="test-secret")
    services = build_services(settings, llm=TestLLM())
    app = create_app(settings, services)
    headers = {"Authorization": "Bearer test-secret"}
    with TestClient(app) as client:
        assert client.post("/api/ai/assessments").status_code == 401
        created = client.post("/api/ai/assessments", headers=headers)
        assert created.status_code == 201
        assessment = created.json()["assessment_id"]
        path = f"/api/ai/assessments/{assessment}"
        response = client.post(path + "/documents", headers=headers, files={"file": ("data.csv", b"name\nWarehouse\n", "text/csv")})
        assert response.status_code == 201
        assert client.post(path + "/extract", headers=headers).json()["data"]["assets"][0]["asset_id"] == "001"
        # Re-extraction replaces per-document output instead of duplicating assets.
        assert len(client.post(path + "/extract", headers=headers).json()["data"]["assets"]) == 1
        findings = client.get(path + "/findings", headers=headers).json()
        assert findings["status"] == "partial"
        assert findings["data"]["model_results"] is None
        assert len(findings["data"]["findings"]) == 2
        assert client.post(path + "/chat", headers=headers, json={"message": "What is missing?"}).status_code == 200
        assert client.post(path + "/analyze", headers=headers).status_code == 503
        assert client.post(path + "/scenarios", headers=headers, json={"name": "baseline"}).status_code == 503
        invalid = client.post(path + "/chat", headers=headers, json={"message": "", "secret": "confidential"})
        assert invalid.status_code == 422
        assert "confidential" not in invalid.text
        assert client.get("/api/ai/assessments/missing/findings", headers=headers).status_code == 404
    services.store.close()
    restarted = build_services(settings, llm=TestLLM())
    assert len(restarted.store.extraction(assessment).assets) == 1
    restarted.store.close()


def test_missing_credentials_and_invalid_documents(tmp_path):
    settings = Settings(_env_file=None, storage_directory=str(tmp_path), openai_api_key=None, gemini_api_key=None)
    with TestClient(create_app(settings)) as client:
        assessment = client.post("/api/ai/assessments").json()["assessment_id"]
        path = f"/api/ai/assessments/{assessment}"
        assert client.post(path + "/extract").status_code == 422
        assert client.post(path + "/documents", files={"file": ("bad.pdf", b"invalid")}).status_code == 422
        client.post(path + "/documents", files={"file": ("data.csv", b"name\nWarehouse")})
        assert client.post(path + "/extract").status_code == 503
        assert client.post(path + "/chat", json={"message": "hello"}).status_code == 503


def test_provider_errors_do_not_leak(tmp_path, caplog):
    class BrokenLLM(TestLLM):
        async def structured(self, *args):
            raise RuntimeError("private customer and API key")
    settings = Settings(_env_file=None, storage_directory=str(tmp_path))
    services = build_services(settings, llm=BrokenLLM())
    with TestClient(create_app(settings, services), raise_server_exceptions=False) as client:
        assessment = client.post("/api/ai/assessments").json()["assessment_id"]
        path = f"/api/ai/assessments/{assessment}"
        client.post(path + "/documents", files={"file": ("data.csv", b"name\nWarehouse")})
        response = client.post(path + "/extract")
        assert response.status_code == 502
        assert "private customer" not in response.text
        assert "private customer" not in caplog.text
    services.store.close()
