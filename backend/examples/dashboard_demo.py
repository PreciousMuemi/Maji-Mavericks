"""Generate dashboard API contract examples from clearly synthetic test fixtures.

Run: python -m examples.dashboard_demo (requires the project's test dependencies).
Never supplies fixtures as an application model fallback.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from ai.config import Settings
from ai.schemas import ResponseEnvelope
from ai.underwriting_schemas import Principal
from api.ai_routes import build_services, get_principal
from app.main import create_app
from app.schemas.dashboard import DashboardResponse
from tests.dashboard_fixtures import assessment_fixture, backend_fixture
from tests.test_underwriting_analysis import ExplainingLLM


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main():
    root = Path(__file__).resolve().parents[1]
    docs = root / 'docs'
    docs.mkdir(exist_ok=True)
    envelope_schema = ResponseEnvelope[DashboardResponse].model_json_schema(mode='serialization')
    envelope_schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
    write_json(docs / 'dashboard.schema.json', envelope_schema)
    payload_schema = DashboardResponse.model_json_schema(mode='serialization')
    payload_schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
    write_json(docs / 'dashboard-data.schema.json', payload_schema)
    report = []
    for complete in (True, False):
        with TemporaryDirectory(prefix='nzoia-dashboard-fixture-') as directory:
            services = build_services(Settings(_env_file=None, storage_directory=directory), llm=ExplainingLLM())
            try:
                assessment, _ = assessment_fixture(services.store, complete=complete)
                if complete:
                    services.model_backend = backend_fixture(assessment)
                app = create_app(services=services)
                app.dependency_overrides[get_principal] = lambda: Principal(subject='alice')
                with TestClient(app) as client:
                    if complete:
                        generated = client.post(f'/api/ai/assessments/{assessment}/underwriting-analysis')
                        generated.raise_for_status()
                    response = client.get(f'/api/assessments/{assessment}/dashboard')
                    response.raise_for_status()
                    result = response.json()
                ResponseEnvelope[DashboardResponse].model_validate(result)
                name = 'dashboard-success.synthetic.json' if complete else 'dashboard-partial.synthetic.json'
                write_json(root / 'examples' / name, result)
                report.append({'example': name, 'synthetic_test_data': True, 'status': result['data']['status'], 'charts': len(result['data']['charts']), 'model_source': 'static TEST_ONLY adapter fixture' if complete else 'unavailable actual integration', 'llm_source': 'scripted test fixture' if complete else 'no LLM calls'})
            finally:
                services.store.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
