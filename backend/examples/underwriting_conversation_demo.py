"""End-to-end demo with a real persisted assessment and explicit synthetic inputs.

Uses a scripted tool-selecting LLM, real API/repository/validation/evidence functions,
and the unavailable-model adapter. No flood results or live LLM responses are invented.
Run from backend: python -m examples.underwriting_conversation_demo
"""
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from ai.config import Settings
from ai.schemas import LLMResponse, ToolCall
from api.ai_routes import build_services
from app.main import create_app
from tests.test_insurance_extraction import FixtureLLM, SYNTHETIC_PAGES, pdf_bytes


class DemoLLM(FixtureLLM):
    async def chat(self, messages, tools):
        if messages[-1]['role'] == 'tool':
            result = json.loads(messages[-1]['content'])
            if result['status'] == 'error':
                return LLMResponse(text=result['message'])
            if messages[-1]['name'] == 'validate_portfolio':
                fields = sorted({item['field_path'] for item in result['data']['review_items']})
                return LLMResponse(text='Review is required for: ' + ', '.join(fields))
            return LLMResponse(text='The requested source evidence is available in the structured response.')
        context = next(m['content'] for m in reversed(messages) if m['role'] == 'user' and m['content'].startswith('Current authorized assessment'))
        assessment_id = json.loads(context.split(': ', 1)[1])['data']['assessment_id']
        question = messages[-1]['content'].casefold()
        periods = [int(p) for p in re.findall(r'(\d+)[- ]year', question)]
        if 'missing' in question:
            name, extra = 'validate_portfolio', {}
        elif 'compare' in question:
            name, extra = 'compare_return_periods', {'periods': periods}
        elif 'run' in question and periods:
            name, extra = 'run_flood_model', {'return_periods': periods}
        elif 'evidence' in question:
            name, extra = 'get_document_evidence', {'field': 'facts.insured.coordinates'}
        else:
            name, extra = 'get_model_results', {}
        return LLMResponse(tool_calls=[ToolCall(id='demo-call', name=name, arguments={'assessment_id': assessment_id, **extra})])


def main():
    root = Path(__file__).parents[1]
    settings = Settings(_env_file=None, storage_directory=str(root / '.nzoia-data' / 'underwriting-demo'))
    services = build_services(settings, llm=DemoLLM())
    transcript = {'demonstration': 'Persisted assessment with synthetic offer and scripted LLM; actual model unavailable', 'turns': []}
    with TestClient(create_app(settings, services)) as client:
        assessment_id = client.post('/api/ai/assessments').json()['assessment_id']
        transcript['assessment_id'] = assessment_id
        base = f'/api/ai/assessments/{assessment_id}'
        upload = client.post(base + '/documents', files={'file': ('synthetic_demo_offer.pdf', pdf_bytes(SYNTHETIC_PAGES), 'application/pdf')})
        document_id = upload.json()['data']['document_id']
        extraction = client.post(base + f'/documents/{document_id}/insurance-extract')
        assert extraction.status_code == 200
        for question in [
            'What information is missing from this insurance offer?',
            'Show the evidence for the facility coordinates.',
            'Analyse this portfolio for flood risk.',
            'Run the 100-year flood scenario.',
            'Which buildings contribute the largest losses?',
            'Compare the 100-year and 500-year flood scenarios.',
        ]:
            response = client.post(base + '/chat', json={'message': question, 'conversation_id': 'demonstration'})
            assert response.status_code == 200
            transcript['turns'].append({'user': question, 'response': response.json()})
        assert services.store.owner(assessment_id) == 'local-development'
        assert len(services.store.conversation(assessment_id, 'demonstration')) == 12
    services.store.close()
    target = root / 'examples' / 'underwriting_conversation_demo.json'
    target.write_text(json.dumps(transcript, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'assessment_id': assessment_id, 'transcript': str(target), 'turns': len(transcript['turns']), 'actual_model_results': False}))


if __name__ == '__main__':
    main()
