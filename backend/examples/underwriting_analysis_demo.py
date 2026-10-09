"""Synthetic evidence demonstration, not analysis of the missing grain offer.

Run from backend: python -m examples.underwriting_analysis_demo --events 2
Uses actual analysis/repository/calculation functions and a fixture narrative provider.
"""
import argparse
import json
from pathlib import Path

from ai.config import Settings
from ai.underwriting_analysis import generate_underwriting_analysis
from api.ai_routes import build_services
from tests.test_underwriting_analysis import ExplainingLLM, build_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.events <= 50:
        parser.error('Use a test event count between 1 and 50')
    root = Path(__file__).parents[1]
    settings = Settings(_env_file=None, storage_directory=str(root / '.nzoia-data' / 'intelligence-demo'))
    services = build_services(settings, llm=ExplainingLLM())
    try:
        assessment = build_document(services.store, count=args.events)
        result = generate_underwriting_analysis(assessment, store=services.store, llm=services.extraction.llm, settings=settings)
    finally:
        services.store.close()
    target = root / 'examples' / 'synthetic_underwriting_analysis.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'assessment_id': assessment, 'output': str(target), 'synthetic_events': args.events, 'model_results_available': False, 'narrative_provider': 'scripted test fixture'}))


if __name__ == '__main__':
    main()
