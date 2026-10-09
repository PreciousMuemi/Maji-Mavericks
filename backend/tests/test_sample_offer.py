"""Real supplied-offer integration tests. Never replace the missing file with a fixture."""
import os
from pathlib import Path

import pytest

from ai.config import Settings
from ai.document_parser import DocumentParser
from ai.insurance_extraction import extract_insurance_document
from ai.insurance_schemas import InsuranceDocumentResult


def sample_path():
    configured = os.getenv('NZOIA_SAMPLE_OFFER_PATH')
    roots = [Path(configured)] if configured else [
        Path(__file__).parent / 'fixtures' / 'OFFER_NZOIA_GRAIN_PROCESSING.pdf',
        Path(__file__).parents[2] / 'OFFER_NZOIA_GRAIN_PROCESSING.pdf',
        Path(__file__).parents[1] / 'OFFER_NZOIA_GRAIN_PROCESSING.pdf',
    ]
    path = next((path for path in roots if path.is_file()), None)
    if path is None:
        pytest.skip('OFFER_NZOIA_GRAIN_PROCESSING.pdf is not available; set NZOIA_SAMPLE_OFFER_PATH')
    return path


def test_sample_pdf_text_and_page_evidence():
    document = DocumentParser(Settings(_env_file=None)).parse_file(sample_path(), 'sample-offer')
    assert document.segments
    assert all(segment.page_number is not None for segment in document.segments)
    assert any(segment.kind == 'text' for segment in document.segments)


def test_sample_offer_live_extraction():
    path = sample_path()
    if os.getenv('NZOIA_RUN_LIVE_EXTRACTION') != '1':
        pytest.skip('Set NZOIA_RUN_LIVE_EXTRACTION=1 to run paid provider integration testing')
    settings = Settings()
    key = settings.openai_api_key if settings.llm_provider == 'openai' else settings.gemini_api_key
    if not key:
        pytest.fail('Live extraction requested but selected provider credentials are absent')
    result = InsuranceDocumentResult.model_validate(extract_insurance_document(path, settings=settings))
    assert result.facts.insured.client_name.status != 'not_provided'
    assert result.facts.assets
    from ai.insurance_extraction import _walk
    for _, fact in _walk(result.facts):
        for source in [*fact.sources, *[s for a in fact.alternatives for s in a.sources]]:
            assert source.page_number is not None
    # No sample amounts, client names, building counts or totals are hardcoded.
