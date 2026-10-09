"""Offline contract demonstration using a synthetic PDF and injected fixture responses.

Run from backend: python -m examples.generate_synthetic_demo
This never reads, impersonates, or extracts the missing user-provided sample offer.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ai.config import Settings
from ai.insurance_extraction import extract_insurance_document
from tests.test_insurance_extraction import FixtureLLM, SYNTHETIC_PAGES, pdf_bytes


def main():
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'synthetic_offer.pdf'
        path.write_bytes(pdf_bytes(SYNTHETIC_PAGES))
        result = extract_insurance_document(path, llm=FixtureLLM(), settings=Settings(_env_file=None))
    output = Path(__file__).parent
    (output / 'synthetic_insurance_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Synthetic extraction demonstration', '', 'This result uses a generated two-page test PDF and an injected fixture LLM. It is not extraction of OFFER_NZOIA_GRAIN_PROCESSING.pdf and does not demonstrate live provider accuracy.', '', 'Review items:', '']
    lines += [f"- `{item['field_path']}`: {item['message']}" for item in result['review_items']]
    (output / 'synthetic_insurance_review.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'demo_json': str(output / 'synthetic_insurance_result.json'), 'review_items': len(result['review_items']), 'contradictions': len(result['contradictions'])}))


if __name__ == '__main__':
    main()
