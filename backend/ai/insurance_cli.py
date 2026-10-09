"""Run real extraction with the configured provider; no fallback fabricated output."""
import argparse
import json
import os
from pathlib import Path

from pydantic import ValidationError

from .document_parser import DocumentError
from .extraction_service import ExtractionError
from .insurance_extraction import extract_insurance_document
from .llm_client import LLMUnavailable


def main():
    parser = argparse.ArgumentParser(description='Extract source-backed insurance facts as validated JSON')
    parser.add_argument('file_path', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = extract_insurance_document(args.file_path)
    except (DocumentError, ExtractionError, LLMUnavailable, ValidationError):
        print(json.dumps({'status': 'error', 'code': 'extraction_unavailable', 'message': 'Check the document, credentials, schema and source evidence.'}))
        return 2
    except Exception:
        print(json.dumps({'status': 'error', 'code': 'dependency_failure', 'message': 'Extraction dependency failed; no facts accepted.'}))
        return 2
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('w', encoding='utf-8', opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
            stream.write(text + '\n')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
