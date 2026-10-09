"""Portfolio validation can run independently of catastrophe-model availability."""
import asyncio
from ai.exposure_mapping import get_missing_model_inputs, map_to_exposure_schema
from ai.store import AssessmentStore
from ai.validator import validate_extraction


async def validate_portfolio(assessment_id: str, *, store: AssessmentStore) -> dict:
    extraction = await asyncio.to_thread(store.extraction, assessment_id)
    documents = await asyncio.to_thread(store.insurance_results, assessment_id)
    findings = [f.model_dump(mode='json') for f in validate_extraction(extraction)]
    reviews = [item.model_dump(mode='json') for doc in documents for item in doc.review_items]
    missing = []
    for doc in documents:
        batch = map_to_exposure_schema(doc)
        missing.extend(item.model_dump(mode='json') for item in get_missing_model_inputs(batch))
    if not extraction.assets and not documents:
        findings.append({'code': 'no_extracted_assets', 'severity': 'warning', 'message': 'No extracted assets are available', 'asset_id': None})
    return {'assessment_id': assessment_id, 'findings': findings, 'review_items': reviews, 'missing_model_inputs': missing}
