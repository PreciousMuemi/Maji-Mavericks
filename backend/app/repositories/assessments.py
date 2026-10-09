"""Real assessment context from the injected durable assessment repository."""
import asyncio
from ai.store import AssessmentStore


async def get_assessment(assessment_id: str, *, store: AssessmentStore) -> dict:
    extraction = await asyncio.to_thread(store.extraction, assessment_id)
    insurance = await asyncio.to_thread(store.insurance_results, assessment_id)
    return {'assessment_id': assessment_id, 'extraction': extraction.model_dump(mode='json'), 'insurance_documents': [r.model_dump(mode='json') for r in insurance], 'historical_claims_are_predictions': False}
