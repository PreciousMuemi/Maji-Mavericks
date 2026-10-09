"""Field-specific retrieval from validated document facts, never filesystem paths."""
import asyncio
import re
from ai.insurance_extraction import _walk
from ai.store import AssessmentStore


async def get_document_evidence(assessment_id: str, field: str, *, store: AssessmentStore) -> dict:
    documents = await asyncio.to_thread(store.insurance_results, assessment_id)
    extraction = await asyncio.to_thread(store.extraction, assessment_id)
    evidence = []
    for doc in documents:
        for path, fact in _walk(doc.facts):
            normalized = re.sub(r'\[\d+\]', '', path)
            if field in {path, normalized, path.removeprefix('facts.'), normalized.removeprefix('facts.')}:
                evidence.append({'field': path, 'status': fact.status, 'value': fact.model_dump(mode='json')['value'], 'sources': [s.model_dump(mode='json') for s in fact.sources], 'alternatives': [a.model_dump(mode='json') for a in fact.alternatives]})
    for category in ('assets', 'claims'):
        for index, record in enumerate(getattr(extraction, category)):
            allowed = {'assets': {'name', 'insured_value', 'latitude', 'longitude', 'address', 'asset_type'}, 'claims': {'event_date', 'cause', 'paid_amount'}}[category]
            for name in allowed:
                path = f'{category}[{index}].{name}'
                if field in {path, f'{category}.{name}'}:
                    evidence.append({'field': path, 'value': record.model_dump(mode='json')[name], 'sources': [s.model_dump(mode='json') for s in record.sources]})
    return {'assessment_id': assessment_id, 'field': field, 'evidence': evidence}
