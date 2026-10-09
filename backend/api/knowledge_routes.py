from fastapi import APIRouter, Depends, Query

from ai.knowledge_base import KnowledgeChatRequest, KnowledgeChatResponse, KnowledgeHit, KnowledgeSummary, SiteHazardProfile, get_knowledge_base
from ai.schemas import ResponseEnvelope
from ai.logging import log_event
from api.ai_routes import AIServices, get_services

router = APIRouter(prefix='/api/knowledge', tags=['Knowledge base'])


@router.get('/summary', response_model=ResponseEnvelope[KnowledgeSummary])
async def knowledge_summary(services: AIServices = Depends(get_services)):
    data = get_knowledge_base(services.settings.dataset_directory).summary()
    return ResponseEnvelope(status='partial' if data.blockers else 'success', data=data, warnings=data.warnings)


@router.get('/search', response_model=ResponseEnvelope[list[KnowledgeHit]])
async def knowledge_search(q: str = Query(min_length=2, max_length=200), limit: int = Query(8, ge=1, le=20),
                           services: AIServices = Depends(get_services)):
    return ResponseEnvelope(status='success', data=get_knowledge_base(services.settings.dataset_directory).search(q, limit))


@router.get('/site-hazard', response_model=ResponseEnvelope[SiteHazardProfile])
async def site_hazard(latitude: float = Query(ge=-90, le=90), longitude: float = Query(ge=-180, le=180),
                      services: AIServices = Depends(get_services)):
    data = get_knowledge_base(services.settings.dataset_directory).sample_site(latitude, longitude)
    return ResponseEnvelope(status='success' if data.coverage_status == 'covered' else 'partial', data=data,
                            warnings=[] if data.coverage_status == 'covered' else ['Location is outside the supplied Nzoia raster extent'])


@router.post('/chat', response_model=ResponseEnvelope[KnowledgeChatResponse])
async def knowledge_chat(body: KnowledgeChatRequest, services: AIServices = Depends(get_services)):
    knowledge = get_knowledge_base(services.settings.dataset_directory)
    warnings = []
    try:
        data = await knowledge.answer_with_llm(body.message, services.extraction.llm)
    except Exception:
        log_event('knowledge_chat_llm_fallback', code='grounding_or_provider_failure')
        data = knowledge.answer(body.message)
        warnings.append('Gemini was unavailable or returned unsupported content; deterministic evidence response used.')
    return ResponseEnvelope(status='partial' if data.status == 'requires_model_input' else 'success', data=data, warnings=warnings)
