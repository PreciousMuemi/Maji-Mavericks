import asyncio
from uuid import uuid4
from dataclasses import dataclass
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from ai.agent import Agent
from ai.config import Settings
from ai.document_parser import DocumentParser, DocumentError
from ai.extraction_service import ExtractionService, ExtractionError
from ai.llm_client import LLMClient, LLMUnavailable, create_llm_client
from ai.logging import log_event
from ai.exposure_mapper import CanonicalExposureMapper, ExposureMapper
from ai.risk_analyzer import FloodModel, RiskAnalyzer, UnavailableFloodModel, IntegrationUnavailable
from ai.schemas import ResponseEnvelope, ChatRequest, ScenarioRequest, DocumentExtraction, AIAnalysis, ValidationFinding
from ai.scenario_service import ScenarioService
from ai.store import AssessmentStore, AssessmentNotFound
from ai.tools import ToolRegistry
from ai.validator import validate_extraction
from ai.insurance_extraction import InsuranceDocumentExtractor
from ai.insurance_schemas import InsuranceDocumentResult
from ai.authorization import AssessmentAuthorizer, OwnershipAuthorizer
from ai.model_backend import UnderwritingModelBackend, UnavailableUnderwritingBackend
from ai.underwriting_schemas import Principal
from ai.underwriting_analysis import UnderwritingAnalysisService
from ai.intelligence_schemas import UnderwritingAnalysis


@dataclass
class AIServices:
    settings: Settings
    store: AssessmentStore
    parser: DocumentParser
    extraction: ExtractionService
    agent: Agent
    analyzer: RiskAnalyzer
    model_backend: UnderwritingModelBackend
    authorizer: AssessmentAuthorizer


def build_services(settings: Settings, llm: LLMClient | None = None, flood_model: FloodModel | None = None, mapper: ExposureMapper | None = None, store: AssessmentStore | None = None, model_backend: UnderwritingModelBackend | None = None, authorizer: AssessmentAuthorizer | None = None) -> AIServices:
    client = llm if llm is not None else create_llm_client(settings)
    analyzer = RiskAnalyzer(flood_model if flood_model is not None else UnavailableFloodModel(), mapper if mapper is not None else CanonicalExposureMapper(), timeout_seconds=settings.agent_tool_timeout_seconds)
    repository = store if store is not None else AssessmentStore(settings.storage_directory)
    agent = Agent(client, settings.max_tool_rounds, max_tool_calls=settings.max_agent_tool_calls, llm_timeout_seconds=settings.llm_timeout_seconds, chat_timeout_seconds=settings.agent_chat_timeout_seconds)
    return AIServices(settings, repository, DocumentParser(settings), ExtractionService(client), agent, analyzer, model_backend if model_backend is not None else UnavailableUnderwritingBackend(), authorizer if authorizer is not None else OwnershipAuthorizer(repository))


def get_services(request: Request) -> AIServices:
    return request.app.state.ai_services


def get_principal(services: AIServices = Depends(get_services)) -> Principal:
    # The app middleware authenticates the configured shared credential. Multi-user
    # hosts replace this dependency with their verified identity provider.
    return Principal(subject='service-account' if services.settings.api_token else 'local-development')


async def authorize_request(request: Request, principal: Principal = Depends(get_principal), services: AIServices = Depends(get_services)):
    assessment_id = request.path_params.get('assessment_id')
    if assessment_id is not None:
        await services.authorizer.require_access(principal, assessment_id)


router = APIRouter(prefix="/api/ai", tags=["AI"], dependencies=[Depends(authorize_request)])


@router.post('/assessments/{assessment_id}/underwriting-analysis', response_model=ResponseEnvelope[UnderwritingAnalysis])
async def underwriting_analysis(assessment_id: str, services: AIServices = Depends(get_services), principal: Principal = Depends(get_principal)):
    snapshot = await asyncio.to_thread(services.store.dashboard_snapshot, assessment_id)
    intelligence = UnderwritingAnalysisService(services.store, services.extraction.llm, model_backend=services.model_backend, settings=services.settings)
    result = await intelligence.generate(assessment_id, principal=principal, authorizer=services.authorizer)
    saved = await asyncio.to_thread(services.store.save_intelligence, assessment_id, result, fingerprint=snapshot['fingerprint'])
    log_event('underwriting_analysis_completed', assessment_id=assessment_id)
    return ResponseEnvelope(status='partial' if result.limitations or not saved else 'success', assessment_id=assessment_id, data=result,
                            warnings=[] if saved else ['Inputs changed during analysis; refresh before displaying this briefing'])


@router.post('/assessments/{assessment_id}/documents/{document_id}/insurance-extract', response_model=ResponseEnvelope[InsuranceDocumentResult])
async def extract_insurance(assessment_id: str, document_id: str, services: AIServices = Depends(get_services)):
    documents = await asyncio.to_thread(services.store.documents, assessment_id)
    if document_id not in documents:
        raise HTTPException(status_code=404)
    try:
        result = await InsuranceDocumentExtractor(services.extraction.llm, services.settings).extract_document(documents[document_id])
    except ExtractionError as exc:
        message = str(exc).casefold()
        code = ('source_mismatch' if 'evidence' in message or 'source' in message
                else 'numeric_mismatch' if 'numeric' in message or 'measurement' in message or 'currency' in message
                else 'extraction_validation')
        log_event('insurance_extraction_rejected', assessment_id=assessment_id, code=code)
        raise
    await asyncio.to_thread(services.store.save_insurance_extraction, assessment_id, result)
    log_event('insurance_extraction_completed', assessment_id=assessment_id)
    return ResponseEnvelope(status='partial' if result.review_status == 'requires_review' or result.review_items or result.contradictions or result.warnings else 'success', assessment_id=assessment_id, data=result, warnings=result.warnings)


@router.get('/assessments/{assessment_id}/documents/{document_id}/insurance-extraction', response_model=ResponseEnvelope[InsuranceDocumentResult])
async def get_insurance_extraction(assessment_id: str, document_id: str, services: AIServices = Depends(get_services)):
    result = await asyncio.to_thread(services.store.insurance_extraction, assessment_id, document_id)
    if result is None:
        raise HTTPException(status_code=404)
    return ResponseEnvelope(status='partial' if result.review_status == 'requires_review' or result.review_items or result.contradictions or result.warnings else 'success', assessment_id=assessment_id, data=result, warnings=result.warnings)


@router.post("/assessments", response_model=ResponseEnvelope[dict[str, str]], status_code=201)
async def create_assessment(services: AIServices = Depends(get_services), principal: Principal = Depends(get_principal)):
    assessment_id = await asyncio.to_thread(services.store.create, principal.subject)
    log_event("assessment_created", assessment_id=assessment_id)
    return ResponseEnvelope(status="success", assessment_id=assessment_id, data={"assessment_id": assessment_id})


@router.post("/assessments/{assessment_id}/documents", response_model=ResponseEnvelope[dict[str, object]], status_code=201)
async def upload_document(assessment_id: str, file: UploadFile = File(...), services: AIServices = Depends(get_services)):
    try:
        await asyncio.to_thread(services.store.documents, assessment_id)
        content = await file.read(services.settings.max_upload_bytes + 1)
        document_id = str(uuid4())
        document = await asyncio.to_thread(services.parser.parse, content, file.filename or "", document_id)
        await asyncio.to_thread(services.store.add_document, assessment_id, document)
    finally:
        await file.close()
    log_event("document_parsed", assessment_id=assessment_id)
    return ResponseEnvelope(status="partial" if document.warnings else "success", assessment_id=assessment_id, data={"document_id": document_id, "segments": len(document.segments)}, warnings=document.warnings)


@router.post("/assessments/{assessment_id}/extract", response_model=ResponseEnvelope[DocumentExtraction])
async def extract(assessment_id: str, services: AIServices = Depends(get_services)):
    documents = await asyncio.to_thread(services.store.documents, assessment_id)
    if not documents:
        raise ExtractionError("Upload at least one document before extraction")
    values = {}
    for document_id, document in documents.items():
        values[document_id] = await services.extraction.extract(document)
    await asyncio.to_thread(services.store.save_extractions, assessment_id, values)
    result = await asyncio.to_thread(services.store.extraction, assessment_id)
    log_event("extraction_completed", assessment_id=assessment_id)
    warnings = list(dict.fromkeys([*result.warnings, *[f.message for f in validate_extraction(result)]]))
    if not result.assets and not result.claims:
        warnings.append('No source-backed assets or claims were extracted; assessment information remains incomplete')
    return ResponseEnvelope(status="partial" if warnings else "success", assessment_id=assessment_id, data=result, warnings=warnings)


@router.post("/assessments/{assessment_id}/chat", response_model=ResponseEnvelope[AIAnalysis])
async def chat(assessment_id: str, body: ChatRequest, services: AIServices = Depends(get_services), principal: Principal = Depends(get_principal)):
    await asyncio.to_thread(services.store.documents, assessment_id)
    conversation_id = body.conversation_id or 'default'
    history = await asyncio.to_thread(services.store.conversation, assessment_id, conversation_id)
    registry = ToolRegistry(assessment_id, services.store, services.analyzer, backend=services.model_backend, principal=principal, authorizer=services.authorizer, timeout_seconds=services.settings.agent_tool_timeout_seconds)
    result, warnings = await services.agent.chat(body.message, registry, history=history)
    await asyncio.to_thread(services.store.save_chat_turn, assessment_id, conversation_id, body.message, result.summary)
    log_event("chat_completed", assessment_id=assessment_id)
    return ResponseEnvelope(status="partial" if warnings else "success", assessment_id=assessment_id, data=result, warnings=warnings)


@router.get("/assessments/{assessment_id}/findings", response_model=ResponseEnvelope[AIAnalysis])
async def findings(assessment_id: str, services: AIServices = Depends(get_services)):
    extraction = await asyncio.to_thread(services.store.extraction, assessment_id)
    analysis = await asyncio.to_thread(services.store.analysis, assessment_id)
    if analysis is None:
        analysis = AIAnalysis(summary="Data validation only; flood model analysis has not been run.", findings=validate_extraction(extraction),
            sources=[source for record in [*extraction.assets, *extraction.claims] for source in record.sources])
    rich = await asyncio.to_thread(services.store.insurance_results, assessment_id)
    for document in rich:
        for item in document.review_items:
            analysis.findings.append(ValidationFinding(code=item.code, severity='warning', message=item.message, field_path=item.field_path))
            analysis.sources.extend(item.sources)
        for item in document.contradictions:
            analysis.findings.append(ValidationFinding(code='document_contradiction', severity='warning', message=item.message, field_path=item.field_path))
            analysis.sources.extend(item.sources)
    warnings = list(extraction.warnings)
    if not extraction.assets:
        warnings.append("No extracted assets are available")
    if analysis.model_results is None:
        warnings.append("No flood model results are available")
    return ResponseEnvelope(status="partial" if warnings or analysis.findings else "success", assessment_id=assessment_id, data=analysis, warnings=warnings)


@router.post("/assessments/{assessment_id}/analyze", response_model=ResponseEnvelope[AIAnalysis])
async def analyze(assessment_id: str, services: AIServices = Depends(get_services)):
    snapshot = await asyncio.to_thread(services.store.dashboard_snapshot, assessment_id)
    result = await services.analyzer.analyze(assessment_id, snapshot['extraction'])
    saved = await asyncio.to_thread(services.store.save_analysis, assessment_id, result, fingerprint=snapshot['fingerprint'])
    if not saved:
        raise IntegrationUnavailable('Inputs changed during calculation; rerun against current records')
    return ResponseEnvelope(status="success", assessment_id=assessment_id, data=result)


@router.post("/assessments/{assessment_id}/scenarios", response_model=ResponseEnvelope[dict[str, object]])
async def scenario(assessment_id: str, body: ScenarioRequest, services: AIServices = Depends(get_services)):
    extraction = await asyncio.to_thread(services.store.extraction, assessment_id)
    result = await ScenarioService(services.analyzer).run(assessment_id, extraction, body)
    return ResponseEnvelope(status="success", assessment_id=assessment_id, data=result)
