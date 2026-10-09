"""ASGI entry point. Dependencies are replaceable for model integration and tests."""
from contextlib import asynccontextmanager
from hmac import compare_digest
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from pydantic import ValidationError
from ai.config import Settings
from ai.document_parser import DocumentError
from ai.extraction_service import ExtractionError
from ai.llm_client import LLMUnavailable
from ai.logging import log_event
from ai.risk_analyzer import IntegrationUnavailable
from ai.schemas import ResponseEnvelope, ErrorDetail
from ai.store import AssessmentNotFound
from ai.authorization import AssessmentAccessDenied
from api.ai_routes import AIServices, build_services, router
from api.knowledge_routes import router as knowledge_router
from app.api.routes.assessments import router as assessments_router


def error_response(request: Request, status: int, code: str, message: str):
    # Neither validation inputs nor provider exception strings are returned/logged.
    assessment_id = request.path_params.get("assessment_id")
    log_event("request_failed", assessment_id=assessment_id, code=code)
    envelope = ResponseEnvelope(status="error", assessment_id=assessment_id, errors=[ErrorDetail(code=code, message=message)])
    return JSONResponse(status_code=status, content=envelope.model_dump(mode="json"))


def create_app(settings: Settings | None = None, services: AIServices | None = None) -> FastAPI:
    settings = settings if settings is not None else (services.settings if services is not None else Settings())

    @asynccontextmanager
    async def lifespan(app):
        owns_services = services is None
        app.state.ai_services = services if services is not None else build_services(settings)
        try:
            yield
        finally:
            if owns_services:
                app.state.ai_services.store.close()
                close = getattr(app.state.ai_services.extraction.llm, "aclose", None)
                if close is not None:
                    await close()

    application = FastAPI(title="Nzoia Risk Intelligence AI", lifespan=lifespan)

    @application.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path.startswith(("/api/ai", "/api/assessments", "/api/knowledge")) and settings.api_token:
            provided = request.headers.get("authorization", "")
            expected = "Bearer " + settings.api_token.get_secret_value()
            if not compare_digest(provided.encode(), expected.encode()):
                return error_response(request, 401, "unauthorized", "A valid API token is required")
        try:
            return await call_next(request)
        except Exception:
            # Handle here so ASGI servers do not log confidential provider tracebacks.
            return error_response(request, 502, "service_failure", "Service dependency failed; no result was accepted")

    @application.exception_handler(Exception)
    async def unexpected(request, exc):
        return error_response(request, 502, "service_failure", "Service dependency failed; no result was accepted")

    @application.exception_handler(RequestValidationError)
    async def request_invalid(request, exc):
        return error_response(request, 422, "invalid_request", "Request does not match the API schema")

    @application.exception_handler(ValidationError)
    async def output_invalid(request, exc):
        return error_response(request, 502, "invalid_llm_output", "Service output did not match the required schema")

    @application.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error_response(request, exc.status_code, "http_error", "Request could not be completed")

    for exception, status, code, message in [
        (AssessmentNotFound, 404, "assessment_not_found", "Assessment does not exist"),
        (AssessmentAccessDenied, 404, "assessment_not_found", "Assessment does not exist"),
        (DocumentError, 422, "invalid_document", "Document is unsupported, unreadable, or exceeds configured limits"),
        (ExtractionError, 422, "extraction_failed", "Extraction requires readable documents and matching source provenance"),
        (LLMUnavailable, 503, "llm_unavailable", "LLM is not configured or did not return a usable response"),
        (IntegrationUnavailable, 503, "integration_unavailable", "Model integration is unavailable or required exposure inputs are incomplete"),
    ]:
        def handler_factory(status, code, message):
            async def handler(request, exc):
                return error_response(request, status, code, message)
            return handler
        application.add_exception_handler(exception, handler_factory(status, code, message))

    application.include_router(router)
    application.include_router(assessments_router)
    application.include_router(knowledge_router)

    @application.get('/underwriter', include_in_schema=False)
    async def underwriter_app():
        return FileResponse(Path(__file__).resolve().parents[2] / 'frontend' / 'public' / 'underwriter.html')
    return application


app = create_app()
