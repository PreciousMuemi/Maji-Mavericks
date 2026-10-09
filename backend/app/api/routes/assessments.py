"""Assessment read models shared by the underwriting dashboard."""
from fastapi import APIRouter, Depends

from ai.logging import log_event
from ai.schemas import ResponseEnvelope
from api.ai_routes import AIServices, authorize_request, get_services
from app.schemas.dashboard import DashboardResponse
from app.services.dashboard import DashboardService

router = APIRouter(prefix='/api/assessments', tags=['Assessments'], dependencies=[Depends(authorize_request)])


@router.get('/{assessment_id}/dashboard', response_model=ResponseEnvelope[DashboardResponse])
async def dashboard(assessment_id: str, services: AIServices = Depends(get_services)):
    """Read validated facts and stored results. Does not execute a model or call an LLM."""
    result = await DashboardService(services.store, services.model_backend, services.settings).get(assessment_id)
    log_event('dashboard_retrieved', assessment_id=assessment_id)
    return ResponseEnvelope(status='success' if result.status == 'calculation_completed' else 'partial',
                            assessment_id=assessment_id, data=result,
                            warnings=[l.message for l in result.ai_insights.limitations])
