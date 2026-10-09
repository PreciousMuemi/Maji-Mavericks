"""Frontend contract: exact decimals, explicit availability and auditable evidence."""
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from ai.base_schema import StrictModel
from ai.intelligence_schemas import AnalysisLimitation, Citation, RecommendedAction, RiskDriver

AssessmentStatus = Literal['document_extracted', 'validation_required', 'model_ready',
                           'calculation_completed', 'partial_assessment',
                           'hazard_coverage_unavailable', 'insufficient_data']
Basis = Literal['document_reported', 'historical', 'modelled', 'validated_exposure', 'workflow']


class KPI(StrictModel):
    id: str
    label: str
    value: Decimal | str | None
    display_value: str
    availability: Literal['available', 'unavailable', 'review_required']
    basis: Basis
    currency: Literal['KES'] | None = None
    citation_ids: list[str] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode='after')
    def honest_availability(self):
        if (self.availability == 'available') != (self.value is not None):
            raise ValueError('Unavailable and review-required values must be null')
        if self.availability == 'available' and self.basis != 'workflow' and not self.citation_ids:
            raise ValueError('Available numerical cards require evidence')
        return self


class ChartPoint(StrictModel):
    label: str
    value: Decimal = Field(ge=0, allow_inf_nan=False)
    display_value: str
    citation_ids: list[str] = Field(min_length=1)


class Chart(StrictModel):
    id: str
    type: Literal['bar', 'line']
    title: str
    x_label: str
    y_label: str
    basis: Basis
    unit: Literal['KES', 'assets']
    points: list[ChartPoint] = Field(min_length=1)
    note: str
    return_period: int | None = None


class MapLocation(StrictModel):
    id: str
    label: str
    lat: float = Field(ge=-90, le=90, allow_inf_nan=False)
    lon: float = Field(ge=-180, le=180, allow_inf_nan=False)
    coordinate_scope: Literal['site', 'building', 'assumed_site']
    citation_ids: list[str] = Field(min_length=1)


class DashboardMap(StrictModel):
    locations: list[MapLocation] = Field(default_factory=list)
    coverage_status: Literal['unverified', 'unavailable', 'model_results_available']
    note: str


class AIInsights(StrictModel):
    summary: str
    origin: Literal['stored_ai_analysis', 'deterministic_evidence']
    risk_drivers: list[RiskDriver] = Field(max_length=3)
    recommendations: list[RecommendedAction]
    limitations: list[AnalysisLimitation]
    missing_data: list[AnalysisLimitation]
    assumptions: list[str]


class DashboardResponse(StrictModel):
    assessment_id: str
    status: AssessmentStatus
    statuses: list[AssessmentStatus] = Field(min_length=1)
    kpis: list[KPI]
    charts: list[Chart]
    map: DashboardMap
    ai_insights: AIInsights
    source_references: list[Citation]

    @model_validator(mode='after')
    def coherent_references(self):
        if self.status not in self.statuses:
            raise ValueError('Primary status must be included in statuses')
        if len({k.id for k in self.kpis}) != len(self.kpis) or len({c.id for c in self.charts}) != len(self.charts):
            raise ValueError('Dashboard IDs must be unique')
        citations = {c.id for c in self.source_references}
        if len(citations) != len(self.source_references):
            raise ValueError('Citation IDs must be unique')
        if any(c.assessment_id != self.assessment_id for c in self.source_references):
            raise ValueError('Evidence belongs to another assessment')
        items = [*self.kpis, *self.map.locations, *self.ai_insights.risk_drivers,
                 *self.ai_insights.recommendations, *self.ai_insights.limitations, *self.ai_insights.missing_data,
                 *[p for chart in self.charts for p in chart.points]]
        if any(not set(item.citation_ids).issubset(citations) for item in items):
            raise ValueError('Dashboard refers to unknown evidence')
        return self
