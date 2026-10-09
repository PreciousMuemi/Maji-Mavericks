from datetime import date
from decimal import Decimal
from typing import Any, Generic, Literal, TypeVar
from pydantic import BaseModel, Field, model_validator
from .underwriting_schemas import AgentScenarioArguments, DashboardInstruction, ToolExecution
from .base_schema import StrictModel


class SourceReference(StrictModel):
    document_id: str
    locator: str
    excerpt: str = Field(max_length=500)
    page_number: int | None = Field(default=None, ge=1)


class FinancialValue(StrictModel):
    amount: Decimal = Field(ge=0, allow_inf_nan=False)
    currency: str = Field(pattern=r"^[A-Z]{3}$")


class Asset(StrictModel):
    asset_id: str
    name: str
    asset_type: str | None = None
    address: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    longitude: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)
    insured_value: FinancialValue | None = None
    sources: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def paired_coordinates(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        return self


class Claim(StrictModel):
    claim_id: str
    asset_id: str | None = None
    event_date: date | None = None
    cause: str | None = None
    paid_amount: FinancialValue | None = None
    sources: list[SourceReference] = Field(default_factory=list)


class DocumentExtraction(StrictModel):
    assets: list[Asset] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentSegment(StrictModel):
    locator: str
    text: str
    page_number: int | None = Field(default=None, ge=1)
    kind: Literal['text', 'table'] = 'text'


class ParsedDocument(StrictModel):
    document_id: str
    segments: list[DocumentSegment]
    warnings: list[str] = Field(default_factory=list)


class ValidationFinding(StrictModel):
    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    asset_id: str | None = None
    field_path: str | None = None


class AIAnalysis(StrictModel):
    summary: str
    findings: list[ValidationFinding] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    model_results: dict[str, Any] | None = None
    dashboard_updates: list[DashboardInstruction] = Field(default_factory=list)
    tool_executions: list[ToolExecution] = Field(default_factory=list)


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, max_length=100)


class ScenarioRequest(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def preserve_assessment_identity(self):
        AgentScenarioArguments(assessment_id='request-validation', parameters=self.parameters)
        return self


class ToolCall(StrictModel):
    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(StrictModel):
    provider_metadata: dict[str, Any] = Field(default_factory=dict, exclude=True)
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ErrorDetail(StrictModel):
    code: str
    message: str


T = TypeVar("T")


class ResponseEnvelope(BaseModel, Generic[T]):
    status: Literal["success", "partial", "error"]
    assessment_id: str | None = None
    data: T | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorDetail] = Field(default_factory=list)
