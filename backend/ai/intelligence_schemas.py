"""Strict frontend schema and an LLM narrative contract without numerical authority."""
import re
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, model_validator

from .base_schema import StrictModel
from .schemas import SourceReference


class Citation(StrictModel):
    id: str
    kind: Literal['document', 'model', 'calculation', 'validation']
    assessment_id: str
    source: SourceReference | None = None
    model_result_id: str | None = None
    model_version: str | None = None
    formula: str | None = None
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    function: str | None = None


class NumericFinding(StrictModel):
    metric: str
    value: Decimal
    unit: str
    citation_id: str


class IntelligenceFinding(StrictModel):
    id: str
    title: str
    statement: str
    evidence_type: Literal['document_reported', 'historical', 'modelled', 'data_quality']
    citation_ids: list[str] = Field(min_length=1)
    numerical_findings: list[NumericFinding] = Field(default_factory=list)


class RiskDriver(IntelligenceFinding):
    why_it_matters: str


class RecommendedAction(StrictModel):
    action: str
    rationale: str
    origin: Literal['independent_analysis', 'broker_proposal', 'insurer_condition', 'insurer_proposal', 'surveyor_proposal', 'insured_proposal', 'unknown_document_author']
    citation_ids: list[str] = Field(min_length=1)


class AnalysisLimitation(StrictModel):
    code: str
    message: str
    citation_ids: list[str] = Field(default_factory=list)


class RiskLevel(StrictModel):
    value: str | None = None
    source: str


class UnderwritingAnalysis(StrictModel):
    risk_summary: str = Field(min_length=1)
    risk_level: RiskLevel
    top_risk_drivers: list[RiskDriver] = Field(max_length=3)
    historical_claims_findings: list[IntelligenceFinding]
    model_findings: list[IntelligenceFinding]
    recommended_actions: list[RecommendedAction]
    limitations: list[AnalysisLimitation]
    citations: list[Citation]

    @model_validator(mode='after')
    def references_exist(self):
        if len(re.findall(r'[.!?](?:\s|$)', self.risk_summary)) > 3:
            raise ValueError('Executive risk summary must have at most three sentences')
        known = {citation.id for citation in self.citations}
        for item in [*self.top_risk_drivers, *self.historical_claims_findings, *self.model_findings, *self.recommended_actions, *self.limitations]:
            if not set(item.citation_ids).issubset(known):
                raise ValueError('Unknown citation reference')
        for item in [*self.top_risk_drivers, *self.historical_claims_findings, *self.model_findings]:
            if any(n.citation_id not in known for n in item.numerical_findings):
                raise ValueError('Unknown numerical calculation citation')
        return self


class NarrativeSelection(StrictModel):
    finding_id: str
    explanation: str = Field(min_length=1, max_length=600)


class NarrativeAction(StrictModel):
    finding_id: str
    action: str = Field(min_length=1, max_length=400)
    rationale: str = Field(min_length=1, max_length=600)


class AnalysisNarrative(StrictModel):
    summary: list[NarrativeSelection] = Field(min_length=1, max_length=3)
    drivers: list[NarrativeSelection] = Field(max_length=3)
    actions: list[NarrativeAction] = Field(max_length=12)
