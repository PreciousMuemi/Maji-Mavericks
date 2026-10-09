"""Agent execution metadata and a typed adapter contract for genuine model outputs."""
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator

from .base_schema import StrictModel


class AssessmentArguments(StrictModel):
    assessment_id: str = Field(min_length=1, max_length=100)


class FloodArguments(AssessmentArguments):
    return_periods: list[StrictInt] = Field(min_length=1, max_length=10)

    @model_validator(mode='after')
    def periods_valid(self):
        if len(set(self.return_periods)) != len(self.return_periods) or any(p < 2 or p > 100_000 for p in self.return_periods):
            raise ValueError('Return periods must be unique integers between 2 and 100000')
        return self


class CompareArguments(AssessmentArguments):
    periods: list[StrictInt] = Field(min_length=2, max_length=10)

    @model_validator(mode='after')
    def periods_valid(self):
        FloodArguments(assessment_id=self.assessment_id, return_periods=self.periods)
        return self


class EvidenceArguments(AssessmentArguments):
    field: str = Field(min_length=1, max_length=200, pattern=r'^[A-Za-z0-9_.\[\]-]+$')


class AgentScenarioArguments(AssessmentArguments):
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def no_identity_override(self):
        def inspect(value):
            if isinstance(value, dict):
                if any(str(key).casefold() in {'assessment_id', 'assessment_ids', 'tenant_id', 'owner_id', 'user_id'} for key in value):
                    raise ValueError('Scenario parameters cannot override assessment identity')
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)
        inspect(self.parameters)
        return self


class BuildingLoss(StrictModel):
    loc_id: str
    loss: Decimal = Field(ge=0, allow_inf_nan=False)
    housing_class: str | None = None


class PeriodLoss(StrictModel):
    return_period: StrictInt = Field(ge=2, le=100_000)
    currency: str = Field(pattern=r'^[A-Z]{3}$')
    total_loss: Decimal = Field(ge=0, allow_inf_nan=False)
    building_losses: list[BuildingLoss] = Field(default_factory=list)


class ModelOutput(StrictModel):
    """AI adapter contract, not a claim about the absent catastrophe engine schema.

    The modelling adapter supplies actual losses, explicit units and model version.
    """
    assessment_id: str
    status: Literal['ready', 'no_results', 'missing_hazard_coverage']
    model_version: str | None = None
    result_id: str | None = None
    period_results: list[PeriodLoss] = Field(default_factory=list)
    missing_return_periods: list[StrictInt] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def results_consistent(self):
        if self.status == 'ready' and (not self.period_results or not self.model_version or self.missing_return_periods):
            raise ValueError('Ready results require genuine period outputs and a model version')
        if self.status != 'ready' and self.period_results:
            raise ValueError('Unavailable output cannot include loss estimates')
        periods = [p.return_period for p in self.period_results]
        if len(set(periods)) != len(periods):
            raise ValueError('Period results must be unique')
        return self


class DashboardInstruction(StrictModel):
    action: Literal['refresh_assessment', 'refresh_findings', 'refresh_model_results', 'show_evidence', 'show_comparison', 'show_scenario']
    assessment_id: str
    result_id: str | None = None
    field: str | None = None
    periods: list[int] = Field(default_factory=list)


class ToolExecution(StrictModel):
    name: str
    status: Literal['success', 'error']
    code: str | None = None


class Principal(StrictModel):
    subject: str = Field(min_length=1)
