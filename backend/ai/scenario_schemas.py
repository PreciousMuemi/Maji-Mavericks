"""Strict contracts for capability-driven what-if analysis."""
from decimal import Decimal
from typing import Annotated, Literal, Union

from pydantic import Field, StrictInt, model_validator

from .base_schema import StrictModel


class ExposureValueChange(StrictModel):
    type: Literal['exposure_value_change']
    percentage: Decimal = Field(gt=-100, le=1000, allow_inf_nan=False)
    asset_ids: list[str] = Field(default_factory=list)


class ReturnPeriodComparison(StrictModel):
    type: Literal['return_period_comparison']
    return_periods: list[StrictInt] = Field(min_length=2, max_length=10)

    @model_validator(mode='after')
    def valid_periods(self):
        if len(set(self.return_periods)) != len(self.return_periods) or any(p < 2 or p > 100_000 for p in self.return_periods):
            raise ValueError('Return periods must be unique integers between 2 and 100000')
        return self


class ConstructionClassChange(StrictModel):
    type: Literal['construction_class_change']
    target_class: str = Field(min_length=1, max_length=100)
    asset_ids: list[str] = Field(default_factory=list)


class DeductibleChange(StrictModel):
    type: Literal['deductible_change']
    amount: Decimal = Field(ge=0, allow_inf_nan=False)
    currency: str = Field(pattern=r'^[A-Z]{3}$')


class ElevationChange(StrictModel):
    type: Literal['elevation_change']
    elevation_delta_m: Decimal = Field(gt=0, le=20, allow_inf_nan=False)
    asset_ids: list[str] = Field(min_length=1)


ScenarioModification = Annotated[Union[
    ExposureValueChange, ReturnPeriodComparison, ConstructionClassChange,
    DeductibleChange, ElevationChange,
], Field(discriminator='type')]


class ScenarioIntent(StrictModel):
    scenario_name: str = Field(min_length=1, max_length=160)
    modifications: list[ScenarioModification] = Field(min_length=1, max_length=10)
    affected_return_periods: list[StrictInt] = Field(min_length=1, max_length=10)

    @model_validator(mode='after')
    def periods_valid(self):
        if len(set(self.affected_return_periods)) != len(self.affected_return_periods) or any(p < 2 or p > 100_000 for p in self.affected_return_periods):
            raise ValueError('Affected return periods must be unique integers between 2 and 100000')
        return self


class ScenarioCapabilities(StrictModel):
    supported_modifications: set[Literal[
        'exposure_value_change', 'return_period_comparison',
        'construction_class_change', 'deductible_change', 'elevation_change'
    ]] = Field(default_factory=set)
    supported_return_periods: set[StrictInt] = Field(default_factory=set)
    supported_construction_classes: set[str] = Field(default_factory=set)
    currency: str = Field(pattern=r'^[A-Z]{3}$')
    limitations: list[str] = Field(default_factory=list)


class ScenarioPeriodGrossLoss(StrictModel):
    return_period: StrictInt = Field(ge=2, le=100_000)
    gross_loss: Decimal = Field(ge=0, allow_inf_nan=False)


class ScenarioEngineOutput(StrictModel):
    assessment_id: str
    model_version: str = Field(min_length=1)
    result_id: str = Field(min_length=1)
    currency: str = Field(pattern=r'^[A-Z]{3}$')
    gross_losses: list[ScenarioPeriodGrossLoss] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique_periods(self):
        periods = [item.return_period for item in self.gross_losses]
        if len(set(periods)) != len(periods):
            raise ValueError('Gross loss return periods must be unique')
        return self


class ScenarioComparisonPoint(StrictModel):
    return_period: StrictInt
    baseline_gross_loss: Decimal
    scenario_gross_loss: Decimal
    absolute_difference: Decimal
    percentage_difference: Decimal | None
    currency: str


class WhatIfScenarioResult(StrictModel):
    scenario_name: str
    baseline_model_version: str
    modified_assumptions: list[ScenarioModification]
    baseline_gross_loss: Decimal
    scenario_gross_loss: Decimal
    absolute_difference: Decimal
    percentage_difference: Decimal | None
    affected_return_periods: list[StrictInt]
    currency: str
    chart_points: list[ScenarioComparisonPoint]
    limitations: list[str] = Field(default_factory=list)
    ai_explanation: str
    baseline_result_id: str
    scenario_result_id: str
