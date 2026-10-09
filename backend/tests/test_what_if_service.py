from decimal import Decimal

import pytest

from ai.risk_analyzer import IntegrationUnavailable
from ai.scenario_schemas import (
    ScenarioCapabilities, ScenarioEngineOutput, ScenarioIntent,
    ScenarioPeriodGrossLoss,
)
from ai.what_if_service import execute_scenario, interpret_scenario_request


class IntentLLM:
    def __init__(self, value):
        self.value = value

    async def structured(self, system, user, schema):
        assert 'Do not calculate losses' in system
        assert '<untrusted_request>' in user
        return schema.model_validate(self.value)


def capabilities(**updates):
    values = dict(
        supported_modifications={'exposure_value_change', 'return_period_comparison'},
        supported_return_periods={100, 500}, supported_construction_classes={'concrete_rcc'},
        currency='KES', limitations=['Industrial contents vulnerability is not configured'],
    )
    values.update(updates)
    return ScenarioCapabilities(**values)


@pytest.mark.asyncio
async def test_interpret_request_uses_strict_schema_and_capabilities():
    llm = IntentLLM({
        'scenario_name': 'Increase building values',
        'modifications': [{'type': 'exposure_value_change', 'percentage': '20', 'asset_ids': []}],
        'affected_return_periods': [100, 500],
    })
    result = await interpret_scenario_request('What if values rise by 20%?', llm=llm, capabilities=capabilities())
    assert result.modifications[0].percentage == Decimal('20')


@pytest.mark.asyncio
async def test_interpret_rejects_engine_unsupported_mitigation():
    llm = IntentLLM({
        'scenario_name': 'Elevate warehouses',
        'modifications': [{'type': 'elevation_change', 'elevation_delta_m': '1', 'asset_ids': ['W3']}],
        'affected_return_periods': [100],
    })
    with pytest.raises(IntegrationUnavailable, match='elevation_change'):
        await interpret_scenario_request('Elevate W3', llm=llm, capabilities=capabilities())


def output(assessment, result_id, losses, *, version='flood-2'):
    return ScenarioEngineOutput(
        assessment_id=assessment, model_version=version, result_id=result_id,
        currency='KES', gross_losses=[
            ScenarioPeriodGrossLoss(return_period=period, gross_loss=loss)
            for period, loss in losses.items()
        ], limitations=[],
    )


class Engine:
    async def get_scenario_capabilities(self, assessment_id):
        return capabilities()

    async def get_scenario_baseline(self, assessment_id, return_periods):
        return output(assessment_id, 'base-1', {100: Decimal('100'), 500: Decimal('0')})

    async def run_what_if_scenario(self, assessment_id, intent):
        return output(assessment_id, 'scenario-1', {100: Decimal('125'), 500: Decimal('10')})


@pytest.mark.asyncio
async def test_execute_computes_all_financial_differences_in_python():
    intent = ScenarioIntent(
        scenario_name='Approved value change',
        modifications=[{'type': 'exposure_value_change', 'percentage': '20'}],
        affected_return_periods=[100, 500],
    )
    result = await execute_scenario('A-1', intent, backend=Engine())
    assert result.baseline_gross_loss == Decimal('100')
    assert result.scenario_gross_loss == Decimal('135')
    assert result.absolute_difference == Decimal('35')
    assert result.percentage_difference == Decimal('35')
    assert result.chart_points[0].percentage_difference == Decimal('25')
    assert result.chart_points[1].percentage_difference is None
    assert 'not a short-term weather forecast' in result.ai_explanation


@pytest.mark.asyncio
async def test_execute_rejects_non_comparable_model_versions():
    class ChangedEngine(Engine):
        async def run_what_if_scenario(self, assessment_id, intent):
            return output(assessment_id, 'scenario-1', {100: Decimal('125'), 500: Decimal('10')}, version='other')

    intent = ScenarioIntent(
        scenario_name='Comparison',
        modifications=[{'type': 'return_period_comparison', 'return_periods': [100, 500]}],
        affected_return_periods=[100, 500],
    )
    with pytest.raises(IntegrationUnavailable, match='different model versions'):
        await execute_scenario('A-1', intent, backend=ChangedEngine())


@pytest.mark.asyncio
async def test_execute_rejects_missing_period_and_identity_mismatch():
    intent = ScenarioIntent(
        scenario_name='Comparison',
        modifications=[{'type': 'return_period_comparison', 'return_periods': [100, 500]}],
        affected_return_periods=[100, 500],
    )

    class Missing(Engine):
        async def run_what_if_scenario(self, assessment_id, intent):
            return output(assessment_id, 'scenario-1', {100: Decimal('125')})

    with pytest.raises(IntegrationUnavailable, match='every requested'):
        await execute_scenario('A-1', intent, backend=Missing())

    class Foreign(Engine):
        async def run_what_if_scenario(self, assessment_id, intent):
            return output('OTHER', 'scenario-1', {100: Decimal('125'), 500: Decimal('10')})

    with pytest.raises(IntegrationUnavailable, match='different assessment'):
        await execute_scenario('A-1', intent, backend=Foreign())
