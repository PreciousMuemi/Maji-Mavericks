"""Scenario/comparison integration. The AI performs no flood-loss calculations."""
from ai.model_backend import UnderwritingModelBackend
from ai.underwriting_schemas import ModelOutput


async def compare_return_periods(assessment_id: str, periods: list[int], *, backend: UnderwritingModelBackend) -> ModelOutput:
    return ModelOutput.model_validate(await backend.compare_return_periods(assessment_id, periods))


async def run_scenario(assessment_id: str, parameters: dict, *, backend: UnderwritingModelBackend) -> ModelOutput:
    return ModelOutput.model_validate(await backend.run_scenario(assessment_id, parameters))
