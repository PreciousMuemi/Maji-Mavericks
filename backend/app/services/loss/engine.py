"""Catastrophe-team entry point. Inject the actual calibrated model implementation."""
from ai.model_backend import UnderwritingModelBackend
from ai.underwriting_schemas import ModelOutput


async def run_flood_model(assessment_id: str, return_periods: list[int], *, backend: UnderwritingModelBackend) -> ModelOutput:
    return ModelOutput.model_validate(await backend.run_flood_model(assessment_id, return_periods))
