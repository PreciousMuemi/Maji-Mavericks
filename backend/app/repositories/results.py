"""Read model outputs from the configured modelling-team repository adapter."""
from ai.model_backend import UnderwritingModelBackend
from ai.underwriting_schemas import ModelOutput


async def get_model_results(assessment_id: str, *, backend: UnderwritingModelBackend) -> ModelOutput:
    return ModelOutput.model_validate(await backend.get_model_results(assessment_id))
