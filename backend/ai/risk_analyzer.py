import json
import asyncio
from typing import Any, Protocol
from .exposure_mapper import ExposureMapper
from .schemas import AIAnalysis, DocumentExtraction
from .validator import validate_extraction
from .underwriting_schemas import ModelOutput
from pydantic import ValidationError


class IntegrationUnavailable(RuntimeError):
    pass


def validate_model_output(result: dict[str, Any], assessment_id: str | None = None) -> dict[str, Any]:
    """Reject non-object and non-finite results before persistence or responses."""
    if not isinstance(result, dict):
        raise IntegrationUnavailable("Model integration returned an invalid output contract")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise IntegrationUnavailable("Model integration returned non-JSON output") from exc
    try:
        output = ModelOutput.model_validate(result)
    except ValidationError as exc:
        raise IntegrationUnavailable('Model integration did not return the required loss-result contract') from exc
    if output.status != 'ready' or assessment_id is not None and output.assessment_id != assessment_id:
        raise IntegrationUnavailable('No completed model results belong to this assessment')
    return output.model_dump(mode='json')


class FloodModel(Protocol):
    """Adapter must return actual model outputs with units, provenance, and version.

    Implementers also own calibration, input preconditions, and model execution.
    AI code never computes hazard or catastrophe loss estimates.
    """
    async def analyze(self, assessment_id: str, exposures: list[dict[str, Any]]) -> dict[str, Any]: ...
    async def scenario(self, assessment_id: str, exposures: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]: ...


class UnavailableFloodModel:
    async def analyze(self, assessment_id, exposures):
        raise IntegrationUnavailable("Flood model integration is not configured")

    async def scenario(self, assessment_id, exposures, parameters):
        raise IntegrationUnavailable("Scenario model integration is not configured")


class RiskAnalyzer:
    def __init__(self, model: FloodModel, mapper: ExposureMapper, *, timeout_seconds: float = 30):
        self.model, self.mapper, self.timeout_seconds = model, mapper, timeout_seconds

    async def call_model(self, name: str, *args):
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await getattr(self.model, name)(*args)
        except TimeoutError as exc:
            raise IntegrationUnavailable('Model request timed out; execution status may be unknown and no result was accepted') from exc

    async def analyze(self, assessment_id: str, extraction: DocumentExtraction) -> AIAnalysis:
        findings = validate_extraction(extraction)
        if not extraction.assets or any(f.code in {"missing_coordinates", "missing_insured_value", "duplicate_asset_id", "mixed_currencies"} for f in findings):
            raise IntegrationUnavailable("Model analysis requires valid assets, coordinates, insured values, and a consistent currency")
        results = validate_model_output(await self.call_model('analyze', assessment_id, self.mapper.map_assets(extraction.assets)), assessment_id)
        return AIAnalysis(summary="Flood model output returned by the configured integration; review model assumptions before underwriting.", findings=findings, sources=[s for a in extraction.assets for s in a.sources], model_results=results)


def generate_underwriting_analysis(assessment_id: str, **kwargs) -> dict:
    """Evidence-grounded intelligence without inventing catastrophe outputs."""
    from .underwriting_analysis import generate_underwriting_analysis as generate
    return generate(assessment_id, **kwargs)
