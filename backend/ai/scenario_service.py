from .risk_analyzer import RiskAnalyzer, IntegrationUnavailable, validate_model_output
from .schemas import DocumentExtraction, ScenarioRequest
from .validator import validate_extraction
from .what_if_service import execute_scenario, interpret_scenario_request

__all__ = ['ScenarioService', 'interpret_scenario_request', 'execute_scenario']


class ScenarioService:
    def __init__(self, analyzer: RiskAnalyzer):
        self.analyzer = analyzer

    async def run(self, assessment_id: str, extraction: DocumentExtraction, request: ScenarioRequest):
        blockers = {"missing_coordinates", "missing_insured_value", "duplicate_asset_id", "mixed_currencies"}
        if not extraction.assets or any(f.code in blockers for f in validate_extraction(extraction)):
            raise IntegrationUnavailable("Scenario analysis requires a complete, valid exposure portfolio")
        return validate_model_output(await self.analyzer.call_model('scenario', assessment_id, self.analyzer.mapper.map_assets(extraction.assets), request.parameters), assessment_id)
