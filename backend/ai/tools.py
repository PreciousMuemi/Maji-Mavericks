"""Allowlisted, assessment-bound tools calling actual backend service entry points."""
import asyncio
import json
from typing import Any

from pydantic import ValidationError

from app.repositories.assessments import get_assessment
from app.repositories.results import get_model_results
from app.services.ingestion.provenance import get_document_evidence
from app.services.loss.engine import run_flood_model
from app.services.loss.scenarios import compare_return_periods, run_scenario
from app.services.validation.portfolio import validate_portfolio

from .authorization import AssessmentAccessDenied, AssessmentAuthorizer
from .model_backend import UnderwritingModelBackend, UnavailableUnderwritingBackend
from .risk_analyzer import RiskAnalyzer, IntegrationUnavailable
from .scenario_service import ScenarioService
from .schemas import ScenarioRequest, StrictModel
from .store import AssessmentStore, AssessmentNotFound
from .underwriting_schemas import (
    AgentScenarioArguments, AssessmentArguments, CompareArguments, EvidenceArguments,
    FloodArguments, ModelOutput, Principal,
)
from .validator import validate_extraction


class EmptyArguments(StrictModel):
    pass


ARGUMENT_SCHEMAS = {
    'get_assessment': AssessmentArguments, 'validate_portfolio': AssessmentArguments,
    'run_flood_model': FloodArguments, 'get_model_results': AssessmentArguments,
    'compare_return_periods': CompareArguments, 'get_document_evidence': EvidenceArguments,
    'run_scenario': AgentScenarioArguments,
}
DESCRIPTIONS = {
    'get_assessment': 'Retrieve the authorized assessment and extracted facts; historical claims are not forecasts.',
    'validate_portfolio': 'Find missing insured values, coordinates and other inputs; document review works without a flood model.',
    'run_flood_model': 'Run the actual backend flood model for explicitly requested return periods. Never infer hazard coverage.',
    'get_model_results': 'Retrieve actual model outputs and per-building losses for ranking or explanation.',
    'compare_return_periods': 'Ask the backend to compare actual results for the requested return periods.',
    'get_document_evidence': 'Retrieve field-specific source excerpts from this authorized assessment.',
    'run_scenario': 'Run the actual backend scenario with explicit parameters; model preconditions and calibration belong to the backend.',
}
TOOL_SCHEMAS = [{'type': 'function', 'function': {'name': name, 'description': DESCRIPTIONS[name], 'parameters': schema.model_json_schema()}} for name, schema in ARGUMENT_SCHEMAS.items()]


class ToolRegistry:
    """HTTP callers supply a trusted principal/authorizer; direct use is internal only."""
    def __init__(self, assessment_id: str, store: AssessmentStore, analyzer: RiskAnalyzer, *, backend: UnderwritingModelBackend | None = None, principal: Principal | None = None, authorizer: AssessmentAuthorizer | None = None, timeout_seconds: float = 30):
        self.assessment_id, self.store, self.analyzer = assessment_id, store, analyzer
        self.backend = backend if backend is not None else UnavailableUnderwritingBackend()
        self.principal, self.authorizer, self.timeout_seconds = principal, authorizer, timeout_seconds

    async def require_access(self):
        if self.authorizer is not None:
            if self.principal is None:
                raise AssessmentAccessDenied('Assessment unavailable')
            await self.authorizer.require_access(self.principal, self.assessment_id)
        else:
            await asyncio.to_thread(self.store.documents, self.assessment_id)

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        phase = 'arguments'
        try:
            async with asyncio.timeout(self.timeout_seconds):
                await self.require_access()
                if name in {'get_exposures', 'validate_exposures', 'analyze_risk'} or (name == 'run_scenario' and 'name' in arguments and 'assessment_id' not in arguments):
                    return await self._legacy(name, arguments)
                if name not in ARGUMENT_SCHEMAS:
                    return self.error('unknown_tool', 'Tool is not allowed')
                parsed = ARGUMENT_SCHEMAS[name].model_validate(arguments)
                if parsed.assessment_id != self.assessment_id:
                    return self.error('assessment_scope_mismatch', 'Tools may only access the current authorized assessment')
                if len(json.dumps(arguments, allow_nan=False)) > 16_000:
                    return self.error('invalid_arguments', 'Tool arguments exceed the input limit')
                phase = 'backend'
                if name == 'get_assessment':
                    result = await get_assessment(self.assessment_id, store=self.store)
                elif name == 'validate_portfolio':
                    result = await validate_portfolio(self.assessment_id, store=self.store)
                elif name == 'get_document_evidence':
                    result = await get_document_evidence(self.assessment_id, parsed.field, store=self.store)
                else:
                    if name in {'run_flood_model', 'run_scenario'}:
                        extraction = await asyncio.to_thread(self.store.extraction, self.assessment_id)
                        blockers = {'missing_coordinates', 'missing_insured_value', 'duplicate_asset_id', 'mixed_currencies'}
                        if not extraction.assets or any(f.code in blockers for f in validate_extraction(extraction)):
                            return self.error('missing_model_inputs', 'Flood modelling requires verified building coordinates, individual insured values and consistent currency. Document/history analysis remains available.')
                    if name == 'run_flood_model':
                        result = await run_flood_model(self.assessment_id, parsed.return_periods, backend=self.backend)
                    elif name == 'get_model_results':
                        result = await get_model_results(self.assessment_id, backend=self.backend)
                    elif name == 'compare_return_periods':
                        result = await compare_return_periods(self.assessment_id, parsed.periods, backend=self.backend)
                    else:
                        result = await run_scenario(self.assessment_id, parsed.parameters, backend=self.backend)
                    output = ModelOutput.model_validate(result)
                    if output.assessment_id != self.assessment_id:
                        return self.error('invalid_model_output', 'Backend output does not belong to the authorized assessment')
                    if output.status == 'missing_hazard_coverage':
                        periods = ', '.join(str(p) for p in output.missing_return_periods)
                        return self.error('missing_hazard_coverage', f'Required hazard coverage is missing{": return periods " + periods if periods else ""}. No flood losses were returned.')
                    if output.status != 'ready':
                        return self.error('no_model_results', 'The backend returned no flood-model results. No losses can be predicted.')
                    requested = getattr(parsed, 'return_periods', getattr(parsed, 'periods', None))
                    if requested is not None and set(requested) != {p.return_period for p in output.period_results}:
                        return self.error('invalid_model_output', 'Backend output does not cover the requested return periods')
                    result = output.model_dump(mode='json')
                json.dumps(result, allow_nan=False)
                return {'status': 'success', 'data': result}
        except (AssessmentAccessDenied, AssessmentNotFound):
            return self.error('assessment_unavailable', 'Assessment is unavailable')
        except ValidationError:
            return self.error('invalid_arguments' if phase == 'arguments' else 'invalid_model_output', 'Tool arguments or backend output do not match the required schema')
        except TimeoutError:
            return self.error('tool_timeout', 'Backend function timed out; execution status may be unknown. No result was accepted.')
        except IntegrationUnavailable:
            return self.error('integration_unavailable', 'Flood model integration is unavailable or required inputs are incomplete')
        except (ValueError, TypeError):
            return self.error('invalid_arguments_or_output', 'Tool arguments or backend output are invalid')
        except Exception:
            return self.error('tool_failed', 'Tool execution failed')

    @staticmethod
    def error(code, message):
        return {'status': 'error', 'code': code, 'message': message}

    async def _legacy(self, name, arguments):
        """Keep internal compatibility; legacy names are not advertised to the LLM."""
        parsed = ScenarioRequest.model_validate(arguments) if name == 'run_scenario' else EmptyArguments.model_validate(arguments)
        if name == 'run_scenario':
            AgentScenarioArguments(assessment_id=self.assessment_id, parameters=parsed.parameters)
        snapshot = await asyncio.to_thread(self.store.dashboard_snapshot, self.assessment_id)
        extraction = snapshot['extraction']
        if name == 'get_exposures':
            result = extraction.model_dump(mode='json')
        elif name == 'validate_exposures':
            result = [f.model_dump(mode='json') for f in validate_extraction(extraction)]
        elif name == 'analyze_risk':
            analysis = await self.analyzer.analyze(self.assessment_id, extraction)
            saved = await asyncio.to_thread(self.store.save_analysis, self.assessment_id, analysis, fingerprint=snapshot['fingerprint'])
            if not saved:
                return self.error('assessment_changed', 'Inputs changed during calculation; no result was accepted. Rerun against current records.')
            result = analysis.model_dump(mode='json')
        else:
            result = await ScenarioService(self.analyzer).run(self.assessment_id, extraction, parsed)
        json.dumps(result, allow_nan=False)
        return {'status': 'success', 'data': result}
