"""Explicit callable adapter; no hazard or financial calculations are implemented here."""
import asyncio
import inspect
from typing import Callable, Protocol

from .risk_analyzer import IntegrationUnavailable
from .underwriting_schemas import ModelOutput


class UnderwritingModelBackend(Protocol):
    async def run_flood_model(self, assessment_id: str, return_periods: list[int]) -> ModelOutput: ...
    async def get_model_results(self, assessment_id: str) -> ModelOutput: ...
    async def compare_return_periods(self, assessment_id: str, periods: list[int]) -> ModelOutput: ...
    async def run_scenario(self, assessment_id: str, parameters: dict) -> ModelOutput: ...


class UnavailableUnderwritingBackend:
    async def run_flood_model(self, *args):
        raise IntegrationUnavailable('Catastrophe backend is not configured')

    async def get_model_results(self, *args):
        raise IntegrationUnavailable('Catastrophe backend is not configured')

    async def compare_return_periods(self, *args):
        raise IntegrationUnavailable('Catastrophe backend is not configured')

    async def run_scenario(self, *args):
        raise IntegrationUnavailable('Catastrophe backend is not configured')


class CallableModelBackend:
    """Bind the team's actual functions; each owns conversion to ModelOutput.

    Sync functions execute in a worker. A timeout cannot stop an already running
    synchronous calculation; the host must provide cancellation/job status semantics.
    """
    def __init__(self, *, run_flood_model: Callable, get_model_results: Callable, compare_return_periods: Callable, run_scenario: Callable):
        self.functions = dict(run_flood_model=run_flood_model, get_model_results=get_model_results, compare_return_periods=compare_return_periods, run_scenario=run_scenario)

    async def _call(self, name, *args):
        function = self.functions[name]
        result = await function(*args) if inspect.iscoroutinefunction(function) else await asyncio.to_thread(function, *args)
        if inspect.isawaitable(result):
            result = await result
        return ModelOutput.model_validate(result)

    async def run_flood_model(self, assessment_id, return_periods):
        return await self._call('run_flood_model', assessment_id, return_periods)

    async def get_model_results(self, assessment_id):
        return await self._call('get_model_results', assessment_id)

    async def compare_return_periods(self, assessment_id, periods):
        return await self._call('compare_return_periods', assessment_id, periods)

    async def run_scenario(self, assessment_id, parameters):
        return await self._call('run_scenario', assessment_id, parameters)
