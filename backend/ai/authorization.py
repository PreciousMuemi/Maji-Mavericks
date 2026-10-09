"""Assessment ownership checks; application hosts may replace this policy."""
import asyncio
from typing import Protocol

from .store import AssessmentStore
from .underwriting_schemas import Principal


class AssessmentAccessDenied(LookupError):
    pass


class AssessmentAuthorizer(Protocol):
    async def require_access(self, principal: Principal, assessment_id: str) -> None: ...


class OwnershipAuthorizer:
    def __init__(self, store: AssessmentStore):
        self.store = store

    async def require_access(self, principal: Principal, assessment_id: str):
        owner = await asyncio.to_thread(self.store.owner, assessment_id)
        if owner != principal.subject:
            raise AssessmentAccessDenied('Assessment is unavailable')
