"""Only allowlisted operational metadata enters application logs."""
import json
import logging
from uuid import UUID

logger = logging.getLogger("nzoia.ai")


def log_event(event: str, *, assessment_id: str | None = None, code: str | None = None):
    # Path parameters are user-controlled; never log arbitrary identifier text.
    if assessment_id is not None:
        try:
            assessment_id = str(UUID(assessment_id))
        except ValueError:
            assessment_id = None
    logger.info(json.dumps({"event": event, "assessment_id": assessment_id, "code": code}, separators=(",", ":")))
