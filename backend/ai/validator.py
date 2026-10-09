from collections import Counter
from .schemas import DocumentExtraction, ValidationFinding


def validate_extraction(extraction: DocumentExtraction) -> list[ValidationFinding]:
    findings = []
    def add(code, message, asset_id=None, severity="warning"):
        findings.append(ValidationFinding(code=code, severity=severity, message=message, asset_id=asset_id))
    ids = Counter(a.asset_id for a in extraction.assets)
    for asset_id, count in ids.items():
        if count > 1:
            add("duplicate_asset_id", "Asset identifier occurs more than once", asset_id, "error")
    for asset in extraction.assets:
        if asset.latitude is None:
            add("missing_coordinates", "Coordinates are required for hazard sampling", asset.asset_id)
        if asset.insured_value is None:
            add("missing_insured_value", "Insured value is required for financial loss estimation", asset.asset_id)
        elif asset.insured_value.amount == 0:
            add("zero_insured_value", "Insured value is zero", asset.asset_id)
    if len({a.insured_value.currency for a in extraction.assets if a.insured_value}) > 1:
        add("mixed_currencies", "Portfolio contains multiple currencies; conversion must be supplied before aggregation")
    claim_ids = Counter(c.claim_id for c in extraction.claims)
    if any(count > 1 for count in claim_ids.values()):
        add("duplicate_claim_id", "Claim identifier occurs more than once", severity="error")
    for claim in extraction.claims:
        if claim.asset_id is not None and claim.asset_id not in ids:
            add("unknown_claim_asset", "Claim references an unknown asset", claim.asset_id, "error")
    return findings


def validate_exposure_records(records, **kwargs):
    from .exposure_mapping import validate_exposure_records as validate
    return validate(records, **kwargs)


def get_missing_model_inputs(records, **kwargs):
    from .exposure_mapping import get_missing_model_inputs as missing
    return missing(records, **kwargs)
