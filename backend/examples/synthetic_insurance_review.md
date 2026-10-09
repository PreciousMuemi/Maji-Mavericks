# Synthetic extraction demonstration

This result uses a generated two-page test PDF and an injected fixture LLM. It is not extraction of OFFER_NZOIA_GRAIN_PROCESSING.pdf and does not demonstrate live provider accuracy.

Review items:

- `facts.assets[0].condition`: Document sections give competing values; confirm whether scope, date or interpretation differs.
- `facts.assets[0].elevation`: Confirm whether elevation is relative to ground, river level, or a geodetic datum.
- `facts.assets[0].elevation_datum`: Not provided in extractable document content; do not substitute zero.
- `facts.assets[0].stated_value`: Not provided in extractable document content; do not substitute zero.
- `facts.financial_exposure.property_sum_insured`: The offer and revised schedule state different property sums.
- `facts.financial_exposure.business_interruption_limit`: Not provided in extractable document content; do not substitute zero.
- `facts.insurance_terms.coverage_requested`: Not provided in extractable document content; do not substitute zero.
- `facts.insurance_terms.limits`: Not provided in extractable document content; do not substitute zero.
- `facts.insurance_terms.retention`: Not provided in extractable document content; do not substitute zero.
- `facts.insurance_terms.reinsurance_participation`: Not provided in extractable document content; do not substitute zero.
- `facts.insurance_terms.premiums`: Not provided in extractable document content; do not substitute zero.
- `facts.flood_history[0].description`: Not provided in extractable document content; do not substitute zero.
- `facts.risk_factors.building_deterioration`: Not provided; do not interpret missing observations as absence of risk.
- `facts.risk_factors.flood_defenses`: Not provided; do not interpret missing observations as absence of risk.
- `facts.financial_exposure.property_sum_insured`: Relationship with machinery_values is unverified. Preserve both amounts separately; no total has been calculated.
- `facts.financial_exposure.property_sum_insured`: Relationship with inventory_values is unverified. Preserve both amounts separately; no total has been calculated.
- `facts.financial_exposure.machinery_values`: Relationship with inventory_values is unverified. Preserve both amounts separately; no total has been calculated.
