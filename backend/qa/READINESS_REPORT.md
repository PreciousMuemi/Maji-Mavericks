# Nzoia AI Intelligence Layer — Demo Readiness Report

Run date: 8 October 2026 (Africa/Nairobi). Verdict: **NOT READY for the critical
end-to-end underwriting demo**. Document parsing and honest partial responses work,
but live AI extraction and catastrophe calculations cannot run in the supplied
configuration.

## Tests executed and outcomes

The real HTTP runner started Uvicorn, used authenticated multipart requests, and
created isolated durable SQLite data. It uploaded the supplied 32-page PDF, the
supplied 500-row CSV, a generated XLSX schedule with independently recorded expected
values, and an unfamiliar DOCX. No HTTP test doubles were used. Full machine-readable
results are in `qa/live_api_results.json`; private response evidence is under the
artifact directory named there.

| Required sequence | Result | Evidence |
|---|---|---|
| Start backend; upload PDF | Pass | Real Uvicorn; HTTP 201 |
| Text extraction covers all pages | Pass | Text stored for all 32 pages |
| Insured, financial terms, assets, claims, contradictions | Blocked | Both AI extraction endpoints return `503 llm_unavailable` |
| Location against raster extent | Blocked | No raster/catalog and no verified extracted coordinate |
| Supported loss calculations | Blocked | No catastrophe backend; explicit HTTP 503 |
| Dashboard | Pass, partial only | HTTP 200; unavailable KPIs are null; no fabricated charts/status |
| Evidence-grounded AI summary | Blocked for live AI | Deterministic fallback is clearly labelled and contains no model claims |
| Follow-up underwriting questions | Blocked | Four real chat requests return HTTP 503 |
| What-if scenario | Blocked in deployment | Capability-driven service is implemented and unit tested; real engine absent |
| Cross-API financial consistency | Pass for unavailable state | Values remain null consistently; populated reconciliation is blocked |
| Unfamiliar document | Parser pass; AI blocked | DOCX produced 13 stored source segments |
| Invalid uploads and missing data | Pass | Empty/corrupt/spoofed/oversize/invalid CSV cases rejected |
| Tool authorization and failures | Pass in automated tests | Identity override, wrong bearer token, timeout and unavailable-result paths covered |

The real-server run recorded **36 passed, 0 failed, and 24 blocked** checks. The
complete automated suite recorded **204 passed and 4 skipped** tests, with one
upstream Starlette TestClient deprecation warning. “Blocked”
means the test could not establish the required successful behavior because a real
dependency was absent; it is not counted as a pass. The complete automated suite
result is recorded in `qa/unit_integration_results.xml`. Tests marked skipped explicitly require live credentials or the
modelling team's verified exposure contract.

## Fixes and high-priority findings

Implemented fixes reject duplicate CSV headings and malformed row widths, prevent
dates from being accepted as flood depths, bind numeric evidence to units, verify
asset-specific values against named source evidence, reject site coordinates as
invented building coordinates, enforce full claim dates, invalidate stale cached
analysis, guard analysis persistence against input races, bound model calls, validate
typed model outputs, and reject nested assessment/tenant identity overrides.

The new what-if service accepts only engine-declared modifications. It rejects
unknown vulnerability classes, unsupported return periods, currency conversion,
different model versions, foreign assessment results and missing period outputs.
It computes monetary deltas and percentages in Python; a zero baseline yields an
unavailable percentage. It does not assign mitigation effectiveness.

## Missing integrations and demo-critical blockers

- No OpenAI/Gemini credential is configured, so correct extraction of the supplied
  offer—including the insured identity, historical floods and terms—has not been
  demonstrated through the AI pipeline.
- No flood raster, CRS/extent contract, hazard catalog, calibrated vulnerability
  catalog, industrial loss functions, or actual catastrophe/scenario backend is
  installed. No return-period loss can be truthfully produced.
- The uploaded CSV is parsed, but the live API does not yet persist an approved
  exposure mapping workflow into the dashboard. Mapping tests use an injected host
  integration fixture.
- Legacy and typed model interfaces both exist. The modelling team must bind them
  consistently and return input/model revision provenance to prevent externally
  cached stale results.
- Individual building GPS positions and insured values must remain missing where
  only facility-level coordinates or aggregate values are documented.
- Production multi-user deployment must replace the local bearer principal with the
  host identity/tenant authorizer.

These are demo-critical: the full PDF → extraction → validated exposure → hazard
coverage → loss → dashboard → conversational explanation path has not succeeded.
The current build is suitable for demonstrating parsing, validation, traceability,
and safe partial/failure behavior only.

## Start and verify

From the repository root:

```bash
cd /home/gachuuri/KenyaRE/Maji-Mavericks/backend
export NZOIA_STORAGE_DIRECTORY=.nzoia-data/demo
export NZOIA_API_TOKEN="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

In another shell, retain the same token value and create/upload an assessment:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/ai/assessments \
  -H "Authorization: Bearer $NZOIA_API_TOKEN" \
  -H 'Content-Type: application/json' -d '{}'

curl -sS -X POST http://127.0.0.1:8000/api/ai/assessments/ASSESSMENT_ID/documents \
  -H "Authorization: Bearer $NZOIA_API_TOKEN" \
  -F 'file=@../OFFER_NZOIA_GRAIN_PROCESSING.pdf;type=application/pdf'

curl -sS http://127.0.0.1:8000/api/assessments/ASSESSMENT_ID/dashboard \
  -H "Authorization: Bearer $NZOIA_API_TOKEN"

curl -sS -X POST http://127.0.0.1:8000/api/ai/assessments/ASSESSMENT_ID/chat \
  -H "Authorization: Bearer $NZOIA_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"What information is missing from this insurance offer?"}'
```

Run automated and real-server QA:

```bash
cd /home/gachuuri/KenyaRE/Maji-Mavericks/backend
.venv/bin/python -m pytest -q -rs tests --junitxml=qa/unit_integration_results.xml
.venv/bin/python -m qa.run_live_api
```

The live runner intentionally exits with status 2 while readiness is `NOT_READY`.

## Remaining risks and limitations

Unit doubles prove schema, routing, arithmetic and failure behavior; they do not
prove calibrated hazards, industrial vulnerability, provider latency, prompt
robustness or live document extraction accuracy. There is no OCR fallback, currency
conversion, approved independent risk-rating framework, or reliable way to cancel
an already-running synchronous host calculation after an API timeout. Historical
claims remain observations and are never represented as future predictions.
