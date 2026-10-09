# Nzoia Risk Intelligence AI service

Python 3.11+ asynchronous FastAPI backend for document extraction, exposure validation,
and tool-assisted underwriting chat. The existing repository contained AI scaffolding
and empty modelling placeholders; this service does not implement a flood model.

## Run

From the repository root:

```bash
python3.11 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -e 'backend[test]'
cd backend
cp .env.example .env
# Edit .env with the selected provider's credentials and an API token.
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Visit `/docs` for the API schema. Configuration loads from environment variables and
`.env` in the current working directory. Use `NZOIA_LLM_PROVIDER=anthropic`, `openai`,
`gemini` or `openrouter`, with the matching `NZOIA_*_API_KEY`. Anthropic uses the
direct Claude Messages API with native structured output and tool use. OpenRouter supports
capability-aware routing; the local demo pins a free structured-output/tool model
to avoid inconsistent schema support from a rotating router. `NZOIA_LLM_MODEL` overrides
the adapter default. Secrets use Pydantic SecretStr. Missing credentials allow
startup and parsing/validation; LLM calls return 503. Provider schema failures or
outages return sanitized errors, never exception messages or request content.
OpenAI and OpenRouter structured extraction use their structured-output APIs; Anthropic
uses `output_config.format` with local Pydantic validation, and Gemini
uses a response schema plus local validation. Both adapters expose the same
`structured` and `chat` methods. Gemini preserves native tool response parts.

Set `NZOIA_API_TOKEN` and send `Authorization: Bearer <token>` on AI API calls.
Without a token the service is unauthenticated, intended for local development.
The shared token does not provide per-user assessment authorization. Add the host
platform's authentication and tenant ownership checks before multi-tenant deployment.

## API workflow

All routes are under `/api/ai`. All responses contain `status`, `assessment_id`,
`data`, `warnings`, and `errors`; errors contain stable `code` and sanitized `message`.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/assessments` | Create an assessment; returns 201 |
| POST | `/assessments/{id}/documents` | Multipart `file` upload; returns document ID |
| POST | `/assessments/{id}/extract` | Extract all uploaded documents and save validated output |
| POST | `/assessments/{id}/chat` | JSON `{"message":"What information is missing?"}` |
| GET | `/assessments/{id}/findings` | Validation findings and saved model analysis |
| POST | `/assessments/{id}/analyze` | Invoke injected flood-model adapter |
| POST | `/assessments/{id}/scenarios` | JSON `{"name":"Scenario", "parameters":{}}`; invoke adapter |

PDF text uses PyMuPDF; XLSX and CSV use pandas/openpyxl; DOCX paragraphs and tables
use python-docx. Locations and verbatim excerpts are preserved. Every extracted
asset and claim must cite a matching document segment. Invalid provenance rejects
the entire extraction. Missing information remains null. Currency conversion,
geocoding and OCR are not supplied. Scanned pages produce explicit warnings.
Uploads, archive expansion and extracted character counts have configurable limits.
Raw upload bytes are not retained; extracted document text and results are stored.

`NZOIA_STORAGE_DIRECTORY` holds a durable SQLite database (restricted directory and
file permissions). Re-extraction replaces results per document and invalidates saved
analysis. SQLite suits a single service deployment; a shared host can inject its
repository through `build_services(..., store=...)`. Implement the same store methods.
No migrations or tenancy integration with the placeholder `app` repositories exist.
Operational logging emits only event, assessment ID and error code; no document text,
filenames, user messages, provider errors or credentials. Configure handler/retention
in the host application's logging setup. Infrastructure/provider logs are separate.

## Integration contract

`ai.risk_analyzer.FloodModel` is a protocol with two async methods:

```python
async def analyze(assessment_id: str, exposures: list[dict]) -> dict: ...
async def scenario(assessment_id: str, exposures: list[dict], parameters: dict) -> dict: ...
```

Implement these with the teammate's actual model service. Supply output units,
provenance and model version in its JSON result; the AI checks JSON serializability
and finite values, not the model's scientific correctness. Exposure conversion is
separate: implement `ExposureMapper.map_assets(assets)` for the modelling contract.
The default mapper forwards canonical asset data with Decimal amounts as strings;
it performs no inference. Model execution requires assets, coordinates, insured
values, unique asset IDs and one currency. The model adapter owns any additional
requirements, calibration and scenario parameter semantics. Scenario names are
UI labels; the model receives only the supplied parameters.

Wire the adapter without changing AI modules:

```python
from ai.config import Settings
from api.ai_routes import build_services
from app.main import create_app

settings = Settings()
services = build_services(settings, flood_model=your_model_adapter,
                          mapper=your_exposure_mapper)
app = create_app(settings, services)
# Caller owns cleanup of injected services.store and injected LLM clients.
```

Existing applications can mount `api.ai_routes.router` and inject `get_services`
through FastAPI dependency overrides. Install the error handlers/authentication
policy from the app factory if hosting the router separately. Services also accept
an injected LLM and store for offline tests. The default model explicitly returns
integration-unavailable errors. No synthetic model outputs are supplied.

The agent only executes four allowlisted, schema-validated tools bound to the current
assessment, with bounded rounds and calls. Documents and tool outputs are untrusted
prompt data. Chat prose is LLM-generated: prompts and provenance checks do not prove
all prose or extracted field values correct. Review outputs before underwriting;
model results are accepted only from the configured model adapter. Chat now stores bounded conversation history. Numerical catastrophe answers are
rendered from typed backend outputs; general explanatory prose remains subject to review. Findings returns validation or the latest actual model analysis,
not a stored chat transcript. LLM extraction sends parsed text to the selected provider.

## Test

```bash
cd backend
.venv/bin/python -m pytest -q
```

Tests use temporary storage and injected LLMs; they make no paid API calls and do not
fabricate catastrophe results. They cover all supported document formats, limits,
provenance rejection, financial/date/coordinate validation, persistence, tool
restrictions, missing integration behavior, adapter forwarding, and HTTP envelopes,
authentication and confidential error sanitization. Live provider credentials and
an actual flood model are required for end-to-end external integration verification.

Provider integration references: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
and [Google Gen AI SDK](https://googleapis.github.io/python-genai/).

Previous AI foundation verification: Python 3.11.17, `python -m pytest -q`:
48 passed, with one upstream Starlette TestClient/httpx deprecation warning.
No live LLM calls or catastrophe-model calculations were performed.

## Field-level intelligent insurance ingestion

`ai.insurance_extraction.extract_insurance_document(file_path)` (also exported by
`ai.extraction_service`) is the reusable synchronous service. It returns a
JSON-compatible dictionary validated by `InsuranceDocumentResult`; serialize using
`json.dumps(result, allow_nan=False)`. Use `aextract_insurance_document` in async code.
Both accept optional `llm=` and `settings=` injection and close clients they create.
Malformed files, unsupported evidence and invalid provider JSON raise sanitized
`DocumentError`, `ExtractionError` or Pydantic validation errors. They never return
fabricated fallback facts when credentials or document text are unavailable.

```python
from ai.extraction_service import extract_insurance_document
result = extract_insurance_document('/path/to/offer.pdf')
print(result['facts'])
print(result['review_items'])
```

The pipeline validates extension, file signatures/container contents, upload size,
archive expansion and extracted text limits. PDF pages include ordered text and
separate table-cell segments with actual page numbers. Excel preserves every row,
including narrative titles and irregular/repeated headings; formulas are not run.
CSV supports comma, semicolon, tab and pipe delimiters with string-preserved values.
DOCX retains paragraph and table references; actual page numbers cannot be supplied
without a layout renderer. PDF parsing is serialized because PyMuPDF has shared
library state. Raw library diagnostics are suppressed; parser limitations become
explicit review findings rather than confidential exception messages.

An evidence-backed LLM classification is followed by bounded section extraction.
All extractable content is retained instead of discarding content based on headings.
Every present field has `value`, `status`, `confidence`, `sources`, `alternatives`,
and `review_reason`. Status is `not_provided`, `provided`, `uncertain`, or `contradicted`.
Confidence is an LLM interpretation score, **not a calibrated probability**.
An explicitly pending or uninterpretable field may be `uncertain` with null value,
evidence and a review reason; it is distinct from `not_provided`.
Parser pagination overrides omitted page numbers; incorrect citations are rejected.
Numeric values, currencies and measurement units must appear in cited evidence;
explicit thousand/million/billion expressions may be normalized. Unsupported number
formats require a revised extraction/human review instead of guessed conversion.
Building values require an explicit building-and-amount source association; citing
an overall property sum cannot populate a building value.

The schema covers insured details, building characteristics and elevations, property,
machinery, inventory and interruption exposure, coverage clauses, deductibles,
limits, retention, participation, premium, flood history, claims versus settlements,
interruption durations, and risk factors. Missing values stay null; documented zero
stays zero. Financial categories are never added. Explicit financial relationships
are kept separately; missing or conflicting relationships generate review items.
The overall property sum is never reconciled against structure values as an equality.

Deterministic reconciliation combines matching documented building names and event
dates, preserves evidence and retains conflicting values as alternatives. Ambiguous
names/dates and different scopes should be reviewed, rather than assumed unique.
A final LLM semantic audit flags narrative/table or cross-field contradictions with
two distinct verified evidence statements. Oversized audit inputs are marked as
incomplete rather than silently treated as contradiction-free. Semantic findings
remain interpretations requiring review, not proof of inconsistency or guaranteed
complete detection. Raw-field factual meaning still needs underwriting review.
Documentary recommendations retain broker/insurer/surveyor/insured/unknown origin;
the extraction schema does not generate AI recommendations.

New API routes use uploaded document IDs, never caller-supplied server file paths:

- POST `/api/ai/assessments/{id}/documents/{document_id}/insurance-extract`
- GET `/api/ai/assessments/{id}/documents/{document_id}/insurance-extraction`

Rich results are persisted separately in SQLite and scoped to the assessment.
Existing canonical asset/claim extraction routes remain compatible. Rich results
are not automatically converted into a flood-model portfolio: missing individual
values, scope and financial relationships must be resolved first.

For real extraction using configured credentials:

```bash
cd backend
.venv/bin/python -m ai.insurance_cli /path/to/offer.pdf --output result.json
```

To test the supplied sample (now present at the repository root):

```bash
export NZOIA_SAMPLE_OFFER_PATH=/absolute/path/OFFER_NZOIA_GRAIN_PROCESSING.pdf
.venv/bin/python -m pytest tests/test_sample_offer.py -q -rs
# With the selected provider credentials configured, run live extraction explicitly:
NZOIA_RUN_LIVE_EXTRACTION=1 .venv/bin/python -m pytest tests/test_sample_offer.py -q -rs
```

The real sample arrived during dashboard implementation: its 32 pages now pass
the parser/provenance integration test. No provider credentials are configured,
so live structured extraction remains unverified and explicitly skips. No sample
values or company names are hardcoded or replaced by fixture conclusions.
An offline contract demonstration is available in
[examples/synthetic_insurance_result.json](examples/synthetic_insurance_result.json)
and its [review report](examples/synthetic_insurance_review.md). This uses a generated
PDF and injected fixture responses, **not the supplied offer or a live AI extraction**.
Regenerate from `backend` with `python -m examples.generate_synthetic_demo`.
Parser reference documentation: [PyMuPDF page/table API](https://pymupdf.readthedocs.io/en/latest/page.html#Page.find_tables)
and [pandas Excel reader](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html).

Earlier ingestion-only verification, before the sample arrived: **83 passed,
2 skipped**. Current combined verification is reported in the dashboard section below.

## Exposure mapping and underwriter review

The supplied root-level `exposure_nzoia_synthetic(in).csv` is now inspected: it has
500 records and columns `loc_id`, `lat`, `lon`, `housing_class`, `floor_area_m2`,
`cost_per_m2_kes`, `tiv_kes`, `synthetic`, and `source`. Observed housing classes are
`concrete_rcc`, `informal_iron_sheet`, `permanent_masonry`, and `semi_permanent`.
The grain-processing PDF is also present at the repository root.
No calibrated vulnerability catalog or catastrophe engine has been supplied; observed
CSV classes do not prove the supported vulnerability enumeration or occupancy
applicability. The six required fields remain a provisional engine interface, with
all original CSV columns retained in mapping provenance. A verified modelling-team
contract is required for executable exports; no catalog is guessed.

`ai.exposure_mapping` implements:

- `map_to_exposure_schema(data, contract=..., column_mapping=...)`
- `validate_exposure_records(records_or_batch, contract=...)`
- `get_missing_model_inputs(records_or_batch, contract=...)`
- `confirm_exposure_mapping(batch, decisions)`
- Async `map_property_description(...)` and `map_insurance_document(...)`

The first three are also available through `ai.exposure_mapper` and `ai.validator`.
`data` accepts CSV/XLSX paths, dictionaries, `InsuranceFacts`, or a validated
`InsuranceDocumentResult`. Natural-language extraction uses the configured LLM,
retains field evidence, and returns historical/document facts independently of model
readiness. These are callable services; exposure-specific HTTP routes/UI are not
implemented in this change. Existing upload and insurance extraction APIs can supply
their validated document results to this mapping service.

```python
from ai.exposure_mapping import (
    map_to_exposure_schema, get_missing_model_inputs, confirm_exposure_mapping,
)
from ai.exposure_schemas import ExposureContract, Confirmation
from pathlib import Path

# Supplied and verified by the modelling teammate, including all extra fields,
# vulnerability class applicability and any construction correspondence rules.
contract = ExposureContract.model_validate_json(Path('model_contract.json').read_text())
batch = map_to_exposure_schema('actual_exposures.csv', contract=contract)
print(batch.model_dump(mode='json'))
print([item.model_dump() for item in get_missing_model_inputs(batch)])
```

`ExposureContract` defines the complete fields, required flags, numeric bounds,
extra enums, supported vulnerability classes and their occupancy/asset applicability.
Construction rules declare a source term, exact target class, rule ID, rationale,
and whether confirmation is required. No residential/industrial class names or
vulnerability curves are provided by the AI. Without a verified contract, model
export remains blocked. Class applicability is checked even after confirmation.
Industry text is retained as evidence; it does not automatically establish occupancy.

CSV values preserve leading-zero identifiers. Excel narrative rows and repeated
headers are supported; ambiguous or duplicate headers must be resolved in a cleaned
schedule. Canonical columns and explicit aliases map transparently. Ambiguous matches
require a caller-selected `column_mapping`; generic financial columns additionally
require per-asset scope confirmation. All original rows are retained, with row/page
citations and separate `derived_fields` for aliases, construction rules, unit/currency
conversion and underwriting decisions. Decimal arithmetic preserves financial values;
JSON-compatible model records encode Decimal numbers as strings. The modelling
adapter must implement its verified engine serialization contract before execution.

Validation covers missing fields, incomplete or out-of-range coordinates, non-finite
or ambiguous numbers, negative values, zero building area, unsupported classes,
occupancy mismatches, duplicate IDs/records and repeated coordinates. Zero TIV is
preserved as zero. Duplicate points are reviewed as potential shared site locations,
not declared to be duplicate assets. Missing building location IDs remain missing;
only internal review record IDs are generated.

Facility coordinates stay in `site_coordinates`, with `coordinate_scope='site'`;
individual building lat/lon stay null. No building values are copied from the overall
property sum, and no automatic allocation/division occurs. The separate categories
buildings, machinery, inventory, contents and business interruption retain their
original financial evidence. Non-building categories never enter building-model
export, even after a housing-class confirmation. They require specialized models.

A confirmation includes record ID, current revision, field, explicit value, approved
flag, underwriter identity, reason, evidence and basis. Unapproved/stale decisions fail.
Missing or facility-level values require `explicit_asset_value_assumption`; all
originals remain intact. A site-coordinate proxy requires `site_coordinate_assumption`
and remains marked `assumed_site`, never represented as observed building GPS.
Confirmation re-runs validation and changes the revision; it cannot approve an
unsupported class or missing model contract. The host must authenticate the underwriter
and enforce tenant ownership before calling this service: core confirmation objects
are trusted application inputs, not an authentication mechanism. Persist batches and
apply revision comparison atomically in the host repository when exposing this via HTTP.

Square feet convert using the exact international-foot factor 0.09290304 m²/ft².
Foreign currency never converts automatically. `FXApproval` requires a supplied rate,
effective date, reference, explicit approval, approver and rationale; no live or
hardcoded market rates are supplied. Currency cells that conflict with `_kes` headers
remain reviewable. The supplied rate is an underwriting input, not verified market data.

Results contain `valid_records`, `review_required_records`, `model_records`,
`model_readiness`, immutable originals, derivation/confirmation history, and optional
`source_analysis`. `partial` means only the approved building subset is usable;
it must not be presented as a complete portfolio result. `unavailable` still permits
historical claims, document facts, missing-data and underwriting review. No loss
estimates or damage curves are calculated by this service.

Real-input test hooks use `NZOIA_EXPOSURE_CSV_PATH`, `NZOIA_EXPOSURE_CONTRACT_PATH`,
and `NZOIA_SAMPLE_OFFER_PATH`; live PDF extraction also uses
`NZOIA_RUN_LIVE_EXTRACTION=1` and the selected provider's credentials. Missing real
inputs skip explicitly, without substituting test-only catalogs or fixtures.

Earlier exposure-only verification, before the input files arrived: **104 passed,
5 skipped**. Current combined verification is below. Test-only catalogs and exchange
rates are explicitly labelled; they are not Nzoia model classes or market quotes.

## Tool-enabled underwriting assistant

`POST /api/ai/assessments/{assessment_id}/chat` accepts
`{"message":"Run the 100-year flood scenario.","conversation_id":"underwriting"}`.
The response retains the standard envelope and includes `summary`, `findings`,
verified `sources`, actual `model_results`, `tool_executions`, and allowlisted
`dashboard_updates`. Conversation history is persisted per assessment/conversation,
bounded to eight prior turns in context, and treated as untrusted/stale information.
Current assessment context and any numerical results are retrieved anew.

The requested backend functions now live in these modules:

| Tool | Backend implementation |
| --- | --- |
| `get_assessment` | `app/repositories/assessments.py` |
| `validate_portfolio` | `app/services/validation/portfolio.py` |
| `run_flood_model` | `app/services/loss/engine.py` |
| `get_model_results` | `app/repositories/results.py` |
| `compare_return_periods` | `app/services/loss/scenarios.py` |
| `get_document_evidence` | `app/services/ingestion/provenance.py` |
| `run_scenario` | `app/services/loss/scenarios.py` |

Assessment, validation and evidence functions read the real injected SQLite store.
The model functions delegate to `UnderwritingModelBackend`; they contain **no flood,
damage or loss calculations**. The catastrophe team's calibrated implementation
should provide `run_flood_model`, `get_model_results`, `compare_return_periods` and
`run_scenario`, using its actual hazard/vulnerability/loss services and result repository.
The existing hazard sampling and vulnerability curve modules remain placeholders.
Without an implementation, the default backend explicitly reports unavailable.

Wire the team's real functions without changing the agent:

```python
from ai.model_backend import CallableModelBackend
from api.ai_routes import build_services
from app.main import create_app
from ai.config import Settings

# Functions supplied by the modelling team; each returns the ModelOutput contract.
backend = CallableModelBackend(
    run_flood_model=team_run_flood_model,
    get_model_results=team_get_model_results,
    compare_return_periods=team_compare_return_periods,
    run_scenario=team_run_scenario,
)
settings = Settings()
services = build_services(settings, model_backend=backend)
app = create_app(settings, services)
```

`ai.underwriting_schemas.ModelOutput` is an explicit **AI adapter contract**, not an
inferred schema for an unavailable engine. The adapter must translate actual engine
outputs into assessment ID, status (`ready`, `no_results`, `missing_hazard_coverage`),
model version, result ID, and period results with currency, total loss and optional
per-building losses. A ready result requires actual period outputs and model version.
Wrong assessment IDs, missing requested periods, negative/non-finite losses and
malformed results are rejected. Model callbacks own calibration, hazard coverage,
input semantics, persistence and invalidation. No missing values or model totals
are synthesized. Building ranking sorts returned losses; it never estimates them.
Comparisons display the actual periods returned by the comparison backend.

Structured tool arguments require the current assessment ID, unique strict integer
periods (maximum ten), field paths or scenario parameter objects. Nested scenario
identity overrides are rejected. Tool schemas advertise only the seven approved
functions; private legacy compatibility tools cannot be invoked by the public agent.
Calls are bounded by rounds, total calls (including context), per-tool timeout,
per-LLM-call timeout, and overall chat timeout. Synchronous functions run in workers:
timeouts cannot terminate a running synchronous model calculation, so execution status
may be unknown and the modelling backend must supply job cancellation/status semantics.
Failures return sanitized codes/messages; provider/model exception text is not exposed.

Numerical catastrophe answers are rendered server-side from validated backend
outputs. Free-form LLM explanations cannot substitute loss values. Missing coverage,
insured values, unavailable integration or empty results produce explicit limitations
and no loss estimates. Historical claim totals are not future predictions. Successful
tools alone create dashboard actions, with server-bound assessment IDs and fixed action
names. Frontend teammates handle these data instructions; they are never executable code.

Every AI route now checks assessment ownership before reading assessment data. Creation
records the authenticated principal. `get_principal` maps the configured shared API
token to one service account; without a token, local development uses one local account.
For multi-user deployment, override `api.ai_routes.get_principal` with a dependency
that verifies the host's login/session and returns `Principal(subject=verified_user_id)`;
optionally inject the host `AssessmentAuthorizer` into `build_services`. Never derive
a production identity from an unverified header. Tool execution repeats the access
check and cannot switch assessment IDs. Unauthorized/not-found assessments both return
404. Existing assessments without an owner are denied until a trusted operator assigns
ownership through the host's migration/admin workflow; no automatic ownership guessing
is performed. Direct internal ToolRegistry use without a principal is trusted-only.

An end-to-end conversation is recorded at
[examples/underwriting_conversation_demo.json](examples/underwriting_conversation_demo.json).
It uses a **real persisted assessment**, created through the API, with clearly synthetic
offer data and a scripted tool-selecting LLM. It exercises actual repository, evidence,
validation, ownership and conversation services and reports unavailable model outputs;
it is not a live production assessment or genuine LLM/flood-model run. Regenerate with:

```bash
cd backend
.venv/bin/python -m examples.underwriting_conversation_demo
```

A genuine calibrated-model/LLM conversation still requires the modelling implementation,
actual assessment documents/exposures and configured provider credentials. Tool-selection
unit tests use scripted function-call responses; they verify routing, not live model
selection accuracy. Static response fixtures test adapter forwarding and grounding;
fixture losses are never used as application fallback results.

## Underwriting risk intelligence

`generate_underwriting_analysis(assessment_id)` is implemented in
`ai.underwriting_analysis` and re-exported by `ai.risk_analyzer`. It returns strict
JSON with exactly the requested keys: `risk_summary`, `risk_level`,
`top_risk_drivers`, `historical_claims_findings`, `model_findings`,
`recommended_actions`, `limitations`, and `citations`. `a_generate_underwriting_analysis`
is the async variant. Optional injection accepts the store, LLM, model backend,
settings and trusted authorization context.

```python
from ai.underwriting_analysis import generate_underwriting_analysis
result = generate_underwriting_analysis(assessment_id)
```

`POST /api/ai/assessments/{assessment_id}/underwriting-analysis` uses the same service,
checks ownership and returns the strict schema inside the standard envelope's `data`.
It reads validated insurance facts, source documents, canonical claims, quality
findings, model assumptions/limitations and actual results from the injected model
backend. It never executes a new flood simulation just to generate a report.

The LLM prioritizes existing evidence findings and explains their implications;
it cannot supply the report's numerical values, citations or risk grade. Narrative
output uses existing finding IDs and single-sentence, non-numerical explanations.
Unknown references, numerical/spelled-out statistical claims and duplicate driver
references are rejected. Safe deterministic findings remain available if narration
fails or credentials are absent, with an explicit limitation; no AI prioritization
is claimed in that fallback. The executive summary has at most three sentences,
labels document-reported/historical/modelled information, and includes only
server-controlled source numbers. At most three supported drivers are returned;
fewer are appropriate when evidence is insufficient.

Deterministic calculations in `ai.risk_calculations` compute:

- Historical totals separately by evidence stream, claimed/settled/paid category
  and currency; different categories, currencies and overlapping source streams
  are never merged. Unknown/repeated or overlapping partial event dates suppress
  cumulative totals and distinct-event counts rather than imply separate floods.
- Largest documented historical amounts within each category/currency, observed
  event counts, and repetitions of the same documented vulnerability description.
- Ranking of actual returned model building losses, class aggregates when coherent,
  shares against the actual same-period reported portfolio total, and comparisons
  between actual scenario totals. Zero denominators, duplicate building IDs and
  inconsistent breakdowns suppress concentration percentages. A returned subset
  is not described as a complete portfolio ranking.

Every consequential numerical finding references a document/model citation or a
calculation citation with function name, formula and input values/source IDs.
Document excerpts and pages are verified against stored source segments; numerical
amounts and units must match their evidence. Model citations retain the actual
backend response snapshot, model version and result ID when supplied. Paid
claims-register statistics require an explicit flood cause. Historical claims are
past observations, not predictions. Neither AAL, exceedance probability nor an
independent risk rating is inferred from claims or event counts. `risk_level.value`
remains null because no approved rating-framework output is supplied by this contract.

Documentary broker/insurer/surveyor/insured proposals retain their author origin;
they are separate from independent analytical actions. Insurer recommendations are
not automatically labelled mandatory conditions. Submitted offer terms are preserved
as document-reported proposals requiring acceptance/scope verification. Limitations
cover missing building values/locations, hazard coverage, unknown vulnerability
assumptions, industrial loss parameters, contradictory facts and ambiguous cover.
Machinery, inventory, contents and interruption losses are never derived using
residential building curves. Financial exposure categories are never added to infer
a property total or individual building values.

The named grain-processing PDF is now present and its parser integration passes,
but live structured extraction cannot run without configured provider credentials.
Its event count, warehouse condition and financial findings therefore remain
unconfirmed by this AI pipeline. Production code contains no company-name,
warehouse-number, event-count or grain-offer conclusions. The live extraction
integration runs when provider settings are supplied. Parameterized synthetic tests demonstrate varying event counts and verify
that unsupported risk statistics cannot enter the output.

An explicitly synthetic strict-schema result is provided at
[examples/synthetic_underwriting_analysis.json](examples/synthetic_underwriting_analysis.json).
It uses a real persisted test assessment, generated evidence and an injected narrative
fixture, with no model results. Regenerate using:

```bash
cd backend
.venv/bin/python -m examples.underwriting_analysis_demo --events 2
```

Earlier assistant/intelligence verification: **151 passed, 6 skipped** before the
input files arrived. Live LLM tool selection, calibrated model execution and
structured grain-offer findings remain unverified. Current verification is below.

## Dashboard intelligence API

`GET /api/assessments/{assessment_id}/dashboard` now returns the standard envelope
with a typed dashboard in `data`: six KPI cards, available charts, scoped map
locations, an evidence-grounded briefing, simultaneous workflow statuses and
auditable source references. See [the dashboard contract and integration guide](docs/dashboard.md)
for status rules, JSON schemas, complete/partial response examples and mapping hooks.

The endpoint reads stored actual model results, never executes a model or calls an
LLM. `POST /api/ai/assessments/{assessment_id}/underwriting-analysis` now persists
the briefing. A document/extraction revision change invalidates stored briefings
and mappings; changed model snapshots invalidate the briefing. A deterministic
briefing remains available if AI narration has not been generated. Financial
figures use exact decimal strings plus KES display text; absent values stay null.
An explicit requested flood-limit field and source-backed disjoint inventory line
items were added without assigning generic limits or manufacturing breakdowns.

The supplied CSV now passes real-file schema/mapping and partial-dashboard tests.
Its known building count and documented locations remain available even when model
class applicability is unverified. The supplied PDF passes actual parsing and
dashboard tests confirming that unextracted financial values remain unavailable.
Live extraction, an authoritative vulnerability catalog and calibrated flood
results still require external configuration/integration.

Current complete backend verification: **177 passed, 4 skipped** on Python 3.11.17,
including **24 dashboard integration/contract tests**. Three skips require live
LLM extraction credentials/opt-in and one requires the modelling team's verified
exposure contract. One upstream Starlette TestClient deprecation warning remains.
Static compilation and `git diff --check` passed. Complete/partial response examples
are explicitly synthetic contract fixtures, never runtime model fallbacks.

## Capability-driven what-if scenarios

`ai.scenario_service` exports `interpret_scenario_request()` and
`execute_scenario()`. Natural language is parsed into the discriminated Pydantic
types in `ai.scenario_schemas`; the LLM is prohibited from calculating losses or
mitigation effects. Local validation then checks the request against capabilities
declared by the numerical engine.

The modelling team should implement `ScenarioEngineBackend` from
`ai.what_if_service` with `get_scenario_capabilities`, `get_scenario_baseline` and
`run_what_if_scenario`. Both engine outputs must identify gross loss explicitly,
use the same assessment, currency and model version, and contain every requested
return period. The AI service computes absolute and percentage differences in
Python. A zero baseline produces a null percentage. The current deployment uses
`UnavailableScenarioEngine`, because no calibrated scenario implementation or
capability catalog is present; construction upgrades, elevation and deductibles
therefore cannot silently imply an effectiveness assumption.

## Supplied dataset knowledge base and underwriter app

The read-only knowledge base uses `NZOIA_DATASET_DIRECTORY` (default
`../datasets`) and derives its figures directly from the supplied CSV, GeoTIFFs,
DOCX metadata/problem statements and modelling guide. It exposes:

- `GET /api/knowledge/summary` for the versioned source catalog, portfolio
  profile, direct raster metadata and portfolio/hazard intersections.
- `GET /api/knowledge/search?q=...` for locally indexed, source-attributed
  reference evidence. This is lexical retrieval and requires no LLM key.
- `POST /api/knowledge/chat` with `{"message":"..."}` for concise,
  source-grounded underwriting answers. Common quantitative questions are answered
  deterministically from the CSV and rasters; unsupported loss questions return a
  clear model-input requirement rather than an estimate.
- `GET /underwriter` for the underwriter workspace.

The workspace labels the portfolio synthetic and reports positive-depth
intersections as exposure rather than loss. It intentionally leaves modelled loss
unavailable until approved depth-damage curves are configured. Start the API from
`backend/`, then open `http://127.0.0.1:8000/underwriter`. If
`NZOIA_API_TOKEN` is configured, enter that token in the connection box.
