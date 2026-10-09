# Nzoia Risk Intelligence: Remaining Work and Recommended Solutions

Last updated: 9 October 2026

## Purpose

This document explains what remains before the upload-to-underwriting-report workflow
can be considered dependable. It also separates AI extraction work from catastrophe
model integrations so that each teammate knows what they need to provide.

## Current working capabilities

- FastAPI creates assessments and accepts PDF, DOCX, CSV and XLSX uploads.
- The supplied `OFFER_NZOIA_GRAIN_PROCESSING.pdf` uploads successfully and produces
  32 parser-controlled document segments.
- PDF page numbers, document locators, narrative text and detected tables are preserved.
- CSV, XLSX and DOCX content can be parsed with stable source locators.
- Pydantic schemas distinguish missing values from zero and reject malformed AI output.
- Source validation rejects invented pages, quotations, amounts, currencies and units.
- Exposure mapping, validation, underwriting analysis, dashboard and assistant interfaces
  are implemented with explicit catastrophe-model integration boundaries.
- The underwriter frontend supports automatic upload processing and assessment chat.
- The supplied Nzoia datasets and flood rasters are indexed by the knowledge service.

## Main demonstration blocker

The uploaded file is normally accepted and parsed. The visible failure occurs during
the next step: rich AI extraction.

The previous configuration used the free OpenRouter model
`nvidia/nemotron-3-super-120b-a12b:free`. During testing, free inference exhibited:

- requests exceeding the configured timeout;
- responses with no usable `choices`;
- JSON that does not satisfy the required Pydantic schema;
- missing evidence, confidence or status fields;
- paraphrased evidence that does not exactly match the parsed document; and
- incorrect measurement structure in large responses.

The backend intentionally rejects these responses. Accepting them would allow
unsupported insurance values into an underwriting report.

## Recommended immediate model configuration

Use Anthropic's direct Claude API with native structured output and tool support. The
configured starting point is:

```env
NZOIA_LLM_PROVIDER=anthropic
NZOIA_LLM_MODEL=claude-haiku-5-5
NZOIA_LLM_TIMEOUT_SECONDS=180
NZOIA_LLM_MAX_OUTPUT_TOKENS=32768
```

Keep the Anthropic API key only in `backend/.env`:

```env
NZOIA_ANTHROPIC_API_KEY=replace-with-the-real-key
```

Never commit `backend/.env` or paste its key into issues, documentation or chat.
Confirm that the Anthropic account can access the configured model, then run repeated
live extraction tests before the demonstration.

Restart the service after changing the configuration:

```bash
cd /home/gachuuri/KenyaRE/Maji-Mavericks/backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The application is then available at:

```text
http://127.0.0.1:8000/underwriter
```

## Required extraction architecture

A better model is necessary, but the model should not control source provenance.
Evidence binding must be deterministic.

### 1. Parser-controlled evidence

Continue assigning a stable locator to every page, paragraph and table row. The LLM
should return a value and one or more existing locator IDs. Python should retrieve the
exact quotation and page from those locators.

Python must then verify:

- names and dates occur in the cited source;
- amounts occur in the source with the stated currency;
- measurements occur next to the stated unit;
- site coordinates are not copied into individual buildings;
- portfolio totals are not assigned to individual assets; and
- claims history is not labelled as a future model prediction.

This approach tolerates harmless LLM paraphrasing while continuing to reject invented
facts.

### 2. Smaller category extractions

Do not request the entire insurance schema in one large completion. Extract these
categories independently:

1. Document classification and insured details
2. Buildings and construction details
3. Financial exposure
4. Insurance and reinsurance terms
5. Historical claims
6. Risk factors and documentary recommendations

Validate and save each successful category independently. Reconcile conflicting values
in deterministic Python code. Never ask the LLM to calculate totals, percentages,
rankings or loss differences.

### 3. Partial processing

An error in one category should not discard verified results from other categories.
The assessment should return one of these truthful states:

- `document_extracted`
- `validation_required`
- `partial_assessment`
- `model_ready`
- `calculation_completed`
- `hazard_coverage_unavailable`
- `insufficient_data`

Partial results must show their missing categories and review requirements.

### 4. Background processing

Upload and AI processing should not be one long HTTP request. The recommended flow is:

```text
Upload document
  -> return assessment and document IDs
  -> enqueue extraction job
  -> expose job/status endpoint
  -> frontend polls or uses server-sent events
  -> render verified categories as they complete
```

Provider timeouts should be reported as retryable extraction failures, not as file-upload
failures.

## OCR requirement

PyMuPDF extracts embedded PDF text; it does not currently OCR scanned pages. Add a
controlled OCR adapter for pages with no extractable text.

The OCR pipeline should:

1. Detect pages without embedded text.
2. Render only those pages at a bounded resolution.
3. Run OCR using an approved local or managed OCR service.
4. Store the OCR result with page number and an `ocr_derived` marker.
5. Require review for low-confidence amounts, coordinates and dates.

OCR text must never silently replace higher-quality embedded text.

## Catastrophe-model integrations still required

The AI service must not invent flood losses. The modelling team needs to supply concrete
implementations for:

- `get_assessment(assessment_id)`
- `validate_portfolio(assessment_id)`
- `run_flood_model(assessment_id, return_periods)`
- `get_model_results(assessment_id)`
- `compare_return_periods(assessment_id, periods)`
- `get_document_evidence(assessment_id, field)`
- `run_scenario(assessment_id, parameters)`

They must also provide:

- the authoritative exposure schema and housing-class enumeration;
- calibrated and versioned vulnerability curves;
- verified raster coverage, CRS and nodata rules;
- model version and result IDs;
- supported return periods;
- treatment of deductibles, limits and retentions; and
- explicit industrial loss functions for machinery, inventory, contents and business
  interruption, if those calculations are supported.

Residential building curves must not be applied to industrial non-building assets.

## Responsibilities

### AI/backend teammate

- Implement locator-based evidence hydration.
- Split extraction into category-level schemas.
- Add retries only for provider and schema failures.
- Preserve strict local Pydantic and provenance validation.
- Add OCR through an explicit adapter.
- Persist partial category results and processing status.
- Keep prompts resistant to instructions embedded in uploaded documents.

### Catastrophe-modelling teammate

- Implement the approved model tool interfaces.
- Supply hazard and vulnerability contracts with model versions.
- Return explicit errors for missing hazard coverage or insured values.
- Provide deterministic building loss, return-period comparison and scenario results.
- Do not calculate unsupported industrial asset losses.

### Frontend teammate

- Display upload, parsing, extraction, validation and modelling as separate stages.
- Render partial results and missing information clearly.
- Present unavailable values as unavailable rather than zero.
- Keep historical claims visually distinct from modelled future loss.
- Display evidence pages and review status beside consequential facts.
- Keep chat available after upload, including when extraction needs review.

### Project owner or administrator

- Fund or provide access to a reliable LLM endpoint.
- Place credentials in `backend/.env` without committing them.
- Provide representative anonymized insurance documents for evaluation.
- Confirm the expected document formats and maximum file size.
- Obtain the modelling team's authoritative integration configuration.

## Testing required before demonstration

Run the following from `backend`:

```bash
.venv/bin/pytest -q
```

Then perform repeated live tests rather than relying only on injected test doubles:

1. Upload the supplied grain-processing PDF.
2. Confirm every text-bearing page is represented.
3. Confirm the insured, assets, financial values, terms and claims against source pages.
4. Confirm contradictions and missing fields are shown for review.
5. Upload an unfamiliar narrative offer.
6. Upload valid CSV and XLSX exposure schedules.
7. Test a scanned PDF and verify that it triggers OCR or a clear OCR-required result.
8. Test malformed, encrypted, empty and oversized files.
9. Simulate provider timeout, malformed JSON and unavailable model responses.
10. Run catastrophe calculations only when the real engine reports readiness.
11. Verify dashboard values match saved validated records.
12. Ask chat questions and confirm every consequential answer uses document evidence or
    an actual model result.

Repeat the live PDF test several times to detect provider instability.

## Acceptance criteria

The critical path is ready only when all of the following are true:

- Upload success is reported independently from extraction success.
- Every supported text-bearing document returns parser diagnostics.
- Scanned pages are OCR-processed or explicitly identified as unsupported.
- Every accepted fact has a valid parser-controlled source reference.
- Missing and zero values remain distinct.
- No asset value is allocated from a facility total without an approved assumption.
- No historical claim is presented as a future prediction.
- Invalid LLM output is rejected without deleting valid partial results.
- The supplied PDF and unfamiliar documents pass repeated live extraction runs.
- No loss appears unless the approved deterministic catastrophe engine returned it.
- Model errors and missing hazard coverage result in honest partial or unavailable states.

## Existing detailed diagnosis

See [`backend/qa/UPLOAD_FAILURE_REPORT.md`](backend/qa/UPLOAD_FAILURE_REPORT.md) for the
observed upload and extraction failure chain and the safeguards already implemented.
