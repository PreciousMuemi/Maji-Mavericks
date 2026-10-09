# Nzoia Risk Intelligence recommendations

## Objective

Make the upload-to-underwriting-report workflow dependable for unfamiliar insurance
documents while preserving strict source evidence and never accepting invented values.

## 1. Replace the free extraction model

Do not depend on a free OpenRouter model for the primary document-extraction path.
Free endpoints have variable queues, timeouts and structured-output reliability.

Use the existing OpenRouter API key with a low-cost model that supports strict JSON
Schema output and tool calling. The recommended initial configuration is:

```env
NZOIA_LLM_PROVIDER=openrouter
NZOIA_LLM_MODEL=openai/gpt-oss-120b
NZOIA_LLM_TIMEOUT_SECONDS=180
```

Model availability, capabilities and prices must be checked before deployment. The
model must be tested against the complete Nzoia extraction schema rather than selected
only from benchmark or marketing claims.

The trained model can replace this model later if it is exposed through an API that the
provider adapter can call and it reliably supports the required structured contract.

## 2. Bind evidence deterministically

The LLM should identify facts and parser-owned source segment IDs. It should not be the
authority that creates final quotations, page numbers or document IDs.

Use this workflow:

1. Parse each paragraph, page block and table row into a stable segment ID.
2. Send the relevant segments and their IDs to the LLM.
3. Ask the LLM to return extracted values and selected segment IDs.
4. Retrieve the exact quotation, page and document ID in Python.
5. Verify names, dates, amounts, currencies, units and coordinates against that source.
6. Reject unsupported fields without discarding unrelated valid categories.

This prevents harmless LLM paraphrasing from breaking provenance while continuing to
reject invented values, changed numbers and incorrect pages.

## 3. Divide extraction into smaller tasks

Run separate structured extraction steps for:

- Insured details
- Assets and construction details
- Financial exposure
- Insurance and reinsurance terms
- Historical claims and flood events
- Risk factors and documentary recommendations
- Contradiction review

Validate and save each category independently. Reconcile categories using deterministic
Python code. Apply bounded concurrency and limited retries so one slow provider request
does not hold the entire assessment indefinitely.

## 4. Separate upload from AI processing

Uploading and parsing should return immediately. Start AI extraction as a background
job and expose progress using a status endpoint or server-sent events.

The frontend should display independent states:

- Upload received
- Document parsed
- OCR required or running
- AI extraction running
- Evidence validation running
- Underwriter review required
- Report ready
- Extraction failed

A provider timeout must be shown as an extraction failure rather than an upload failure.

## 5. Add OCR for scanned reports

PyMuPDF extracts embedded text but does not recover text from image-only pages. Add a
controlled OCR adapter for pages with no extractable text.

OCR-derived text must:

- Retain its page number.
- Be labelled as OCR-derived evidence.
- Carry a review warning.
- Receive additional checks for monetary amounts, decimal points and dates.
- Never silently replace the original document evidence.

## 6. Support partial assessments

Do not discard all results because one category fails. Save and render each category
that passes schema and provenance validation. Mark the assessment as partial and list
the failed or unavailable categories.

The system must still avoid:

- Treating missing values as zero.
- Assigning a facility total to individual buildings.
- Adding potentially overlapping financial categories.
- Producing catastrophe losses without approved model results.
- Treating historical claims as future predictions.

## 7. Test unfamiliar documents

Create a regression collection containing:

- Text-based PDFs
- Scanned PDFs
- Multi-page offers with tables
- DOCX offers
- CSV exposure schedules
- XLSX schedules with multiple sheets
- Documents with inconsistent headings
- Documents with contradictory amounts
- Documents with missing coordinates or currencies
- Malformed, encrypted and oversized files

Run repeated live extraction tests. One successful provider response is not sufficient
to establish reliability.

## 8. Required user and team inputs

The project owner should provide:

- OpenRouter credit and the configured API key.
- Access details for the trained model when available.
- Representative anonymized insurance reports.
- Confirmation of supported document formats and maximum sizes.
- The catastrophe modelling API contract.
- Approved vulnerability classes and industrial loss assumptions.

API keys must remain in `backend/.env` and must never be committed or shared in chat.

## Recommended implementation order

1. Fund OpenRouter and configure a dependable structured-output model.
2. Implement parser-segment evidence binding.
3. Split extraction into independently validated categories.
4. Move extraction to a background job with visible progress.
5. Add OCR and OCR review statuses.
6. Add partial-result rendering.
7. Run repeated live tests against the supplied PDF and unfamiliar documents.
8. Connect the approved catastrophe model after validated exposures are available.

## Acceptance criteria

- Every supported file receives a clear upload and parsing result.
- Image-only pages trigger OCR or a specific OCR-required status.
- Every accepted fact has trusted document, locator, page and quotation evidence.
- Missing information is unavailable rather than zero.
- Invalid LLM output never produces a completed report.
- Valid categories survive unrelated category failures.
- The chatbot distinguishes document facts, historical claims and modelled results.
- The supplied and unfamiliar reports pass repeated end-to-end tests.
- No loss is generated when catastrophe-model inputs or integrations are unavailable.
