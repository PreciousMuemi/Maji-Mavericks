# Report upload and extraction readiness

## Current result

The file-upload API is working for the supplied grain-processing PDF. A live probe
returned HTTP 201, parsed all 32 document segments and reported no upload errors.
The visible failure occurs in the subsequent rich AI extraction request, so the UI
message can be mistaken for an upload failure even though the document was stored.

## Failure chain

| Stage | Current state | Failure mode |
|---|---|---|
| Assessment creation | Working | Returns HTTP 201. |
| Multipart upload | Working for supplied PDF | Rejects empty, oversized, encrypted, malformed, or unsupported files. |
| PDF/DOCX/CSV/XLSX parsing | Working for text-bearing supported files | Scanned PDF pages have no OCR; DOCX has no reliable page numbering; spreadsheet formulas are not executed. |
| LLM classification | Intermittent | Free OpenRouter capacity can be slow or unavailable. |
| Structured fact extraction | Demo blocker | The configured free model can time out, omit required schema fields, or paraphrase evidence. |
| Provenance validation | Correctly fail-closed | Any invented locator, wrong page, changed number/unit, or non-matching quotation rejects the result with HTTP 422. Whitespace-only quote differences are repaired. |
| Underwriting analysis | Depends on accepted extraction | It cannot safely create the final report when extraction is rejected. |
| Chat | Available after upload | It remains subject to free-provider latency and must not present rejected extraction as verified fact. |

## Confirmed causes

1. The current provider is OpenRouter using `nvidia/nemotron-3-super-120b-a12b:free`
   with a 60-second request timeout. Free routing has variable queueing, latency and
   structured-output reliability.
2. The rich `InsuranceFacts` contract is large. The model must return nested values,
   confidence, status and exact evidence for every present fact. Large completions can
   exceed the provider timeout or fail local Pydantic validation.
3. Model-generated evidence is untrusted. The service deliberately requires the
   document ID, locator, page and quotation to match parser-controlled content. This is
   why paraphrased or fabricated citations are rejected rather than silently accepted.
4. PyMuPDF extracts embedded text but the application has no OCR integration. A scanned
   or image-only report therefore cannot be extracted reliably.
5. The frontend runs upload, extraction and analysis as one synchronous sequence. A
   downstream extraction error is displayed in the upload workflow, and no partial
   parser report is currently rendered.

## Changes already applied

- Provider-facing schemas remove an unsupported Decimal look-ahead expression while
  retaining full local Pydantic validation.
- OpenRouter handles missing choices and attempts one JSON fallback; invalid fallback
  output is rejected.
- Extraction sections are bounded to reduce request and response size.
- PDF line-wrap and whitespace-only citation differences are aligned to the exact
  parser text. Changed values remain rejected.
- Safe operational logs distinguish source, numeric and general extraction rejection.
- Chat appears after document upload even when rich extraction fails.
- The browser no longer advertises unsupported legacy `.xls`; supported spreadsheet
  uploads are `.xlsx`.

## Required work for a dependable demo

### P0: replace the free extraction dependency

Configure a provider/model with dependable JSON-schema output, sufficient output-token
limits and predictable latency. A trained model is useful only if it is exposed through
the provider adapter and can satisfy the same structured schema and evidence contract.
Run the supplied PDF repeatedly before the demo; one successful request is insufficient
to establish reliability.

### P0: make evidence binding deterministic

Have the LLM return parser-owned segment IDs and candidate values. The server should
copy the exact quotation and page from the selected segment, then verify that numbers,
currency, units, names and dates occur in that source. This avoids rejecting harmless
model paraphrases without weakening the rule against fabricated facts.

### P0: separate upload from processing

Return immediately after upload, enqueue extraction as a job, and expose a processing
status endpoint or event stream. The UI should show independent states such as uploaded,
text extracted, AI extraction running, review required and failed. Provider timeouts
must not look like file-upload failures.

### P1: reduce each LLM task

Extract insured details, assets, financial exposure, terms, claims and risk factors with
smaller schemas, then reconcile them deterministically. Use bounded concurrency and
per-stage retries. Never retry a provenance rejection by accepting less evidence.

### P1: add OCR and document diagnostics

Add a controlled OCR adapter for pages without embedded text. Before starting the LLM,
return page count, pages with text, pages requiring OCR, extracted tables and truncation
warnings. Keep OCR text marked as OCR-derived evidence.

### P1: support partial results

Render parser results and review warnings even when AI extraction fails. Save successful
category extractions independently, label them partial, and do not produce modelled loss
or a completed status until required inputs and integrations are available.

## Acceptance criteria

- Supported files upload and parse with a clear stage-specific status.
- Image-only pages trigger OCR or a specific OCR-required response.
- Every accepted fact resolves to parser-controlled evidence and the correct page.
- Missing fields remain unavailable rather than zero.
- Provider timeout or invalid JSON produces a retryable extraction failure, not an
  upload failure or fake report.
- The supplied PDF and an unfamiliar document pass repeated live end-to-end runs.
- Invalid files, altered citations, incorrect units and invented monetary values remain
  rejected.
- Chat clearly distinguishes raw document retrieval, accepted facts, historical claims
  and actual catastrophe-model results.
