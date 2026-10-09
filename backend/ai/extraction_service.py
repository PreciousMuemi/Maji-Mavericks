from .llm_client import LLMClient
from .prompts import EXTRACTION_SYSTEM
from .schemas import DocumentExtraction, ParsedDocument
import re
from decimal import Decimal


class ExtractionError(ValueError):
    pass


class ExtractionService:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def extract(self, document: ParsedDocument) -> DocumentExtraction:
        if not document.segments:
            raise ExtractionError("Document has no extractable text")
        result = DocumentExtraction.model_validate(await self.llm.structured(EXTRACTION_SYSTEM, document.model_dump_json(), DocumentExtraction))
        # Reject fabricated provenance rather than treating it as verified data.
        segments = {segment.locator: segment for segment in document.segments}
        for record in [*result.assets, *result.claims]:
            if not record.sources:
                raise ExtractionError("Extracted records require document provenance")
            for source in record.sources:
                segment = segments.get(source.locator)
                if source.document_id != document.document_id or segment is None or not source.excerpt.strip() or source.excerpt not in segment.text:
                    raise ExtractionError("Extracted provenance does not match the document")
                if source.page_number is not None and source.page_number != segment.page_number:
                    raise ExtractionError("Extracted source page does not match the document")
                source.page_number = segment.page_number
        # Reuse numerical checks from rich ingestion. Import locally to keep the
        # shared ExtractionError interface free of circular module initialization.
        from .insurance_extraction import _numbers, _validate_numeric
        from .insurance_schemas import Coordinates, Money

        def normalized(text):
            return ' '.join(text.casefold().split())

        for asset in result.assets:
            if not any(normalized(asset.name) in normalized(s.excerpt) for s in asset.sources):
                raise ExtractionError('Asset identity is not supported by its evidence')
            facility = (asset.asset_type or '').casefold() in {'facility', 'site', 'property'}
            if asset.insured_value is not None:
                _validate_numeric(Money(amount=asset.insured_value.amount, currency=asset.insured_value.currency), asset.sources)
                bound = any(normalized(asset.name) in normalized(s.excerpt) and asset.insured_value.amount in _numbers(s.excerpt)
                    and (facility or not re.search(r'(?:overall|total)\s+property|property\s+sum\s+insured', s.excerpt, re.I)) for s in asset.sources)
                if not bound:
                    raise ExtractionError('Individual asset value lacks asset-specific evidence')
            if asset.latitude is not None:
                coordinates = Coordinates(latitude=asset.latitude, longitude=asset.longitude)
                _validate_numeric(coordinates, asset.sources)
                if not facility:
                    bound = any(normalized(asset.name) in normalized(s.excerpt)
                        and not re.search(r'(?:site|facility).{0,20}(?:GPS|coordinates)', s.excerpt, re.I)
                        and all(number in _numbers(s.excerpt) for number in (Decimal(str(asset.latitude)), Decimal(str(asset.longitude)))) for s in asset.sources)
                    if not bound:
                        raise ExtractionError('Building coordinates require individual source evidence; site coordinates cannot be assigned')
        for claim in result.claims:
            if claim.paid_amount is not None:
                _validate_numeric(Money(amount=claim.paid_amount.amount, currency=claim.paid_amount.currency), claim.sources)
            if claim.event_date is not None:
                # Partial dates cannot become invented days. Numeric locale
                # formats remain ambiguous; use ISO or an explicit month name.
                formats = ('%Y-%m-%d', '%Y/%m/%d', '%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y')
                dates = [normalized(claim.event_date.strftime(fmt)) for fmt in formats]
                unpadded = [re.sub(r'\b0([1-9])\b', r'\1', text) for text in dates[2:]]
                if not any(text in normalized(s.excerpt) for s in claim.sources for text in [*dates, *unpadded]):
                    raise ExtractionError('Historical event date lacks an explicit full source date')
        result.warnings = [*document.warnings, *result.warnings]
        return result


def extract_insurance_document(file_path, *, llm=None, settings=None):
    """Rich insurance extraction; legacy canonical asset extraction stays compatible."""
    from .insurance_extraction import extract_insurance_document as extract
    return extract(file_path, llm=llm, settings=settings)
