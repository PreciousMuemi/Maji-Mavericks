"""Reusable staged extraction, evidence validation and conservative reconciliation."""
import asyncio
import json
import re
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from .config import Settings
from .document_parser import DocumentError, DocumentParser
from .extraction_service import ExtractionError
from .insurance_schemas import (
    Contradiction, ContradictionAudit, Coordinates, DocumentClassification, Fact, InsuranceDocumentResult,
    InsuranceFacts, InsuranceFinancialTerms, InsuranceHistoryRisk, InsuranceIdentityAssets,
    Interpretation, Measurement, Money, ReviewItem,
)
from .llm_client import LLMClient, create_llm_client
from .prompts import INSURANCE_CLASSIFICATION_SYSTEM, INSURANCE_EXTRACTION_SYSTEM, INSURANCE_CONTRADICTION_SYSTEM
from .schemas import DocumentSegment, ParsedDocument, SourceReference


def _key(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _unique_sources(sources):
    return list({_key(source): source for source in sources}.values())


def _walk(value, path='facts'):
    if isinstance(value, Fact):
        yield path, value
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _walk(getattr(value, name), f'{path}.{name}')
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _walk(item, f'{path}[{i}]')


def _numbers(excerpts: str) -> set[Decimal]:
    """Support literal numbers and explicit thousand/million/billion multipliers.

    No guessed currencies, exchange rates, totals or undocumented arithmetic.
    """
    values = set()
    pattern = r'(?<![\w.])[+-]?\d+(?:[,\u00a0 ]\d{3})*(?:\.\d+)?'
    for match in re.finditer(pattern, excerpts):
        number = Decimal(re.sub(r'[,\u00a0 ]', '', match.group()))
        values.add(number)
        scale = re.match(r'\s*(thousand|million|billion)\b', excerpts[match.end():], re.I)
        if scale:
            values.add(number * {'thousand': 1000, 'million': 1_000_000, 'billion': 1_000_000_000}[scale[1].lower()])
    return values


def _validate_numeric(value, sources):
    numbers = _numbers('\n'.join(source.excerpt for source in sources))
    if isinstance(value, Money):
        required = [value.amount]
        if value.currency is not None:
            text = ' '.join(source.excerpt for source in sources).upper()
            # Abbreviations with unambiguous ISO relationships only. "$" alone
            # does not identify USD and is deliberately not a supported alias.
            aliases = {'KES': ('KES', 'KSH', 'KSHS', 'KENYAN SHILLING'), 'USD': ('USD', 'US$', 'US DOLLAR'), 'EUR': ('EUR', 'EURO', '€'), 'GBP': ('GBP', 'POUND STERLING', '£')}
            if not any(alias in text for alias in aliases.get(value.currency, (value.currency,))):
                raise ExtractionError('Extracted currency is not supported by its evidence')
    elif isinstance(value, Measurement):
        required = [number for number in (value.value, value.lower, value.upper) if number is not None]
        unit_aliases = {
            'm': ('m', 'metres', 'meters', 'metre', 'meter'),
            'cm': ('cm', 'centimetres', 'centimeters'),
            'mm': ('mm', 'millimetres', 'millimeters'),
            'ft': ('ft', 'feet', 'foot'),
            'm2': ('m2', 'm²', 'sqm', 'sq m', 'square metres', 'square meters'),
            'days': ('days', 'day'), 'hours': ('hours', 'hour'),
            'weeks': ('weeks', 'week'), 'months': ('months', 'month'),
            'years': ('years', 'year'),
        }
        aliases = unit_aliases.get(value.unit.casefold(), (value.unit,))
        text = ' '.join(s.excerpt for s in sources)
        if not any(re.search(r'(?<![A-Za-z])' + re.escape(alias) + r'(?![\w²])', text, re.I) for alias in aliases):
            raise ExtractionError('Extracted measurement unit is not supported by its evidence')
        # A date, claim amount or other number on the same page is not evidence
        # of a depth/area/duration. Require a value tied to this exact unit.
        unit_pattern = r'(?:' + '|'.join(re.escape(a) for a in aliases) + r')(?![\w²])'
        number_pattern = r'(?<![\w.])([+-]?\d+(?:[,\u00a0 ]\d{3})*(?:\.\d+)?)'
        gap = r'\s*[\"\']?\s*(?:,\s*[\"\']?)?\s*'
        evidenced = set()
        suffix = number_pattern + gap + unit_pattern
        prefix = r'(?<![A-Za-z])' + unit_pattern + r'\s*[\"\')\]]?\s*[:=]\s*[\"\']?\s*' + number_pattern
        for pattern in (suffix, prefix):
            for match in re.finditer(pattern, text, re.I):
                evidenced.add(Decimal(re.sub(r'[,\u00a0 ]', '', match[1])))
        range_pattern = number_pattern + r'\s*(?:' + unit_pattern + r')?\s*(?:to|[-–—])\s*' + number_pattern + gap + unit_pattern
        for match in re.finditer(range_pattern, text, re.I):
            evidenced.update(Decimal(re.sub(r'[,\u00a0 ]', '', match[i])) for i in (1, 2))
        # Full table excerpts can also establish unit-bearing column headers.
        flat_rows = []
        for source in sources:
            try:
                table = json.loads(source.excerpt)
            except (ValueError, TypeError):
                continue
            row_locator = re.fullmatch(r'(sheet:.+):row:(\d+)', source.locator)
            if isinstance(table, list) and table and not any(isinstance(cell, (list, dict)) for cell in table) and row_locator:
                flat_rows.append((source.document_id, row_locator[1], int(row_locator[2]), table))
            if not isinstance(table, list) or not table or not all(isinstance(row, list) for row in table):
                continue
            for column, header in enumerate(table[0]):
                if not re.search(r'(?<![A-Za-z])' + unit_pattern, str(header), re.I):
                    continue
                for row in table[1:]:
                    cell = str(row[column]).strip() if column < len(row) else ''
                    if re.fullmatch(r'[+-]?\d+(?:[,\u00a0 ]\d{3})*(?:\.\d+)?', cell):
                        evidenced.add(Decimal(re.sub(r'[,\u00a0 ]', '', cell)))
        # XLSX preserves each row separately. Unit-bearing header and value
        # evidence must refer to the same source sheet and the same column.
        for document_id, sheet, row_number, header in flat_rows:
            for column, label in enumerate(header):
                if not re.search(r'(?<![A-Za-z])' + unit_pattern, str(label), re.I) or not re.search(r'[A-Za-z]', re.sub(unit_pattern, '', str(label), flags=re.I)):
                    continue
                for other_doc, other_sheet, other_row, values in flat_rows:
                    if (other_doc, other_sheet) != (document_id, sheet) or other_row <= row_number or len(values) != len(header):
                        continue
                    cell = str(values[column]).strip()
                    if re.fullmatch(r'[+-]?\d+(?:[,\u00a0 ]\d{3})*(?:\.\d+)?', cell):
                        evidenced.add(Decimal(re.sub(r'[,\u00a0 ]', '', cell)))
        if any(number not in evidenced for number in required):
            raise ExtractionError('Measurement value is not tied to its stated unit in the evidence')
    elif isinstance(value, Coordinates):
        required = [Decimal(str(value.latitude)), Decimal(str(value.longitude))]
    else:
        return
    if any(number not in numbers for number in required):
        raise ExtractionError('Extracted numeric value is not supported by its evidence')


class InsuranceDocumentExtractor:
    def __init__(self, llm: LLMClient, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or Settings()

    def _validate_sources(self, sources, document, *, repair_from_locator=False):
        segments = {segment.locator: segment for segment in document.segments}
        for source in sources:
            segment = segments.get(source.locator)
            if source.document_id != document.document_id or segment is None:
                raise ExtractionError('Extracted evidence does not match the document')
            if not source.excerpt.strip() or source.excerpt not in segment.text:
                # PDF/DOCX parsers often represent line wrapping as different runs of
                # whitespace. Recover the exact parser-owned substring when the LLM's
                # quote differs only in whitespace, then retain strict provenance.
                words = re.split(r'\s+', source.excerpt.strip())
                if words and all(words):
                    match = re.search(r'\s+'.join(re.escape(word) for word in words), segment.text)
                    if match:
                        source.excerpt = segment.text[match.start():match.end()]
                # Classification needs a provenance anchor rather than a field value.
                # A provider may paraphrase its excerpt despite selecting a genuine
                # locator. Replace only short, parser-controlled classification text;
                # fact and contradiction citations remain exact-match only.
                if source.excerpt in segment.text:
                    pass
                elif repair_from_locator and len(segment.text) <= 500:
                    source.excerpt = segment.text
                else:
                    raise ExtractionError('Extracted evidence does not match the document')
            if source.page_number is not None and source.page_number != segment.page_number:
                raise ExtractionError('Extracted evidence has an incorrect page number')
            source.page_number = segment.page_number  # Trusted parser supplies pagination.

    def _anchor_classification(self, classification, document):
        if classification.kind == 'unknown' or classification.sources:
            return classification
        terms = {
            'insurance_offer': ('insurance offer', 'offer identification'),
            'reinsurance_offer': ('reinsurance offer', 'facultative'),
            'exposure_schedule': ('exposure schedule', 'location schedule'),
            'policy': ('insurance policy', 'policy schedule'),
            'claims_report': ('claims report', 'claims history'),
            'mixed': ('insurance', 'claim'),
        }[classification.kind]
        segment = next((s for s in document.segments
                        if len(s.text) <= 500 and any(term in s.text.casefold() for term in terms)), None)
        if segment is None:
            raise ExtractionError('Document classification lacks source evidence')
        classification.sources = [SourceReference(document_id=document.document_id,
            locator=segment.locator, excerpt=segment.text, page_number=segment.page_number)]
        return classification

    def _validate_facts(self, facts, document):
        for _, fact in _walk(facts):
            self._validate_sources(fact.sources, document)
            if fact.value is not None:
                _validate_numeric(fact.value, fact.sources)
            for alternative in fact.alternatives:
                self._validate_sources(alternative.sources, document)
                _validate_numeric(alternative.value, alternative.sources)
        for asset in facts.assets:
            values = asset.stated_value.alternatives if asset.stated_value.status == 'contradicted' else ([Interpretation(value=asset.stated_value.value, sources=asset.stated_value.sources)] if asset.stated_value.value is not None else [])
            names = [a.value for a in asset.name.alternatives] if asset.name.status == 'contradicted' else [asset.name.value]
            for interpretation in values:
                # A portfolio amount cited near a building is insufficient. Require
                # explicit building identity and amount in the same source excerpt.
                bound = any(
                    any(name and name.casefold() in source.excerpt.casefold() for name in names)
                    and interpretation.value.amount in _numbers(source.excerpt)
                    and not re.search(r'(?:overall|total)\s+property|property\s+sum\s+insured', source.excerpt, re.I)
                    for source in interpretation.sources
                )
                if not bound:
                    raise ExtractionError('Individual asset value lacks explicit asset-specific source evidence')

    def _sections(self, document):
        """Keep all content; split oversized pages without losing original locators."""
        limit = self.settings.extraction_section_characters
        group, length = [], 200  # Reserve container/payload metadata overhead.
        for segment in document.segments:
            start = 0
            while start < len(segment.text):
                end = min(len(segment.text), start + limit - 400)
                piece = segment.model_copy(update={'text': segment.text[start:end]})
                while len(piece.model_dump_json()) > limit - 200:
                    end = start + (end - start) // 2
                    if end <= start:
                        raise ExtractionError('Source metadata exceeds the section budget')
                    piece = segment.model_copy(update={'text': segment.text[start:end]})
                cost = len(piece.model_dump_json()) + 1
                if group and (length + cost > limit or any(s.locator == piece.locator for s in group)):
                    yield ParsedDocument(document_id=document.document_id, segments=group)
                    group, length = [], 200
                group.append(piece)
                length += cost
                start = end
        if group:
            yield ParsedDocument(document_id=document.document_id, segments=group)

    def _merge_fact(self, left, right):
        if left.status == 'not_provided':
            return right.model_copy(deep=True)
        if right.status == 'not_provided':
            return left.model_copy(deep=True)
        if (left.status == 'uncertain' and left.value is None) or (right.status == 'uncertain' and right.value is None):
            known = right if left.value is None and left.status != 'contradicted' else left
            result = known.model_copy(deep=True)
            result.sources = _unique_sources([*left.sources, *right.sources])
            if result.status != 'contradicted':
                result.status = 'uncertain'
                result.confidence = min(left.confidence, right.confidence)
                result.review_reason = left.review_reason or right.review_reason
            return result
        interpretations = []
        for fact in (left, right):
            interpretations.extend(fact.alternatives if fact.status == 'contradicted' else [Interpretation(value=fact.value, sources=fact.sources)])
        grouped = {}
        for alternative in interpretations:
            key = _key(alternative.value)
            if key in grouped:
                grouped[key].sources = _unique_sources([*grouped[key].sources, *alternative.sources])
            else:
                grouped[key] = alternative.model_copy(deep=True)
        if len(grouped) > 1:
            return type(left)(status='contradicted', alternatives=list(grouped.values()), sources=_unique_sources([s for a in grouped.values() for s in a.sources]), review_reason='Document sections give competing values; confirm whether scope, date or interpretation differs.')
        result = left.model_copy(deep=True)
        result.sources = _unique_sources([*left.sources, *right.sources])
        if left.status == 'uncertain' or right.status == 'uncertain':
            result.status = 'uncertain'
            result.review_reason = left.review_reason or right.review_reason
        result.confidence = min(left.confidence, right.confidence)
        return result

    def _merge(self, left, right):
        if isinstance(left, Fact):
            return self._merge_fact(left, right)
        if isinstance(left, BaseModel):
            result = left.model_copy(deep=True)
            for name in type(left).model_fields:
                setattr(result, name, self._merge(getattr(left, name), getattr(right, name)))
            return result
        if isinstance(left, list):
            result = list(left)
            for incoming in right:
                identity = None
                if hasattr(incoming, 'name') and isinstance(incoming.name, Fact) and incoming.name.value is not None:
                    identity = ('name', incoming.name.value.strip().casefold())
                elif hasattr(incoming, 'event_date') and incoming.event_date.value is not None:
                    identity = ('event_date', incoming.event_date.value.strip().casefold())
                found = None
                if identity is not None:
                    for i, item in enumerate(result):
                        field = getattr(item, identity[0])
                        if field.value is not None and field.value.strip().casefold() == identity[1]:
                            found = i
                            break
                if found is not None:
                    result[found] = self._merge(result[found], incoming)
                elif _key(incoming) not in {_key(item) for item in result}:
                    result.append(incoming.model_copy(deep=True) if isinstance(incoming, BaseModel) else incoming)
            return result
        # No scalar numeric reconciliation or summation occurs here.
        return left if left == right else right

    def _reviews(self, facts, classification, document):
        reviews, contradictions = [], []
        if classification.kind == 'unknown' or classification.confidence < self.settings.extraction_review_confidence:
            reviews.append(ReviewItem(code='document_type_uncertain', field_path='document_type', message='Confirm the document type.', sources=classification.sources))
        for path, fact in _walk(facts):
            if fact.status == 'not_provided':
                reviews.append(ReviewItem(code='not_provided', field_path=path, message='Not provided in extractable document content; do not substitute zero.'))
            elif fact.status == 'contradicted':
                evidence = [s for alternative in fact.alternatives for s in alternative.sources]
                contradictions.append(Contradiction(field_path=path, message=fact.review_reason, sources=evidence))
                reviews.append(ReviewItem(code='contradiction', field_path=path, message=fact.review_reason, sources=evidence))
            elif fact.status == 'uncertain' or fact.confidence < self.settings.extraction_review_confidence:
                reviews.append(ReviewItem(code='uncertain_interpretation', field_path=path, message=fact.review_reason or 'Low-confidence interpretation requires source review.', sources=fact.sources))
            if isinstance(fact.value, Money) and fact.value.currency is None:
                reviews.append(ReviewItem(code='currency_not_provided', field_path=path, message='The amount has no evidenced currency; do not infer one.', sources=fact.sources))
            if path.endswith('.elevation') and fact.value is not None:
                datum = next((a.elevation_datum for a in facts.assets if a.elevation is fact), None)
                if datum is not None and datum.status == 'not_provided':
                    reviews.append(ReviewItem(code='elevation_datum_missing', field_path=path, message='Confirm whether elevation is relative to ground, river level, or a geodetic datum.', sources=fact.sources))
        for name in ('assets', 'flood_history', 'document_recommendations'):
            if not getattr(facts, name):
                reviews.append(ReviewItem(code='not_provided', field_path=f'facts.{name}', message='No source-backed records found; absence does not establish that none exist.'))
        for name in type(facts.risk_factors).model_fields:
            if not getattr(facts.risk_factors, name):
                reviews.append(ReviewItem(code='not_provided', field_path=f'facts.risk_factors.{name}', message='Not provided; do not interpret missing observations as absence of risk.'))
        categories = ['property_sum_insured', 'machinery_values', 'inventory_values', 'contents_values', 'business_interruption_limit']
        present = [name for name in categories if getattr(facts.financial_exposure, name).status != 'not_provided']
        for first, second in combinations(present, 2):
            relationships = [r for r in facts.financial_exposure.relationships if {r.category, r.other_category} == {first, second}]
            confirmed = [r for r in relationships if r.relationship != 'unknown' and r.evidence.status == 'provided' and r.evidence.confidence >= self.settings.extraction_review_confidence]
            if not confirmed:
                reviews.append(ReviewItem(code='financial_overlap_unverified', field_path=f'facts.financial_exposure.{first}', message=f'Relationship with {second} is unverified. Preserve both amounts separately; no total has been calculated.'))
            elif len({(r.relationship, r.category if r.relationship == 'included_in' else '') for r in confirmed}) > 1:
                evidence = [s for r in confirmed for s in r.evidence.sources]
                contradictions.append(Contradiction(field_path='facts.financial_exposure.relationships', message='Conflicting relationships between financial categories.', sources=evidence))
                reviews.append(ReviewItem(code='contradiction', field_path='facts.financial_exposure.relationships', message='Review competing financial relationships; do not aggregate.', sources=evidence))
        for i, recommendation in enumerate(facts.document_recommendations):
            if recommendation.author_role == 'unknown':
                reviews.append(ReviewItem(code='recommendation_author_unknown', field_path=f'facts.document_recommendations[{i}]', message='Confirm who made this documentary recommendation.', sources=recommendation.recommendation.sources))
        for warning in document.warnings:
            reviews.append(ReviewItem(code='ingestion_limitation', field_path='document', message=warning))
        return reviews, contradictions

    async def _extract_section_facts(self, payload: str) -> InsuranceFacts:
        if not getattr(self.llm, 'requires_split_insurance_schema', False):
            return InsuranceFacts.model_validate(
                await self.llm.structured(INSURANCE_EXTRACTION_SYSTEM, payload, InsuranceFacts))

        identity, financial, history = await asyncio.gather(
            self.llm.structured(INSURANCE_EXTRACTION_SYSTEM, payload, InsuranceIdentityAssets),
            self.llm.structured(INSURANCE_EXTRACTION_SYSTEM, payload, InsuranceFinancialTerms),
            self.llm.structured(INSURANCE_EXTRACTION_SYSTEM, payload, InsuranceHistoryRisk),
        )
        identity = InsuranceIdentityAssets.model_validate(identity)
        financial = InsuranceFinancialTerms.model_validate(financial)
        history = InsuranceHistoryRisk.model_validate(history)
        return InsuranceFacts(
            insured=identity.insured,
            assets=identity.assets,
            financial_exposure=financial.financial_exposure,
            insurance_terms=financial.insurance_terms,
            flood_history=history.flood_history,
            risk_factors=history.risk_factors,
            document_recommendations=history.document_recommendations,
        )

    async def extract_document(self, document: ParsedDocument) -> InsuranceDocumentResult:
        if not document.segments:
            raise ExtractionError('Document has no extractable text')
        # Uniform overview sampling keeps huge schedules within the same budget.
        # Extraction below still processes every segment, including omitted rows.
        budget = self.settings.extraction_section_characters
        count = min(len(document.segments), 64, max(1, budget // 500))
        indices = sorted({round(i * (len(document.segments) - 1) / max(1, count - 1)) for i in range(count)})
        per_segment = max(1, (budget - 200) // count - 200)
        sampled = [document.segments[i].model_copy(update={'text': document.segments[i].text[:per_segment]}) for i in indices]
        overview = ParsedDocument(document_id=document.document_id, segments=sampled)
        while len(overview.model_dump_json()) > budget:
            if len(overview.segments) > 1:
                overview.segments = overview.segments[::2]
            else:
                overview.segments[0].text = overview.segments[0].text[:len(overview.segments[0].text) // 2]
                if not overview.segments[0].text:
                    raise ExtractionError('Source metadata exceeds the classification budget')
        classification = DocumentClassification.model_validate(await self.llm.structured(INSURANCE_CLASSIFICATION_SYSTEM, overview.model_dump_json(), DocumentClassification))
        classification = self._anchor_classification(classification, overview)
        self._validate_sources(classification.sources, overview, repair_from_locator=True)
        merged = InsuranceFacts()
        for section in self._sections(document):
            payload = json.dumps({'document_type': classification.kind, 'section': section.model_dump(mode='json')}, ensure_ascii=False)
            facts = await self._extract_section_facts(payload)
            self._validate_facts(facts, section)
            merged = self._merge(merged, facts)
        # Revalidate after deterministic reconciliation, then collect review work.
        merged = InsuranceFacts.model_validate_json(merged.model_dump_json())
        reviews, contradictions = self._reviews(merged, classification, document)
        audit_payload = merged.model_dump_json()
        if len(audit_payload) <= self.settings.max_document_characters:
            audit = ContradictionAudit.model_validate(await self.llm.structured(INSURANCE_CONTRADICTION_SYSTEM, audit_payload, ContradictionAudit))
            paths = [path for path, _ in _walk(merged)]
            for contradiction in audit.contradictions:
                if contradiction.field_path == 'facts' or not any(path == contradiction.field_path or path.startswith(contradiction.field_path + '[') or path.startswith(contradiction.field_path + '.') for path in paths):
                    raise ExtractionError('Contradiction references an unknown fact field')
                self._validate_sources(contradiction.sources, document)
                if len({(s.locator, s.excerpt) for s in contradiction.sources}) < 2:
                    raise ExtractionError('Contradiction requires two distinct evidence statements')
                if not any(c.field_path == contradiction.field_path and c.message == contradiction.message for c in contradictions):
                    contradictions.append(contradiction)
                    reviews.append(ReviewItem(code='semantic_contradiction', field_path=contradiction.field_path, message=contradiction.message, sources=contradiction.sources))
        else:
            reviews.append(ReviewItem(code='semantic_audit_incomplete', field_path='document', message='Extracted facts exceed the semantic audit budget; review cross-field contradictions manually.'))
        return InsuranceDocumentResult(document_id=document.document_id, document_type=classification, facts=merged, review_status='requires_review' if reviews else 'ready', review_items=reviews, contradictions=contradictions, warnings=document.warnings)


async def aextract_insurance_document(file_path: str | Path, *, llm: LLMClient | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """Async variant for FastAPI. Returns a JSON-compatible, validated object."""
    config = settings or Settings()
    document = await asyncio.to_thread(DocumentParser(config).parse_file, file_path, str(uuid4()))
    client = llm if llm is not None else create_llm_client(config)
    try:
        result = await InsuranceDocumentExtractor(client, config).extract_document(document)
        return json.loads(result.model_dump_json())
    finally:
        if llm is None:
            close = getattr(client, 'aclose', None)
            if close:
                await close()


def extract_insurance_document(file_path: str | Path, *, llm: LLMClient | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """Synchronous reusable entry point; use aextract_insurance_document in async code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aextract_insurance_document(file_path, llm=llm, settings=settings))
    raise RuntimeError('Use await aextract_insurance_document(...) inside an event loop')
