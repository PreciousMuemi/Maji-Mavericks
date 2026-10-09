"""Synthetic fixtures exercise extraction contracts; they are not the user's offer."""
import json
from decimal import Decimal

import pymupdf as fitz
import pytest
from pydantic import ValidationError

from ai.config import Settings
from ai.document_parser import DocumentError, DocumentParser
from ai.extraction_service import ExtractionError, extract_insurance_document
from ai.insurance_extraction import InsuranceDocumentExtractor, aextract_insurance_document
from ai.insurance_schemas import (
    ContradictionAudit, DocumentClassification, Fact, InsuranceDocumentResult, InsuranceFacts, Money,
)
from ai.schemas import DocumentSegment, ParsedDocument, SourceReference


SYNTHETIC_PAGES = [
    'Insurance offer\nClient: Synthetic Test Mill\nSite: Example County\nIndustry: Grain processing\nCoordinates: 0.5, 34.2\nProperty sum insured: KES 1 million\nMachinery: KES 400,000\nInventory: KES 0\nBuilding: Warehouse A\nConstruction: masonry\nFloor area: 120 m2\nCondition: fair\nElevation: 2 m\nBroker recommends improving drainage.\nDrainage: blocked channels.\nDeductible: KES 0',
    'Revised schedule\nProperty sum insured: KES 1,200,000\nBuilding: Warehouse A\nCondition: deteriorated\nFlood event: May 2022\nFlood depth: 0.5 to 1 m\nClaimed: KES 60,000\nSettled: KES 40,000\nInterruption: 10 days\nPrevious damage: flooded switchgear.\nCritical vulnerability: unprotected switchgear.',
]


def pdf_bytes(pages):
    with fitz.open() as document:
        for text in pages:
            page = document.new_page()
            page.insert_text((50, 50), text, fontsize=10)
        return document.tobytes()


def fact(value, segment, document_id, excerpt, status='provided', confidence=0.99, review_reason=None):
    return {'value': value, 'status': status, 'confidence': confidence, 'sources': [{'document_id': document_id, 'locator': segment['locator'], 'page_number': segment['page_number'], 'excerpt': excerpt}], 'review_reason': review_reason}


class FixtureLLM:
    """Injected structured responses derived from a known synthetic fixture only."""
    async def structured(self, system, user, schema):
        if schema is ContradictionAudit:
            return schema()
        if schema is DocumentClassification:
            overview = json.loads(user)
            first = overview['segments'][0]
            return schema(kind='insurance_offer', confidence=0.99, reason='Explicit offer heading.', sources=[{'document_id': overview['document_id'], 'locator': first['locator'], 'excerpt': 'Insurance offer'}])
        section = json.loads(user)['section']
        document_id = section['document_id']
        result = {'insured': {}, 'financial_exposure': {}, 'insurance_terms': {}, 'assets': [], 'flood_history': [], 'risk_factors': {}, 'document_recommendations': []}
        for segment in section['segments']:
            text = segment['text']
            def f(value, excerpt, **kwargs):
                return fact(value, segment, document_id, excerpt, **kwargs)
            if 'Client: Synthetic Test Mill' in text:
                result['insured'] = {
                    'client_name': f('Synthetic Test Mill', 'Client: Synthetic Test Mill'),
                    'facility_location': f('Example County', 'Site: Example County'),
                    'industry': f('Grain processing', 'Industry: Grain processing'),
                    'coordinates': f({'latitude': 0.5, 'longitude': 34.2}, 'Coordinates: 0.5, 34.2'),
                }
                result['financial_exposure'] = {
                    'property_sum_insured': f({'amount': '1000000', 'currency': 'KES'}, 'Property sum insured: KES 1 million'),
                    'machinery_values': f({'amount': '400000', 'currency': 'KES'}, 'Machinery: KES 400,000'),
                    'inventory_values': f({'amount': '0', 'currency': 'KES'}, 'Inventory: KES 0'),
                    'currency': f('KES', 'Property sum insured: KES 1 million'),
                }
                result['insurance_terms'] = {'deductibles': f('KES 0', 'Deductible: KES 0')}
                result['assets'].append({'name': f('Warehouse A', 'Building: Warehouse A'), 'construction_type': f('masonry', 'Construction: masonry'), 'floor_area': f({'value': '120', 'unit': 'm2'}, 'Floor area: 120 m2'), 'condition': f('fair', 'Condition: fair'), 'elevation': f({'value': '2', 'unit': 'm'}, 'Elevation: 2 m')})
                result['document_recommendations'] = [{'author_role': 'broker', 'recommendation': f('Improve drainage', 'Broker recommends improving drainage.')}]
                result['risk_factors']['drainage_weaknesses'] = [f('Blocked channels', 'Drainage: blocked channels.')]
            if 'Revised schedule' in text:
                # Inside one section preserve both assertions rather than choose one.
                updated = f({'amount': '1200000', 'currency': 'KES'}, 'Property sum insured: KES 1,200,000')
                if 'property_sum_insured' in result['financial_exposure']:
                    old = result['financial_exposure']['property_sum_insured']
                    updated = {'status': 'contradicted', 'review_reason': 'The offer and revised schedule state different property sums.', 'alternatives': [{'value': old['value'], 'sources': old['sources']}, {'value': updated['value'], 'sources': updated['sources']}]}
                result['financial_exposure']['property_sum_insured'] = updated
                result['assets'].append({'name': f('Warehouse A', 'Building: Warehouse A'), 'condition': f('deteriorated', 'Condition: deteriorated')})
                result['flood_history'].append({'event_date': f('May 2022', 'Flood event: May 2022'), 'reported_depth': f({'lower': '0.5', 'upper': '1', 'unit': 'm'}, 'Flood depth: 0.5 to 1 m'), 'historical_claim_value': f({'amount': '60000', 'currency': 'KES'}, 'Claimed: KES 60,000'), 'settlement_value': f({'amount': '40000', 'currency': 'KES'}, 'Settled: KES 40,000'), 'business_interruption_duration': f({'value': '10', 'unit': 'days'}, 'Interruption: 10 days')})
                result['risk_factors']['previous_damage'] = [f('Flooded switchgear', 'Previous damage: flooded switchgear.')]
                result['risk_factors']['operational_vulnerabilities'] = [f('Unprotected switchgear', 'Critical vulnerability: unprotected switchgear.')]
        return schema.model_validate(result)


@pytest.fixture
def synthetic_pdf(tmp_path):
    path = tmp_path / 'synthetic_offer.pdf'
    path.write_bytes(pdf_bytes(SYNTHETIC_PAGES))
    return path


def test_reusable_service_returns_valid_json_and_all_categories(synthetic_pdf):
    result = extract_insurance_document(synthetic_pdf, llm=FixtureLLM(), settings=Settings(_env_file=None, extraction_section_characters=1000))
    validated = InsuranceDocumentResult.model_validate_json(json.dumps(result, allow_nan=False))
    assert validated.facts.insured.client_name.value == 'Synthetic Test Mill'
    assert validated.facts.insured.coordinates.value.longitude == 34.2
    assert validated.document_type.sources[0].page_number == 1
    assert validated.facts.flood_history[0].reported_depth.value.upper == Decimal('1')
    assert validated.facts.flood_history[0].historical_claim_value.value.amount == Decimal('60000')
    assert validated.facts.flood_history[0].settlement_value.value.amount == Decimal('40000')
    assert validated.facts.flood_history[0].business_interruption_duration.value.value == Decimal('10')
    assert validated.facts.document_recommendations[0].author_role == 'broker'
    assert validated.facts.risk_factors.operational_vulnerabilities
    assert 'ai_recommendations' not in result['facts']


def test_source_quotes_are_aligned_only_when_difference_is_whitespace():
    extractor = InsuranceDocumentExtractor(FixtureLLM(), Settings(_env_file=None))
    document = ParsedDocument(document_id='doc', segments=[DocumentSegment(
        locator='page:1:block:1', page_number=1,
        text='Property sum insured:\nKES 1 million')])
    source = SourceReference(document_id='doc', locator='page:1:block:1', page_number=1,
                             excerpt='Property sum insured: KES 1 million')
    extractor._validate_sources([source], document)
    assert source.excerpt == 'Property sum insured:\nKES 1 million'

    fabricated = SourceReference(document_id='doc', locator='page:1:block:1', page_number=1,
                                  excerpt='Property sum insured: KES 2 million')
    with pytest.raises(ExtractionError):
        extractor._validate_sources([fabricated], document)


def test_zero_missing_values_conflicts_and_overlap_are_preserved(synthetic_pdf):
    result = extract_insurance_document(synthetic_pdf, llm=FixtureLLM(), settings=Settings(_env_file=None, extraction_section_characters=1000))
    facts = result['facts']
    assert facts['financial_exposure']['inventory_values']['value']['amount'] == '0'
    assert facts['financial_exposure']['inventory_values']['status'] == 'provided'
    assert facts['financial_exposure']['business_interruption_limit']['value'] is None
    assert facts['financial_exposure']['business_interruption_limit']['status'] == 'not_provided'
    assert len(facts['assets']) == 1
    assert facts['assets'][0]['stated_value']['value'] is None
    assert facts['assets'][0]['condition']['status'] == 'contradicted'
    prop = facts['financial_exposure']['property_sum_insured']
    assert prop['value'] is None
    assert {a['value']['amount'] for a in prop['alternatives']} == {'1000000', '1200000'}
    assert {s['page_number'] for a in prop['alternatives'] for s in a['sources']} == {1, 2}
    assert {c['field_path'] for c in result['contradictions']} == {'facts.assets[0].condition', 'facts.financial_exposure.property_sum_insured'}
    assert 'total' not in facts['financial_exposure']
    codes = {r['code'] for r in result['review_items']}
    assert {'not_provided', 'contradiction', 'financial_overlap_unverified', 'elevation_datum_missing'} <= codes
    assert result['review_status'] == 'requires_review'


@pytest.mark.parametrize('output', [
    {'insured': {'client_name': {'value': 'Invented', 'status': 'provided', 'confidence': 1, 'sources': []}}},
    {'financial_exposure': {'total': 999}},
    {'insured': {'client_name': {'value': 'Wrong', 'status': 'not_provided'}}},
    {'insured': {'coordinates': {'value': {'latitude': 99, 'longitude': 34}, 'status': 'provided', 'confidence': 1, 'sources': []}}},
])
def test_invalid_schema_output(output):
    with pytest.raises(ValidationError):
        InsuranceFacts.model_validate(output)


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['excerpt', 'locator', 'document_id', 'page_number', 'amount', 'currency', 'unit'])
async def test_fabricated_evidence_numbers_and_units_rejected(change):
    document = ParsedDocument(document_id='doc', segments=[DocumentSegment(locator='page:1', page_number=1, text='Insurance offer\nProperty: KES 100\nDepth: 2 m')])
    class BadLLM:
        async def structured(self, system, user, schema):
            if schema is DocumentClassification:
                return schema(kind='insurance_offer', confidence=1, reason='Offer', sources=[{'document_id': 'doc', 'locator': 'page:1', 'excerpt': 'Insurance offer'}])
            source = {'document_id': 'doc', 'locator': 'page:1', 'page_number': 1, 'excerpt': 'Property: KES 100'}
            if change in {'excerpt', 'locator', 'document_id', 'page_number'}:
                source[change] = 2 if change == 'page_number' else 'invented'
            value = {'amount': '999' if change == 'amount' else '100', 'currency': 'USD' if change == 'currency' else 'KES'}
            output = {'financial_exposure': {'property_sum_insured': {'value': value, 'status': 'provided', 'confidence': 1, 'sources': [source]}}}
            if change == 'unit':
                output = {'assets': [{'name': {'value': 'A', 'status': 'provided', 'confidence': 1, 'sources': [source]}, 'elevation': {'value': {'value': '2', 'unit': 'ft'}, 'status': 'provided', 'confidence': 1, 'sources': [{'document_id': 'doc', 'locator': 'page:1', 'excerpt': 'Depth: 2 m'}]}}]}
            return schema.model_validate(output)
    with pytest.raises(ExtractionError):
        await InsuranceDocumentExtractor(BadLLM(), Settings(_env_file=None)).extract_document(document)


@pytest.mark.parametrize('filename,content', [('broken.pdf', b'%PDF-invalid'), ('renamed.pdf', b'client,value\nA,100'), ('broken.xlsx', b'not zip'), ('broken.docx', b'not zip'), ('binary.csv', b'\x00\x01'), ('unexpected.exe', b'x')])
def test_malformed_files_rejected_before_llm(tmp_path, filename, content):
    path = tmp_path / filename
    path.write_bytes(content)
    class NeverCalled:
        async def structured(self, *args):
            raise AssertionError('Invalid upload reached LLM')
    with pytest.raises(DocumentError):
        extract_insurance_document(path, llm=NeverCalled(), settings=Settings(_env_file=None))


def test_oversized_and_missing_paths(tmp_path):
    path = tmp_path / 'large.csv'
    path.write_bytes(b'a,b\nx,y\n')
    with pytest.raises(DocumentError):
        extract_insurance_document(path, settings=Settings(_env_file=None, max_upload_bytes=3))
    with pytest.raises(DocumentError):
        extract_insurance_document(tmp_path / 'missing.pdf', settings=Settings(_env_file=None))


@pytest.mark.asyncio
async def test_async_entry_point_and_sync_misuse(synthetic_pdf):
    with pytest.raises(RuntimeError, match='aextract_insurance_document'):
        extract_insurance_document(synthetic_pdf)
    result = await aextract_insurance_document(synthetic_pdf, llm=FixtureLLM(), settings=Settings(_env_file=None))
    assert result['document_type']['kind'] == 'insurance_offer'


def test_uncertain_missing_currency_and_unknown_attribution_require_review():
    extractor = InsuranceDocumentExtractor(FixtureLLM(), Settings(_env_file=None))
    segment = {'locator': 'page:1', 'page_number': 1}
    facts = InsuranceFacts.model_validate({'financial_exposure': {'machinery_values': fact({'amount': '100', 'currency': None}, segment, 'doc', 'Machinery 100')}, 'insured': {'industry': fact('Milling', segment, 'doc', 'Industry possibly milling', status='uncertain', confidence=0.5, review_reason='Tentative wording')}, 'document_recommendations': [{'author_role': 'unknown', 'recommendation': fact('Improve defenses', segment, 'doc', 'Recommend improve defenses')}]})
    classification = DocumentClassification(kind='unknown', confidence=0.2, reason='Unclear document')
    reviews, _ = extractor._reviews(facts, classification, ParsedDocument(document_id='doc', segments=[]))
    assert {'document_type_uncertain', 'uncertain_interpretation', 'currency_not_provided', 'recommendation_author_unknown'} <= {r.code for r in reviews}


@pytest.mark.asyncio
async def test_cross_chunk_reconciliation_retains_original_page_evidence():
    pages = [SYNTHETIC_PAGES[0] + '\n' + ('Administrative narrative. ' * 100), SYNTHETIC_PAGES[1]]
    document = ParsedDocument(document_id='doc', segments=[DocumentSegment(locator=f'page:{i + 1}', page_number=i + 1, text=text) for i, text in enumerate(pages)])
    extractor = InsuranceDocumentExtractor(FixtureLLM(), Settings(_env_file=None, extraction_section_characters=1000))
    sections = list(extractor._sections(document))
    assert len(sections) > 2
    assert all(sum(len(s.text) for s in section.segments) <= 1000 for section in sections)
    result = await extractor.extract_document(document)
    assert len(result.facts.assets) == 1
    assert result.facts.assets[0].condition.status == 'contradicted'
    assert {source.page_number for alt in result.facts.financial_exposure.property_sum_insured.alternatives for source in alt.sources} == {1, 2}


@pytest.mark.asyncio
async def test_overall_property_value_cannot_be_assigned_to_building():
    document = ParsedDocument(document_id='doc', segments=[DocumentSegment(locator='page:1', page_number=1, text='Insurance offer\nBuilding Warehouse A\nProperty sum insured: KES 100')])
    class WrongAllocation:
        async def structured(self, system, user, schema):
            source = {'document_id': 'doc', 'locator': 'page:1', 'excerpt': 'Insurance offer'}
            if schema is DocumentClassification:
                return schema(kind='insurance_offer', confidence=1, reason='Offer', sources=[source])
            return schema.model_validate({'assets': [{'name': fact('Warehouse A', {'locator': 'page:1', 'page_number': 1}, 'doc', 'Building Warehouse A'), 'stated_value': fact({'amount': '100', 'currency': 'KES'}, {'locator': 'page:1', 'page_number': 1}, 'doc', 'Building Warehouse A\nProperty sum insured: KES 100')}]})
    with pytest.raises(ExtractionError, match='asset-specific'):
        await InsuranceDocumentExtractor(WrongAllocation(), Settings(_env_file=None)).extract_document(document)


def test_verified_financial_relationship_suppresses_only_its_overlap_review():
    segment = {'locator': 'page:1', 'page_number': 1}
    data = {'financial_exposure': {'property_sum_insured': fact({'amount': '100', 'currency': 'KES'}, segment, 'doc', 'Property KES 100'), 'machinery_values': fact({'amount': '30', 'currency': 'KES'}, segment, 'doc', 'Machinery KES 30'), 'relationships': [{'category': 'machinery_values', 'other_category': 'property_sum_insured', 'relationship': 'included_in', 'evidence': fact('Machinery is included in property sum insured', segment, 'doc', 'Machinery is included in property sum insured')}]}}
    facts = InsuranceFacts.model_validate(data)
    extractor = InsuranceDocumentExtractor(FixtureLLM(), Settings(_env_file=None))
    reviews, _ = extractor._reviews(facts, DocumentClassification(kind='unknown', confidence=0, reason='Unknown'), ParsedDocument(document_id='doc', segments=[]))
    assert not any(r.code == 'financial_overlap_unverified' for r in reviews)
    assert not hasattr(facts.financial_exposure, 'total')


@pytest.mark.asyncio
async def test_semantic_contradiction_audit_retains_two_distinct_statements():
    text = 'Insurance offer\nDrainage: regularly maintained.\nDrainage: channels blocked.'
    document = ParsedDocument(document_id='doc', segments=[DocumentSegment(locator='page:1', page_number=1, text=text)])
    class SemanticLLM:
        async def structured(self, system, user, schema):
            segment = {'locator': 'page:1', 'page_number': 1}
            source = lambda excerpt: {'document_id': 'doc', 'locator': 'page:1', 'excerpt': excerpt}
            if schema is DocumentClassification:
                return schema(kind='insurance_offer', confidence=1, reason='Offer', sources=[source('Insurance offer')])
            if schema is ContradictionAudit:
                return schema(contradictions=[{'field_path': 'facts.risk_factors.drainage_weaknesses', 'message': 'Maintenance and blocked-channel statements conflict unless dates or scopes differ.', 'sources': [source('Drainage: regularly maintained.'), source('Drainage: channels blocked.')]}])
            return schema.model_validate({'risk_factors': {'drainage_weaknesses': [fact('Maintained drainage', segment, 'doc', 'Drainage: regularly maintained.'), fact('Blocked channels', segment, 'doc', 'Drainage: channels blocked.')]}})
    result = await InsuranceDocumentExtractor(SemanticLLM(), Settings(_env_file=None)).extract_document(document)
    assert len(result.contradictions) == 1
    assert all(s.page_number == 1 for s in result.contradictions[0].sources)
    assert any(r.code == 'semantic_contradiction' for r in result.review_items)


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_path,bad_excerpt', [('facts.nonexistent', None), ('facts.insured.client_name', 'fabricated source')])
async def test_semantic_audit_rejects_unknown_fields_and_fabricated_evidence(synthetic_pdf, bad_path, bad_excerpt):
    class InvalidAudit(FixtureLLM):
        async def structured(self, system, user, schema):
            if schema is ContradictionAudit:
                payload = json.loads(user)
                source = payload['insured']['client_name']['sources'][0]
                other = dict(source)
                other['excerpt'] = bad_excerpt or 'Client: Synthetic Test Mill'
                return schema(contradictions=[{'field_path': bad_path, 'message': 'Invalid audit', 'sources': [source, other]}])
            return await super().structured(system, user, schema)
    with pytest.raises(ExtractionError):
        await aextract_insurance_document(synthetic_pdf, llm=InvalidAudit(), settings=Settings(_env_file=None))


def test_cli_stdout_is_strict_json_on_failure(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-m', 'ai.insurance_cli', str(tmp_path / 'missing.pdf')], capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)['status'] == 'error'
    assert 'Traceback' not in result.stderr


def test_uninterpretable_present_field_is_distinct_from_absence_and_zero():
    source = {'document_id': 'doc', 'locator': 'page:1', 'excerpt': 'Property sum insured: to be confirmed'}
    uncertain = Fact[Money](status='uncertain', value=None, confidence=0.2, sources=[source], review_reason='Value is explicitly pending confirmation')
    assert uncertain.status != 'not_provided'
    assert uncertain.sources
    with pytest.raises(ValidationError):
        Fact[Money](status='provided', value=None, confidence=1, sources=[source])
    extractor = InsuranceDocumentExtractor(FixtureLLM(), Settings(_env_file=None))
    known = Fact[Money](status='provided', value=Money(amount='0', currency='KES'), confidence=0.99, sources=[{'document_id': 'doc', 'locator': 'page:2', 'excerpt': 'Property sum insured: KES 0'}])
    merged = extractor._merge_fact(uncertain, known)
    assert merged.value.amount == 0
    assert merged.status == 'uncertain'
    assert len(merged.sources) == 2
    assert merged.alternatives == []


@pytest.mark.asyncio
async def test_llm_payload_budgets_with_many_escaped_segments():
    calls = []
    class EmptyLLM:
        async def structured(self, system, user, schema):
            calls.append((schema, len(user)))
            if schema is DocumentClassification:
                return schema(kind='unknown', confidence=0.1, reason='No insurance context')
            return schema()
    document = ParsedDocument(document_id='doc', segments=[DocumentSegment(locator=f'sheet:Schedule:row:{i + 1}', kind='table', text='{"note":"' + ('quoted \\\" text ' * 8) + '"}') for i in range(100)])
    extractor = InsuranceDocumentExtractor(EmptyLLM(), Settings(_env_file=None, extraction_section_characters=1000))
    sections = list(extractor._sections(document))
    assert ''.join(s.text for section in sections for s in section.segments) == ''.join(s.text for s in document.segments)
    assert all(len(section.model_dump_json()) <= 1000 for section in sections)
    assert all(len({s.locator for s in section.segments}) == len(section.segments) for section in sections)
    await extractor.extract_document(document)
    assert all(length <= 1000 for schema, length in calls if schema is not ContradictionAudit)


def test_numeric_unit_without_space_is_grounded_without_accepting_wrong_dimension():
    from ai.insurance_extraction import _validate_numeric
    from ai.insurance_schemas import Measurement
    from ai.schemas import SourceReference
    source = SourceReference(document_id='doc', locator='page:1', excerpt='Flood depth 0.5m')
    _validate_numeric(Measurement(value='0.5', unit='m'), [source])
    with pytest.raises(ExtractionError):
        _validate_numeric(Measurement(value='120', unit='m'), [SourceReference(document_id='doc', locator='page:1', excerpt='Floor area 120m2')])
