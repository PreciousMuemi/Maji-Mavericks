import pytest
from ai.extraction_service import ExtractionError, ExtractionService
from ai.llm_client import UnconfiguredClient, LLMUnavailable
from ai.schemas import Asset, DocumentExtraction, DocumentSegment, ParsedDocument, SourceReference


class StubLLM:
    def __init__(self, output):
        self.output = output

    async def structured(self, system, user, schema):
        assert 'untrusted' in system
        return schema.model_validate(self.output)


@pytest.mark.asyncio
async def test_extraction_preserves_provenance_and_warnings():
    source = SourceReference(document_id='doc', locator='page:1', excerpt='Warehouse')
    output = DocumentExtraction(assets=[Asset(asset_id='a', name='Warehouse', sources=[source])])
    document = ParsedDocument(document_id='doc', segments=[DocumentSegment(locator='page:1', text='Warehouse KES 100')], warnings=['Some pages require OCR'])
    result = await ExtractionService(StubLLM(output)).extract(document)
    assert result.assets[0].insured_value is None
    assert result.warnings == document.warnings


@pytest.mark.asyncio
@pytest.mark.parametrize('sources', [[], [SourceReference(document_id='wrong', locator='page:1', excerpt='Warehouse')], [SourceReference(document_id='doc', locator='page:2', excerpt='Warehouse')], [SourceReference(document_id='doc', locator='page:1', excerpt='invented')]])
async def test_rejects_missing_or_fabricated_provenance(sources):
    document = ParsedDocument(document_id='doc', segments=[DocumentSegment(locator='page:1', text='Warehouse')])
    output = DocumentExtraction(assets=[Asset(asset_id='a', name='Warehouse', sources=sources)])
    with pytest.raises(ExtractionError):
        await ExtractionService(StubLLM(output)).extract(document)


@pytest.mark.asyncio
async def test_empty_document_and_unconfigured_llm():
    with pytest.raises(ExtractionError):
        await ExtractionService(UnconfiguredClient()).extract(ParsedDocument(document_id='doc', segments=[]))
    with pytest.raises(LLMUnavailable):
        await UnconfiguredClient().structured('', '', DocumentExtraction)
