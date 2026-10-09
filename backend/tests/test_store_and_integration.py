import pytest
from ai.risk_analyzer import IntegrationUnavailable, validate_model_output
from ai.schemas import AIAnalysis, DocumentExtraction, ParsedDocument
from ai.store import AssessmentStore


def test_store_survives_reopen_and_invalidates_analysis(tmp_path):
    store = AssessmentStore(str(tmp_path))
    assessment_id = store.create()
    store.add_document(assessment_id, ParsedDocument(document_id='doc', segments=[]))
    store.save_extractions(assessment_id, {'doc': DocumentExtraction()})
    store.save_analysis(assessment_id, AIAnalysis(summary='Validation only'))
    store.close()
    store = AssessmentStore(str(tmp_path))
    try:
        assert 'doc' in store.documents(assessment_id)
        assert store.analysis(assessment_id).summary == 'Validation only'
        store.save_extractions(assessment_id, {'doc': DocumentExtraction()})
        assert store.analysis(assessment_id) is None
    finally:
        store.close()


@pytest.mark.parametrize('output', [[], {'loss': float('nan')}, {'value': object()}])
def test_invalid_model_outputs_are_rejected(output):
    with pytest.raises(IntegrationUnavailable):
        validate_model_output(output)


def test_logs_do_not_echo_arbitrary_assessment_ids(caplog):
    from ai.logging import log_event
    caplog.set_level('INFO', logger='nzoia.ai')
    log_event('request_failed', assessment_id='CONFIDENTIAL-CUSTOMER', code='assessment_not_found')
    assert 'CONFIDENTIAL-CUSTOMER' not in caplog.text
    assert '"assessment_id":null' in caplog.text
