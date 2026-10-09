"""Evidence-grounded underwriting intelligence with deterministic numerical authority."""
import asyncio
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal

from pydantic import ValidationError

from .authorization import AssessmentAuthorizer
from .config import Settings
from .insurance_extraction import _walk, _validate_numeric
from .insurance_schemas import Money
from .intelligence_schemas import (
    AnalysisLimitation, AnalysisNarrative, Citation, IntelligenceFinding, NarrativeSelection,
    NumericFinding, RecommendedAction, RiskDriver, RiskLevel, UnderwritingAnalysis,
)
from .llm_client import LLMUnavailable, create_llm_client
from .model_backend import UnavailableUnderwritingBackend
from .prompts import UNDERWRITING_ANALYSIS_SYSTEM
from .risk_analyzer import IntegrationUnavailable
from .risk_calculations import event_count, historical_totals, largest_historical_amount, model_concentrations, repeated_description_count
from .store import AssessmentStore
from .underwriting_schemas import ModelOutput
from .validator import validate_extraction


class UnderwritingAnalysisService:
    def __init__(self, store, llm, *, model_backend=None, settings=None):
        self.store, self.llm = store, llm
        self.backend = model_backend if model_backend is not None else UnavailableUnderwritingBackend()
        self.settings = settings or Settings()

    async def generate(self, assessment_id, *, principal=None, authorizer: AssessmentAuthorizer | None = None) -> UnderwritingAnalysis:
        if authorizer is not None:
            await authorizer.require_access(principal, assessment_id)
        documents = await asyncio.to_thread(self.store.insurance_results, assessment_id)
        extraction = await asyncio.to_thread(self.store.extraction, assessment_id)
        parsed = await asyncio.to_thread(self.store.documents, assessment_id)
        observations, history, model_findings, citations, limitations, proposals = [], [], [], {}, [], []
        records, descriptions, event_keys, event_audit = [], [], [], []
        validation_id = f'validation.{assessment_id}'
        citations[validation_id] = Citation(id=validation_id, kind='validation', assessment_id=assessment_id, function='ai.validator.validate_extraction', inputs=[{'extraction': extraction.model_dump(mode='json'), 'insurance_documents': [d.model_dump(mode='json') for d in documents]}])

        def cite(sources):
            ids = []
            for source in sources:
                document = parsed.get(source.document_id)
                segment = next((s for s in document.segments if s.locator == source.locator), None) if document else None
                if segment is None or not source.excerpt.strip() or source.excerpt not in segment.text or (source.page_number is not None and source.page_number != segment.page_number):
                    continue
                source = source.model_copy(update={'page_number': segment.page_number})
                identifier = 'doc.' + hashlib.sha256(source.model_dump_json().encode()).hexdigest()[:16]
                citations[identifier] = Citation(id=identifier, kind='document', assessment_id=assessment_id, source=source)
                ids.append(identifier)
            return list(dict.fromkeys(ids))

        def observation(identifier, title, statement, kind, source_ids, numbers=None):
            if not source_ids:
                return None
            item = IntelligenceFinding(id=identifier, title=title, statement=statement, evidence_type=kind, citation_ids=source_ids, numerical_findings=numbers or [])
            observations.append(item)
            return item

        def limitation(code, message, ids=None):
            limitations.append(AnalysisLimitation(code=code, message=message, citation_ids=ids or []))

        for doc_index, document in enumerate(documents):
            facts = document.facts
            for path, fact in _walk(facts):
                if fact.status != 'provided' or fact.confidence < self.settings.extraction_review_confidence:
                    continue
                ids = cite(fact.sources)
                if not ids:
                    limitation('document_evidence_unverified', f'{path}: source evidence does not match the stored document and is excluded.')
                    continue
                if ids:
                    try:
                        _validate_numeric(fact.value, [citations[i].source for i in ids])
                    except ValueError:
                        limitation('numerical_evidence_unverified', f'{path}: a numerical field does not match its cited evidence and is excluded.', ids)
                        continue
                identifier = f'document.{doc_index}.{path}'
                value = fact.model_dump(mode='json')['value']
                if path.startswith('facts.risk_factors.') or re.search(r'\.assets\[\d+\]\.(condition|construction_type|elevation|elevation_datum)$', path):
                    asset = next((a.name.value for index, a in enumerate(facts.assets) if path.startswith(f'facts.assets[{index}]')), None)
                    title = f'{path.rsplit(".", 1)[-1].replace("_", " ")}' + (f': {asset}' if asset else '')
                    observation(identifier, title, f'Document-reported {title}: {json.dumps(value, ensure_ascii=False)}', 'document_reported', ids)
                elif path.startswith('facts.financial_exposure.') and hasattr(fact.value, 'amount'):
                    money = fact.value
                    if money.currency is None:
                        limitation('currency_unknown', 'A document-reported amount lacks an evidenced currency; it is not aggregated.', ids)
                        continue
                    metric = path.rsplit('.', 1)[-1]
                    item = observation(identifier, metric.replace('_', ' '), f'Document-reported {metric.replace("_", " ")}: {money.amount} {money.currency}; no financial categories have been added together.', 'document_reported', ids, [NumericFinding(metric=metric, value=money.amount, unit=money.currency, citation_id=ids[0])] if ids else [])
                elif path.startswith('facts.insurance_terms.'):
                    observation(identifier, 'Offer term: ' + path.rsplit('.', 1)[-1].replace('_', ' '), f'Document-reported offer term: {json.dumps(value, ensure_ascii=False)}; verify whether this is proposed or agreed.', 'document_reported', ids)
            for index, event in enumerate(facts.flood_history):
                date_fact = event.event_date
                date_ids = cite(date_fact.sources) if date_fact.status == 'provided' else []
                documented_date = date_fact.value and any(date_fact.value.casefold() in s.excerpt.casefold() for s in date_fact.sources)
                key = date_fact.value.strip().casefold() if date_ids and documented_date and date_fact.confidence >= self.settings.extraction_review_confidence else None
                event_keys.append(key)
                event_audit.append({'event_key': key, 'citation_ids': date_ids})
                for kind, amount_fact in [('claimed', event.historical_claim_value), ('settled', event.settlement_value)]:
                    if amount_fact.status != 'provided' or amount_fact.confidence < self.settings.extraction_review_confidence:
                        continue
                    ids = cite(amount_fact.sources)
                    if not ids:
                        limitation('historical_evidence_unverified', 'A historical amount lacks matching document evidence and is excluded from calculations.')
                        continue
                    money = amount_fact.value
                    try:
                        _validate_numeric(money, [citations[i].source for i in ids])
                    except ValueError:
                        limitation('numerical_evidence_unverified', 'A historical amount or currency does not match its cited evidence and is excluded.', ids)
                        continue
                    records.append({'stream': 'document_history', 'kind': kind, 'event_key': key, 'amount': money.amount, 'currency': money.currency, 'citation_ids': list(dict.fromkeys([*ids, *date_ids]))})
                    identifier = f'historical.{doc_index}.{index}.{kind}'
                    if money.currency is None:
                        limitation('historical_currency_unknown', 'A historical amount has no verified currency; a total is withheld.', ids)
                    finding = observation(identifier, f'Historical {kind} amount', f'Documented historical {kind} amount for {date_fact.value if key else "an undated or unverified event"}: {money.amount} {money.currency or "currency not provided"}; not a future prediction.', 'historical', ids, [NumericFinding(metric=f'historical_{kind}_amount', value=money.amount, unit=money.currency or 'unknown_currency', citation_id=ids[0])])
                    if finding:
                        history.append(finding)
                for field in ('reported_depth', 'business_interruption_duration', 'description'):
                    fact = getattr(event, field)
                    if fact.status == 'provided' and fact.confidence >= self.settings.extraction_review_confidence:
                        ids = cite(fact.sources)
                        if ids:
                            try:
                                _validate_numeric(fact.value, [citations[i].source for i in ids])
                            except ValueError:
                                limitation('numerical_evidence_unverified', 'A historical measurement does not match its cited evidence and is excluded.', ids)
                                continue
                        finding = observation(f'historical.{doc_index}.{index}.{field}', 'Observed historical ' + field.replace('_', ' '), f'Documented past event {field.replace("_", " ")}: {json.dumps(fact.model_dump(mode="json")["value"], ensure_ascii=False)}; no future severity is inferred.', 'historical', ids)
                        if finding:
                            history.append(finding)
                        if field == 'description' and key is not None:
                            descriptions.append((key, fact.value, ids))
            for recommendation in facts.document_recommendations:
                ids = cite(recommendation.recommendation.sources)
                if ids and recommendation.recommendation.value:
                    origin = {'broker': 'broker_proposal', 'insurer': 'insurer_proposal', 'surveyor': 'surveyor_proposal', 'insured': 'insured_proposal'}.get(recommendation.author_role, 'unknown_document_author')
                    proposals.append(RecommendedAction(action=recommendation.recommendation.value, rationale='This is a documentary proposal/condition; it is not an independent AI recommendation or confirmation of agreed cover.', origin=origin, citation_ids=ids))
            for item in document.review_items:
                ids = cite(item.sources) or [validation_id]
                limitation(item.code, f'{item.field_path}: {item.message}', ids)
            for contradiction in document.contradictions:
                limitation('document_contradiction', contradiction.message, cite(contradiction.sources))
            if document.document_type.kind in {'insurance_offer', 'reinsurance_offer'}:
                limitation('coverage_ambiguity', 'The submitted offer contains proposed terms; acceptance, insured scope, limits and reinsurance attachment require verification.', cite(document.document_type.sources) or [validation_id])

        for claim in extraction.claims:
            if claim.paid_amount is None:
                continue
            cause = (claim.cause or '').casefold()
            if not re.search(r'\bflood(?:ing)?\b|\binundation\b', cause) or re.search(r'non[- ]?flood|not.*flood', cause):
                if not cause:
                    limitation('historical_cause_unknown', 'A claims-register record has no verified flood cause and is excluded from flood statistics.')
                continue
            ids = cite(claim.sources)
            if not ids:
                limitation('historical_evidence_unverified', 'A claims-register amount lacks matching source evidence and is excluded.')
                continue
            try:
                _validate_numeric(Money(amount=claim.paid_amount.amount, currency=claim.paid_amount.currency), [citations[i].source for i in ids])
            except ValueError:
                limitation('numerical_evidence_unverified', 'A registered claim amount does not match its cited evidence and is excluded.', ids)
                continue
            # Claim IDs must be documented, not just labels generated during extraction.
            documented_id = any(claim.claim_id in s.excerpt for s in claim.sources)
            key = claim.claim_id if documented_id else str(claim.event_date) if claim.event_date else None
            records.append({'stream': 'claims_register', 'kind': 'paid', 'event_key': key, 'amount': claim.paid_amount.amount, 'currency': claim.paid_amount.currency, 'citation_ids': ids})
            finding = observation('claim.' + claim.claim_id, 'Historical paid claim', f'Documented historical paid claim {claim.claim_id}: {claim.paid_amount.amount} {claim.paid_amount.currency}; past payments are not forecasts.', 'historical', ids, [NumericFinding(metric='historical_paid_amount', value=claim.paid_amount.amount, unit=claim.paid_amount.currency, citation_id=ids[0])])
            if finding:
                history.append(finding)

        totals, audits, notes = historical_totals(assessment_id, records)
        for item in audits:
            citations[item.id] = item
        history.extend(totals)
        observations.extend(totals)
        for note in notes:
            limitation('historical_identity_ambiguous', note)
        streams = {row['stream'] for row in records}
        if len(streams) > 1:
            limitation('historical_stream_overlap', 'Document events and the claims register may describe the same losses; their totals are kept separate.')
        # Event count is auditable only when every documented event has a unique,
        # evidenced date identity. Unknown/repeated dates do not become distinct floods.
        counted_events = event_count(event_keys)
        if counted_events is not None:
            identifier = 'calc.historical.event_count'
            citations[identifier] = Citation(id=identifier, kind='calculation', assessment_id=assessment_id, formula='count unique, evidenced document event date identities with nonoverlapping calendar intervals', function='ai.risk_calculations.event_count', inputs=event_audit)
            event_source_ids = list(dict.fromkeys(source for row in event_audit for source in row['citation_ids']))
            item = observation(identifier, 'Documented historical flood events', f'The document history identifies {len(event_keys)} distinct documented historical flood events; this count is not a future recurrence forecast.', 'historical', [identifier, *event_source_ids], [NumericFinding(metric='documented_historical_flood_events', value=len(event_keys), unit='events', citation_id=identifier)])
            history.append(item)
        elif event_keys:
            limitation('event_count_unverified', 'Unknown or repeated event dates prevent a verified count of distinct historical floods.')
        for currency in sorted({record['currency'] for record in records if record['currency']}):
            for kind in ('claimed', 'settled', 'paid'):
                candidates = [record for record in records if record['currency'] == currency and record['kind'] == kind]
                if not candidates:
                    continue
                largest = largest_historical_amount(candidates)
                identifier = f'calc.historical.largest.{kind}.{currency}'
                citations[identifier] = Citation(id=identifier, kind='calculation', assessment_id=assessment_id, function='ai.risk_calculations.largest_historical_amount', formula='max documented amounts within one financial category and currency', inputs=[{'amount': str(row['amount']), 'event_key': row['event_key'], 'citation_ids': row['citation_ids']} for row in candidates])
                item = observation(identifier, f'Largest documented historical {kind} amount', f'The largest documented historical {kind} amount is {largest["amount"]} {currency}; claimed, settled and paid amounts remain separate.', 'historical', [identifier, *largest['citation_ids']], [NumericFinding(metric=f'largest_historical_{kind}_amount', value=largest['amount'], unit=currency, citation_id=identifier)])
                history.append(item)
        grouped_descriptions = {}
        for key, description, ids in descriptions:
            normalized = re.sub(r'\W+', ' ', description.casefold()).strip()
            grouped_descriptions.setdefault(normalized, []).append((key, description, ids))
        for index, rows in enumerate(grouped_descriptions.values()):
            keys = {row[0] for row in rows}
            if repeated_description_count(keys) < 2:
                continue
            identifier = f'calc.historical.repeated_description.{index}'
            ids = list(dict.fromkeys(source for row in rows for source in row[2]))
            citations[identifier] = Citation(id=identifier, kind='calculation', assessment_id=assessment_id, function='ai.risk_calculations.repeated_description_count', formula='count distinct evidenced event identities sharing the same normalized reported vulnerability description', inputs=[{'event_key': row[0], 'description': row[1], 'citation_ids': row[2]} for row in rows])
            item = observation(identifier, 'Repeated documented vulnerability description', f'The same reported vulnerability description appears in {len(keys)} distinct historical events: {rows[0][1]}', 'historical', [identifier, *ids], [NumericFinding(metric='events_with_repeated_description', value=len(keys), unit='events', citation_id=identifier)])
            history.append(item)

        for item in validate_extraction(extraction):
            limitation(item.code, item.message, [validation_id])
        for document in documents:
            for asset in document.facts.assets:
                missing = [field for field in ('coordinates', 'construction_type', 'floor_area', 'stated_value') if getattr(asset, field).status != 'provided']
                if missing:
                    limitation('missing_building_model_inputs', f'{asset.name.value or "An identified building"}: verify {", ".join(missing)} before model execution; site coordinates and overall property totals are not individual building inputs.', [validation_id])
        model_status_id = f'validation.model_availability.{assessment_id}'
        citations[model_status_id] = Citation(id=model_status_id, kind='validation', assessment_id=assessment_id, function='UnderwritingModelBackend.get_model_results', inputs=[])
        try:
            async with asyncio.timeout(self.settings.agent_tool_timeout_seconds):
                received = await self.backend.get_model_results(assessment_id)
            model = ModelOutput.model_validate_json(received.model_dump_json() if isinstance(received, ModelOutput) else json.dumps(received, allow_nan=False))
            if model.assessment_id != assessment_id:
                raise ValueError('Wrong assessment')
            if model.status == 'ready':
                citations[model_status_id].inputs = [{'backend_status': 'ready', 'model_result_id': model.result_id, 'model_version': model.model_version}]
                model_findings, audits, notes = model_concentrations(assessment_id, model)
                for item in audits:
                    citations[item.id] = item
                observations.extend(model_findings)
                for note in notes:
                    limitation('model_breakdown_limitation', note)
                for assumption in model.assumptions:
                    limitation('model_assumption', assumption, [audits[0].id])
                for note in model.limitations:
                    limitation('backend_model_limitation', note, [audits[0].id])
                if not model.assumptions:
                    limitation('vulnerability_assumptions_unverified', 'The backend did not supply vulnerability assumptions; verify class applicability and industrial asset treatment.', [audits[0].id])
                if model.result_id is None:
                    limitation('model_result_identity_missing', 'An external model result ID was not supplied; the actual response snapshot and version are retained for audit.', [audits[0].id])
            elif model.status == 'missing_hazard_coverage':
                citations[model_status_id].inputs = [{'backend_status': model.status, 'missing_return_periods': model.missing_return_periods}]
                limitation('missing_hazard_coverage', 'The backend reports unsupported hazard coverage; no modelled losses or concentrations are available.', [model_status_id])
            else:
                citations[model_status_id].inputs = [{'backend_status': model.status}]
                limitation('no_model_results', 'No catastrophe-model results are available; only historical and document-reported information can be assessed.', [model_status_id])
        except (IntegrationUnavailable, TimeoutError, ValidationError, ValueError, TypeError):
            citations[model_status_id].inputs = [{'backend_status': 'unavailable_or_invalid'}]
            limitation('model_unavailable', 'Model results are unavailable, timed out or failed validation; no modelled statistics are generated.', [model_status_id])
        except Exception:
            citations[model_status_id].inputs = [{'backend_status': 'failed'}]
            limitation('model_unavailable', 'The model dependency failed; no modelled statistics are generated.', [model_status_id])
        if not documents and not extraction.claims:
            limitation('missing_document_facts', 'No validated insurance document or historical claims facts are available.')
        if not model_findings:
            limitation('hazard_coverage_unverified', 'Verify facility/building location and supported hazard coverage before relying on catastrophe-loss estimates.', [model_status_id])
            limitation('vulnerability_assumptions_unverified', 'No verified vulnerability assumptions are available; class applicability and non-building loss treatment have not been inferred.', [model_status_id])
        if any(getattr(doc.facts.financial_exposure, field).status != 'not_provided' for doc in documents for field in ('machinery_values', 'inventory_values', 'contents_values', 'business_interruption_limit')):
            limitation('industrial_parameters_unknown', 'Machinery, inventory, contents and business interruption require verified industrial loss parameters; building curves do not establish their losses.')
        limitation('risk_rating_not_supplied', 'No approved risk-rating framework output was supplied; an independent categorical or numerical risk level is not assigned.')
        observation('quality.assessment', 'Assessment information status', 'Underwriting conclusions require source verification; historical observations and document reports are not future predictions.', 'data_quality', [validation_id])

        # Data-quality limitations are eligible for mitigation reviews, not unsupported
        # top risk drivers. Create auditable review observations for the LLM to reference.
        for index, item in enumerate(list(limitations)):
            if item.citation_ids:
                observation(f'quality.{index}', item.code.replace('_', ' '), item.message, 'data_quality', item.citation_ids)
        known = {item.id: item for item in observations}
        eligible = [item for item in observations if item.evidence_type != 'data_quality']
        fallback = AnalysisNarrative(summary=[NarrativeSelection(finding_id=item.id, explanation='This evidence requires underwriting review') for item in eligible[:3]] or [NarrativeSelection(finding_id=next(iter(known)), explanation='Available information remains limited')], drivers=[NarrativeSelection(finding_id=item.id, explanation='Review its implications for asset protection and model applicability') for item in eligible[:3]], actions=[])
        narrative = fallback
        try:
            async with asyncio.timeout(self.settings.llm_timeout_seconds):
                payload = json.dumps({'assessment_id': assessment_id, 'findings': [item.model_dump(mode='json') for item in observations]}, ensure_ascii=False)
                if len(payload) > self.settings.max_document_characters * 4:
                    raise ValueError('Narrative budget exceeded')
                draft = AnalysisNarrative.model_validate(await self.llm.structured(UNDERWRITING_ANALYSIS_SYSTEM, payload, AnalysisNarrative))
            self._validate_narrative(draft, known)
            narrative = draft
        except Exception:
            limitation('ai_narrative_unavailable', 'AI narration was unavailable or failed grounding/format checks; deterministic findings are returned without claiming AI prioritization.')
        drivers = [RiskDriver(**known[item.finding_id].model_dump(), why_it_matters=item.explanation) for item in narrative.drivers if known[item.finding_id].evidence_type != 'data_quality'][:3]
        sentences = []
        for item in narrative.summary[:3]:
            fact = known[item.finding_id]
            # Source facts and numerals are server-controlled. Use one compact sentence
            # per selection; document text punctuation cannot create extra sentences.
            fragment = re.sub(r'(?<!\d)[.!?]|[.!?](?!\d)', ';', fact.statement)
            fragment = re.sub(r'\s+', ' ', fragment).strip('; ')
            if len(fragment) > 220:
                fragment = fragment[:220].rsplit(' ', 1)[0].rstrip('; ')
            sentences.append(f'{fragment}; {item.explanation.rstrip(".!?")}.')
        actions = [RecommendedAction(action=item.action, rationale=item.rationale, origin='independent_analysis', citation_ids=known[item.finding_id].citation_ids) for item in narrative.actions]
        actions.extend(proposals)
        # Missing information must remain visible even if the LLM does not select it.
        for item in limitations:
            if item.citation_ids and item.code in {'missing_coordinates', 'missing_insured_value', 'missing_building_model_inputs', 'not_provided', 'document_contradiction', 'financial_overlap_unverified', 'coverage_ambiguity', 'model_unavailable', 'missing_hazard_coverage', 'hazard_coverage_unverified', 'no_model_results', 'vulnerability_assumptions_unverified'}:
                actions.append(RecommendedAction(action='Verify or obtain the referenced information before relying on model or coverage conclusions.', rationale=item.message, origin='independent_analysis', citation_ids=item.citation_ids))
        return UnderwritingAnalysis(risk_summary=' '.join(sentences), risk_level=RiskLevel(source='No approved risk-rating framework output is available'), top_risk_drivers=drivers, historical_claims_findings=history, model_findings=model_findings, recommended_actions=actions, limitations=limitations, citations=list(citations.values()))

    @staticmethod
    def _validate_narrative(narrative, known):
        if len({item.finding_id for item in narrative.drivers}) != len(narrative.drivers):
            raise ValueError('Risk drivers must refer to distinct findings')
        for item in [*narrative.summary, *narrative.drivers, *narrative.actions]:
            if item.finding_id not in known:
                raise ValueError('Unsupported finding reference')
            texts = [item.explanation] if hasattr(item, 'explanation') else [item.action, item.rationale]
            for text in texts:
                if re.search(r'\d|[%$€£]|\b(?:zero|one|two|three|four|five|hundred|thousand|million|billion|percent|percentage|twice|double|triple|half|AAL|EAL)\b', text, re.I):
                    raise ValueError('Numerical narrative claims are not authorized')
                if re.search(r'[.!?]\s+\S', text):
                    raise ValueError('Narrative explanations must be a single sentence')


async def a_generate_underwriting_analysis(assessment_id: str, *, store=None, llm=None, model_backend=None, settings=None, principal=None, authorizer=None) -> dict:
    config = settings or Settings()
    repository = store if store is not None else AssessmentStore(config.storage_directory)
    client = llm if llm is not None else create_llm_client(config)
    try:
        result = await UnderwritingAnalysisService(repository, client, model_backend=model_backend, settings=config).generate(assessment_id, principal=principal, authorizer=authorizer)
        return json.loads(result.model_dump_json())
    finally:
        if store is None:
            repository.close()
        if llm is None and getattr(client, 'aclose', None):
            await client.aclose()


def generate_underwriting_analysis(assessment_id: str, **kwargs) -> dict:
    """Reusable synchronous entry point; returns the exact strict frontend schema."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(a_generate_underwriting_analysis(assessment_id, **kwargs))
    raise RuntimeError('Use await a_generate_underwriting_analysis inside async code')
