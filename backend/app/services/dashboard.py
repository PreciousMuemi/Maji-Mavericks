"""Read-only dashboard aggregation. No simulation, allocation, FX guessing or LLM calls."""
import asyncio
import hashlib
import json
import re
from collections import defaultdict
from decimal import Decimal

from ai.config import Settings
from ai.insurance_extraction import InsuranceDocumentExtractor, _validate_numeric, _walk
from ai.insurance_schemas import Coordinates, Money
from ai.intelligence_schemas import AnalysisLimitation, Citation
from ai.llm_client import LLMUnavailable
from ai.risk_analyzer import IntegrationUnavailable
from ai.underwriting_analysis import UnderwritingAnalysisService
from ai.underwriting_schemas import ModelOutput
from ai.validator import validate_extraction
from app.schemas.dashboard import AIInsights, Chart, ChartPoint, DashboardMap, DashboardResponse, KPI, MapLocation


def kes(amount: Decimal) -> str:
    return f'KES {amount:,.2f}'


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


class _SnapshotStore:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def insurance_results(self, _):
        return self.snapshot['insurance']

    def extraction(self, _):
        return self.snapshot['extraction']

    def documents(self, _):
        return self.snapshot['documents']


class _CachedModel:
    def __init__(self, model):
        self.model = model

    async def get_model_results(self, _):
        if self.model is None:
            raise IntegrationUnavailable('No validated stored model output')
        return self.model


class _EvidenceOnlyNarrator:
    async def structured(self, *args):
        # Deliberately avoid a paid provider call on dashboard GET.
        raise LLMUnavailable('Use the explicit analysis endpoint to create an AI briefing')


class DashboardService:
    def __init__(self, store, model_backend, settings: Settings):
        self.store, self.backend, self.settings = store, model_backend, settings

    async def get(self, assessment_id: str) -> DashboardResponse:
        snapshot = await asyncio.to_thread(self.store.dashboard_snapshot, assessment_id)
        original_document_count = len(snapshot['insurance'])
        notes = []
        # Recheck provenance at the ingestion boundary, even for host-written records.
        accepted = []
        for document in snapshot['insurance']:
            try:
                parsed = snapshot['documents'][document.document_id]
                InsuranceDocumentExtractor(None, self.settings)._validate_facts(document.facts, parsed)
                accepted.append(document)
            except (ValueError, KeyError):
                notes.append(AnalysisLimitation(code='document_evidence_unverified', message='An extraction has invalid source evidence and is excluded from dashboard metrics.'))
        snapshot['insurance'] = accepted
        model = None
        try:
            async with asyncio.timeout(self.settings.agent_tool_timeout_seconds):
                received = await self.backend.get_model_results(assessment_id)
            model = ModelOutput.model_validate_json(received.model_dump_json() if isinstance(received, ModelOutput) else json.dumps(received, allow_nan=False))
            if model.assessment_id != assessment_id:
                raise ValueError('Wrong assessment')
        except Exception:
            model = None
            notes.append(AnalysisLimitation(code='model_unavailable', message='Stored model results are unavailable, timed out or failed validation; no loss estimates are displayed.'))

        calculated = await UnderwritingAnalysisService(_SnapshotStore(snapshot), _EvidenceOnlyNarrator(),
                        model_backend=_CachedModel(model), settings=self.settings).generate(assessment_id)
        cached = snapshot['intelligence']
        # A briefing tied to an old model run must not contradict the current charts.
        if cached:
            cached_models = [i.get('backend_output') for c in cached.citations if c.kind == 'model' for i in c.inputs if 'backend_output' in i]
            current = model.model_dump(mode='json') if model and model.status == 'ready' else None
            cached_status = next((i.get('backend_status') for c in cached.citations if c.id.startswith('validation.model_availability.') for i in c.inputs), None)
            expected_status = model.status if model else 'unavailable_or_invalid'
            if cached_models != ([current] if current else []) or cached_status != expected_status or len(accepted) != original_document_count:
                cached = None
                notes.append(AnalysisLimitation(code='briefing_stale', message='The previous briefing no longer matches the current evidence or model results; regenerate underwriting analysis.'))
        analysis = cached or calculated
        citations = {c.id: c for c in [*calculated.citations, *analysis.citations]}

        def cite(sources):
            identifiers = []
            for source in sources:
                document = snapshot['documents'].get(source.document_id)
                segment = next((s for s in document.segments if s.locator == source.locator), None) if document else None
                if segment is None or not source.excerpt.strip() or source.excerpt not in segment.text or (source.page_number is not None and source.page_number != segment.page_number):
                    continue
                trusted = source.model_copy(update={'page_number': segment.page_number})
                identifier = 'doc.' + hashlib.sha256(trusted.model_dump_json().encode()).hexdigest()[:16]
                citations[identifier] = Citation(id=identifier, kind='document', assessment_id=assessment_id, source=trusted)
                identifiers.append(identifier)
            return list(dict.fromkeys(identifiers))

        def usable(fact):
            return fact.status == 'provided' and fact.confidence >= self.settings.extraction_review_confidence and bool(cite(fact.sources))

        def audit(name, formula, inputs, source_ids):
            identifier = 'dashboard.' + name + '.' + digest(inputs)
            citations[identifier] = Citation(id=identifier, kind='calculation', assessment_id=assessment_id,
                function='app.services.dashboard.DashboardService.get', formula=formula, inputs=inputs)
            return [identifier, *list(dict.fromkeys(source_ids))]

        kpis, charts = [], []

        def card(identifier, label, basis, amount=None, ids=None, note=None, review=False, monetary=True):
            kpis.append(KPI(id=identifier, label=label, value=amount,
                display_value=kes(amount) if amount is not None and monetary else str(amount) if amount is not None else 'Unavailable',
                currency='KES' if monetary else None, basis=basis, citation_ids=ids or [], note=note,
                availability='available' if amount is not None else 'review_required' if review else 'unavailable'))

        def money_card(identifier, label, facts):
            present = [f for f in facts if f.status != 'not_provided']
            eligible = [f for f in present if usable(f) and f.value.currency == 'KES']
            unique = {(f.value.amount, f.value.currency) for f in eligible}
            if len(unique) == 1 and len(eligible) == len(present) == 1:
                amount = eligible[0].value.amount
                card(identifier, label, 'document_reported', amount, [i for f in eligible for i in cite(f.sources)],
                     'Explicit document-reported amount; separate financial categories are not added.')
            else:
                card(identifier, label, 'document_reported', note='Requires one unambiguous, source-backed KES amount; multiple document scopes require confirmation, with no inferred limit or currency conversion.', review=bool(present))

        docs = snapshot['insurance']
        money_card('property_sum_insured', 'Property sum insured', [d.facts.financial_exposure.property_sum_insured for d in docs])
        money_card('requested_flood_limit', 'Requested flood limit (proposed)', [d.facts.insurance_terms.requested_flood_limit for d in docs])

        # Select one stream; do not combine claimed, settled and paid amounts.
        history = calculated.historical_claims_findings
        historical = next((f for metric in ('historical_claimed_total', 'historical_paid_total', 'historical_settled_total')
            for f in history if any(n.metric == metric and n.unit == 'KES' for n in f.numerical_findings)), None)
        if historical:
            number = next(n for n in historical.numerical_findings if n.unit == 'KES')
            audit_ref = citations[number.citation_id]
            kind = number.metric.removeprefix('historical_').removesuffix('_total')
            card('historical_flood_claims', f'Historical flood claims ({kind}; KES only)', 'historical', number.value, historical.citation_ids,
                 'Available documented amounts only; claimed, settled and paid streams remain separate. Past losses are not forecasts.')
            grouped, sources, inputs_by_year = defaultdict(lambda: Decimal('0')), defaultdict(list), defaultdict(list)
            for row in audit_ref.inputs:
                # Chart dates must be evidenced; a register claim ID is not a year.
                key = row['event_key']
                if '.claims_register.' in audit_ref.id:
                    claim = next((c for c in snapshot['extraction'].claims if c.claim_id == key or str(c.event_date) == key), None)
                    key = str(claim.event_date) if claim and claim.event_date and any(str(claim.event_date) in s.excerpt for s in claim.sources) else ''
                years = re.findall(r'(?<!\d)(\d{4})(?!\d)', key or '')
                if len(set(years)) != 1:
                    break
                year = years[0]
                grouped[year] += Decimal(row['amount'])
                sources[year].extend(row['citation_ids'])
                inputs_by_year[year].append({**row, 'event_year': year})
            else:
                if grouped:
                    points = [ChartPoint(label=year, value=amount, display_value=kes(amount),
                        citation_ids=audit('historical_year.' + year, 'sum verified same-stream KES amounts with the same evidenced event year',
                            inputs_by_year[year], sources[year]))
                        for year, amount in sorted(grouped.items())]
                    charts.append(Chart(id='historical_flood_claims_by_year', type='bar', title=f'Documented historical flood claims by year ({kind}; KES only)', x_label='Historical event year', y_label='Documented amount (KES)', basis='historical', unit='KES', points=points,
                        note='Available evidenced events only; no missing years filled with zero. This is historical experience, not a prediction.'))
        else:
            card('historical_flood_claims', 'Historical flood claims', 'historical', note='No unambiguous cumulative KES claims stream is available; no claimed/settled/paid streams are combined.')

        mapping = snapshot['mapping']
        if mapping:
            from ai.exposure_mapping import get_missing_model_inputs, validate_exposure_records
            mapping = validate_exposure_records(mapping)
            missing_groups = defaultdict(list)
            for missing in get_missing_model_inputs(mapping):
                missing_groups[(tuple(sorted(missing.fields)), tuple(sorted(missing.blockers)))].append(missing.record_id)
            for (fields, blockers), record_ids in missing_groups.items():
                ids = audit('missing_inputs', 'count records with the same missing-field and validation-blocker set',
                            [{'record_ids': record_ids, 'count': len(record_ids), 'missing_fields': list(fields), 'blockers': list(blockers)}], [])
                notes.append(AnalysisLimitation(code='missing_model_inputs', message=f'{len(record_ids)} exposure records: fields {", ".join(fields) or "none"}; blockers {", ".join(blockers)}.', citation_ids=ids))
            rows = mapping.valid_records
            class_rows = defaultdict(list)
            for row in rows:
                class_rows[row.normalized_values['housing_class']].append(row)
            points = []
            for construction, members in sorted(class_rows.items()):
                ids = [i for r in members for field in ('loc_id', 'housing_class', 'tiv_kes') for i in cite(r.sources.get(field, []))]
                inputs = [{'record_id': r.record_id, 'loc_id': r.normalized_values['loc_id'], 'housing_class': construction,
                           'tiv_kes': str(r.normalized_values['tiv_kes']), 'derived_fields': [f.model_dump(mode='json') for f in r.derived_fields],
                           'confirmations': [c.model_dump(mode='json') for c in r.confirmations]} for r in members]
                amount = sum((Decimal(str(r.normalized_values['tiv_kes'])) for r in members), Decimal('0'))
                points.append(ChartPoint(label=construction, value=amount, display_value=kes(amount),
                    citation_ids=audit('exposure_class', 'sum validated building-specific tiv_kes within the same supported construction class; excludes specialized industrial categories', inputs, ids)))
            if points:
                charts.append(Chart(id='exposure_by_construction_class', type='bar', title='Validated building exposure by supported construction class', x_label='Supported construction class', y_label='Building insured values (KES)', basis='validated_exposure', unit='KES', points=points,
                    note='Only validated building records; review-required rows and machinery, inventory, contents and interruption are excluded. No overall property sum is allocated.'))

        assets = [a for d in docs for a in d.facts.assets]
        names = [a.name.value.casefold().strip() for a in assets if usable(a.name)]
        if assets and len(names) == len(assets) == len(set(names)):
            ids = [i for a in assets for i in cite(a.name.sources)]
            card('number_of_assets', 'Documented buildings', 'document_reported', Decimal(len(assets)),
                 audit('asset_count', 'count uniquely named source-backed buildings; financial categories are not counted as structures', [{'name': a.name.value} for a in assets], ids), monetary=False)
        elif mapping and (scheduled := [r for r in [*mapping.valid_records, *mapping.review_required_records] if r.asset_category == 'buildings']) and all(r.normalized_values.get('loc_id') and cite(r.sources.get('loc_id', [])) for r in scheduled) and len({r.normalized_values['loc_id'] for r in scheduled}) == len(scheduled):
            card('number_of_assets', 'Scheduled buildings', 'document_reported', Decimal(len(scheduled)),
                 audit('asset_count', 'count unique evidenced building loc_id values; class applicability and model readiness are assessed separately', [{'loc_id': r.normalized_values['loc_id']} for r in scheduled], [i for r in scheduled for i in cite(r.sources.get('loc_id', []))]), monetary=False)
        else:
            canonical = snapshot['extraction'].assets
            identifiers = [a.asset_id for a in canonical if cite(a.sources)]
            if not assets and identifiers and len(identifiers) == len(canonical) == len(set(identifiers)):
                card('number_of_assets', 'Documented assets', 'document_reported', Decimal(len(canonical)),
                     audit('asset_count', 'count unique extracted asset IDs with matching source evidence', [{'asset_id': a.asset_id} for a in canonical], [i for a in canonical for i in cite(a.sources)]), monetary=False)
            else:
                card('number_of_assets', 'Number of assets', 'document_reported', note='Asset identities are absent or may overlap; no asset count is inferred.', review=bool(assets or canonical), monetary=False)

        # A breakdown requires explicit non-overlap evidence; totals alone are not a breakdown.
        inventories = [d.facts.financial_exposure for d in docs if d.facts.financial_exposure.inventory_breakdown]
        if len(inventories) == 1:
            financial = inventories[0]
            items = financial.inventory_breakdown
            disjoint = financial.inventory_breakdown_disjoint
            if usable(disjoint) and disjoint.value is True and len(items) >= 2 and all(usable(i.name) and usable(i.stated_value) and i.stated_value.value.currency == 'KES' for i in items) and len({i.name.value.casefold() for i in items}) == len(items):
                charts.append(Chart(id='inventory_breakdown', type='bar', title='Document-reported inventory breakdown', x_label='Inventory item', y_label='Reported item value (KES)', basis='document_reported', unit='KES',
                    points=[ChartPoint(label=i.name.value, value=i.stated_value.value.amount, display_value=kes(i.stated_value.value.amount), citation_ids=[*cite(i.name.sources), *cite(i.stated_value.sources), *cite(disjoint.sources)]) for i in items],
                    note='Explicitly separate reported inventory items; this is exposure, not a modelled inventory loss.'))
            else:
                notes.append(AnalysisLimitation(code='inventory_breakdown_unverified', message='Inventory line items lack complete KES values, identities or explicit non-overlap evidence; the breakdown chart is omitted.'))

        model_ids = [c.id for c in calculated.citations if c.kind == 'model']
        periods = sorted([p for p in model.period_results if p.currency == 'KES'], key=lambda p: p.return_period) if model and model.status == 'ready' else []
        rp100 = next((p for p in periods if p.return_period == 100), None)
        card('rp100_modelled_loss', 'RP100 modelled flood loss', 'modelled', rp100.total_loss if rp100 else None, model_ids,
             'Actual stored catastrophe-model result; not a historical claim total.' if rp100 else 'No accepted KES RP100 result is available; no scenario is interpolated.')
        if periods:
            points = [ChartPoint(label=str(p.return_period), value=p.total_loss, display_value=kes(p.total_loss), citation_ids=model_ids) for p in periods]
            charts.append(Chart(id='return_period_loss_curve', type='line', title='Actual modelled flood losses by return period', x_label='Return period (years)', y_label='Modelled loss (KES)', basis='modelled', unit='KES', points=points,
                note='Reported scenarios only; no interpolation, probability, AAL or missing scenarios inferred.'))
            if len(periods) >= 2:
                charts.append(Chart(id='scenario_comparison', type='bar', title='Comparison of actual return-period scenarios', x_label='Return period (years)', y_label='Modelled loss (KES)', basis='modelled', unit='KES', points=points,
                    note='Same-currency stored return-period scenario totals; no unexecuted mitigation scenarios are shown.'))
            for period in periods:
                rows = period.building_losses
                if rows and len({r.loc_id for r in rows}) == len(rows) and sum((r.loss for r in rows), Decimal('0')) <= period.total_loss:
                    charts.append(Chart(id=f'building_loss_ranking_rp{period.return_period}', type='bar', title=f'Modelled building loss ranking — RP{period.return_period}', x_label='Backend building ID', y_label='Modelled building loss (KES)', basis='modelled', unit='KES', return_period=period.return_period,
                        points=[ChartPoint(label=r.loc_id, value=r.loss, display_value=kes(r.loss), citation_ids=audit('building_rank', 'sort actual same-period building losses descending, then loc_id', [{'loc_id': r.loc_id, 'loss': str(r.loss), 'return_period': period.return_period, 'model_result_id': model.result_id}], model_ids)) for r in sorted(rows, key=lambda r: (-r.loss, r.loc_id))],
                        note='Ranks only returned building rows; completeness of the modelled building breakdown must be verified.'))

        locations = []
        for document in docs:
            fact = document.facts.insured.coordinates
            if usable(fact):
                locations.append(MapLocation(id=document.document_id + '.site', label=document.facts.insured.facility_location.value or 'Documented facility', lat=fact.value.latitude, lon=fact.value.longitude, coordinate_scope='site', citation_ids=cite(fact.sources)))
            for index, asset in enumerate(document.facts.assets):
                if usable(asset.coordinates):
                    locations.append(MapLocation(id=f'{document.document_id}.building.{index}', label=asset.name.value or 'Documented building', lat=asset.coordinates.value.latitude, lon=asset.coordinates.value.longitude, coordinate_scope='building', citation_ids=cite(asset.coordinates.sources)))
        if mapping:
            for row in [*mapping.valid_records, *mapping.review_required_records]:
                if row.coordinate_scope not in {'building', 'assumed_site'} or any(i.blocking and (i.field in {'coordinates', 'lat', 'lon'} or i.code == 'site_proxy_unapproved') for i in [*row.mapping_issues, *row.validation_issues]):
                    continue
                sources = [s for field in ('lat', 'lon') for s in row.sources.get(field, [])]
                try:
                    coordinates = Coordinates(latitude=row.normalized_values.get('lat'), longitude=row.normalized_values.get('lon'))
                    if not cite(sources):
                        continue
                    _validate_numeric(coordinates, sources)
                except ValueError:
                    continue
                if any(p.coordinate_scope == row.coordinate_scope and p.lat == coordinates.latitude and p.lon == coordinates.longitude for p in locations):
                    continue
                ids = audit('mapped_location', 'use validated coordinates at their recorded scope; approved site proxies remain labelled assumed_site', [{'loc_id': row.normalized_values['loc_id'], 'lat': str(row.normalized_values['lat']), 'lon': str(row.normalized_values['lon']), 'scope': row.coordinate_scope, 'confirmations': [c.model_dump(mode='json') for c in row.confirmations]}], [i for field in ('lat', 'lon') for i in cite(row.sources.get(field, []))])
                locations.append(MapLocation(id=row.record_id, label=row.normalized_values['loc_id'], lat=row.normalized_values['lat'], lon=row.normalized_values['lon'], coordinate_scope=row.coordinate_scope, citation_ids=ids))

        missing_cards = [k for k in kpis if k.availability != 'available']
        for k in missing_cards:
            notes.append(AnalysisLimitation(code='missing_' + k.id, message=f'{k.label}: {k.note}'))
        foreign_currencies = {f.value.currency for d in docs for _, f in _walk(d.facts) if f.status == 'provided' and isinstance(f.value, Money) and f.value.currency and f.value.currency != 'KES'}
        if model:
            foreign_currencies.update(p.currency for p in model.period_results if p.currency != 'KES')
        if foreign_currencies:
            notes.append(AnalysisLimitation(code='currency_conversion_required', message='Non-KES document/model amounts are withheld from dashboard figures; an approved currency conversion is required.'))
        validation = bool(notes or validate_extraction(snapshot['extraction']) or any(d.review_status == 'requires_review' or d.review_items or d.contradictions or any(f.status in {'uncertain', 'contradicted'} or f.confidence is not None and f.confidence < self.settings.extraction_review_confidence for _, f in _walk(d.facts)) for d in docs) or mapping and mapping.review_required_records)
        statuses = []
        if docs or snapshot['extraction'].assets or snapshot['extraction'].claims or mapping and (mapping.valid_records or mapping.review_required_records):
            statuses.append('document_extracted')
        if validation:
            statuses.append('validation_required')
        if mapping and mapping.model_readiness == 'ready':
            statuses.append('model_ready')
        if model and model.status == 'ready':
            statuses.append('calculation_completed')
        if model and model.status == 'missing_hazard_coverage':
            status = 'hazard_coverage_unavailable'
        elif not docs and not snapshot['extraction'].assets and not snapshot['extraction'].claims and not periods and not any(k.availability == 'available' for k in kpis):
            status = 'insufficient_data'
        elif missing_cards or validation or not periods:
            status = 'partial_assessment' if periods or any(k.availability == 'available' for k in kpis) else 'validation_required'
        else:
            status = 'calculation_completed'
        if status not in statuses:
            statuses.append(status)
        kpis.append(KPI(id='assessment_status', label='Assessment status', value=status, display_value=status.replace('_', ' ').capitalize(), availability='available', basis='workflow'))
        limitations = list({(l.code, l.message): l for l in [*calculated.limitations, *analysis.limitations, *notes]}.values())
        if cached:
            limitations = [l for l in limitations if l.code != 'ai_narrative_unavailable']
        missing = [l for l in limitations if l.code.startswith(('missing_', 'not_provided')) or l.code in {'document_evidence_unverified', 'currency_conversion_required', 'inventory_breakdown_unverified'}]
        # Format display prose; exact underlying evidence remains unchanged in citations.
        def display(text):
            text = re.sub(r'\bKES\s+([0-9]+(?:\.[0-9]+)?)(?![\d.,])', lambda m: kes(Decimal(m[1])), text)
            return re.sub(r'(?<![\w.,])([0-9]+(?:\.[0-9]+)?) KES\b', lambda m: kes(Decimal(m[1])), text)
        def foreign(text):
            return any(re.search(r'\b' + re.escape(c) + r'\b', text) for c in foreign_currencies)
        drivers = [d.model_copy(update={'statement': display(d.statement)}) for d in analysis.top_risk_drivers
                   if not foreign(d.statement) and not any(n.unit not in {'KES', 'events', 'rank', 'percent'} for n in d.numerical_findings)]
        summary = display(analysis.risk_summary) if not foreign(analysis.risk_summary) else 'The documented evidence requires underwriting review; non-KES financial findings require an approved conversion before dashboard display.'
        recommendations = [r.model_copy(update={'action': display(r.action), 'rationale': display(r.rationale)}) if not foreign(r.action + ' ' + r.rationale) else r.model_copy(update={'action': 'Review the source recommendation and arrange an approved currency conversion.', 'rationale': 'The original recommendation contains non-KES amounts; consult its source evidence.'}) for r in analysis.recommended_actions]
        # Validation snapshots are represented by their audit function, rather than
        # duplicating the complete uploaded documents in the frontend payload.
        references = [c.model_copy(update={'inputs': []}) if c.kind == 'validation' else c for c in citations.values()]
        return DashboardResponse(assessment_id=assessment_id, status=status, statuses=statuses, kpis=kpis, charts=charts,
            map=DashboardMap(locations=locations, coverage_status='unavailable' if model and model.status == 'missing_hazard_coverage' else 'model_results_available' if periods else 'unverified',
                note='A site point is not an individual building position. Returned losses do not establish complete spatial hazard coverage; verify extent with the modelling team.'),
            ai_insights=AIInsights(summary=summary, origin='stored_ai_analysis' if cached and not any(l.code == 'ai_narrative_unavailable' for l in cached.limitations) else 'deterministic_evidence', risk_drivers=drivers, recommendations=recommendations, limitations=limitations, missing_data=missing, assumptions=model.assumptions if model else []),
            source_references=references)
