"""Conservative exposure normalization. This module never runs vulnerability curves."""
import hashlib
import json
import re
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Settings
from .document_parser import DocumentError, DocumentParser
from .exposure_schemas import (
    Confirmation, ContractObservation, DerivedField, ExposureCandidate, ExposureContract,
    ExposureIssue, ExposureMappingResult, FXApproval, MissingModelInput,
)
from .insurance_schemas import InsuranceDocumentResult, InsuranceFacts
from .schemas import SourceReference

CORE_FIELDS = ('loc_id', 'lat', 'lon', 'housing_class', 'floor_area_m2', 'tiv_kes')
ALIASES = {
    'loc_id': ('location_id', 'building_id', 'asset_id', 'location_ref'),
    'lat': ('latitude',), 'lon': ('longitude', 'lng'),
    'housing_class': ('vulnerability_class',),
    'floor_area_m2': ('floor_area', 'floor_area_sqm', 'floor_area_sqft', 'area_m2', 'area_sqft'),
    'tiv_kes': ('building_value_kes', 'structure_value_kes', 'insured_value', 'sum_insured', 'insured_value_kes'),
    'construction': ('construction_type', 'construction_material'),
    'currency': ('currency_code',), 'occupancy': ('industry', 'occupancy_type'),
    'asset_category': ('category', 'asset_type'), 'coordinate_scope': ('gps_scope',),
    'value_scope': ('insured_value_scope',), 'floor_area_unit': ('area_unit',),
}


def _label(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().casefold()).strip('_')


def _issue(record, code, field, message):
    record.mapping_issues.append(ExposureIssue(code=code, field=field, message=message))


def _number(value):
    if value is None or isinstance(value, bool) or str(value).strip() == '':
        return None
    text = str(value).strip()
    # Ambiguous decimal/thousands locales are not guessed.
    if ',' in text:
        if not re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', text):
            raise ValueError('Ambiguous numeric formatting')
        text = text.replace(',', '')
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError('Invalid number') from exc
    if not result.is_finite():
        raise ValueError('Non-finite number')
    return result


def _json_safe(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return str(value) if not Decimal(str(value)).is_finite() else value
    return str(value)


def _point(record):
    try:
        latitude = _number(record.normalized_values.get('lat'))
        longitude = _number(record.normalized_values.get('lon'))
        return (latitude, longitude) if latitude is not None and longitude is not None else None
    except ValueError:
        return None


def _column_plan(columns, explicit, contract):
    plan, ambiguous = {}, []
    for field in dict.fromkeys([*contract.fields, *ALIASES]):
        if field in explicit:
            if explicit[field] not in columns:
                ambiguous.append((field, 'Selected source column does not exist'))
            else:
                plan[field] = explicit[field]
            continue
        matches = [column for column in columns if _label(column) in {field, *ALIASES.get(field, ())}]
        if len(matches) == 1:
            plan[field] = matches[0]
        elif len(matches) > 1:
            ambiguous.append((field, 'Multiple source columns match; select one explicitly'))
    return plan, ambiguous


def _read_table(path, settings):
    if Path(path).suffix.casefold() not in {'.csv', '.xlsx'}:
        raise DocumentError('Exposure tables must be CSV or XLSX')
    document = DocumentParser(settings).parse_file(path, str(uuid4()))
    rows = []
    if Path(path).suffix.casefold() == '.csv':
        for segment in document.segments:
            rows.append((json.loads(segment.text), segment))
    else:
        headers = {}
        for segment in document.segments:
            cells = json.loads(segment.text)
            sheet = segment.locator.rsplit(':row:', 1)[0]
            recognized = {_label(cell) for cell in cells} & {name for field, aliases in ALIASES.items() for name in (field, *aliases)}
            if len(recognized) >= 2:
                headers[sheet] = cells
                continue
            if sheet in headers:
                names = headers[sheet]
                if len(set(names)) != len(names) or any(not str(name).strip() for name in names):
                    raise DocumentError('Spreadsheet headers must be unique and nonempty; supply a cleaned schedule')
                rows.append((dict(zip(names, cells)), segment))
        if not rows:
            raise DocumentError('No recognizable spreadsheet exposure header was found')
    return document, rows


def inspect_exposure_csv(file_path, *, settings=None) -> ContractObservation:
    """Inspect actual headers and observed values; never infer an authoritative enum."""
    if Path(file_path).suffix.casefold() != '.csv':
        raise DocumentError('Contract observation requires the actual CSV')
    document, rows = _read_table(file_path, settings or Settings())
    columns = list(rows[0][0]) if rows else []
    class_column = next((name for name in columns if _label(name) == 'housing_class'), None)
    classes = sorted({str(row[class_column]).strip() for row, _ in rows if class_column and str(row[class_column]).strip()})
    return ContractObservation(columns=columns, observed_housing_classes=classes, row_count=len(rows))


def _classify_construction(record, contract):
    if record.normalized_values.get('housing_class') is not None:
        return
    construction = record.original_values.get('construction')
    if not construction:
        return
    rules = [rule for rule in contract.construction_rules if _label(rule.construction) == _label(construction)]
    if len(rules) != 1:
        _issue(record, 'construction_unmapped', 'housing_class', 'No unique, declared construction-to-vulnerability mapping is available')
        return
    rule = rules[0]
    record.normalized_values['housing_class'] = rule.housing_class
    record.derived_fields.append(DerivedField(field='housing_class', value=rule.housing_class, method='construction_rule', explanation=f'{rule.rule_id}: {rule.rationale}', input_fields=['construction'], sources=record.sources.get('construction', [])))
    if rule.requires_confirmation:
        _issue(record, 'construction_confirmation_required', 'housing_class', 'Underwriter must confirm the proposed construction mapping')


def _normalize(raw, contract, column_mapping, sources, fx_approval):
    plan, ambiguity = _column_plan(list(raw), column_mapping, contract)
    original = {field: raw.get(column) for field, column in plan.items()}
    original['_source_row'] = _json_safe(raw)
    record = ExposureCandidate(record_id=str(uuid4()), asset_category='unclassified', original_values=_json_safe(original), sources={field: sources.get(field, sources.get('*', [])) for field in plan})
    for field, message in ambiguity:
        _issue(record, 'ambiguous_column', field, message)
    category = _label(original.get('asset_category', ''))
    aliases = {'building': 'buildings', 'bi': 'business_interruption'}
    category = aliases.get(category, category)
    if category in {'buildings', 'machinery', 'inventory', 'contents', 'business_interruption'}:
        record.asset_category = category
    elif {'loc_id', 'housing_class', 'floor_area_m2', 'tiv_kes'}.issubset(plan):
        record.asset_category = 'buildings'
        record.derived_fields.append(DerivedField(field='asset_category', value='buildings', method='column_alias', explanation='The supplied table uses the explicit building exposure columns'))
    else:
        _issue(record, 'asset_category_unknown', 'asset_category', 'Confirm the asset category; non-building assets require separate loss models')
    record.occupancy = str(original.get('occupancy') or 'unknown').strip().casefold()
    for field in contract.fields:
        value = original.get(field)
        specification = contract.fields[field]
        try:
            if specification.kind in {'number', 'integer'}:
                parsed = _number(value)
                if specification.kind == 'integer' and parsed is not None and parsed != parsed.to_integral_value():
                    raise ValueError('Expected integer')
                record.normalized_values[field] = parsed
            elif specification.kind == 'boolean':
                if value is None or value == '':
                    record.normalized_values[field] = None
                elif value in (True, 'true', 'True', '1', 1):
                    record.normalized_values[field] = True
                elif value in (False, 'false', 'False', '0', 0):
                    record.normalized_values[field] = False
                else:
                    raise ValueError('Expected boolean')
            else:
                record.normalized_values[field] = str(value).strip() if value is not None and str(value).strip() else None
        except ValueError:
            record.normalized_values[field] = None
            _issue(record, 'invalid_number', field, 'Value is invalid, non-finite or uses ambiguous numerical formatting')
        if field in plan and _label(plan[field]) != field:
            record.derived_fields.append(DerivedField(field=field, value=record.normalized_values[field], method='column_alias', explanation=f'Source column {plan[field]} mapped to {field}', input_fields=[plan[field]], sources=record.sources.get(field, [])))
    coordinate_scope = _label(original.get('coordinate_scope', ''))
    if coordinate_scope in {'building', 'site'}:
        record.coordinate_scope = coordinate_scope
    elif record.normalized_values.get('lat') is not None or record.normalized_values.get('lon') is not None:
        # Exact building model columns are a declared per-location schedule. Aliases
        # from generic property schedules do not establish coordinate scope.
        record.coordinate_scope = 'building' if all(plan.get(f) == f for f in CORE_FIELDS) else 'unknown'
    if record.coordinate_scope == 'site':
        lat, lon = record.normalized_values.get('lat'), record.normalized_values.get('lon')
        if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
            record.site_coordinates = {'lat': float(lat), 'lon': float(lon)}
        record.normalized_values['lat'] = record.normalized_values['lon'] = None
    area = record.normalized_values.get('floor_area_m2')
    area_column = _label(plan.get('floor_area_m2', ''))
    unit = _label(original.get('floor_area_unit', ''))
    if area is not None:
        if 'sqft' in area_column or unit in {'ft2', 'sqft', 'square_feet'}:
            converted = area * Decimal('0.09290304')
            record.normalized_values['floor_area_m2'] = converted
            record.derived_fields.append(DerivedField(field='floor_area_m2', value=converted, method='unit_conversion', explanation='Square feet × 0.09290304 = square metres', input_fields=['floor_area_m2', 'floor_area_unit'], sources=record.sources.get('floor_area_m2', [])))
        elif area_column not in {'floor_area_m2', 'floor_area_sqm', 'area_m2'} and unit not in {'m2', 'sqm', 'square_metres', 'square_meters'}:
            _issue(record, 'area_unit_unknown', 'floor_area_m2', 'The source does not establish square metres')
    amount = record.normalized_values.get('tiv_kes')
    value_column = _label(plan.get('tiv_kes', ''))
    currency = str(original.get('currency') or '').strip().upper()
    if value_column.endswith('_kes') and not currency:
        currency = 'KES'
    record.normalized_values['_source_currency'] = currency or None
    if currency == 'KES' and not original.get('currency') and value_column.endswith('_kes'):
        record.derived_fields.append(DerivedField(field='currency', value='KES', method='column_alias', explanation='Currency is explicit in the source column suffix _kes', input_fields=[plan['tiv_kes']], sources=record.sources.get('tiv_kes', [])))
    if amount is not None:
        if value_column.endswith('_kes') and currency and currency != 'KES':
            _issue(record, 'currency_header_conflict', 'tiv_kes', 'The source column declares KES but the currency cell declares another currency')
        if value_column not in {'tiv_kes', 'building_value_kes', 'structure_value_kes', 'insured_value_kes'} and _label(original.get('value_scope', '')) != 'per_asset':
            _issue(record, 'value_scope_unknown', 'tiv_kes', 'Confirm that the amount belongs to this asset, not an overall facility sum')
        if _label(original.get('value_scope', '')) in {'site', 'site_total', 'facility', 'facility_total', 'overall'}:
            record.normalized_values['tiv_kes'] = None
            _issue(record, 'facility_value_not_allocated', 'tiv_kes', 'A facility total cannot populate individual structure values')
        elif currency == 'KES':
            pass
        elif currency and fx_approval and fx_approval.approved and fx_approval.currency == currency:
            date.fromisoformat(fx_approval.effective_date)
            converted = amount * fx_approval.kes_per_unit
            record.normalized_values['tiv_kes'] = converted
            record.derived_fields.append(DerivedField(field='tiv_kes', value=converted, method='currency_conversion', explanation=f'Approved {currency}/KES rate {fx_approval.kes_per_unit} on {fx_approval.effective_date}; reference {fx_approval.reference}; {fx_approval.reason}', input_fields=['tiv_kes', 'currency'], sources=record.sources.get('tiv_kes', []), approved_by=fx_approval.approved_by))
        else:
            record.normalized_values['tiv_kes'] = None
            _issue(record, 'currency_conversion_required' if currency else 'currency_unknown', 'tiv_kes', 'A documented currency and explicit approved KES conversion are required')
    _classify_construction(record, contract)
    return record


def _insurance_rows(facts, review_confidence=0.85):
    site = facts.insured.coordinates.value
    occupancy = 'unknown'  # Industry text alone does not establish curve applicability.
    for asset in facts.assets:
        def present(fact):
            return fact.value if fact.status == 'provided' and fact.confidence >= review_confidence else None
        building_coordinates = present(asset.coordinates)
        if building_coordinates is not None and not any(asset.name.value and asset.name.value.casefold() in source.excerpt.casefold() and not re.search(r'(?:site|facility).{0,20}(?:GPS|coordinates)', source.excerpt, re.I) for source in asset.coordinates.sources):
            building_coordinates = None
        raw = {'asset_category': 'buildings', 'occupancy': occupancy, 'construction': present(asset.construction_type), 'coordinate_scope': 'building' if building_coordinates else 'site' if site else 'unknown', 'value_scope': 'per_asset', '_extracted_asset': asset.model_dump(mode='json')}
        name = present(asset.name)
        # Names are preserved but are not silently turned into model location IDs.
        raw['building_name'] = name
        raw['loc_id'] = None
        area = present(asset.floor_area)
        raw['floor_area'] = area.value if area else None
        raw['floor_area_unit'] = area.unit if area else None
        money = present(asset.stated_value)
        raw['insured_value'] = money.amount if money else None
        raw['currency'] = money.currency if money else None
        coords = building_coordinates or site
        raw['lat'], raw['lon'] = (coords.latitude, coords.longitude) if coords else (None, None)
        sources = {'construction': asset.construction_type.sources, 'floor_area_m2': asset.floor_area.sources, 'tiv_kes': asset.stated_value.sources, 'lat': asset.coordinates.sources or facts.insured.coordinates.sources, 'lon': asset.coordinates.sources or facts.insured.coordinates.sources, 'building_name': asset.name.sources}
        yield raw, sources, [('uncertain_document_field', field, 'The source field is uncertain, contradicted or below the review confidence threshold') for field, fact in [('construction', asset.construction_type), ('floor_area_m2', asset.floor_area), ('tiv_kes', asset.stated_value), ('coordinates', asset.coordinates)] if fact.status in {'uncertain', 'contradicted'} or (fact.confidence is not None and fact.confidence < review_confidence)]
    for category, field in [('machinery', 'machinery_values'), ('inventory', 'inventory_values'), ('contents', 'contents_values'), ('business_interruption', 'business_interruption_limit')]:
        fact = getattr(facts.financial_exposure, field)
        if fact.status == 'not_provided':
            continue
        money = fact.value if fact.status == 'provided' else None
        raw = {'asset_category': category, 'coordinate_scope': 'site' if site else 'unknown', 'occupancy': occupancy, 'insured_value': money.amount if money else None, 'currency': money.currency if money else None, 'value_scope': 'site_total', 'lat': site.latitude if site else None, 'lon': site.longitude if site else None, '_extracted_financial_fact': fact.model_dump(mode='json')}
        yield raw, {'tiv_kes': fact.sources, 'lat': facts.insured.coordinates.sources, 'lon': facts.insured.coordinates.sources}, []


def validate_exposure_records(records, *, contract=None) -> ExposureMappingResult:
    """Revalidate a batch or list of normalized candidates; invalid rows remain reviewable."""
    batch = records if isinstance(records, ExposureMappingResult) else None
    catalog = contract or (batch.contract if batch else ExposureContract())
    candidates = [record.model_copy(deep=True) for record in ([*batch.valid_records, *batch.review_required_records] if batch else records)]
    classes = {item.name: item for item in catalog.vulnerability_classes}
    counts = Counter(r.normalized_values.get('loc_id') for r in candidates if r.normalized_values.get('loc_id'))
    # Repeated points are review flags, not duplicate assets by themselves.
    points = Counter(point for r in candidates if (point := _point(r)) is not None)
    fingerprints = Counter((r.asset_category, json.dumps(_json_safe(r.original_values), sort_keys=True)) for r in candidates)
    valid, review = [], []
    for record in candidates:
        record.validation_issues = []
        def flag(code, field, message):
            record.validation_issues.append(ExposureIssue(code=code, field=field, message=message))
        if not catalog.verified:
            flag('model_contract_unverified', 'contract', 'The complete model schema and supported classes have not been verified')
        if record.asset_category != 'buildings':
            flag('specialized_model_required', 'asset_category', 'Machinery, inventory, contents and business interruption cannot use building depth-damage curves')
        for field, spec in catalog.fields.items():
            value = record.normalized_values.get(field)
            if value is None or value == '':
                if spec.required:
                    flag('missing_model_input', field, 'Required model input is missing')
                continue
            if spec.kind in {'number', 'integer'}:
                try:
                    value = _number(value)
                    if value is None or (spec.kind == 'integer' and value != value.to_integral_value()) or (spec.minimum is not None and value < spec.minimum) or (spec.maximum is not None and value > spec.maximum):
                        raise ValueError('Outside contract bounds')
                    record.normalized_values[field] = value
                except ValueError:
                    flag('invalid_model_number', field, 'Input is non-finite, malformed or outside contract bounds')
            elif spec.kind == 'enum' and field != 'housing_class' and value not in spec.allowed_values:
                flag('unsupported_enum', field, 'Value is not in the supplied contract enumeration')
        area = record.normalized_values.get('floor_area_m2')
        try:
            if area is not None and _number(area) <= 0:
                flag('invalid_floor_area', 'floor_area_m2', 'Building area must be positive; zero is not treated as missing')
        except ValueError:
            pass
        lat, lon = record.normalized_values.get('lat'), record.normalized_values.get('lon')
        point = _point(record)
        if point is not None and (not -90 <= point[0] <= 90 or not -180 <= point[1] <= 180):
            flag('invalid_coordinates', 'coordinates', 'Coordinates must be finite decimal WGS84 degrees within geographic bounds')
        if (lat is None) != (lon is None):
            flag('incomplete_coordinate_pair', 'coordinates', 'Latitude and longitude must be supplied together')
        if record.coordinate_scope not in {'building', 'assumed_site'}:
            flag('building_coordinates_unknown', 'coordinates', 'Only site-level or unknown coordinates are available; individual building GPS positions were not inferred')
        if record.coordinate_scope == 'assumed_site' and not any(c.approved and c.basis == 'site_coordinate_assumption' and c.field == 'coordinates' for c in record.confirmations):
            flag('site_proxy_unapproved', 'coordinates', 'Use of a site coordinate as a model proxy requires an explicit underwriter assumption')
        housing = record.normalized_values.get('housing_class')
        if housing is not None:
            definition = classes.get(housing)
            if definition is None:
                flag('unsupported_housing_class', 'housing_class', 'Housing class is absent from the supplied verified catalog')
            elif record.asset_category not in definition.asset_categories:
                flag('curve_asset_category_mismatch', 'housing_class', 'Vulnerability class does not support this asset category')
            elif record.occupancy not in definition.occupancies:
                flag('curve_occupancy_unverified', 'housing_class', 'Class applicability to this occupancy is not established')
        if counts[record.normalized_values.get('loc_id')] > 1:
            flag('duplicate_loc_id', 'loc_id', 'Location identifier is duplicated')
        if fingerprints[(record.asset_category, json.dumps(_json_safe(record.original_values), sort_keys=True))] > 1:
            flag('duplicate_record', 'record', 'Identical source values occur more than once')
        if point is not None and points[point] > 1 and record.coordinate_scope != 'assumed_site':
            flag('repeated_coordinates', 'coordinates', 'Several assets share a point; confirm whether these are individual building positions or a site coordinate')
        record.model_ready = not any(item.blocking for item in [*record.mapping_issues, *record.validation_issues])
        (valid if record.model_ready else review).append(record)
    readiness = 'ready' if valid and not review else 'partial' if valid else 'unavailable'
    return ExposureMappingResult(batch_id=batch.batch_id if batch else str(uuid4()), revision=batch.revision if batch else '', contract=catalog, valid_records=valid, review_required_records=review, model_records=[_json_safe({field: r.normalized_values.get(field) for field in catalog.fields}) for r in valid], model_readiness=readiness, source_analysis=batch.source_analysis if batch else None, warnings=batch.warnings if batch else ['Historical/document analysis remains available even when model execution is blocked.'])


def map_to_exposure_schema(data, *, contract=None, column_mapping=None, settings=None, fx_approval=None) -> ExposureMappingResult:
    """Map a table path, row dictionaries, InsuranceFacts or InsuranceDocumentResult."""
    catalog = contract or ExposureContract()
    config = settings or Settings()
    mapping = column_mapping or {}
    candidates, analysis, warnings = [], None, []
    if isinstance(data, (InsuranceDocumentResult, InsuranceFacts)):
        facts = data.facts if isinstance(data, InsuranceDocumentResult) else data
        analysis = data.model_dump(mode='json')
        for raw, sources, issues in _insurance_rows(facts, config.extraction_review_confidence):
            candidate = _normalize(raw, catalog, mapping, sources, fx_approval)
            # No building TIV or coordinate is copied from the facility aggregate.
            candidate.original_values['building_name'] = raw.get('building_name')
            if candidate.coordinate_scope == 'site' and (facts.insured.coordinates.status != 'provided' or facts.insured.coordinates.confidence < config.extraction_review_confidence):
                _issue(candidate, 'site_coordinates_uncertain', 'coordinates', 'Facility coordinates require review')
            for code, field, message in issues:
                _issue(candidate, code, field, message)
            candidates.append(candidate)
    elif isinstance(data, (str, Path)):
        document, rows = _read_table(data, settings or Settings())
        warnings.extend(document.warnings)
        for raw, segment in rows:
            source = SourceReference(document_id=document.document_id, locator=segment.locator, excerpt=segment.text[:500], page_number=segment.page_number)
            candidates.append(_normalize(raw, catalog, mapping, {'*': [source]}, fx_approval))
    else:
        document_id = str(uuid4())
        for index, raw in enumerate(data, 1):
            text = json.dumps(_json_safe(raw), ensure_ascii=False)
            source = SourceReference(document_id=document_id, locator=f'row:{index}', excerpt=text[:500])
            candidates.append(_normalize(raw, catalog, mapping, {'*': [source]}, fx_approval))
    payload = json.dumps({'original': [c.original_values for c in candidates], 'contract': catalog.model_dump(mode='json'), 'mapping': mapping}, sort_keys=True)
    revision = hashlib.sha256(payload.encode()).hexdigest()
    batch = ExposureMappingResult(batch_id=str(uuid4()), revision=revision, contract=catalog, valid_records=[], review_required_records=candidates, model_records=[], model_readiness='unavailable', source_analysis=analysis, warnings=[*warnings, 'Document/history analysis is independent of catastrophe-model readiness. No loss results are calculated.'])
    return validate_exposure_records(batch)


def get_missing_model_inputs(records, *, contract=None) -> list[MissingModelInput]:
    validated = validate_exposure_records(records, contract=contract)
    return [MissingModelInput(record_id=record.record_id, fields=sorted({issue.field for issue in record.validation_issues if issue.code == 'missing_model_input'}), blockers=sorted({issue.code for issue in [*record.mapping_issues, *record.validation_issues] if issue.blocking})) for record in validated.review_required_records]


def confirm_exposure_mapping(batch: ExposureMappingResult, decisions: list[Confirmation]) -> ExposureMappingResult:
    """Apply explicit decisions to a copy; originals and every decision remain auditable."""
    updated = batch.model_copy(deep=True)
    records = {r.record_id: r for r in [*updated.valid_records, *updated.review_required_records]}
    for decision in decisions:
        if not decision.approved or decision.revision != batch.revision or decision.record_id not in records:
            raise ValueError('Confirmation must be explicitly approved and match the current batch revision and record')
        record = records[decision.record_id]
        field = decision.field
        if field == 'coordinates':
            if not isinstance(decision.value, dict) or set(decision.value) != {'lat', 'lon'}:
                raise ValueError('Coordinates require a latitude/longitude pair')
            lat, lon = _number(decision.value['lat']), _number(decision.value['lon'])
            if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError('Confirmed coordinates are invalid')
            if decision.basis == 'site_coordinate_assumption':
                if record.site_coordinates is None or (lat, lon) != (_number(record.site_coordinates['lat']), _number(record.site_coordinates['lon'])):
                    raise ValueError('A site proxy must use the documented site point')
                record.coordinate_scope = 'assumed_site'
            else:
                record.coordinate_scope = 'building'
            record.normalized_values.update(lat=lat, lon=lon)
        elif field == 'asset_category':
            if decision.value not in {'buildings', 'machinery', 'inventory', 'contents', 'business_interruption'}:
                raise ValueError('Unknown asset category')
            if record.asset_category not in {'unclassified', decision.value}:
                raise ValueError('An evidenced non-building asset cannot be reclassified to bypass its model requirements')
            record.asset_category = decision.value
        elif field == 'occupancy':
            record.occupancy = str(decision.value).strip().casefold()
        elif field in {'floor_area_m2', 'tiv_kes'}:
            value = _number(decision.value)
            if value is None or value < 0:
                raise ValueError('Confirmed numerical value is invalid')
            value_unknown = record.original_values.get('tiv_kes') is None or any(issue.code in {'facility_value_not_allocated', 'value_scope_unknown'} for issue in record.mapping_issues)
            if field == 'tiv_kes' and value_unknown and decision.basis != 'explicit_asset_value_assumption':
                raise ValueError('Missing asset values require an explicit asset-value assumption')
            record.normalized_values[field] = value
        else:
            if not isinstance(decision.value, str) or not decision.value.strip():
                raise ValueError('Confirmed text value is invalid')
            record.normalized_values[field] = decision.value.strip()
        record.mapping_issues = [issue for issue in record.mapping_issues if issue.field != field and not (field == 'coordinates' and issue.field in {'lat', 'lon'})]
        record.confirmations.append(decision)
        record.derived_fields.append(DerivedField(field=field, value=decision.value, method='site_coordinate_assumption' if decision.basis == 'site_coordinate_assumption' else 'underwriter', explanation=f'{decision.reason}; evidence: {decision.evidence}; basis: {decision.basis}', approved_by=decision.underwriter_id))
    # New revision prevents stale approvals or concurrent edits silently replacing data.
    updated.revision = hashlib.sha256((batch.revision + json.dumps([d.model_dump(mode='json') for d in decisions], sort_keys=True)).encode()).hexdigest()
    return validate_exposure_records(updated)


async def map_property_description(description: str, *, llm=None, contract=None, settings=None) -> ExposureMappingResult:
    """LLM-backed, evidenced extraction; unsupported model classes are never invented."""
    from .insurance_extraction import InsuranceDocumentExtractor
    from .llm_client import create_llm_client
    from .schemas import DocumentSegment, ParsedDocument
    config = settings or Settings()
    if not description.strip() or len(description) > config.max_document_characters:
        raise DocumentError('Description is empty or exceeds the extraction limit')
    client = llm if llm is not None else create_llm_client(config)
    try:
        parsed = ParsedDocument(document_id=str(uuid4()), segments=[DocumentSegment(locator='description:1', text=description)])
        facts = await InsuranceDocumentExtractor(client, config).extract_document(parsed)
        return map_to_exposure_schema(facts, contract=contract, settings=config)
    finally:
        if llm is None and getattr(client, 'aclose', None):
            await client.aclose()


async def map_insurance_document(file_path, *, llm=None, contract=None, settings=None) -> ExposureMappingResult:
    from .insurance_extraction import aextract_insurance_document
    result = InsuranceDocumentResult.model_validate(await aextract_insurance_document(file_path, llm=llm, settings=settings))
    return map_to_exposure_schema(result, contract=contract, settings=settings)
