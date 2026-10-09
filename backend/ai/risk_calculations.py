"""Auditable arithmetic over documented claims and actual returned model losses.

No hazard, damage functions, financial categories or catastrophe losses are inferred.
"""
from collections import defaultdict
import calendar
import re
from datetime import date, datetime
from decimal import Decimal

from .intelligence_schemas import Citation, IntelligenceFinding, NumericFinding
from .underwriting_schemas import ModelOutput


def event_identities_disjoint(identities):
    """Partial dates are intervals, so a year and a day in that year may overlap."""
    intervals = []
    for identity in identities:
        if not identity:
            return False
        text = str(identity).strip().casefold()
        if re.search(r'\b(?:before|after|unknown|undated)\b', text):
            return False
        try:
            precise = date.fromisoformat(text)
            interval = (precise, precise)
        except ValueError:
            interval = None
            for pattern in ('%B %Y', '%b %Y', '%d %B %Y', '%d %b %Y'):
                try:
                    parsed = datetime.strptime(text, pattern).date()
                    end = date(parsed.year, parsed.month, calendar.monthrange(parsed.year, parsed.month)[1]) if pattern.startswith('%B') or pattern.startswith('%b') else parsed
                    interval = (parsed, end)
                    break
                except ValueError:
                    pass
            if interval is None:
                years = sorted({int(year) for year in re.findall(r'(?<!\d)(\d{4})(?!\d)', text) if 1 <= int(year) <= 9999})
                if not years:
                    return False
                interval = (date(years[0], 1, 1), date(years[-1], 12, 31))
        if any(interval[0] <= other[1] and other[0] <= interval[1] for other in intervals):
            return False
        intervals.append(interval)
    return True


def event_count(identities):
    return len(identities) if identities and event_identities_disjoint(identities) else None


def largest_historical_amount(records):
    return max(records, key=lambda record: record['amount']) if records else None


def repeated_description_count(event_keys):
    return len(set(event_keys))


def historical_totals(assessment_id, records):
    """Totals stay separate by source stream, financial category and currency.

    Every event must have an evidenced unique event key. Repeated or missing keys
    suppress totals rather than double-count possibly overlapping claims.
    """
    grouped = defaultdict(list)
    for record in records:
        if record['currency'] is not None:
            grouped[(record['stream'], record['kind'], record['currency'])].append(record)
    findings, citations, limitations = [], [], []
    for (stream, kind, currency), rows in grouped.items():
        identities = [row['event_key'] for row in rows]
        if any(identity is None for identity in identities) or len(set(identities)) != len(identities) or (stream == 'document_history' and not event_identities_disjoint(identities)):
            limitations.append(f'{stream}: {kind} amounts have ambiguous or repeated event identities; cumulative totals are withheld.')
            continue
        total = sum((row['amount'] for row in rows), Decimal('0'))
        identifier = f'calc.historical.{stream}.{kind}.{currency}'
        source_ids = sorted({citation for row in rows for citation in row['citation_ids']})
        citations.append(Citation(id=identifier, kind='calculation', assessment_id=assessment_id, formula='sum(amount for uniquely identified events in one source stream, amount category and currency)', function='ai.risk_calculations.historical_totals', inputs=[{'event_key': row['event_key'], 'amount': str(row['amount']), 'currency': currency, 'citation_ids': row['citation_ids']} for row in rows]))
        findings.append(IntelligenceFinding(id=identifier, title=f'Cumulative historical {kind} amounts', statement=f'Historical {kind} amounts in {stream} total {total} {currency}; these are documented past amounts, not future predictions.', evidence_type='historical', citation_ids=[identifier, *source_ids], numerical_findings=[NumericFinding(metric=f'historical_{kind}_total', value=total, unit=currency, citation_id=identifier)]))
    return findings, citations, limitations


def model_concentrations(assessment_id: str, model: ModelOutput):
    """Rank returned buildings; shares use the explicit same-period portfolio total."""
    model = ModelOutput.model_validate_json(model.model_dump_json())
    if model.status != 'ready' or model.assessment_id != assessment_id:
        return [], [], ['No valid model outputs are available for concentration analysis.']
    findings, citations, limitations = [], [], []
    model_citation = f'model.{model.result_id or model.model_version}'
    citations.append(Citation(id=model_citation, kind='model', assessment_id=assessment_id, model_result_id=model.result_id, model_version=model.model_version, inputs=[{'backend_output': model.model_dump(mode='json')}]))
    for period in model.period_results:
        prefix = f'calc.model.{period.return_period}'
        rows = sorted(period.building_losses, key=lambda row: (-row.loss, row.loc_id))
        portfolio = IntelligenceFinding(id=f'{prefix}.portfolio', title='Modelled portfolio loss', statement=f'The actual backend reports {period.total_loss} {period.currency} for the {period.return_period}-year scenario.', evidence_type='modelled', citation_ids=[model_citation], numerical_findings=[NumericFinding(metric='modelled_portfolio_loss', value=period.total_loss, unit=period.currency, citation_id=model_citation)])
        findings.append(portfolio)
        if len({row.loc_id for row in rows}) != len(rows):
            limitations.append(f'{period.return_period}-year output contains repeated building IDs; ranking and concentration are withheld.')
            continue
        subtotal = sum((row.loss for row in rows), Decimal('0'))
        shares_valid = period.total_loss > 0 and subtotal <= period.total_loss
        if not rows:
            limitations.append(f'{period.return_period}-year output has no building breakdown; contributions cannot be ranked.')
        if rows and not shares_valid:
            limitations.append(f'{period.return_period}-year building shares cannot be computed: the total is zero or the breakdown exceeds the reported portfolio total.')
        for rank, row in enumerate(rows, 1):
            identifier = f'{prefix}.building.{rank}'
            inputs = [{'loc_id': row.loc_id, 'loss': str(row.loss), 'portfolio_total': str(period.total_loss), 'currency': period.currency, 'model_citation': model_citation}]
            citations.append(Citation(id=identifier, kind='calculation', assessment_id=assessment_id, model_result_id=model.result_id, model_version=model.model_version, formula='sort returned building losses descending; loss / reported portfolio total * 100 when denominator and breakdown are valid', function='ai.risk_calculations.model_concentrations', inputs=inputs))
            numbers = [NumericFinding(metric='reported_building_loss', value=row.loss, unit=period.currency, citation_id=identifier), NumericFinding(metric='rank_among_returned_buildings', value=rank, unit='rank', citation_id=identifier)]
            statement = f'Among returned buildings, {row.loc_id} ranks {rank} with modelled loss {row.loss} {period.currency} in the {period.return_period}-year scenario.'
            if shares_valid:
                percentage = row.loss / period.total_loss * 100
                numbers.append(NumericFinding(metric='share_of_reported_portfolio_loss', value=percentage, unit='percent', citation_id=identifier))
                statement += f' This is {percentage:.2f}% of the reported portfolio loss.'
            findings.append(IntelligenceFinding(id=identifier, title=f'Modelled building contribution: {row.loc_id}', statement=statement, evidence_type='modelled', citation_ids=[identifier, model_citation], numerical_findings=numbers))
        by_class = defaultdict(list)
        for row in (rows if subtotal <= period.total_loss else []):
            if row.housing_class is not None:
                by_class[row.housing_class].append(row)
        for housing, members in sorted(by_class.items()):
            amount = sum((row.loss for row in members), Decimal('0'))
            identifier = f'{prefix}.class.{len(citations)}'
            citations.append(Citation(id=identifier, kind='calculation', assessment_id=assessment_id, model_result_id=model.result_id, model_version=model.model_version, function='ai.risk_calculations.model_concentrations', formula='sum non-duplicate returned building losses with the same backend-reported housing class', inputs=[{'loc_id': row.loc_id, 'loss': str(row.loss), 'housing_class': housing, 'model_citation': model_citation} for row in members]))
            findings.append(IntelligenceFinding(id=identifier, title=f'Modelled class contribution: {housing}', statement=f'Backend-reported class {housing} contributes {amount} {period.currency} among returned buildings in the {period.return_period}-year scenario.', evidence_type='modelled', citation_ids=[identifier, model_citation], numerical_findings=[NumericFinding(metric='reported_class_loss', value=amount, unit=period.currency, citation_id=identifier)]))
        if rows and not by_class:
            limitations.append(f'{period.return_period}-year housing-class concentration is unavailable without a coherent, class-labelled building breakdown.')
        limitations.append(f'{period.return_period}-year rankings concern returned building rows; completeness of the portfolio breakdown must be verified.')
    for first, second in zip(sorted(model.period_results, key=lambda p: p.return_period), sorted(model.period_results, key=lambda p: p.return_period)[1:]):
        if first.currency != second.currency:
            limitations.append('Model scenarios use different currencies; a numerical comparison is withheld.')
            continue
        identifier = f'calc.compare.{first.return_period}.{second.return_period}'
        difference = second.total_loss - first.total_loss
        citations.append(Citation(id=identifier, kind='calculation', assessment_id=assessment_id, model_result_id=model.result_id, function='ai.risk_calculations.model_concentrations', formula='second scenario reported total - first scenario reported total; no new losses calculated', inputs=[{'return_period': p.return_period, 'total_loss': str(p.total_loss), 'currency': p.currency, 'model_citation': model_citation} for p in (first, second)]))
        findings.append(IntelligenceFinding(id=identifier, title='Modelled scenario comparison', statement=f'The {second.return_period}-year total differs from the {first.return_period}-year total by {difference} {first.currency}.', evidence_type='modelled', citation_ids=[identifier, model_citation], numerical_findings=[NumericFinding(metric='difference_between_reported_scenario_totals', value=difference, unit=first.currency, citation_id=identifier)]))
    return findings, citations, limitations
