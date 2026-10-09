import pytest
from pydantic import ValidationError
from ai.schemas import Asset, Claim, DocumentExtraction, FinancialValue
from ai.validator import validate_extraction


@pytest.mark.parametrize('kwargs', [{'latitude': 1}, {'latitude': 91, 'longitude': 0}, {'latitude': float('nan'), 'longitude': 0}])
def test_invalid_coordinates(kwargs):
    with pytest.raises(ValidationError):
        Asset(asset_id='a', name='A', **kwargs)


@pytest.mark.parametrize('amount,currency', [(-1, 'KES'), (float('inf'), 'KES'), (1, 'kes')])
def test_invalid_financial_values(amount, currency):
    with pytest.raises(ValidationError):
        FinancialValue(amount=amount, currency=currency)


def test_claim_date_must_be_real_date():
    with pytest.raises(ValidationError):
        Claim(claim_id='c', event_date='2026-02-30')


def test_portfolio_findings():
    assets = [Asset(asset_id='a', name='A'), Asset(asset_id='a', name='B', insured_value=FinancialValue(amount=0, currency='KES')), Asset(asset_id='b', name='C', insured_value=FinancialValue(amount=10, currency='USD'))]
    claims = [Claim(claim_id='c', asset_id='unknown'), Claim(claim_id='c')]
    codes = {f.code for f in validate_extraction(DocumentExtraction(assets=assets, claims=claims))}
    assert codes == {'duplicate_asset_id', 'missing_coordinates', 'missing_insured_value', 'zero_insured_value', 'mixed_currencies', 'duplicate_claim_id', 'unknown_claim_asset'}
