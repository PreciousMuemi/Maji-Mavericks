"""Versioned deterministic Nzoia flood model using supplied JRC rasters.

Reference curves are from Huizinga, de Moel & Szewczyk (JRC, 2017), tables
3-1 and 3-10. Housing-class modifiers are deliberately absent because the
supplied material provides no calibrated class-specific factors for Kenya.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import rasterio
from pydantic import Field, model_validator

from .base_schema import StrictModel
from .config import Settings
from .exposure_schemas import ExposureContract, ModelField, VulnerabilityClass
from .risk_analyzer import IntegrationUnavailable
from .underwriting_schemas import BuildingLoss, ModelOutput, PeriodLoss

MODEL_VERSION = 'NZOIA-JRC-AFRICA-2017-v1.0.0'
SUPPORTED_PERIODS = (10, 20, 50, 100, 200, 500)
JRC_SOURCE = 'Huizinga et al. (2017), JRC105688, doi:10.2760/16510'

RESIDENTIAL_CURVE = ((0, 0), (.5, .22), (1, .38), (1.5, .53), (2, .64),
                     (3, .82), (4, .90), (5, .96), (6, 1))
INDUSTRIAL_CURVE = ((0, 0), (.5, .06), (1, .25), (1.5, .40), (2, .49),
                    (3, .68), (4, .92), (5, 1), (6, 1))
HOUSING_CLASSES = ('informal_iron_sheet', 'semi_permanent',
                   'permanent_masonry', 'concrete_rcc')


class FinancialTerms(StrictModel):
    per_risk_deductible_kes: Decimal = Field(default=Decimal('0'), ge=0)
    per_risk_limit_kes: Decimal | None = Field(default=None, ge=0)
    quota_share: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    cat_xol_attachment_kes: Decimal | None = Field(default=None, ge=0)
    cat_xol_limit_kes: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def complete_xol(self):
        if (self.cat_xol_attachment_kes is None) != (self.cat_xol_limit_kes is None):
            raise ValueError('Cat XOL requires both attachment and limit')
        return self


class FinancialBreakdown(StrictModel):
    ground_up: Decimal
    gross: Decimal
    quota_share_ceded: Decimal
    cat_xol_ceded: Decimal
    net: Decimal


def apply_financial_terms(losses: list[Decimal], terms: FinancialTerms) -> FinancialBreakdown:
    ground_up = sum(losses, Decimal('0'))
    insured = []
    for loss in losses:
        payable = max(loss - terms.per_risk_deductible_kes, Decimal('0'))
        if terms.per_risk_limit_kes is not None:
            payable = min(payable, terms.per_risk_limit_kes)
        insured.append(payable)
    gross = sum(insured, Decimal('0'))
    quota = gross * terms.quota_share
    after_quota = gross - quota
    xol = Decimal('0')
    if terms.cat_xol_attachment_kes is not None:
        xol = min(max(after_quota - terms.cat_xol_attachment_kes, Decimal('0')),
                  terms.cat_xol_limit_kes)
    return FinancialBreakdown(ground_up=ground_up, gross=gross,
                              quota_share_ceded=quota, cat_xol_ceded=xol,
                              net=after_quota - xol)


def nzoia_exposure_contract() -> ExposureContract:
    classes = [VulnerabilityClass(
        name=name, occupancies=['unknown', 'residential'], asset_categories=['buildings'],
        description=('Mapped without a class modifier to the JRC 2017 Africa residential '
                     'depth-damage curve; local Kenyan class calibration is unavailable.'))
        for name in HOUSING_CLASSES]
    return ExposureContract(
        version=MODEL_VERSION, verified=True,
        verified_by='Nzoia project dataset guide and JRC105688 reference implementation',
        fields={
            'loc_id': ModelField(kind='text'),
            'lat': ModelField(kind='number', minimum=-90, maximum=90, unit='degrees'),
            'lon': ModelField(kind='number', minimum=-180, maximum=180, unit='degrees'),
            'housing_class': ModelField(kind='enum', allowed_values=list(HOUSING_CLASSES)),
            'floor_area_m2': ModelField(kind='number', minimum=0, unit='m2'),
            'tiv_kes': ModelField(kind='number', minimum=0, unit='KES'),
        }, vulnerability_classes=classes)


def damage_ratio(depth_m: float, *, occupancy: str = 'residential') -> Decimal:
    if not np.isfinite(depth_m) or depth_m <= 0:
        return Decimal('0')
    curve = INDUSTRIAL_CURVE if occupancy == 'industrial' else RESIDENTIAL_CURVE
    return Decimal(str(float(np.interp(depth_m, [p[0] for p in curve],
                                       [p[1] for p in curve]))))


class NzoiaCatModelBackend:
    def __init__(self, store, settings: Settings):
        self.store, self.settings = store, settings
        self._results: dict[str, ModelOutput] = {}
        self._lock = RLock()

    def _raster(self, period: int) -> Path:
        root = Path(self.settings.dataset_directory)
        path = root / 'data' / 'team_b_nzoia' / f'nzoia_rp{period}y.tif'
        if not path.is_file():
            raise IntegrationUnavailable(f'Flood raster is unavailable for RP{period}')
        return path

    def _records(self, assessment_id: str) -> list[dict[str, Any]]:
        mapping = self.store.dashboard_snapshot(assessment_id)['mapping']
        if mapping is None or mapping.model_readiness == 'unavailable' or not mapping.model_records:
            raise IntegrationUnavailable('Reviewed building exposure mapping is required')
        return mapping.model_records

    def _calculate(self, assessment_id: str, periods: list[int], terms: FinancialTerms) -> ModelOutput:
        missing = sorted(set(periods) - set(SUPPORTED_PERIODS))
        if missing:
            return ModelOutput(assessment_id=assessment_id, status='missing_hazard_coverage',
                               missing_return_periods=missing,
                               limitations=['Only the supplied Nzoia return-period rasters are supported.'])
        records = self._records(assessment_id)
        results = []
        audit = []
        for period in periods:
            path = self._raster(period)
            points = [(float(r['lon']), float(r['lat'])) for r in records]
            with rasterio.open(path) as src:
                in_extent = [src.bounds.left <= x <= src.bounds.right and
                             src.bounds.bottom <= y <= src.bounds.top for x, y in points]
                if not all(in_extent):
                    return ModelOutput(assessment_id=assessment_id,
                        status='missing_hazard_coverage', missing_return_periods=[period],
                        limitations=['One or more buildings are outside the supplied raster extent.'])
                sampled = [value[0] for value in src.sample(points, masked=True)]
                nodata = src.nodata
            building_losses, per_risk = [], []
            for record, raw_depth in zip(records, sampled):
                depth = 0.0 if np.ma.is_masked(raw_depth) or (nodata is not None and raw_depth == nodata) else float(raw_depth)
                ratio = damage_ratio(depth)
                loss = Decimal(str(record['tiv_kes'])) * ratio
                payable = max(loss - terms.per_risk_deductible_kes, Decimal('0'))
                if terms.per_risk_limit_kes is not None:
                    payable = min(payable, terms.per_risk_limit_kes)
                per_risk.append(loss)
                building_losses.append(BuildingLoss(loc_id=str(record['loc_id']), loss=payable,
                                                    housing_class=str(record['housing_class'])))
                audit.append((period, record['loc_id'], depth, str(ratio), str(record['tiv_kes'])))
            financial = apply_financial_terms(per_risk, terms)
            results.append(PeriodLoss(return_period=period, currency='KES',
                                      total_loss=financial.gross,
                                      building_losses=building_losses))
        result_id = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()[:20]
        assumptions = [
            f'Vulnerability source: {JRC_SOURCE}.',
            'All four supplied housing classes use the same Africa residential curve; no unsupported class modifier is applied.',
            'Losses cover buildings only; machinery, inventory, contents and business interruption are excluded.',
            'Raster depth is sampled at the supplied building point without local elevation, defence or floor-height adjustment.',
            ('Financial terms applied: ' + terms.model_dump_json()),
        ]
        output = ModelOutput(assessment_id=assessment_id, status='ready',
                             model_version=MODEL_VERSION, result_id=result_id,
                             period_results=results, assumptions=assumptions,
                             limitations=['Regional reference curves are not a locally calibrated Kenyan claims model.'])
        with self._lock:
            self._results[assessment_id] = output
        return output

    async def run_flood_model(self, assessment_id, return_periods):
        return self._calculate(assessment_id, return_periods, FinancialTerms())

    async def get_model_results(self, assessment_id):
        with self._lock:
            result = self._results.get(assessment_id)
        if result is None:
            raise IntegrationUnavailable('No model run is stored for this assessment')
        return result

    async def compare_return_periods(self, assessment_id, periods):
        return self._calculate(assessment_id, periods, FinancialTerms())

    async def run_scenario(self, assessment_id, parameters):
        allowed = set(FinancialTerms.model_fields)
        unknown = set(parameters) - allowed
        if unknown:
            raise IntegrationUnavailable('Scenario contains unsupported parameters: ' + ', '.join(sorted(unknown)))
        return self._calculate(assessment_id, list(SUPPORTED_PERIODS), FinancialTerms.model_validate(parameters))
