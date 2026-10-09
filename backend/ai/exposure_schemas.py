"""Exposure contracts, provenance, review work and explicit underwriting decisions."""
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, model_validator

from .schemas import SourceReference, StrictModel

AssetCategory = Literal['buildings', 'machinery', 'inventory', 'contents', 'business_interruption', 'unclassified']


class ModelField(StrictModel):
    kind: Literal['text', 'number', 'integer', 'boolean', 'enum']
    required: bool = True
    allowed_values: list[str] = Field(default_factory=list)
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    unit: str | None = None


class VulnerabilityClass(StrictModel):
    name: str = Field(min_length=1)
    occupancies: list[str] = Field(min_length=1)
    asset_categories: list[AssetCategory] = Field(min_length=1)
    description: str


class ConstructionRule(StrictModel):
    rule_id: str
    construction: str
    housing_class: str
    rationale: str
    requires_confirmation: bool = True


class ExposureContract(StrictModel):
    """The modelling team owns verification and the supported class catalog.

    The six supplied fields are a provisional interface, not an inferred engine schema.
    """
    version: str = 'unverified'
    verified: bool = False
    verified_by: str | None = None
    fields: dict[str, ModelField] = Field(default_factory=lambda: {
        'loc_id': ModelField(kind='text'),
        'lat': ModelField(kind='number', minimum=-90, maximum=90, unit='degrees'),
        'lon': ModelField(kind='number', minimum=-180, maximum=180, unit='degrees'),
        'housing_class': ModelField(kind='enum'),
        'floor_area_m2': ModelField(kind='number', minimum=0, unit='m2'),
        'tiv_kes': ModelField(kind='number', minimum=0, unit='KES'),
    })
    vulnerability_classes: list[VulnerabilityClass] = Field(default_factory=list)
    construction_rules: list[ConstructionRule] = Field(default_factory=list)

    @model_validator(mode='after')
    def consistent_contract(self):
        required = {'loc_id', 'lat', 'lon', 'housing_class', 'floor_area_m2', 'tiv_kes'}
        if not required.issubset(self.fields):
            raise ValueError('Contract must include the six requested building fields')
        names = [item.name for item in self.vulnerability_classes]
        if len(names) != len(set(names)):
            raise ValueError('Vulnerability class names must be unique')
        if any(rule.housing_class not in names for rule in self.construction_rules):
            raise ValueError('Construction rules must target declared classes')
        if self.verified and (not self.verified_by or not names or self.version == 'unverified'):
            raise ValueError('Verified contracts require an authority, version and class catalog')
        return self


class ContractObservation(StrictModel):
    columns: list[str]
    observed_housing_classes: list[str]
    row_count: int
    warnings: list[str] = Field(default_factory=lambda: ['Observed CSV classes are not an authoritative list of accepted vulnerability classes.'])


class ExposureIssue(StrictModel):
    code: str
    field: str
    message: str
    blocking: bool = True


class DerivedField(StrictModel):
    field: str
    value: Any
    method: Literal['column_alias', 'unit_conversion', 'currency_conversion', 'construction_rule', 'underwriter', 'site_coordinate_assumption']
    explanation: str
    input_fields: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    approved_by: str | None = None


class Confirmation(StrictModel):
    record_id: str
    revision: str
    field: Literal['loc_id', 'coordinates', 'housing_class', 'floor_area_m2', 'tiv_kes', 'occupancy', 'asset_category']
    value: Any
    approved: bool
    underwriter_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    basis: Literal['correction', 'explicit_asset_value_assumption', 'site_coordinate_assumption', 'construction_mapping'] = 'correction'
    evidence: str = Field(min_length=1)


class FXApproval(StrictModel):
    currency: str = Field(pattern=r'^[A-Z]{3}$')
    kes_per_unit: Decimal = Field(gt=0, allow_inf_nan=False)
    effective_date: str
    reference: str = Field(min_length=1)
    approved: bool
    approved_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ExposureCandidate(StrictModel):
    record_id: str
    asset_category: AssetCategory
    original_values: dict[str, Any]
    normalized_values: dict[str, Any] = Field(default_factory=dict)
    site_coordinates: dict[str, float] | None = None
    coordinate_scope: Literal['building', 'site', 'unknown', 'assumed_site'] = 'unknown'
    occupancy: str = 'unknown'
    sources: dict[str, list[SourceReference]] = Field(default_factory=dict)
    derived_fields: list[DerivedField] = Field(default_factory=list)
    confirmations: list[Confirmation] = Field(default_factory=list)
    mapping_issues: list[ExposureIssue] = Field(default_factory=list)
    validation_issues: list[ExposureIssue] = Field(default_factory=list)
    model_ready: bool = False


class ExposureMappingResult(StrictModel):
    batch_id: str
    revision: str
    contract: ExposureContract
    valid_records: list[ExposureCandidate]
    review_required_records: list[ExposureCandidate]
    model_records: list[dict[str, Any]]
    model_readiness: Literal['unavailable', 'partial', 'ready']
    source_analysis: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class MissingModelInput(StrictModel):
    record_id: str
    fields: list[str]
    blockers: list[str]
