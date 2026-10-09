"""Field-level insurance facts. Missing, zero, uncertainty and conflict are distinct."""
import json
from decimal import Decimal
from typing import Generic, Literal, TypeVar

from pydantic import Field, model_validator

from .schemas import SourceReference, StrictModel

T = TypeVar('T')


class Interpretation(StrictModel, Generic[T]):
    value: T
    sources: list[SourceReference] = Field(min_length=1)


class Fact(StrictModel, Generic[T]):
    value: T | None = None
    status: Literal['not_provided', 'provided', 'uncertain', 'contradicted'] = 'not_provided'
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    sources: list[SourceReference] = Field(default_factory=list)
    alternatives: list[Interpretation[T]] = Field(default_factory=list)
    review_reason: str | None = None

    @model_validator(mode='after')
    def consistent_state(self):
        if self.status == 'not_provided':
            if self.value is not None or self.sources or self.alternatives or self.confidence is not None:
                raise ValueError('Not-provided facts cannot carry values, evidence or confidence')
        elif self.status == 'contradicted':
            if self.value is not None or len(self.alternatives) < 2 or not self.review_reason:
                raise ValueError('Contradictions require competing interpretations and review reason')
            values = [json.dumps(a.value.model_dump(mode='json') if isinstance(a.value, StrictModel) else a.value, sort_keys=True) for a in self.alternatives]
            if len(set(values)) < 2:
                raise ValueError('Contradictions require distinct values')
        else:
            if not self.sources or self.confidence is None or self.alternatives:
                raise ValueError('Present facts require evidence and confidence')
            if self.status == 'provided' and self.value is None:
                raise ValueError('Provided facts require a value')
            if isinstance(self.value, str) and not self.value.strip():
                raise ValueError('Provided text facts cannot be empty')
            if self.status == 'uncertain' and not self.review_reason:
                raise ValueError('Uncertain facts require a review reason')
        return self


class Money(StrictModel):
    amount: Decimal = Field(ge=0, allow_inf_nan=False)
    currency: str | None = Field(default=None, pattern=r'^[A-Z]{3}$')


class Measurement(StrictModel):
    value: Decimal | None = Field(default=None, allow_inf_nan=False)
    lower: Decimal | None = Field(default=None, allow_inf_nan=False)
    upper: Decimal | None = Field(default=None, allow_inf_nan=False)
    unit: str = Field(min_length=1, max_length=50)

    @model_validator(mode='after')
    def exact_or_range(self):
        if self.value is not None:
            if self.lower is not None or self.upper is not None:
                raise ValueError('A measurement is an exact value or a range')
        elif self.lower is None or self.upper is None or self.lower > self.upper:
            raise ValueError('A range requires ordered lower and upper bounds')
        return self


class Coordinates(StrictModel):
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)


TextFact = Fact[str]
MoneyFact = Fact[Money]
MeasureFact = Fact[Measurement]


class InsuredDetails(StrictModel):
    client_name: TextFact = Field(default_factory=TextFact)
    facility_location: TextFact = Field(default_factory=TextFact)
    industry: TextFact = Field(default_factory=TextFact)
    coordinates: Fact[Coordinates] = Field(default_factory=Fact[Coordinates])


class InsuranceAsset(StrictModel):
    name: TextFact  # Keep the documented name; do not generate building IDs.
    construction_type: TextFact = Field(default_factory=TextFact)
    floor_area: MeasureFact = Field(default_factory=MeasureFact)
    condition: TextFact = Field(default_factory=TextFact)
    elevation: MeasureFact = Field(default_factory=MeasureFact)
    elevation_datum: TextFact = Field(default_factory=TextFact)
    stated_value: MoneyFact = Field(default_factory=MoneyFact)
    coordinates: Fact[Coordinates] = Field(default_factory=Fact[Coordinates])

    @model_validator(mode='after')
    def identified(self):
        if self.name.status == 'not_provided':
            raise ValueError('An asset requires a documented identity')
        return self


FinancialCategory = Literal['property_sum_insured', 'machinery_values', 'inventory_values', 'contents_values', 'business_interruption_limit']


class FinancialRelationship(StrictModel):
    category: FinancialCategory
    other_category: FinancialCategory
    relationship: Literal['included_in', 'disjoint_from', 'overlaps_with', 'unknown']
    evidence: TextFact

    @model_validator(mode='after')
    def distinct_categories(self):
        if self.category == self.other_category or self.evidence.status == 'not_provided':
            raise ValueError('Relationships require distinct categories and source evidence')
        return self


class InventoryItem(StrictModel):
    name: TextFact
    stated_value: MoneyFact


class FinancialExposure(StrictModel):
    property_sum_insured: MoneyFact = Field(default_factory=MoneyFact)
    machinery_values: MoneyFact = Field(default_factory=MoneyFact)
    inventory_values: MoneyFact = Field(default_factory=MoneyFact)
    contents_values: MoneyFact = Field(default_factory=MoneyFact)
    business_interruption_limit: MoneyFact = Field(default_factory=MoneyFact)
    currency: TextFact = Field(default_factory=TextFact)
    relationships: list[FinancialRelationship] = Field(default_factory=list)
    inventory_breakdown: list[InventoryItem] = Field(default_factory=list)
    inventory_breakdown_disjoint: Fact[bool] = Field(default_factory=Fact[bool])
    # No aggregate/total field: categories may overlap and are never added here.


class InsuranceTerms(StrictModel):
    coverage_requested: TextFact = Field(default_factory=TextFact)
    deductibles: TextFact = Field(default_factory=TextFact)
    limits: TextFact = Field(default_factory=TextFact)
    requested_flood_limit: MoneyFact = Field(default_factory=MoneyFact)
    retention: TextFact = Field(default_factory=TextFact)
    reinsurance_participation: TextFact = Field(default_factory=TextFact)
    premiums: MoneyFact = Field(default_factory=MoneyFact)


class FloodEvent(StrictModel):
    event_date: TextFact  # Preserve partial dates, e.g. "April 2022", verbatim.
    reported_depth: MeasureFact = Field(default_factory=MeasureFact)
    historical_claim_value: MoneyFact = Field(default_factory=MoneyFact)
    settlement_value: MoneyFact = Field(default_factory=MoneyFact)
    business_interruption_duration: MeasureFact = Field(default_factory=MeasureFact)
    description: TextFact = Field(default_factory=TextFact)

    @model_validator(mode='after')
    def evidenced_event(self):
        if not any(getattr(self, name).status != 'not_provided' for name in type(self).model_fields):
            raise ValueError('A flood event requires source evidence')
        return self


class RiskFactors(StrictModel):
    drainage_weaknesses: list[TextFact] = Field(default_factory=list)
    building_deterioration: list[TextFact] = Field(default_factory=list)
    flood_defenses: list[TextFact] = Field(default_factory=list)
    previous_damage: list[TextFact] = Field(default_factory=list)
    operational_vulnerabilities: list[TextFact] = Field(default_factory=list)


class DocumentRecommendation(StrictModel):
    author_role: Literal['broker', 'insurer', 'surveyor', 'insured', 'unknown']
    recommendation: TextFact

    @model_validator(mode='after')
    def evidenced_recommendation(self):
        if self.recommendation.status == 'not_provided':
            raise ValueError('Recommendations require document evidence')
        return self


class InsuranceFacts(StrictModel):
    insured: InsuredDetails = Field(default_factory=InsuredDetails)
    assets: list[InsuranceAsset] = Field(default_factory=list)
    financial_exposure: FinancialExposure = Field(default_factory=FinancialExposure)
    insurance_terms: InsuranceTerms = Field(default_factory=InsuranceTerms)
    flood_history: list[FloodEvent] = Field(default_factory=list)
    risk_factors: RiskFactors = Field(default_factory=RiskFactors)
    document_recommendations: list[DocumentRecommendation] = Field(default_factory=list)


class InsuranceIdentityAssets(StrictModel):
    insured: InsuredDetails = Field(default_factory=InsuredDetails)
    assets: list[InsuranceAsset] = Field(default_factory=list)


class InsuranceFinancialTerms(StrictModel):
    financial_exposure: FinancialExposure = Field(default_factory=FinancialExposure)
    insurance_terms: InsuranceTerms = Field(default_factory=InsuranceTerms)


class InsuranceHistoryRisk(StrictModel):
    flood_history: list[FloodEvent] = Field(default_factory=list)
    risk_factors: RiskFactors = Field(default_factory=RiskFactors)
    document_recommendations: list[DocumentRecommendation] = Field(default_factory=list)


class DocumentClassification(StrictModel):
    kind: Literal['insurance_offer', 'reinsurance_offer', 'exposure_schedule', 'policy', 'claims_report', 'mixed', 'unknown']
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    sources: list[SourceReference] = Field(default_factory=list)
    reason: str

    # The extraction service attaches and validates an exact parser-controlled
    # heading when a provider correctly classifies the document but omits citations.


class ReviewItem(StrictModel):
    code: str
    field_path: str
    message: str
    sources: list[SourceReference] = Field(default_factory=list)


class Contradiction(StrictModel):
    field_path: str
    message: str
    sources: list[SourceReference] = Field(min_length=2)


class ContradictionAudit(StrictModel):
    contradictions: list[Contradiction] = Field(default_factory=list)


class InsuranceDocumentResult(StrictModel):
    document_id: str
    document_type: DocumentClassification
    facts: InsuranceFacts
    review_status: Literal['ready', 'requires_review']
    review_items: list[ReviewItem] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
