"""Pydantic models for privacy evaluation results."""

from typing import Any

from pydantic import BaseModel, Field


class BasicPrivacyResults(BaseModel):
    """Results from basic privacy evaluation methods."""

    copies: dict[str, Any] = Field(
        default_factory=dict, description='Information about exact copies between real and synthetic data'
    )
    duplicates: dict[str, Any] = Field(
        default_factory=dict, description='Information about duplicates within each dataset'
    )
    row_distances: dict[str, float] = Field(
        default_factory=dict, description='Statistical information about row-wise distances'
    )
    error: str | None = Field(default=None, description='Error message if basic evaluation failed')

    class Config:
        extra = 'allow'


class KAnonymityResults(BaseModel):
    """Results from k-anonymity analysis."""

    analysis: dict[str, Any] = Field(default_factory=dict, description='Detailed k-anonymity analysis results')
    recommendations: list[str] = Field(
        default_factory=list, description='Specific recommendations for improving k-anonymity'
    )
    privacy_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description='Privacy score based on k-anonymity metrics (0-1, higher is better)'
    )
    quasi_identifiers_used: list[str] | None = Field(
        default=None, description='List of quasi-identifier columns used in analysis'
    )
    sensitive_attributes_used: list[str] | None = Field(
        default=None, description='List of sensitive attribute columns used in analysis'
    )
    error: str | None = Field(default=None, description='Error message if analysis failed')

    class Config:
        extra = 'allow'


class MembershipInferenceResults(BaseModel):
    """Results from membership inference attack analysis."""

    attack_results: dict[str, Any] = Field(
        default_factory=dict, description='Detailed results from membership inference attacks'
    )
    vulnerability_assessment: dict[str, Any] = Field(
        default_factory=dict, description='Assessment of privacy vulnerability'
    )
    recommendations: list[str] = Field(
        default_factory=list, description='Recommendations for reducing membership inference risk'
    )
    model_performances: list[dict[str, Any]] = Field(
        default_factory=list, description='Performance metrics for different attack models'
    )
    error: str | None = Field(default=None, description='Error message if analysis failed')

    class Config:
        extra = 'allow'


class OverallPrivacyAssessment(BaseModel):
    """Overall privacy risk assessment combining multiple analyses."""

    overall_risk_level: str = Field(default='Low', description='Overall privacy risk level (Low/Moderate/High)')
    identified_risks: list[str] = Field(default_factory=list, description='List of identified privacy risk factors')
    privacy_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description='Combined privacy score (0-1, higher is better)'
    )
    risk_breakdown: dict[str, float] = Field(
        default_factory=dict, description='Breakdown of risk scores from different analyses'
    )
    error: str | None = Field(default=None, description='Error message if assessment failed')

    class Config:
        extra = 'allow'


class PrivacyEvaluationResults(BaseModel):
    """Complete results from privacy evaluation."""

    basic_privacy: BasicPrivacyResults = Field(description='Results from basic privacy checks')
    k_anonymity: KAnonymityResults = Field(description='Results from k-anonymity analysis')
    membership_inference: MembershipInferenceResults = Field(
        description='Results from membership inference attack analysis'
    )
    overall_assessment: OverallPrivacyAssessment = Field(description='Overall privacy risk assessment')
    recommendations: list[str] = Field(
        default_factory=list, description='Combined recommendations from all privacy analyses'
    )

    # Configuration flags
    included_basic: bool = Field(default=True, description='Whether basic privacy analysis was included')
    included_k_anonymity: bool = Field(default=True, description='Whether k-anonymity analysis was included')
    included_membership_inference: bool = Field(
        default=True, description='Whether membership inference analysis was included'
    )

    class Config:
        extra = 'allow'
