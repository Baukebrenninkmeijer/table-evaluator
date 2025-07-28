"""Result models for privacy metrics functions."""

from pydantic import BaseModel, Field

from table_evaluator.models.privacy_models import KAnonymityAnalysis, MembershipInferenceAnalysis

from .error_models import ErrorResult


class OverallPrivacyRiskResult(BaseModel):
    """Model for assess_overall_privacy_risk results."""

    risk_level: str = Field(description='Overall risk level (High/Moderate/Low)')
    risk_factors: list[str] = Field(description='List of identified risk factors')
    recommendations: list[str] = Field(description='Privacy improvement recommendations')
    privacy_score: float = Field(ge=0, le=1, description='Overall privacy score (higher is better)')
    k_anonymity_score: float | None = Field(default=None, description='K-anonymity component score')
    membership_inference_score: float | None = Field(default=None, description='Membership inference component score')
    success: bool = True


class ComprehensivePrivacyAnalysisResult(BaseModel):
    """Model for comprehensive_privacy_analysis results."""

    k_anonymity: KAnonymityAnalysis | ErrorResult | None = Field(description='K-anonymity analysis results')
    membership_inference: MembershipInferenceAnalysis | ErrorResult | None = Field(
        description='Membership inference analysis results'
    )
    overall_assessment: OverallPrivacyRiskResult | ErrorResult = Field(description='Overall privacy assessment')
    success: bool = True
