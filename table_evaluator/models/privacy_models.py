"""Privacy evaluation result models using clean, strongly-typed Pydantic models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pendulum
from pydantic import BaseModel, Field


class AnalysisMetadata(BaseModel):
    """Metadata for privacy analysis results."""

    analysis_timestamp: datetime = Field(default_factory=lambda: pendulum.now('Europe/Paris'))
    library_version: str = Field(default='1.9.0')
    analysis_duration_seconds: float | None = None


class BaseStatisticalResult(BaseModel):
    """Base class for statistical analysis results with common metadata."""

    analysis_timestamp: datetime = Field(default_factory=lambda: pendulum.now('Europe/Paris'))
    sample_size: int = Field(ge=0, description='Number of samples analyzed')
    computation_method: str = Field(default='standard', description='Method used for computation')


class StatisticalDistribution(BaseModel):
    """Statistical distribution summary from pandas describe()."""

    count: float
    mean: float
    std: float
    min: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    max: float

    class Config:
        populate_by_name = True


class LDiversityAnalysis(BaseModel):
    """L-diversity analysis for a single sensitive attribute."""

    min_l_value: int = Field(ge=0)
    avg_l_value: float = Field(ge=0.0)
    l_violations: int = Field(ge=0)
    diversity_distribution: StatisticalDistribution


class KAnonymityAnalysis(BaseStatisticalResult):
    """Core k-anonymity analysis results."""

    k_value: int = Field(ge=0)
    anonymity_level: Literal['Good', 'Moderate', 'Weak', 'Poor', 'Error', 'Perfect']
    violations: int = Field(ge=0)
    equivalence_classes: int = Field(ge=0)
    avg_class_size: float = Field(ge=0.0)
    class_size_distribution: StatisticalDistribution
    l_diversity_results: dict[str, LDiversityAnalysis] = Field(default_factory=dict)


class AttackModelResult(BaseModel):
    """Results from a single attack model."""

    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    auc_score: float = Field(ge=0.0, le=1.0)
    privacy_risk: Literal['High', 'Medium', 'Low']
    baseline_accuracy: float = Field(default=0.5)


class MembershipInferenceAnalysis(BaseStatisticalResult):
    """Complete membership inference attack analysis."""

    # Individual model results (optional for detailed analysis)
    logistic_regression: AttackModelResult | None = None
    random_forest: AttackModelResult | None = None

    # Summary results (always present)
    max_attack_accuracy: float = Field(ge=0.0, le=1.0)
    avg_attack_accuracy: float = Field(ge=0.0, le=1.0)
    privacy_vulnerability: Literal['High', 'Medium', 'Low']
    recommendation: str


class BasicPrivacyAnalysis(BaseModel):
    """Basic privacy metrics."""

    exact_copies_count: int = Field(ge=0)
    exact_copies_fraction: float = Field(ge=0.0, le=1.0)
    real_duplicates_count: int = Field(ge=0)
    synthetic_duplicates_count: int = Field(ge=0)
    mean_row_distance: float = Field(ge=0.0)
    std_row_distance: float = Field(ge=0.0)


class PrivacyRiskAssessment(BaseModel):
    """Overall privacy risk assessment."""

    risk_level: Literal['Low', 'Moderate', 'High']
    privacy_score: float = Field(ge=0.0, le=1.0)
    identified_risks: list[str] = Field(default_factory=list)
    k_anonymity_contribution: float = Field(ge=0.0, le=1.0)
    inference_risk_contribution: float = Field(ge=0.0, le=1.0)


class PrivacyEvaluationResults(BaseModel):
    """Complete privacy evaluation results with clean, strongly-typed structure."""

    # Analysis metadata
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)

    # Core analysis results
    basic_privacy: BasicPrivacyAnalysis
    k_anonymity: KAnonymityAnalysis | None = None
    membership_inference: MembershipInferenceAnalysis | None = None
    overall_assessment: PrivacyRiskAssessment

    # Additional context
    recommendations: list[str] = Field(default_factory=list)
    quasi_identifiers_used: list[str] | None = None
    sensitive_attributes_used: list[str] | None = None
    analysis_errors: list[str] = Field(default_factory=list)

    # Configuration flags
    included_basic: bool = True
    included_k_anonymity: bool = False
    included_membership_inference: bool = False
