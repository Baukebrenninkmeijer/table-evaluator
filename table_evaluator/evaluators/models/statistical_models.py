"""Pydantic models for statistical evaluation results."""

from typing import Any

from pydantic import BaseModel, Field


class BasicStatisticalResults(BaseModel):
    """Results from basic statistical evaluation methods."""

    basic_statistical_evaluation: float | None = Field(
        description='Correlation coefficient between basic statistical properties'
    )
    correlation_correlation: float | None = Field(description='Correlation coefficient between association matrices')
    pca_correlation: float | None = Field(description='Correlation coefficient between PCA explained variances')
    error: str | None = Field(default=None, description='Error message if basic evaluation failed')

    class Config:
        extra = 'allow'


class WassersteinResults(BaseModel):
    """Results from Wasserstein distance evaluation."""

    distances_p1: Any = Field(description='Wasserstein distances of order 1 (DataFrame, dict, or list)')
    distances_p2: Any = Field(description='Wasserstein distances of order 2 (DataFrame, dict, or list)')
    summary: dict[str, Any] = Field(description='Summary analysis of Wasserstein distances')
    distances_2d: dict[str, Any] | None = Field(
        default=None, description='2D Wasserstein distances for column pairs (if computed)'
    )
    column_details: dict[str, dict[str, Any]] = Field(
        description='Per-column detailed analysis including transport cost'
    )
    quality_metrics: dict[str, Any] = Field(description='Overall quality metrics derived from Wasserstein distances')
    error: str | None = Field(default=None, description='Error message if evaluation failed')

    class Config:
        extra = 'allow'


class MMDResults(BaseModel):
    """Results from Maximum Mean Discrepancy evaluation."""

    column_wise: Any = Field(
        default=None, description='Column-wise MMD results for different kernels (dict or DataFrame)'
    )
    multivariate: Any = Field(
        default=None, description='Multivariate MMD results for different kernels (dict or DataFrame)'
    )
    summary: dict[str, Any] | None = Field(default=None, description='Summary statistics from MMD analysis')
    quality_metrics: dict[str, Any] | None = Field(default=None, description='Quality metrics derived from MMD results')
    best_kernel: dict[str, Any] | None = Field(default=None, description='Information about the best performing kernel')
    error: str | None = Field(default=None, description='Error message if evaluation failed')

    class Config:
        extra = 'allow'


class CombinedStatisticalMetrics(BaseModel):
    """Combined metrics from Wasserstein and MMD analyses."""

    overall_similarity: float = Field(
        ge=0.0, le=1.0, description='Overall distribution similarity score (0-1, higher is better)'
    )
    quality_consensus: str = Field(description='Agreement between Wasserstein and MMD quality ratings')
    wasserstein_rating: str = Field(description='Quality rating from Wasserstein analysis')
    mmd_rating: str = Field(description='Quality rating from MMD analysis')
    statistical_significance: dict[str, Any] = Field(description='Statistical significance information from MMD tests')
    error: str | None = Field(default=None, description='Error message if combined analysis failed')

    class Config:
        extra = 'allow'


class StatisticalEvaluationResults(BaseModel):
    """Complete results from statistical evaluation."""

    basic_statistics: BasicStatisticalResults = Field(description='Results from basic statistical methods')
    wasserstein: WassersteinResults = Field(description='Results from Wasserstein distance analysis')
    mmd: MMDResults = Field(description='Results from Maximum Mean Discrepancy analysis')
    combined_metrics: CombinedStatisticalMetrics = Field(
        description='Combined metrics from advanced statistical methods'
    )
    recommendations: list[str] = Field(
        default_factory=list, description='Actionable recommendations based on statistical analysis'
    )

    # Configuration information
    numerical_columns: list[str] = Field(description='List of numerical columns used in analysis')
    categorical_columns: list[str] | None = Field(
        default=None, description='List of categorical columns used in analysis'
    )
    included_basic: bool = Field(default=True, description='Whether basic statistical analysis was included')
    included_wasserstein: bool = Field(default=True, description='Whether Wasserstein analysis was included')
    included_mmd: bool = Field(default=True, description='Whether MMD analysis was included')

    class Config:
        extra = 'allow'
