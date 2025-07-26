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


class WassersteinSummary(BaseModel):
    """Summary analysis of Wasserstein distances."""

    overall_metrics: 'WassersteinOverallMetrics' = Field(
        default_factory=lambda: WassersteinOverallMetrics(), description='Overall Wasserstein metrics'
    )
    best_column: str | None = Field(default=None, description='Column with best (lowest) Wasserstein distance')
    worst_column: str | None = Field(default=None, description='Column with worst (highest) Wasserstein distance')

    class Config:
        extra = 'allow'


class WassersteinOverallMetrics(BaseModel):
    """Overall metrics from Wasserstein distance analysis."""

    mean_distance: float = Field(default=0.0, description='Mean Wasserstein distance across all columns')
    median_distance: float = Field(default=0.0, description='Median Wasserstein distance')
    max_distance: float = Field(default=0.0, description='Maximum Wasserstein distance')
    min_distance: float = Field(default=0.0, description='Minimum Wasserstein distance')

    class Config:
        extra = 'allow'


class ColumnDetail(BaseModel):
    """Per-column detailed analysis including transport cost."""

    transport_cost: float = Field(default=0.0, description='Optimal transport cost for this column')
    transport_plan_shape: tuple[int, int] = Field(default=(0, 0), description='Shape of the transport plan matrix')
    n_bins: int = Field(default=0, description='Number of bins used in transport calculation')
    error: str | None = Field(default=None, description='Error message if calculation failed')

    class Config:
        extra = 'allow'


class WassersteinQualityMetrics(BaseModel):
    """Overall quality metrics derived from Wasserstein distances."""

    mean_wasserstein_p1: float = Field(default=0.0, description='Mean Wasserstein distance of order 1')
    median_wasserstein_p1: float = Field(default=0.0, description='Median Wasserstein distance of order 1')
    std_wasserstein_p1: float = Field(default=0.0, description='Standard deviation of Wasserstein distances')
    max_wasserstein_p1: float = Field(default=0.0, description='Maximum Wasserstein distance')
    distribution_similarity_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description='Distribution similarity score'
    )
    quality_rating: str = Field(default='Unknown', description='Quality rating based on Wasserstein distances')

    class Config:
        extra = 'allow'


class WassersteinResults(BaseModel):
    """Results from Wasserstein distance evaluation."""

    distances_p1: Any = Field(description='Wasserstein distances of order 1 (DataFrame, dict, or list)')
    distances_p2: Any = Field(description='Wasserstein distances of order 2 (DataFrame, dict, or list)')
    summary: WassersteinSummary = Field(
        default_factory=WassersteinSummary, description='Summary analysis of Wasserstein distances'
    )
    distances_2d: Any = Field(default=None, description='2D Wasserstein distances for column pairs (if computed)')
    column_details: dict[str, ColumnDetail] = Field(
        default_factory=dict, description='Per-column detailed analysis including transport cost'
    )
    quality_metrics: WassersteinQualityMetrics = Field(
        default_factory=WassersteinQualityMetrics,
        description='Overall quality metrics derived from Wasserstein distances',
    )
    error: str | None = Field(default=None, description='Error message if evaluation failed')

    class Config:
        extra = 'allow'


class MMDSummary(BaseModel):
    """Summary statistics from MMD analysis."""

    mean_mmd: float = Field(default=0.0, description='Mean MMD value across all columns and kernels')
    fraction_significant: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Fraction of columns with significant differences'
    )
    overall_quality_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description='Overall quality score from MMD analysis'
    )
    total_comparisons: int = Field(default=0, description='Total number of MMD comparisons performed')

    class Config:
        extra = 'allow'


class MMDQualityMetrics(BaseModel):
    """Quality metrics derived from MMD results."""

    mean_mmd: float = Field(default=0.0, description='Mean MMD value')
    fraction_significant_differences: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Fraction of significant differences'
    )
    overall_quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description='Overall quality score')
    mmd_rating: str = Field(default='Unknown', description='Quality rating based on MMD values')

    class Config:
        extra = 'allow'


class BestKernel(BaseModel):
    """Information about the best performing kernel."""

    kernel: str = Field(default='', description='Name of the best kernel')
    mmd_squared: float = Field(default=0.0, description='MMD squared value for the best kernel')
    discriminative_power: str = Field(default='Low', description='Discriminative power level')

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
    summary: MMDSummary | None = Field(default=None, description='Summary statistics from MMD analysis')
    quality_metrics: MMDQualityMetrics | None = Field(
        default=None, description='Quality metrics derived from MMD results'
    )
    best_kernel: BestKernel | None = Field(default=None, description='Information about the best performing kernel')
    error: str | None = Field(default=None, description='Error message if evaluation failed')

    class Config:
        extra = 'allow'


class StatisticalSignificance(BaseModel):
    """Statistical significance information from MMD tests."""

    fraction_columns_different: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Fraction of columns showing significant differences'
    )
    interpretation: str = Field(default='Low', description='Interpretation of statistical significance level')

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
    statistical_significance: StatisticalSignificance = Field(
        default_factory=StatisticalSignificance, description='Statistical significance information from MMD tests'
    )
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


# Update forward references
WassersteinSummary.model_rebuild()
