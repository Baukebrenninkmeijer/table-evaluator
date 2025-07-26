"""Pydantic models for comprehensive evaluation results."""

from pydantic import BaseModel, Field

from table_evaluator.evaluators.models.ml_models import MLEvaluationResults
from table_evaluator.evaluators.models.statistical_models import StatisticalEvaluationResults
from table_evaluator.evaluators.models.textual_models import TextualEvaluationResults
from table_evaluator.models.privacy_models import PrivacyEvaluationResults


class BasicEvaluationResults(BaseModel):
    """Results from basic table evaluation."""

    basic_statistics: float = Field(description='Basic statistical evaluation score')
    correlation_correlation: float = Field(description='Correlation between correlation matrices')
    mean_correlation: float = Field(description='Mean correlation between fake and real columns')
    ml_efficacy: float = Field(description='ML efficacy score (1-MAPE or correlation)')
    similarity_score: float = Field(description='Overall similarity score')

    # Additional basic metrics
    duplicates_real: int = Field(description='Number of duplicate rows in real data')
    duplicates_fake: int = Field(description='Number of duplicate rows in fake data')
    copies_between_sets: int = Field(description='Number of copied rows between real and fake')
    nearest_neighbor_mean: float = Field(description='Mean nearest neighbor distance')
    nearest_neighbor_std: float = Field(description='Standard deviation of nearest neighbor distances')

    # Privacy metrics
    correlation_distance_rmse: float = Field(description='RMSE correlation distance')
    correlation_distance_mae: float = Field(description='MAE correlation distance')

    class Config:
        extra = 'allow'


class CombinedEvaluationSummary(BaseModel):
    """Summary of combined evaluation results."""

    overall_similarity: float = Field(description='Overall similarity score across all evaluation types')
    tabular_similarity: float | None = Field(default=None, description='Tabular data similarity score')
    textual_similarity: float | None = Field(default=None, description='Textual data similarity score')

    # Weights used in combination
    text_weight: float = Field(default=0.0, description='Weight given to textual metrics')
    tabular_weight: float = Field(default=1.0, description='Weight given to tabular metrics')

    # Quality assessments
    data_quality: str = Field(description='Overall data quality rating')
    privacy_assessment: str = Field(description='Privacy risk assessment')
    recommendations: list[str] = Field(default_factory=list, description='Key recommendations')

    class Config:
        extra = 'allow'


class ComprehensiveEvaluationResults(BaseModel):
    """Complete results from comprehensive table evaluation."""

    # Core evaluation results
    basic: BasicEvaluationResults | None = Field(default=None, description='Basic evaluation results')
    advanced_statistical: StatisticalEvaluationResults | None = Field(
        default=None, description='Advanced statistical evaluation results'
    )
    advanced_privacy: PrivacyEvaluationResults | None = Field(
        default=None, description='Advanced privacy evaluation results'
    )
    ml_evaluation: MLEvaluationResults | None = Field(default=None, description='Machine learning evaluation results')
    textual: TextualEvaluationResults | None = Field(default=None, description='Textual evaluation results')

    # Combined analysis
    combined_similarity: CombinedEvaluationSummary | None = Field(
        default=None, description='Combined similarity analysis'
    )

    # Metadata
    evaluation_config: dict = Field(default_factory=dict, description='Configuration used for evaluation')
    target_column: str | None = Field(default=None, description='Target column used for ML evaluation')
    target_type: str | None = Field(default=None, description='Type of ML task (class/regr)')

    # Error tracking
    evaluation_errors: dict[str, str] = Field(default_factory=dict, description='Errors encountered during evaluation')

    class Config:
        extra = 'allow'
