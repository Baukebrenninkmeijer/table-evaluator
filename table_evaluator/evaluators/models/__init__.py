"""Evaluator result models."""

# Statistical models
# ML models
from table_evaluator.evaluators.models.ml_models import (
    ClassificationResults,
    ClassificationSummary,
    MLEvaluationResults,
    MLSummary,
    RegressionResults,
    RegressionSummary,
    TargetsEvaluated,
)
from table_evaluator.evaluators.models.statistical_models import (
    BasicStatisticalResults,
    BestKernel,
    ColumnDetail,
    CombinedStatisticalMetrics,
    MMDQualityMetrics,
    MMDResults,
    MMDSummary,
    StatisticalEvaluationResults,
    StatisticalSignificance,
    WassersteinOverallMetrics,
    WassersteinQualityMetrics,
    WassersteinResults,
    WassersteinSummary,
)

__all__ = [
    # Statistical models
    'BasicStatisticalResults',
    'BestKernel',
    'ColumnDetail',
    'CombinedStatisticalMetrics',
    'MMDQualityMetrics',
    'MMDResults',
    'MMDSummary',
    'StatisticalEvaluationResults',
    'StatisticalSignificance',
    'WassersteinOverallMetrics',
    'WassersteinQualityMetrics',
    'WassersteinResults',
    'WassersteinSummary',
    # ML models
    'ClassificationResults',
    'ClassificationSummary',
    'MLEvaluationResults',
    'MLSummary',
    'RegressionResults',
    'RegressionSummary',
    'TargetsEvaluated',
]
