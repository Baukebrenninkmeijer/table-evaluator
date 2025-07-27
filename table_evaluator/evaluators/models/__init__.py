"""Evaluator result models."""

from table_evaluator.evaluators.models.comprehensive_models import (
    BasicEvaluationResults,
    CombinedEvaluationSummary,
    ComprehensiveEvaluationResults,
)

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
from table_evaluator.evaluators.models.textual_models import (
    CombinedTextualMetrics,
    LexicalDiversityResults,
    SemanticSimilarityResults,
    TextualEvaluationResults,
    TfidfSimilarityResults,
)

__all__ = [
    'BasicEvaluationResults',
    'BasicStatisticalResults',
    'BestKernel',
    'ClassificationResults',
    'ClassificationSummary',
    'ColumnDetail',
    'CombinedEvaluationSummary',
    'CombinedStatisticalMetrics',
    'CombinedTextualMetrics',
    'ComprehensiveEvaluationResults',
    'LexicalDiversityResults',
    'MLEvaluationResults',
    'MLSummary',
    'MMDQualityMetrics',
    'MMDResults',
    'MMDSummary',
    'RegressionResults',
    'RegressionSummary',
    'SemanticSimilarityResults',
    'StatisticalEvaluationResults',
    'StatisticalSignificance',
    'TargetsEvaluated',
    'TextualEvaluationResults',
    'TfidfSimilarityResults',
    'WassersteinOverallMetrics',
    'WassersteinQualityMetrics',
    'WassersteinResults',
    'WassersteinSummary',
]
