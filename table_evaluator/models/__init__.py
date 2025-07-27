"""Pydantic models for table evaluation results."""

# Textual models
# Comprehensive models
from .comprehensive_models import (
    BasicEvaluationResults,
    CombinedEvaluationSummary,
    ComprehensiveEvaluationResults,
)

# ML models
from .ml_models import (
    ClassificationResults,
    ClassificationSummary,
    MLEvaluationResults,
    MLSummary,
    RegressionResults,
    RegressionSummary,
    TargetsEvaluated,
)

# Privacy models
from .privacy_models import (
    AttackModelResult,
    BasicPrivacyAnalysis,
    KAnonymityAnalysis,
    LDiversityAnalysis,
    MembershipInferenceAnalysis,
    PrivacyEvaluationResults,
    PrivacyRiskAssessment,
    StatisticalDistribution,
)

# Statistical models
from .statistical_models import (
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
from .textual_models import (
    ComprehensiveTextualResult,
    LexicalDiversityResult,
    QuickTextualResult,
    SemanticSimilarityResult,
    TextualEvaluationSummary,
    TfidfSimilarityResult,
)

__all__ = [
    # Textual models
    'ComprehensiveTextualResult',
    'LexicalDiversityResult',
    'QuickTextualResult',
    'SemanticSimilarityResult',
    'TextualEvaluationSummary',
    'TfidfSimilarityResult',
    # Privacy models
    'AttackModelResult',
    'BasicPrivacyAnalysis',
    'KAnonymityAnalysis',
    'LDiversityAnalysis',
    'MembershipInferenceAnalysis',
    'PrivacyEvaluationResults',
    'PrivacyRiskAssessment',
    'StatisticalDistribution',
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
    # Comprehensive models
    'BasicEvaluationResults',
    'CombinedEvaluationSummary',
    'ComprehensiveEvaluationResults',
]
