"""Pydantic models for table evaluation results."""

# Error models
# Comprehensive models
from .comprehensive_models import (
    BasicEvaluationResults,
    CombinedEvaluationSummary,
    ComprehensiveEvaluationResults,
)
from .error_models import ErrorResult, create_error_result

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

# ML result models
from .ml_result_models import (
    MLUtilityEvaluationResult,
    MLUtilityEvaluationSummary,
    SingleModelEvaluationResult,
    TrainTestSyntheticResult,
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

# Privacy result models
from .privacy_result_models import (
    ComprehensivePrivacyAnalysisResult,
    OverallPrivacyRiskResult,
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

# Statistical result models
from .statistical_result_models import (
    JensenShannonResult,
    MMDAnalysisResult,
    MultivariateMMDResult,
    WassersteinDistanceResult,
    WassersteinDistanceSummaryResult,
)
from .textual_models import (
    BasicTextualEvaluationResults,
    ComprehensiveEvaluationWithTextResults,
    ComprehensiveTextualAnalysisResult,
    ComprehensiveTextualResult,
    LexicalDiversityResult,
    QuickTextualResult,
    SemanticSimilarityResult,
    TextualEvaluationResults,
    TextualEvaluationSummary,
    TfidfSimilarityResult,
)

__all__ = [
    # Privacy models
    'AttackModelResult',
    # Comprehensive models
    'BasicEvaluationResults',
    'BasicPrivacyAnalysis',
    # Statistical models
    'BasicStatisticalResults',
    # Textual models
    'BasicTextualEvaluationResults',
    'BestKernel',
    # ML models
    'ClassificationResults',
    'ClassificationSummary',
    'ColumnDetail',
    'CombinedEvaluationSummary',
    'CombinedStatisticalMetrics',
    'ComprehensiveEvaluationResults',
    'ComprehensiveEvaluationWithTextResults',
    # Privacy result models
    'ComprehensivePrivacyAnalysisResult',
    'ComprehensiveTextualAnalysisResult',
    'ComprehensiveTextualResult',
    # Error models
    'ErrorResult',
    # Statistical result models
    'JensenShannonResult',
    'KAnonymityAnalysis',
    'LDiversityAnalysis',
    'LexicalDiversityResult',
    'MLEvaluationResults',
    'MLSummary',
    # ML result models
    'MLUtilityEvaluationResult',
    'MLUtilityEvaluationSummary',
    'MMDAnalysisResult',
    'MMDQualityMetrics',
    'MMDResults',
    'MMDSummary',
    'MembershipInferenceAnalysis',
    'MultivariateMMDResult',
    'OverallPrivacyRiskResult',
    'PrivacyEvaluationResults',
    'PrivacyRiskAssessment',
    'QuickTextualResult',
    'RegressionResults',
    'RegressionSummary',
    'SemanticSimilarityResult',
    'SingleModelEvaluationResult',
    'StatisticalDistribution',
    'StatisticalEvaluationResults',
    'StatisticalSignificance',
    'TargetsEvaluated',
    'TextualEvaluationResults',
    'TextualEvaluationSummary',
    'TfidfSimilarityResult',
    'TrainTestSyntheticResult',
    'WassersteinDistanceResult',
    'WassersteinDistanceSummaryResult',
    'WassersteinOverallMetrics',
    'WassersteinQualityMetrics',
    'WassersteinResults',
    'WassersteinSummary',
    'create_error_result',
]
