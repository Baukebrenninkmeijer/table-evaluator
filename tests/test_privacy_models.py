"""Test validation and edge cases for privacy Pydantic models."""

from datetime import datetime

import pytest
from pydantic import ValidationError
from table_evaluator.models.privacy_models import (
    AnalysisMetadata,
    AttackModelResult,
    BaseStatisticalResult,
    BasicPrivacyAnalysis,
    KAnonymityAnalysis,
    LDiversityAnalysis,
    MembershipInferenceAnalysis,
    PrivacyEvaluationResults,
    PrivacyRiskAssessment,
    StatisticalDistribution,
)


class TestAnalysisMetadata:
    """Test AnalysisMetadata model validation."""

    def test_default_metadata(self):
        """Test default metadata creation."""
        metadata = AnalysisMetadata()
        assert metadata.library_version == '1.9.0'
        assert isinstance(metadata.analysis_timestamp, datetime)
        assert metadata.analysis_duration_seconds is None

    def test_custom_metadata(self):
        """Test custom metadata values."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)  # noqa: DTZ001
        metadata = AnalysisMetadata(
            analysis_timestamp=custom_time,
            library_version='2.0.0',
            analysis_duration_seconds=15.5,
        )
        assert metadata.analysis_timestamp == custom_time
        assert metadata.library_version == '2.0.0'
        assert metadata.analysis_duration_seconds == 15.5


class TestBaseStatisticalResult:
    """Test BaseStatisticalResult model validation."""

    def test_default_base_result(self):
        """Test default base statistical result creation."""
        result = BaseStatisticalResult(sample_size=100)
        assert result.sample_size == 100
        assert isinstance(result.analysis_timestamp, datetime)
        assert result.computation_method == 'standard'

    def test_custom_base_result(self):
        """Test custom base statistical result values."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)  # noqa: DTZ001
        result = BaseStatisticalResult(
            analysis_timestamp=custom_time,
            sample_size=500,
            computation_method='optimized',
        )
        assert result.analysis_timestamp == custom_time
        assert result.sample_size == 500
        assert result.computation_method == 'optimized'

    def test_negative_sample_size_fails(self):
        """Test that negative sample_size raises ValidationError."""
        with pytest.raises(ValidationError, match='greater than or equal to 0'):
            BaseStatisticalResult(sample_size=-1)


class TestStatisticalDistribution:
    """Test StatisticalDistribution model validation."""

    def test_valid_distribution(self):
        """Test valid statistical distribution creation."""
        dist = StatisticalDistribution(
            count=100.0,
            mean=50.0,
            std=15.0,
            min=0.0,
            percentile_25=35.0,
            percentile_50=50.0,
            percentile_75=65.0,
            max=100.0,
        )
        assert dist.count == 100.0
        assert dist.mean == 50.0

    def test_negative_count(self):
        """Test that negative count is accepted (pandas can return negative in edge cases)."""
        # Note: We don't constrain count to be >= 0 as pandas describe() can return negative values
        dist = StatisticalDistribution(
            count=-1.0,
            mean=0.0,
            std=0.0,
            min=0.0,
            percentile_25=0.0,
            percentile_50=0.0,
            percentile_75=0.0,
            max=0.0,
        )
        assert dist.count == -1.0


class TestLDiversityAnalysis:
    """Test LDiversityAnalysis model validation."""

    def test_valid_l_diversity(self):
        """Test valid l-diversity analysis creation."""
        dist = StatisticalDistribution(
            count=10.0, mean=2.5, std=1.0, min=1.0, percentile_25=2.0, percentile_50=2.5, percentile_75=3.0, max=4.0
        )
        analysis = LDiversityAnalysis(
            min_l_value=2,
            avg_l_value=2.5,
            l_violations=5,
            diversity_distribution=dist,
        )
        assert analysis.min_l_value == 2
        assert analysis.avg_l_value == 2.5

    def test_negative_min_l_value_fails(self):
        """Test that negative min_l_value raises ValidationError."""
        dist = StatisticalDistribution(
            count=10.0, mean=2.5, std=1.0, min=1.0, percentile_25=2.0, percentile_50=2.5, percentile_75=3.0, max=4.0
        )
        with pytest.raises(ValidationError, match='greater than or equal to 0'):
            LDiversityAnalysis(
                min_l_value=-1,
                avg_l_value=2.5,
                l_violations=5,
                diversity_distribution=dist,
            )

    def test_negative_avg_l_value_fails(self):
        """Test that negative avg_l_value raises ValidationError."""
        dist = StatisticalDistribution(
            count=10.0, mean=2.5, std=1.0, min=1.0, percentile_25=2.0, percentile_50=2.5, percentile_75=3.0, max=4.0
        )
        with pytest.raises(ValidationError, match='greater than or equal to 0'):
            LDiversityAnalysis(
                min_l_value=2,
                avg_l_value=-1.0,
                l_violations=5,
                diversity_distribution=dist,
            )

    def test_negative_violations_fails(self):
        """Test that negative violations raises ValidationError."""
        dist = StatisticalDistribution(
            count=10.0, mean=2.5, std=1.0, min=1.0, percentile_25=2.0, percentile_50=2.5, percentile_75=3.0, max=4.0
        )
        with pytest.raises(ValidationError, match='greater than or equal to 0'):
            LDiversityAnalysis(
                min_l_value=2,
                avg_l_value=2.5,
                l_violations=-1,
                diversity_distribution=dist,
            )


class TestKAnonymityAnalysis:
    """Test KAnonymityAnalysis model validation."""

    def test_valid_k_anonymity(self):
        """Test valid k-anonymity analysis creation."""
        dist = StatisticalDistribution(
            count=10.0, mean=5.0, std=2.0, min=2.0, percentile_25=3.0, percentile_50=5.0, percentile_75=7.0, max=10.0
        )
        analysis = KAnonymityAnalysis(
            k_value=3,
            anonymity_level='Good',
            violations=2,
            equivalence_classes=50,
            avg_class_size=5.5,
            class_size_distribution=dist,
            sample_size=100,
        )
        assert analysis.k_value == 3
        assert analysis.anonymity_level == 'Good'

    def test_invalid_anonymity_level_fails(self):
        """Test that invalid anonymity level raises ValidationError."""
        dist = StatisticalDistribution(
            count=10.0, mean=5.0, std=2.0, min=2.0, percentile_25=3.0, percentile_50=5.0, percentile_75=7.0, max=10.0
        )
        with pytest.raises(ValidationError, match='Input should be'):
            KAnonymityAnalysis(
                k_value=3,
                anonymity_level='Invalid',  # Not in Literal options
                violations=2,
                equivalence_classes=50,
                avg_class_size=5.5,
                class_size_distribution=dist,
                sample_size=100,
            )

    def test_negative_k_value_fails(self):
        """Test that negative k_value raises ValidationError."""
        dist = StatisticalDistribution(
            count=10.0, mean=5.0, std=2.0, min=2.0, percentile_25=3.0, percentile_50=5.0, percentile_75=7.0, max=10.0
        )
        with pytest.raises(ValidationError, match='greater than or equal to 0'):
            KAnonymityAnalysis(
                k_value=-1,
                anonymity_level='Good',
                violations=2,
                equivalence_classes=50,
                avg_class_size=5.5,
                class_size_distribution=dist,
                sample_size=100,
            )


class TestAttackModelResult:
    """Test AttackModelResult model validation."""

    def test_valid_attack_result(self):
        """Test valid attack model result creation."""
        result = AttackModelResult(
            accuracy=0.75,
            precision=0.80,
            recall=0.70,
            auc_score=0.85,
            privacy_risk='Medium',
        )
        assert result.accuracy == 0.75
        assert result.privacy_risk == 'Medium'
        assert result.baseline_accuracy == 0.5  # Default value

    def test_accuracy_greater_than_one_fails(self):
        """Test that accuracy > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError, match='less than or equal to 1'):
            AttackModelResult(
                accuracy=1.5,  # > 1.0 should fail
                precision=0.80,
                recall=0.70,
                auc_score=0.85,
                privacy_risk='Medium',
            )

    def test_accuracy_less_than_zero_fails(self):
        """Test that accuracy < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError, match='greater than or equal to 0'):
            AttackModelResult(
                accuracy=-0.1,  # < 0.0 should fail
                precision=0.80,
                recall=0.70,
                auc_score=0.85,
                privacy_risk='Medium',
            )

    def test_invalid_privacy_risk_fails(self):
        """Test that invalid privacy risk raises ValidationError."""
        with pytest.raises(ValidationError, match='Input should be'):
            AttackModelResult(
                accuracy=0.75,
                precision=0.80,
                recall=0.70,
                auc_score=0.85,
                privacy_risk='Invalid',  # Not in Literal['High', 'Medium', 'Low']
            )

    def test_all_metrics_boundary_values(self):
        """Test boundary values for all probability metrics."""
        # Test 0.0 values
        result = AttackModelResult(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            auc_score=0.0,
            privacy_risk='Low',
        )
        assert result.accuracy == 0.0

        # Test 1.0 values
        result = AttackModelResult(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            auc_score=1.0,
            privacy_risk='High',
        )
        assert result.accuracy == 1.0


class TestMembershipInferenceAnalysis:
    """Test MembershipInferenceAnalysis model validation."""

    def test_valid_inference_analysis(self):
        """Test valid membership inference analysis creation."""
        lr_result = AttackModelResult(accuracy=0.75, precision=0.80, recall=0.70, auc_score=0.85, privacy_risk='Medium')
        analysis = MembershipInferenceAnalysis(
            logistic_regression=lr_result,
            max_attack_accuracy=0.75,
            avg_attack_accuracy=0.72,
            privacy_vulnerability='Medium',
            recommendation='Consider data augmentation techniques.',
            sample_size=1000,
        )
        assert analysis.max_attack_accuracy == 0.75
        assert analysis.privacy_vulnerability == 'Medium'

    def test_max_accuracy_greater_than_one_fails(self):
        """Test that max_attack_accuracy > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError, match='less than or equal to 1'):
            MembershipInferenceAnalysis(
                max_attack_accuracy=1.2,  # > 1.0 should fail
                avg_attack_accuracy=0.72,
                privacy_vulnerability='Medium',
                recommendation='Test recommendation.',
                sample_size=1000,
            )

    def test_invalid_vulnerability_level_fails(self):
        """Test that invalid vulnerability level raises ValidationError."""
        with pytest.raises(ValidationError, match='Input should be'):
            MembershipInferenceAnalysis(
                max_attack_accuracy=0.75,
                avg_attack_accuracy=0.72,
                privacy_vulnerability='Critical',  # Not in Literal options
                recommendation='Test recommendation.',
                sample_size=1000,
            )


class TestBasicPrivacyAnalysis:
    """Test BasicPrivacyAnalysis model validation."""

    def test_valid_basic_privacy(self):
        """Test valid basic privacy analysis creation."""
        analysis = BasicPrivacyAnalysis(
            exact_copies_count=5,
            exact_copies_fraction=0.05,
            real_duplicates_count=10,
            synthetic_duplicates_count=8,
            mean_row_distance=2.5,
            std_row_distance=1.2,
        )
        assert analysis.exact_copies_count == 5
        assert analysis.exact_copies_fraction == 0.05

    def test_negative_counts_fail(self):
        """Test that negative counts raise ValidationError."""
        with pytest.raises(ValidationError, match='greater than or equal to 0'):
            BasicPrivacyAnalysis(
                exact_copies_count=-1,  # Negative count should fail
                exact_copies_fraction=0.05,
                real_duplicates_count=10,
                synthetic_duplicates_count=8,
                mean_row_distance=2.5,
                std_row_distance=1.2,
            )

    def test_fraction_greater_than_one_fails(self):
        """Test that fraction > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError, match='less than or equal to 1'):
            BasicPrivacyAnalysis(
                exact_copies_count=5,
                exact_copies_fraction=1.5,  # > 1.0 should fail
                real_duplicates_count=10,
                synthetic_duplicates_count=8,
                mean_row_distance=2.5,
                std_row_distance=1.2,
            )


class TestPrivacyRiskAssessment:
    """Test PrivacyRiskAssessment model validation."""

    def test_valid_risk_assessment(self):
        """Test valid privacy risk assessment creation."""
        assessment = PrivacyRiskAssessment(
            risk_level='Moderate',
            privacy_score=0.65,
            identified_risks=['Low k-anonymity', 'Membership inference vulnerability'],
            k_anonymity_contribution=0.4,
            inference_risk_contribution=0.6,
        )
        assert assessment.risk_level == 'Moderate'
        assert len(assessment.identified_risks) == 2

    def test_invalid_risk_level_fails(self):
        """Test that invalid risk level raises ValidationError."""
        with pytest.raises(ValidationError, match='Input should be'):
            PrivacyRiskAssessment(
                risk_level='Critical',  # Not in Literal options
                privacy_score=0.65,
                k_anonymity_contribution=0.4,
                inference_risk_contribution=0.6,
            )

    def test_score_out_of_bounds_fails(self):
        """Test that privacy_score outside [0,1] raises ValidationError."""
        with pytest.raises(ValidationError, match='less than or equal to 1'):
            PrivacyRiskAssessment(
                risk_level='High',
                privacy_score=1.2,  # > 1.0 should fail
                k_anonymity_contribution=0.4,
                inference_risk_contribution=0.6,
            )


class TestPrivacyEvaluationResults:
    """Test complete PrivacyEvaluationResults model validation."""

    def test_valid_complete_results(self):
        """Test valid complete privacy evaluation results."""
        basic = BasicPrivacyAnalysis(
            exact_copies_count=5,
            exact_copies_fraction=0.05,
            real_duplicates_count=10,
            synthetic_duplicates_count=8,
            mean_row_distance=2.5,
            std_row_distance=1.2,
        )
        assessment = PrivacyRiskAssessment(
            risk_level='Moderate', privacy_score=0.65, k_anonymity_contribution=0.4, inference_risk_contribution=0.6
        )

        results = PrivacyEvaluationResults(
            basic_privacy=basic,
            overall_assessment=assessment,
            recommendations=['Increase data diversity', 'Consider differential privacy'],
            quasi_identifiers_used=['age', 'zipcode'],
            sensitive_attributes_used=['salary'],
        )

        assert results.included_basic is True
        assert results.included_k_anonymity is False
        assert len(results.recommendations) == 2
        assert isinstance(results.metadata, AnalysisMetadata)
        assert results.metadata.library_version == '1.9.0'

    def test_minimal_valid_results(self):
        """Test minimal valid privacy evaluation results."""
        basic = BasicPrivacyAnalysis(
            exact_copies_count=0,
            exact_copies_fraction=0.0,
            real_duplicates_count=0,
            synthetic_duplicates_count=0,
            mean_row_distance=0.0,
            std_row_distance=0.0,
        )
        assessment = PrivacyRiskAssessment(
            risk_level='Low', privacy_score=0.9, k_anonymity_contribution=0.0, inference_risk_contribution=0.0
        )

        results = PrivacyEvaluationResults(
            basic_privacy=basic,
            overall_assessment=assessment,
        )

        assert results.recommendations == []  # Default empty list
        assert results.analysis_errors == []  # Default empty list
        assert results.quasi_identifiers_used is None  # Default None
        assert isinstance(results.metadata, AnalysisMetadata)  # Default metadata
