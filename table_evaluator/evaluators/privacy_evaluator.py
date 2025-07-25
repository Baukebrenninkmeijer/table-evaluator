"""Unified privacy evaluation functionality combining basic and advanced privacy analysis."""

import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from table_evaluator.metrics.privacy import (
    calculate_k_anonymity,
    identify_quasi_identifiers,
    simulate_membership_inference_attack,
)
from table_evaluator.models.privacy_models import (
    BasicPrivacyAnalysis,
    KAnonymityAnalysis,
    MembershipInferenceAnalysis,
    PrivacyEvaluationResults,
    PrivacyRiskAssessment,
)

logger = logging.getLogger(__name__)


class PrivacyEvaluator:
    """Handles privacy evaluation of real vs synthetic data."""

    def __init__(self, verbose: bool = False):
        """
        Initialize the privacy evaluator.

        Args:
            verbose: Whether to print detailed output
        """
        self.verbose = verbose

    def get_copies(self, real: pd.DataFrame, fake: pd.DataFrame, return_len: bool = False) -> pd.DataFrame | int:
        """
        Check whether any real values occur in the fake data (exact matches).

        Args:
            real: DataFrame containing real data
            fake: DataFrame containing synthetic data
            return_len: If True, return the count instead of the DataFrame

        Returns:
            DataFrame with copies found or count if return_len is True
        """
        try:
            # Convert to string for exact matching
            real_str = real.astype(str)
            fake_str = fake.astype(str)

            # Find exact matches row-wise
            real_tuples = [tuple(row) for row in real_str.values]
            fake_tuples = [tuple(row) for row in fake_str.values]

            real_set = set(real_tuples)
            copies = [fake.iloc[i] for i, fake_tuple in enumerate(fake_tuples) if fake_tuple in real_set]

            if return_len:
                return len(copies)

            return pd.DataFrame(copies) if copies else pd.DataFrame()

        except Exception as e:
            logger.error(f'Error finding copies: {e}')
            if return_len:
                return 0
            return pd.DataFrame()

    def get_duplicates(
        self, real: pd.DataFrame, fake: pd.DataFrame, return_values: bool = True
    ) -> tuple[int, int] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get duplicated rows within each dataset.

        Args:
            real: DataFrame containing real data
            fake: DataFrame containing synthetic data
            return_values: If True, return the duplicate rows; if False, return counts

        Returns:
            Tuple of duplicate DataFrames or counts
        """
        try:
            real_duplicates = real[real.duplicated(keep=False)]
            fake_duplicates = fake[fake.duplicated(keep=False)]

            if return_values:
                return real_duplicates, fake_duplicates
            return len(real_duplicates), len(fake_duplicates)

        except Exception as e:
            logger.error(f'Error finding duplicates: {e}')
            if return_values:
                return pd.DataFrame(), pd.DataFrame()
            return 0, 0

    def row_distance(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[float, float]:
        """
        Calculate mean and std of row-wise distances between real and synthetic data.

        Args:
            real: DataFrame containing real data
            synthetic: DataFrame containing synthetic data

        Returns:
            Tuple of (mean_distance, std_distance)
        """
        try:
            # Select only numerical columns
            real_num = real.select_dtypes(include=[np.number])
            synthetic_num = synthetic.select_dtypes(include=[np.number])

            # Use common columns
            common_cols = list(set(real_num.columns) & set(synthetic_num.columns))
            if not common_cols:
                return 0.0, 0.0

            real_num = real_num[common_cols]
            synthetic_num = synthetic_num[common_cols]

            # Sample for computational efficiency
            n_samples = min(1000, len(real_num), len(synthetic_num))
            real_sample = real_num.sample(n=n_samples, random_state=42)
            synthetic_sample = synthetic_num.sample(n=n_samples, random_state=42)

            # Calculate pairwise distances
            distances = cdist(real_sample.values, synthetic_sample.values, metric='euclidean')
            min_distances = distances.min(axis=1)

            return float(np.mean(min_distances)), float(np.std(min_distances))

        except Exception as e:
            logger.error(f'Error calculating row distances: {e}')
            return 0.0, 0.0

    def _assess_privacy_risk(
        self,
        basic_privacy: BasicPrivacyAnalysis,
        k_anonymity: KAnonymityAnalysis | None,
        membership_inference: MembershipInferenceAnalysis | None,
    ) -> PrivacyRiskAssessment:
        """Assess overall privacy risk based on analysis results."""
        risks = []

        # Basic privacy risks
        if basic_privacy.exact_copies_fraction > 0.1:
            risks.append('High proportion of exact copies detected')
        elif basic_privacy.exact_copies_fraction > 0.01:
            risks.append('Moderate proportion of exact copies detected')

        # K-anonymity risks
        k_anon_contribution = 1.0
        if k_anonymity:
            if k_anonymity.k_value < 2:
                risks.append('High k-anonymity risk')
                k_anon_contribution = 0.2
            elif k_anonymity.k_value < 5:
                risks.append('Moderate k-anonymity risk')
                k_anon_contribution = min(1.0, k_anonymity.k_value / 5.0)

        # Membership inference risks
        inference_contribution = 0.0
        if membership_inference:
            if membership_inference.max_attack_accuracy > 0.75:
                risks.append('High membership inference risk')
                inference_contribution = (membership_inference.max_attack_accuracy - 0.5) * 2
            elif membership_inference.max_attack_accuracy > 0.6:
                risks.append('Moderate membership inference risk')
                inference_contribution = (membership_inference.max_attack_accuracy - 0.5) * 2

        # Overall risk assessment
        if any('High' in risk for risk in risks):
            risk_level = 'High'
        elif any('Moderate' in risk for risk in risks):
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'

        # Calculate combined privacy score
        base_score = 1.0
        base_score *= k_anon_contribution
        base_score *= max(0.0, 1.0 - inference_contribution)
        base_score *= max(0.0, 1.0 - basic_privacy.exact_copies_fraction * 2)

        privacy_score = max(0.0, min(1.0, base_score))

        return PrivacyRiskAssessment(
            risk_level=risk_level,
            privacy_score=privacy_score,
            identified_risks=risks,
            k_anonymity_contribution=k_anon_contribution,
            inference_risk_contribution=inference_contribution,
        )

    def _generate_recommendations(
        self,
        basic_privacy: BasicPrivacyAnalysis,
        k_anonymity: KAnonymityAnalysis | None,
        membership_inference: MembershipInferenceAnalysis | None,
        overall_assessment: PrivacyRiskAssessment,
    ) -> list[str]:
        """Generate privacy improvement recommendations."""
        recommendations = []

        # Basic privacy recommendations
        if basic_privacy.exact_copies_fraction > 0.01:
            recommendations.append('Reduce exact copies by adding more noise or using different generation parameters')

        # K-anonymity recommendations
        if k_anonymity and k_anonymity.k_value < 5:
            recommendations.append(
                f'Improve k-anonymity (current k={k_anonymity.k_value}) by generalizing quasi-identifiers'
            )

        # Membership inference recommendations
        if membership_inference and membership_inference.max_attack_accuracy > 0.6:
            recommendations.append(membership_inference.recommendation)

        # Overall recommendations
        if overall_assessment.risk_level == 'High':
            recommendations.append('Consider regenerating synthetic data with stronger privacy protections')
        elif overall_assessment.risk_level == 'Moderate':
            recommendations.append('Review and improve specific privacy risk factors identified')

        return recommendations

    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        quasi_identifiers: list[str] | None = None,
        sensitive_attributes: list[str] | None = None,
        include_basic: bool = True,
        include_k_anonymity: bool = True,
        include_membership_inference: bool = True,
        **kwargs,
    ) -> 'PrivacyEvaluationResults':
        """
        Comprehensive privacy evaluation with all methods.

        Args:
            real_data: Original dataset
            synthetic_data: Synthetic dataset
            quasi_identifiers: List of quasi-identifier column names
            sensitive_attributes: List of sensitive attribute column names
            include_basic: Whether to include basic privacy checks
            include_k_anonymity: Whether to include k-anonymity analysis
            include_membership_inference: Whether to include membership inference analysis
            **kwargs: Additional parameters for specific methods

        Returns:
            PrivacyEvaluationResults: Comprehensive privacy analysis results
        """
        if self.verbose:
            print('Running comprehensive privacy evaluation...')

        analysis_errors = []

        # Basic privacy analysis
        basic_privacy = BasicPrivacyAnalysis(
            exact_copies_count=0,
            exact_copies_fraction=0.0,
            real_duplicates_count=0,
            synthetic_duplicates_count=0,
            mean_row_distance=0.0,
            std_row_distance=0.0,
        )

        if include_basic:
            try:
                copies_count = self.get_copies(real_data, synthetic_data, return_len=True)
                copies_fraction = copies_count / len(synthetic_data) if len(synthetic_data) > 0 else 0.0

                real_dup_count, synthetic_dup_count = self.get_duplicates(
                    real_data, synthetic_data, return_values=False
                )
                mean_dist, std_dist = self.row_distance(real_data, synthetic_data)

                basic_privacy = BasicPrivacyAnalysis(
                    exact_copies_count=copies_count,
                    exact_copies_fraction=copies_fraction,
                    real_duplicates_count=real_dup_count,
                    synthetic_duplicates_count=synthetic_dup_count,
                    mean_row_distance=mean_dist,
                    std_row_distance=std_dist,
                )
            except Exception as e:
                logger.error(f'Basic privacy evaluation failed: {e}')
                analysis_errors.append(f'Basic privacy analysis: {e!s}')

        # K-anonymity analysis
        k_anonymity = None
        if include_k_anonymity:
            try:
                if quasi_identifiers is None:
                    quasi_identifiers = identify_quasi_identifiers(synthetic_data)

                k_anonymity = calculate_k_anonymity(synthetic_data, quasi_identifiers, sensitive_attributes)
            except Exception as e:
                logger.error(f'K-anonymity evaluation failed: {e}')
                analysis_errors.append(f'K-anonymity analysis: {e!s}')

        # Membership inference analysis
        membership_inference = None
        if include_membership_inference:
            try:
                membership_inference = simulate_membership_inference_attack(real_data, synthetic_data)
            except Exception as e:
                logger.error(f'Membership inference evaluation failed: {e}')
                analysis_errors.append(f'Membership inference analysis: {e!s}')

        # Overall assessment
        overall_assessment = self._assess_privacy_risk(basic_privacy, k_anonymity, membership_inference)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            basic_privacy, k_anonymity, membership_inference, overall_assessment
        )

        return PrivacyEvaluationResults(
            basic_privacy=basic_privacy,
            k_anonymity=k_anonymity,
            membership_inference=membership_inference,
            overall_assessment=overall_assessment,
            recommendations=recommendations,
            quasi_identifiers_used=quasi_identifiers,
            sensitive_attributes_used=sensitive_attributes,
            analysis_errors=analysis_errors,
            included_basic=include_basic,
            included_k_anonymity=include_k_anonymity,
            included_membership_inference=include_membership_inference,
        )
