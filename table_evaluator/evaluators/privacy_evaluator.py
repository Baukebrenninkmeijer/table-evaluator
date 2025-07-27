"""Unified privacy evaluation functionality combining basic and advanced privacy analysis."""

import logging
from functools import lru_cache
from typing import Literal

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
    """Handles privacy evaluation of real vs synthetic data with LRU caching for performance."""

    def __init__(self, *, verbose: bool = False, cache_size: int = 128) -> None:
        """
        Initialize the privacy evaluator.

        Args:
            verbose: Whether to print detailed output
            cache_size: Maximum size of the LRU cache for expensive computations
        """
        self.verbose = verbose

        # Apply LRU caching to expensive methods that only use hashable arguments
        self._cached_row_distance = lru_cache(maxsize=cache_size)(self._compute_row_distance_cached)
        self._cached_duplicates = lru_cache(maxsize=cache_size)(self._compute_duplicates_cached)

        # Store data for cache lookups (simple in-memory storage)
        self._data_cache: dict[str, pd.DataFrame] = {}

    def _compute_data_hash(self, data_repr: str) -> int:
        """
        Compute a hash of DataFrame for caching purposes.

        Args:
            data_repr: String representation of DataFrame

        Returns:
            Hash value for the data
        """
        return hash(data_repr)

    def _dataframe_to_cache_key(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to a string representation for caching.

        Args:
            df: DataFrame to convert

        Returns:
            String representation suitable for hashing
        """
        return f'{df.shape}_{hash(df.to_string())}'

    def _compute_duplicates_cached(self, data_key: str) -> tuple[pd.DataFrame, int]:
        """
        Cached computation of duplicate rows within a dataset.

        Args:
            data_key: Cache key for the data

        Returns:
            Tuple of (duplicate_rows, duplicate_count)
        """
        if data_key not in self._data_cache:
            return pd.DataFrame(), 0

        data = self._data_cache[data_key]
        duplicates = data[data.duplicated(keep=False)]
        return duplicates, len(duplicates)

    def _compute_row_distance_cached(
        self, real_key: str, synthetic_key: str, max_samples: int = 5000
    ) -> tuple[float, float]:
        """
        Cached computation of row distances between real and synthetic data.

        Args:
            real_key: Cache key for real data
            synthetic_key: Cache key for synthetic data
            max_samples: Maximum number of samples to use for distance calculation

        Returns:
            Tuple of (mean_distance, std_distance)
        """
        try:
            if real_key not in self._data_cache or synthetic_key not in self._data_cache:
                return 0.0, 0.0

            real_data = self._data_cache[real_key]
            synthetic_data = self._data_cache[synthetic_key]

            # Select only numerical columns
            real_num = real_data.select_dtypes(include=[np.number])
            synthetic_num = synthetic_data.select_dtypes(include=[np.number])

            # Use common columns
            common_cols = list(set(real_num.columns) & set(synthetic_num.columns))
            if not common_cols:
                return 0.0, 0.0

            real_num = real_num[common_cols]
            synthetic_num = synthetic_num[common_cols]

            # Sample for computational efficiency
            n_samples = min(max_samples, len(real_num), len(synthetic_num))
            real_sample = real_num.sample(n=n_samples, random_state=42)
            synthetic_sample = synthetic_num.sample(n=n_samples, random_state=42)

            # Calculate pairwise distances
            distances = cdist(real_sample.values, synthetic_sample.values, metric='euclidean')
            min_distances = distances.min(axis=1)

            return float(np.mean(min_distances)), float(np.std(min_distances))

        except Exception:
            logger.exception('Error calculating row distances')
            return 0.0, 0.0

    def get_copies(self, real: pd.DataFrame, fake: pd.DataFrame, *, return_len: bool = False) -> pd.DataFrame | int:
        """
        Check whether any real values occur in the fake data (exact matches).

        Args:
            real: DataFrame containing real data
            fake: DataFrame containing synthetic data
            return_len: If True, return the count instead of the DataFrame

        Returns:
            DataFrame with copies found or count if return_len is True
        """
        # Create hashes of each row to efficiently find exact matches
        real_hashes = real.apply(lambda x: hash(tuple(x)), axis=1)
        fake_hashes = fake.apply(lambda x: hash(tuple(x)), axis=1)

        # Find fake rows that match real rows
        duplicate_indices = fake_hashes.isin(real_hashes.values)
        duplicate_indices = duplicate_indices[duplicate_indices].sort_index().index.tolist()

        if self.verbose:
            logger.info(f'Number of copied rows: {len(duplicate_indices)}')

        if return_len:
            return len(duplicate_indices)

        return pd.DataFrame(duplicate_indices) if duplicate_indices else pd.DataFrame()

    def get_duplicates(
        self, real: pd.DataFrame, fake: pd.DataFrame, *, return_values: bool = True
    ) -> tuple[int, int] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get duplicated rows within each dataset (with caching).

        Args:
            real: DataFrame containing real data
            fake: DataFrame containing synthetic data
            return_values: If True, return the duplicate rows; if False, return counts

        Returns:
            Tuple of duplicate DataFrames or counts
        """
        try:
            # Use cached computation for duplicates
            real_key = self._dataframe_to_cache_key(real)
            fake_key = self._dataframe_to_cache_key(fake)

            # Store data in cache for the cached methods to access
            self._data_cache[real_key] = real
            self._data_cache[fake_key] = fake

            real_duplicates, real_count = self._cached_duplicates(real_key)
            fake_duplicates, fake_count = self._cached_duplicates(fake_key)

            if return_values:
                return real_duplicates, fake_duplicates
            return real_count, fake_count

        except Exception:
            logger.exception('Error finding duplicates')
            if return_values:
                return pd.DataFrame(), pd.DataFrame()
            return 0, 0

    def row_distance(self, real: pd.DataFrame, synthetic: pd.DataFrame, max_samples: int = 5000) -> tuple[float, float]:
        """
        Calculate mean and std of row-wise distances between real and synthetic data (with caching).

        Args:
            real: DataFrame containing real data
            synthetic: DataFrame containing synthetic data
            max_samples: Maximum number of samples to use for distance calculation

        Returns:
            Tuple of (mean_distance, std_distance)
        """
        try:
            # Use cached computation for row distances
            real_key = self._dataframe_to_cache_key(real)
            synthetic_key = self._dataframe_to_cache_key(synthetic)

            # Store data in cache for the cached methods to access
            self._data_cache[real_key] = real
            self._data_cache[synthetic_key] = synthetic

            return self._cached_row_distance(real_key, synthetic_key, max_samples)

        except Exception:
            logger.exception('Error calculating row distances')
            return 0.0, 0.0

    def _assess_basic_privacy_risks(self, basic_privacy: BasicPrivacyAnalysis) -> list[str]:
        """Assess basic privacy risks."""
        risks = []
        if basic_privacy.exact_copies_fraction > 0.1:
            risks.append('High proportion of exact copies detected')
        elif basic_privacy.exact_copies_fraction > 0.01:
            risks.append('Moderate proportion of exact copies detected')
        return risks

    def _assess_k_anonymity_risks(self, k_anonymity: KAnonymityAnalysis | None) -> tuple[list[str], float]:
        """Assess k-anonymity risks and return contribution score."""
        risks = []
        k_anon_contribution = 1.0

        if k_anonymity:
            if k_anonymity.k_value < 2:
                risks.append('High k-anonymity risk')
                k_anon_contribution = 0.2
            elif k_anonymity.k_value < 5:
                risks.append('Moderate k-anonymity risk')
                k_anon_contribution = min(1.0, k_anonymity.k_value / 5.0)

        return risks, k_anon_contribution

    def _assess_membership_inference_risks(
        self, membership_inference: MembershipInferenceAnalysis | None
    ) -> tuple[list[str], float]:
        """Assess membership inference risks and return contribution score."""
        risks = []
        inference_contribution = 0.0

        if membership_inference:
            if membership_inference.max_attack_accuracy > 0.75:
                risks.append('High membership inference risk')
                inference_contribution = (membership_inference.max_attack_accuracy - 0.5) * 2
            elif membership_inference.max_attack_accuracy > 0.6:
                risks.append('Moderate membership inference risk')
                inference_contribution = (membership_inference.max_attack_accuracy - 0.5) * 2

        return risks, inference_contribution

    def _calculate_overall_risk_level(self, all_risks: list[str]) -> Literal['Low', 'Moderate', 'High']:
        """Determine overall risk level from identified risks."""
        if any('High' in risk for risk in all_risks):
            return 'High'
        if any('Moderate' in risk for risk in all_risks):
            return 'Moderate'
        return 'Low'

    def _calculate_privacy_score(
        self,
        k_anon_contribution: float,
        inference_contribution: float,
        basic_privacy: BasicPrivacyAnalysis,
    ) -> float:
        """Calculate combined privacy score."""
        base_score = 1.0
        base_score *= k_anon_contribution
        base_score *= max(0.0, 1.0 - inference_contribution)
        base_score *= max(0.0, 1.0 - basic_privacy.exact_copies_fraction * 2)

        return max(0.0, min(1.0, base_score))

    def _assess_privacy_risk(
        self,
        basic_privacy: BasicPrivacyAnalysis,
        k_anonymity: KAnonymityAnalysis | None,
        membership_inference: MembershipInferenceAnalysis | None,
    ) -> PrivacyRiskAssessment:
        """Assess overall privacy risk based on analysis results."""
        # Assess different types of privacy risks
        basic_risks = self._assess_basic_privacy_risks(basic_privacy)
        k_anon_risks, k_anon_contribution = self._assess_k_anonymity_risks(k_anonymity)
        inference_risks, inference_contribution = self._assess_membership_inference_risks(membership_inference)

        # Combine all risks
        all_risks = basic_risks + k_anon_risks + inference_risks

        # Determine overall risk level
        risk_level = self._calculate_overall_risk_level(all_risks)

        # Calculate combined privacy score
        privacy_score = self._calculate_privacy_score(k_anon_contribution, inference_contribution, basic_privacy)

        return PrivacyRiskAssessment(
            risk_level=risk_level,
            privacy_score=privacy_score,
            identified_risks=all_risks,
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
        *,
        include_basic: bool = True,
        include_k_anonymity: bool = True,
        include_membership_inference: bool = True,
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
            logger.info('Running comprehensive privacy evaluation...')

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
                copies_count_result = self.get_copies(real_data, synthetic_data, return_len=True)
                copies_count = int(copies_count_result) if isinstance(copies_count_result, int) else 0
                copies_fraction = copies_count / len(synthetic_data) if len(synthetic_data) > 0 else 0.0

                duplicate_counts = self.get_duplicates(real_data, synthetic_data, return_values=False)
                if isinstance(duplicate_counts, tuple) and len(duplicate_counts) == 2:
                    real_dup_count = int(duplicate_counts[0]) if isinstance(duplicate_counts[0], int) else 0
                    synthetic_dup_count = int(duplicate_counts[1]) if isinstance(duplicate_counts[1], int) else 0
                else:
                    real_dup_count, synthetic_dup_count = 0, 0

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
                logger.exception('Basic privacy evaluation failed')
                analysis_errors.append(f'Basic privacy analysis: {e!s}')

        # K-anonymity analysis
        k_anonymity = None
        if include_k_anonymity:
            try:
                if quasi_identifiers is None:
                    quasi_identifiers = identify_quasi_identifiers(synthetic_data)

                k_anonymity = calculate_k_anonymity(synthetic_data, quasi_identifiers, sensitive_attributes)
            except Exception as e:
                logger.exception('K-anonymity evaluation failed')
                analysis_errors.append(f'K-anonymity analysis: {e!s}')

        # Membership inference analysis
        membership_inference = None
        if include_membership_inference:
            try:
                membership_inference = simulate_membership_inference_attack(real_data, synthetic_data)
            except Exception as e:
                logger.exception('Membership inference evaluation failed')
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
