"""Unified statistical evaluation functionality combining basic and advanced statistical analysis."""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error

from table_evaluator.evaluators.models.statistical_models import (
    BasicStatisticalResults,
    CombinedStatisticalMetrics,
    MMDResults,
    StatisticalEvaluationResults,
    WassersteinResults,
)
from table_evaluator.metrics.statistical import (
    associations,
    earth_movers_distance_summary,
    mmd_column_wise,
    mmd_comprehensive_analysis,
    optimal_transport_cost,
    wasserstein_distance_df,
)

logger = logging.getLogger(__name__)


class StatisticalEvaluator:
    """Unified statistical evaluator combining basic and advanced statistical analysis."""

    def __init__(self, comparison_metric: Callable | None = None, *, verbose: bool = False) -> None:
        """
        Initialize the statistical evaluator.

        Args:
            comparison_metric: Function to compare two arrays (e.g., stats.pearsonr)
                              If None, defaults to stats.pearsonr
            verbose: Whether to print detailed output
        """
        self.comparison_metric = comparison_metric or stats.pearsonr
        self.verbose = verbose

    def basic_statistical_evaluation(
        self, real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: list[str]
    ) -> float:
        """
        Calculate the correlation coefficient between the basic properties of real and fake data.

        Uses Spearman's Rho as it's more resilient to outliers and magnitude differences.

        Args:
            real: Real dataset
            fake: Synthetic dataset
            numerical_columns: List of numerical column names

        Returns:
            float: Correlation coefficient between basic statistical properties
        """
        total_metrics = pd.DataFrame()

        for ds_name, ds in [('real', real), ('fake', fake)]:
            metrics = {}
            num_ds = ds[numerical_columns]

            # Calculate basic statistics
            for idx, value in num_ds.mean().items():
                metrics[f'mean_{idx}'] = value
            for idx, value in num_ds.median().items():
                metrics[f'median_{idx}'] = value
            for idx, value in num_ds.std().items():
                metrics[f'std_{idx}'] = value
            for idx, value in num_ds.var().items():
                metrics[f'variance_{idx}'] = value

            total_metrics[ds_name] = metrics.values()

        total_metrics.index = list(metrics.keys())

        if self.verbose:
            print('\nBasic statistical attributes:')
            print(total_metrics.to_string())

        corr, p = stats.spearmanr(total_metrics['real'], total_metrics['fake'])
        return corr

    def correlation_correlation(self, real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: list[str]) -> float:
        """
        Calculate correlation coefficient between association matrices of real and fake data.

        Args:
            real: Real dataset
            fake: Synthetic dataset
            categorical_columns: List of categorical column names

        Returns:
            float: Correlation coefficient between association matrices
        """
        total_metrics = pd.DataFrame()

        for ds_name, ds in [('real', real), ('fake', fake)]:
            corr_df: pd.DataFrame = associations(
                ds,
                nominal_columns=categorical_columns,
                nom_nom_assoc='theil',
            )['corr']

            values = corr_df.to_numpy()
            # Remove diagonal elements (self-correlations)
            values = values[~np.eye(values.shape[0], dtype=bool)].reshape(values.shape[0], -1)
            total_metrics[ds_name] = values.flatten()

        if self.verbose:
            print('\nColumn correlation between datasets:')
            print(total_metrics.to_string())

        corr, p = self.comparison_metric(total_metrics['real'], total_metrics['fake'])
        return corr

    def pca_correlation(self, real: pd.DataFrame, fake: pd.DataFrame, *, lingress: bool = False) -> float:
        """
        Calculate the relation between PCA explained variance values.

        Args:
            real: Real dataset (must be numerical)
            fake: Synthetic dataset (must be numerical)
            lingress: Whether to use linear regression (Pearson's r) vs MAPE

        Returns:
            float: Correlation coefficient if lingress=True, otherwise 1 - MAPE(log)
        """
        # Limit PCA components to min of 5 or number of features/samples
        n_components = min(5, real.shape[1], real.shape[0] - 1)
        pca_real = PCA(n_components=n_components)
        pca_fake = PCA(n_components=n_components)

        pca_real.fit(real)
        pca_fake.fit(fake)

        if self.verbose:
            results = pd.DataFrame(
                {
                    'real': pca_real.explained_variance_,
                    'fake': pca_fake.explained_variance_,
                }
            )
            print('\nTop 5 PCA components:')
            print(results.to_string())

        if lingress:
            corr, p = self.comparison_metric(pca_real.explained_variance_, pca_fake.explained_variance_)
            return corr
        pca_error = mean_absolute_percentage_error(pca_real.explained_variance_, pca_fake.explained_variance_)
        return float(1 - pca_error)

    # Advanced Statistical Methods (from AdvancedStatisticalEvaluator)

    def wasserstein_evaluation(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        numerical_columns: list[str],
        *,
        include_2d: bool = False,
        enable_sampling: bool = False,
        max_samples: int = 5000,
    ) -> dict:
        """
        Comprehensive Wasserstein distance evaluation.

        Args:
            real: DataFrame containing real data
            fake: DataFrame containing synthetic data
            numerical_columns: List of numerical column names
            include_2d: Whether to include 2D pairwise analysis
            enable_sampling: Whether to enable sampling for large datasets
            max_samples: Maximum samples per dataset when sampling is enabled

        Returns:
            Dictionary containing comprehensive Wasserstein analysis results
        """
        if self.verbose:
            print('Computing Wasserstein distances...')

        results = {}

        try:
            # 1D Wasserstein distances (order 1 and 2)
            distances_p1 = wasserstein_distance_df(real, fake, numerical_columns, p=1)
            distances_p2 = wasserstein_distance_df(real, fake, numerical_columns, p=2)

            results['distances_p1'] = distances_p1
            results['distances_p2'] = distances_p2

            # Summary analysis
            summary = earth_movers_distance_summary(real, fake, numerical_columns)
            results['summary'] = summary

            # 2D analysis for column pairs (optional, computationally expensive)
            if include_2d and len(numerical_columns) >= 2:
                if self.verbose:
                    print('Computing 2D Wasserstein distances...')

                distances_2d = wasserstein_distance_df(
                    real,
                    fake,
                    numerical_columns,
                    method='2d',
                    enable_sampling=enable_sampling,
                    max_samples_2d=max_samples,
                )
                results['distances_2d'] = distances_2d

            # Per-column detailed analysis
            def _compute_transport_cost_safe(col: str) -> tuple[float, np.ndarray, np.ndarray]:
                try:
                    cost, plan, bins = optimal_transport_cost(real[col], fake[col])
                    return {
                        'transport_cost': cost,
                        'transport_plan_shape': plan.shape,
                        'n_bins': len(bins) - 1,
                    }
                except Exception as e:
                    logger.warning(f'Failed to compute transport cost for {col}: {e}')
                    return {'error': str(e)}

            column_details = {col: _compute_transport_cost_safe(col) for col in numerical_columns}

            results['column_details'] = column_details

            # Overall quality metrics
            p1_distances = distances_p1['wasserstein_distance'].to_numpy()
            results['quality_metrics'] = {
                'mean_wasserstein_p1': np.mean(p1_distances),
                'median_wasserstein_p1': np.median(p1_distances),
                'std_wasserstein_p1': np.std(p1_distances),
                'max_wasserstein_p1': np.max(p1_distances),
                'distribution_similarity_score': np.exp(-np.mean(p1_distances)),
                'quality_rating': self._rate_wasserstein_quality(np.mean(p1_distances)),
            }

        except Exception as e:
            logger.exception('Error in Wasserstein evaluation')
            results['error'] = str(e)

        return results

    def mmd_evaluation(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        numerical_columns: list[str],
        kernel_types: list[str] | None = None,
        *,
        include_multivariate: bool = True,
    ) -> dict:
        """
        Comprehensive Maximum Mean Discrepancy evaluation.

        Args:
            real: DataFrame containing real data
            fake: DataFrame containing synthetic data
            numerical_columns: List of numerical column names
            kernel_types: List of kernels to use (default: ["rbf", "polynomial", "linear"])
            include_multivariate: Whether to include multivariate MMD analysis
            enable_sampling: Whether to enable sampling for large datasets
            max_samples: Maximum samples per dataset when sampling is enabled

        Returns:
            Dictionary containing comprehensive MMD analysis results
        """
        if kernel_types is None:
            kernel_types = ['rbf', 'polynomial', 'linear']

        if self.verbose:
            print(f'Computing MMD with kernels: {kernel_types}')

        results = {}

        try:
            if include_multivariate:
                # Comprehensive analysis (both column-wise and multivariate)
                comprehensive = mmd_comprehensive_analysis(real, fake, numerical_columns, kernel_types)
                results.update(comprehensive)
            else:
                # Column-wise analysis only
                results['column_wise'] = {}
                for kernel_type in kernel_types:
                    results['column_wise'][kernel_type] = mmd_column_wise(real, fake, numerical_columns, kernel_type)

            # Extract key metrics for summary
            if 'summary' in results:
                summary = results['summary']
                results['quality_metrics'] = {
                    'mean_mmd': summary['mean_mmd'],
                    'fraction_significant_differences': summary['fraction_significant'],
                    'overall_quality_score': summary['overall_quality_score'],
                    'mmd_rating': self._rate_mmd_quality(summary['mean_mmd']),
                }

            # Best kernel analysis
            if 'multivariate' in results:
                best_kernel_info = self._find_best_kernel(results['multivariate'])
                results['best_kernel'] = best_kernel_info

        except Exception as e:
            logger.exception('Error in MMD evaluation')
            results['error'] = str(e)

        return results

    def _run_basic_statistical_evaluation(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        numerical_columns: list[str],
        categorical_columns: list[str],
    ) -> dict:
        """Run basic statistical evaluation with error handling."""
        try:
            return {
                'basic_statistical_evaluation': self.basic_statistical_evaluation(
                    real_data, synthetic_data, numerical_columns
                ),
                'correlation_correlation': self.correlation_correlation(real_data, synthetic_data, categorical_columns)
                if categorical_columns
                else None,
                'pca_correlation': self.pca_correlation(
                    real_data[numerical_columns], synthetic_data[numerical_columns]
                ),
            }
        except Exception as e:
            logger.exception('Basic statistical evaluation failed')
            return {'error': str(e)}

    def _run_wasserstein_evaluation(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        numerical_columns: list[str],
        wasserstein_config: dict,
    ) -> dict:
        """Run Wasserstein evaluation with error handling."""
        try:
            return self.wasserstein_evaluation(real_data, synthetic_data, numerical_columns, **wasserstein_config)
        except Exception as e:
            logger.exception('Wasserstein evaluation failed')
            return {'error': str(e)}

    def _run_mmd_evaluation(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, numerical_columns: list[str], mmd_config: dict
    ) -> dict:
        """Run MMD evaluation with error handling."""
        try:
            return self.mmd_evaluation(real_data, synthetic_data, numerical_columns, **mmd_config)
        except Exception as e:
            logger.exception('MMD evaluation failed')
            return {'error': str(e)}

    def _run_combined_analysis(self, wasserstein_results: dict, mmd_results: dict) -> tuple[dict, list[str]]:
        """Run combined analysis with error handling."""
        try:
            combined_metrics_dict = self._combine_metrics(wasserstein_results, mmd_results)
            recommendations = self._generate_recommendations(
                {
                    'wasserstein': wasserstein_results,
                    'mmd': mmd_results,
                    'basic_statistics': {},
                    'combined_metrics': combined_metrics_dict,
                    'recommendations': [],
                }
            )
        except Exception as e:
            logger.exception('Combined analysis failed')
            return {'error': str(e)}, []
        else:
            return combined_metrics_dict, recommendations

    def _build_evaluation_results(
        self,
        results: dict,
        combined_metrics_dict: dict,
        recommendations: list[str],
        numerical_columns: list[str],
        categorical_columns: list[str],
        *,
        include_basic: bool,
        include_wasserstein: bool,
        include_mmd: bool,
    ) -> 'StatisticalEvaluationResults':
        """Build the final evaluation results with error handling."""
        try:
            # Basic statistical results
            basic_statistics = BasicStatisticalResults(**results['basic_statistics'])

            # Wasserstein results
            wasserstein = WassersteinResults(**results['wasserstein'])

            # MMD results
            mmd = MMDResults(**results['mmd'])

            # Combined metrics
            combined_metrics = CombinedStatisticalMetrics(**combined_metrics_dict)

            # Create comprehensive results
            return StatisticalEvaluationResults(
                basic_statistics=basic_statistics,
                wasserstein=wasserstein,
                mmd=mmd,
                combined_metrics=combined_metrics,
                recommendations=recommendations,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                included_basic=include_basic,
                included_wasserstein=include_wasserstein,
                included_mmd=include_mmd,
            )
        except Exception:
            logger.exception('Error building StatisticalEvaluationResults')
            # Fallback: create models with error handling
            return StatisticalEvaluationResults(
                basic_statistics=BasicStatisticalResults(**results.get('basic_statistics', {})),
                wasserstein=WassersteinResults(**results.get('wasserstein', {})),
                mmd=MMDResults(**results.get('mmd', {})),
                combined_metrics=CombinedStatisticalMetrics(**combined_metrics_dict),
                recommendations=recommendations,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns or [],
                included_basic=include_basic,
                included_wasserstein=include_wasserstein,
                included_mmd=include_mmd,
            )

    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        numerical_columns: list[str],
        categorical_columns: list[str] | None = None,
        *,
        include_basic: bool = True,
        include_wasserstein: bool = True,
        include_mmd: bool = True,
        wasserstein_config: dict | None = None,
        mmd_config: dict | None = None,
    ) -> 'StatisticalEvaluationResults':
        """
        Comprehensive statistical evaluation with all methods.

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            numerical_columns: List of numerical column names
            categorical_columns: List of categorical column names
            include_basic: Whether to include basic statistical evaluation
            include_wasserstein: Whether to include Wasserstein distance analysis
            include_mmd: Whether to include MMD analysis
            wasserstein_config: Configuration for Wasserstein evaluation
            mmd_config: Configuration for MMD evaluation
            **kwargs: Additional parameters

        Returns:
            dict: Comprehensive statistical analysis results
        """
        if self.verbose:
            print('Running comprehensive statistical evaluation...')

        # Initialize configuration defaults
        if categorical_columns is None:
            categorical_columns = []
        if wasserstein_config is None:
            wasserstein_config = {}
        if mmd_config is None:
            mmd_config = {}

        results = {
            'basic_statistics': {},
            'wasserstein': {},
            'mmd': {},
        }

        # Run evaluations
        if include_basic:
            results['basic_statistics'] = self._run_basic_statistical_evaluation(
                real_data, synthetic_data, numerical_columns, categorical_columns
            )

        if include_wasserstein:
            results['wasserstein'] = self._run_wasserstein_evaluation(
                real_data, synthetic_data, numerical_columns, wasserstein_config
            )

        if include_mmd:
            results['mmd'] = self._run_mmd_evaluation(real_data, synthetic_data, numerical_columns, mmd_config)

        # Combined analysis
        combined_metrics_dict, recommendations = self._run_combined_analysis(results['wasserstein'], results['mmd'])

        # Build and return final results
        return self._build_evaluation_results(
            results,
            combined_metrics_dict,
            recommendations,
            numerical_columns,
            categorical_columns,
            include_basic,
            include_wasserstein,
            include_mmd,
        )

    # Helper methods

    def _rate_wasserstein_quality(self, mean_distance: float) -> str:
        """Rate the quality based on mean Wasserstein distance."""
        if mean_distance < 0.1:
            return 'Excellent'
        if mean_distance < 0.25:
            return 'Good'
        if mean_distance < 0.5:
            return 'Fair'
        if mean_distance < 1.0:
            return 'Poor'
        return 'Very Poor'

    def _rate_mmd_quality(self, mean_mmd: float) -> str:
        """Rate the quality based on mean MMD."""
        if np.isnan(mean_mmd):
            return 'Unknown'

        if mean_mmd < 0.01:
            return 'Excellent'
        if mean_mmd < 0.05:
            return 'Good'
        if mean_mmd < 0.1:
            return 'Fair'
        if mean_mmd < 0.2:
            return 'Poor'
        return 'Very Poor'

    def _find_best_kernel(self, multivariate_results: dict) -> dict:
        """Find the kernel that gives the most discriminative results."""
        best_kernel = None
        best_score = -1

        for kernel_type, results in multivariate_results.items():
            if 'error' not in results:
                # Score based on discriminative power (higher MMD = more discriminative)
                score = results.get('mmd_squared', 0)
                if score > best_score:
                    best_score = score
                    best_kernel = kernel_type

        return {
            'kernel': best_kernel,
            'mmd_squared': best_score,
            'discriminative_power': 'High' if best_score > 0.1 else 'Medium' if best_score > 0.05 else 'Low',
        }

    def _combine_metrics(self, wasserstein_results: dict, mmd_results: dict) -> dict:
        """Combine Wasserstein and MMD results into unified metrics."""
        combined = {}

        # Extract key metrics
        wasserstein_quality = wasserstein_results.get('quality_metrics', {})
        mmd_quality = mmd_results.get('quality_metrics', {})

        # Overall distribution similarity (0-1 scale, higher is better)
        wass_similarity = wasserstein_quality.get('distribution_similarity_score', 0)
        mmd_similarity = mmd_quality.get('overall_quality_score', 0)

        combined['overall_similarity'] = (wass_similarity + mmd_similarity) / 2

        # Agreement between methods
        wass_rating = wasserstein_quality.get('quality_rating', 'Unknown')
        mmd_rating = mmd_quality.get('mmd_rating', 'Unknown')

        combined['quality_consensus'] = wass_rating if wass_rating == mmd_rating else 'Mixed'
        combined['wasserstein_rating'] = wass_rating
        combined['mmd_rating'] = mmd_rating

        # Statistical significance
        mmd_significant_fraction = mmd_quality.get('fraction_significant_differences', 0)
        combined['statistical_significance'] = {
            'fraction_columns_different': mmd_significant_fraction,
            'interpretation': 'High'
            if mmd_significant_fraction > 0.5
            else 'Medium'
            if mmd_significant_fraction > 0.2
            else 'Low',
        }

        return combined

    def _generate_recommendations(self, results: dict) -> list[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        combined = results.get('combined_metrics', {})
        wasserstein = results.get('wasserstein', {})

        # Overall similarity recommendation
        similarity = combined.get('overall_similarity', 0)
        if similarity < 0.3:
            recommendations.append('Low overall similarity detected. Consider improving the data generation model.')
        elif similarity < 0.7:
            recommendations.append('Moderate similarity achieved. Fine-tuning the model may improve results.')
        else:
            recommendations.append('Good similarity achieved. The synthetic data quality is acceptable.')

        # Specific metric recommendations
        if 'quality_metrics' in wasserstein:
            worst_wass = wasserstein['summary'].get('overall_metrics', {}).get('worst_column')
            if worst_wass:
                recommendations.append(
                    f"Column '{worst_wass}' has the highest Wasserstein distance. "
                    "Focus on improving this variable's distribution."
                )

        # Statistical significance recommendations
        sig_info = combined.get('statistical_significance', {})
        if sig_info.get('interpretation') == 'High':
            recommendations.append(
                'Many columns show statistically significant differences. '
                'Review the data generation process for systematic biases.'
            )

        return recommendations
