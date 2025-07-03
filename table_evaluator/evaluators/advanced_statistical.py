"""Advanced statistical evaluation using Wasserstein distance and Maximum Mean Discrepancy."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from table_evaluator.metrics.wasserstein import (
    earth_movers_distance_summary,
    wasserstein_distance_df,
    optimal_transport_cost,
)
from table_evaluator.metrics.mmd import mmd_comprehensive_analysis, mmd_column_wise

logger = logging.getLogger(__name__)


class AdvancedStatisticalEvaluator:
    """Advanced statistical evaluation using modern distribution comparison methods."""

    def __init__(self, verbose: bool = False):
        """
        Initialize the advanced statistical evaluator.

        Args:
            verbose: Whether to print detailed output during evaluation
        """
        self.verbose = verbose

    def wasserstein_evaluation(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        numerical_columns: List[str],
        include_2d: bool = False,
    ) -> Dict:
        """
        Comprehensive Wasserstein distance evaluation.

        Args:
            real: DataFrame containing real data
            fake: DataFrame containing synthetic data
            numerical_columns: List of numerical column names
            include_2d: Whether to include 2D pairwise analysis

        Returns:
            Dictionary containing comprehensive Wasserstein analysis results
        """
        if self.verbose:
            print("Computing Wasserstein distances...")

        results = {}

        try:
            # 1D Wasserstein distances (order 1 and 2)
            distances_p1 = wasserstein_distance_df(real, fake, numerical_columns, p=1)
            distances_p2 = wasserstein_distance_df(real, fake, numerical_columns, p=2)

            results["distances_p1"] = distances_p1
            results["distances_p2"] = distances_p2

            # Summary analysis
            summary = earth_movers_distance_summary(real, fake, numerical_columns)
            results["summary"] = summary

            # 2D analysis for column pairs (optional, computationally expensive)
            if include_2d and len(numerical_columns) >= 2:
                if self.verbose:
                    print("Computing 2D Wasserstein distances...")

                distances_2d = wasserstein_distance_df(
                    real, fake, numerical_columns, method="2d"
                )
                results["distances_2d"] = distances_2d

            # Per-column detailed analysis
            column_details = {}
            for col in numerical_columns:
                try:
                    cost, plan, bins = optimal_transport_cost(real[col], fake[col])
                    column_details[col] = {
                        "transport_cost": cost,
                        "transport_plan_shape": plan.shape,
                        "n_bins": len(bins) - 1,
                    }
                except Exception as e:
                    logger.warning(f"Failed to compute transport cost for {col}: {e}")
                    column_details[col] = {"error": str(e)}

            results["column_details"] = column_details

            # Overall quality metrics
            p1_distances = distances_p1["wasserstein_distance"].values
            results["quality_metrics"] = {
                "mean_wasserstein_p1": np.mean(p1_distances),
                "median_wasserstein_p1": np.median(p1_distances),
                "std_wasserstein_p1": np.std(p1_distances),
                "max_wasserstein_p1": np.max(p1_distances),
                "distribution_similarity_score": np.exp(-np.mean(p1_distances)),
                "quality_rating": self._rate_wasserstein_quality(np.mean(p1_distances)),
            }

        except Exception as e:
            logger.error(f"Error in Wasserstein evaluation: {e}")
            results["error"] = str(e)

        return results

    def mmd_evaluation(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        numerical_columns: List[str],
        kernel_types: Optional[List[str]] = None,
        include_multivariate: bool = True,
    ) -> Dict:
        """
        Comprehensive Maximum Mean Discrepancy evaluation.

        Args:
            real: DataFrame containing real data
            fake: DataFrame containing synthetic data
            numerical_columns: List of numerical column names
            kernel_types: List of kernels to use (default: ["rbf", "polynomial", "linear"])
            include_multivariate: Whether to include multivariate MMD analysis

        Returns:
            Dictionary containing comprehensive MMD analysis results
        """
        if kernel_types is None:
            kernel_types = ["rbf", "polynomial", "linear"]

        if self.verbose:
            print(f"Computing MMD with kernels: {kernel_types}")

        results = {}

        try:
            if include_multivariate:
                # Comprehensive analysis (both column-wise and multivariate)
                comprehensive = mmd_comprehensive_analysis(
                    real, fake, numerical_columns, kernel_types
                )
                results.update(comprehensive)
            else:
                # Column-wise analysis only
                results["column_wise"] = {}
                for kernel_type in kernel_types:
                    results["column_wise"][kernel_type] = mmd_column_wise(
                        real, fake, numerical_columns, kernel_type
                    )

            # Extract key metrics for summary
            if "summary" in results:
                summary = results["summary"]
                results["quality_metrics"] = {
                    "mean_mmd": summary["mean_mmd"],
                    "fraction_significant_differences": summary["fraction_significant"],
                    "overall_quality_score": summary["overall_quality_score"],
                    "mmd_rating": self._rate_mmd_quality(summary["mean_mmd"]),
                }

            # Best kernel analysis
            if "multivariate" in results:
                best_kernel_info = self._find_best_kernel(results["multivariate"])
                results["best_kernel"] = best_kernel_info

        except Exception as e:
            logger.error(f"Error in MMD evaluation: {e}")
            results["error"] = str(e)

        return results

    def comprehensive_evaluation(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        numerical_columns: List[str],
        wasserstein_config: Optional[Dict] = None,
        mmd_config: Optional[Dict] = None,
    ) -> Dict:
        """
        Run comprehensive advanced statistical evaluation.

        Args:
            real: DataFrame containing real data
            fake: DataFrame containing synthetic data
            numerical_columns: List of numerical column names
            wasserstein_config: Configuration for Wasserstein evaluation
            mmd_config: Configuration for MMD evaluation

        Returns:
            Dictionary with complete advanced statistical analysis
        """
        if wasserstein_config is None:
            wasserstein_config = {"include_2d": False}

        if mmd_config is None:
            mmd_config = {
                "kernel_types": ["rbf", "polynomial"],
                "include_multivariate": True,
            }

        if self.verbose:
            print("Running comprehensive advanced statistical evaluation...")

        results = {
            "wasserstein": {},
            "mmd": {},
            "combined_metrics": {},
            "recommendations": [],
        }

        # Wasserstein analysis
        try:
            results["wasserstein"] = self.wasserstein_evaluation(
                real, fake, numerical_columns, **wasserstein_config
            )
        except Exception as e:
            logger.error(f"Wasserstein evaluation failed: {e}")
            results["wasserstein"] = {"error": str(e)}

        # MMD analysis
        try:
            results["mmd"] = self.mmd_evaluation(
                real, fake, numerical_columns, **mmd_config
            )
        except Exception as e:
            logger.error(f"MMD evaluation failed: {e}")
            results["mmd"] = {"error": str(e)}

        # Combined analysis
        try:
            results["combined_metrics"] = self._combine_metrics(
                results["wasserstein"], results["mmd"]
            )
            results["recommendations"] = self._generate_recommendations(results)
        except Exception as e:
            logger.error(f"Combined analysis failed: {e}")
            results["combined_metrics"] = {"error": str(e)}

        return results

    def _rate_wasserstein_quality(self, mean_distance: float) -> str:
        """Rate the quality based on mean Wasserstein distance."""
        if mean_distance < 0.1:
            return "Excellent"
        elif mean_distance < 0.25:
            return "Good"
        elif mean_distance < 0.5:
            return "Fair"
        elif mean_distance < 1.0:
            return "Poor"
        else:
            return "Very Poor"

    def _rate_mmd_quality(self, mean_mmd: float) -> str:
        """Rate the quality based on mean MMD."""
        if np.isnan(mean_mmd):
            return "Unknown"

        if mean_mmd < 0.01:
            return "Excellent"
        elif mean_mmd < 0.05:
            return "Good"
        elif mean_mmd < 0.1:
            return "Fair"
        elif mean_mmd < 0.2:
            return "Poor"
        else:
            return "Very Poor"

    def _find_best_kernel(self, multivariate_results: Dict) -> Dict:
        """Find the kernel that gives the most discriminative results."""
        best_kernel = None
        best_score = -1

        for kernel_type, results in multivariate_results.items():
            if "error" not in results:
                # Score based on discriminative power (higher MMD = more discriminative)
                score = results.get("mmd_squared", 0)
                if score > best_score:
                    best_score = score
                    best_kernel = kernel_type

        return {
            "kernel": best_kernel,
            "mmd_squared": best_score,
            "discriminative_power": "High"
            if best_score > 0.1
            else "Medium"
            if best_score > 0.05
            else "Low",
        }

    def _combine_metrics(self, wasserstein_results: Dict, mmd_results: Dict) -> Dict:
        """Combine Wasserstein and MMD results into unified metrics."""
        combined = {}

        # Extract key metrics
        wasserstein_quality = wasserstein_results.get("quality_metrics", {})
        mmd_quality = mmd_results.get("quality_metrics", {})

        # Overall distribution similarity (0-1 scale, higher is better)
        wass_similarity = wasserstein_quality.get("distribution_similarity_score", 0)
        mmd_similarity = mmd_quality.get("overall_quality_score", 0)

        combined["overall_similarity"] = (wass_similarity + mmd_similarity) / 2

        # Agreement between methods
        wass_rating = wasserstein_quality.get("quality_rating", "Unknown")
        mmd_rating = mmd_quality.get("mmd_rating", "Unknown")

        combined["quality_consensus"] = (
            wass_rating if wass_rating == mmd_rating else "Mixed"
        )
        combined["wasserstein_rating"] = wass_rating
        combined["mmd_rating"] = mmd_rating

        # Statistical significance
        mmd_significant_fraction = mmd_quality.get(
            "fraction_significant_differences", 0
        )
        combined["statistical_significance"] = {
            "fraction_columns_different": mmd_significant_fraction,
            "interpretation": "High"
            if mmd_significant_fraction > 0.5
            else "Medium"
            if mmd_significant_fraction > 0.2
            else "Low",
        }

        return combined

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        combined = results.get("combined_metrics", {})
        wasserstein = results.get("wasserstein", {})

        # Overall similarity recommendation
        similarity = combined.get("overall_similarity", 0)
        if similarity < 0.3:
            recommendations.append(
                "Low overall similarity detected. Consider improving the data generation model."
            )
        elif similarity < 0.7:
            recommendations.append(
                "Moderate similarity achieved. Fine-tuning the model may improve results."
            )
        else:
            recommendations.append(
                "Good similarity achieved. The synthetic data quality is acceptable."
            )

        # Specific metric recommendations
        if "quality_metrics" in wasserstein:
            worst_wass = (
                wasserstein["summary"].get("overall_metrics", {}).get("worst_column")
            )
            if worst_wass:
                recommendations.append(
                    f"Column '{worst_wass}' has the highest Wasserstein distance. "
                    "Focus on improving this variable's distribution."
                )

        # Statistical significance recommendations
        sig_info = combined.get("statistical_significance", {})
        if sig_info.get("interpretation") == "High":
            recommendations.append(
                "Many columns show statistically significant differences. "
                "Review the data generation process for systematic biases."
            )

        return recommendations
