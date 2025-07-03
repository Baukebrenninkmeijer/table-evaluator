"""Wasserstein distance plugin for the plugin architecture."""

import time
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from table_evaluator.plugins.base_metric import (
    StatisticalMetric,
    MetricMetadata,
    MetricResult,
    MetricType,
    OutputFormat,
)
from table_evaluator.metrics.wasserstein import (
    wasserstein_distance_df,
    earth_movers_distance_summary,
)


class WassersteinDistanceMetric(StatisticalMetric):
    """Wasserstein distance metric plugin."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="wasserstein_distance",
            version="1.0.0",
            description="Calculate Wasserstein (Earth Mover's) distance between distributions",
            author="TableEvaluator Team",
            metric_type=MetricType.STATISTICAL,
            output_format=OutputFormat.DICT,
            dependencies=["scipy", "numpy", "pandas"],
            supports_backends=["pandas", "polars"],
        )

    def get_required_parameters(self) -> List[str]:
        return ["numerical_columns"]

    def get_optional_parameters(self) -> Dict[str, Any]:
        return {
            "p": 1,  # Order of Wasserstein distance
            "method": "1d",  # "1d" or "2d"
            "include_summary": True,
        }

    def calculate_statistic(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """Calculate Wasserstein distance statistics."""
        numerical_columns = kwargs.get("numerical_columns", [])
        p = kwargs.get("p", 1)
        method = kwargs.get("method", "1d")
        include_summary = kwargs.get("include_summary", True)

        if not numerical_columns:
            # Auto-detect numerical columns
            numerical_columns = real_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        # Calculate distances
        distances_df = wasserstein_distance_df(
            real_data, synthetic_data, numerical_columns, p=p, method=method
        )

        results = {
            "distances": distances_df.to_dict("records"),
            "mean_distance": distances_df["wasserstein_distance"].mean(),
            "max_distance": distances_df["wasserstein_distance"].max(),
            "min_distance": distances_df["wasserstein_distance"].min(),
        }

        if include_summary:
            summary = earth_movers_distance_summary(
                real_data, synthetic_data, numerical_columns
            )
            results["summary"] = summary

        return results

    def evaluate(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> MetricResult:
        """Evaluate Wasserstein distance metric."""
        start_time = time.time()
        warnings = []

        try:
            # Validate inputs
            numerical_columns = kwargs.get("numerical_columns", [])
            if not numerical_columns:
                # Auto-detect numerical columns
                numerical_columns = real_data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                if not numerical_columns:
                    return MetricResult(
                        metric_name=self.metadata.name,
                        success=False,
                        value=None,
                        metadata={"error": "No numerical columns found"},
                        execution_time=time.time() - start_time,
                        error_message="No numerical columns available for Wasserstein distance calculation",
                    )
                warnings.append(f"Auto-detected numerical columns: {numerical_columns}")

            # Calculate statistics
            result_value = self.calculate_statistic(
                real_data, synthetic_data, numerical_columns=numerical_columns, **kwargs
            )

            # Create metadata
            metadata = {
                "columns_analyzed": numerical_columns,
                "parameters": {
                    "p": kwargs.get("p", 1),
                    "method": kwargs.get("method", "1d"),
                },
                "data_shapes": {
                    "real": real_data.shape,
                    "synthetic": synthetic_data.shape,
                },
            }

            return MetricResult(
                metric_name=self.metadata.name,
                success=True,
                value=result_value,
                metadata=metadata,
                execution_time=time.time() - start_time,
                warnings=warnings,
            )

        except Exception as e:
            return MetricResult(
                metric_name=self.metadata.name,
                success=False,
                value=None,
                metadata={},
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def validate_data(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> bool:
        """Validate data for Wasserstein distance calculation."""
        # Call parent validation
        if not super().validate_data(real_data, synthetic_data):
            return False

        # Check for numerical columns
        real_numerical = real_data.select_dtypes(include=[np.number]).columns
        synthetic_numerical = synthetic_data.select_dtypes(include=[np.number]).columns

        if len(real_numerical) == 0 or len(synthetic_numerical) == 0:
            return False

        # Check that numerical columns match
        if set(real_numerical) != set(synthetic_numerical):
            return False

        return True
