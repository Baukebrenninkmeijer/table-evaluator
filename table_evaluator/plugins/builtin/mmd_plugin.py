"""Maximum Mean Discrepancy (MMD) plugin for the plugin architecture."""

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
from table_evaluator.metrics.mmd import mmd_column_wise, mmd_comprehensive_analysis


class MMDMetric(StatisticalMetric):
    """Maximum Mean Discrepancy metric plugin."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="mmd",
            version="1.0.0",
            description="Calculate Maximum Mean Discrepancy using various kernels",
            author="TableEvaluator Team",
            metric_type=MetricType.STATISTICAL,
            output_format=OutputFormat.DICT,
            dependencies=["scikit-learn", "numpy", "pandas"],
            supports_backends=["pandas", "polars"],
        )

    def get_required_parameters(self) -> List[str]:
        return ["numerical_columns"]

    def get_optional_parameters(self) -> Dict[str, Any]:
        return {
            "kernel_types": ["rbf", "polynomial", "linear"],
            "include_multivariate": True,
            "max_samples": 5000,
            "n_permutations": 100,
        }

    def calculate_statistic(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """Calculate MMD statistics."""
        numerical_columns = kwargs.get("numerical_columns", [])
        kernel_types = kwargs.get("kernel_types", ["rbf", "polynomial", "linear"])
        include_multivariate = kwargs.get("include_multivariate", True)

        if not numerical_columns:
            # Auto-detect numerical columns
            numerical_columns = real_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        if include_multivariate:
            # Comprehensive analysis
            results = mmd_comprehensive_analysis(
                real_data, synthetic_data, numerical_columns, kernel_types
            )
        else:
            # Column-wise only
            results = {"column_wise": {}}
            for kernel_type in kernel_types:
                results["column_wise"][kernel_type] = mmd_column_wise(
                    real_data, synthetic_data, numerical_columns, kernel_type
                )

        return results

    def evaluate(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> MetricResult:
        """Evaluate MMD metric."""
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
                        error_message="No numerical columns available for MMD calculation",
                    )
                warnings.append(f"Auto-detected numerical columns: {numerical_columns}")

            # Limit data size for computational efficiency
            max_samples = kwargs.get("max_samples", 5000)
            real_sample = real_data
            synthetic_sample = synthetic_data

            if len(real_data) > max_samples:
                real_sample = real_data.sample(n=max_samples, random_state=42)
                warnings.append(f"Sampled {max_samples} rows from real data")

            if len(synthetic_data) > max_samples:
                synthetic_sample = synthetic_data.sample(n=max_samples, random_state=42)
                warnings.append(f"Sampled {max_samples} rows from synthetic data")

            # Calculate statistics
            result_value = self.calculate_statistic(
                real_sample,
                synthetic_sample,
                numerical_columns=numerical_columns,
                **kwargs,
            )

            # Create metadata
            metadata = {
                "columns_analyzed": numerical_columns,
                "parameters": {
                    "kernel_types": kwargs.get(
                        "kernel_types", ["rbf", "polynomial", "linear"]
                    ),
                    "include_multivariate": kwargs.get("include_multivariate", True),
                    "max_samples": max_samples,
                },
                "data_shapes": {
                    "real": real_sample.shape,
                    "synthetic": synthetic_sample.shape,
                    "original_real": real_data.shape,
                    "original_synthetic": synthetic_data.shape,
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
        """Validate data for MMD calculation."""
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

        # Check minimum data size
        if len(real_data) < 10 or len(synthetic_data) < 10:
            return False

        return True
