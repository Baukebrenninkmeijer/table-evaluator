"""Privacy evaluation plugin for the plugin architecture."""

import time
from typing import Any, Dict, List

import pandas as pd

from table_evaluator.plugins.base_metric import (
    PrivacyMetric,
    MetricMetadata,
    MetricResult,
    MetricType,
    OutputFormat,
)
from table_evaluator.metrics.privacy_attacks import (
    comprehensive_privacy_analysis,
    identify_quasi_identifiers,
)


class PrivacyAnalysisMetric(PrivacyMetric):
    """Comprehensive privacy analysis metric plugin."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="privacy_analysis",
            version="1.0.0",
            description="Comprehensive privacy analysis including k-anonymity and membership inference",
            author="TableEvaluator Team",
            metric_type=MetricType.PRIVACY,
            output_format=OutputFormat.DICT,
            dependencies=["scikit-learn", "numpy", "pandas"],
            supports_backends=["pandas", "polars"],
        )

    def get_required_parameters(self) -> List[str]:
        return []  # All parameters are optional with sensible defaults

    def get_optional_parameters(self) -> Dict[str, Any]:
        return {
            "quasi_identifiers": None,  # Auto-detected if None
            "sensitive_attributes": None,
            "include_k_anonymity": True,
            "include_membership_inference": True,
            "max_samples_attack": 5000,
        }

    def assess_privacy_risk(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """Assess privacy risk using multiple techniques."""
        quasi_identifiers = kwargs.get("quasi_identifiers")
        sensitive_attributes = kwargs.get("sensitive_attributes")

        # Auto-detect quasi-identifiers if not provided
        if quasi_identifiers is None:
            quasi_identifiers = identify_quasi_identifiers(synthetic_data)

        # Run comprehensive analysis
        results = comprehensive_privacy_analysis(
            real_data, synthetic_data, quasi_identifiers, sensitive_attributes
        )

        return results

    def evaluate(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> MetricResult:
        """Evaluate privacy analysis metric."""
        start_time = time.time()
        warnings = []

        try:
            # Limit data size for computational efficiency in membership inference
            max_samples = kwargs.get("max_samples_attack", 5000)
            real_sample = real_data
            synthetic_sample = synthetic_data

            if len(real_data) > max_samples:
                real_sample = real_data.sample(n=max_samples, random_state=42)
                warnings.append(
                    f"Sampled {max_samples} rows from real data for membership inference"
                )

            if len(synthetic_data) > max_samples:
                synthetic_sample = synthetic_data.sample(n=max_samples, random_state=42)
                warnings.append(
                    f"Sampled {max_samples} rows from synthetic data for membership inference"
                )

            # Run privacy analysis
            result_value = self.assess_privacy_risk(
                real_sample, synthetic_sample, **kwargs
            )

            # Extract key metrics for metadata
            k_anon = result_value.get("k_anonymity", {})
            overall = result_value.get("overall_assessment", {})

            metadata = {
                "parameters": {
                    "quasi_identifiers": kwargs.get(
                        "quasi_identifiers", "auto-detected"
                    ),
                    "sensitive_attributes": kwargs.get("sensitive_attributes"),
                    "max_samples_attack": max_samples,
                },
                "data_shapes": {
                    "real": real_sample.shape,
                    "synthetic": synthetic_sample.shape,
                    "original_real": real_data.shape,
                    "original_synthetic": synthetic_data.shape,
                },
                "key_findings": {
                    "k_value": k_anon.get("k_value", "N/A"),
                    "overall_risk": overall.get("overall_risk_level", "Unknown"),
                    "privacy_score": overall.get("privacy_score", 0.0),
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
        """Validate data for privacy analysis."""
        # Call parent validation
        if not super().validate_data(real_data, synthetic_data):
            return False

        # Check minimum data size for meaningful analysis
        if len(real_data) < 10 or len(synthetic_data) < 10:
            return False

        # Check that we have some columns to analyze
        if len(real_data.columns) == 0:
            return False

        return True


class KAnonymityMetric(PrivacyMetric):
    """K-anonymity specific metric plugin."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="k_anonymity",
            version="1.0.0",
            description="K-anonymity analysis for privacy evaluation",
            author="TableEvaluator Team",
            metric_type=MetricType.PRIVACY,
            output_format=OutputFormat.DICT,
            dependencies=["pandas", "numpy"],
            supports_backends=["pandas", "polars"],
        )

    def get_required_parameters(self) -> List[str]:
        return []

    def get_optional_parameters(self) -> Dict[str, Any]:
        return {
            "quasi_identifiers": None,  # Auto-detected if None
            "sensitive_attributes": None,
        }

    def assess_privacy_risk(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """Assess k-anonymity privacy risk."""
        from table_evaluator.metrics.privacy_attacks import (
            calculate_k_anonymity,
            identify_quasi_identifiers,
        )

        quasi_identifiers = kwargs.get("quasi_identifiers")
        sensitive_attributes = kwargs.get("sensitive_attributes")

        # Auto-detect quasi-identifiers if not provided
        if quasi_identifiers is None:
            quasi_identifiers = identify_quasi_identifiers(synthetic_data)

        # Calculate k-anonymity for synthetic data
        k_anon_results = calculate_k_anonymity(
            synthetic_data, quasi_identifiers, sensitive_attributes
        )

        return {
            "k_anonymity": k_anon_results,
            "quasi_identifiers_used": quasi_identifiers,
            "sensitive_attributes_used": sensitive_attributes,
        }

    def evaluate(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> MetricResult:
        """Evaluate k-anonymity metric."""
        start_time = time.time()
        warnings = []

        try:
            quasi_identifiers = kwargs.get("quasi_identifiers")
            if quasi_identifiers is None:
                quasi_identifiers = identify_quasi_identifiers(synthetic_data)
                warnings.append(f"Auto-detected quasi-identifiers: {quasi_identifiers}")

            result_value = self.assess_privacy_risk(real_data, synthetic_data, **kwargs)

            k_value = result_value["k_anonymity"].get("k_value", 0)
            anonymity_level = result_value["k_anonymity"].get(
                "anonymity_level", "Unknown"
            )

            metadata = {
                "parameters": {
                    "quasi_identifiers": quasi_identifiers,
                    "sensitive_attributes": kwargs.get("sensitive_attributes"),
                },
                "key_findings": {
                    "k_value": k_value,
                    "anonymity_level": anonymity_level,
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
