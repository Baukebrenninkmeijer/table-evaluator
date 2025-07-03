"""Base classes and interfaces for plugin architecture."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class MetricType(Enum):
    """Types of evaluation metrics."""

    STATISTICAL = "statistical"
    PRIVACY = "privacy"
    ML_EFFICACY = "ml_efficacy"
    UTILITY = "utility"
    CUSTOM = "custom"


class OutputFormat(Enum):
    """Supported output formats for metric results."""

    DATAFRAME = "dataframe"
    DICT = "dict"
    SCALAR = "scalar"
    VISUALIZATION = "visualization"


@dataclass
class MetricMetadata:
    """Metadata for a metric plugin."""

    name: str
    version: str
    description: str
    author: str
    metric_type: MetricType
    output_format: OutputFormat
    dependencies: List[str]
    min_python_version: str = "3.8"
    min_pandas_version: str = "1.5.0"
    supports_backends: List[str] = None  # ["pandas", "polars"] or None for all

    def __post_init__(self):
        if self.supports_backends is None:
            self.supports_backends = ["pandas", "polars"]


@dataclass
class MetricConfig:
    """Configuration for metric execution."""

    enabled: bool = True
    parameters: Dict[str, Any] = None
    timeout_seconds: Optional[int] = None
    cache_results: bool = True
    parallel_execution: bool = False

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class MetricResult:
    """Result from metric evaluation."""

    metric_name: str
    success: bool
    value: Any
    metadata: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""

    def __init__(self, config: Optional[MetricConfig] = None):
        """
        Initialize the metric with configuration.

        Args:
            config: Configuration for the metric
        """
        self.config = config or MetricConfig()
        self._metadata = None

    @property
    @abstractmethod
    def metadata(self) -> MetricMetadata:
        """Return metadata about this metric."""
        pass

    @abstractmethod
    def evaluate(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> MetricResult:
        """
        Evaluate the metric on real and synthetic data.

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            **kwargs: Additional parameters for evaluation

        Returns:
            MetricResult containing evaluation results
        """
        pass

    def validate_data(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> bool:
        """
        Validate that the data is suitable for this metric.

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data

        Returns:
            True if data is valid, False otherwise
        """
        # Basic validation
        if real_data.empty or synthetic_data.empty:
            return False

        # Check column compatibility
        if set(real_data.columns) != set(synthetic_data.columns):
            return False

        return True

    def get_required_parameters(self) -> List[str]:
        """
        Get list of required parameters for this metric.

        Returns:
            List of required parameter names
        """
        return []

    def get_optional_parameters(self) -> Dict[str, Any]:
        """
        Get dictionary of optional parameters with default values.

        Returns:
            Dictionary of parameter names and default values
        """
        return {}

    def supports_backend(self, backend: str) -> bool:
        """
        Check if this metric supports a specific backend.

        Args:
            backend: Backend name ("pandas", "polars", etc.)

        Returns:
            True if backend is supported
        """
        return backend in self.metadata.supports_backends


class StatisticalMetric(BaseMetric):
    """Base class for statistical evaluation metrics."""

    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)

    @abstractmethod
    def calculate_statistic(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate the statistical measure.

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            **kwargs: Additional parameters

        Returns:
            Statistical measure value(s)
        """
        pass


class PrivacyMetric(BaseMetric):
    """Base class for privacy evaluation metrics."""

    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)

    @abstractmethod
    def assess_privacy_risk(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Assess privacy risk.

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            **kwargs: Additional parameters

        Returns:
            Dictionary with privacy risk assessment
        """
        pass


class MLEfficacyMetric(BaseMetric):
    """Base class for machine learning efficacy metrics."""

    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)

    @abstractmethod
    def evaluate_ml_utility(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate machine learning utility.

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            target_column: Name of target column for ML tasks
            **kwargs: Additional parameters

        Returns:
            Dictionary with ML utility assessment
        """
        pass


class UtilityMetric(BaseMetric):
    """Base class for data utility metrics."""

    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)

    @abstractmethod
    def measure_utility(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs
    ) -> Union[float, Dict[str, float]]:
        """
        Measure data utility.

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            **kwargs: Additional parameters

        Returns:
            Utility measure value(s)
        """
        pass


class MetricPlugin:
    """Container for a metric plugin with its metadata and implementation."""

    def __init__(self, metric_class: type, metadata: MetricMetadata):
        """
        Initialize the plugin.

        Args:
            metric_class: Class implementing the metric
            metadata: Metadata about the metric
        """
        self.metric_class = metric_class
        self.metadata = metadata
        self._instance = None

    def create_instance(self, config: Optional[MetricConfig] = None) -> BaseMetric:
        """
        Create an instance of the metric.

        Args:
            config: Configuration for the metric instance

        Returns:
            Configured metric instance
        """
        return self.metric_class(config)

    def get_instance(self, config: Optional[MetricConfig] = None) -> BaseMetric:
        """
        Get a cached instance or create a new one.

        Args:
            config: Configuration for the metric instance

        Returns:
            Metric instance
        """
        if self._instance is None or config is not None:
            self._instance = self.create_instance(config)
        return self._instance

    def validate_dependencies(self) -> bool:
        """
        Validate that all dependencies are available.

        Returns:
            True if all dependencies are satisfied
        """
        try:
            for dependency in self.metadata.dependencies:
                __import__(dependency)
            return True
        except ImportError:
            return False


class MetricRegistry:
    """Registry for managing available metrics."""

    def __init__(self):
        self._metrics: Dict[str, MetricPlugin] = {}
        self._metrics_by_type: Dict[MetricType, List[str]] = {
            metric_type: [] for metric_type in MetricType
        }

    def register_metric(self, plugin: MetricPlugin) -> bool:
        """
        Register a metric plugin.

        Args:
            plugin: MetricPlugin to register

        Returns:
            True if registration was successful
        """
        if not plugin.validate_dependencies():
            return False

        name = plugin.metadata.name
        if name in self._metrics:
            # Handle version conflicts
            existing_version = self._metrics[name].metadata.version
            new_version = plugin.metadata.version

            # Simple version comparison (assumes semantic versioning)
            if self._compare_versions(new_version, existing_version) <= 0:
                return False  # Don't replace with older version

        self._metrics[name] = plugin

        # Update type index
        metric_type = plugin.metadata.metric_type
        if name not in self._metrics_by_type[metric_type]:
            self._metrics_by_type[metric_type].append(name)

        return True

    def get_metric(self, name: str) -> Optional[MetricPlugin]:
        """
        Get a metric plugin by name.

        Args:
            name: Name of the metric

        Returns:
            MetricPlugin or None if not found
        """
        return self._metrics.get(name)

    def get_metrics_by_type(self, metric_type: MetricType) -> List[MetricPlugin]:
        """
        Get all metrics of a specific type.

        Args:
            metric_type: Type of metrics to retrieve

        Returns:
            List of MetricPlugin instances
        """
        return [
            self._metrics[name]
            for name in self._metrics_by_type[metric_type]
            if name in self._metrics
        ]

    def list_available_metrics(self) -> Dict[str, MetricMetadata]:
        """
        List all available metrics with their metadata.

        Returns:
            Dictionary mapping metric names to metadata
        """
        return {name: plugin.metadata for name, plugin in self._metrics.items()}

    def unregister_metric(self, name: str) -> bool:
        """
        Unregister a metric plugin.

        Args:
            name: Name of the metric to unregister

        Returns:
            True if unregistration was successful
        """
        if name not in self._metrics:
            return False

        plugin = self._metrics[name]
        metric_type = plugin.metadata.metric_type

        # Remove from type index
        if name in self._metrics_by_type[metric_type]:
            self._metrics_by_type[metric_type].remove(name)

        # Remove from main registry
        del self._metrics[name]

        return True

    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings.

        Args:
            version1: First version string
            version2: Second version string

        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """

        def parse_version(version: str) -> List[int]:
            return [int(x) for x in version.split(".")]

        v1_parts = parse_version(version1)
        v2_parts = parse_version(version2)

        # Pad with zeros to same length
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))

        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1

        return 0


# Global registry instance
metric_registry = MetricRegistry()
