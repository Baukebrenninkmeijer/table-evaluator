"""Plugin architecture for extensible table evaluation."""

from .base_metric import (
    BaseMetric,
    StatisticalMetric,
    PrivacyMetric,
    MLEfficacyMetric,
    UtilityMetric,
    MetricMetadata,
    MetricConfig,
    MetricResult,
    MetricType,
    OutputFormat,
    MetricPlugin,
    MetricRegistry,
    metric_registry,
)

from .plugin_manager import PluginManager, plugin_manager

__all__ = [
    # Base classes
    "BaseMetric",
    "StatisticalMetric",
    "PrivacyMetric",
    "MLEfficacyMetric",
    "UtilityMetric",
    # Data classes
    "MetricMetadata",
    "MetricConfig",
    "MetricResult",
    "MetricPlugin",
    # Enums
    "MetricType",
    "OutputFormat",
    # Registry and manager
    "MetricRegistry",
    "metric_registry",
    "PluginManager",
    "plugin_manager",
]
