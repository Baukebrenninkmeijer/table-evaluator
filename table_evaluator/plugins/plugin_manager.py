"""Plugin manager for dynamic loading and execution of evaluation metrics."""

import importlib
import importlib.util
import inspect
import logging
import pkgutil
import time
from typing import Any, Dict, List, Optional, Type
import concurrent.futures
from dataclasses import asdict

import pandas as pd

from .base_metric import (
    BaseMetric,
    MetricConfig,
    MetricMetadata,
    MetricPlugin,
    MetricResult,
    MetricType,
    metric_registry,
)

logger = logging.getLogger(__name__)


class PluginManager:
    """Manager for loading, configuring, and executing metric plugins."""

    def __init__(self, plugin_directories: Optional[List[str]] = None):
        """
        Initialize the plugin manager.

        Args:
            plugin_directories: List of directories to search for plugins
        """
        self.plugin_directories = plugin_directories or []
        self.loaded_plugins: Dict[str, MetricPlugin] = {}
        self.execution_cache: Dict[str, MetricResult] = {}
        self.default_config = MetricConfig()

    def discover_plugins(
        self, package_name: str = "table_evaluator.plugins.builtin"
    ) -> int:
        """
        Discover and load plugins from specified package.

        Args:
            package_name: Python package to search for plugins

        Returns:
            Number of plugins discovered and loaded
        """
        loaded_count = 0

        try:
            # Import the package
            package = importlib.import_module(package_name)
            package_path = package.__path__

            # Walk through all modules in the package
            for _, module_name, _ in pkgutil.iter_modules(package_path):
                full_module_name = f"{package_name}.{module_name}"

                try:
                    module = importlib.import_module(full_module_name)
                    loaded_count += self._load_metrics_from_module(module)
                except Exception as e:
                    logger.warning(
                        f"Failed to load plugin module {full_module_name}: {e}"
                    )

        except ImportError as e:
            logger.error(f"Failed to import plugin package {package_name}: {e}")

        return loaded_count

    def load_plugin_from_file(self, file_path: str) -> bool:
        """
        Load a plugin from a Python file.

        Args:
            file_path: Path to the Python file containing the plugin

        Returns:
            True if plugin was loaded successfully
        """
        try:
            # Import module from file
            spec = importlib.util.spec_from_file_location("plugin_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            return self._load_metrics_from_module(module) > 0

        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")
            return False

    def register_plugin(
        self, metric_class: Type[BaseMetric], metadata: Optional[MetricMetadata] = None
    ) -> bool:
        """
        Register a metric plugin programmatically.

        Args:
            metric_class: Class implementing BaseMetric
            metadata: Metadata for the metric (auto-detected if None)

        Returns:
            True if registration was successful
        """
        try:
            if metadata is None:
                # Try to get metadata from the class
                if hasattr(metric_class, "metadata") and isinstance(
                    metric_class.metadata, property
                ):
                    # Create temporary instance to get metadata
                    temp_instance = metric_class()
                    metadata = temp_instance.metadata
                else:
                    logger.error(f"No metadata provided for {metric_class.__name__}")
                    return False

            plugin = MetricPlugin(metric_class, metadata)
            success = metric_registry.register_metric(plugin)

            if success:
                self.loaded_plugins[metadata.name] = plugin
                logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")

            return success

        except Exception as e:
            logger.error(f"Failed to register plugin {metric_class.__name__}: {e}")
            return False

    def execute_metric(
        self,
        metric_name: str,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        config: Optional[MetricConfig] = None,
        **kwargs,
    ) -> MetricResult:
        """
        Execute a specific metric.

        Args:
            metric_name: Name of the metric to execute
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            config: Configuration for metric execution
            **kwargs: Additional parameters for the metric

        Returns:
            MetricResult with execution results
        """
        start_time = time.time()

        try:
            # Get plugin
            plugin = metric_registry.get_metric(metric_name)
            if plugin is None:
                return MetricResult(
                    metric_name=metric_name,
                    success=False,
                    value=None,
                    metadata={},
                    execution_time=time.time() - start_time,
                    error_message=f"Metric '{metric_name}' not found",
                )

            # Use provided config or default
            effective_config = config or self.default_config

            # Check cache if enabled
            cache_key = self._generate_cache_key(
                metric_name, real_data, synthetic_data, effective_config, kwargs
            )
            if effective_config.cache_results and cache_key in self.execution_cache:
                cached_result = self.execution_cache[cache_key]
                logger.debug(f"Using cached result for {metric_name}")
                return cached_result

            # Create metric instance
            metric_instance = plugin.get_instance(effective_config)

            # Validate data
            if not metric_instance.validate_data(real_data, synthetic_data):
                return MetricResult(
                    metric_name=metric_name,
                    success=False,
                    value=None,
                    metadata={},
                    execution_time=time.time() - start_time,
                    error_message="Data validation failed",
                )

            # Execute with timeout if specified
            if effective_config.timeout_seconds:
                result = self._execute_with_timeout(
                    metric_instance,
                    real_data,
                    synthetic_data,
                    effective_config.timeout_seconds,
                    **kwargs,
                )
            else:
                result = metric_instance.evaluate(real_data, synthetic_data, **kwargs)

            # Update execution time
            result.execution_time = time.time() - start_time

            # Cache result if enabled
            if effective_config.cache_results:
                self.execution_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Error executing metric {metric_name}: {e}")
            return MetricResult(
                metric_name=metric_name,
                success=False,
                value=None,
                metadata={},
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def execute_metrics_parallel(
        self,
        metric_names: List[str],
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        configs: Optional[Dict[str, MetricConfig]] = None,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, MetricResult]:
        """
        Execute multiple metrics in parallel.

        Args:
            metric_names: List of metric names to execute
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            configs: Dictionary mapping metric names to configs
            max_workers: Maximum number of parallel workers
            **kwargs: Additional parameters for metrics

        Returns:
            Dictionary mapping metric names to results
        """
        if configs is None:
            configs = {}

        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all metric executions
            future_to_metric = {
                executor.submit(
                    self.execute_metric,
                    metric_name,
                    real_data,
                    synthetic_data,
                    configs.get(metric_name),
                    **kwargs,
                ): metric_name
                for metric_name in metric_names
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    result = future.result()
                    results[metric_name] = result
                except Exception as e:
                    logger.error(f"Parallel execution failed for {metric_name}: {e}")
                    results[metric_name] = MetricResult(
                        metric_name=metric_name,
                        success=False,
                        value=None,
                        metadata={},
                        execution_time=0.0,
                        error_message=str(e),
                    )

        return results

    def get_available_metrics(
        self, metric_type: Optional[MetricType] = None, backend: Optional[str] = None
    ) -> Dict[str, MetricMetadata]:
        """
        Get available metrics with optional filtering.

        Args:
            metric_type: Filter by metric type
            backend: Filter by supported backend

        Returns:
            Dictionary mapping metric names to metadata
        """
        all_metrics = metric_registry.list_available_metrics()

        # Apply filters
        filtered_metrics = {}
        for name, metadata in all_metrics.items():
            # Filter by type
            if metric_type is not None and metadata.metric_type != metric_type:
                continue

            # Filter by backend
            if backend is not None and backend not in metadata.supports_backends:
                continue

            filtered_metrics[name] = metadata

        return filtered_metrics

    def get_metric_info(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with detailed metric information
        """
        plugin = metric_registry.get_metric(metric_name)
        if plugin is None:
            return None

        # Get instance to access methods
        instance = plugin.get_instance()

        return {
            "metadata": asdict(plugin.metadata),
            "required_parameters": instance.get_required_parameters(),
            "optional_parameters": instance.get_optional_parameters(),
            "dependencies_satisfied": plugin.validate_dependencies(),
        }

    def validate_metric_config(
        self, metric_name: str, config: MetricConfig
    ) -> List[str]:
        """
        Validate configuration for a specific metric.

        Args:
            metric_name: Name of the metric
            config: Configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        plugin = metric_registry.get_metric(metric_name)
        if plugin is None:
            errors.append(f"Metric '{metric_name}' not found")
            return errors

        try:
            instance = plugin.get_instance(config)
            required_params = instance.get_required_parameters()

            # Check required parameters
            for param in required_params:
                if param not in config.parameters:
                    errors.append(f"Missing required parameter: {param}")

            # Validate timeout
            if config.timeout_seconds is not None and config.timeout_seconds <= 0:
                errors.append("Timeout must be positive")

        except Exception as e:
            errors.append(f"Configuration validation error: {e}")

        return errors

    def clear_cache(self, metric_name: Optional[str] = None) -> None:
        """
        Clear execution cache.

        Args:
            metric_name: Clear cache for specific metric (all if None)
        """
        if metric_name is None:
            self.execution_cache.clear()
        else:
            # Remove cache entries for specific metric
            keys_to_remove = [
                key
                for key in self.execution_cache.keys()
                if key.startswith(f"{metric_name}:")
            ]
            for key in keys_to_remove:
                del self.execution_cache[key]

    def _load_metrics_from_module(self, module) -> int:
        """Load metric classes from a module."""
        loaded_count = 0

        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseMetric)
                and obj != BaseMetric
                and not inspect.isabstract(obj)
            ):  # Skip abstract classes
                try:
                    # Try to register the metric
                    if self.register_plugin(obj):
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to register metric {name}: {e}")

        return loaded_count

    def _generate_cache_key(
        self,
        metric_name: str,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        config: MetricConfig,
        kwargs: Dict[str, Any],
    ) -> str:
        """Generate a cache key for metric execution."""
        # Create a simple hash-based key
        # In production, you might want a more sophisticated approach
        real_hash = pd.util.hash_pandas_object(real_data).sum()
        synthetic_hash = pd.util.hash_pandas_object(synthetic_data).sum()
        config_str = str(sorted(config.parameters.items()))
        kwargs_str = str(sorted(kwargs.items()))

        return f"{metric_name}:{real_hash}:{synthetic_hash}:{hash(config_str)}:{hash(kwargs_str)}"

    def _execute_with_timeout(
        self,
        metric_instance: BaseMetric,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        timeout_seconds: int,
        **kwargs,
    ) -> MetricResult:
        """Execute metric with timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                metric_instance.evaluate, real_data, synthetic_data, **kwargs
            )

            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                return MetricResult(
                    metric_name=metric_instance.metadata.name,
                    success=False,
                    value=None,
                    metadata={},
                    execution_time=timeout_seconds,
                    error_message=f"Execution timed out after {timeout_seconds} seconds",
                )


# Global plugin manager instance
plugin_manager = PluginManager()
