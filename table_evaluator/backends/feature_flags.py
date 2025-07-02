"""Feature flags system for enabling/disabling backend features based on availability."""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureFlags:
    """Feature flags system for controlling backend availability and features.

    This class manages feature flags that control which backends and features
    are available based on dependencies, environment variables, and configuration.
    """

    def __init__(self):
        """Initialize feature flags system."""
        self._flags: Dict[str, bool] = {}
        self._initialize_default_flags()
        self._load_environment_overrides()

    def _initialize_default_flags(self) -> None:
        """Initialize default feature flags based on available dependencies."""
        # Backend availability flags
        try:
            import polars

            self._flags["polars_backend"] = True
            self._flags["polars_lazy_evaluation"] = True
            logger.debug("Polars backend enabled (dependency available)")
        except ImportError:
            self._flags["polars_backend"] = False
            self._flags["polars_lazy_evaluation"] = False
            logger.debug("Polars backend disabled (dependency not available)")

        # Always enable pandas backend
        self._flags["pandas_backend"] = True

        # Feature flags for specific capabilities
        self._flags["automatic_backend_detection"] = True
        self._flags["lazy_conversion"] = True
        self._flags["memory_optimization"] = True
        self._flags["schema_validation"] = True
        self._flags["dtype_preservation"] = True

        # Performance features
        self._flags["conversion_caching"] = True
        self._flags["performance_heuristics"] = True

        # Development and debugging features
        self._flags["conversion_logging"] = False
        self._flags["performance_monitoring"] = False
        self._flags["strict_validation"] = False

    def _load_environment_overrides(self) -> None:
        """Load feature flag overrides from environment variables."""
        env_prefix = "TABLE_EVALUATOR_"

        for key in self._flags.keys():
            env_key = f"{env_prefix}{key.upper()}"
            env_value = os.getenv(env_key)

            if env_value is not None:
                # Convert string to boolean
                if env_value.lower() in ("true", "1", "yes", "on"):
                    self._flags[key] = True
                    logger.debug(f"Environment override: {key} = True")
                elif env_value.lower() in ("false", "0", "no", "off"):
                    self._flags[key] = False
                    logger.debug(f"Environment override: {key} = False")
                else:
                    logger.warning(f"Invalid boolean value for {env_key}: {env_value}")

    def is_enabled(self, flag: str) -> bool:
        """Check if a feature flag is enabled.

        Args:
            flag: Feature flag name

        Returns:
            True if feature is enabled, False otherwise
        """
        return self._flags.get(flag, False)

    def enable(self, flag: str) -> None:
        """Enable a feature flag.

        Args:
            flag: Feature flag name
        """
        if flag in self._flags:
            self._flags[flag] = True
            logger.debug(f"Enabled feature flag: {flag}")
        else:
            logger.warning(f"Unknown feature flag: {flag}")

    def disable(self, flag: str) -> None:
        """Disable a feature flag.

        Args:
            flag: Feature flag name
        """
        if flag in self._flags:
            self._flags[flag] = False
            logger.debug(f"Disabled feature flag: {flag}")
        else:
            logger.warning(f"Unknown feature flag: {flag}")

    def set_flag(self, flag: str, value: bool) -> None:
        """Set a feature flag to a specific value.

        Args:
            flag: Feature flag name
            value: Boolean value to set
        """
        if value:
            self.enable(flag)
        else:
            self.disable(flag)

    def get_all_flags(self) -> Dict[str, bool]:
        """Get all feature flags and their current values.

        Returns:
            Dictionary of flag names and their boolean values
        """
        return self._flags.copy()

    def get_backend_flags(self) -> Dict[str, bool]:
        """Get backend-specific feature flags.

        Returns:
            Dictionary of backend-related flags
        """
        backend_flags = {}
        for key, value in self._flags.items():
            if "backend" in key:
                backend_flags[key] = value
        return backend_flags

    def get_available_backends(self) -> List[str]:
        """Get list of available backends based on feature flags.

        Returns:
            List of available backend names
        """
        backends = []
        if self.is_enabled("pandas_backend"):
            backends.append("pandas")
        if self.is_enabled("polars_backend"):
            backends.append("polars")
        return backends

    def require_backend(self, backend: str) -> None:
        """Require a specific backend to be available.

        Args:
            backend: Backend name

        Raises:
            RuntimeError: If required backend is not available
        """
        flag_name = f"{backend}_backend"
        if not self.is_enabled(flag_name):
            raise RuntimeError(f"Required backend '{backend}' is not available")

    def configure_for_environment(self, environment: str) -> None:
        """Configure feature flags for specific environment.

        Args:
            environment: Environment name ("development", "production", "testing")
        """
        if environment == "development":
            self.enable("conversion_logging")
            self.enable("performance_monitoring")
            self.disable("strict_validation")

        elif environment == "production":
            self.disable("conversion_logging")
            self.disable("performance_monitoring")
            self.enable("strict_validation")

        elif environment == "testing":
            self.enable("conversion_logging")
            self.enable("performance_monitoring")
            self.enable("strict_validation")

        else:
            logger.warning(f"Unknown environment: {environment}")

        logger.info(f"Configured feature flags for {environment} environment")

    def validate_configuration(self) -> List[str]:
        """Validate current feature flag configuration.

        Returns:
            List of configuration warnings/issues
        """
        issues = []

        # Check for conflicting flags
        if not self.is_enabled("pandas_backend") and not self.is_enabled(
            "polars_backend"
        ):
            issues.append("No backends are enabled")

        if self.is_enabled("polars_lazy_evaluation") and not self.is_enabled(
            "polars_backend"
        ):
            issues.append("Polars lazy evaluation enabled but Polars backend disabled")

        if self.is_enabled("conversion_caching") and not self.is_enabled(
            "lazy_conversion"
        ):
            issues.append("Conversion caching enabled but lazy conversion disabled")

        # Check for dependency availability
        if self.is_enabled("polars_backend"):
            try:
                import polars
            except ImportError:
                issues.append(
                    "Polars backend enabled but polars dependency not available"
                )

        return issues

    def reset_to_defaults(self) -> None:
        """Reset all feature flags to their default values."""
        self._flags.clear()
        self._initialize_default_flags()
        logger.info("Reset feature flags to defaults")

    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for serialization.

        Returns:
            Dictionary with configuration details
        """
        return {
            "flags": self._flags.copy(),
            "available_backends": self.get_available_backends(),
            "validation_issues": self.validate_configuration(),
        }

    def import_configuration(self, config: Dict[str, Any]) -> None:
        """Import configuration from serialized format.

        Args:
            config: Configuration dictionary
        """
        if "flags" in config:
            for flag, value in config["flags"].items():
                if isinstance(value, bool):
                    self._flags[flag] = value
                else:
                    logger.warning(f"Invalid flag value for {flag}: {value}")

        logger.info("Imported feature flag configuration")


# Global feature flags instance
_global_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    """Get the global feature flags instance.

    Returns:
        Global FeatureFlags instance
    """
    global _global_flags
    if _global_flags is None:
        _global_flags = FeatureFlags()
    return _global_flags


def is_feature_enabled(flag: str) -> bool:
    """Check if a feature flag is enabled (convenience function).

    Args:
        flag: Feature flag name

    Returns:
        True if feature is enabled, False otherwise
    """
    return get_feature_flags().is_enabled(flag)


def require_feature(flag: str) -> None:
    """Require a feature to be enabled (convenience function).

    Args:
        flag: Feature flag name

    Raises:
        RuntimeError: If feature is not enabled
    """
    if not is_feature_enabled(flag):
        raise RuntimeError(f"Required feature '{flag}' is not enabled")


def is_backend_available(backend: str) -> bool:
    """Check if a backend is available (convenience function).

    Args:
        backend: Backend name

    Returns:
        True if backend is available, False otherwise
    """
    return is_feature_enabled(f"{backend}_backend")


def get_available_backends() -> List[str]:
    """Get list of available backends (convenience function).

    Returns:
        List of available backend names
    """
    return get_feature_flags().get_available_backends()
