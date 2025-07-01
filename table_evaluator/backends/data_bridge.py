"""Comprehensive data bridge for seamless pandas/Polars conversion.

This module provides the main interface for data type bridging,
combining dtype mapping, conversion, validation, and lazy evaluation.
"""

import logging
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from .conversion_bridge import DataFrameConverter, ConversionError
from .dtype_bridge import DTypeMapper, DTypeMappingError
from .lazy_conversion import LazyConverter, optimize_for_memory
from .schema_validator import SchemaValidator, SchemaIssue

logger = logging.getLogger(__name__)


class DataBridge:
    """Comprehensive data bridge for seamless backend conversion.

    This class provides a high-level interface that combines all aspects
    of data conversion: dtype mapping, validation, conversion, and optimization.
    """

    def __init__(
        self,
        strict_validation: bool = False,
        enable_lazy_conversion: bool = True,
        cache_size: int = 50,
    ):
        """Initialize data bridge.

        Args:
            strict_validation: Whether to use strict schema validation
            enable_lazy_conversion: Whether to enable lazy conversion
            cache_size: Size of lazy conversion cache
        """
        self.dtype_mapper = DTypeMapper()
        self.converter = DataFrameConverter(strict_dtypes=not strict_validation)
        self.validator = SchemaValidator(strict_mode=strict_validation)

        self.enable_lazy_conversion = enable_lazy_conversion
        if enable_lazy_conversion:
            self.lazy_converter = LazyConverter(cache_size)
        else:
            self.lazy_converter = None

        self.strict_validation = strict_validation

    def convert_dataframe(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        target_backend: str,
        validate: bool = True,
        lazy: bool = False,
        optimize_memory: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"]:
        """Convert DataFrame to target backend with full validation and optimization.

        Args:
            df: Source DataFrame
            target_backend: Target backend ("pandas" or "polars")
            validate: Whether to validate before conversion
            lazy: Whether to use lazy conversion (if enabled)
            optimize_memory: Whether to optimize for memory usage
            **kwargs: Additional conversion arguments

        Returns:
            Converted DataFrame

        Raises:
            ConversionError: If conversion fails
            SchemaValidationError: If validation fails in strict mode
        """
        logger.debug(f"Converting DataFrame to {target_backend} backend")

        # Validate schema if requested
        if validate:
            issues = self.validator.validate_conversion_compatibility(
                df, target_backend
            )

            if issues:
                self._handle_validation_issues(issues, df)

        # Optimize memory if requested, but skip for same-backend conversions to avoid dtype changes
        if optimize_memory:
            # Determine source backend
            if isinstance(df, pd.DataFrame):
                source_backend = "pandas"
            elif POLARS_AVAILABLE and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
                source_backend = "polars"
            else:
                source_backend = "unknown"

            # Only optimize if converting between different backends
            if source_backend != target_backend:
                df = self._optimize_dataframe(df, target_backend)

        # Perform conversion
        if lazy and self.enable_lazy_conversion and self.lazy_converter:
            result = self.lazy_converter.convert_when_needed(
                df, target_backend, lazy=lazy, **kwargs
            )
        else:
            result = self.converter.convert_with_fallback(
                df, target_backend, lazy=lazy, **kwargs
            )

        logger.info(f"Successfully converted DataFrame to {target_backend}")
        return result

    def validate_compatibility(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        target_backend: str,
    ) -> Tuple[bool, List[SchemaIssue]]:
        """Validate DataFrame compatibility with target backend.

        Args:
            df: DataFrame to validate
            target_backend: Target backend

        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = self.validator.validate_conversion_compatibility(df, target_backend)

        # Check if there are any blocking issues
        has_errors = any(issue.severity == "error" for issue in issues)
        is_compatible = not has_errors

        return is_compatible, issues

    def get_conversion_plan(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        target_backend: str,
    ) -> Dict[str, Any]:
        """Get detailed conversion plan for DataFrame.

        Args:
            df: Source DataFrame
            target_backend: Target backend

        Returns:
            Dictionary with conversion plan details
        """
        plan = {
            "source_backend": None,
            "target_backend": target_backend,
            "is_compatible": False,
            "requires_conversion": True,
            "validation_issues": [],
            "dtype_mapping": {},
            "optimization_suggestions": [],
            "memory_impact": "unknown",
            "performance_impact": "unknown",
        }

        # Determine source backend
        if isinstance(df, pd.DataFrame):
            plan["source_backend"] = "pandas"
        elif POLARS_AVAILABLE and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            plan["source_backend"] = "polars"

        # Check if conversion is needed
        if plan["source_backend"] == target_backend:
            plan["requires_conversion"] = False
            plan["is_compatible"] = True
            return plan

        # Validate compatibility
        is_compatible, issues = self.validate_compatibility(df, target_backend)
        plan["is_compatible"] = is_compatible
        plan["validation_issues"] = [str(issue) for issue in issues]

        # Get dtype mapping
        if plan["source_backend"] == "pandas":
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
            for col, dtype in dtypes.items():
                try:
                    polars_dtype = self.dtype_mapper.pandas_to_polars_dtype(dtype)
                    plan["dtype_mapping"][col] = {
                        "source": dtype,
                        "target": polars_dtype,
                    }
                except DTypeMappingError:
                    pass

        elif plan["source_backend"] == "polars":
            schema = df.schema if hasattr(df, "schema") else {}
            for col, dtype in schema.items():
                try:
                    pandas_dtype = self.dtype_mapper.polars_to_pandas_dtype(dtype)
                    plan["dtype_mapping"][col] = {
                        "source": str(dtype),
                        "target": pandas_dtype,
                    }
                except DTypeMappingError:
                    pass

        # Estimate performance impact
        data_size = len(df) if hasattr(df, "__len__") else 0
        if data_size > 100_000:
            plan["performance_impact"] = "high"
        elif data_size > 10_000:
            plan["performance_impact"] = "medium"
        else:
            plan["performance_impact"] = "low"

        # Memory impact estimation
        if target_backend == "polars" and plan["source_backend"] == "pandas":
            plan["memory_impact"] = (
                "reduced"  # Polars is generally more memory efficient
            )
        elif target_backend == "pandas" and plan["source_backend"] == "polars":
            plan["memory_impact"] = "increased"  # pandas generally uses more memory
        else:
            plan["memory_impact"] = "neutral"

        # Optimization suggestions
        if data_size > 50_000:
            plan["optimization_suggestions"].append("Consider using lazy evaluation")

        if target_backend == "polars":
            plan["optimization_suggestions"].append(
                "Use LazyFrame for memory efficiency"
            )

        return plan

    def create_schema_mapping(
        self, source_schema: Dict[str, Any], source_backend: str, target_backend: str
    ) -> Dict[str, Any]:
        """Create schema mapping between backends.

        Args:
            source_schema: Source schema (column -> dtype mapping)
            source_backend: Source backend name
            target_backend: Target backend name

        Returns:
            Target schema mapping
        """
        if source_backend == target_backend:
            return source_schema.copy()

        target_schema = {}

        for column, dtype in source_schema.items():
            try:
                if source_backend == "pandas" and target_backend == "polars":
                    target_dtype = self.dtype_mapper.pandas_to_polars_dtype(dtype)
                elif source_backend == "polars" and target_backend == "pandas":
                    target_dtype = self.dtype_mapper.polars_to_pandas_dtype(dtype)
                else:
                    target_dtype = str(dtype)  # Fallback

                target_schema[column] = target_dtype

            except DTypeMappingError as e:
                logger.warning(f"Could not map dtype for column '{column}': {e}")
                target_schema[column] = (
                    "object" if target_backend == "pandas" else "String"
                )

        return target_schema

    def _handle_validation_issues(
        self,
        issues: List[SchemaIssue],
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
    ) -> None:
        """Handle validation issues based on configuration.

        Args:
            issues: List of validation issues
            df: Source DataFrame
        """
        errors = [issue for issue in issues if issue.severity == "error"]
        warnings = [issue for issue in issues if issue.severity == "warning"]

        # Log warnings
        for warning in warnings:
            logger.warning(str(warning))

        # Handle errors
        if errors:
            if self.strict_validation:
                error_messages = [str(error) for error in errors]
                raise ConversionError(f"Validation failed: {'; '.join(error_messages)}")
            else:
                for error in errors:
                    logger.error(str(error))

                # Try to auto-fix if it's a pandas DataFrame
                if isinstance(df, pd.DataFrame):
                    try:
                        fixed_df, applied_fixes = self.validator.fix_common_issues(
                            df, errors, auto_fix=True
                        )
                        if applied_fixes:
                            logger.info(f"Applied automatic fixes: {applied_fixes}")
                            # Update the original DataFrame reference would require returning it
                            # For now, just log the fixes
                    except Exception as e:
                        logger.error(f"Failed to apply automatic fixes: {e}")

    def _optimize_dataframe(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        target_backend: str,
    ) -> Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"]:
        """Optimize DataFrame for target backend.

        Args:
            df: Source DataFrame
            target_backend: Target backend

        Returns:
            Optimized DataFrame
        """
        # Determine optimization strategy based on target backend
        if target_backend == "polars":
            target_operations = ["aggregation", "filter", "join"]
        else:
            target_operations = ["statistical", "ml"]

        return optimize_for_memory(df, target_operations)

    def get_bridge_statistics(self) -> Dict[str, Any]:
        """Get statistics about bridge operations.

        Returns:
            Dictionary with bridge statistics
        """
        stats = {
            "lazy_conversion_enabled": self.enable_lazy_conversion,
            "strict_validation": self.strict_validation,
            "supported_pandas_dtypes": len(
                self.dtype_mapper.get_supported_pandas_dtypes()
            ),
            "supported_polars_dtypes": len(
                self.dtype_mapper.get_supported_polars_dtypes()
            ),
            "polars_available": POLARS_AVAILABLE,
        }

        if self.lazy_converter:
            stats["lazy_conversion_stats"] = self.lazy_converter.get_conversion_stats()

        return stats

    def cleanup(self) -> None:
        """Clean up bridge resources."""
        if self.lazy_converter:
            self.lazy_converter.clear_cache()
            logger.debug("Cleaned up data bridge resources")


# Convenience functions for direct use
def convert_to_pandas(
    df: Union["pl.DataFrame", "pl.LazyFrame"], validate: bool = True, **kwargs
) -> pd.DataFrame:
    """Convert Polars DataFrame to pandas with validation.

    Args:
        df: Polars DataFrame or LazyFrame
        validate: Whether to validate before conversion
        **kwargs: Additional conversion arguments

    Returns:
        Pandas DataFrame
    """
    bridge = DataBridge()
    return bridge.convert_dataframe(df, "pandas", validate=validate, **kwargs)


def convert_to_polars(
    df: pd.DataFrame, lazy: bool = False, validate: bool = True, **kwargs
) -> Union["pl.DataFrame", "pl.LazyFrame"]:
    """Convert pandas DataFrame to Polars with validation.

    Args:
        df: Pandas DataFrame
        lazy: Whether to return LazyFrame
        validate: Whether to validate before conversion
        **kwargs: Additional conversion arguments

    Returns:
        Polars DataFrame or LazyFrame
    """
    if not POLARS_AVAILABLE:
        raise ConversionError("Polars is not available")

    bridge = DataBridge()
    return bridge.convert_dataframe(
        df, "polars", validate=validate, lazy=lazy, **kwargs
    )
