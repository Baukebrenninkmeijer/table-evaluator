"""Schema validator for ensuring data consistency across backend conversions."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from .dtype_bridge import DTypeMapper

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Exception raised when schema validation fails."""

    pass


class SchemaIssue:
    """Represents a schema validation issue."""

    def __init__(
        self,
        issue_type: str,
        column: str,
        description: str,
        severity: str = "warning",
        suggestion: Optional[str] = None,
    ):
        """Initialize schema issue.

        Args:
            issue_type: Type of issue ("dtype", "missing", "extra", "value", etc.)
            column: Column name where issue occurs
            description: Description of the issue
            severity: Severity level ("error", "warning", "info")
            suggestion: Optional suggestion for fixing the issue
        """
        self.issue_type = issue_type
        self.column = column
        self.description = description
        self.severity = severity
        self.suggestion = suggestion

    def __str__(self) -> str:
        """String representation of the issue."""
        base = f"[{self.severity.upper()}] {self.issue_type} in '{self.column}': {self.description}"
        if self.suggestion:
            base += f" | Suggestion: {self.suggestion}"
        return base

    def __repr__(self) -> str:
        """Detailed representation of the issue."""
        return (
            f"SchemaIssue(type='{self.issue_type}', column='{self.column}', "
            f"severity='{self.severity}', description='{self.description}')"
        )


class SchemaValidator:
    """Validates and ensures data consistency across backend conversions.

    This class provides comprehensive schema validation to detect potential
    issues during backend conversions and suggests fixes for common problems.
    """

    def __init__(self, strict_mode: bool = False):
        """Initialize schema validator.

        Args:
            strict_mode: Whether to be strict about schema validation
        """
        self.strict_mode = strict_mode
        self.dtype_mapper = DTypeMapper()

        # Thresholds for validation
        self.max_categorical_unique_ratio = 0.5
        self.min_numerical_unique_count = 2
        self.max_string_length_variance = 1000

    def validate_conversion_compatibility(
        self,
        source_df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        target_backend: str,
    ) -> List[SchemaIssue]:
        """Validate that DataFrame can be safely converted to target backend.

        Args:
            source_df: Source DataFrame to validate
            target_backend: Target backend ("pandas" or "polars")

        Returns:
            List of schema issues found
        """
        issues = []

        # Determine source backend
        if isinstance(source_df, pd.DataFrame):
            source_backend = "pandas"
        elif POLARS_AVAILABLE and isinstance(source_df, (pl.DataFrame, pl.LazyFrame)):
            source_backend = "polars"
        else:
            issues.append(
                SchemaIssue(
                    "unsupported",
                    "DataFrame",
                    f"Unsupported DataFrame type: {type(source_df)}",
                    "error",
                )
            )
            return issues

        if source_backend == target_backend:
            return issues  # No conversion needed

        # Validate dtype compatibility
        issues.extend(
            self._validate_dtype_compatibility(
                source_df, source_backend, target_backend
            )
        )

        # Validate data integrity
        issues.extend(self._validate_data_integrity(source_df, source_backend))

        # Validate column names
        issues.extend(self._validate_column_names(source_df, source_backend))

        # Validate for specific backend limitations
        issues.extend(
            self._validate_backend_limitations(
                source_df, source_backend, target_backend
            )
        )

        return issues

    def _validate_dtype_compatibility(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        source_backend: str,
        target_backend: str,
    ) -> List[SchemaIssue]:
        """Validate dtype compatibility between backends."""
        issues = []

        if source_backend == "pandas":
            dtypes = {col: dtype for col, dtype in df.dtypes.items()}
            unsupported = self.dtype_mapper.validate_pandas_dtypes(dtypes)

            for column in unsupported:
                dtype_str = str(dtypes[column])
                suggestions = self.dtype_mapper.get_conversion_suggestions([dtype_str])

                issues.append(
                    SchemaIssue(
                        "dtype",
                        column,
                        f"Pandas dtype '{dtype_str}' cannot be converted to Polars",
                        "error" if self.strict_mode else "warning",
                        suggestions.get(dtype_str),
                    )
                )

        elif source_backend == "polars":
            if isinstance(df, pl.LazyFrame):
                schema = df.schema
            else:
                schema = df.schema

            dtypes = {col: dtype for col, dtype in schema.items()}
            unsupported = self.dtype_mapper.validate_polars_dtypes(dtypes)

            for column in unsupported:
                dtype_str = str(dtypes[column])
                suggestions = self.dtype_mapper.get_conversion_suggestions([dtype_str])

                issues.append(
                    SchemaIssue(
                        "dtype",
                        column,
                        f"Polars dtype '{dtype_str}' cannot be converted to pandas",
                        "error" if self.strict_mode else "warning",
                        suggestions.get(dtype_str),
                    )
                )

        return issues

    def _validate_data_integrity(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        source_backend: str,
    ) -> List[SchemaIssue]:
        """Validate data integrity issues that could affect conversion."""
        issues = []

        if source_backend == "pandas":
            # Check for mixed types in object columns
            for column in df.select_dtypes(include=["object"]).columns:
                sample_types = set()
                for value in df[column].dropna().head(1000):
                    sample_types.add(type(value).__name__)

                if len(sample_types) > 1:
                    issues.append(
                        SchemaIssue(
                            "mixed_types",
                            column,
                            f"Column contains mixed types: {sample_types}",
                            "warning",
                            "Consider converting to string or separate into multiple columns",
                        )
                    )

            # Check for extreme string lengths
            for column in df.select_dtypes(include=["object", "string"]).columns:
                try:
                    str_lengths = df[column].astype(str).str.len()
                    if (
                        str_lengths.max() - str_lengths.min()
                        > self.max_string_length_variance
                    ):
                        issues.append(
                            SchemaIssue(
                                "string_variance",
                                column,
                                f"Large variance in string lengths (min: {str_lengths.min()}, max: {str_lengths.max()})",
                                "info",
                                "Consider normalizing string lengths or using categorical type",
                            )
                        )
                except Exception:  # nosec B110
                    pass  # Skip if error in string length calculation

        elif source_backend == "polars":
            # Check for schema consistency (LazyFrame vs DataFrame)
            if isinstance(df, pl.LazyFrame):
                try:
                    # Try to collect a small sample to validate
                    sample = df.limit(10).collect()
                    if sample.is_empty():
                        issues.append(
                            SchemaIssue(
                                "empty_data",
                                "DataFrame",
                                "LazyFrame appears to be empty",
                                "warning",
                            )
                        )
                except Exception as e:
                    issues.append(
                        SchemaIssue(
                            "lazy_execution",
                            "DataFrame",
                            f"Error executing LazyFrame: {str(e)}",
                            "error",
                        )
                    )

        return issues

    def _validate_column_names(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        source_backend: str,
    ) -> List[SchemaIssue]:
        """Validate column names for potential issues."""
        issues = []

        if source_backend == "pandas":
            columns = df.columns.tolist()
        else:
            columns = df.columns

        # Check for duplicate column names
        seen = set()
        duplicates = set()
        for col in columns:
            if col in seen:
                duplicates.add(col)
            seen.add(col)

        for col in duplicates:
            issues.append(
                SchemaIssue(
                    "duplicate_column",
                    col,
                    "Duplicate column name found",
                    "error",
                    "Rename duplicate columns before conversion",
                )
            )

        # Check for problematic column names
        for col in columns:
            col_str = str(col)

            # Empty or whitespace-only names
            if not col_str or col_str.isspace():
                issues.append(
                    SchemaIssue(
                        "empty_column_name",
                        col_str,
                        "Column name is empty or whitespace-only",
                        "error",
                        "Provide meaningful column names",
                    )
                )

            # Names with special characters
            if any(char in col_str for char in ["\n", "\r", "\t"]):
                issues.append(
                    SchemaIssue(
                        "special_chars",
                        col_str,
                        "Column name contains special characters (newlines, tabs)",
                        "warning",
                        "Remove or replace special characters",
                    )
                )

            # Very long column names
            if len(col_str) > 255:
                issues.append(
                    SchemaIssue(
                        "long_column_name",
                        col_str,
                        f"Column name is very long ({len(col_str)} characters)",
                        "warning",
                        "Consider shortening column name",
                    )
                )

        return issues

    def _validate_backend_limitations(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        source_backend: str,
        target_backend: str,
    ) -> List[SchemaIssue]:
        """Validate for specific backend limitations."""
        issues = []

        if target_backend == "polars" and source_backend == "pandas":
            # Polars limitations

            # Check for MultiIndex
            if (
                hasattr(df, "index")
                and hasattr(df.index, "nlevels")
                and df.index.nlevels > 1
            ):
                issues.append(
                    SchemaIssue(
                        "multiindex",
                        "index",
                        "Polars does not support MultiIndex",
                        "error",
                        "Reset index or flatten MultiIndex before conversion",
                    )
                )

            # Check for complex dtypes
            for column in df.columns:
                try:
                    # Handle case where duplicate columns exist
                    col_data = df[column]
                    if hasattr(col_data, "dtype"):
                        dtype = col_data.dtype
                    else:
                        # Skip if this is a DataFrame (duplicate columns)
                        continue

                    if dtype.name.startswith("complex"):
                        issues.append(
                            SchemaIssue(
                                "complex_dtype",
                                column,
                                "Polars does not support complex number dtypes",
                                "error",
                                "Convert to separate real/imaginary columns or string representation",
                            )
                        )
                except (AttributeError, KeyError):
                    # Skip columns that can't be accessed properly (e.g., duplicates)
                    continue

        elif target_backend == "pandas" and source_backend == "polars":
            # Pandas limitations (fewer than Polars)

            # Check for very large integers that might overflow
            if source_backend == "polars":
                schema = df.schema if hasattr(df, "schema") else {}
                for column, dtype in schema.items():
                    if str(dtype) == "UInt64":
                        issues.append(
                            SchemaIssue(
                                "uint64_overflow",
                                column,
                                "UInt64 values might overflow in pandas (max int64)",
                                "warning",
                                "Check if values exceed pandas int64 range",
                            )
                        )

        return issues

    def validate_schema_consistency(
        self,
        df1: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        df2: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        check_order: bool = True,
    ) -> List[SchemaIssue]:
        """Validate schema consistency between two DataFrames.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            check_order: Whether to check column order

        Returns:
            List of schema consistency issues
        """
        issues = []

        # Get column information
        if isinstance(df1, pd.DataFrame):
            cols1 = df1.columns.tolist()
            dtypes1 = {col: str(dtype) for col, dtype in df1.dtypes.items()}
        else:
            cols1 = df1.columns
            dtypes1 = {col: str(dtype) for col, dtype in df1.schema.items()}

        if isinstance(df2, pd.DataFrame):
            cols2 = df2.columns.tolist()
            dtypes2 = {col: str(dtype) for col, dtype in df2.dtypes.items()}
        else:
            cols2 = df2.columns
            dtypes2 = {col: str(dtype) for col, dtype in df2.schema.items()}

        # Check column presence
        missing_in_df2 = set(cols1) - set(cols2)
        extra_in_df2 = set(cols2) - set(cols1)

        for col in missing_in_df2:
            issues.append(
                SchemaIssue(
                    "missing_column", col, "Column missing in second DataFrame", "error"
                )
            )

        for col in extra_in_df2:
            issues.append(
                SchemaIssue(
                    "extra_column", col, "Extra column in second DataFrame", "warning"
                )
            )

        # Check column order
        if check_order and cols1 != cols2:
            common_cols = [col for col in cols1 if col in cols2]
            df2_common_order = [col for col in cols2 if col in common_cols]

            if common_cols != df2_common_order:
                issues.append(
                    SchemaIssue(
                        "column_order",
                        "DataFrame",
                        "Column order differs between DataFrames",
                        "warning",
                        "Reorder columns to match",
                    )
                )

        # Check dtype consistency for common columns
        common_columns = set(cols1) & set(cols2)
        for col in common_columns:
            if dtypes1[col] != dtypes2[col]:
                issues.append(
                    SchemaIssue(
                        "dtype_mismatch",
                        col,
                        f"Dtype mismatch: {dtypes1[col]} vs {dtypes2[col]}",
                        "warning",
                        "Cast to common dtype before operations",
                    )
                )

        return issues

    def get_validation_summary(self, issues: List[SchemaIssue]) -> Dict[str, Any]:
        """Get summary of validation issues.

        Args:
            issues: List of schema issues

        Returns:
            Dictionary with validation summary
        """
        summary = {
            "total_issues": len(issues),
            "by_severity": {"error": 0, "warning": 0, "info": 0},
            "by_type": {},
            "has_blocking_issues": False,
            "recommendations": [],
        }

        for issue in issues:
            # Count by severity
            summary["by_severity"][issue.severity] += 1

            # Count by type
            if issue.issue_type not in summary["by_type"]:
                summary["by_type"][issue.issue_type] = 0
            summary["by_type"][issue.issue_type] += 1

            # Check for blocking issues
            if issue.severity == "error":
                summary["has_blocking_issues"] = True

            # Collect recommendations
            if issue.suggestion and issue.suggestion not in summary["recommendations"]:
                summary["recommendations"].append(issue.suggestion)

        return summary

    def fix_common_issues(
        self, df: pd.DataFrame, issues: List[SchemaIssue], auto_fix: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Attempt to fix common schema issues.

        Args:
            df: DataFrame to fix
            issues: List of issues to address
            auto_fix: Whether to automatically apply fixes

        Returns:
            Tuple of (fixed_dataframe, list_of_applied_fixes)
        """
        fixed_df = df.copy()
        applied_fixes = []

        for issue in issues:
            if not auto_fix and issue.severity == "error":
                continue  # Skip errors unless auto_fix is enabled

            try:
                if (
                    issue.issue_type == "mixed_types"
                    and issue.column in fixed_df.columns
                ):
                    # Convert mixed types to string
                    fixed_df[issue.column] = fixed_df[issue.column].astype(str)
                    applied_fixes.append(f"Converted '{issue.column}' to string type")

                elif issue.issue_type == "empty_column_name":
                    # Generate a name for empty columns
                    new_name = f"column_{len(applied_fixes)}"
                    fixed_df = fixed_df.rename(columns={issue.column: new_name})
                    applied_fixes.append(f"Renamed empty column to '{new_name}'")

                elif issue.issue_type == "duplicate_column":
                    # Rename duplicate columns
                    columns = fixed_df.columns.tolist()
                    duplicates = [
                        i for i, col in enumerate(columns) if col == issue.column
                    ]

                    for i, dup_idx in enumerate(
                        duplicates[1:], 1
                    ):  # Skip first occurrence
                        new_name = f"{issue.column}_{i}"
                        columns[dup_idx] = new_name
                        applied_fixes.append(
                            f"Renamed duplicate column to '{new_name}'"
                        )

                    fixed_df.columns = columns

            except Exception as e:
                logger.warning(
                    f"Failed to fix issue {issue.issue_type} for column {issue.column}: {e}"
                )

        return fixed_df, applied_fixes
