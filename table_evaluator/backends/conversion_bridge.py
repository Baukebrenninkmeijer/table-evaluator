"""Bidirectional conversion functions for pandas/Polars with dtype preservation."""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from .dtype_bridge import DTypeMapper, DTypeMappingError

logger = logging.getLogger(__name__)


class ConversionError(Exception):
    """Exception raised when DataFrame conversion fails."""

    pass


class DataFrameConverter:
    """Bidirectional converter for pandas and Polars DataFrames with dtype preservation.

    This class handles the complex task of converting DataFrames between backends
    while preserving data types, handling missing values, and maintaining data integrity.
    """

    def __init__(self, strict_dtypes: bool = True):
        """Initialize DataFrame converter.

        Args:
            strict_dtypes: Whether to enforce strict dtype compatibility
        """
        self.dtype_mapper = DTypeMapper()
        self.strict_dtypes = strict_dtypes

    def pandas_to_polars(
        self,
        df: pd.DataFrame,
        lazy: bool = False,
        preserve_dtypes: bool = True,
        handle_unsupported: str = "warn",
    ) -> Union["pl.DataFrame", "pl.LazyFrame"]:
        """Convert pandas DataFrame to Polars DataFrame/LazyFrame.

        Args:
            df: Pandas DataFrame to convert
            lazy: Whether to return LazyFrame
            preserve_dtypes: Whether to preserve original dtypes
            handle_unsupported: How to handle unsupported dtypes ("warn", "error", "coerce")

        Returns:
            Polars DataFrame or LazyFrame

        Raises:
            ConversionError: If conversion fails
        """
        if not POLARS_AVAILABLE:
            raise ConversionError("Polars is not available")

        if df.empty:
            # Handle empty DataFrame
            if lazy:
                return pl.LazyFrame()
            else:
                return pl.DataFrame()

        try:
            # Handle mixed types before conversion
            df_to_convert = df.copy()

            # Check for mixed types in object columns and convert them to string
            for col in df_to_convert.select_dtypes(include=["object"]).columns:
                # Check if the column has mixed types by attempting type inference
                sample_values = df_to_convert[col].dropna()
                if len(sample_values) > 0:
                    type_set = set(
                        type(val).__name__
                        for val in sample_values.iloc[: min(100, len(sample_values))]
                    )
                    if len(type_set) > 1:
                        logger.debug(
                            f"Converting mixed-type column '{col}' to string for Polars compatibility"
                        )
                        df_to_convert[col] = df_to_convert[col].astype(str)

            # Validate dtypes if preserve_dtypes is True
            if preserve_dtypes:
                unsupported = self.dtype_mapper.validate_pandas_dtypes(
                    {col: dtype for col, dtype in df_to_convert.dtypes.items()}
                )

                if unsupported:
                    self._handle_unsupported_dtypes(
                        unsupported, handle_unsupported, "pandas to Polars"
                    )

            # Create Polars DataFrame
            polars_df = pl.from_pandas(df_to_convert)

            # Apply dtype conversions if needed (only if they seem necessary)
            if preserve_dtypes:
                try:
                    polars_df = self._apply_polars_dtypes(df, polars_df)
                except Exception as e:
                    # If dtype conversion fails, log and continue without conversion
                    # Polars often does a good job detecting types automatically
                    logger.warning(
                        f"Dtype conversion failed, using polars auto-detected types: {e}"
                    )
                    # Continue with the polars-detected types

            # Return lazy or eager
            if lazy:
                return polars_df.lazy()
            else:
                return polars_df

        except Exception as e:
            raise ConversionError(
                f"Failed to convert pandas to Polars: {str(e)}"
            ) from e

    def polars_to_pandas(
        self,
        df: Union["pl.DataFrame", "pl.LazyFrame"],
        preserve_dtypes: bool = True,
        handle_unsupported: str = "warn",
    ) -> pd.DataFrame:
        """Convert Polars DataFrame/LazyFrame to pandas DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame to convert
            preserve_dtypes: Whether to preserve original dtypes
            handle_unsupported: How to handle unsupported dtypes ("warn", "error", "coerce")

        Returns:
            Pandas DataFrame

        Raises:
            ConversionError: If conversion fails
        """
        if not POLARS_AVAILABLE:
            raise ConversionError("Polars is not available")

        try:
            # Collect if LazyFrame
            if isinstance(df, pl.LazyFrame):
                df = df.collect()

            if df.is_empty():
                # Handle empty DataFrame
                return pd.DataFrame()

            # Validate dtypes if preserve_dtypes is True
            if preserve_dtypes:
                polars_dtypes = {col: str(dtype) for col, dtype in df.schema.items()}
                unsupported = self.dtype_mapper.validate_polars_dtypes(polars_dtypes)

                if unsupported:
                    self._handle_unsupported_dtypes(
                        unsupported, handle_unsupported, "Polars to pandas"
                    )

            # Convert to pandas
            pandas_df = df.to_pandas()

            # Apply dtype conversions if needed
            if preserve_dtypes:
                pandas_df = self._apply_pandas_dtypes(df, pandas_df)

            return pandas_df

        except Exception as e:
            raise ConversionError(
                f"Failed to convert Polars to pandas: {str(e)}"
            ) from e

    def _apply_polars_dtypes(
        self, pandas_df: pd.DataFrame, polars_df: "pl.DataFrame"
    ) -> "pl.DataFrame":
        """Apply proper Polars dtypes based on pandas DataFrame.

        Args:
            pandas_df: Original pandas DataFrame
            polars_df: Converted Polars DataFrame

        Returns:
            Polars DataFrame with corrected dtypes
        """
        conversions = []

        for column in pandas_df.columns:
            try:
                pandas_dtype = pandas_df[column].dtype
                current_polars_dtype = polars_df[column].dtype

                # Smart dtype detection for object columns
                if pandas_dtype == "object":
                    # Check if polars already detected a better type
                    if current_polars_dtype in [pl.Boolean, pl.Int64, pl.Float64]:
                        # Trust polars' detection and skip conversion
                        logger.debug(
                            f"Column '{column}': Skipping conversion, polars detected {current_polars_dtype} for object column"
                        )
                        continue

                target_polars_dtype = self.dtype_mapper.pandas_to_polars_dtype(
                    pandas_dtype
                )

                # Get the appropriate Polars dtype class
                polars_dtype_class = self._get_polars_dtype_class(target_polars_dtype)

                if polars_dtype_class is not None:
                    # Only convert if the types are actually different
                    if str(current_polars_dtype) != target_polars_dtype:
                        logger.debug(
                            f"Column '{column}': pandas {pandas_dtype} -> polars {current_polars_dtype} -> target {target_polars_dtype}"
                        )
                        conversions.append(pl.col(column).cast(polars_dtype_class))

            except (DTypeMappingError, AttributeError) as e:
                logger.debug(f"Could not convert dtype for column '{column}': {e}")
                continue

        if conversions:
            polars_df = polars_df.with_columns(conversions)

        return polars_df

    def _apply_pandas_dtypes(
        self, polars_df: "pl.DataFrame", pandas_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply proper pandas dtypes based on Polars DataFrame.

        Args:
            polars_df: Original Polars DataFrame
            pandas_df: Converted pandas DataFrame

        Returns:
            Pandas DataFrame with corrected dtypes
        """
        dtype_dict = {}

        for column in polars_df.columns:
            try:
                polars_dtype = polars_df.schema[column]
                target_pandas_dtype = self.dtype_mapper.polars_to_pandas_dtype(
                    polars_dtype
                )
                dtype_dict[column] = target_pandas_dtype

            except DTypeMappingError:
                logger.debug(f"Could not convert dtype for column '{column}'")
                continue

        if dtype_dict:
            for column, dtype in dtype_dict.items():
                try:
                    pandas_df[column] = pandas_df[column].astype(dtype)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not cast column '{column}' to {dtype}: {e}")

        return pandas_df

    def _get_polars_dtype_class(self, dtype_str: str) -> Optional[Any]:
        """Get Polars dtype class from string representation.

        Args:
            dtype_str: String representation of Polars dtype

        Returns:
            Polars dtype class or None if not found
        """
        if not POLARS_AVAILABLE:
            return None

        dtype_mapping = {
            "Int8": pl.Int8,
            "Int16": pl.Int16,
            "Int32": pl.Int32,
            "Int64": pl.Int64,
            "UInt8": pl.UInt8,
            "UInt16": pl.UInt16,
            "UInt32": pl.UInt32,
            "UInt64": pl.UInt64,
            "Float32": pl.Float32,
            "Float64": pl.Float64,
            "Boolean": pl.Boolean,
            "String": pl.String,
            "Utf8": pl.Utf8,
            "Datetime": pl.Datetime,
            "Date": pl.Date,
            "Duration": pl.Duration,
            "Time": pl.Time,
            "Categorical": pl.Categorical,
        }

        return dtype_mapping.get(dtype_str)

    def _handle_unsupported_dtypes(
        self, unsupported: List[str], handle_mode: str, conversion_direction: str
    ) -> None:
        """Handle unsupported dtypes during conversion.

        Args:
            unsupported: List of unsupported dtype strings
            handle_mode: How to handle ("warn", "error", "coerce")
            conversion_direction: Direction of conversion for error messages
        """
        message = f"Unsupported dtypes for {conversion_direction}: {unsupported}"

        if handle_mode == "error":
            raise ConversionError(message)
        elif handle_mode == "warn":
            logger.warning(message)
            suggestions = self.dtype_mapper.get_conversion_suggestions(unsupported)
            for dtype, suggestion in suggestions.items():
                logger.info(f"Suggestion for '{dtype}': {suggestion}")
        elif handle_mode == "coerce":
            logger.info(f"Coercing unsupported dtypes: {unsupported}")
        else:
            raise ValueError(f"Unknown handle_mode: {handle_mode}")

    def convert_with_fallback(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        target_backend: str,
        **kwargs,
    ) -> Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"]:
        """Convert DataFrame with automatic fallback strategies.

        Args:
            df: Source DataFrame
            target_backend: Target backend ("pandas" or "polars")
            **kwargs: Additional conversion arguments

        Returns:
            Converted DataFrame
        """
        if target_backend == "pandas":
            if isinstance(df, pd.DataFrame):
                return df.copy()
            else:
                # Remove 'lazy' parameter as polars_to_pandas doesn't support it
                pandas_kwargs = {k: v for k, v in kwargs.items() if k != "lazy"}
                return self.polars_to_pandas(df, **pandas_kwargs)

        elif target_backend == "polars":
            if POLARS_AVAILABLE and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
                return df.clone()
            else:
                return self.pandas_to_polars(df, **kwargs)

        else:
            raise ValueError(f"Unknown target backend: {target_backend}")

    def get_conversion_info(
        self, df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"]
    ) -> Dict[str, Any]:
        """Get information about potential conversion issues.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with conversion information
        """
        info = {
            "source_backend": None,
            "total_columns": 0,
            "supported_columns": 0,
            "unsupported_columns": [],
            "dtype_mapping": {},
            "conversion_warnings": [],
        }

        if isinstance(df, pd.DataFrame):
            info["source_backend"] = "pandas"
            info["total_columns"] = len(df.columns)

            dtypes = {col: dtype for col, dtype in df.dtypes.items()}
            unsupported = self.dtype_mapper.validate_pandas_dtypes(dtypes)

            info["supported_columns"] = len(df.columns) - len(unsupported)
            info["unsupported_columns"] = unsupported

            # Create dtype mapping for supported columns
            for col, dtype in dtypes.items():
                if col not in unsupported:
                    try:
                        polars_dtype = self.dtype_mapper.pandas_to_polars_dtype(dtype)
                        info["dtype_mapping"][col] = {
                            "pandas": str(dtype),
                            "polars": polars_dtype,
                        }
                    except DTypeMappingError:
                        pass

        elif POLARS_AVAILABLE and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            info["source_backend"] = "polars"
            info["total_columns"] = len(df.columns)

            dtypes = {col: dtype for col, dtype in df.schema.items()}
            unsupported = self.dtype_mapper.validate_polars_dtypes(dtypes)

            info["supported_columns"] = len(df.columns) - len(unsupported)
            info["unsupported_columns"] = unsupported

            # Create dtype mapping for supported columns
            for col, dtype in dtypes.items():
                if col not in unsupported:
                    try:
                        pandas_dtype = self.dtype_mapper.polars_to_pandas_dtype(dtype)
                        info["dtype_mapping"][col] = {
                            "polars": str(dtype),
                            "pandas": pandas_dtype,
                        }
                    except DTypeMappingError:
                        pass

        return info
