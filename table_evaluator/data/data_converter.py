"""Data conversion utilities for table evaluation."""

import logging
from typing import List, Optional, Tuple, Union

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from ..backends import (
    BackendType,
    DataFrameWrapper,
    is_backend_available,
    is_feature_enabled,
    BackendManager,
    convert_to_pandas,
    convert_to_polars,
)

logger = logging.getLogger(__name__)


class DataConverter:
    """Utilities for converting data between different representations with backend support."""

    def __init__(self, backend: Optional[Union[str, BackendType]] = None):
        """Initialize DataConverter with optional backend preference.

        Args:
            backend: Preferred backend for operations ("pandas", "polars", or BackendType)
        """
        self.backend = self._parse_backend(backend)
        self.backend_manager = (
            BackendManager()
            if is_feature_enabled("automatic_backend_detection")
            else None
        )

    def _parse_backend(
        self, backend: Optional[Union[str, BackendType]]
    ) -> Optional[BackendType]:
        """Parse backend parameter to BackendType."""
        if backend is None:
            return None
        elif isinstance(backend, str):
            if backend.lower() == "pandas":
                return BackendType.PANDAS
            elif backend.lower() == "polars":
                return BackendType.POLARS
            elif backend.lower() == "auto":
                return BackendType.AUTO
            else:
                raise ValueError(f"Unknown backend: {backend}")
        else:
            return backend

    def to_numerical(
        self,
        real: Union[pd.DataFrame, DataFrameWrapper],
        fake: Union[pd.DataFrame, DataFrameWrapper],
        categorical_columns: List[str],
        backend: Optional[Union[str, BackendType]] = None,
    ) -> Tuple[
        Union[pd.DataFrame, DataFrameWrapper], Union[pd.DataFrame, DataFrameWrapper]
    ]:
        """
        Convert datasets to numerical representations using factorization.

        Categorical columns are factorized to ensure both datasets have
        identical column representations.

        Args:
            real: Real dataset (pandas DataFrame or DataFrameWrapper)
            fake: Synthetic dataset (pandas DataFrame or DataFrameWrapper)
            categorical_columns: List of categorical column names
            backend: Backend to use for operations (overrides instance backend)

        Returns:
            Tuple of numerically encoded datasets (same type as input)
        """
        # Determine target backend
        target_backend = (
            self._parse_backend(backend) or self.backend or BackendType.PANDAS
        )

        # Convert inputs to consistent format for processing
        real_df, real_is_wrapper = self._extract_dataframe(real)
        fake_df, fake_is_wrapper = self._extract_dataframe(fake)

        # Convert to pandas for factorization (most mature implementation)
        if isinstance(real_df, DataFrameWrapper):
            real_pandas = real_df.to_pandas()
        else:
            real_pandas = (
                real_df
                if isinstance(real_df, pd.DataFrame)
                else convert_to_pandas(real_df)
            )

        if isinstance(fake_df, DataFrameWrapper):
            fake_pandas = fake_df.to_pandas()
        else:
            fake_pandas = (
                fake_df
                if isinstance(fake_df, pd.DataFrame)
                else convert_to_pandas(fake_df)
            )

        # Perform factorization
        real_converted = real_pandas.copy()
        fake_converted = fake_pandas.copy()

        for column in categorical_columns:
            if (
                column in real_converted.columns
                and real_converted[column].dtype == "object"
            ):
                # Use consistent factorization across both datasets
                combined_values = pd.concat(
                    [real_converted[column], fake_converted[column]]
                ).dropna()

                # Handle mixed types safely by converting everything to string if needed
                try:
                    unique_values = sorted(combined_values.unique())
                except TypeError:
                    # If direct sorting fails (mixed types), convert both columns to string first
                    logger.debug(
                        f"Mixed types detected in column '{column}', converting all values to string for consistent encoding"
                    )
                    real_converted[column] = real_converted[column].astype(str)
                    fake_converted[column] = fake_converted[column].astype(str)

                    # Recalculate combined values after string conversion
                    combined_values = pd.concat(
                        [real_converted[column], fake_converted[column]]
                    ).dropna()
                    unique_values = sorted(combined_values.unique())

                # Create mapping for consistent encoding
                value_map = {val: idx for idx, val in enumerate(unique_values)}

                real_converted[column] = (
                    real_converted[column].map(value_map).fillna(-1).astype(int)
                )
                fake_converted[column] = (
                    fake_converted[column].map(value_map).fillna(-1).astype(int)
                )

        # Return in appropriate format
        return self._return_in_original_format(
            real_converted,
            fake_converted,
            real_is_wrapper,
            fake_is_wrapper,
            target_backend,
        )

    def to_one_hot(
        self,
        real: Union[pd.DataFrame, DataFrameWrapper],
        fake: Union[pd.DataFrame, DataFrameWrapper],
        categorical_columns: List[str],
        backend: Optional[Union[str, BackendType]] = None,
    ) -> Tuple[
        Union[pd.DataFrame, DataFrameWrapper], Union[pd.DataFrame, DataFrameWrapper]
    ]:
        """
        Convert datasets to numerical representations using one-hot encoding.

        Categorical and boolean columns are one-hot encoded to ensure both datasets
        have identical column representations.

        Args:
            real: Real dataset (pandas DataFrame or DataFrameWrapper)
            fake: Synthetic dataset (pandas DataFrame or DataFrameWrapper)
            categorical_columns: List of categorical column names
            backend: Backend to use for operations (overrides instance backend)

        Returns:
            Tuple of one-hot encoded datasets (same type as input)
        """
        # Determine target backend
        target_backend = (
            self._parse_backend(backend) or self.backend or BackendType.PANDAS
        )

        # Convert inputs to consistent format for processing
        real_df, real_is_wrapper = self._extract_dataframe(real)
        fake_df, fake_is_wrapper = self._extract_dataframe(fake)

        # Convert to pandas for one-hot encoding
        if isinstance(real_df, DataFrameWrapper):
            real_pandas = real_df.to_pandas()
        else:
            real_pandas = (
                real_df
                if isinstance(real_df, pd.DataFrame)
                else convert_to_pandas(real_df)
            )

        if isinstance(fake_df, DataFrameWrapper):
            fake_pandas = fake_df.to_pandas()
        else:
            fake_pandas = (
                fake_df
                if isinstance(fake_df, pd.DataFrame)
                else convert_to_pandas(fake_df)
            )

        # Include boolean columns in encoding
        extended_categorical_cols = self._extend_categorical_columns(
            categorical_columns, real_pandas, fake_pandas
        )

        # Apply one-hot encoding using native implementation
        real_encoded = self._numerical_encoding(real_pandas, extended_categorical_cols)
        fake_encoded = self._numerical_encoding(fake_pandas, extended_categorical_cols)

        # Ensure both datasets have the same columns
        real_aligned, fake_aligned = self._align_columns(real_encoded, fake_encoded)

        # Return in appropriate format
        return self._return_in_original_format(
            real_aligned, fake_aligned, real_is_wrapper, fake_is_wrapper, target_backend
        )

    def to_numerical_one_hot(
        self,
        real: Union[pd.DataFrame, DataFrameWrapper],
        fake: Union[pd.DataFrame, DataFrameWrapper],
        categorical_columns: List[str],
        backend: Optional[Union[str, BackendType]] = None,
    ) -> Tuple[
        Union[pd.DataFrame, DataFrameWrapper], Union[pd.DataFrame, DataFrameWrapper]
    ]:
        """
        Convert datasets to numerical representations using one-hot encoding.

        This method is identical to to_one_hot and exists for backward compatibility.

        Args:
            real: Real dataset (pandas DataFrame or DataFrameWrapper)
            fake: Synthetic dataset (pandas DataFrame or DataFrameWrapper)
            categorical_columns: List of categorical column names
            backend: Backend to use for operations (overrides instance backend)

        Returns:
            Tuple of one-hot encoded datasets (same type as input)
        """
        return self.to_one_hot(real, fake, categorical_columns, backend)

    def _extract_dataframe(
        self, df: Union[pd.DataFrame, DataFrameWrapper]
    ) -> Tuple[Union[pd.DataFrame, DataFrameWrapper], bool]:
        """Extract the actual DataFrame and determine if it was a wrapper.

        Args:
            df: Input DataFrame or wrapper

        Returns:
            Tuple of (dataframe, is_wrapper)
        """
        if isinstance(df, DataFrameWrapper):
            return df, True
        else:
            return df, False

    def _return_in_original_format(
        self,
        real_df: pd.DataFrame,
        fake_df: pd.DataFrame,
        real_was_wrapper: bool,
        fake_was_wrapper: bool,
        target_backend: BackendType,
    ) -> Tuple[
        Union[pd.DataFrame, DataFrameWrapper], Union[pd.DataFrame, DataFrameWrapper]
    ]:
        """Return DataFrames in the appropriate format based on input types.

        Args:
            real_df: Processed real DataFrame
            fake_df: Processed fake DataFrame
            real_was_wrapper: Whether real input was a wrapper
            fake_was_wrapper: Whether fake input was a wrapper
            target_backend: Target backend type

        Returns:
            Tuple of DataFrames in appropriate format
        """
        # Convert to target backend if needed
        if target_backend == BackendType.POLARS and is_backend_available("polars"):
            real_result = convert_to_polars(real_df, lazy=False)
            fake_result = convert_to_polars(fake_df, lazy=False)

            if real_was_wrapper or fake_was_wrapper:
                return DataFrameWrapper(real_result), DataFrameWrapper(fake_result)
            else:
                return real_result, fake_result
        else:
            # Return pandas format
            if real_was_wrapper or fake_was_wrapper:
                return DataFrameWrapper(real_df), DataFrameWrapper(fake_df)
            else:
                return real_df, fake_df

    @staticmethod
    def _extend_categorical_columns(
        categorical_columns: List[str], real: pd.DataFrame, fake: pd.DataFrame
    ) -> List[str]:
        """
        Extend categorical columns to include boolean columns from both datasets.

        Args:
            categorical_columns: Base categorical column names
            real: Real dataset
            fake: Synthetic dataset

        Returns:
            List[str]: Extended list of categorical columns without duplicates
        """
        extended_cols = (
            categorical_columns
            + real.select_dtypes("bool").columns.tolist()
            + fake.select_dtypes("bool").columns.tolist()
        )
        # Remove duplicates while preserving order
        return list(dict.fromkeys(extended_cols))

    @staticmethod
    def _align_columns(
        real_encoded: pd.DataFrame, fake_encoded: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ensure both datasets have identical columns in the same order.

        Args:
            real_encoded: Real dataset after encoding
            fake_encoded: Fake dataset after encoding

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Aligned datasets with same columns
        """
        # Get all unique columns and sort for consistency
        all_columns = sorted(set(real_encoded.columns) | set(fake_encoded.columns))

        # Add missing columns with zeros
        for col in all_columns:
            if col not in real_encoded.columns:
                logger.warning(
                    f"Adding missing column {col} with all 0s to real dataset"
                )
                real_encoded[col] = 0.0
            if col not in fake_encoded.columns:
                logger.warning(
                    f"Adding missing column {col} with all 0s to fake dataset"
                )
                fake_encoded[col] = 0.0

        # Reorder columns to match
        real_encoded = real_encoded[all_columns]
        fake_encoded = fake_encoded[all_columns]

        return real_encoded, fake_encoded

    def ensure_compatible_columns(
        self,
        real: Union[pd.DataFrame, DataFrameWrapper],
        fake: Union[pd.DataFrame, DataFrameWrapper],
        backend: Optional[Union[str, BackendType]] = None,
    ) -> Tuple[
        Union[pd.DataFrame, DataFrameWrapper], Union[pd.DataFrame, DataFrameWrapper]
    ]:
        """
        Ensure both datasets have exactly the same columns in the same order.

        Args:
            real: Real dataset (pandas DataFrame or DataFrameWrapper)
            fake: Synthetic dataset (pandas DataFrame or DataFrameWrapper)
            backend: Backend to use for operations (overrides instance backend)

        Returns:
            Tuple of datasets with aligned columns (same type as input)
        """
        # Determine target backend
        target_backend = (
            self._parse_backend(backend) or self.backend or BackendType.PANDAS
        )

        # Convert inputs to consistent format for processing
        real_df, real_is_wrapper = self._extract_dataframe(real)
        fake_df, fake_is_wrapper = self._extract_dataframe(fake)

        # Convert to pandas for column operations
        if isinstance(real_df, DataFrameWrapper):
            real_pandas = real_df.to_pandas()
        else:
            real_pandas = (
                real_df
                if isinstance(real_df, pd.DataFrame)
                else convert_to_pandas(real_df)
            )

        if isinstance(fake_df, DataFrameWrapper):
            fake_pandas = fake_df.to_pandas()
        else:
            fake_pandas = (
                fake_df
                if isinstance(fake_df, pd.DataFrame)
                else convert_to_pandas(fake_df)
            )

        # Get intersection of columns
        common_columns = list(set(real_pandas.columns) & set(fake_pandas.columns))

        if len(common_columns) != len(real_pandas.columns) or len(
            common_columns
        ) != len(fake_pandas.columns):
            logger.warning("Datasets have different columns. Using intersection.")

        # Sort columns for consistency
        common_columns.sort()

        real_aligned = real_pandas[common_columns].copy()
        fake_aligned = fake_pandas[common_columns].copy()

        # Return in appropriate format
        return self._return_in_original_format(
            real_aligned, fake_aligned, real_is_wrapper, fake_is_wrapper, target_backend
        )

    @staticmethod
    def _numerical_encoding(
        df: pd.DataFrame, nominal_columns: List[str]
    ) -> pd.DataFrame:
        """
        Convert a DataFrame to numerical encoding using one-hot encoding for categorical columns.

        This is a native implementation to replace dython.nominal.numerical_encoding.

        Args:
            df: DataFrame to encode
            nominal_columns: List of column names to treat as categorical

        Returns:
            pd.DataFrame: Numerically encoded DataFrame
        """
        result_df = df.copy()

        # Apply one-hot encoding to categorical columns
        for col in nominal_columns:
            if col in result_df.columns:
                # Use pandas get_dummies for one-hot encoding
                encoded = pd.get_dummies(result_df[col], prefix=col, dummy_na=False)

                # Drop the original column and add the encoded columns
                result_df = result_df.drop(columns=[col])
                result_df = pd.concat([result_df, encoded], axis=1)

        # Convert all columns to float (original behavior for backward compatibility)
        # But handle problematic columns more gracefully
        for col in result_df.columns:
            try:
                if result_df[col].dtype == "bool":
                    # Convert boolean columns to float (0.0/1.0)
                    result_df[col] = result_df[col].astype(float)
                else:
                    # Try to convert to numeric
                    result_df[col] = pd.to_numeric(
                        result_df[col], errors="coerce"
                    ).astype(float)
            except Exception as e:
                logger.warning(
                    f"Could not convert column '{col}' to float, trying fallback encoding: {e}"
                )
                try:
                    # Fallback: categorical codes
                    result_df[col] = pd.Categorical(result_df[col]).codes.astype(float)
                except Exception:
                    # Last resort: drop the column
                    logger.warning(
                        f"Dropping column '{col}' as it cannot be converted to numeric format"
                    )
                    result_df = result_df.drop(columns=[col])

        return result_df


# Backward compatibility: create static methods that use default instance
_default_converter = DataConverter()


# Static method wrappers for backward compatibility
def to_numerical(
    real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward compatibility wrapper for to_numerical."""
    return _default_converter.to_numerical(real, fake, categorical_columns)


def to_one_hot(
    real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward compatibility wrapper for to_one_hot."""
    return _default_converter.to_one_hot(real, fake, categorical_columns)


def to_numerical_one_hot(
    real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward compatibility wrapper for to_numerical_one_hot."""
    return _default_converter.to_numerical_one_hot(real, fake, categorical_columns)
