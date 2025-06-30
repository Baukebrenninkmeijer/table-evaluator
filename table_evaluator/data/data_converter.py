"""Data conversion utilities for table evaluation."""

import logging
from typing import List, Tuple

import pandas as pd
# Removed dython dependency - using native implementation

logger = logging.getLogger(__name__)


class DataConverter:
    """Utilities for converting data between different representations."""

    @staticmethod
    def to_numerical(
        real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert datasets to numerical representations using factorization.

        Categorical columns are factorized to ensure both datasets have
        identical column representations.

        Args:
            real: Real dataset
            fake: Synthetic dataset
            categorical_columns: List of categorical column names

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Numerically encoded real and fake datasets
        """
        real_converted = real.copy()
        fake_converted = fake.copy()

        for column in categorical_columns:
            if real_converted[column].dtype == "object":
                real_converted[column] = pd.factorize(
                    real_converted[column], sort=True
                )[0]
                fake_converted[column] = pd.factorize(
                    fake_converted[column], sort=True
                )[0]

        return real_converted, fake_converted

    @staticmethod
    def to_one_hot(
        real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert datasets to numerical representations using one-hot encoding.

        Categorical and boolean columns are one-hot encoded to ensure both datasets
        have identical column representations.

        Args:
            real: Real dataset
            fake: Synthetic dataset
            categorical_columns: List of categorical column names

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: One-hot encoded real and fake datasets
        """
        # Include boolean columns in encoding
        extended_categorical_cols = DataConverter._extend_categorical_columns(
            categorical_columns, real, fake
        )

        # Apply one-hot encoding using native implementation
        real_encoded = DataConverter._numerical_encoding(
            real, extended_categorical_cols
        )
        fake_encoded = DataConverter._numerical_encoding(
            fake, extended_categorical_cols
        )

        # Ensure both datasets have the same columns
        return DataConverter._align_columns(real_encoded, fake_encoded)

    @staticmethod
    def to_numerical_one_hot(
        real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert datasets to numerical representations using one-hot encoding.

        This method is identical to to_one_hot and exists for backward compatibility.

        Args:
            real: Real dataset
            fake: Synthetic dataset
            categorical_columns: List of categorical column names

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: One-hot encoded real and fake datasets
        """
        return DataConverter.to_one_hot(real, fake, categorical_columns)

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

    @staticmethod
    def ensure_compatible_columns(
        real: pd.DataFrame, fake: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ensure both datasets have exactly the same columns in the same order.

        Args:
            real: Real dataset
            fake: Synthetic dataset

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Datasets with aligned columns
        """
        # Get intersection of columns
        common_columns = list(set(real.columns) & set(fake.columns))

        if len(common_columns) != len(real.columns) or len(common_columns) != len(
            fake.columns
        ):
            logger.warning("Datasets have different columns. Using intersection.")

        # Sort columns for consistency
        common_columns.sort()

        return real[common_columns].copy(), fake[common_columns].copy()

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

        return result_df.astype(float)
