"""Data conversion utilities for table evaluator."""

from typing import List, Tuple

import pandas as pd
from loguru import logger


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
        categorical_cols = (
            categorical_columns
            + real.select_dtypes("bool").columns.tolist()
            + fake.select_dtypes("bool").columns.tolist()
        )

        # Remove duplicates while preserving order
        categorical_cols = list(dict.fromkeys(categorical_cols))

        # Apply one-hot encoding
        real_encoded: pd.DataFrame = numerical_encoding(
            real, nominal_columns=categorical_cols
        ).astype(float)
        real_encoded = real_encoded.sort_index(axis=1)

        fake_encoded: pd.DataFrame = numerical_encoding(
            fake, nominal_columns=categorical_cols
        ).astype(float)

        # Ensure both datasets have the same columns
        all_columns = sorted(set(real_encoded.columns) | set(fake_encoded.columns))

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
