"""Data conversion utilities for table evaluation."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataConverter:
    """Utilities for converting data between different representations."""

    @staticmethod
    def factorize(
        real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            if real_converted[column].dtype in ['object', 'category', 'bool']:
                real_converted[column] = pd.factorize(real_converted[column], sort=True)[0]
                fake_converted[column] = pd.factorize(fake_converted[column], sort=True)[0]

        return real_converted, fake_converted

    @staticmethod
    def to_one_hot(
        real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        # Remove duplicates while preserving order
        categorical_cols = list(dict.fromkeys(categorical_columns))

        # Apply simple one-hot encoding
        real_encoded: pd.DataFrame = pd.get_dummies(real, columns=categorical_cols).astype(float)
        real_encoded = real_encoded.sort_index(axis=1)

        fake_encoded: pd.DataFrame = pd.get_dummies(fake, columns=categorical_cols).astype(float)

        # Ensure both datasets have the same columns
        all_columns = sorted(set(real_encoded.columns) | set(fake_encoded.columns))

        for col in all_columns:
            if col not in real_encoded.columns:
                logger.warning(f'Adding missing column {col} with all 0s to real dataset')
                real_encoded[col] = 0.0
            if col not in fake_encoded.columns:
                logger.warning(f'Adding missing column {col} with all 0s to fake dataset')
                fake_encoded[col] = 0.0

        real_encoded = real_encoded[all_columns]
        fake_encoded = fake_encoded[all_columns]

        return real_encoded, fake_encoded

    @staticmethod
    def numeric_encoding(
        real: pd.DataFrame,
        fake: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        drop_single_label: bool = False,
        nan_strategy: str = 'replace',
        nan_replace_value: Any = 0.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """
        Encode datasets with mixed data to numerical-only datasets using adaptive logic.

        Uses different encoding strategies based on the number of unique values:
        * categorical with only a single value will be marked as zero (or dropped if requested)
        * categorical with two values will be replaced with factorization
        * categorical with more than two values will be replaced with one-hot encoding
        * numerical columns will not be modified

        Args:
            real: Real dataset
            fake: Synthetic dataset
            categorical_columns: List of categorical column names. If None, auto-detect
            drop_single_label: If True, columns with only one unique value will be dropped
            nan_strategy: How to handle missing values ('replace', 'drop_samples', 'drop_features')
            nan_replace_value: Value to replace missing values with when strategy is 'replace'

        Returns:
            Tuple containing:
            - Encoded real dataset
            - Encoded fake dataset
            - Dictionary with encoding information for each column
        """
        real_df = real.copy()
        fake_df = fake.copy()

        # Handle missing values
        if nan_strategy == 'replace':
            real_df.fillna(nan_replace_value, inplace=True)
            fake_df.fillna(nan_replace_value, inplace=True)
        elif nan_strategy == 'drop_samples':
            real_df.dropna(axis=0, inplace=True)
            fake_df.dropna(axis=0, inplace=True)
        elif nan_strategy == 'drop_features':
            real_df.dropna(axis=1, inplace=True)
            fake_df.dropna(axis=1, inplace=True)

        # Auto-detect categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = []
            for col in real_df.columns:
                if real_df[col].dtype in ['object', 'category', 'bool']:
                    categorical_columns.append(col)

        # Remove duplicates while preserving order
        categorical_cols = list(dict.fromkeys(categorical_columns))

        # Track encoding decisions
        encoding_info = {}

        # Separate columns by encoding strategy
        single_value_cols = []
        binary_cols = []
        multi_value_cols = []

        for col in categorical_cols:
            if col not in real_df.columns:
                continue

            # Combine unique values from both datasets
            real_unique = set(pd.unique(real_df[col]))
            fake_unique = set(pd.unique(fake_df[col]))
            all_unique = real_unique | fake_unique

            num_unique = len(all_unique)

            if num_unique <= 1:
                single_value_cols.append(col)
                encoding_info[col] = {
                    'strategy': 'single_value',
                    'unique_count': num_unique,
                }
            elif num_unique == 2:
                binary_cols.append(col)
                encoding_info[col] = {
                    'strategy': 'factorize',
                    'unique_count': num_unique,
                }
            else:
                multi_value_cols.append(col)
                encoding_info[col] = {'strategy': 'one_hot', 'unique_count': num_unique}

        # Apply encoding strategies
        real_encoded = real_df.copy()
        fake_encoded = fake_df.copy()

        # Handle single-value columns
        for col in single_value_cols:
            if drop_single_label:
                real_encoded = real_encoded.drop(columns=[col])
                fake_encoded = fake_encoded.drop(columns=[col])
                logger.info(f'Dropped single-value column: {col}')
            else:
                real_encoded[col] = 0
                fake_encoded[col] = 0
                logger.info(f'Set single-value column {col} to 0')

        # Handle binary columns using factorization (similar to to_numerical)
        if binary_cols:
            real_binary, fake_binary = DataConverter.factorize(
                real=real_encoded, fake=fake_encoded, categorical_columns=binary_cols
            )
            real_encoded = real_binary
            fake_encoded = fake_binary
            logger.info(f'Applied factorization to binary columns: {binary_cols}')

        # Handle multi-value columns using one-hot encoding
        if multi_value_cols:
            real_onehot, fake_onehot = DataConverter.to_one_hot(
                real=real_encoded, fake=fake_encoded, categorical_columns=multi_value_cols
            )
            real_encoded = real_onehot
            fake_encoded = fake_onehot
            logger.info(f'Applied one-hot encoding to multi-value columns: {multi_value_cols}')

        return real_encoded, fake_encoded, encoding_info

    @staticmethod
    def ensure_compatible_columns(real: pd.DataFrame, fake: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        if len(common_columns) != len(real.columns) or len(common_columns) != len(fake.columns):
            logger.warning('Datasets have different columns. Using intersection.')

        # Sort columns for consistency
        common_columns.sort()

        return real[common_columns].copy(), fake[common_columns].copy()
