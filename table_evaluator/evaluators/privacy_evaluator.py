"""Privacy evaluation functionality extracted from TableEvaluator."""

from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class PrivacyEvaluator:
    """Handles privacy evaluation of real vs synthetic data."""

    def __init__(self, verbose: bool = False):
        """
        Initialize the privacy evaluator.

        Args:
            verbose: Whether to print detailed output
        """
        self.verbose = verbose

    def get_copies(
        self, real: pd.DataFrame, fake: pd.DataFrame, return_len: bool = False
    ) -> Union[pd.DataFrame, int]:
        """
        Check whether any real values occur in the fake data (exact matches).

        Args:
            real: Real dataset
            fake: Synthetic dataset
            return_len: Whether to return count instead of the actual rows

        Returns:
            Union[pd.DataFrame, int]: DataFrame with copied rows or count of copies
        """
        # Create hashes of each row to efficiently find exact matches
        real_hashes = real.apply(lambda x: hash(tuple(x)), axis=1)
        fake_hashes = fake.apply(lambda x: hash(tuple(x)), axis=1)

        # Find fake rows that match real rows
        duplicate_indices = fake_hashes.isin(real_hashes.values)
        duplicate_indices = (
            duplicate_indices[duplicate_indices].sort_index().index.tolist()
        )

        if self.verbose:
            print(f"Number of copied rows: {len(duplicate_indices)}")

        copies = fake.loc[duplicate_indices, :]

        return len(copies) if return_len else copies

    def get_duplicates(
        self, real: pd.DataFrame, fake: pd.DataFrame, return_values: bool = False
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[int, int]]:
        """
        Find duplicates within each dataset separately.

        Args:
            real: Real dataset
            fake: Synthetic dataset
            return_values: Whether to return actual duplicate rows or just counts

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[int, int]]:
                Duplicate rows or counts for (real, fake) datasets
        """
        real_duplicates = real[real.duplicated(keep=False)]
        fake_duplicates = fake[fake.duplicated(keep=False)]

        if return_values:
            return real_duplicates, fake_duplicates
        else:
            return len(real_duplicates), len(fake_duplicates)

    def row_distance(
        self, real: pd.DataFrame, fake: pd.DataFrame, n_samples: int = None
    ) -> Tuple[float, float]:
        """
        Calculate mean and standard deviation of minimum distances between fake and real rows.

        This measures how close each synthetic row is to its nearest real row,
        which can indicate privacy risk.

        Args:
            real: Real dataset (must be numerical and one-hot encoded)
            fake: Synthetic dataset (must be numerical and one-hot encoded)
            n_samples: Number of samples to use (None = use all data)

        Returns:
            Tuple[float, float]: (mean_distance, std_distance) of minimum distances
        """
        if n_samples is None:
            n_samples = len(real)

        # Ensure both datasets have the same columns in the same order
        columns = sorted(real.columns.tolist())
        real_aligned = real[columns].copy()

        # Add missing columns to fake dataset with zeros
        fake_aligned = fake.copy()
        for col in columns:
            if col not in fake_aligned.columns:
                fake_aligned[col] = 0
        fake_aligned = fake_aligned[columns]

        # Standardize columns with more than 2 unique values
        for column in columns:
            if len(real_aligned[column].unique()) > 2:
                real_mean, real_std = (
                    real_aligned[column].mean(),
                    real_aligned[column].std(),
                )
                fake_mean, fake_std = (
                    fake_aligned[column].mean(),
                    fake_aligned[column].std(),
                )

                if real_std > 0:
                    real_aligned[column] = (real_aligned[column] - real_mean) / real_std
                if fake_std > 0:
                    fake_aligned[column] = (fake_aligned[column] - fake_mean) / fake_std

        # Calculate pairwise distances and find minimum distance for each fake row
        distances = cdist(real_aligned[:n_samples], fake_aligned[:n_samples])
        min_distances = np.min(distances, axis=1)

        return float(np.mean(min_distances)), float(np.std(min_distances))
