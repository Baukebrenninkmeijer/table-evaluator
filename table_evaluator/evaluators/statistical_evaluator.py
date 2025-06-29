"""Statistical evaluation functionality extracted from TableEvaluator."""

from typing import Callable, List

import numpy as np
import pandas as pd
from table_evaluator.association_metrics import associations
from scipy import stats
from sklearn.decomposition import PCA

from table_evaluator.metrics import mean_absolute_percentage_error


class StatisticalEvaluator:
    """Handles statistical evaluation of real vs synthetic data."""

    def __init__(self, comparison_metric: Callable, verbose: bool = False):
        """
        Initialize the statistical evaluator.

        Args:
            comparison_metric: Function to compare two arrays (e.g., stats.pearsonr)
            verbose: Whether to print detailed output
        """
        self.comparison_metric = comparison_metric
        self.verbose = verbose

    def basic_statistical_evaluation(
        self, real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: List[str]
    ) -> float:
        """
        Calculate the correlation coefficient between the basic properties of real and fake data.

        Uses Spearman's Rho as it's more resilient to outliers and magnitude differences.

        Args:
            real: Real dataset
            fake: Synthetic dataset
            numerical_columns: List of numerical column names

        Returns:
            float: Correlation coefficient between basic statistical properties
        """
        total_metrics = pd.DataFrame()

        for ds_name, ds in [("real", real), ("fake", fake)]:
            metrics = {}
            num_ds = ds[numerical_columns]

            # Calculate basic statistics
            for idx, value in num_ds.mean().items():
                metrics[f"mean_{idx}"] = value
            for idx, value in num_ds.median().items():
                metrics[f"median_{idx}"] = value
            for idx, value in num_ds.std().items():
                metrics[f"std_{idx}"] = value
            for idx, value in num_ds.var().items():
                metrics[f"variance_{idx}"] = value

            total_metrics[ds_name] = metrics.values()

        total_metrics.index = list(metrics.keys())

        if self.verbose:
            print("\nBasic statistical attributes:")
            print(total_metrics.to_string())

        corr, p = stats.spearmanr(total_metrics["real"], total_metrics["fake"])
        return corr

    def correlation_correlation(
        self, real: pd.DataFrame, fake: pd.DataFrame, categorical_columns: List[str]
    ) -> float:
        """
        Calculate correlation coefficient between association matrices of real and fake data.

        Args:
            real: Real dataset
            fake: Synthetic dataset
            categorical_columns: List of categorical column names

        Returns:
            float: Correlation coefficient between association matrices
        """
        total_metrics = pd.DataFrame()

        for ds_name, ds in [("real", real), ("fake", fake)]:
            corr_df: pd.DataFrame = associations(
                ds,
                nominal_columns=categorical_columns,
                nom_nom_assoc="theil",
                compute_only=True,
            )["corr"]

            values = corr_df.values
            # Remove diagonal elements (self-correlations)
            values = values[~np.eye(values.shape[0], dtype=bool)].reshape(
                values.shape[0], -1
            )
            total_metrics[ds_name] = values.flatten()

        if self.verbose:
            print("\nColumn correlation between datasets:")
            print(total_metrics.to_string())

        corr, p = self.comparison_metric(total_metrics["real"], total_metrics["fake"])
        return corr

    def pca_correlation(
        self, real: pd.DataFrame, fake: pd.DataFrame, lingress: bool = False
    ) -> float:
        """
        Calculate the relation between PCA explained variance values.

        Args:
            real: Real dataset (must be numerical)
            fake: Synthetic dataset (must be numerical)
            lingress: Whether to use linear regression (Pearson's r) vs MAPE

        Returns:
            float: Correlation coefficient if lingress=True, otherwise 1 - MAPE(log)
        """
        # Limit PCA components to min of 5 or number of features/samples
        n_components = min(5, real.shape[1], real.shape[0] - 1)
        pca_real = PCA(n_components=n_components)
        pca_fake = PCA(n_components=n_components)

        pca_real.fit(real)
        pca_fake.fit(fake)

        if self.verbose:
            results = pd.DataFrame(
                {
                    "real": pca_real.explained_variance_,
                    "fake": pca_fake.explained_variance_,
                }
            )
            print("\nTop 5 PCA components:")
            print(results.to_string())

        if lingress:
            corr, p = self.comparison_metric(
                pca_real.explained_variance_, pca_fake.explained_variance_
            )
            return corr
        else:
            pca_error = mean_absolute_percentage_error(
                pca_real.explained_variance_, pca_fake.explained_variance_
            )
            return 1 - pca_error
