"""Wasserstein Distance (Earth Mover's Distance) implementation for distribution comparison."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist


def wasserstein_distance_1d(
    real_col: pd.Series, fake_col: pd.Series, p: int = 1
) -> float:
    """
    Calculate 1D Wasserstein distance between two distributions.

    The Wasserstein distance (also known as Earth Mover's Distance) measures
    the minimum cost of transforming one distribution into another.

    Args:
        real_col: Series containing real data values
        fake_col: Series containing synthetic data values
        p: Order of the Wasserstein distance (1 or 2)

    Returns:
        float: Wasserstein distance between the two distributions

    Raises:
        ValueError: If p is not 1 or 2
    """
    if p not in [1, 2]:
        raise ValueError("p must be 1 or 2")

    # Remove NaN values
    real_clean = real_col.dropna().values
    fake_clean = fake_col.dropna().values

    if len(real_clean) == 0 or len(fake_clean) == 0:
        return np.inf

    return stats.wasserstein_distance(real_clean, fake_clean)


def wasserstein_distance_2d(
    real_data: np.ndarray, fake_data: np.ndarray, reg: float = 0.1, max_iter: int = 1000
) -> float:
    """
    Calculate 2D Wasserstein distance using Sinkhorn algorithm approximation.

    For higher dimensional data, we use the Sinkhorn algorithm to approximate
    the optimal transport distance, which is computationally more feasible.

    Args:
        real_data: Array of shape (n_samples, 2) with real data points
        fake_data: Array of shape (m_samples, 2) with synthetic data points
        reg: Regularization parameter for entropy regularization
        max_iter: Maximum number of iterations for Sinkhorn algorithm

    Returns:
        float: Approximate 2D Wasserstein distance
    """
    if real_data.shape[1] != 2 or fake_data.shape[1] != 2:
        raise ValueError("Input data must be 2D")

    n, m = len(real_data), len(fake_data)

    # Uniform distributions
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Cost matrix (squared Euclidean distance)
    C = cdist(real_data, fake_data, metric="sqeuclidean")

    # Sinkhorn algorithm
    K = np.exp(-C / reg)
    u = np.ones(n) / n

    for _ in range(max_iter):
        v = b / (K.T @ u + 1e-16)
        u_new = a / (K @ v + 1e-16)

        # Check convergence
        if np.allclose(u, u_new, rtol=1e-6):
            break
        u = u_new

    # Calculate optimal transport cost
    transport_cost = np.sum(u[:, None] * K * v[None, :] * C)

    return transport_cost


def wasserstein_distance_df(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    numerical_columns: List[str],
    p: int = 1,
    method: str = "1d",
) -> pd.DataFrame:
    """
    Calculate Wasserstein distances for all numerical columns.

    Args:
        real: DataFrame containing real data
        fake: DataFrame containing synthetic data
        numerical_columns: List of numerical column names to evaluate
        p: Order of Wasserstein distance (1 or 2)
        method: Method to use ("1d" for column-wise, "2d" for pairwise)

    Returns:
        DataFrame with column names and their Wasserstein distances

    Raises:
        AssertionError: If real and fake DataFrames don't have identical columns
    """
    assert (
        real.columns.tolist() == fake.columns.tolist()
    ), "Columns are not identical between real and fake DataFrames"

    distances = []

    if method == "1d":
        for col in numerical_columns:
            distance = wasserstein_distance_1d(real[col], fake[col], p=p)
            distances.append(
                {"column": col, "wasserstein_distance": distance, "method": f"1D_p{p}"}
            )

    elif method == "2d" and len(numerical_columns) >= 2:
        # Calculate 2D Wasserstein for all pairs of columns
        for i, col1 in enumerate(numerical_columns):
            for j, col2 in enumerate(numerical_columns[i + 1 :], i + 1):
                real_2d = real[[col1, col2]].dropna().values
                fake_2d = fake[[col1, col2]].dropna().values

                if len(real_2d) > 0 and len(fake_2d) > 0:
                    distance = wasserstein_distance_2d(real_2d, fake_2d)
                    distances.append(
                        {
                            "column": f"{col1}__{col2}",
                            "wasserstein_distance": distance,
                            "method": "2D_sinkhorn",
                        }
                    )

    else:
        raise ValueError("Invalid method or insufficient columns for 2D analysis")

    return pd.DataFrame(distances)


def optimal_transport_cost(
    real_col: pd.Series, fake_col: pd.Series, bins: int = 50
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate optimal transport cost and transport plan.

    Args:
        real_col: Series with real data values
        fake_col: Series with synthetic data values
        bins: Number of bins for discretization

    Returns:
        Tuple of (transport_cost, transport_plan, bin_edges)
    """
    # Remove NaN values
    real_clean = real_col.dropna().values
    fake_clean = fake_col.dropna().values

    # Create common bins
    combined_data = np.concatenate([real_clean, fake_clean])
    bin_edges = np.linspace(combined_data.min(), combined_data.max(), bins + 1)

    # Calculate histograms (distributions)
    real_hist, _ = np.histogram(real_clean, bins=bin_edges, density=True)
    fake_hist, _ = np.histogram(fake_clean, bins=bin_edges, density=True)

    # Normalize to probability distributions
    real_hist = real_hist / real_hist.sum()
    fake_hist = fake_hist / fake_hist.sum()

    # Calculate cost matrix (distance between bin centers)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cost_matrix = np.abs(bin_centers[:, None] - bin_centers[None, :])

    # Simple greedy transport (for visualization purposes)
    transport_plan = np.zeros_like(cost_matrix)
    real_remaining = real_hist.copy()
    fake_remaining = fake_hist.copy()

    # Greedy assignment
    for _ in range(bins):
        # Find minimum cost cell with remaining mass
        valid_mask = (real_remaining > 1e-10) & (fake_remaining > 1e-10)
        if not valid_mask.any():
            break

        masked_cost = np.where(valid_mask, cost_matrix, np.inf)
        min_idx = np.unravel_index(np.argmin(masked_cost), cost_matrix.shape)
        i, j = min_idx

        # Transport minimum available mass
        transport_amount = min(real_remaining[i], fake_remaining[j])
        transport_plan[i, j] = transport_amount
        real_remaining[i] -= transport_amount
        fake_remaining[j] -= transport_amount

    # Calculate total transport cost
    transport_cost = np.sum(transport_plan * cost_matrix)

    return transport_cost, transport_plan, bin_edges


def earth_movers_distance_summary(
    real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: List[str]
) -> dict:
    """
    Generate comprehensive Wasserstein distance summary.

    Args:
        real: DataFrame with real data
        fake: DataFrame with synthetic data
        numerical_columns: List of numerical columns to analyze

    Returns:
        Dictionary with detailed Wasserstein analysis results
    """
    results = {"summary_stats": {}, "column_distances": {}, "overall_metrics": {}}

    # Calculate 1D distances for all columns
    distances_1d = wasserstein_distance_df(real, fake, numerical_columns, p=1)

    # Store individual column results
    for _, row in distances_1d.iterrows():
        col = row["column"]
        distance = row["wasserstein_distance"]
        results["column_distances"][col] = {
            "wasserstein_1": distance,
            "normalized_distance": distance
            / (real[col].std() + fake[col].std() + 1e-10),
        }

    # Calculate summary statistics
    distances = distances_1d["wasserstein_distance"].values
    results["summary_stats"] = {
        "mean_distance": np.mean(distances),
        "median_distance": np.median(distances),
        "std_distance": np.std(distances),
        "max_distance": np.max(distances),
        "min_distance": np.min(distances),
    }

    # Overall quality score (lower is better)
    results["overall_metrics"] = {
        "quality_score": 1.0 / (1.0 + np.mean(distances)),
        "distribution_similarity": np.exp(-np.mean(distances)),
        "worst_column": distances_1d.loc[
            distances_1d["wasserstein_distance"].idxmax(), "column"
        ],
    }

    return results
