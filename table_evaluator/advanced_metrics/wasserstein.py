"""Wasserstein Distance (Earth Mover's Distance) implementation for distribution comparison."""

import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def _sample_for_performance(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    max_samples: int = 10000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample data for performance optimization while maintaining statistical properties.

    Args:
        real_data: Real data array
        fake_data: Fake data array
        max_samples: Maximum number of samples to keep per dataset
        random_state: Random seed for reproducibility

    Returns:
        Tuple of sampled real and fake data
    """
    np.random.seed(random_state)

    # Sample real data if needed
    if len(real_data) > max_samples:
        indices = np.random.choice(len(real_data), max_samples, replace=False)
        real_sampled = real_data[indices]
        logger.info(f"Sampled real data from {len(real_data)} to {max_samples} rows")
    else:
        real_sampled = real_data

    # Sample fake data if needed
    if len(fake_data) > max_samples:
        indices = np.random.choice(len(fake_data), max_samples, replace=False)
        fake_sampled = fake_data[indices]
        logger.info(f"Sampled fake data from {len(fake_data)} to {max_samples} rows")
    else:
        fake_sampled = fake_data

    return real_sampled, fake_sampled


def _check_large_dataset_and_warn(
    real_data: Union[np.ndarray, pd.DataFrame],
    fake_data: Union[np.ndarray, pd.DataFrame],
    threshold: int = 250000,
    operation_name: str = "calculation",
) -> None:
    """
    Check for large datasets and provide sampling recommendations.

    Args:
        real_data: Real data array or DataFrame
        fake_data: Fake data array or DataFrame
        threshold: Row count threshold for large dataset warning
        operation_name: Name of the operation being performed for context
    """
    total_rows = len(real_data) + len(fake_data)

    if total_rows > threshold:
        logger.warning(
            f"Large dataset detected for {operation_name} ({total_rows:,} total rows). "
            f"This may be slow and memory-intensive. Consider enabling sampling by setting "
            f"'enable_sampling=True' and 'max_samples=5000' parameters to improve performance."
        )


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
        ValueError: If p is not 1 or 2, or if inputs are invalid
        TypeError: If inputs are not pandas Series
    """
    # Input validation
    if not isinstance(real_col, pd.Series):
        raise TypeError("real_col must be a pandas Series")
    if not isinstance(fake_col, pd.Series):
        raise TypeError("fake_col must be a pandas Series")
    if p not in [1, 2]:
        raise ValueError("p must be 1 or 2")

    # Remove NaN values and validate
    real_clean = real_col.dropna().values
    fake_clean = fake_col.dropna().values

    if len(real_clean) == 0:
        raise ValueError("real_col contains no valid (non-NaN) values")
    if len(fake_clean) == 0:
        raise ValueError("fake_col contains no valid (non-NaN) values")

    # Check for infinite values
    if np.any(np.isinf(real_clean)) or np.any(np.isinf(fake_clean)):
        raise ValueError("Input data contains infinite values")

    try:
        return stats.wasserstein_distance(real_clean, fake_clean)
    except Exception as e:
        raise ValueError(f"Failed to calculate Wasserstein distance: {str(e)}")


def wasserstein_distance_2d(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 1000,
    convergence_threshold: float = 1e-6,
    enable_sampling: bool = False,
    max_samples: int = 5000,
) -> float:
    """
    Calculate 2D Wasserstein distance using Sinkhorn algorithm approximation.

    For higher dimensional data, we use the Sinkhorn algorithm to approximate
    the optimal transport distance, which is computationally more feasible.

    Args:
        real_data: Array of shape (n_samples, 2) with real data points
        fake_data: Array of shape (m_samples, 2) with synthetic data points
        reg: Regularization parameter for entropy regularization (must be > 0)
        max_iter: Maximum number of iterations for Sinkhorn algorithm
        convergence_threshold: Threshold for convergence check
        enable_sampling: Whether to enable sampling for large datasets
        max_samples: Maximum samples per dataset when sampling is enabled

    Returns:
        float: Approximate 2D Wasserstein distance

    Raises:
        ValueError: If input data is invalid or algorithm parameters are incorrect
        TypeError: If inputs are not numpy arrays
    """
    # Input validation
    if not isinstance(real_data, np.ndarray):
        raise TypeError("real_data must be a numpy array")
    if not isinstance(fake_data, np.ndarray):
        raise TypeError("fake_data must be a numpy array")

    if real_data.ndim != 2 or fake_data.ndim != 2:
        raise ValueError("Input data must be 2D arrays")
    if real_data.shape[1] != 2 or fake_data.shape[1] != 2:
        raise ValueError("Input data must have exactly 2 features")
    if len(real_data) == 0 or len(fake_data) == 0:
        raise ValueError("Input arrays cannot be empty")

    if reg <= 0:
        raise ValueError("Regularization parameter must be positive")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be positive")
    if convergence_threshold <= 0:
        raise ValueError("Convergence threshold must be positive")

    # Check for NaN or infinite values
    if np.any(np.isnan(real_data)) or np.any(np.isnan(fake_data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(real_data)) or np.any(np.isinf(fake_data)):
        raise ValueError("Input data contains infinite values")

    # Check for large datasets and warn user
    _check_large_dataset_and_warn(
        real_data, fake_data, operation_name="2D Wasserstein distance"
    )

    # Performance optimization: sample large datasets only if enabled
    if enable_sampling and (
        len(real_data) > max_samples or len(fake_data) > max_samples
    ):
        real_data, fake_data = _sample_for_performance(
            real_data, fake_data, max_samples
        )

    n, m = len(real_data), len(fake_data)

    try:
        # Uniform distributions
        a = np.ones(n) / n
        b = np.ones(m) / m

        # Cost matrix (squared Euclidean distance)
        C = cdist(real_data, fake_data, metric="sqeuclidean")

        # Check for numerical issues in cost matrix
        if np.any(np.isnan(C)) or np.any(np.isinf(C)):
            raise ValueError("Cost matrix contains invalid values")

        # Sinkhorn algorithm with improved numerical stability
        K = np.exp(-C / reg)
        u = np.ones(n) / n

        converged = False
        for iteration in range(max_iter):
            # Add small epsilon for numerical stability
            epsilon = 1e-16
            v = b / (K.T @ u + epsilon)
            u_new = a / (K @ v + epsilon)

            # Check convergence
            if np.allclose(u, u_new, rtol=convergence_threshold, atol=1e-12):
                converged = True
                break
            u = u_new

        if not converged:
            raise ValueError(
                f"Sinkhorn algorithm did not converge after {max_iter} iterations"
            )

        # Calculate optimal transport cost
        transport_cost = np.sum(u[:, None] * K * v[None, :] * C)

        if np.isnan(transport_cost) or np.isinf(transport_cost):
            raise ValueError("Transport cost calculation resulted in invalid value")

        return float(transport_cost)

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to calculate 2D Wasserstein distance: {str(e)}")


def wasserstein_distance_df(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    numerical_columns: List[str],
    p: int = 1,
    method: str = "1d",
    enable_sampling: bool = False,
    max_samples_2d: int = 5000,
) -> pd.DataFrame:
    """
    Calculate Wasserstein distances for all numerical columns.

    Args:
        real: DataFrame containing real data
        fake: DataFrame containing synthetic data
        numerical_columns: List of numerical column names to evaluate
        p: Order of Wasserstein distance (1 or 2)
        method: Method to use ("1d" for column-wise, "2d" for pairwise)
        enable_sampling: Whether to enable sampling for large datasets in 2D calculations
        max_samples_2d: Maximum samples for 2D calculations when sampling is enabled

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
                    distance = wasserstein_distance_2d(
                        real_2d,
                        fake_2d,
                        enable_sampling=enable_sampling,
                        max_samples=max_samples_2d,
                    )
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
