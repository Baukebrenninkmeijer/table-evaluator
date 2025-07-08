"""Maximum Mean Discrepancy (MMD) implementation for two-sample testing."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel


def _sample_for_mmd_performance(
    X: np.ndarray, Y: np.ndarray, max_samples: int = 5000, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample data for MMD performance optimization.

    Args:
        X: First dataset
        Y: Second dataset
        max_samples: Maximum number of samples to keep per dataset
        random_state: Random seed for reproducibility

    Returns:
        Tuple of sampled X and Y data
    """
    np.random.seed(random_state)

    # Sample X if needed
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sampled = X[indices]
    else:
        X_sampled = X

    # Sample Y if needed
    if len(Y) > max_samples:
        indices = np.random.choice(len(Y), max_samples, replace=False)
        Y_sampled = Y[indices]
    else:
        Y_sampled = Y

    return X_sampled, Y_sampled


class MMDKernel:
    """Base class for MMD kernel functions."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix."""
        raise NotImplementedError


class RBFKernel(MMDKernel):
    """Radial Basis Function (Gaussian) kernel."""

    def __init__(self, gamma: Optional[float] = None):
        super().__init__("rbf")
        self.gamma = gamma

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if Y is not None and not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy array or None")

        gamma = self.gamma
        if gamma is None:
            # Use median heuristic for bandwidth selection
            try:
                if Y is None:
                    pairwise_dists = np.sqrt(
                        np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
                    )
                else:
                    pairwise_dists = np.sqrt(
                        np.sum((X[:, None] - Y[None, :]) ** 2, axis=2)
                    )

                valid_dists = pairwise_dists[pairwise_dists > 0]
                if len(valid_dists) == 0:
                    raise ValueError(
                        "All pairwise distances are zero - cannot compute gamma"
                    )

                median_dist = np.median(valid_dists)
                if median_dist == 0:
                    raise ValueError("Median distance is zero - cannot compute gamma")

                gamma = 1.0 / (2 * median_dist**2)
            except Exception as e:
                raise ValueError(
                    f"Failed to compute gamma using median heuristic: {str(e)}"
                )

        try:
            return rbf_kernel(X, Y, gamma=gamma)
        except Exception as e:
            raise ValueError(f"Failed to compute RBF kernel: {str(e)}")


class PolynomialKernel(MMDKernel):
    """Polynomial kernel."""

    def __init__(
        self, degree: int = 3, gamma: Optional[float] = None, coef0: float = 1.0
    ):
        super().__init__("polynomial")
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        return polynomial_kernel(
            X, Y, degree=self.degree, gamma=self.gamma, coef0=self.coef0
        )


class LinearKernel(MMDKernel):
    """Linear kernel."""

    def __init__(self):
        super().__init__("linear")

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        return linear_kernel(X, Y)


def mmd_squared(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: MMDKernel,
    unbiased: bool = True,
    max_samples: Optional[int] = 5000,
) -> float:
    """
    Calculate squared Maximum Mean Discrepancy between two samples.

    MMD² = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    where x,x' ~ P and y,y' ~ Q

    Args:
        X: Sample from first distribution, shape (n, d)
        Y: Sample from second distribution, shape (m, d)
        kernel: Kernel function to use
        unbiased: Whether to use unbiased estimator
        max_samples: Maximum samples per dataset for performance (None for no limit)

    Returns:
        float: Squared MMD value

    Raises:
        ValueError: If input data is invalid
        TypeError: If inputs are not numpy arrays or kernel is invalid
    """
    # Input validation
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array")
    if not isinstance(kernel, MMDKernel):
        raise TypeError("kernel must be an instance of MMDKernel")

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Input arrays must be 2D")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        raise ValueError("Input data contains infinite values")

    # Performance optimization: sample large datasets
    if max_samples is not None and (len(X) > max_samples or len(Y) > max_samples):
        X, Y = _sample_for_mmd_performance(X, Y, max_samples)

    n, m = len(X), len(Y)

    # Check minimum sample sizes for unbiased estimator
    if unbiased and (n < 2 or m < 2):
        raise ValueError("Unbiased estimator requires at least 2 samples in each group")

    try:
        # Compute kernel matrices
        Kxx = kernel(X, X)
        Kyy = kernel(Y, Y)
        Kxy = kernel(X, Y)

        # Validate kernel matrices
        if np.any(np.isnan(Kxx)) or np.any(np.isnan(Kyy)) or np.any(np.isnan(Kxy)):
            raise ValueError("Kernel computation resulted in NaN values")
        if np.any(np.isinf(Kxx)) or np.any(np.isinf(Kyy)) or np.any(np.isinf(Kxy)):
            raise ValueError("Kernel computation resulted in infinite values")

        if unbiased:
            # Unbiased estimator (removes diagonal elements)
            # E[k(x,x')] for x ≠ x'
            Kxx_sum = np.sum(Kxx) - np.trace(Kxx)
            term1 = Kxx_sum / (n * (n - 1))

            # E[k(y,y')] for y ≠ y'
            Kyy_sum = np.sum(Kyy) - np.trace(Kyy)
            term3 = Kyy_sum / (m * (m - 1))
        else:
            # Biased estimator
            term1 = np.mean(Kxx)
            term3 = np.mean(Kyy)

        # E[k(x,y)]
        term2 = np.mean(Kxy)

        mmd_sq = term1 - 2 * term2 + term3

        # Validate result
        if np.isnan(mmd_sq) or np.isinf(mmd_sq):
            raise ValueError("MMD computation resulted in invalid value")

        return max(0.0, float(mmd_sq))  # Ensure non-negative

    except Exception as e:
        if isinstance(e, (ValueError, TypeError)):
            raise
        raise ValueError(f"Failed to calculate MMD: {str(e)}")


def mmd_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: MMDKernel,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Perform permutation test for MMD significance.

    Args:
        X: Sample from first distribution
        Y: Sample from second distribution
        kernel: Kernel function to use
        n_permutations: Number of permutation samples
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (mmd_observed, p_value, null_distribution)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n, m = len(X), len(Y)

    # Observed MMD
    mmd_observed = mmd_squared(X, Y, kernel, unbiased=True)

    # Combine samples for permutation
    combined = np.vstack([X, Y])

    # Generate null distribution
    null_mmds = []
    for _ in range(n_permutations):
        # Randomly permute combined data
        perm_idx = np.random.permutation(n + m)
        perm_data = combined[perm_idx]

        # Split into two groups of original sizes
        X_perm = perm_data[:n]
        Y_perm = perm_data[n:]

        # Calculate MMD for permuted data
        mmd_perm = mmd_squared(X_perm, Y_perm, kernel, unbiased=True)
        null_mmds.append(mmd_perm)

    null_mmds = np.array(null_mmds)

    # Calculate p-value
    p_value = np.mean(null_mmds >= mmd_observed)

    return mmd_observed, p_value, null_mmds


def mmd_bootstrap_test(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: MMDKernel,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Bootstrap-based MMD test for distribution differences.

    Args:
        X: Sample from first distribution
        Y: Sample from second distribution
        kernel: Kernel function to use
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        Tuple of (mmd_observed, p_value, bootstrap_distribution)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n, m = len(X), len(Y)

    # Observed MMD
    mmd_observed = mmd_squared(X, Y, kernel, unbiased=True)

    # Bootstrap distribution under null hypothesis
    bootstrap_mmds = []
    combined = np.vstack([X, Y])

    for _ in range(n_bootstrap):
        # Bootstrap samples from combined data
        bootstrap_idx = np.random.choice(n + m, size=n + m, replace=True)
        bootstrap_data = combined[bootstrap_idx]

        # Split into two groups
        X_boot = bootstrap_data[:n]
        Y_boot = bootstrap_data[n:]

        mmd_boot = mmd_squared(X_boot, Y_boot, kernel, unbiased=True)
        bootstrap_mmds.append(mmd_boot)

    bootstrap_mmds = np.array(bootstrap_mmds)

    # Calculate p-value
    p_value = np.mean(bootstrap_mmds >= mmd_observed)

    return mmd_observed, p_value, bootstrap_mmds


def mmd_column_wise(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    numerical_columns: List[str],
    kernel_type: str = "rbf",
    kernel_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Calculate MMD for each numerical column independently.

    Args:
        real: DataFrame with real data
        fake: DataFrame with synthetic data
        numerical_columns: List of column names to evaluate
        kernel_type: Type of kernel ("rbf", "polynomial", "linear")
        kernel_params: Parameters for kernel function

    Returns:
        DataFrame with MMD results for each column
    """
    if kernel_params is None:
        kernel_params = {}

    # Initialize kernel
    if kernel_type == "rbf":
        kernel = RBFKernel(**kernel_params)
    elif kernel_type == "polynomial":
        kernel = PolynomialKernel(**kernel_params)
    elif kernel_type == "linear":
        kernel = LinearKernel()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    results = []

    for col in numerical_columns:
        real_col = real[col].dropna().values.reshape(-1, 1)
        fake_col = fake[col].dropna().values.reshape(-1, 1)

        if len(real_col) == 0 or len(fake_col) == 0:
            mmd_val = np.nan
            p_val = np.nan
        else:
            mmd_val = mmd_squared(real_col, fake_col, kernel, unbiased=True)

            # Quick permutation test (reduced iterations for speed)
            _, p_val, _ = mmd_permutation_test(
                real_col, fake_col, kernel, n_permutations=100
            )

        results.append(
            {
                "column": col,
                "mmd_squared": mmd_val,
                "p_value": p_val,
                "significant": p_val < 0.05 if not np.isnan(p_val) else False,
                "kernel": kernel_type,
            }
        )

    return pd.DataFrame(results)


def mmd_multivariate(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    numerical_columns: List[str],
    kernel_type: str = "rbf",
    kernel_params: Optional[Dict] = None,
    max_samples: int = 5000,
) -> Dict:
    """
    Calculate multivariate MMD using all numerical columns together.

    Args:
        real: DataFrame with real data
        fake: DataFrame with synthetic data
        numerical_columns: Columns to include in analysis
        kernel_type: Type of kernel to use
        kernel_params: Kernel parameters
        max_samples: Maximum samples to use (for computational efficiency)

    Returns:
        Dictionary with multivariate MMD results
    """
    if kernel_params is None:
        kernel_params = {}

    # Initialize kernel
    if kernel_type == "rbf":
        kernel = RBFKernel(**kernel_params)
    elif kernel_type == "polynomial":
        kernel = PolynomialKernel(**kernel_params)
    elif kernel_type == "linear":
        kernel = LinearKernel()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Prepare data
    real_data = real[numerical_columns].dropna().values
    fake_data = fake[numerical_columns].dropna().values

    # Subsample if datasets are too large
    if len(real_data) > max_samples:
        idx = np.random.choice(len(real_data), max_samples, replace=False)
        real_data = real_data[idx]

    if len(fake_data) > max_samples:
        idx = np.random.choice(len(fake_data), max_samples, replace=False)
        fake_data = fake_data[idx]

    # Calculate MMD
    mmd_val = mmd_squared(real_data, fake_data, kernel, unbiased=True)

    # Permutation test
    mmd_obs, p_val, null_dist = mmd_permutation_test(
        real_data, fake_data, kernel, n_permutations=200
    )

    return {
        "mmd_squared": mmd_val,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "kernel": kernel_type,
        "n_real_samples": len(real_data),
        "n_fake_samples": len(fake_data),
        "dimensions": len(numerical_columns),
        "null_distribution_mean": np.mean(null_dist),
        "null_distribution_std": np.std(null_dist),
    }


def mmd_comprehensive_analysis(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    numerical_columns: List[str],
    kernel_types: List[str] = ["rbf", "polynomial", "linear"],
) -> Dict:
    """
    Comprehensive MMD analysis with multiple kernels.

    Args:
        real: DataFrame with real data
        fake: DataFrame with synthetic data
        numerical_columns: Columns to analyze
        kernel_types: List of kernel types to use

    Returns:
        Dictionary with comprehensive MMD analysis results
    """
    results = {"column_wise": {}, "multivariate": {}, "summary": {}}

    # Column-wise analysis for each kernel
    for kernel_type in kernel_types:
        results["column_wise"][kernel_type] = mmd_column_wise(
            real, fake, numerical_columns, kernel_type
        )

    # Multivariate analysis
    for kernel_type in kernel_types:
        results["multivariate"][kernel_type] = mmd_multivariate(
            real, fake, numerical_columns, kernel_type
        )

    # Summary statistics
    all_mmds = []
    all_pvals = []

    for kernel_type in kernel_types:
        df = results["column_wise"][kernel_type]
        all_mmds.extend(df["mmd_squared"].dropna().tolist())
        all_pvals.extend(df["p_value"].dropna().tolist())

    results["summary"] = {
        "mean_mmd": np.mean(all_mmds) if all_mmds else np.nan,
        "median_mmd": np.median(all_mmds) if all_mmds else np.nan,
        "max_mmd": np.max(all_mmds) if all_mmds else np.nan,
        "fraction_significant": np.mean([p < 0.05 for p in all_pvals])
        if all_pvals
        else np.nan,
        "overall_quality_score": 1.0 / (1.0 + np.mean(all_mmds)) if all_mmds else 0.0,
    }

    return results
