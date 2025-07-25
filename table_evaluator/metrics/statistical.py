"""Statistical metrics and distribution comparison utilities for synthetic data evaluation."""

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from scipy import stats as ss
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

logger = logging.getLogger(__name__)


# =============================================================================
# Basic Statistical Metrics
# =============================================================================


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any]:
    """
    Returns the mean absolute error between y_true and y_pred.

    Args:
        y_true: NumPy.ndarray with the ground truth values.
        y_pred: NumPy.ndarray with the ground predicted values.

    Returns:
        Mean absolute error (float).
    """
    return np.mean(np.abs(np.subtract(y_true, y_pred)))


def euclidean_distance(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Returns the euclidean distance between y_true and y_pred.

    Args:
        y_true: The ground truth values.
        y_pred: The predicted values.

    Returns:
        The euclidean distance.
    """
    return np.sqrt(np.sum(np.power(np.subtract(y_true, y_pred), 2)))


def mean_absolute_percentage_error(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series):
    """
    Returns the mean absolute percentage error between y_true and y_pred.

    Args:
        y_true: The ground truth values.
        y_pred: The predicted values.

    Returns:
        Mean absolute percentage error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """
    Returns the root mean squared error between y_true and y_pred.

    Args:
        y_true: NumPy.ndarray with the ground truth values.
        y_pred: NumPy.ndarray with the ground predicted values.

    Returns:
        Root mean squared error (float).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))  # type: ignore


def cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    return np.sum(y_true * y_pred) / (np.sqrt(np.sum(y_true**2)) * np.sqrt(np.sum(y_pred**2)))


# =============================================================================
# Association Metrics (from association_metrics.py)
# =============================================================================


def cramers_v(
    x: list | np.ndarray | pd.Series,
    y: list | np.ndarray | pd.Series,
    bias_correction: bool = True,
) -> float:
    """
    Calculate Cramer's V statistic for categorical-categorical association.
    This is a symmetric coefficient: V(x,y) = V(y,x)

    Args:
        x: A sequence of categorical measurements
        y: A sequence of categorical measurements
        bias_correction: Use bias correction from Bergsma and Wicher

    Returns:
        Cramer's V in the range [0,1]

    Raises:
        ValueError: If inputs are empty or have mismatched lengths
        TypeError: If inputs cannot be converted to pandas Series
    """
    # Type and input validation
    if not isinstance(bias_correction, bool):
        raise TypeError("bias_correction must be a boolean")

    try:
        # Convert to pandas Series for easier handling
        x_series = pd.Series(x) if not isinstance(x, pd.Series) else x
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot convert inputs to pandas Series: {e}") from e

    # Validate inputs
    if len(x_series) != len(y_series):
        raise ValueError(f"Input arrays must have same length: {len(x_series)} vs {len(y_series)}")

    if len(x_series) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Handle missing values by dropping them
    mask = ~(x_series.isna() | y_series.isna())
    x_clean = x_series[mask]
    y_clean = y_series[mask]

    if len(x_clean) == 0:
        return 0.0  # No valid data points - return 0 association

    # Create contingency table
    confusion_matrix = pd.crosstab(x_clean, y_clean)

    if confusion_matrix.shape[0] == 1 or confusion_matrix.shape[1] == 1:
        return 0.0

    # Calculate chi-squared statistic (disable Yates correction for consistency)
    chi2, _, _, _ = chi2_contingency(confusion_matrix, correction=False)

    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1

    # Calculate Cramer's V
    cramers_v_val = np.sqrt(chi2 / (n * min_dim))

    if bias_correction:
        # Bias correction from Bergsma and Wicher
        k = confusion_matrix.shape[0]  # number of rows
        r = confusion_matrix.shape[1]  # number of columns

        phi_corrected = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
        k_corrected = k - (k - 1) ** 2 / (n - 1)
        r_corrected = r - (r - 1) ** 2 / (n - 1)

        min_dim_corrected = min(k_corrected - 1, r_corrected - 1)
        if min_dim_corrected > 0:
            cramers_v_val = np.sqrt(phi_corrected / min_dim_corrected)
        else:
            cramers_v_val = 0.0

    return min(1.0, cramers_v_val)


def theils_u(x: list | np.ndarray | pd.Series, y: list | np.ndarray | pd.Series) -> float:
    """
    Calculate Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
    This is the uncertainty of x given y: U(x,y) != U(y,x)

    Args:
        x: A sequence of categorical measurements
        y: A sequence of categorical measurements

    Returns:
        Theil's U in the range [0,1]

    Raises:
        ValueError: If inputs are empty or have mismatched lengths
        TypeError: If inputs cannot be converted to pandas Series
    """
    try:
        # Convert to pandas Series for easier handling
        x_series = pd.Series(x) if not isinstance(x, pd.Series) else x
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot convert inputs to pandas Series: {e}") from e

    # Validate inputs
    if len(x_series) != len(y_series):
        raise ValueError(f"Input arrays must have same length: {len(x_series)} vs {len(y_series)}")

    if len(x_series) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Handle missing values by dropping them
    mask = ~(x_series.isna() | y_series.isna())
    x_clean = x_series[mask]
    y_clean = y_series[mask]

    if len(x_clean) == 0:
        return 0.0

    # Calculate entropy of x using Shannon entropy formula: H(X) = -Σ p(x) * log2(p(x))
    x_counter = x_clean.value_counts()
    x_probs = x_counter / len(x_clean)
    EPSILON = 1e-10  # Small constant to prevent log(0), typical value for numerical stability
    h_x = -np.sum(x_probs * np.log2(x_probs + EPSILON))  # Shannon entropy in bits

    if h_x == 0:
        return 0.0

    # Calculate conditional entropy H(X|Y)
    h_x_given_y = 0.0
    y_counter = y_clean.value_counts()

    for y_val in y_counter.index:
        y_mask = y_clean == y_val
        x_given_y = x_clean[y_mask]

        if len(x_given_y) > 0:
            p_y = len(x_given_y) / len(x_clean)
            x_given_y_counter = x_given_y.value_counts()
            x_given_y_probs = x_given_y_counter / len(x_given_y)
            h_x_given_y_val = -np.sum(x_given_y_probs * np.log2(x_given_y_probs + EPSILON))
            h_x_given_y += p_y * h_x_given_y_val

    # Calculate Theil's U = (H(X) - H(X|Y)) / H(X)
    # This represents the uncertainty coefficient: how much knowing Y reduces uncertainty about X
    u = (h_x - h_x_given_y) / h_x
    return max(0.0, min(1.0, u))  # Clamp to [0,1] range for numerical stability


def correlation_ratio(
    categorical: list | np.ndarray | pd.Series,
    numerical: list | np.ndarray | pd.Series,
) -> float:
    """
    Calculate correlation ratio (eta) for categorical-numerical association.

    Args:
        categorical: A sequence of categorical measurements
        numerical: A sequence of numerical measurements

    Returns:
        Correlation ratio in the range [0,1]
    """
    # Convert to pandas Series for easier handling
    cat_series = pd.Series(categorical) if not isinstance(categorical, pd.Series) else categorical
    num_series = pd.Series(numerical) if not isinstance(numerical, pd.Series) else numerical

    # Handle missing values by dropping them
    mask = ~(cat_series.isna() | num_series.isna())
    cat_clean = cat_series[mask]
    num_clean = num_series[mask]

    if len(cat_clean) == 0 or len(cat_clean.unique()) == 1:
        return 0.0

    # Calculate overall mean
    overall_mean = num_clean.mean()

    # Calculate sum of squares total
    ss_total = np.sum((num_clean - overall_mean) ** 2)

    if ss_total == 0:
        return 0.0

    # Calculate sum of squares between groups
    ss_between = 0.0
    for category in cat_clean.unique():
        group_data = num_clean[cat_clean == category]
        if len(group_data) > 0:
            group_mean = group_data.mean()
            ss_between += len(group_data) * (group_mean - overall_mean) ** 2

    # Calculate correlation ratio (eta): sqrt(SS_between / SS_total)
    # This is the square root of the eta-squared (coefficient of determination)
    # representing the proportion of variance in numerical variable explained by categorical variable
    eta_squared = ss_between / ss_total
    return np.sqrt(max(0.0, min(1.0, eta_squared)))  # Clamp for numerical stability


def associations(
    dataset: pd.DataFrame,
    nominal_columns: list[str] | str | None = "auto",
    nom_nom_assoc: str = "cramer",
    num_num_assoc: str = "pearson",
    nom_num_assoc: str = "correlation_ratio",
) -> dict[str, pd.DataFrame]:
    """
    Calculate the correlation/strength-of-association matrix for mixed data types.

    Args:
        dataset: DataFrame with mixed data types
        nominal_columns: List of categorical column names, 'auto' to detect, or 'all'
        nom_nom_assoc: Method for categorical-categorical associations ('cramer' or 'theil')
        num_num_assoc: Method for numerical-numerical associations ('pearson', 'spearman', 'kendall')
        nom_num_assoc: Method for categorical-numerical associations ('correlation_ratio')
        compute_only: If True, skip plotting and only return correlation matrix

    Returns:
        dict: Dictionary containing 'corr' key with correlation DataFrame
    """
    if nominal_columns == "auto":
        # Auto-detect categorical columns based on dtype
        nominal_columns = list(dataset.select_dtypes(include=["object", "category", "bool"]).columns)
    elif nominal_columns == "all":
        nominal_columns = list(dataset.columns)
    elif nominal_columns is None:
        nominal_columns = []

    # Initialize correlation matrix
    corr_matrix = pd.DataFrame(index=dataset.columns, columns=dataset.columns, dtype=float)

    # Fill diagonal with 1s
    np.fill_diagonal(corr_matrix.values, 1.0)

    # Calculate associations for each pair
    for i, col1 in enumerate(dataset.columns):
        for j, col2 in enumerate(dataset.columns):
            if i < j:  # Only calculate upper triangle, then mirror
                col1_is_nominal = col1 in nominal_columns
                col2_is_nominal = col2 in nominal_columns

                try:
                    if col1_is_nominal and col2_is_nominal:
                        # Categorical-categorical association
                        if nom_nom_assoc == "cramer":
                            assoc_value = cramers_v(dataset[col1], dataset[col2])
                        elif nom_nom_assoc == "theil":
                            assoc_value = theils_u(dataset[col1], dataset[col2])
                        else:
                            raise ValueError(f"Unknown nom_nom_assoc: {nom_nom_assoc}")

                    elif not col1_is_nominal and not col2_is_nominal:
                        # Numerical-numerical association
                        if num_num_assoc == "pearson":
                            assoc_value, _ = stats.pearsonr(dataset[col1].dropna(), dataset[col2].dropna())
                            # Keep original sign for Pearson correlation to match dython behavior
                        elif num_num_assoc == "spearman":
                            assoc_value, _ = stats.spearmanr(dataset[col1].dropna(), dataset[col2].dropna())
                            assoc_value = abs(assoc_value)
                        elif num_num_assoc == "kendall":
                            assoc_value, _ = stats.kendalltau(dataset[col1].dropna(), dataset[col2].dropna())
                            assoc_value = abs(assoc_value)
                        else:
                            raise ValueError(f"Unknown num_num_assoc: {num_num_assoc}")

                    else:
                        # Categorical-numerical association
                        if nom_num_assoc == "correlation_ratio":
                            if col1_is_nominal:
                                assoc_value = correlation_ratio(dataset[col1], dataset[col2])
                            else:
                                assoc_value = correlation_ratio(dataset[col2], dataset[col1])
                        else:
                            raise ValueError(f"Unknown nom_num_assoc: {nom_num_assoc}")

                    # Handle NaN results
                    if np.isnan(assoc_value):
                        assoc_value = 0.0

                except (
                    ValueError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                    TypeError,
                ) as e:
                    warnings.warn(
                        f"Could not calculate association between {col1} and {col2}: {e}",
                        RuntimeWarning,
                    )
                    assoc_value = 0.0
                except Exception as e:
                    # Log unexpected errors but don't crash - return 0 association
                    warnings.warn(
                        f"Unexpected error calculating association between {col1} and {col2}: {e}",
                        RuntimeWarning,
                    )
                    assoc_value = 0.0

                # Fill both upper and lower triangles (make symmetric for most cases)
                corr_matrix.loc[col1, col2] = assoc_value
                if nom_nom_assoc != "theil" or not (col1_is_nominal and col2_is_nominal):
                    # Make symmetric except for Theil's U which is asymmetric
                    corr_matrix.loc[col2, col1] = assoc_value
                else:
                    # For Theil's U, calculate the reverse direction
                    try:
                        reverse_assoc = theils_u(dataset[col2], dataset[col1])
                        corr_matrix.loc[col2, col1] = reverse_assoc if not np.isnan(reverse_assoc) else 0.0
                    except (ValueError, RuntimeError) as e:
                        warnings.warn(
                            f"Could not calculate reverse Theil's U between {col2} and {col1}: {e}",
                            RuntimeWarning,
                        )
                        corr_matrix.loc[col2, col1] = 0.0

    return {"corr": corr_matrix.astype(float)}


def column_correlations(
    dataset_a: pd.DataFrame,
    dataset_b: pd.DataFrame,
    categorical_columns: list[str] | None,
    theil_u=True,
) -> float:
    """
    Column-wise correlation calculation between ``dataset_a`` and ``dataset_b``.

    Args:
        dataset_a: First DataFrame
        dataset_b: Second DataFrame
        categorical_columns: The columns containing categorical values
        theil_u: Whether to use Theil's U. If False, use Cramer's V.

    Returns:
        Mean correlation between all columns.
    """
    if categorical_columns is None:
        categorical_columns = list()
    elif categorical_columns == "all":
        categorical_columns = dataset_a.columns
    assert dataset_a.columns.tolist() == dataset_b.columns.tolist()
    corr = pd.DataFrame(columns=dataset_a.columns, index=["correlation"])

    for column in dataset_a.columns.tolist():
        if column in categorical_columns:
            if theil_u:
                corr[column] = theils_u(dataset_a[column].sort_values(), dataset_b[column].sort_values())
            else:
                corr[column] = cramers_v(dataset_a[column].sort_values(), dataset_b[column].sort_values())
        else:
            corr[column], _ = ss.pearsonr(dataset_a[column].sort_values(), dataset_b[column].sort_values())
    corr.fillna(value=np.nan, inplace=True)
    correlation = np.mean(corr.values.flatten())
    return float(correlation)


# =============================================================================
# Distribution Comparison Metrics
# =============================================================================


def js_distance_df(real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: list[str]) -> pd.DataFrame:
    """
    Calculate Jensen-Shannon distances between real and fake data for numerical columns.

    This function computes the Jensen-Shannon distance for each numerical column
    in parallel using joblib's Parallel and delayed functions.

    Args:
        real: DataFrame containing the real data.
        fake: DataFrame containing the fake data.
        numerical_columns: List of column names to compute distances for.

    Returns:
        A DataFrame with column names as index and Jensen-Shannon
        distances as values.

    Raises:
        AssertionError: If the columns in real and fake DataFrames are not identical.
    """
    assert real.columns.tolist() == fake.columns.tolist(), "Columns are not identical between `real` and `fake`. "
    distances = Parallel(n_jobs=-1)(
        delayed(jensenshannon_distance)(col, real[col], fake[col]) for col in numerical_columns
    )

    distances_df = pd.DataFrame(distances)
    return distances_df.set_index("col_name")


def jensenshannon_distance(colname: str, real_col: pd.Series, fake_col: pd.Series, bins: int = 25) -> dict[str, Any]:
    """
    Calculate the Jensen-Shannon distance between real and fake data columns.

    This function bins the data, calculates probability distributions, and then
    computes the Jensen-Shannon distance between these distributions.

    Args:
        colname: Name of the column being analyzed.
        real_col: Series containing the real data.
        fake_col: Series containing the fake data.
        bins: Number of bins to use for discretization. Defaults to 25.

    Returns:
        A dictionary containing:
        - 'col_name': Name of the column.
        - 'js_distance': The calculated Jensen-Shannon distance.

    Note:
        The number of bins is capped at the length of the real column to avoid empty bins.
    """
    bins = min(bins, len(real_col))
    binned_values_real, actual_bins = pd.cut(x=real_col, bins=bins, retbins=True)
    binned_probs_real = binned_values_real.value_counts(normalize=True, sort=False)
    binned_probs_fake = pd.cut(fake_col, bins=actual_bins).value_counts(normalize=True, sort=False)
    js_distance = jensenshannon(binned_probs_real, binned_probs_fake)
    return {"col_name": colname, "js_distance": js_distance}


def kolmogorov_smirnov_test(col_name: str, real_col: pd.Series, fake_col: pd.Series) -> dict[str, Any]:
    """
    Perform Kolmogorov-Smirnov test on real and fake data columns.

    Args:
        col_name: Name of the column being tested.
        real_col: Series containing the real data.
        fake_col: Series containing the fake data.

    Returns:
        A dictionary containing:
        - 'col_name': Name of the column.
        - 'statistic': The KS statistic.
        - 'p-value': The p-value of the test.
        - 'equality': 'identical' if p-value > 0.01, else 'different'.
    """
    statistic, p_value = ks_2samp(real_col, fake_col)
    equality = "identical" if p_value > 0.01 else "different"  # type: ignore
    return {
        "col_name": col_name,
        "statistic": statistic,
        "p-value": p_value,
        "equality": equality,
    }


def kolmogorov_smirnov_df(real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
    """Calculate Kolmogorov-Smirnov test results for numerical columns."""
    assert real.columns.tolist() == fake.columns.tolist(), "Columns are not identical between `real` and `fake`. "
    distances = Parallel(n_jobs=-1)(
        delayed(kolmogorov_smirnov_test)(col, real[col], fake[col]) for col in numerical_columns
    )
    distances_df = pd.DataFrame(distances)
    return distances_df.set_index("col_name")


# =============================================================================
# Wasserstein Distance Implementation
# =============================================================================


def _sample_for_performance(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    max_samples: int = 10000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
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
    real_data: np.ndarray | pd.DataFrame,
    fake_data: np.ndarray | pd.DataFrame,
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


def wasserstein_distance_1d(real_col: pd.Series, fake_col: pd.Series, p: int = 1) -> float:
    """
    Calculate 1D Wasserstein distance between two distributions.

    The Wasserstein distance (also known as Earth Mover's Distance) measures
    the minimum cost of transforming one distribution into another.

    Args:
        real_col: Series containing real data values
        fake_col: Series containing synthetic data values
        p: Order of the Wasserstein distance (1 or 2)

    Returns:
        Wasserstein distance between the two distributions

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
        raise ValueError(f"Failed to calculate Wasserstein distance: {e!s}")


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
        Approximate 2D Wasserstein distance

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
    _check_large_dataset_and_warn(real_data, fake_data, operation_name="2D Wasserstein distance")

    # Performance optimization: sample large datasets only if enabled
    if enable_sampling and (len(real_data) > max_samples or len(fake_data) > max_samples):
        real_data, fake_data = _sample_for_performance(real_data, fake_data, max_samples)

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
            raise ValueError(f"Sinkhorn algorithm did not converge after {max_iter} iterations")

        # Calculate optimal transport cost
        transport_cost = np.sum(u[:, None] * K * v[None, :] * C)

        if np.isnan(transport_cost) or np.isinf(transport_cost):
            raise ValueError("Transport cost calculation resulted in invalid value")

        return float(transport_cost)

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to calculate 2D Wasserstein distance: {e!s}")


def wasserstein_distance_df(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    numerical_columns: list[str],
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
    assert real.columns.tolist() == fake.columns.tolist(), "Columns are not identical between real and fake DataFrames"

    distances = []

    if method == "1d":
        for col in numerical_columns:
            distance = wasserstein_distance_1d(real[col], fake[col], p=p)
            distances.append({"column": col, "wasserstein_distance": distance, "method": f"1D_p{p}"})

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


def earth_movers_distance_summary(real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: list[str]) -> dict:
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
            "normalized_distance": distance / (real[col].std() + fake[col].std() + 1e-10),
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
        "worst_column": distances_1d.loc[distances_1d["wasserstein_distance"].idxmax(), "column"],
    }

    return results


def optimal_transport_cost(
    real_col: pd.Series, fake_col: pd.Series, bins: int = 50
) -> tuple[float, np.ndarray, np.ndarray]:
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


def mmd_comprehensive_analysis(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    numerical_columns: list[str],
    kernel_types: list[str] = ["rbf", "polynomial", "linear"],
) -> dict:
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
        results["column_wise"][kernel_type] = mmd_column_wise(real, fake, numerical_columns, kernel_type)

    # Multivariate analysis
    for kernel_type in kernel_types:
        results["multivariate"][kernel_type] = mmd_multivariate(real, fake, numerical_columns, kernel_type)

    # Summary statistics
    all_mmds = []

    for kernel_type in kernel_types:
        df = results["column_wise"][kernel_type]
        all_mmds.extend(df["mmd_squared"].dropna().tolist())

    results["summary"] = {
        "mean_mmd": np.mean(all_mmds) if all_mmds else np.nan,
        "median_mmd": np.median(all_mmds) if all_mmds else np.nan,
        "max_mmd": np.max(all_mmds) if all_mmds else np.nan,
        "overall_quality_score": 1.0 / (1.0 + np.mean(all_mmds)) if all_mmds else 0.0,
    }

    return results


# =============================================================================
# Maximum Mean Discrepancy (MMD) Implementation
# =============================================================================


def _sample_for_mmd_performance(
    X: np.ndarray, Y: np.ndarray, max_samples: int = 5000, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
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
        logger.info(f"Sampled X data from {len(X)} to {max_samples} rows")
    else:
        X_sampled = X

    # Sample Y if needed
    if len(Y) > max_samples:
        indices = np.random.choice(len(Y), max_samples, replace=False)
        Y_sampled = Y[indices]
        logger.info(f"Sampled Y data from {len(Y)} to {max_samples} rows")
    else:
        Y_sampled = Y

    return X_sampled, Y_sampled


def _check_large_dataset_and_warn_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    threshold: int = 250000,
    operation_name: str = "MMD calculation",
) -> None:
    """
    Check for large datasets and provide sampling recommendations for MMD.

    Args:
        X: First dataset
        Y: Second dataset
        threshold: Row count threshold for large dataset warning
        operation_name: Name of the operation being performed for context
    """
    total_rows = len(X) + len(Y)

    if total_rows > threshold:
        logger.warning(
            f"Large dataset detected for {operation_name} ({total_rows:,} total rows). "
            f"This may be slow and memory-intensive. Consider enabling sampling by setting "
            f"'enable_sampling=True' and 'max_samples=5000' parameters to improve performance."
        )


class MMDKernel:
    """Base class for MMD kernel functions."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """Compute kernel matrix."""
        raise NotImplementedError


class RBFKernel(MMDKernel):
    """Radial Basis Function (Gaussian) kernel."""

    def __init__(self, gamma: float | None = None):
        super().__init__("rbf")
        self.gamma = gamma

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if Y is not None and not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy array or None")

        gamma = self.gamma
        if gamma is None:
            # Use median heuristic for bandwidth selection
            try:
                if Y is None:
                    pairwise_dists = np.sqrt(np.sum((X[:, None] - X[None, :]) ** 2, axis=2))
                else:
                    pairwise_dists = np.sqrt(np.sum((X[:, None] - Y[None, :]) ** 2, axis=2))

                valid_dists = pairwise_dists[pairwise_dists > 0]
                if len(valid_dists) == 0:
                    raise ValueError("All pairwise distances are zero - cannot compute gamma")

                median_dist = np.median(valid_dists)
                if median_dist == 0:
                    raise ValueError("Median distance is zero - cannot compute gamma")

                gamma = 1.0 / (2 * median_dist**2)
            except Exception as e:
                raise ValueError(f"Failed to compute gamma using median heuristic: {e!s}")

        try:
            return rbf_kernel(X, Y, gamma=gamma)
        except Exception as e:
            raise ValueError(f"Failed to compute RBF kernel: {e!s}")


class PolynomialKernel(MMDKernel):
    """Polynomial kernel."""

    def __init__(self, degree: int = 3, gamma: float | None = None, coef0: float = 1.0):
        super().__init__("polynomial")
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        return polynomial_kernel(X, Y, degree=self.degree, gamma=self.gamma, coef0=self.coef0)


class LinearKernel(MMDKernel):
    """Linear kernel."""

    def __init__(self):
        super().__init__("linear")

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        return linear_kernel(X, Y)


def mmd_squared(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: MMDKernel,
    unbiased: bool = True,
    enable_sampling: bool = False,
    max_samples: int = 5000,
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
        enable_sampling: Whether to enable sampling for large datasets
        max_samples: Maximum samples per dataset when sampling is enabled

    Returns:
        Squared MMD value

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

    # Check for large datasets and warn user
    _check_large_dataset_and_warn_mmd(X, Y, operation_name="MMD calculation")

    # Performance optimization: sample large datasets only if enabled
    if enable_sampling and (len(X) > max_samples or len(Y) > max_samples):
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
        raise ValueError(f"Failed to calculate MMD: {e!s}")


def mmd_column_wise(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    numerical_columns: list[str],
    kernel_type: str = "rbf",
    kernel_params: dict | None = None,
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
        else:
            mmd_val = mmd_squared(real_col, fake_col, kernel, unbiased=True)

        results.append(
            {
                "column": col,
                "mmd_squared": mmd_val,
                "kernel": kernel_type,
            }
        )

    return pd.DataFrame(results)


def mmd_multivariate(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    numerical_columns: list[str],
    kernel_type: str = "rbf",
    kernel_params: dict | None = None,
    enable_sampling: bool = False,
    max_samples: int = 5000,
) -> dict:
    """
    Calculate multivariate MMD using all numerical columns together.

    Args:
        real: DataFrame with real data
        fake: DataFrame with synthetic data
        numerical_columns: Columns to include in analysis
        kernel_type: Type of kernel to use
        kernel_params: Kernel parameters
        enable_sampling: Whether to enable sampling for large datasets
        max_samples: Maximum samples to use when sampling is enabled

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

    # Calculate MMD
    mmd_val = mmd_squared(
        real_data, fake_data, kernel, unbiased=True, enable_sampling=enable_sampling, max_samples=max_samples
    )

    return {
        "mmd_squared": mmd_val,
        "kernel": kernel_type,
        "n_real_samples": len(real_data),
        "n_fake_samples": len(fake_data),
        "dimensions": len(numerical_columns),
    }
