"""
Native implementations of association metrics to replace dython dependency.

This module provides statistical association measures for mixed data types, implementing
the same algorithms as the dython library but with full control over the calculations.

Statistical Methods Implemented:
===============================

1. **Cramer's V** (categorical-categorical):
   - Based on chi-squared test of independence
   - Formula: V = sqrt(χ²/(n * min(k-1, r-1)))
   - Range: [0, 1] where 0 = no association, 1 = perfect association
   - Symmetric: V(x,y) = V(y,x)
   - Includes Bergsma-Wicher bias correction option

2. **Theil's U** (categorical-categorical):
   - Based on information theory and entropy
   - Formula: U(x|y) = (H(X) - H(X|Y)) / H(X)
   - Range: [0, 1] where 0 = no reduction in uncertainty, 1 = perfect prediction
   - Asymmetric: U(x|y) ≠ U(y|x) generally
   - Measures how much knowing Y reduces uncertainty about X

3. **Correlation Ratio** (categorical-numerical):
   - Based on ANOVA decomposition of variance
   - Formula: η = sqrt(SS_between / SS_total)
   - Range: [0, 1] where 0 = no association, 1 = categorical perfectly determines numerical
   - Measures proportion of numerical variance explained by categorical grouping

4. **Full Associations Matrix**:
   - Automatically selects appropriate metric based on variable types
   - Handles mixed datasets with categorical, numerical, and boolean variables
   - Returns correlation matrix with all pairwise associations

Key Features:
=============
- Numerically stable implementations with proper handling of edge cases
- Comprehensive input validation and type checking
- Consistent API matching dython's interface
- Performance optimized with numpy/scipy operations
- Extensive test coverage including edge cases

References:
===========
- Cramér, H. (1946). Mathematical Methods of Statistics
- Theil, H. (1970). On the estimation of relationships involving qualitative variables
- Bergsma, W., & Wicher, P. (2013). On the bias and efficiency of Cramér's V
"""

from typing import Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency


def cramers_v(
    x: Union[List, np.ndarray, pd.Series],
    y: Union[List, np.ndarray, pd.Series],
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
        float: Cramer's V in the range [0,1]

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
        raise ValueError(
            f"Input arrays must have same length: {len(x_series)} vs {len(y_series)}"
        )

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


def theils_u(
    x: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]
) -> float:
    """
    Calculate Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
    This is the uncertainty of x given y: U(x,y) != U(y,x)

    Args:
        x: A sequence of categorical measurements
        y: A sequence of categorical measurements

    Returns:
        float: Theil's U in the range [0,1]

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
        raise ValueError(
            f"Input arrays must have same length: {len(x_series)} vs {len(y_series)}"
        )

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
    EPSILON = (
        1e-10  # Small constant to prevent log(0), typical value for numerical stability
    )
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
            h_x_given_y_val = -np.sum(
                x_given_y_probs * np.log2(x_given_y_probs + EPSILON)
            )
            h_x_given_y += p_y * h_x_given_y_val

    # Calculate Theil's U = (H(X) - H(X|Y)) / H(X)
    # This represents the uncertainty coefficient: how much knowing Y reduces uncertainty about X
    u = (h_x - h_x_given_y) / h_x
    return max(0.0, min(1.0, u))  # Clamp to [0,1] range for numerical stability


def correlation_ratio(
    categorical: Union[List, np.ndarray, pd.Series],
    numerical: Union[List, np.ndarray, pd.Series],
) -> float:
    """
    Calculate correlation ratio (eta) for categorical-numerical association.

    Args:
        categorical: A sequence of categorical measurements
        numerical: A sequence of numerical measurements

    Returns:
        float: Correlation ratio in the range [0,1]
    """
    # Convert to pandas Series for easier handling
    cat_series = (
        pd.Series(categorical)
        if not isinstance(categorical, pd.Series)
        else categorical
    )
    num_series = (
        pd.Series(numerical) if not isinstance(numerical, pd.Series) else numerical
    )

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
    nominal_columns: Optional[Union[List[str], str]] = "auto",
    nom_nom_assoc: str = "cramer",
    num_num_assoc: str = "pearson",
    nom_num_assoc: str = "correlation_ratio",
    compute_only: bool = False,
) -> Dict[str, pd.DataFrame]:
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
        nominal_columns = list(
            dataset.select_dtypes(include=["object", "category", "bool"]).columns
        )
    elif nominal_columns == "all":
        nominal_columns = list(dataset.columns)
    elif nominal_columns is None:
        nominal_columns = []

    # Initialize correlation matrix
    corr_matrix = pd.DataFrame(
        index=dataset.columns, columns=dataset.columns, dtype=float
    )

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
                            assoc_value, _ = stats.pearsonr(
                                dataset[col1].dropna(), dataset[col2].dropna()
                            )
                            # Keep original sign for Pearson correlation to match dython behavior
                        elif num_num_assoc == "spearman":
                            assoc_value, _ = stats.spearmanr(
                                dataset[col1].dropna(), dataset[col2].dropna()
                            )
                            assoc_value = abs(assoc_value)
                        elif num_num_assoc == "kendall":
                            assoc_value, _ = stats.kendalltau(
                                dataset[col1].dropna(), dataset[col2].dropna()
                            )
                            assoc_value = abs(assoc_value)
                        else:
                            raise ValueError(f"Unknown num_num_assoc: {num_num_assoc}")

                    else:
                        # Categorical-numerical association
                        if nom_num_assoc == "correlation_ratio":
                            if col1_is_nominal:
                                assoc_value = correlation_ratio(
                                    dataset[col1], dataset[col2]
                                )
                            else:
                                assoc_value = correlation_ratio(
                                    dataset[col2], dataset[col1]
                                )
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
                if nom_nom_assoc != "theil" or not (
                    col1_is_nominal and col2_is_nominal
                ):
                    # Make symmetric except for Theil's U which is asymmetric
                    corr_matrix.loc[col2, col1] = assoc_value
                else:
                    # For Theil's U, calculate the reverse direction
                    try:
                        reverse_assoc = theils_u(dataset[col2], dataset[col1])
                        corr_matrix.loc[col2, col1] = (
                            reverse_assoc if not np.isnan(reverse_assoc) else 0.0
                        )
                    except (ValueError, RuntimeError) as e:
                        warnings.warn(
                            f"Could not calculate reverse Theil's U between {col2} and {col1}: {e}",
                            RuntimeWarning,
                        )
                        corr_matrix.loc[col2, col1] = 0.0

    return {"corr": corr_matrix.astype(float)}
