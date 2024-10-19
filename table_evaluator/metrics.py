from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.stats as ss
from dython.nominal import cramers_v, theils_u
from joblib import Parallel, delayed
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.metrics import root_mean_squared_error


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any]:
    """
    Returns the mean absolute error between y_true and y_pred.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: Mean absolute error (float).
    """
    return np.mean(np.abs(np.subtract(y_true, y_pred)))


def euclidean_distance(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Returns the euclidean distance between y_true and y_pred.

    Args:
        y_true (numpy.ndarray): The ground truth values.
        y_pred (numpy.ndarray): The predicted values.

    Returns:
        float: The mean absolute error.
    """
    return np.sqrt(np.sum(np.power(np.subtract(y_true, y_pred), 2)))


def mean_absolute_percentage_error(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series):
    """
    Returns the mean absolute percentage error between y_true and y_pred. Throws ValueError if y_true contains zero
    values.

    Args:
        y_true (numpy.ndarray): The ground truth values.
        y_pred (numpy.ndarray): The predicted values.

    Returns:
        float: Mean absolute percentage error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """
    Returns the root mean squared error between y_true and y_pred.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: root mean squared error (float).
    """
    return root_mean_squared_error(y_true, y_pred)  # type: ignore


def cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    return np.sum(y_true * y_pred) / (np.sqrt(np.sum(y_true**2)) * np.sqrt(np.sum(y_pred**2)))


def column_correlations(
    dataset_a: pd.DataFrame, dataset_b: pd.DataFrame, categorical_columns: list[str] | None, theil_u=True
):
    """
    Column-wise correlation calculation between ``dataset_a`` and ``dataset_b``.

    Args:
        dataset_a (pd.DataFrame): First DataFrame
        dataset_b (pd.DataFrame): Second DataFrame
        categorical_columns (list[str]): The columns containing categorical values
        theil_u (bool): Whether to use Theil's U. If False, use Cramer's V.

    Returns:
        float: Mean correlation between all columns.
    """
    if categorical_columns is None:
        categorical_columns = list()
    elif categorical_columns == 'all':
        categorical_columns = dataset_a.columns
    assert dataset_a.columns.tolist() == dataset_b.columns.tolist()
    corr = pd.DataFrame(columns=dataset_a.columns, index=['correlation'])

    for column in dataset_a.columns.tolist():
        if column in categorical_columns:
            if theil_u:
                corr[column] = theils_u(dataset_a[column].sort_values(), dataset_b[column].sort_values())
            else:
                corr[column] = cramers_v(dataset_a[column].sort_values(), dataset_b[column].sort_vaues())
        else:
            corr[column], _ = ss.pearsonr(dataset_a[column].sort_values(), dataset_b[column].sort_values())
    corr.fillna(value=np.nan, inplace=True)
    correlation = np.mean(corr.values.flatten())
    return correlation


def js_distance_df(real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
    """
    Calculate Jensen-Shannon distances between real and fake data for numerical columns.

    This function computes the Jensen-Shannon distance for each numerical column
    in parallel using joblib's Parallel and delayed functions.

    Args:
        real (pd.DataFrame): DataFrame containing the real data.
        fake (pd.DataFrame): DataFrame containing the fake data.
        numerical_columns (List[str]): List of column names to compute distances for.

    Returns:
        pd.DataFrame: A DataFrame with column names as index and Jensen-Shannon
                      distances as values.

    Raises:
        AssertionError: If the columns in real and fake DataFrames are not identical.
    """
    assert real.columns.tolist() == fake.columns.tolist(), 'Columns are not identical between `real` and `fake`. '
    distances = Parallel(n_jobs=-1)(
        delayed(jensenshannon_distance)(col, real[col], fake[col]) for col in numerical_columns
    )

    distances_df = pd.DataFrame(distances)
    return distances_df.set_index('col_name')


def jensenshannon_distance(colname: str, real_col: pd.Series, fake_col: pd.Series, bins: int = 25) -> Dict[str, Any]:
    """
    Calculate the Jensen-Shannon distance between real and fake data columns.

    This function bins the data, calculates probability distributions, and then
    computes the Jensen-Shannon distance between these distributions.

    Args:
        colname (str): Name of the column being analyzed.
        real_col (pd.Series): Series containing the real data.
        fake_col (pd.Series): Series containing the fake data.
        bins (int, optional): Number of bins to use for discretization. Defaults to 25.

    Returns:
        Dict[str, Any]: A dictionary containing:
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
    return {'col_name': colname, 'js_distance': js_distance}


def kolmogorov_smirnov_test(col_name: str, real_col: pd.Series, fake_col: pd.Series) -> Dict[str, Any]:
    """
    Perform Kolmogorov-Smirnov test on real and fake data columns.

    Args:
        col_name (str): Name of the column being tested.
        real_col (pd.Series): Series containing the real data.
        fake_col (pd.Series): Series containing the fake data.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'col_name': Name of the column.
            - 'statistic': The KS statistic.
            - 'p-value': The p-value of the test.
            - 'equality': 'identical' if p-value > 0.01, else 'different'.
    """
    statistic, p_value = ks_2samp(real_col, fake_col)
    equality = 'identical' if p_value > 0.01 else 'different'  # type: ignore
    return {'col_name': col_name, 'statistic': statistic, 'p-value': p_value, 'equality': equality}


def kolmogorov_smirnov_df(real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: List) -> List[Dict[str, Any]]:
    assert real.columns.tolist() == fake.columns.tolist(), 'Colums are not identical between `real` and `fake`. '
    distances = Parallel(n_jobs=-1)(
        delayed(kolmogorov_smirnov_test)(col, real[col], fake[col]) for col in numerical_columns
    )
    distances_df = pd.DataFrame(distances)
    return distances_df.set_index('col_name')
