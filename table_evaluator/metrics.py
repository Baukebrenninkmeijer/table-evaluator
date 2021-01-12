import pandas as pd
import numpy as np
import scipy.stats as ss
from dython.nominal import theils_u, cramers_v
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import jensenshannon
from joblib import Parallel, delayed
from typing import Dict, Any, List
from scipy.stats import ks_2samp

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the mean absolute error between y_true and y_pred.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: Mean absolute error (float).
    """
    return np.mean(np.abs(np.subtract(y_true, y_pred)))


def euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the euclidean distance between y_true and y_pred.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :returns: Mean absolute error (float).
    """
    return np.sqrt(np.sum(np.power(np.subtract(y_true, y_pred), 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the mean absolute percentage error between y_true and y_pred. Throws ValueError if y_true contains zero values.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: Mean absolute percentage error (float).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the root mean squared error between y_true and y_pred.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: root mean squared error (float).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    return np.sum(y_true * y_pred) / (np.sqrt(np.sum(y_true ** 2)) * np.sqrt(np.sum(y_pred ** 2)))


def column_correlations(dataset_a, dataset_b, categorical_columns, theil_u=True):
    """
    Column-wise correlation calculation between ``dataset_a`` and ``dataset_b``.

    :param dataset_a: First DataFrame
    :param dataset_b: Second DataFrame
    :param categorical_columns: The columns containing categorical values
    :param theil_u: Whether to use Theil's U. If False, use Cramer's V.
    :return: Mean correlation between all columns.
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


def js_distance_df(real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: List) -> pd.DataFrame:
    assert real.columns.tolist() == fake.columns.tolist(), f'Colums are not identical between `real` and `fake`. '
    real_iter = real[numerical_columns].iteritems()
    fake_iter = fake[numerical_columns].iteritems()
    distances = Parallel(n_jobs=-1)(
        delayed(jensenshannon_distance)
        (colname, real_col, fake_col) for (colname, real_col), (_, fake_col) in zip(real_iter, fake_iter))

    distances_df = pd.DataFrame(distances)
    # distances_df = distances_df.append({'colname': 'mean', 'js_distance': distances_df.js_distance.mean()})
    return distances_df.set_index('col_name')


def jensenshannon_distance(colname: str, real_col: pd.Series, fake_col: pd.Series, bins=25) -> Dict[str, Any]:
    binned_values_real, bins = pd.cut(real_col, bins=bins, retbins=True)
    binned_probs_real = binned_values_real.value_counts(normalize=True, sort=False)
    binned_probs_fake = pd.cut(fake_col, bins=bins).value_counts(normalize=True, sort=False)
    js_distance = jensenshannon(binned_probs_real, binned_probs_fake)
    return {'col_name': colname, 'js_distance': js_distance}


def kolmogorov_smirnov_test(col_name, real_col, fake_col):
    statistic, p_value = ks_2samp(real_col, fake_col)
    equality = 'identical' if p_value > 0.01 else 'different'
    return {'col_name': col_name, 'statistic': statistic, 'p-value': p_value, 'equality': equality}

def kolmogorov_smirnov_df(real: pd.DataFrame, fake: pd.DataFrame, numerical_columns: List) -> List[Dict[str, Any]]:
    assert real.columns.tolist() == fake.columns.tolist(), f'Colums are not identical between `real` and `fake`. '
    real_iter = real[numerical_columns].iteritems()
    fake_iter = fake[numerical_columns].iteritems()
    distances = Parallel(n_jobs=-1)(
        delayed(kolmogorov_smirnov_test)
        (colname, real_col, fake_col) for (colname, real_col), (_, fake_col) in zip(real_iter, fake_iter))
    distances_df = pd.DataFrame(distances)
    return distances_df.set_index('col_name')
