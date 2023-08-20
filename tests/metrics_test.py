import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append('..')
from pathlib import Path

from dython.nominal import associations, numerical_encoding

from table_evaluator.metrics import *
from table_evaluator.utils import load_data

data_folder = Path('data')
test_data_folder = Path('data/tests')
real, fake = load_data(data_folder/'real_test_sample.csv', data_folder/'fake_test_sample.csv')
cat_cols = ['trans_type', 'trans_operation', 'trans_k_symbol']


def test_mape():
    assert mean_absolute_percentage_error([1], [2]) == 1.0


def test_mean_absolute_error():
    assert mean_absolute_error([1, 1], [2, 2]) == 1.0


def test_euclidean_distance():
    np.testing.assert_almost_equal(euclidean_distance([0, 0], [1, 1]), 1.41421356)


def test_rmse():
    assert rmse([0, 0], [2, 2]) == 2.0


def test_column_correlation():
    column_correlations(real, fake, cat_cols)


def test_associations():
    """
    Tests that check wether the dython associations are still computed as is expected.
    """
    # load test data
    real_assoc = pd.read_parquet(test_data_folder/'real_associations.parquet')
    real_assoc_theil = pd.read_parquet(test_data_folder/'real_associations_theil.parquet')
    fake_assoc = pd.read_parquet(test_data_folder/'fake_associations.parquet')
    fake_assoc_theil = pd.read_parquet(test_data_folder/'fake_associations_theil.parquet')

    # Assert equality with saved data
    pd.testing.assert_frame_equal(real_assoc, associations(real, nominal_columns=cat_cols, compute_only=True)['corr'])
    pd.testing.assert_frame_equal(real_assoc_theil, associations(real, nominal_columns=cat_cols, nom_nom_assoc='theil', compute_only=True)['corr'])
    pd.testing.assert_frame_equal(fake_assoc, associations(fake, nominal_columns=cat_cols, compute_only=True)['corr'])
    pd.testing.assert_frame_equal(fake_assoc_theil, associations(fake, nominal_columns=cat_cols, nom_nom_assoc='theil', compute_only=True)['corr'])


def test_numerical_encoding():
    """
    Tests that check wether the dython numerical_encoding are still computed as is expected.
    """
    num_encoding = numerical_encoding(real, nominal_columns=cat_cols)
    stored_encoding = pd.read_parquet(test_data_folder/'real_test_sample_numerical_encoded.parquet')
    pd.testing.assert_frame_equal(num_encoding, stored_encoding)

    num_encoding = numerical_encoding(fake, nominal_columns=cat_cols)
    stored_encoding = pd.read_parquet(test_data_folder/'fake_test_sample_numerical_encoded.parquet')
    pd.testing.assert_frame_equal(num_encoding, stored_encoding)


def test_jensenshannon_distance():
    # create some sample data
    colname = "age"
    real_col = pd.Series([20, 25, 30, 35, 40])
    fake_col = pd.Series([22, 27, 32, 37, 42])

    # call the function and get the result
    result = jensenshannon_distance(colname, real_col, fake_col)

    # check that the result is a dictionary with the correct keys and values
    assert isinstance(result, dict)
    assert result["col_name"] == colname
    assert result["js_distance"] == 0.2736453208486386 # this is the expected JS distance for these data
