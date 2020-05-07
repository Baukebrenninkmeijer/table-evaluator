import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from table_evaluator.helpers import *
from dython.nominal import compute_associations, numerical_encoding
from pathlib import Path

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
    # load test data
    real_assoc = pd.read_csv(test_data_folder/'real_associations.csv', index_col='Unnamed: 0')
    real_assoc_theil = pd.read_csv(test_data_folder/'real_associations_theil.csv', index_col='Unnamed: 0')
    fake_assoc = pd.read_csv(test_data_folder/'fake_associations.csv', index_col='Unnamed: 0')
    fake_assoc_theil = pd.read_csv(test_data_folder/'fake_associations_theil.csv', index_col='Unnamed: 0')

    # Assert equality with saved data
    pd.testing.assert_frame_equal(real_assoc, compute_associations(real, nominal_columns=cat_cols))
    pd.testing.assert_frame_equal(real_assoc_theil, compute_associations(real, nominal_columns=cat_cols, theil_u=True))
    pd.testing.assert_frame_equal(fake_assoc, compute_associations(fake, nominal_columns=cat_cols))
    pd.testing.assert_frame_equal(fake_assoc_theil, compute_associations(fake, nominal_columns=cat_cols, theil_u=True))


def test_numerical_encoding():
    num_encoding = numerical_encoding(real, nominal_columns=cat_cols)
    uint_cols = num_encoding.select_dtypes(include=['uint8']).columns.tolist()
    num_encoding[uint_cols] = num_encoding[uint_cols].astype('int64')
    stored_encoding = pd.read_csv(test_data_folder/'real_test_sample_numerical_encoded.csv')
    pd.testing.assert_frame_equal(num_encoding, stored_encoding)

    num_encoding = numerical_encoding(fake, nominal_columns=cat_cols)
    uint_cols = num_encoding.select_dtypes(include=['uint8']).columns.tolist()
    num_encoding[uint_cols] = num_encoding[uint_cols].astype('int64')
    stored_encoding = pd.read_csv(test_data_folder/'fake_test_sample_numerical_encoded.csv')
    pd.testing.assert_frame_equal(num_encoding, stored_encoding)
