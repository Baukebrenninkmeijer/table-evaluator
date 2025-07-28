from pathlib import Path

import numpy as np
import pandas as pd

import table_evaluator.metrics as te_metrics
from table_evaluator.data.data_converter import DataConverter
from table_evaluator.metrics.statistical import associations
from table_evaluator.utils import load_data

data_folder = Path('data')
test_data_folder = Path('data/tests')
real, fake = load_data(data_folder / 'real_test_sample.csv', data_folder / 'fake_test_sample.csv')
cat_cols = ['trans_type', 'trans_operation', 'trans_k_symbol']


def test_mape():
    assert te_metrics.mean_absolute_percentage_error(np.array([1]), np.array([2])) == 1.0


def test_mean_absolute_error():
    assert te_metrics.mean_absolute_error(np.array([1, 1]), np.array([2, 2])) == 1.0


def test_euclidean_distance():
    np.testing.assert_almost_equal(te_metrics.euclidean_distance(np.array([0, 0]), np.array([1, 1])), 1.41421356)


def test_rmse():
    assert te_metrics.rmse(np.array([0, 0]), np.array([2, 2])) == 2.0


def test_column_correlation():
    te_metrics.column_correlations(real, fake, cat_cols)


def test_associations():
    """
    Tests that check wether the dython associations are still computed as is expected.
    """
    # load test data
    real_assoc = pd.read_parquet(test_data_folder / 'real_associations.parquet')
    real_assoc_theil = pd.read_parquet(test_data_folder / 'real_associations_theil.parquet')
    fake_assoc = pd.read_parquet(test_data_folder / 'fake_associations.parquet')
    fake_assoc_theil = pd.read_parquet(test_data_folder / 'fake_associations_theil.parquet')

    # Assert equality with saved data
    pd.testing.assert_frame_equal(
        real_assoc,
        associations(real, nominal_columns=cat_cols),
    )
    pd.testing.assert_frame_equal(
        real_assoc_theil,
        associations(real, nominal_columns=cat_cols, nom_nom_assoc='theil'),
    )
    pd.testing.assert_frame_equal(
        fake_assoc,
        associations(fake, nominal_columns=cat_cols),
    )
    pd.testing.assert_frame_equal(
        fake_assoc_theil,
        associations(fake, nominal_columns=cat_cols, nom_nom_assoc='theil'),
    )


def test_numerical_encoding():
    """
    Tests that check wether the dython numerical_encoding are still computed as is expected.
    """

    real_enc, fake_enc, _ = DataConverter.numeric_encoding(real=real, fake=fake, categorical_columns=cat_cols)
    fake_expected = pd.read_parquet(test_data_folder / 'fake_test_sample_numerical_encoded.parquet')
    real_expected = pd.read_parquet(test_data_folder / 'real_test_sample_numerical_encoded.parquet')

    pd.testing.assert_frame_equal(
        fake_enc.sort_index(axis=1).sort_values('account_id'),
        fake_expected.sort_index(axis=1).sort_values('account_id'),
    )
    pd.testing.assert_frame_equal(
        real_enc.sort_index(axis=1).sort_values('account_id'),
        real_expected.sort_index(axis=1).sort_values('account_id'),
    )


def test_jensenshannon_distance():
    # create some sample data
    colname = 'age'
    real_col = pd.Series([20, 25, 30, 35, 40])
    fake_col = pd.Series([22, 27, 32, 37, 42])

    # call the function and get the result
    result = te_metrics.jensenshannon_distance(colname, real_col, fake_col)

    # check that the result is a JensenShannonResult object with the correct values
    assert hasattr(result, 'col_name')
    assert hasattr(result, 'js_distance')
    assert hasattr(result, 'success')
    assert result.col_name == colname
    assert abs(result.js_distance - 0.2736453208486386) < 1e-15  # this is the expected JS distance for these data
    assert result.success is True
