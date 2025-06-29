import pandas as pd
import numpy as np
from table_evaluator.utils import _preprocess_data

def test_preprocess_data():
    real_data = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['A', 'B', 'A', 'C', 'B'],
        'col3': [10.1, 11.2, np.nan, 13.4, 14.5]
    })
    fake_data = pd.DataFrame({
        'col1': [6, 7, 8, 9, 10],
        'col2': ['B', 'C', 'A', 'A', 'C'],
        'col3': [15.1, 16.2, 17.3, np.nan, 19.5]
    })

    # Test with default parameters
    preprocessed_real, preprocessed_fake, numerical_cols, categorical_cols = _preprocess_data(real_data.copy(), fake_data.copy())
    assert len(preprocessed_real) == len(preprocessed_fake)
    assert 'col1' in numerical_cols
    assert 'col3' in numerical_cols
    assert 'col2' in categorical_cols
    assert not preprocessed_real.isnull().values.any()
    assert not preprocessed_fake.isnull().values.any()

    # Test with specified n_samples
    preprocessed_real, preprocessed_fake, _, _ = _preprocess_data(real_data.copy(), fake_data.copy(), n_samples=3)
    assert len(preprocessed_real) == 3
    assert len(preprocessed_fake) == 3

    # Test with specified cat_cols
    preprocessed_real, preprocessed_fake, numerical_cols, categorical_cols = _preprocess_data(real_data.copy(), fake_data.copy(), cat_cols=['col1', 'col2'])
    assert 'col3' in numerical_cols
    assert 'col1' in categorical_cols
    assert 'col2' in categorical_cols

    # Test with unique_thresh
    real_data_thresh = pd.DataFrame({
        'col1': [1, 1, 2, 2, 3],
        'col2': [10, 11, 12, 13, 14]
    })
    fake_data_thresh = pd.DataFrame({
        'col1': [1, 1, 2, 2, 3],
        'col2': [10, 11, 12, 13, 14]
    })
    preprocessed_real, preprocessed_fake, numerical_cols, categorical_cols = _preprocess_data(real_data_thresh.copy(), fake_data_thresh.copy(), unique_thresh=2)
    assert 'col1' in numerical_cols # col1 has 3 unique values, so it remains numerical
    assert 'col2' in numerical_cols

    # Test with different column order
    fake_data_reordered = pd.DataFrame({
        'col3': [15.1, 16.2, 17.3, np.nan, 19.5],
        'col1': [6, 7, 8, 9, 10],
        'col2': ['B', 'C', 'A', 'A', 'C']
    })
    preprocessed_real, preprocessed_fake, _, _ = _preprocess_data(real_data.copy(), fake_data_reordered.copy())
    assert preprocessed_real.columns.tolist() == preprocessed_fake.columns.tolist()

    # Test with NaN values and check if they are filled
    real_data_nan = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': ['A', np.nan, 'A', 'C', 'B']
    })
    fake_data_nan = pd.DataFrame({
        'col1': [6, np.nan, 8, 9, 10],
        'col2': [np.nan, 'C', 'A', 'A', 'C']
    })
    preprocessed_real, preprocessed_fake, _, _ = _preprocess_data(real_data_nan.copy(), fake_data_nan.copy())
    assert not preprocessed_real.isnull().values.any()
    assert not preprocessed_fake.isnull().values.any()
