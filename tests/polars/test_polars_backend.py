import pytest
import numpy as np
import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from table_evaluator.backends import (
    PolarsBackend,
)


@pytest.fixture
def sample_data():
    """Create sample test data."""
    np.random.seed(42)
    data = {
        "int_col": [1, 2, 3, 4, 5],
        "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
        "str_col": ["a", "b", "c", "d", "e"],
        "bool_col": [True, False, True, False, True],
        "cat_col": ["cat1", "cat2", "cat1", "cat3", "cat2"],
    }
    return pd.DataFrame(data)


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
class TestPolarsBackend:
    """Test PolarsBackend functionality."""

    def test_load_csv(self, tmp_path, sample_data):
        """Test CSV loading with Polars backend."""
        backend = PolarsBackend(lazy=False)
        csv_path = tmp_path / "test.csv"
        sample_data.to_csv(csv_path, index=False)

        loaded_df = backend.load_csv(csv_path)
        assert isinstance(loaded_df, pl.DataFrame)
        assert loaded_df.shape == sample_data.shape

    def test_basic_operations(self, sample_data):
        """Test basic DataFrame operations."""
        backend = PolarsBackend(lazy=False)
        polars_df = pl.from_pandas(sample_data)

        # Test select_columns
        selected = backend.select_columns(polars_df, ["int_col", "str_col"])
        assert selected.columns == ["int_col", "str_col"]

        # Test get_shape
        shape = backend.get_shape(polars_df)
        assert shape == sample_data.shape

        # Test get_columns
        columns = backend.get_columns(polars_df)
        assert columns == sample_data.columns.tolist()

    def test_conversion(self, sample_data):
        """Test pandas/Polars conversion."""
        backend = PolarsBackend()

        # Test from_pandas
        polars_df = backend.from_pandas(sample_data)
        assert isinstance(polars_df, (pl.DataFrame, pl.LazyFrame))

        # Test to_pandas
        converted_back = backend.to_pandas(polars_df)
        assert isinstance(converted_back, pd.DataFrame)
        pd.testing.assert_frame_equal(sample_data, converted_back)

    # TODO: Add more tests for PolarsBackend methods
