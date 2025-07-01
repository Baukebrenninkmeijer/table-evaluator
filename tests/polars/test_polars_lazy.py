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
class TestPolarsLazy:
    """Test PolarsBackend lazy functionality."""

    def test_lazy_loading(self, tmp_path, sample_data):
        """Test lazy CSV loading."""
        backend = PolarsBackend(lazy=True)
        csv_path = tmp_path / "test.csv"
        sample_data.to_csv(csv_path, index=False)

        loaded_df = backend.load_csv(csv_path)
        assert isinstance(loaded_df, pl.LazyFrame)

    # TODO: Add more tests for lazy evaluation
