"""Integration tests for full evaluation workflows with both backends."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from table_evaluator.backends import (
    BackendManager,
    DataFrameWrapper,
    DataBridge,
)
from table_evaluator.utils import load_data, _preprocess_data
from table_evaluator.data.data_converter import DataConverter


@pytest.fixture
def sample_datasets():
    """Create sample real and fake datasets for testing."""
    np.random.seed(42)

    # Create real dataset
    real_data = {
        "numeric_col1": np.random.normal(0, 1, 1000),
        "numeric_col2": np.random.exponential(2, 1000),
        "categorical_col1": np.random.choice(["A", "B", "C"], 1000),
        "categorical_col2": np.random.choice(["X", "Y", "Z", "W"], 1000),
        "boolean_col": np.random.choice([True, False], 1000),
        "mixed_col": np.random.choice(["low", "medium", "high"], 1000),
    }
    real_df = pd.DataFrame(real_data)

    # Create synthetic dataset with similar distribution
    fake_data = {
        "numeric_col1": np.random.normal(0.1, 1.1, 950),  # Slightly different
        "numeric_col2": np.random.exponential(2.1, 950),
        "categorical_col1": np.random.choice(["A", "B", "C"], 950),
        "categorical_col2": np.random.choice(["X", "Y", "Z", "W"], 950),
        "boolean_col": np.random.choice([True, False], 950),
        "mixed_col": np.random.choice(["low", "medium", "high"], 950),
    }
    fake_df = pd.DataFrame(fake_data)

    return real_df, fake_df


@pytest.fixture
def temp_csv_files(sample_datasets):
    """Create temporary CSV files for testing data loading."""
    real_df, fake_df = sample_datasets

    with tempfile.TemporaryDirectory() as temp_dir:
        real_path = Path(temp_dir) / "real_data.csv"
        fake_path = Path(temp_dir) / "fake_data.csv"

        real_df.to_csv(real_path, index=False)
        fake_df.to_csv(fake_path, index=False)

        yield str(real_path), str(fake_path)


@pytest.mark.parametrize(
    "backend", ["pandas"] + (["polars"] if POLARS_AVAILABLE else [])
)
class TestFullEvaluationWorkflow:
    """Test complete evaluation workflows with both backends."""

    def test_data_loading_workflow(self, backend, temp_csv_files):
        """Test complete data loading workflow."""
        real_path, fake_path = temp_csv_files

        # Load data with specified backend
        real, fake = load_data(
            real_path, fake_path, backend=backend, auto_detect_backend=True
        )

        # Verify data is loaded correctly
        assert len(real) > 0
        assert len(fake) > 0
        assert list(real.columns) == list(fake.columns)

        # Check backend-specific types
        if backend == "polars" and POLARS_AVAILABLE:
            assert isinstance(real, DataFrameWrapper)
            assert isinstance(fake, DataFrameWrapper)
        else:
            # Should fallback to pandas for compatibility
            assert isinstance(real, (pd.DataFrame, DataFrameWrapper))
            assert isinstance(fake, (pd.DataFrame, DataFrameWrapper))

    def test_preprocessing_workflow(self, backend, sample_datasets):
        """Test complete preprocessing workflow."""
        real_df, fake_df = sample_datasets

        # Convert to appropriate format for backend
        if backend == "polars" and POLARS_AVAILABLE:
            bridge = DataBridge()
            real_input = bridge.convert_dataframe(real_df, "polars")
            fake_input = bridge.convert_dataframe(fake_df, "polars")
            real_wrapped = DataFrameWrapper(real_input)
            fake_wrapped = DataFrameWrapper(fake_input)
        else:
            real_wrapped = real_df
            fake_wrapped = fake_df

        # Run preprocessing
        real_processed, fake_processed, num_cols, cat_cols = _preprocess_data(
            real_wrapped, fake_wrapped, backend=backend, n_samples=500, seed=42
        )

        # Verify preprocessing results
        assert len(real_processed) == len(fake_processed) == 500
        assert len(num_cols) > 0
        assert len(cat_cols) > 0

        # Verify column classification
        expected_num_cols = ["numeric_col1", "numeric_col2"]
        expected_cat_cols = [
            "categorical_col1",
            "categorical_col2",
            "boolean_col",
            "mixed_col",
        ]

        assert all(col in num_cols for col in expected_num_cols)
        assert all(col in cat_cols for col in expected_cat_cols)

        # Check for missing values handling
        # Convert to pandas for isnull check if needed
        if hasattr(real_processed, "to_pandas"):
            real_check = real_processed.to_pandas()
            fake_check = fake_processed.to_pandas()
        else:
            real_check = real_processed
            fake_check = fake_processed

        assert not real_check.isnull().any().any()
        assert not fake_check.isnull().any().any()

    def test_data_conversion_workflow(self, backend, sample_datasets):
        """Test data conversion workflows."""
        real_df, fake_df = sample_datasets

        # Initialize converter with backend preference
        converter = DataConverter(backend=backend)

        # Test categorical encoding
        cat_cols = ["categorical_col1", "categorical_col2", "boolean_col"]

        # Test numerical encoding
        real_num, fake_num = converter.to_numerical(
            real_df, fake_df, cat_cols, backend=backend
        )

        assert len(real_num) == len(real_df)
        assert len(fake_num) == len(fake_df)
        assert list(real_num.columns) == list(fake_num.columns)

        # Test one-hot encoding
        real_onehot, fake_onehot = converter.to_one_hot(
            real_df, fake_df, cat_cols, backend=backend
        )

        assert len(real_onehot.columns) > len(
            real_df.columns
        )  # More columns due to one-hot
        assert len(real_onehot) == len(real_df)
        assert list(real_onehot.columns) == list(fake_onehot.columns)

    def test_backend_optimization_workflow(self, backend, sample_datasets):
        """Test backend optimization for different operations."""
        real_df, fake_df = sample_datasets

        manager = BackendManager()

        # Test optimization for different operation types
        real_wrapped = DataFrameWrapper(real_df)
        fake_wrapped = DataFrameWrapper(fake_df)

        # Test aggregation optimization
        real_optimized = manager.optimize_for_operation(real_wrapped, "aggregation")
        fake_optimized = manager.optimize_for_operation(fake_wrapped, "aggregation")

        assert isinstance(real_optimized, DataFrameWrapper)
        assert isinstance(fake_optimized, DataFrameWrapper)

        # Verify data integrity after optimization
        assert real_optimized.shape == real_wrapped.shape
        assert fake_optimized.shape == fake_wrapped.shape

    def test_mixed_backend_workflow(self, backend, sample_datasets):
        """Test workflows that mix different backends."""
        real_df, fake_df = sample_datasets
        bridge = DataBridge()

        # Start with pandas
        real_pandas = real_df
        fake_pandas = fake_df

        # Convert to target backend
        if backend == "polars" and POLARS_AVAILABLE:
            real_target = bridge.convert_dataframe(real_pandas, "polars")
            fake_target = bridge.convert_dataframe(fake_pandas, "polars")

            # Verify conversion worked
            assert real_target is not None
            assert fake_target is not None

            # Convert back to pandas
            real_back = bridge.convert_dataframe(real_target, "pandas")
            fake_back = bridge.convert_dataframe(fake_target, "pandas")

            # Verify roundtrip preserves basic structure
            assert real_back.shape == real_pandas.shape
            assert fake_back.shape == fake_pandas.shape
            assert list(real_back.columns) == list(real_pandas.columns)
            assert list(fake_back.columns) == list(fake_pandas.columns)
        else:
            # For pandas backend, conversion should be no-op
            real_target = bridge.convert_dataframe(real_pandas, "pandas")
            fake_target = bridge.convert_dataframe(fake_pandas, "pandas")

            pd.testing.assert_frame_equal(real_target, real_pandas)
            pd.testing.assert_frame_equal(fake_target, fake_pandas)


class TestWorkflowWithMissingValues:
    """Test workflows with various missing value patterns."""

    @pytest.fixture
    def data_with_missing(self):
        """Create datasets with missing values."""
        np.random.seed(42)

        real_data = {
            "numeric_col": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0],
            "categorical_col": ["A", "B", None, "A", "C", "B", None],
            "boolean_col": [True, False, None, True, False, True, False],
            "mixed_col": [1, "text", None, 3.14, "more_text", None, 42],
        }
        real_df = pd.DataFrame(real_data)

        fake_data = {
            "numeric_col": [1.1, np.nan, 3.1, 4.1, np.nan, 6.1, 7.1],
            "categorical_col": ["A", None, "C", "A", "C", None, "B"],
            "boolean_col": [True, False, True, None, False, True, None],
            "mixed_col": [1, None, "text", 3.14, None, "more_text", 42],
        }
        fake_df = pd.DataFrame(fake_data)

        return real_df, fake_df

    @pytest.mark.parametrize(
        "backend", ["pandas"] + (["polars"] if POLARS_AVAILABLE else [])
    )
    def test_missing_value_workflow(self, backend, data_with_missing):
        """Test complete workflow with missing values."""
        real_df, fake_df = data_with_missing

        # Test preprocessing with missing values
        real_processed, fake_processed, num_cols, cat_cols = _preprocess_data(
            real_df,
            fake_df,
            backend=backend,
            n_samples=None,  # Use all data
            seed=42,
        )

        # Verify missing values are handled
        assert not real_processed.isnull().any().any()
        assert not fake_processed.isnull().any().any()

        # Test data conversion with missing values
        converter = DataConverter(backend=backend)

        real_num, fake_num = converter.to_numerical(
            real_df, fake_df, cat_cols, backend=backend
        )

        # Should handle missing values gracefully
        assert len(real_num) == len(real_df)
        assert len(fake_num) == len(fake_df)


class TestLargeDatasetWorkflow:
    """Test workflows with larger datasets to verify scalability."""

    @pytest.fixture
    def large_datasets(self):
        """Create larger datasets for scalability testing."""
        np.random.seed(42)
        size = 10000

        real_data = {
            "id": range(size),
            "numeric1": np.random.normal(0, 1, size),
            "numeric2": np.random.exponential(1, size),
            "numeric3": np.random.uniform(0, 100, size),
            "cat1": np.random.choice(["A", "B", "C", "D", "E"], size),
            "cat2": np.random.choice(["X", "Y", "Z"], size),
            "bool1": np.random.choice([True, False], size),
            "bool2": np.random.choice([True, False], size),
        }
        real_df = pd.DataFrame(real_data)

        fake_data = {
            "id": range(size),
            "numeric1": np.random.normal(0.1, 1.1, size),
            "numeric2": np.random.exponential(1.1, size),
            "numeric3": np.random.uniform(0, 105, size),
            "cat1": np.random.choice(["A", "B", "C", "D", "E"], size),
            "cat2": np.random.choice(["X", "Y", "Z"], size),
            "bool1": np.random.choice([True, False], size),
            "bool2": np.random.choice([True, False], size),
        }
        fake_df = pd.DataFrame(fake_data)

        return real_df, fake_df

    @pytest.mark.parametrize(
        "backend", ["pandas"] + (["polars"] if POLARS_AVAILABLE else [])
    )
    def test_large_dataset_workflow(self, backend, large_datasets):
        """Test workflow with larger datasets."""
        real_df, fake_df = large_datasets

        # Test preprocessing with larger dataset
        real_processed, fake_processed, num_cols, cat_cols = _preprocess_data(
            real_df,
            fake_df,
            backend=backend,
            n_samples=5000,  # Sample subset
            seed=42,
        )

        assert len(real_processed) == len(fake_processed) == 5000
        assert len(num_cols) >= 3  # numeric1, numeric2, numeric3
        assert len(cat_cols) >= 4  # cat1, cat2, bool1, bool2

        # Test data conversion
        converter = DataConverter(backend=backend)

        real_encoded, fake_encoded = converter.to_one_hot(
            real_processed, fake_processed, cat_cols, backend=backend
        )

        # Should handle large dataset efficiently
        assert len(real_encoded) == 5000
        assert len(fake_encoded) == 5000
        assert len(real_encoded.columns) > len(real_processed.columns)


class TestErrorHandlingWorkflows:
    """Test error handling in various workflow scenarios."""

    def test_incompatible_columns_workflow(self):
        """Test workflow with incompatible column structures."""
        real_df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["A", "B", "C"], "col3": [True, False, True]}
        )

        fake_df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["A", "B", "C"],
                "col4": [1.1, 2.2, 3.3],  # Different column name
            }
        )

        # Should handle column mismatch gracefully
        with pytest.raises(
            ValueError, match="Columns in real and fake dataframe are not the same"
        ):
            _preprocess_data(real_df, fake_df, n_samples=3)

    def test_insufficient_samples_workflow(self):
        """Test workflow with insufficient samples."""
        real_df = pd.DataFrame(
            {
                "col1": [1, 2],  # Only 2 rows
                "col2": ["A", "B"],
            }
        )

        fake_df = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})

        # Should raise error when requesting more samples than available
        with pytest.raises(Exception, match="Make sure n_samples < len"):
            _preprocess_data(real_df, fake_df, n_samples=5)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_backend_conversion_error_handling(self):
        """Test error handling in backend conversions."""
        # Create DataFrame with unsupported dtype for conversion
        df_with_complex = pd.DataFrame(
            {"complex_col": [1 + 2j, 3 + 4j, 5 + 6j], "normal_col": [1, 2, 3]}
        )

        bridge = DataBridge(strict_validation=True)

        # Should handle conversion gracefully or raise appropriate error
        try:
            result = bridge.convert_dataframe(df_with_complex, "polars")
            # If it succeeds, verify it's not None
            assert result is not None
        except Exception as e:
            # If it fails, should be a meaningful error
            assert "complex" in str(e).lower() or "unsupported" in str(e).lower()


class TestPerformanceCharacteristics:
    """Test performance characteristics of workflows."""

    @pytest.fixture
    def performance_datasets(self):
        """Create datasets for performance testing."""
        np.random.seed(42)
        size = 1000

        # Create datasets with different characteristics
        wide_data = {f"col_{i}": np.random.normal(0, 1, size) for i in range(50)}
        wide_df = pd.DataFrame(wide_data)

        categorical_data = {
            f"cat_{i}": np.random.choice([f"val_{j}" for j in range(10)], size)
            for i in range(10)
        }
        categorical_df = pd.DataFrame(categorical_data)

        return wide_df, categorical_df

    @pytest.mark.parametrize(
        "backend", ["pandas"] + (["polars"] if POLARS_AVAILABLE else [])
    )
    def test_wide_dataframe_workflow(self, backend, performance_datasets):
        """Test workflow with wide DataFrames (many columns)."""
        wide_df, _ = performance_datasets

        # Create fake version
        fake_wide = wide_df + np.random.normal(0, 0.1, wide_df.shape)

        # Test preprocessing
        real_processed, fake_processed, num_cols, cat_cols = _preprocess_data(
            wide_df, fake_wide, backend=backend, n_samples=500, seed=42
        )

        assert len(real_processed.columns) == len(wide_df.columns)
        assert len(fake_processed.columns) == len(fake_wide.columns)
        assert len(num_cols) == len(wide_df.columns)  # All numeric
        assert len(cat_cols) == 0

    @pytest.mark.parametrize(
        "backend", ["pandas"] + (["polars"] if POLARS_AVAILABLE else [])
    )
    def test_categorical_heavy_workflow(self, backend, performance_datasets):
        """Test workflow with many categorical columns."""
        _, categorical_df = performance_datasets

        # Create fake version with slightly different distribution
        fake_categorical = categorical_df.copy()
        for col in fake_categorical.columns:
            # Randomly change some values
            mask = np.random.random(len(fake_categorical)) < 0.1
            fake_categorical.loc[mask, col] = np.random.choice(
                [f"val_{j}" for j in range(10)], mask.sum()
            )

        # Test preprocessing
        real_processed, fake_processed, num_cols, cat_cols = _preprocess_data(
            categorical_df, fake_categorical, backend=backend, n_samples=500, seed=42
        )

        assert len(cat_cols) == len(categorical_df.columns)  # All categorical
        assert len(num_cols) == 0

        # Test one-hot encoding
        converter = DataConverter(backend=backend)
        real_encoded, fake_encoded = converter.to_one_hot(
            real_processed, fake_processed, cat_cols, backend=backend
        )

        # Should create many more columns due to one-hot encoding
        assert len(real_encoded.columns) > len(categorical_df.columns) * 5


if __name__ == "__main__":
    pytest.main([__file__])
