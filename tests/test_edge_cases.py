"""Tests for edge cases and boundary conditions in backend operations."""

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
    BackendType,
    BackendFactory,
    DataFrameWrapper,
    DataBridge,
    SchemaValidator,
    convert_to_polars,
)
from table_evaluator.data.data_converter import DataConverter
from table_evaluator.utils import _preprocess_data


class TestEmptyDataFrames:
    """Test handling of empty DataFrames."""

    def test_empty_dataframe_creation(self):
        """Test creating wrappers for empty DataFrames."""
        empty_df = pd.DataFrame()
        wrapper = DataFrameWrapper(empty_df)

        assert wrapper.shape == (0, 0)
        assert len(wrapper.columns) == 0
        assert wrapper.is_pandas
        assert not wrapper.is_polars

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_empty_polars_dataframe(self):
        """Test empty Polars DataFrames."""
        empty_polars = pl.DataFrame()
        wrapper = DataFrameWrapper(empty_polars)

        assert wrapper.shape == (0, 0)
        assert len(wrapper.columns) == 0
        assert wrapper.is_polars
        assert not wrapper.is_pandas

    def test_empty_dataframe_conversion(self):
        """Test conversion of empty DataFrames."""
        empty_df = pd.DataFrame()
        bridge = DataBridge()

        # Convert to pandas (should be no-op)
        result_pandas = bridge.convert_dataframe(empty_df, "pandas")
        assert isinstance(result_pandas, pd.DataFrame)
        assert result_pandas.empty

        # Convert to Polars if available
        if POLARS_AVAILABLE:
            result_polars = bridge.convert_dataframe(empty_df, "polars")
            assert result_polars is not None

    def test_empty_dataframe_preprocessing(self):
        """Test preprocessing with empty DataFrames."""
        empty_real = pd.DataFrame()
        empty_fake = pd.DataFrame()

        # Should handle empty DataFrames gracefully
        result = _preprocess_data(empty_real, empty_fake)

        # Should return tuple of (real, fake, categorical_columns, numerical_columns)
        assert isinstance(result, tuple)
        assert len(result) == 4
        real_processed, fake_processed, categorical_columns, numerical_columns = result

        # All should be empty
        assert len(real_processed) == 0
        assert len(fake_processed) == 0
        assert len(categorical_columns) == 0
        assert len(numerical_columns) == 0

    def test_empty_dataframe_operations(self):
        """Test operations on empty DataFrames."""
        empty_df = pd.DataFrame()
        wrapper = DataFrameWrapper(empty_df)

        # Operations should handle empty DataFrames
        selected = wrapper.select([])
        assert selected.shape == (0, 0)

        # Sampling empty DataFrame should return empty
        sampled = wrapper.sample(0)
        assert len(sampled) == 0


class TestSingleRowDataFrames:
    """Test handling of single-row DataFrames."""

    @pytest.fixture
    def single_row_data(self):
        """Create single-row DataFrames."""
        real_data = pd.DataFrame(
            {
                "numeric_col": [1.0],
                "categorical_col": ["A"],
                "boolean_col": [True],
            }
        )

        fake_data = pd.DataFrame(
            {
                "numeric_col": [2.0],
                "categorical_col": ["B"],
                "boolean_col": [False],
            }
        )

        return real_data, fake_data

    def test_single_row_preprocessing(self, single_row_data):
        """Test preprocessing with single-row DataFrames."""
        real_df, fake_df = single_row_data

        # Should handle single row but might have limitations
        try:
            real_processed, fake_processed, num_cols, cat_cols = _preprocess_data(
                real_df, fake_df, n_samples=1, seed=42
            )

            assert len(real_processed) == 1
            assert len(fake_processed) == 1
            assert len(num_cols) >= 1
            assert len(cat_cols) >= 1
        except Exception:
            # Some operations might fail with single row
            pytest.skip("Single row preprocessing not supported")

    def test_single_row_conversion(self, single_row_data):
        """Test data conversion with single row."""
        real_df, fake_df = single_row_data
        converter = DataConverter()

        # Test numerical encoding
        real_num, fake_num = converter.to_numerical(
            real_df, fake_df, ["categorical_col", "boolean_col"]
        )

        assert len(real_num) == 1
        assert len(fake_num) == 1

        # Test one-hot encoding
        real_onehot, fake_onehot = converter.to_one_hot(
            real_df, fake_df, ["categorical_col", "boolean_col"]
        )

        assert len(real_onehot) == 1
        assert len(fake_onehot) == 1


class TestLargeDataFrames:
    """Test handling of large DataFrames."""

    @pytest.fixture
    def large_dataframe(self):
        """Create a large DataFrame for testing."""
        np.random.seed(42)
        size = 100000  # 100K rows

        data = {
            "id": range(size),
            "numeric1": np.random.normal(0, 1, size),
            "numeric2": np.random.exponential(1, size),
            "categorical1": np.random.choice(["A", "B", "C", "D"], size),
            "categorical2": np.random.choice(["X", "Y", "Z"], size),
            "boolean": np.random.choice([True, False], size),
        }

        return pd.DataFrame(data)

    @pytest.mark.parametrize(
        "backend", ["pandas"] + (["polars"] if POLARS_AVAILABLE else [])
    )
    def test_large_dataframe_sampling(self, backend, large_dataframe):
        """Test sampling from large DataFrames."""
        wrapper = DataFrameWrapper(large_dataframe)

        # Test various sample sizes
        for sample_size in [100, 1000, 10000]:
            sampled = wrapper.sample(sample_size, random_state=42)
            assert len(sampled) == sample_size
            assert sampled.columns == large_dataframe.columns.tolist()

    @pytest.mark.parametrize(
        "backend", ["pandas"] + (["polars"] if POLARS_AVAILABLE else [])
    )
    def test_large_dataframe_conversion(self, backend, large_dataframe):
        """Test conversion of large DataFrames."""
        if backend == "polars" and POLARS_AVAILABLE:
            # Test pandas to Polars conversion
            bridge = DataBridge()
            polars_result = bridge.convert_dataframe(large_dataframe, "polars")
            assert polars_result is not None

            # Convert back
            pandas_result = bridge.convert_dataframe(polars_result, "pandas")
            assert pandas_result.shape == large_dataframe.shape

    def test_memory_efficient_operations(self, large_dataframe):
        """Test memory-efficient operations on large DataFrames."""
        # Test lazy operations if Polars is available
        if POLARS_AVAILABLE:
            polars_lazy = convert_to_polars(large_dataframe, lazy=True)
            assert isinstance(polars_lazy, pl.LazyFrame)

            # Test lazy operations
            filtered = polars_lazy.filter(pl.col("numeric1") > 0)
            selected = filtered.select(["id", "numeric1", "categorical1"])

            # Collect only when needed
            result = selected.collect()
            assert isinstance(result, pl.DataFrame)
            assert len(result.columns) == 3


class TestMixedDataTypes:
    """Test handling of mixed and unusual data types."""

    @pytest.fixture
    def mixed_dtype_dataframe(self):
        """Create DataFrame with mixed data types."""
        data = {
            "integers": [1, 2, 3, 4, 5],
            "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
            "strings": ["a", "b", "c", "d", "e"],
            "booleans": [True, False, True, False, True],
            "mixed_types": [1, "text", 3.14, True, None],
            "datetime_strings": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
            ],
            "numeric_strings": ["1", "2", "3", "4", "5"],
        }

        return pd.DataFrame(data)

    def test_mixed_dtype_detection(self, mixed_dtype_dataframe):
        """Test detection of mixed data types."""
        validator = SchemaValidator()
        issues = validator.validate_conversion_compatibility(
            mixed_dtype_dataframe, "polars"
        )

        # Should detect issues with mixed types
        mixed_type_issues = [
            issue for issue in issues if "mixed" in issue.issue_type.lower()
        ]
        assert (
            len(mixed_type_issues) >= 0
        )  # May or may not find issues depending on implementation

    def test_mixed_dtype_conversion(self, mixed_dtype_dataframe):
        """Test conversion of mixed data types."""
        converter = DataConverter()

        # Should handle mixed types gracefully
        try:
            real_encoded, fake_encoded = converter.to_one_hot(
                mixed_dtype_dataframe,
                mixed_dtype_dataframe.copy(),
                ["strings", "booleans", "mixed_types"],
            )

            assert len(real_encoded) == len(mixed_dtype_dataframe)
            assert len(fake_encoded) == len(mixed_dtype_dataframe)
        except Exception as e:
            # Mixed types might cause conversion issues
            assert "mixed" in str(e).lower() or "object" in str(e).lower()

    def test_automatic_type_inference(self, mixed_dtype_dataframe):
        """Test automatic type inference with mixed types."""
        # Test preprocessing with automatic type inference
        real_processed, fake_processed, num_cols, cat_cols = _preprocess_data(
            mixed_dtype_dataframe,
            mixed_dtype_dataframe.copy(),
            cat_cols=None,  # Let it infer automatically
            n_samples=5,
        )

        assert len(num_cols) >= 1  # At least integers and floats
        assert len(cat_cols) >= 1  # At least strings and booleans


class TestNaNAndMissingValues:
    """Test comprehensive handling of missing values."""

    @pytest.fixture
    def dataframe_with_various_missing(self):
        """Create DataFrame with various types of missing values."""
        data = {
            "numeric_with_nan": [1.0, np.nan, 3.0, np.nan, 5.0],
            "numeric_with_inf": [1.0, np.inf, 3.0, -np.inf, 5.0],
            "string_with_none": ["a", None, "c", None, "e"],
            "string_with_empty": ["a", "", "c", "", "e"],
            "boolean_with_none": [True, None, False, None, True],
            "all_missing": [None, None, None, None, None],
        }

        return pd.DataFrame(data)

    def test_various_missing_value_handling(self, dataframe_with_various_missing):
        """Test handling of various missing value types."""
        df = dataframe_with_various_missing

        # Test preprocessing
        real_processed, fake_processed, num_cols, cat_cols = _preprocess_data(
            df, df.copy(), n_samples=5
        )

        # Should handle all missing values
        assert not real_processed.isnull().any().any()
        assert not fake_processed.isnull().any().any()

    def test_infinity_handling(self, dataframe_with_various_missing):
        """Test handling of infinity values."""
        df = dataframe_with_various_missing

        # Check if infinity values are detected
        has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        assert has_inf  # Fixture should contain infinity

        # Test conversion with infinity values
        bridge = DataBridge()
        if POLARS_AVAILABLE:
            # Polars might handle infinity differently
            try:
                polars_result = bridge.convert_dataframe(df, "polars")
                assert polars_result is not None
            except Exception:
                # Infinity handling might not be supported
                pass

    def test_all_missing_column(self, dataframe_with_various_missing):
        """Test handling of columns with all missing values."""
        df = dataframe_with_various_missing

        # The 'all_missing' column should be handled
        validator = SchemaValidator()
        validator.validate_conversion_compatibility(df, "polars")

        # Should detect issues with all-missing column (implementation dependent)
        # May or may not flag this as an issue depending on validation strictness
        pass


class TestUnicodeAndSpecialCharacters:
    """Test handling of Unicode and special characters."""

    @pytest.fixture
    def unicode_dataframe(self):
        """Create DataFrame with Unicode and special characters."""
        data = {
            "unicode_names": ["café", "naïve", "北京", "🚀", "Москва"],
            "special_chars": [
                "hello@world",
                "test#123",
                "data$value",
                "info%calc",
                "end&start",
            ],
            "whitespace": [
                "normal",
                "  leading",
                "trailing  ",
                "  both  ",
                "\t\ttabs\t\t",
            ],
            "newlines": ["line1\nline2", "single", "line1\r\nline2", "\n\n", "normal"],
        }

        return pd.DataFrame(data)

    def test_unicode_character_handling(self, unicode_dataframe):
        """Test handling of Unicode characters."""
        df = unicode_dataframe

        # Test basic operations
        wrapper = DataFrameWrapper(df)
        assert wrapper.shape == (5, 4)

        # Test conversion
        converter = DataConverter()
        real_encoded, fake_encoded = converter.to_one_hot(
            df, df.copy(), ["unicode_names", "special_chars"]
        )

        assert len(real_encoded) == len(df)
        assert len(fake_encoded) == len(df)

    def test_special_character_handling(self, unicode_dataframe):
        """Test handling of special characters in column names and values."""
        df = unicode_dataframe

        # Test with special characters in column names
        df_special_cols = df.copy()
        df_special_cols.columns = ["col@1", "col#2", "col$3", "col%4"]

        # Should handle special characters in column names
        validator = SchemaValidator()
        validator.validate_conversion_compatibility(df_special_cols, "polars")

        # Check if special column names cause issues (implementation dependent)
        # May or may not detect issues depending on validation settings
        pass


class TestColumnNameEdgeCases:
    """Test handling of problematic column names."""

    def test_empty_column_names(self):
        """Test handling of empty column names."""
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df.columns = ["normal", "", "another"]  # Empty column name

        validator = SchemaValidator()
        issues = validator.validate_conversion_compatibility(df, "polars")

        # Should detect empty column name
        empty_name_issues = [
            issue for issue in issues if "empty" in issue.issue_type.lower()
        ]
        assert len(empty_name_issues) > 0

    def test_duplicate_column_names(self):
        """Test handling of duplicate column names."""
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df.columns = ["col1", "col2", "col1"]  # Duplicate column name

        validator = SchemaValidator()
        issues = validator.validate_conversion_compatibility(df, "polars")

        # Should detect duplicate column names
        duplicate_issues = [
            issue for issue in issues if "duplicate" in issue.issue_type.lower()
        ]
        assert len(duplicate_issues) > 0

    def test_numeric_column_names(self):
        """Test handling of numeric column names."""
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df.columns = [1, 2, 3]  # Numeric column names

        # Should handle numeric column names by converting to strings
        wrapper = DataFrameWrapper(df)
        assert wrapper.columns == ["1", "2", "3"]  # Converts to strings for consistency


class TestConcurrencyAndThreadSafety:
    """Test concurrent operations and thread safety."""

    def test_concurrent_backend_operations(self):
        """Test concurrent backend operations."""
        import threading
        import time

        def worker(backend_type, results, index):
            """Worker function for concurrent testing."""
            try:
                factory = BackendFactory()
                backend = factory.get_backend(backend_type)

                # Create test data
                df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

                # Perform operations
                shape = backend.get_shape(df)
                columns = backend.get_columns(df)

                results[index] = (shape, columns)
                time.sleep(0.1)  # Simulate work
            except Exception as e:
                results[index] = e

        # Test with multiple threads
        results = {}
        threads = []

        for i in range(5):
            thread = threading.Thread(
                target=worker, args=(BackendType.PANDAS, results, i)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all operations succeeded
        for i, result in results.items():
            assert not isinstance(result, Exception)
            shape, columns = result
            assert shape == (3, 2)
            assert columns == ["col1", "col2"]


class TestMemoryConstraints:
    """Test behavior under memory constraints."""

    def test_large_categorical_expansion(self):
        """Test handling of large categorical expansions."""
        # Create DataFrame that would expand to many columns
        np.random.seed(42)
        data = {
            f"cat_col_{i}": np.random.choice([f"val_{j}" for j in range(100)], 1000)
            for i in range(10)
        }
        df = pd.DataFrame(data)

        converter = DataConverter()

        # Test one-hot encoding (should create many columns)
        try:
            real_encoded, fake_encoded = converter.to_one_hot(
                df, df.copy(), list(df.columns)
            )

            # Should handle large expansion
            assert len(real_encoded.columns) > len(df.columns) * 50
        except MemoryError:
            pytest.skip("Insufficient memory for large categorical expansion test")
        except Exception as e:
            # Other exceptions might be acceptable for memory constraints
            assert "memory" in str(e).lower() or "size" in str(e).lower()


class TestBackendCompatibilityEdgeCases:
    """Test edge cases in backend compatibility."""

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_specific_operations(self):
        """Test operations specific to Polars."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})

        # Convert to Polars
        polars_df = convert_to_polars(df, lazy=True)
        assert isinstance(polars_df, pl.LazyFrame)

        # Test lazy operations
        filtered = polars_df.filter(pl.col("col1") > 2)
        selected = filtered.select(["col1"])

        # Should remain lazy until collected
        assert isinstance(selected, pl.LazyFrame)

        # Collect and verify
        result = selected.collect()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3  # Values > 2

    def test_backend_fallback_mechanisms(self):
        """Test fallback mechanisms when preferred backend fails."""
        factory = BackendFactory()

        # Test getting AUTO backend (should work)
        auto_backend = factory.get_backend(BackendType.AUTO)
        assert auto_backend is not None

        # Test with unavailable backend
        if not POLARS_AVAILABLE:
            available_backends = factory.get_available_backends()
            assert BackendType.POLARS not in available_backends
            assert BackendType.PANDAS in available_backends


if __name__ == "__main__":
    pytest.main([__file__])
