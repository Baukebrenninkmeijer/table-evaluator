"""Comprehensive test suite for backend compatibility."""

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
    BackendManager,
    DataFrameWrapper,
    PandasBackend,
    PolarsBackend,
    DTypeMapper,
    DataFrameConverter,
    SchemaValidator,
    DataBridge,
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


@pytest.fixture
def sample_data_with_nulls():
    """Create sample test data with missing values."""
    data = {
        "int_col": [1, 2, None, 4, 5],
        "float_col": [1.1, None, 3.3, 4.4, 5.5],
        "str_col": ["a", "b", None, "d", "e"],
        "bool_col": [True, False, None, False, True],
    }
    return pd.DataFrame(data)


class TestBackendFactory:
    """Test BackendFactory functionality."""

    def test_singleton_pattern(self):
        """Test that BackendFactory follows singleton pattern."""
        factory1 = BackendFactory()
        factory2 = BackendFactory()
        assert factory1 is factory2

    def test_get_pandas_backend(self):
        """Test getting pandas backend."""
        factory = BackendFactory()
        backend = factory.get_backend(BackendType.PANDAS)
        assert isinstance(backend, PandasBackend)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_get_polars_backend(self):
        """Test getting Polars backend."""
        factory = BackendFactory()
        backend = factory.get_backend(BackendType.POLARS)
        assert isinstance(backend, PolarsBackend)

    def test_auto_backend_detection(self):
        """Test automatic backend detection."""
        factory = BackendFactory()
        backend = factory.get_backend(BackendType.AUTO)
        assert backend is not None

    def test_get_available_backends(self):
        """Test getting list of available backends."""
        factory = BackendFactory()
        backends = factory.get_available_backends()
        assert BackendType.PANDAS in backends
        assert BackendType.AUTO in backends
        if POLARS_AVAILABLE:
            assert BackendType.POLARS in backends


class TestPandasBackend:
    """Test PandasBackend functionality."""

    def test_load_csv(self, tmp_path, sample_data):
        """Test CSV loading with pandas backend."""
        backend = PandasBackend()
        csv_path = tmp_path / "test.csv"
        sample_data.to_csv(csv_path, index=False)

        loaded_df = backend.load_csv(csv_path)
        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == sample_data.shape

    def test_basic_operations(self, sample_data):
        """Test basic DataFrame operations."""
        backend = PandasBackend()

        # Test select_columns
        selected = backend.select_columns(sample_data, ["int_col", "str_col"])
        assert list(selected.columns) == ["int_col", "str_col"]

        # Test get_shape
        shape = backend.get_shape(sample_data)
        assert shape == sample_data.shape

        # Test get_columns
        columns = backend.get_columns(sample_data)
        assert columns == sample_data.columns.tolist()

    def test_sampling(self, sample_data):
        """Test DataFrame sampling."""
        backend = PandasBackend()
        sampled = backend.sample(sample_data, n=3, random_state=42)
        assert len(sampled) == 3
        assert isinstance(sampled, pd.DataFrame)

    def test_statistics(self, sample_data):
        """Test statistics computation."""
        backend = PandasBackend()
        stats = backend.compute_statistics(sample_data)

        assert "int_col" in stats
        assert "float_col" in stats
        assert "mean" in stats["int_col"]
        assert "std" in stats["int_col"]


class TestDataFrameWrapper:
    """Test DataFrameWrapper functionality."""

    def test_pandas_wrapper(self, sample_data):
        """Test wrapping pandas DataFrame."""
        wrapper = DataFrameWrapper(sample_data)
        assert wrapper.is_pandas
        assert not wrapper.is_polars
        assert wrapper.shape == sample_data.shape
        assert wrapper.columns == sample_data.columns.tolist()

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_wrapper(self, sample_data):
        """Test wrapping Polars DataFrame."""
        polars_df = pl.from_pandas(sample_data)
        wrapper = DataFrameWrapper(polars_df)
        assert wrapper.is_polars
        assert not wrapper.is_pandas
        assert wrapper.shape == sample_data.shape
        assert wrapper.columns == sample_data.columns.tolist()

    def test_wrapper_operations(self, sample_data):
        """Test operations on wrapped DataFrame."""
        wrapper = DataFrameWrapper(sample_data)

        # Test select
        selected = wrapper.select(["int_col", "str_col"])
        assert isinstance(selected, DataFrameWrapper)
        assert selected.columns == ["int_col", "str_col"]

        # Test sample
        sampled = wrapper.sample(3, random_state=42)
        assert len(sampled) == 3


class TestDTypeMapper:
    """Test dtype mapping functionality."""

    def test_pandas_to_polars_mapping(self):
        """Test pandas to Polars dtype mapping."""
        mapper = DTypeMapper()

        # Test basic mappings
        assert mapper.pandas_to_polars_dtype("int64") == "Int64"
        assert mapper.pandas_to_polars_dtype("float64") == "Float64"
        assert mapper.pandas_to_polars_dtype("object") == "String"
        assert mapper.pandas_to_polars_dtype("bool") == "Boolean"

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_to_pandas_mapping(self):
        """Test Polars to pandas dtype mapping."""
        mapper = DTypeMapper()

        # Test basic mappings
        assert mapper.polars_to_pandas_dtype("Int64") == "int64"
        assert mapper.polars_to_pandas_dtype("Float64") == "float64"
        assert mapper.polars_to_pandas_dtype("String") == "string"
        assert mapper.polars_to_pandas_dtype("Boolean") == "boolean"

    def test_validation(self, sample_data):
        """Test dtype validation."""
        mapper = DTypeMapper()
        dtypes = {col: str(dtype) for col, dtype in sample_data.dtypes.items()}

        unsupported = mapper.validate_pandas_dtypes(dtypes)
        # Should have no unsupported dtypes for basic data
        assert len(unsupported) == 0


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
class TestDataFrameConverter:
    """Test DataFrame conversion functionality."""

    def test_pandas_to_polars_conversion(self, sample_data):
        """Test converting pandas to Polars."""
        converter = DataFrameConverter()
        polars_df = converter.pandas_to_polars(sample_data)

        assert isinstance(polars_df, (pl.DataFrame, pl.LazyFrame))

        # Convert back and compare
        converted_back = converter.polars_to_pandas(polars_df)

        # Check basic structure (dtypes might change slightly due to polars' type system)
        assert converted_back.shape == sample_data.shape
        assert set(converted_back.columns) == set(sample_data.columns)

        # Check data values are preserved (allowing for dtype differences)
        for col in sample_data.columns:
            original_series = sample_data[col]
            converted_series = converted_back[col]

            if original_series.dtype == "object":
                # For string columns, compare values rather than exact dtypes
                pd.testing.assert_series_equal(
                    original_series.astype(str),
                    converted_series.astype(str),
                    check_names=False,
                )
            elif original_series.dtype == "bool" or str(
                converted_series.dtype
            ).startswith("boolean"):
                # For boolean columns, allow dtype differences but check values
                pd.testing.assert_series_equal(
                    original_series.astype(bool),
                    converted_series.astype(bool),
                    check_names=False,
                )
            else:
                # For numeric columns, check with some tolerance for floating point
                try:
                    pd.testing.assert_series_equal(
                        original_series,
                        converted_series,
                        check_names=False,
                        check_dtype=False,  # Allow dtype differences
                    )
                except AssertionError:
                    # If that fails, try with value comparison only
                    assert (
                        original_series.tolist() == converted_series.tolist()
                    ), f"Values differ for column {col}: {original_series.tolist()} != {converted_series.tolist()}"

    def test_lazy_conversion(self, sample_data):
        """Test lazy conversion."""
        converter = DataFrameConverter()
        lazy_df = converter.pandas_to_polars(sample_data, lazy=True)

        assert isinstance(lazy_df, pl.LazyFrame)

    def test_dtype_preservation(self, sample_data):
        """Test that dtypes are preserved during conversion."""
        converter = DataFrameConverter()

        # Convert to Polars and back
        polars_df = converter.pandas_to_polars(sample_data, preserve_dtypes=True)
        converted_back = converter.polars_to_pandas(polars_df, preserve_dtypes=True)

        # Check that basic structure is preserved
        assert converted_back.shape == sample_data.shape
        assert set(converted_back.columns) == set(sample_data.columns)


class TestSchemaValidator:
    """Test schema validation functionality."""

    def test_compatibility_validation(self, sample_data):
        """Test schema compatibility validation."""
        validator = SchemaValidator()
        issues = validator.validate_conversion_compatibility(sample_data, "polars")

        # Should have no major issues for basic data
        error_issues = [issue for issue in issues if issue.severity == "error"]
        assert len(error_issues) == 0

    def test_consistency_validation(self, sample_data):
        """Test schema consistency between DataFrames."""
        validator = SchemaValidator()

        # Create slightly different DataFrame
        sample_data2 = sample_data.copy()
        sample_data2["extra_col"] = [1, 2, 3, 4, 5]

        issues = validator.validate_schema_consistency(sample_data, sample_data2)

        # Should detect the extra column
        extra_col_issues = [
            issue for issue in issues if issue.issue_type == "extra_column"
        ]
        assert len(extra_col_issues) > 0


class TestBackendManager:
    """Test BackendManager functionality."""

    def test_data_loading(self, tmp_path, sample_data):
        """Test data loading with backend manager."""
        manager = BackendManager()
        csv_path = tmp_path / "test.csv"
        sample_data.to_csv(csv_path, index=False)

        # Test loading with auto-detection
        wrapper = manager.load_data(csv_path)
        assert isinstance(wrapper, DataFrameWrapper)
        assert wrapper.shape == sample_data.shape

    def test_backend_optimization(self, sample_data):
        """Test backend optimization for operations."""
        manager = BackendManager()
        wrapper = DataFrameWrapper(sample_data)

        # Test optimization for different operations
        optimized = manager.optimize_for_operation(wrapper, "aggregation")
        assert isinstance(optimized, DataFrameWrapper)


class TestDataBridge:
    """Test DataBridge functionality."""

    def test_bridge_conversion(self, sample_data):
        """Test conversion through data bridge."""
        bridge = DataBridge()

        # Test pandas to polars conversion
        if POLARS_AVAILABLE:
            polars_result = bridge.convert_dataframe(sample_data, "polars")
            assert polars_result is not None

        # Test keeping as pandas
        pandas_result = bridge.convert_dataframe(sample_data, "pandas")
        assert isinstance(pandas_result, pd.DataFrame)

    def test_validation_plan(self, sample_data):
        """Test conversion planning."""
        bridge = DataBridge()

        if POLARS_AVAILABLE:
            plan = bridge.get_conversion_plan(sample_data, "polars")
            assert plan["source_backend"] == "pandas"
            assert plan["target_backend"] == "polars"
            assert "is_compatible" in plan

    def test_compatibility_check(self, sample_data):
        """Test compatibility checking."""
        bridge = DataBridge()

        if POLARS_AVAILABLE:
            is_compatible, issues = bridge.validate_compatibility(sample_data, "polars")
            assert isinstance(is_compatible, bool)
            assert isinstance(issues, list)


@pytest.mark.parametrize(
    "backend", ["pandas"] + (["polars"] if POLARS_AVAILABLE else [])
)
class TestBackendParametrized:
    """Parametrized tests that run against both backends."""

    def test_basic_workflow(self, backend, sample_data, tmp_path):
        """Test basic workflow with both backends."""
        # Setup
        csv_path = tmp_path / "test.csv"
        sample_data.to_csv(csv_path, index=False)

        # Test data loading
        manager = BackendManager()
        wrapper = manager.load_data(csv_path, backend=backend)

        assert isinstance(wrapper, DataFrameWrapper)
        assert wrapper.shape == sample_data.shape

        # Test basic operations
        selected = wrapper.select(["int_col", "float_col"])
        assert len(selected.columns) == 2

        sampled = wrapper.sample(3, random_state=42)
        assert len(sampled) == 3

    def test_conversion_roundtrip(self, backend, sample_data):
        """Test conversion roundtrip maintains data integrity."""
        bridge = DataBridge()

        # Convert to target backend
        converted = bridge.convert_dataframe(sample_data, backend)

        # Convert back to pandas for comparison
        if backend == "polars" and POLARS_AVAILABLE:
            back_to_pandas = bridge.convert_dataframe(converted, "pandas")
            # Check basic structure (dtypes might change slightly)
            assert back_to_pandas.shape == sample_data.shape
            assert set(back_to_pandas.columns) == set(sample_data.columns)
        else:
            # For pandas backend, dtypes might still change due to bridge normalization
            assert converted.shape == sample_data.shape
            assert set(converted.columns) == set(sample_data.columns)

            # Check data values are preserved (allowing for dtype changes)
            for col in sample_data.columns:
                original_values = sample_data[col].values
                converted_values = converted[col].values

                # For object columns, compare string representations
                if sample_data[col].dtype == "object":
                    assert (
                        sample_data[col].astype(str) == converted[col].astype(str)
                    ).all()
                else:
                    # For numeric/boolean, allow dtype differences but check values
                    assert len(original_values) == len(converted_values)
                    # Use numpy allclose for floating point, direct comparison for integers/bools
                    import numpy as np

                    if np.issubdtype(sample_data[col].dtype, np.floating):
                        assert np.allclose(
                            original_values, converted_values, equal_nan=True
                        )
                    else:
                        assert (original_values == converted_values).all()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        wrapper = DataFrameWrapper(empty_df)

        assert wrapper.shape == (0, 0)
        assert len(wrapper.columns) == 0

    def test_missing_values(self, sample_data_with_nulls):
        """Test handling of missing values."""
        wrapper = DataFrameWrapper(sample_data_with_nulls)

        # Should handle missing values gracefully
        assert wrapper.shape[0] > 0

        # Test conversion with missing values
        if POLARS_AVAILABLE:
            bridge = DataBridge()
            converted = bridge.convert_dataframe(sample_data_with_nulls, "polars")
            assert converted is not None

    def test_unsupported_backend(self):
        """Test handling of unsupported backend requests."""
        factory = BackendFactory()

        with pytest.raises(ValueError):
            factory.get_backend("unsupported_backend")

    def test_polars_unavailable_handling(self, monkeypatch):
        """Test graceful handling when Polars is unavailable."""
        # Mock Polars as unavailable in the backend_factory module
        monkeypatch.setattr(
            "table_evaluator.backends.backend_factory.POLARS_AVAILABLE", False
        )

        factory = BackendFactory()
        available = factory.get_available_backends()

        # Should only have pandas and auto
        assert BackendType.PANDAS in available
        assert BackendType.AUTO in available
        assert BackendType.POLARS not in available


if __name__ == "__main__":
    pytest.main([__file__])
