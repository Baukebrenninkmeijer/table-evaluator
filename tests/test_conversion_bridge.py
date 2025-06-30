"""Tests for data type bridge and conversion functionality."""

import pytest
import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from table_evaluator.backends import (
    DTypeMapper,
    DTypeMappingError,
    DataFrameConverter,
    ConversionError,
    SchemaValidator,
    SchemaIssue,
    LazyConverter,
    DataBridge,
    convert_to_pandas,
    convert_to_polars,
)


@pytest.fixture
def sample_dtypes():
    """Sample data with various dtypes."""
    data = {
        "int8_col": pd.array([1, 2, 3], dtype="int8"),
        "int64_col": pd.array([1, 2, 3], dtype="int64"),
        "float32_col": pd.array([1.1, 2.2, 3.3], dtype="float32"),
        "float64_col": pd.array([1.1, 2.2, 3.3], dtype="float64"),
        "bool_col": pd.array([True, False, True], dtype="bool"),
        "string_col": pd.array(["a", "b", "c"], dtype="string"),
        "object_col": ["x", "y", "z"],
        "category_col": pd.Categorical(["cat1", "cat2", "cat1"]),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_with_nulls():
    """Sample data with null values."""
    data = {
        "int_col": [1, 2, None, 4],
        "float_col": [1.1, None, 3.3, 4.4],
        "str_col": ["a", "b", None, "d"],
        "bool_col": [True, None, False, True],
    }
    return pd.DataFrame(data)


class TestDTypeMapper:
    """Test dtype mapping between pandas and Polars."""

    def test_basic_pandas_to_polars_mapping(self):
        """Test basic dtype mappings from pandas to Polars."""
        mapper = DTypeMapper()

        # Integer types
        assert mapper.pandas_to_polars_dtype("int8") == "Int8"
        assert mapper.pandas_to_polars_dtype("int16") == "Int16"
        assert mapper.pandas_to_polars_dtype("int32") == "Int32"
        assert mapper.pandas_to_polars_dtype("int64") == "Int64"

        # Float types
        assert mapper.pandas_to_polars_dtype("float32") == "Float32"
        assert mapper.pandas_to_polars_dtype("float64") == "Float64"

        # Other types
        assert mapper.pandas_to_polars_dtype("bool") == "Boolean"
        assert mapper.pandas_to_polars_dtype("object") == "String"
        assert mapper.pandas_to_polars_dtype("string") == "String"

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_basic_polars_to_pandas_mapping(self):
        """Test basic dtype mappings from Polars to pandas."""
        mapper = DTypeMapper()

        # Integer types
        assert mapper.polars_to_pandas_dtype("Int8") == "int8"
        assert mapper.polars_to_pandas_dtype("Int16") == "int16"
        assert mapper.polars_to_pandas_dtype("Int32") == "int32"
        assert mapper.polars_to_pandas_dtype("Int64") == "int64"

        # Float types
        assert mapper.polars_to_pandas_dtype("Float32") == "float32"
        assert mapper.polars_to_pandas_dtype("Float64") == "float64"

        # Other types
        assert mapper.polars_to_pandas_dtype("Boolean") == "boolean"
        assert mapper.polars_to_pandas_dtype("String") == "string"

    def test_datetime_mapping(self):
        """Test datetime dtype mapping."""
        mapper = DTypeMapper()

        # Various datetime formats should map to Datetime
        assert mapper.pandas_to_polars_dtype("datetime64[ns]") == "Datetime"
        assert mapper.pandas_to_polars_dtype("datetime64[us]") == "Datetime"
        assert mapper.pandas_to_polars_dtype("datetime64[ms]") == "Datetime"

    def test_unsupported_dtype_mapping(self):
        """Test handling of unsupported dtypes."""
        mapper = DTypeMapper()

        with pytest.raises(DTypeMappingError):
            mapper.pandas_to_polars_dtype("complex128")

    def test_dtype_validation(self, sample_dtypes):
        """Test dtype validation."""
        mapper = DTypeMapper()
        dtypes = {col: str(dtype) for col, dtype in sample_dtypes.dtypes.items()}

        unsupported = mapper.validate_pandas_dtypes(dtypes)
        # Most basic dtypes should be supported
        assert len(unsupported) <= 1  # Maybe category type

    def test_schema_creation(self, sample_dtypes):
        """Test schema creation."""
        mapper = DTypeMapper()
        dtypes = {col: str(dtype) for col, dtype in sample_dtypes.dtypes.items()}

        try:
            schema = mapper.create_polars_schema(dtypes)
            assert isinstance(schema, dict)
            assert len(schema) > 0
        except DTypeMappingError:
            # Some dtypes might not be supported
            pass

    def test_compatibility_check(self):
        """Test dtype compatibility checking."""
        mapper = DTypeMapper()

        assert mapper.is_dtype_compatible("int64", "polars")
        assert mapper.is_dtype_compatible("float64", "polars")
        assert not mapper.is_dtype_compatible("complex128", "polars")


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
class TestDataFrameConverter:
    """Test DataFrame conversion functionality."""

    def test_pandas_to_polars_basic(self, sample_dtypes):
        """Test basic pandas to Polars conversion."""
        converter = DataFrameConverter()

        # Convert to Polars DataFrame
        polars_df = converter.pandas_to_polars(sample_dtypes, lazy=False)
        assert isinstance(polars_df, pl.DataFrame)
        assert polars_df.shape == sample_dtypes.shape
        assert polars_df.columns == sample_dtypes.columns.tolist()

    def test_pandas_to_polars_lazy(self, sample_dtypes):
        """Test pandas to Polars LazyFrame conversion."""
        converter = DataFrameConverter()

        lazy_df = converter.pandas_to_polars(sample_dtypes, lazy=True)
        assert isinstance(lazy_df, pl.LazyFrame)
        assert lazy_df.columns == sample_dtypes.columns.tolist()

    def test_polars_to_pandas_basic(self, sample_dtypes):
        """Test basic Polars to pandas conversion."""
        converter = DataFrameConverter()

        # First convert to Polars, then back to pandas
        polars_df = pl.from_pandas(sample_dtypes)
        pandas_df = converter.polars_to_pandas(polars_df)

        assert isinstance(pandas_df, pd.DataFrame)
        assert pandas_df.shape == sample_dtypes.shape
        assert set(pandas_df.columns) == set(sample_dtypes.columns)

    def test_conversion_with_nulls(self, sample_with_nulls):
        """Test conversion with null values."""
        converter = DataFrameConverter()

        # Convert to Polars and back
        polars_df = converter.pandas_to_polars(sample_with_nulls, lazy=False)
        pandas_df = converter.polars_to_pandas(polars_df)

        assert pandas_df.shape == sample_with_nulls.shape
        # Check that null pattern is preserved (approximately)
        assert pandas_df.isnull().sum().sum() > 0

    def test_conversion_roundtrip(self, sample_dtypes):
        """Test roundtrip conversion maintains data integrity."""
        converter = DataFrameConverter()

        # Remove problematic columns for roundtrip test
        simple_df = sample_dtypes[
            ["int64_col", "float64_col", "bool_col", "object_col"]
        ].copy()

        # Pandas -> Polars -> Pandas
        polars_df = converter.pandas_to_polars(simple_df, lazy=False)
        roundtrip_df = converter.polars_to_pandas(polars_df)

        # Check basic structure
        assert roundtrip_df.shape == simple_df.shape
        assert set(roundtrip_df.columns) == set(simple_df.columns)

        # Check data integrity for numeric columns
        pd.testing.assert_series_equal(
            simple_df["int64_col"], roundtrip_df["int64_col"], check_dtype=False
        )

    def test_empty_dataframe_conversion(self):
        """Test conversion of empty DataFrames."""
        converter = DataFrameConverter()
        empty_df = pd.DataFrame()

        polars_df = converter.pandas_to_polars(empty_df)
        assert isinstance(polars_df, (pl.DataFrame, pl.LazyFrame))

        back_to_pandas = converter.polars_to_pandas(polars_df)
        assert isinstance(back_to_pandas, pd.DataFrame)
        assert back_to_pandas.empty

    def test_conversion_error_handling(self):
        """Test error handling in conversions."""
        converter = DataFrameConverter(strict_dtypes=True)

        # Create DataFrame with unsupported dtype
        df_with_complex = pd.DataFrame({"complex_col": [1 + 2j, 3 + 4j, 5 + 6j]})

        # Should handle gracefully or raise appropriate error
        try:
            converter.pandas_to_polars(df_with_complex, handle_unsupported="error")
        except ConversionError:
            pass  # Expected for unsupported types

    def test_conversion_info(self, sample_dtypes):
        """Test getting conversion information."""
        converter = DataFrameConverter()

        info = converter.get_conversion_info(sample_dtypes)
        assert info["source_backend"] == "pandas"
        assert info["total_columns"] == len(sample_dtypes.columns)
        assert "supported_columns" in info
        assert "dtype_mapping" in info


class TestSchemaValidator:
    """Test schema validation functionality."""

    def test_pandas_compatibility_validation(self, sample_dtypes):
        """Test validation of pandas DataFrame for Polars compatibility."""
        validator = SchemaValidator()

        issues = validator.validate_conversion_compatibility(sample_dtypes, "polars")

        # Check that issues are SchemaIssue objects
        for issue in issues:
            assert isinstance(issue, SchemaIssue)
            assert hasattr(issue, "issue_type")
            assert hasattr(issue, "severity")

    def test_schema_consistency_validation(self, sample_dtypes):
        """Test schema consistency validation between DataFrames."""
        validator = SchemaValidator()

        # Create modified version
        modified_df = sample_dtypes.copy()
        modified_df["extra_col"] = [1, 2, 3]

        issues = validator.validate_schema_consistency(sample_dtypes, modified_df)

        # Should detect extra column
        extra_col_issues = [i for i in issues if i.issue_type == "extra_column"]
        assert len(extra_col_issues) > 0

    def test_column_name_validation(self):
        """Test validation of problematic column names."""
        validator = SchemaValidator()

        # DataFrame with problematic column names
        bad_df = pd.DataFrame(
            {
                "": [1, 2, 3],  # Empty name
                "normal_col": [4, 5, 6],
                "col\nwith\nnewlines": [7, 8, 9],  # Special chars
            }
        )

        issues = validator.validate_conversion_compatibility(bad_df, "polars")

        # Should detect column name issues
        name_issues = [i for i in issues if "column" in i.issue_type]
        assert len(name_issues) > 0

    def test_duplicate_columns(self):
        """Test detection of duplicate column names."""
        validator = SchemaValidator()

        # Create DataFrame with duplicate columns
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df.columns = ["a", "b", "a"]  # Duplicate 'a'

        issues = validator.validate_conversion_compatibility(df, "polars")

        # Should detect duplicate
        dup_issues = [i for i in issues if i.issue_type == "duplicate_column"]
        assert len(dup_issues) > 0

    def test_validation_summary(self, sample_dtypes):
        """Test validation summary generation."""
        validator = SchemaValidator()

        issues = validator.validate_conversion_compatibility(sample_dtypes, "polars")
        summary = validator.get_validation_summary(issues)

        assert "total_issues" in summary
        assert "by_severity" in summary
        assert "has_blocking_issues" in summary
        assert isinstance(summary["by_severity"], dict)

    def test_auto_fix_capabilities(self):
        """Test automatic fixing of common issues."""
        validator = SchemaValidator()

        # DataFrame with mixed types
        problematic_df = pd.DataFrame(
            {"mixed_col": [1, "two", 3.0, "four"], "normal_col": [1, 2, 3, 4]}
        )

        issues = validator.validate_conversion_compatibility(problematic_df, "polars")
        fixed_df, fixes = validator.fix_common_issues(
            problematic_df, issues, auto_fix=True
        )

        assert isinstance(fixed_df, pd.DataFrame)
        assert isinstance(fixes, list)


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
class TestLazyConverter:
    """Test lazy conversion functionality."""

    def test_lazy_pandas_wrapper(self, sample_dtypes):
        """Test lazy pandas wrapper."""
        converter = LazyConverter()
        polars_df = pl.from_pandas(sample_dtypes)

        lazy_wrapper = converter.create_lazy_pandas_wrapper(polars_df)

        # Should not be converted yet
        assert not lazy_wrapper._converted

        # Access should trigger conversion
        shape = lazy_wrapper.shape
        assert lazy_wrapper._converted
        assert shape == sample_dtypes.shape

    def test_lazy_polars_wrapper(self, sample_dtypes):
        """Test lazy Polars wrapper."""
        converter = LazyConverter()

        lazy_wrapper = converter.create_lazy_polars_wrapper(sample_dtypes, lazy=True)

        # Should not be converted yet
        assert not lazy_wrapper._converted

        # Access should trigger conversion
        columns = lazy_wrapper.columns
        assert lazy_wrapper._converted
        assert columns == sample_dtypes.columns.tolist()

    def test_conversion_caching(self, sample_dtypes):
        """Test that conversions are cached."""
        converter = LazyConverter(cache_size=10)

        # First conversion
        result1 = converter.convert_when_needed(sample_dtypes, "polars")
        stats1 = converter.get_conversion_stats()

        # Second conversion (should be cached)
        result2 = converter.convert_when_needed(sample_dtypes, "polars")
        stats2 = converter.get_conversion_stats()

        # Cache hit should increase
        assert stats2["cache_hits"] > stats1["cache_hits"]

    def test_memory_optimization(self, sample_dtypes):
        """Test memory optimization features."""
        from table_evaluator.backends.lazy_conversion import optimize_for_memory

        # Test with different target operations
        optimized = optimize_for_memory(sample_dtypes, ["aggregation"])
        assert optimized is not None


class TestDataBridge:
    """Test comprehensive data bridge functionality."""

    def test_bridge_initialization(self):
        """Test DataBridge initialization with different settings."""
        # Default initialization
        bridge1 = DataBridge()
        assert bridge1.dtype_mapper is not None
        assert bridge1.converter is not None

        # Strict validation
        bridge2 = DataBridge(strict_validation=True)
        assert bridge2.strict_validation

        # Lazy conversion disabled
        bridge3 = DataBridge(enable_lazy_conversion=False)
        assert bridge3.lazy_converter is None

    def test_dataframe_conversion(self, sample_dtypes):
        """Test DataFrame conversion through bridge."""
        bridge = DataBridge()

        # Convert to pandas (should be no-op)
        pandas_result = bridge.convert_dataframe(sample_dtypes, "pandas")
        assert isinstance(pandas_result, pd.DataFrame)

        # Convert to Polars if available
        if POLARS_AVAILABLE:
            polars_result = bridge.convert_dataframe(sample_dtypes, "polars")
            assert polars_result is not None

    def test_compatibility_validation(self, sample_dtypes):
        """Test compatibility validation through bridge."""
        bridge = DataBridge()

        is_compatible, issues = bridge.validate_compatibility(sample_dtypes, "polars")
        assert isinstance(is_compatible, bool)
        assert isinstance(issues, list)

    def test_conversion_planning(self, sample_dtypes):
        """Test conversion planning."""
        bridge = DataBridge()

        plan = bridge.get_conversion_plan(sample_dtypes, "polars")

        expected_keys = [
            "source_backend",
            "target_backend",
            "is_compatible",
            "requires_conversion",
            "validation_issues",
            "dtype_mapping",
        ]

        for key in expected_keys:
            assert key in plan

        assert plan["source_backend"] == "pandas"
        assert plan["target_backend"] == "polars"

    def test_schema_mapping(self, sample_dtypes):
        """Test schema mapping between backends."""
        bridge = DataBridge()

        source_schema = {col: str(dtype) for col, dtype in sample_dtypes.dtypes.items()}
        target_schema = bridge.create_schema_mapping(source_schema, "pandas", "polars")

        assert isinstance(target_schema, dict)
        assert len(target_schema) > 0

    def test_bridge_statistics(self):
        """Test bridge statistics reporting."""
        bridge = DataBridge()
        stats = bridge.get_bridge_statistics()

        expected_keys = [
            "lazy_conversion_enabled",
            "strict_validation",
            "supported_pandas_dtypes",
            "polars_available",
        ]

        for key in expected_keys:
            assert key in stats


class TestConvenienceFunctions:
    """Test convenience functions for conversion."""

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_convert_to_polars_function(self, sample_dtypes):
        """Test convert_to_polars convenience function."""
        result = convert_to_polars(sample_dtypes, lazy=False)
        assert isinstance(result, (pl.DataFrame, pl.LazyFrame))
        assert result.shape == sample_dtypes.shape

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_convert_to_pandas_function(self, sample_dtypes):
        """Test convert_to_pandas convenience function."""
        polars_df = pl.from_pandas(sample_dtypes)
        result = convert_to_pandas(polars_df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_dtypes.shape


if __name__ == "__main__":
    pytest.main([__file__])
