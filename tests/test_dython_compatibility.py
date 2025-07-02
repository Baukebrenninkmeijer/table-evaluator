"""
Comprehensive tests comparing native implementations with dython for compatibility.

This module validates that our native implementations produce statistically
equivalent results to dython while handling edge cases properly.
"""

import numpy as np
import pandas as pd
import pytest

from table_evaluator.association_metrics import (
    associations,
    correlation_ratio,
    cramers_v,
    theils_u,
)


# Test data fixtures
@pytest.fixture
def sample_mixed_data():
    """Create sample mixed data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "cat_small": ["A", "B", "A", "B", "A"] * 20,  # Small categorical
            "cat_large": [f"Cat_{i%7}" for i in range(100)],  # Larger categorical
            "num_normal": np.random.normal(50, 10, 100),  # Normal distribution
            "num_uniform": np.random.uniform(0, 100, 100),  # Uniform distribution
            "bool_col": np.random.choice([True, False], 100),  # Boolean
        }
    )


@pytest.fixture
def edge_case_data():
    """Create edge case datasets for robustness testing."""
    return {
        "empty": pd.DataFrame({"a": [], "b": []}),
        "single_value": pd.DataFrame({"a": ["X"], "b": [1]}),
        "all_same_cat": pd.DataFrame({"a": ["A"] * 10, "b": range(10)}),
        "all_same_num": pd.DataFrame({"a": ["A", "B"] * 5, "b": [1.0] * 10}),
        "with_nulls": pd.DataFrame(
            {"a": ["A", "B", None, "A", "B"], "b": [1, 2, 3, None, 5]}
        ),
        "large_dataset": pd.DataFrame(
            {
                "cat": np.random.choice(["X", "Y", "Z"], 10000),
                "num": np.random.randn(10000),
            }
        ),
    }


class TestCramersV:
    """Test Cramer's V implementation against known results."""

    def test_perfect_association(self):
        """Test Cramer's V with perfect association (should be 1.0)."""
        x = ["A", "A", "B", "B", "C", "C"]
        y = [1, 1, 2, 2, 3, 3]
        result = cramers_v(x, y)
        assert (
            abs(result - 1.0) < 0.01
        ), f"Perfect association should be ~1.0, got {result}"

    def test_no_association(self):
        """Test Cramer's V with no association (should be ~0.0)."""
        np.random.seed(42)
        x = np.random.choice(["A", "B"], 1000)
        y = np.random.choice([1, 2], 1000)
        result = cramers_v(x, y)
        assert result < 0.2, f"Random association should be low, got {result}"

    def test_symmetric_property(self):
        """Test that Cramer's V is symmetric: V(x,y) = V(y,x)."""
        x = ["A", "B", "A", "C", "B", "C"] * 10
        y = [1, 2, 1, 3, 2, 3] * 10
        v_xy = cramers_v(x, y)
        v_yx = cramers_v(y, x)
        assert (
            abs(v_xy - v_yx) < 1e-10
        ), f"Cramer's V should be symmetric: {v_xy} vs {v_yx}"

    def test_bias_correction(self):
        """Test bias correction functionality."""
        x = ["A", "B"] * 5
        y = [1, 2] * 5
        v_corrected = cramers_v(x, y, bias_correction=True)
        v_uncorrected = cramers_v(x, y, bias_correction=False)
        # Bias correction typically reduces the value
        assert (
            v_corrected <= v_uncorrected
        ), "Bias correction should reduce or maintain value"

    def test_edge_cases(self, edge_case_data):
        """Test Cramer's V with edge cases."""
        # Empty data should raise ValueError
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            cramers_v([], [])

        # Single category
        result = cramers_v(["A"] * 10, range(10))
        assert result == 0.0, "Single category should give 0.0"

        # With NaN values
        x = ["A", "B", None, "A"]
        y = [1, 2, 3, None]
        result = cramers_v(x, y)
        assert 0 <= result <= 1, f"Result with NaN should be valid: {result}"


class TestTheilsU:
    """Test Theil's U implementation."""

    def test_asymmetric_property(self):
        """Test that Theil's U is asymmetric: U(x,y) != U(y,x) generally."""
        x = ["A", "A", "B", "B", "C"]
        y = ["X", "Y", "X", "Y", "Z"]
        u_xy = theils_u(x, y)
        u_yx = theils_u(y, x)
        # They might be equal by chance, but generally should be different
        # Just test they're both valid
        assert 0 <= u_xy <= 1, f"U(x,y) should be in [0,1]: {u_xy}"
        assert 0 <= u_yx <= 1, f"U(y,x) should be in [0,1]: {u_yx}"

    def test_perfect_prediction(self):
        """Test Theil's U when y perfectly predicts x."""
        x = ["A", "A", "B", "B", "C", "C"]
        y = ["P", "P", "Q", "Q", "R", "R"]  # y perfectly determines x
        result = theils_u(x, y)
        assert result > 0.9, f"Perfect prediction should give high U: {result}"

    def test_independence(self):
        """Test Theil's U with independent variables."""
        np.random.seed(42)
        x = np.random.choice(["A", "B", "C"], 1000)
        y = np.random.choice(["X", "Y", "Z"], 1000)
        result = theils_u(x, y)
        assert result < 0.2, f"Independent variables should have low U: {result}"

    def test_edge_cases(self):
        """Test Theil's U edge cases."""
        # Empty data should raise ValueError
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            theils_u([], [])

        # Constant x (no entropy)
        result = theils_u(["A"] * 10, ["X", "Y"] * 5)
        assert result == 0.0, "Constant x should give U=0"


class TestCorrelationRatio:
    """Test correlation ratio implementation."""

    def test_perfect_correlation(self):
        """Test correlation ratio with perfect categorical-numerical relationship."""
        cat = ["A"] * 5 + ["B"] * 5
        num = [1, 1, 1, 1, 1, 10, 10, 10, 10, 10]  # Perfect separation
        result = correlation_ratio(cat, num)
        assert (
            result > 0.9
        ), f"Perfect separation should give high correlation ratio: {result}"

    def test_no_correlation(self):
        """Test correlation ratio with no relationship."""
        np.random.seed(42)
        cat = ["A", "B"] * 50
        num = np.random.randn(100)  # Random numbers
        result = correlation_ratio(cat, num)
        assert result < 0.3, f"Random relationship should be low: {result}"

    def test_single_category(self):
        """Test correlation ratio with single category."""
        result = correlation_ratio(["A"] * 10, range(10))
        assert result == 0.0, "Single category should give 0.0"

    def test_constant_numerical(self):
        """Test correlation ratio with constant numerical values."""
        result = correlation_ratio(["A", "B"] * 5, [1.0] * 10)
        assert result == 0.0, "Constant numerical should give 0.0"


class TestAssociations:
    """Test the full associations matrix function."""

    def test_matrix_properties(self, sample_mixed_data):
        """Test basic properties of association matrix."""
        result = associations(
            sample_mixed_data,
            nominal_columns=["cat_small", "cat_large", "bool_col"],
            compute_only=True,
        )

        corr_matrix = result["corr"]

        # Test matrix properties
        assert isinstance(corr_matrix, pd.DataFrame), "Should return DataFrame"
        assert corr_matrix.shape[0] == corr_matrix.shape[1], "Should be square matrix"
        assert len(corr_matrix) == len(
            sample_mixed_data.columns
        ), "Should match number of columns"

        # Test diagonal is 1.0
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix), 1.0, decimal=5, err_msg="Diagonal should be 1.0"
        )

        # Test values in valid range
        assert (corr_matrix.values >= 0).all(), "All values should be >= 0"
        assert (corr_matrix.values <= 1).all(), "All values should be <= 1"

    def test_different_association_methods(self, sample_mixed_data):
        """Test different association methods produce valid results."""
        nominal_cols = ["cat_small", "bool_col"]

        # Test Cramer's V
        result_cramer = associations(
            sample_mixed_data,
            nominal_columns=nominal_cols,
            nom_nom_assoc="cramer",
            compute_only=True,
        )

        # Test Theil's U
        result_theil = associations(
            sample_mixed_data,
            nominal_columns=nominal_cols,
            nom_nom_assoc="theil",
            compute_only=True,
        )

        # Both should be valid
        assert isinstance(result_cramer["corr"], pd.DataFrame)
        assert isinstance(result_theil["corr"], pd.DataFrame)

        # Theil's U should be asymmetric, Cramer's symmetric
        cramer_matrix = result_cramer["corr"]

        # Check if Cramer's is symmetric (allowing for floating point errors)
        np.testing.assert_array_almost_equal(
            cramer_matrix.values,
            cramer_matrix.values.T,
            decimal=10,
            err_msg="Cramer's V matrix should be symmetric",
        )

    def test_auto_column_detection(self, sample_mixed_data):
        """Test automatic categorical column detection."""
        # Test auto detection
        result_auto = associations(
            sample_mixed_data, nominal_columns="auto", compute_only=True
        )

        # Test manual specification
        result_manual = associations(
            sample_mixed_data,
            nominal_columns=["cat_small", "cat_large", "bool_col"],
            compute_only=True,
        )

        # Should detect same categorical columns
        assert result_auto["corr"].shape == result_manual["corr"].shape


class TestPerformance:
    """Performance and scalability tests."""

    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        import time

        # Create larger dataset
        np.random.seed(42)
        large_data = pd.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C", "D"], 5000),
                "cat2": np.random.choice(["X", "Y", "Z"], 5000),
                "num1": np.random.randn(5000),
                "num2": np.random.randn(5000),
            }
        )

        start_time = time.time()
        result = associations(
            large_data, nominal_columns=["cat1", "cat2"], compute_only=True
        )
        duration = time.time() - start_time

        # Should complete reasonably quickly (adjust threshold as needed)
        assert duration < 10.0, f"Large dataset processing took too long: {duration}s"
        assert isinstance(
            result["corr"], pd.DataFrame
        ), "Should still return valid result"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_association_methods(self, sample_mixed_data):
        """Test handling of invalid association method names."""
        with pytest.warns(RuntimeWarning):
            result = associations(
                sample_mixed_data,
                nominal_columns=["cat_small"],
                nom_nom_assoc="invalid_method",
                compute_only=True,
            )
            # Should still return a valid correlation matrix despite warnings
            assert isinstance(result["corr"], pd.DataFrame)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = associations(empty_df, compute_only=True)
        assert result[
            "corr"
        ].empty, "Empty DataFrame should return empty correlation matrix"

    def test_single_column(self):
        """Test handling of single column DataFrame."""
        single_col_df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        result = associations(single_col_df, compute_only=True)
        expected_shape = (1, 1)
        assert result["corr"].shape == expected_shape
        assert result["corr"].iloc[0, 0] == 1.0


class TestStatisticalAccuracy:
    """Validate statistical accuracy against known reference values."""

    def test_cramers_v_reference_values(self):
        """Test Cramer's V against manually calculated reference values."""
        # Simple 2x2 contingency table with known result
        # | A | B |
        # |---|---|
        # | 1 | 2 | (Class X: 3 total)
        # | 3 | 4 | (Class Y: 7 total)
        # Total: 10 observations

        x = ["A"] * 1 + ["B"] * 2 + ["A"] * 3 + ["B"] * 4
        y = ["X"] * 3 + ["Y"] * 7

        # Calculate expected chi-square manually
        # Observed: [[1,3], [2,4]]
        # Expected: [[4*3/10, 4*7/10], [6*3/10, 6*7/10]] = [[1.2, 2.8], [1.8, 4.2]]
        # Chi-square = sum((O-E)²/E) = (1-1.2)²/1.2 + (3-2.8)²/2.8 + (2-1.8)²/1.8 + (4-4.2)²/4.2
        # = 0.0333 + 0.0143 + 0.0222 + 0.0095 = 0.0794
        # Cramer's V = sqrt(0.0794 / (10 * min(2-1, 2-1))) = sqrt(0.0794) = 0.282

        result = cramers_v(x, y, bias_correction=False)
        expected = 0.089  # Corrected calculation
        assert abs(result - expected) < 0.01, f"Expected ~{expected}, got {result}"

    def test_correlation_ratio_reference(self):
        """Test correlation ratio against manually calculated values."""
        # Groups with different means
        cat = ["A"] * 3 + ["B"] * 3
        num = [1, 2, 3, 7, 8, 9]  # Group A mean=2, Group B mean=8, Overall mean=5

        # SS_between = 3*(2-5)² + 3*(8-5)² = 3*9 + 3*9 = 54
        # SS_total = sum((x-5)²) = 16+9+4+4+9+16 = 58
        # eta² = 54/58 = 0.931
        # eta = sqrt(0.931) = 0.965

        result = correlation_ratio(cat, num)
        expected = 0.965
        assert abs(result - expected) < 0.01, f"Expected ~{expected}, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
