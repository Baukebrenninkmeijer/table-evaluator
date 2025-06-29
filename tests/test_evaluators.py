"""Tests for the new evaluator classes."""

import pandas as pd
import pytest
from scipy import stats

from table_evaluator.data.data_converter import DataConverter
from table_evaluator.evaluators.ml_evaluator import MLEvaluator
from table_evaluator.evaluators.privacy_evaluator import PrivacyEvaluator
from table_evaluator.evaluators.statistical_evaluator import StatisticalEvaluator


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample data for testing."""
    real_data = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [0.1, 0.2, 0.3, 0.4, 0.5],
            "D": [True, False, True, False, True],
        }
    )
    fake_data = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "f"],
            "C": [0.15, 0.25, 0.35, 0.45, 0.55],
            "D": [True, True, False, False, True],
        }
    )
    return real_data, fake_data


def test_statistical_evaluator_basic_evaluation(sample_data):
    """Test StatisticalEvaluator basic statistical evaluation."""
    real, fake = sample_data
    evaluator = StatisticalEvaluator(stats.pearsonr, verbose=False)

    numerical_columns = ["A", "C"]
    result = evaluator.basic_statistical_evaluation(real, fake, numerical_columns)

    assert isinstance(result, float)
    assert -1 <= result <= 1  # Correlation should be between -1 and 1


def test_statistical_evaluator_correlation_correlation(sample_data):
    """Test StatisticalEvaluator correlation analysis."""
    real, fake = sample_data
    evaluator = StatisticalEvaluator(stats.pearsonr, verbose=False)

    categorical_columns = ["B", "D"]
    result = evaluator.correlation_correlation(real, fake, categorical_columns)

    assert isinstance(result, float)
    assert -1 <= result <= 1  # Correlation should be between -1 and 1


def test_statistical_evaluator_pca_correlation(sample_data):
    """Test StatisticalEvaluator PCA correlation."""
    real, fake = sample_data
    evaluator = StatisticalEvaluator(stats.pearsonr, verbose=False)

    # Convert to numerical for PCA
    converter = DataConverter()
    real_num, fake_num = converter.to_numerical(real, fake, ["B", "D"])

    result = evaluator.pca_correlation(real_num, fake_num, lingress=True)
    assert isinstance(result, float)

    result_mape = evaluator.pca_correlation(real_num, fake_num, lingress=False)
    assert isinstance(result_mape, float)


def test_privacy_evaluator_get_copies(sample_data):
    """Test PrivacyEvaluator copy detection."""
    real, fake = sample_data
    evaluator = PrivacyEvaluator(verbose=False)

    # Test with no copies (original data)
    copies = evaluator.get_copies(real, fake, return_len=False)
    assert isinstance(copies, pd.DataFrame)

    copy_count = evaluator.get_copies(real, fake, return_len=True)
    assert isinstance(copy_count, int)
    assert copy_count >= 0


def test_privacy_evaluator_get_duplicates(sample_data):
    """Test PrivacyEvaluator duplicate detection."""
    real, fake = sample_data
    evaluator = PrivacyEvaluator(verbose=False)

    # Test return values
    real_dups, fake_dups = evaluator.get_duplicates(real, fake, return_values=True)
    assert isinstance(real_dups, pd.DataFrame)
    assert isinstance(fake_dups, pd.DataFrame)

    # Test return counts
    real_count, fake_count = evaluator.get_duplicates(real, fake, return_values=False)
    assert isinstance(real_count, int)
    assert isinstance(fake_count, int)


def test_privacy_evaluator_row_distance(sample_data):
    """Test PrivacyEvaluator row distance calculation."""
    real, fake = sample_data
    evaluator = PrivacyEvaluator(verbose=False)

    # Convert to one-hot for distance calculation
    converter = DataConverter()
    real_encoded, fake_encoded = converter.to_one_hot(real, fake, ["B", "D"])

    mean_dist, std_dist = evaluator.row_distance(
        real_encoded, fake_encoded, n_samples=5
    )

    assert isinstance(mean_dist, float)
    assert isinstance(std_dist, float)
    assert mean_dist >= 0
    assert std_dist >= 0


def test_ml_evaluator_classification(sample_data):
    """Test MLEvaluator for classification task."""
    real, fake = sample_data
    evaluator = MLEvaluator(stats.pearsonr, random_seed=42, verbose=False)

    # Convert to numerical for ML evaluation
    converter = DataConverter()
    real_num, fake_num = converter.to_numerical(real, fake, ["B", "D"])

    result = evaluator.estimator_evaluation(
        real_num, fake_num, target_col="A", target_type="class", kfold=False
    )

    assert isinstance(result, float)
    # For very small datasets, the result might be NaN due to division by zero
    # This is expected behavior, so we just test that it's a float
    if not pd.isna(result):
        assert (
            0 <= result <= 1
        )  # Should be between 0 and 1 for classification when valid


def test_ml_evaluator_regression(sample_data):
    """Test MLEvaluator for regression task."""
    real, fake = sample_data
    evaluator = MLEvaluator(stats.pearsonr, random_seed=42, verbose=False)

    # Convert to numerical for ML evaluation
    converter = DataConverter()
    real_num, fake_num = converter.to_numerical(real, fake, ["B", "D"])

    result = evaluator.estimator_evaluation(
        real_num, fake_num, target_col="C", target_type="regr", kfold=False
    )

    assert isinstance(result, float)
    assert -1 <= result <= 1  # Correlation should be between -1 and 1


def test_data_converter_to_numerical(sample_data):
    """Test DataConverter numerical conversion."""
    real, fake = sample_data
    converter = DataConverter()

    real_num, fake_num = converter.to_numerical(real, fake, ["B", "D"])

    assert real_num.shape == real.shape
    assert fake_num.shape == fake.shape
    assert real_num.columns.tolist() == real.columns.tolist()
    assert fake_num.columns.tolist() == fake.columns.tolist()


def test_data_converter_to_one_hot(sample_data):
    """Test DataConverter one-hot encoding."""
    real, fake = sample_data
    converter = DataConverter()

    real_encoded, fake_encoded = converter.to_one_hot(real, fake, ["B", "D"])

    # One-hot encoding should increase number of columns
    assert real_encoded.shape[1] >= real.shape[1]
    assert fake_encoded.shape[1] >= fake.shape[1]
    # After alignment, both datasets should have the same columns
    assert set(real_encoded.columns) == set(fake_encoded.columns)
    assert real_encoded.columns.tolist() == fake_encoded.columns.tolist()


def test_data_converter_ensure_compatible_columns():
    """Test DataConverter column alignment."""
    real = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    fake = pd.DataFrame({"A": [1, 2], "B": [3, 4], "D": [7, 8]})

    converter = DataConverter()
    real_aligned, fake_aligned = converter.ensure_compatible_columns(real, fake)

    assert real_aligned.columns.tolist() == fake_aligned.columns.tolist()
    assert set(real_aligned.columns) == {"A", "B"}  # Only common columns
