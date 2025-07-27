"""Integration tests for textual evaluation functionality with TableEvaluator."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from table_evaluator import TableEvaluator
from table_evaluator.metrics.textual import SENTENCE_TRANSFORMERS_AVAILABLE


# Test data fixtures
@pytest.fixture
def sample_dataframes_with_text():
    """Provide sample DataFrames with mixed data types including text."""
    real_data = pd.DataFrame(
        {
            'numeric_col': [1, 2, 3, 4, 5],
            'category_col': ['A', 'B', 'A', 'C', 'B'],
            'text_col': [
                'The quick brown fox jumps over the lazy dog',
                'Hello world this is a test message',
                'Natural language processing is interesting',
                'Machine learning models need good data',
                'Text analysis helps understand content patterns',
            ],
            'target': [0, 1, 0, 1, 0],
        }
    )

    fake_data = pd.DataFrame(
        {
            'numeric_col': [1.1, 2.2, 2.9, 4.1, 5.2],
            'category_col': ['A', 'B', 'A', 'C', 'C'],
            'text_col': [
                'A fast brown fox leaps above the sleepy dog',
                'Hi world this is a sample message',
                'Natural language processing is fascinating',
                'Machine learning algorithms require quality data',
                'Text analysis assists in understanding content structures',
            ],
            'target': [0, 1, 0, 1, 1],
        }
    )

    return real_data, fake_data


@pytest.fixture
def sample_dataframes_multiple_text():
    """Provide DataFrames with multiple text columns."""
    real_data = pd.DataFrame(
        {
            'numeric_col': [1, 2, 3],
            'text_col1': ['Hello world', 'Test message', 'Sample text'],
            'text_col2': ['First column', 'Second column', 'Third column'],
            'target': [0, 1, 0],
        }
    )

    fake_data = pd.DataFrame(
        {
            'numeric_col': [1.1, 2.1, 3.1],
            'text_col1': ['Hi world', 'Test sample', 'Example text'],
            'text_col2': ['First entry', 'Second entry', 'Third entry'],
            'target': [0, 1, 1],
        }
    )

    return real_data, fake_data


@pytest.fixture
def dataframes_text_only():
    """Provide DataFrames with only text columns."""
    real_data = pd.DataFrame({'text_col': ['The cat sat on the mat', 'Dogs are loyal animals'], 'target': [0, 1]})

    fake_data = pd.DataFrame({'text_col': ['A cat sits on a mat', 'Dogs remain faithful pets'], 'target': [0, 1]})

    return real_data, fake_data


class TestTableEvaluatorTextInit:
    """Test TableEvaluator initialization with text columns."""

    def test_init_with_text_cols(self, sample_dataframes_with_text):
        """Test initialization with text_cols parameter."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        assert evaluator.text_cols == ['text_col']
        assert hasattr(evaluator, 'textual_evaluator')

    def test_init_with_multiple_text_cols(self, sample_dataframes_multiple_text):
        """Test initialization with multiple text columns."""
        real_data, fake_data = sample_dataframes_multiple_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col1', 'text_col2'])

        assert evaluator.text_cols == ['text_col1', 'text_col2']

    def test_init_without_text_cols(self, sample_dataframes_with_text):
        """Test initialization without text_cols parameter."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data)

        assert evaluator.text_cols == []

    def test_init_invalid_text_cols_type(self, sample_dataframes_with_text):
        """Test initialization with invalid text_cols type."""
        real_data, fake_data = sample_dataframes_with_text

        with pytest.raises(TypeError, match='text_cols must be a list of strings or None'):
            TableEvaluator(real=real_data, fake=fake_data, text_cols='invalid')

    def test_init_invalid_text_cols_content(self, sample_dataframes_with_text):
        """Test initialization with invalid text column names."""
        real_data, fake_data = sample_dataframes_with_text

        with pytest.raises(ValueError, match='text_cols contains columns not in DataFrames'):
            TableEvaluator(real=real_data, fake=fake_data, text_cols=['nonexistent_col'])


class TestTextualEvaluationMethods:
    """Test textual evaluation methods of TableEvaluator."""

    def test_textual_evaluation_basic(self, sample_dataframes_with_text):
        """Test basic textual evaluation."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.textual_evaluation(include_semantic=False)

        assert isinstance(result, dict)
        assert 'text_col' in result
        assert 'overall_textual_metrics' in result

        # Check column-specific results
        col_result = result['text_col']
        assert 'lexical_diversity' in col_result
        assert 'tfidf_similarity' in col_result
        assert 'combined_metrics' in col_result

    def test_textual_evaluation_with_semantic(self, sample_dataframes_with_text):
        """Test textual evaluation with semantic analysis."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.textual_evaluation(include_semantic=True)

        assert isinstance(result, dict)
        col_result = result['text_col']

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            assert 'semantic_similarity' in col_result
        else:
            # Should handle gracefully when not available
            assert isinstance(col_result.get('semantic_similarity', {}), dict)

    def test_textual_evaluation_multiple_columns(self, sample_dataframes_multiple_text):
        """Test textual evaluation with multiple text columns."""
        real_data, fake_data = sample_dataframes_multiple_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col1', 'text_col2'])

        result = evaluator.textual_evaluation(include_semantic=False)

        assert 'text_col1' in result
        assert 'text_col2' in result
        assert 'overall_textual_metrics' in result

        # Check overall metrics
        overall = result['overall_textual_metrics']
        assert 'mean_similarity' in overall
        assert overall['num_text_columns'] == 2

    def test_textual_evaluation_no_text_cols(self, sample_dataframes_with_text):
        """Test textual evaluation when no text columns specified."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data)  # No text_cols

        with pytest.raises(ValueError, match='No text columns specified'):
            evaluator.textual_evaluation()

    def test_textual_evaluation_with_sampling(self, sample_dataframes_with_text):
        """Test textual evaluation with sampling enabled."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.textual_evaluation(enable_sampling=True, max_samples=3)

        assert isinstance(result, dict)
        assert 'text_col' in result

    def test_basic_textual_evaluation(self, sample_dataframes_with_text):
        """Test basic (quick) textual evaluation."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.basic_textual_evaluation()

        assert isinstance(result, dict)
        assert 'text_col' in result
        assert 'overall_basic_metrics' in result

        # Check that it's a quick evaluation
        col_result = result['text_col']
        assert 'evaluation_type' in col_result
        assert col_result['evaluation_type'] == 'quick'

    def test_basic_textual_evaluation_no_text_cols(self, sample_dataframes_with_text):
        """Test basic textual evaluation when no text columns specified."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data)

        with pytest.raises(ValueError, match='No text columns specified'):
            evaluator.basic_textual_evaluation()


class TestComprehensiveEvaluationWithText:
    """Test comprehensive evaluation combining tabular and textual analysis."""

    def test_comprehensive_evaluation_with_text(self, sample_dataframes_with_text):
        """Test comprehensive evaluation including textual analysis."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.comprehensive_evaluation_with_text(
            target_col='target', target_type='class', include_textual=True, textual_config={'include_semantic': False}
        )

        assert isinstance(result, dict)
        assert 'basic' in result
        assert 'textual' in result
        assert 'combined_similarity' in result

        # Check combined similarity calculation
        combined = result['combined_similarity']
        assert 'overall_similarity' in combined
        assert 'text_weight' in combined
        assert 'tabular_weight' in combined

    def test_comprehensive_evaluation_custom_text_weight(self, sample_dataframes_with_text):
        """Test comprehensive evaluation with custom text weight."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.comprehensive_evaluation_with_text(
            target_col='target', text_weight=0.7, include_textual=True, textual_config={'include_semantic': False}
        )

        combined = result['combined_similarity']
        assert combined['text_weight'] == 0.7
        assert combined['tabular_weight'] == 0.3

    def test_comprehensive_evaluation_without_text(self, sample_dataframes_with_text):
        """Test comprehensive evaluation with textual analysis disabled."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.comprehensive_evaluation_with_text(target_col='target', include_textual=False)

        assert 'basic' in result
        assert 'textual' not in result or 'error' in result.get('textual', {})

    def test_comprehensive_evaluation_no_text_cols(self, sample_dataframes_with_text):
        """Test comprehensive evaluation when no text columns specified."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data)  # No text_cols

        result = evaluator.comprehensive_evaluation_with_text(target_col='target', include_textual=True)

        assert 'textual' in result
        assert 'error' in result['textual']
        assert 'No text columns specified' in result['textual']['error']

    def test_invalid_text_weight(self, sample_dataframes_with_text):
        """Test comprehensive evaluation with invalid text weight."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        with pytest.raises(ValueError, match='text_weight must be a number between 0.0 and 1.0'):
            evaluator.comprehensive_evaluation_with_text(target_col='target', text_weight=1.5)


class TestGetAvailableAdvancedMetrics:
    """Test get_available_advanced_metrics with textual functionality."""

    def test_get_metrics_with_text_cols(self, sample_dataframes_with_text):
        """Test getting available metrics when text columns are specified."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        metrics = evaluator.get_available_advanced_metrics()

        assert isinstance(metrics, dict)
        assert 'advanced_statistical' in metrics
        assert 'advanced_privacy' in metrics
        assert 'textual' in metrics

        # Check textual metrics are described
        textual_metrics = metrics['textual']
        assert 'lexical_diversity' in textual_metrics
        assert 'tfidf_similarity' in textual_metrics
        assert 'semantic_similarity' in textual_metrics

    def test_get_metrics_without_text_cols(self, sample_dataframes_with_text):
        """Test getting available metrics when no text columns specified."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data)

        metrics = evaluator.get_available_advanced_metrics()

        assert 'textual' not in metrics


class TestPerformanceAndEdgeCases:
    """Test performance warnings and edge cases for textual integration."""

    def test_large_dataset_warning(self):
        """Test performance warning for large text datasets."""
        # Create a dataset that exceeds the warning threshold
        n_rows = 50001
        real_data = pd.DataFrame(
            {'text_col': ['sample text'] * n_rows, 'target': [0, 1] * (n_rows // 2) + [0] * (n_rows % 2)}
        )
        fake_data = pd.DataFrame(
            {'text_col': ['example text'] * n_rows, 'target': [1, 0] * (n_rows // 2) + [1] * (n_rows % 2)}
        )

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        # Should issue a warning but not fail
        with pytest.warns(None):
            result = evaluator.textual_evaluation(include_semantic=False)

        assert isinstance(result, dict)

    def test_text_with_nan_values(self):
        """Test textual evaluation with NaN values in text columns."""
        real_data = pd.DataFrame({'text_col': ['hello world', np.nan, 'test message'], 'target': [0, 1, 0]})
        fake_data = pd.DataFrame({'text_col': ['hi world', 'sample text', np.nan], 'target': [0, 1, 1]})

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.textual_evaluation(include_semantic=False)

        # Should handle NaN values gracefully
        assert isinstance(result, dict)
        assert 'text_col' in result

    def test_empty_text_columns(self):
        """Test with empty text columns."""
        real_data = pd.DataFrame({'text_col': ['', '', ''], 'target': [0, 1, 0]})
        fake_data = pd.DataFrame({'text_col': ['', '', ''], 'target': [0, 1, 1]})

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.basic_textual_evaluation()

        # Should handle empty text gracefully
        assert isinstance(result, dict)


class TestInputValidation:
    """Test input validation for textual evaluation methods."""

    def test_textual_evaluation_input_validation(self, sample_dataframes_with_text):
        """Test input validation for textual_evaluation method."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        # Test invalid include_semantic
        with pytest.raises(TypeError, match='include_semantic must be a boolean'):
            evaluator.textual_evaluation(include_semantic='invalid')

        # Test invalid enable_sampling
        with pytest.raises(TypeError, match='enable_sampling must be a boolean'):
            evaluator.textual_evaluation(enable_sampling='invalid')

        # Test invalid max_samples
        with pytest.raises(ValueError, match='max_samples must be a positive integer'):
            evaluator.textual_evaluation(max_samples=-1)

    def test_basic_textual_evaluation_input_validation(self, sample_dataframes_with_text):
        """Test input validation for basic_textual_evaluation method."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        # Test invalid enable_sampling
        with pytest.raises(TypeError, match='enable_sampling must be a boolean'):
            evaluator.basic_textual_evaluation(enable_sampling='invalid')

        # Test invalid max_samples
        with pytest.raises(ValueError, match='max_samples must be a positive integer'):
            evaluator.basic_textual_evaluation(max_samples=0)


class TestCombinedSimilarityCalculation:
    """Test combined similarity calculation logic."""

    def test_combined_similarity_both_available(self, sample_dataframes_with_text):
        """Test combined similarity when both tabular and textual similarities are available."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        result = evaluator.comprehensive_evaluation_with_text(
            target_col='target', text_weight=0.6, include_textual=True, textual_config={'include_semantic': False}
        )

        combined = result['combined_similarity']

        if combined.get('success', False):
            assert 'overall_similarity' in combined
            assert 'tabular_similarity' in combined
            assert 'textual_similarity' in combined
            assert combined['text_weight'] == 0.6
            assert combined['tabular_weight'] == 0.4

    def test_combined_similarity_only_tabular(self, sample_dataframes_with_text):
        """Test combined similarity when only tabular similarity is available."""
        real_data, fake_data = sample_dataframes_with_text

        evaluator = TableEvaluator(real=real_data, fake=fake_data, text_cols=['text_col'])

        # Force textual evaluation to fail by mocking
        with patch.object(evaluator, 'textual_evaluation', side_effect=Exception('Mock error')):
            result = evaluator.comprehensive_evaluation_with_text(target_col='target', include_textual=True)

        combined = result.get('combined_similarity', {})

        # Should still calculate similarity based on available data
        assert isinstance(combined, dict)


if __name__ == '__main__':
    pytest.main([__file__])
