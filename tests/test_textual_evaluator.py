"""Tests for textual evaluation functionality."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from table_evaluator.evaluators.textual_evaluator import TextualEvaluator
from table_evaluator.metrics.textual import (
    SENTENCE_TRANSFORMERS_AVAILABLE,
    comprehensive_textual_analysis,
    semantic_similarity_embeddings,
    text_length_distribution_similarity,
    tfidf_corpus_similarity,
    vocabulary_overlap_analysis,
)
from table_evaluator.models.textual_models import (
    ComprehensiveTextualAnalysisResult,
    ComprehensiveTextualResult,
    LengthDistributionResult,
    LexicalDiversityResult,
    QuickTextualResult,
    SemanticRawResult,
    SemanticSimilarityResult,
    TfidfRawResult,
    TfidfSimilarityResult,
    VocabularyOverlapResult,
)


# Test data fixtures
@pytest.fixture
def sample_texts():
    """Provide sample text data for testing."""
    real_texts = pd.Series(
        [
            'The quick brown fox jumps over the lazy dog',
            'Hello world this is a test message',
            'Natural language processing is interesting',
            'Machine learning models need good data',
            'Text analysis helps understand content patterns',
        ]
    )

    fake_texts = pd.Series(
        [
            'A fast brown fox leaps above the sleepy dog',
            'Hi world this is a sample message',
            'Natural language processing is fascinating',
            'Machine learning algorithms require quality data',
            'Text analysis assists in understanding content structures',
        ]
    )

    return real_texts, fake_texts


@pytest.fixture
def empty_texts():
    """Provide empty text data for edge case testing."""
    return pd.Series([], dtype=str), pd.Series([], dtype=str)


@pytest.fixture
def single_word_texts():
    """Provide single word text data."""
    real_texts = pd.Series(['cat', 'dog', 'bird', 'fish'])
    fake_texts = pd.Series(['cat', 'dog', 'mouse', 'shark'])
    return real_texts, fake_texts


@pytest.fixture
def texts_with_nan():
    """Provide text data with NaN values."""
    real_texts = pd.Series(['hello world', np.nan, 'test message', ''])
    fake_texts = pd.Series(['hello world', 'sample text', np.nan, 'test'])
    return real_texts, fake_texts


class TestTextLengthDistributionSimilarity:
    """Test text length distribution similarity functions."""

    def test_word_length_similarity(self, sample_texts):
        """Test word length distribution similarity."""
        real_texts, fake_texts = sample_texts
        result = text_length_distribution_similarity(real_texts, fake_texts, unit='word')

        assert hasattr(result, 'ks_statistic')
        assert hasattr(result, 'ks_p_value')
        assert hasattr(result, 'similarity_score')
        assert 0 <= result.ks_statistic <= 1
        assert 0 <= result.similarity_score <= 1

    def test_char_length_similarity(self, sample_texts):
        """Test character length distribution similarity."""
        real_texts, fake_texts = sample_texts
        result = text_length_distribution_similarity(real_texts, fake_texts, unit='char')

        assert hasattr(result, 'ks_statistic')
        assert hasattr(result, 'ks_p_value')
        assert hasattr(result, 'similarity_score')
        assert 0 <= result.ks_statistic <= 1
        assert 0 <= result.similarity_score <= 1

    def test_invalid_unit(self, sample_texts):
        """Test invalid unit parameter."""
        real_texts, fake_texts = sample_texts
        with pytest.raises(ValueError, match="unit must be 'word' or 'char'"):
            text_length_distribution_similarity(real_texts, fake_texts, unit='invalid')

    def test_invalid_input_types(self):
        """Test invalid input types."""
        with pytest.raises(TypeError):
            text_length_distribution_similarity('not_series', pd.Series(['test']))

        with pytest.raises(TypeError):
            text_length_distribution_similarity(pd.Series(['test']), 'not_series')

    def test_empty_data(self, empty_texts):
        """Test with empty text data."""
        real_texts, fake_texts = empty_texts
        result = text_length_distribution_similarity(real_texts, fake_texts)

        assert result.ks_statistic == 1.0
        assert result.ks_p_value == 0.0

    def test_with_nan_values(self, texts_with_nan):
        """Test with NaN values in text data."""
        real_texts, fake_texts = texts_with_nan
        result = text_length_distribution_similarity(real_texts, fake_texts)

        # Should handle NaN values gracefully
        assert hasattr(result, 'similarity_score')
        assert hasattr(result, 'ks_statistic')


class TestVocabularyOverlapAnalysis:
    """Test vocabulary overlap analysis functions."""

    def test_basic_vocabulary_overlap(self, sample_texts):
        """Test basic vocabulary overlap analysis."""
        real_texts, fake_texts = sample_texts
        result = vocabulary_overlap_analysis(real_texts, fake_texts)

        assert isinstance(result, VocabularyOverlapResult)
        assert hasattr(result, 'jaccard_similarity')
        assert hasattr(result, 'vocab_diversity_ratio')
        assert hasattr(result, 'real_vocab_size')
        assert hasattr(result, 'fake_vocab_size')
        assert hasattr(result, 'shared_vocab_size')

        assert 0 <= result.jaccard_similarity <= 1
        assert result.vocab_diversity_ratio > 0
        assert result.real_vocab_size >= 0
        assert result.fake_vocab_size >= 0
        assert result.shared_vocab_size >= 0

    def test_minimum_frequency_filter(self, sample_texts):
        """Test minimum frequency filtering."""
        real_texts, fake_texts = sample_texts
        result = vocabulary_overlap_analysis(real_texts, fake_texts, min_frequency=2)

        assert isinstance(result, VocabularyOverlapResult)
        # With higher frequency threshold, vocabulary sizes should be smaller or equal
        result_default = vocabulary_overlap_analysis(real_texts, fake_texts, min_frequency=1)
        assert result.real_vocab_size <= result_default.real_vocab_size

    def test_identical_texts(self):
        """Test with identical text corpora."""
        texts = pd.Series(['hello world', 'test message'])
        result = vocabulary_overlap_analysis(texts, texts)

        assert result.jaccard_similarity == 1.0
        assert result.vocab_diversity_ratio == 1.0

    def test_no_overlap(self):
        """Test with texts that have no vocabulary overlap."""
        real_texts = pd.Series(['cat dog'])
        fake_texts = pd.Series(['bird fish'])
        result = vocabulary_overlap_analysis(real_texts, fake_texts)

        assert result.jaccard_similarity == 0.0
        assert result.shared_vocab_size == 0

    def test_invalid_min_frequency(self, sample_texts):
        """Test invalid minimum frequency parameter."""
        real_texts, fake_texts = sample_texts
        with pytest.raises(ValueError, match='min_frequency'):
            vocabulary_overlap_analysis(real_texts, fake_texts, min_frequency=0)


class TestTfidfCorpusSimilarity:
    """Test TF-IDF corpus similarity functions."""

    def test_basic_tfidf_similarity(self, sample_texts):
        """Test basic TF-IDF similarity calculation."""
        real_texts, fake_texts = sample_texts
        result = tfidf_corpus_similarity(real_texts, fake_texts)

        assert isinstance(result, TfidfRawResult)
        assert hasattr(result, 'cosine_similarity')
        assert hasattr(result, 'tfidf_distance')
        assert hasattr(result, 'vocabulary_size')

        assert 0 <= result.cosine_similarity <= 1
        assert result.tfidf_distance >= 0
        assert result.vocabulary_size >= 0

    def test_tfidf_parameters(self, sample_texts):
        """Test TF-IDF with different parameters."""
        real_texts, fake_texts = sample_texts
        result = tfidf_corpus_similarity(
            real_texts, fake_texts, max_features=5, ngram_range=(1, 1), min_df=1, max_df=1.0
        )

        assert isinstance(result, TfidfRawResult)
        assert result.vocabulary_size <= 5

    def test_identical_corpora(self):
        """Test with identical text corpora."""
        texts = pd.Series(['hello world test', 'sample message here'])
        result = tfidf_corpus_similarity(texts, texts)

        # Identical corpora should have perfect similarity
        assert abs(result.cosine_similarity - 1.0) < 1e-6

    def test_empty_texts(self, empty_texts):
        """Test with empty text data."""
        real_texts, fake_texts = empty_texts
        result = tfidf_corpus_similarity(real_texts, fake_texts)

        assert result.cosine_similarity == 0.0
        assert result.tfidf_distance == 1.0


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason='sentence-transformers not available')
class TestSemanticSimilarityEmbeddings:
    """Test semantic similarity with embeddings (when available)."""

    def test_basic_semantic_similarity(self, sample_texts):
        """Test basic semantic similarity calculation."""
        real_texts, fake_texts = sample_texts
        result = semantic_similarity_embeddings(real_texts, fake_texts)

        assert isinstance(result, SemanticRawResult)
        assert hasattr(result, 'semantic_similarity')
        assert hasattr(result, 'embedding_distance')
        assert hasattr(result, 'model_name')

        assert 0 <= result.semantic_similarity <= 1
        assert result.embedding_distance >= 0

    def test_with_sampling(self, sample_texts):
        """Test semantic similarity with sampling enabled."""
        real_texts, fake_texts = sample_texts
        result = semantic_similarity_embeddings(real_texts, fake_texts, enable_sampling=True, max_samples=3)

        assert isinstance(result, SemanticRawResult)
        assert hasattr(result, 'semantic_similarity')

    def test_custom_model(self, sample_texts):
        """Test with custom model name."""
        real_texts, fake_texts = sample_texts
        model_name = 'all-MiniLM-L6-v2'
        result = semantic_similarity_embeddings(real_texts, fake_texts, model_name=model_name)

        assert result.model_name == model_name


class TestSemanticSimilarityWithoutDependency:
    """Test semantic similarity when dependencies are not available."""

    @patch('table_evaluator.metrics.textual.SENTENCE_TRANSFORMERS_AVAILABLE', new=False)
    def test_missing_dependency(self, sample_texts):
        """Test behavior when sentence-transformers is not available."""
        real_texts, fake_texts = sample_texts

        with pytest.raises(ImportError, match='sentence-transformers is required'):
            semantic_similarity_embeddings(real_texts, fake_texts)


class TestComprehensiveTextualAnalysis:
    """Test comprehensive textual analysis function."""

    def test_full_analysis(self, sample_texts):
        """Test comprehensive analysis with all metrics."""
        real_texts, fake_texts = sample_texts
        result = comprehensive_textual_analysis(real_texts, fake_texts, include_semantic=False)

        assert isinstance(result, ComprehensiveTextualAnalysisResult)
        assert hasattr(result, 'word_length_dist')
        assert hasattr(result, 'char_length_dist')
        assert hasattr(result, 'vocabulary_overlap')
        assert hasattr(result, 'tfidf_similarity')
        assert hasattr(result, 'overall_similarity')
        assert result.success is True

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason='sentence-transformers not available')
    def test_analysis_with_semantic(self, sample_texts):
        """Test comprehensive analysis including semantic similarity."""
        real_texts, fake_texts = sample_texts
        result = comprehensive_textual_analysis(real_texts, fake_texts, include_semantic=True)

        assert hasattr(result, 'semantic_similarity')
        assert result.semantic_similarity is not None

    def test_analysis_without_semantic(self, sample_texts):
        """Test comprehensive analysis without semantic similarity."""
        real_texts, fake_texts = sample_texts
        result = comprehensive_textual_analysis(real_texts, fake_texts, include_semantic=False)

        assert result.semantic_similarity is None

    def test_error_handling_in_analysis(self):
        """Test error handling in comprehensive analysis."""
        # Test with problematic data that might cause errors
        real_texts = pd.Series(['', '', ''], dtype=str)
        fake_texts = pd.Series(['', '', ''], dtype=str)

        result = comprehensive_textual_analysis(real_texts, fake_texts)

        # Should handle errors gracefully
        assert isinstance(result, ComprehensiveTextualAnalysisResult)
        # Some metrics might fail with empty strings, but result should still be valid
        assert hasattr(result, 'overall_similarity')


class TestTextualEvaluator:
    """Test TextualEvaluator class."""

    def test_initialization(self):
        """Test TextualEvaluator initialization."""
        evaluator = TextualEvaluator(verbose=True)
        assert evaluator.verbose is True

        evaluator = TextualEvaluator(verbose=False)
        assert evaluator.verbose is False

    def test_lexical_diversity_evaluation(self, sample_texts):
        """Test lexical diversity evaluation."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        result = evaluator.lexical_diversity_evaluation(real_texts, fake_texts)

        assert isinstance(result, LexicalDiversityResult)
        assert hasattr(result, 'word_length_distribution')
        assert hasattr(result, 'char_length_distribution')
        assert hasattr(result, 'vocabulary_overlap')
        assert hasattr(result, 'summary')

    def test_tfidf_similarity_evaluation(self, sample_texts):
        """Test TF-IDF similarity evaluation."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        result = evaluator.tfidf_similarity_evaluation(real_texts, fake_texts)

        assert isinstance(result, TfidfSimilarityResult)
        assert hasattr(result, 'summary')
        assert hasattr(result.summary, 'tfidf_cosine_similarity')

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason='sentence-transformers not available')
    def test_semantic_similarity_evaluation(self, sample_texts):
        """Test semantic similarity evaluation."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        result = evaluator.semantic_similarity_evaluation(real_texts, fake_texts)

        assert isinstance(result, SemanticSimilarityResult)
        assert result.available is True
        assert hasattr(result, 'summary')

    def test_semantic_similarity_unavailable(self, sample_texts):
        """Test semantic similarity when unavailable."""
        with patch('table_evaluator.evaluators.textual_evaluator.SENTENCE_TRANSFORMERS_AVAILABLE', new=False):
            real_texts, fake_texts = sample_texts
            evaluator = TextualEvaluator()
            result = evaluator.semantic_similarity_evaluation(real_texts, fake_texts)

            assert result.available is False
            assert result.error is not None

    def test_comprehensive_evaluation(self, sample_texts):
        """Test comprehensive evaluation."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        result = evaluator.comprehensive_evaluation(real_texts, fake_texts, include_semantic=False)

        assert isinstance(result, ComprehensiveTextualResult)
        assert hasattr(result, 'lexical_diversity')
        assert hasattr(result, 'tfidf_similarity')
        assert hasattr(result, 'recommendations')

    def test_quick_evaluation(self, sample_texts):
        """Test quick evaluation."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        result = evaluator.quick_evaluation(real_texts, fake_texts)

        assert isinstance(result, QuickTextualResult)
        assert hasattr(result, 'overall_similarity')
        assert hasattr(result, 'evaluation_type')
        assert result.evaluation_type == 'quick'

    def test_get_summary_for_integration(self, sample_texts):
        """Test summary extraction for TableEvaluator integration."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()

        # Test with comprehensive evaluation results
        eval_results = evaluator.comprehensive_evaluation(real_texts, fake_texts, include_semantic=False)
        summary = evaluator.get_summary_for_integration(eval_results)

        from table_evaluator.models.textual_models import TextualEvaluationSummary

        assert isinstance(summary, TextualEvaluationSummary)
        assert hasattr(summary, 'textual_similarity')
        assert hasattr(summary, 'quality_rating')
        assert hasattr(summary, 'evaluation_type')

    def test_quality_rating_methods(self):
        """Test quality rating methods."""
        evaluator = TextualEvaluator()

        # Test different quality levels
        assert evaluator._rate_lexical_quality(0.95) == 'Excellent'
        assert evaluator._rate_lexical_quality(0.85) == 'Good'
        assert evaluator._rate_lexical_quality(0.7) == 'Fair'
        assert evaluator._rate_lexical_quality(0.5) == 'Poor'
        assert evaluator._rate_lexical_quality(0.2) == 'Very Poor'

    def test_error_handling_in_evaluations(self, empty_texts):
        """Test error handling in various evaluation methods."""
        evaluator = TextualEvaluator()
        empty_real, empty_fake = empty_texts

        result = evaluator.quick_evaluation(empty_real, empty_fake)
        # Should handle gracefully, might have error but shouldn't crash
        assert isinstance(result, QuickTextualResult)
        assert 'error' in result.model_dump()


class TestPerformanceWarnings:
    """Test performance warnings for large datasets."""

    def test_large_dataset_warning(self):
        """Test warning for large text datasets."""
        # Create large dataset
        large_real = pd.Series(['test text'] * 50001)
        large_fake = pd.Series(['sample text'] * 50001)

        # Should not fail with large datasets (warning is logged via loguru, not warnings module)
        result = text_length_distribution_similarity(large_real, large_fake)

        assert isinstance(result, LengthDistributionResult)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_special_characters(self):
        """Test with special characters and unicode."""
        real_texts = pd.Series(['Hello! @#$%', 'Test with émojis 😀', 'Numbers 123 and symbols &*()'])
        fake_texts = pd.Series(['Hi! @#$%', 'Sample with émojis 😀', 'Digits 456 and symbols &*()'])

        result = vocabulary_overlap_analysis(real_texts, fake_texts)
        assert isinstance(result, VocabularyOverlapResult)

    def test_very_long_texts(self):
        """Test with very long text strings."""
        long_text = 'word ' * 1000  # 1000 words
        real_texts = pd.Series([long_text])
        fake_texts = pd.Series([long_text])

        result = tfidf_corpus_similarity(real_texts, fake_texts)
        assert isinstance(result, TfidfRawResult)

    def test_mixed_languages(self):
        """Test with mixed language content."""
        real_texts = pd.Series(['Hello world', 'Bonjour monde', 'Hola mundo'])
        fake_texts = pd.Series(['Hi world', 'Salut monde', 'Hola mundo'])

        result = vocabulary_overlap_analysis(real_texts, fake_texts)
        assert isinstance(result, VocabularyOverlapResult)

    def test_numeric_strings(self):
        """Test with numeric string content."""
        real_texts = pd.Series(['123 456', '789 012', '345 678'])
        fake_texts = pd.Series(['111 222', '333 444', '555 666'])

        result = text_length_distribution_similarity(real_texts, fake_texts)
        assert isinstance(result, LengthDistributionResult)


if __name__ == '__main__':
    pytest.main([__file__])
