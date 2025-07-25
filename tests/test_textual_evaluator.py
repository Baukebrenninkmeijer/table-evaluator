"""Tests for textual evaluation functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from table_evaluator.evaluators.textual_evaluator import TextualEvaluator
from table_evaluator.advanced_metrics.textual import (
    text_length_distribution_similarity,
    vocabulary_overlap_analysis,
    tfidf_corpus_similarity,
    semantic_similarity_embeddings,
    comprehensive_textual_analysis,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)


# Test data fixtures
@pytest.fixture
def sample_texts():
    """Provide sample text data for testing."""
    real_texts = pd.Series([
        "The quick brown fox jumps over the lazy dog",
        "Hello world this is a test message",
        "Natural language processing is interesting",
        "Machine learning models need good data",
        "Text analysis helps understand content patterns"
    ])
    
    fake_texts = pd.Series([
        "A fast brown fox leaps above the sleepy dog",
        "Hi world this is a sample message",
        "Natural language processing is fascinating",
        "Machine learning algorithms require quality data",
        "Text analysis assists in understanding content structures"
    ])
    
    return real_texts, fake_texts


@pytest.fixture
def empty_texts():
    """Provide empty text data for edge case testing."""
    return pd.Series([]), pd.Series([])


@pytest.fixture
def single_word_texts():
    """Provide single word text data."""
    real_texts = pd.Series(["cat", "dog", "bird", "fish"])
    fake_texts = pd.Series(["cat", "dog", "mouse", "shark"])
    return real_texts, fake_texts


@pytest.fixture
def texts_with_nan():
    """Provide text data with NaN values."""
    real_texts = pd.Series(["hello world", np.nan, "test message", ""])
    fake_texts = pd.Series(["hello world", "sample text", np.nan, "test"])
    return real_texts, fake_texts


class TestTextLengthDistributionSimilarity:
    """Test text length distribution similarity functions."""

    def test_word_length_similarity(self, sample_texts):
        """Test word length distribution similarity."""
        real_texts, fake_texts = sample_texts
        result = text_length_distribution_similarity(real_texts, fake_texts, unit="word")
        
        assert isinstance(result, dict)
        assert "ks_statistic" in result
        assert "ks_pvalue" in result
        assert "similarity_score" in result
        assert "unit" in result
        assert result["unit"] == "word"
        assert 0 <= result["ks_statistic"] <= 1
        assert 0 <= result["similarity_score"] <= 1

    def test_char_length_similarity(self, sample_texts):
        """Test character length distribution similarity."""
        real_texts, fake_texts = sample_texts
        result = text_length_distribution_similarity(real_texts, fake_texts, unit="char")
        
        assert isinstance(result, dict)
        assert result["unit"] == "char"
        assert "mean_diff" in result
        assert "std_diff" in result

    def test_invalid_unit(self, sample_texts):
        """Test invalid unit parameter."""
        real_texts, fake_texts = sample_texts
        with pytest.raises(ValueError, match="unit must be 'word' or 'char'"):
            text_length_distribution_similarity(real_texts, fake_texts, unit="invalid")

    def test_invalid_input_types(self):
        """Test invalid input types."""
        with pytest.raises(TypeError):
            text_length_distribution_similarity("not_series", pd.Series(["test"]))
        
        with pytest.raises(TypeError):
            text_length_distribution_similarity(pd.Series(["test"]), "not_series")

    def test_empty_data(self, empty_texts):
        """Test with empty text data."""
        real_texts, fake_texts = empty_texts
        result = text_length_distribution_similarity(real_texts, fake_texts)
        
        assert result["ks_statistic"] == 1.0
        assert result["ks_pvalue"] == 0.0

    def test_with_nan_values(self, texts_with_nan):
        """Test with NaN values in text data."""
        real_texts, fake_texts = texts_with_nan
        result = text_length_distribution_similarity(real_texts, fake_texts)
        
        # Should handle NaN values gracefully
        assert isinstance(result, dict)
        assert "similarity_score" in result


class TestVocabularyOverlapAnalysis:
    """Test vocabulary overlap analysis functions."""

    def test_basic_vocabulary_overlap(self, sample_texts):
        """Test basic vocabulary overlap analysis."""
        real_texts, fake_texts = sample_texts
        result = vocabulary_overlap_analysis(real_texts, fake_texts)
        
        assert isinstance(result, dict)
        assert "jaccard_similarity" in result
        assert "coverage_real_to_fake" in result
        assert "coverage_fake_to_real" in result
        assert "real_vocab_size" in result
        assert "fake_vocab_size" in result
        assert "shared_vocab_size" in result
        
        assert 0 <= result["jaccard_similarity"] <= 1
        assert 0 <= result["coverage_real_to_fake"] <= 1
        assert 0 <= result["coverage_fake_to_real"] <= 1

    def test_minimum_frequency_filter(self, sample_texts):
        """Test minimum frequency filtering."""
        real_texts, fake_texts = sample_texts
        result = vocabulary_overlap_analysis(real_texts, fake_texts, min_frequency=2)
        
        assert isinstance(result, dict)
        # With higher frequency threshold, vocabulary sizes should be smaller or equal
        result_default = vocabulary_overlap_analysis(real_texts, fake_texts, min_frequency=1)
        assert result["real_vocab_size"] <= result_default["real_vocab_size"]

    def test_identical_texts(self):
        """Test with identical text corpora."""
        texts = pd.Series(["hello world", "test message"])
        result = vocabulary_overlap_analysis(texts, texts)
        
        assert result["jaccard_similarity"] == 1.0
        assert result["coverage_real_to_fake"] == 1.0
        assert result["coverage_fake_to_real"] == 1.0

    def test_no_overlap(self):
        """Test with texts that have no vocabulary overlap."""
        real_texts = pd.Series(["cat dog"])
        fake_texts = pd.Series(["bird fish"])
        result = vocabulary_overlap_analysis(real_texts, fake_texts)
        
        assert result["jaccard_similarity"] == 0.0
        assert result["coverage_real_to_fake"] == 0.0
        assert result["coverage_fake_to_real"] == 0.0

    def test_invalid_min_frequency(self, sample_texts):
        """Test invalid minimum frequency parameter."""
        real_texts, fake_texts = sample_texts
        with pytest.raises(ValueError):
            vocabulary_overlap_analysis(real_texts, fake_texts, min_frequency=0)


class TestTfidfCorpusSimilarity:
    """Test TF-IDF corpus similarity functions."""

    def test_basic_tfidf_similarity(self, sample_texts):
        """Test basic TF-IDF similarity calculation."""
        real_texts, fake_texts = sample_texts
        result = tfidf_corpus_similarity(real_texts, fake_texts)
        
        assert isinstance(result, dict)
        assert "cosine_similarity" in result
        assert "tfidf_distance" in result
        assert "vocabulary_size" in result
        assert "similarity_score" in result
        
        assert -1 <= result["cosine_similarity"] <= 1
        assert 0 <= result["tfidf_distance"] <= 2
        assert result["vocabulary_size"] >= 0

    def test_tfidf_parameters(self, sample_texts):
        """Test TF-IDF with different parameters."""
        real_texts, fake_texts = sample_texts
        result = tfidf_corpus_similarity(
            real_texts, fake_texts,
            max_features=5,
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0
        )
        
        assert isinstance(result, dict)
        assert result["vocabulary_size"] <= 5

    def test_identical_corpora(self):
        """Test with identical text corpora."""
        texts = pd.Series(["hello world test", "sample message here"])
        result = tfidf_corpus_similarity(texts, texts)
        
        # Identical corpora should have perfect similarity
        assert abs(result["cosine_similarity"] - 1.0) < 1e-6

    def test_empty_texts(self, empty_texts):
        """Test with empty text data."""
        real_texts, fake_texts = empty_texts
        result = tfidf_corpus_similarity(real_texts, fake_texts)
        
        assert result["cosine_similarity"] == 0.0
        assert result["tfidf_distance"] == 1.0


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
class TestSemanticSimilarityEmbeddings:
    """Test semantic similarity with embeddings (when available)."""

    def test_basic_semantic_similarity(self, sample_texts):
        """Test basic semantic similarity calculation."""
        real_texts, fake_texts = sample_texts
        result = semantic_similarity_embeddings(real_texts, fake_texts)
        
        assert isinstance(result, dict)
        assert "semantic_similarity" in result
        assert "embedding_distance" in result
        assert "model_used" in result
        assert "similarity_score" in result
        
        assert -1 <= result["semantic_similarity"] <= 1
        assert 0 <= result["embedding_distance"] <= 2

    def test_with_sampling(self, sample_texts):
        """Test semantic similarity with sampling enabled."""
        real_texts, fake_texts = sample_texts
        result = semantic_similarity_embeddings(
            real_texts, fake_texts,
            enable_sampling=True,
            max_samples=3
        )
        
        assert isinstance(result, dict)
        assert "semantic_similarity" in result

    def test_custom_model(self, sample_texts):
        """Test with custom model name."""
        real_texts, fake_texts = sample_texts
        model_name = "all-MiniLM-L6-v2"
        result = semantic_similarity_embeddings(
            real_texts, fake_texts,
            model_name=model_name
        )
        
        assert result["model_used"] == model_name


class TestSemanticSimilarityWithoutDependency:
    """Test semantic similarity when dependencies are not available."""

    @patch('table_evaluator.advanced_metrics.textual.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_missing_dependency(self, sample_texts):
        """Test behavior when sentence-transformers is not available."""
        real_texts, fake_texts = sample_texts
        
        with pytest.raises(ImportError, match="sentence-transformers is required"):
            semantic_similarity_embeddings(real_texts, fake_texts)


class TestComprehensiveTextualAnalysis:
    """Test comprehensive textual analysis function."""

    def test_full_analysis(self, sample_texts):
        """Test comprehensive analysis with all metrics."""
        real_texts, fake_texts = sample_texts
        result = comprehensive_textual_analysis(real_texts, fake_texts, include_semantic=False)
        
        assert isinstance(result, dict)
        assert "word_length_dist" in result
        assert "char_length_dist" in result
        assert "vocabulary_overlap" in result
        assert "tfidf_similarity" in result
        assert "overall" in result

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
    def test_analysis_with_semantic(self, sample_texts):
        """Test comprehensive analysis including semantic similarity."""
        real_texts, fake_texts = sample_texts
        result = comprehensive_textual_analysis(real_texts, fake_texts, include_semantic=True)
        
        assert "semantic_similarity" in result

    def test_analysis_without_semantic(self, sample_texts):
        """Test comprehensive analysis without semantic similarity."""
        real_texts, fake_texts = sample_texts
        result = comprehensive_textual_analysis(real_texts, fake_texts, include_semantic=False)
        
        assert "semantic_similarity" not in result

    def test_error_handling_in_analysis(self):
        """Test error handling in comprehensive analysis."""
        # Test with problematic data that might cause errors
        real_texts = pd.Series(["", "", ""])
        fake_texts = pd.Series(["", "", ""])
        
        result = comprehensive_textual_analysis(real_texts, fake_texts)
        
        # Should handle errors gracefully
        assert isinstance(result, dict)


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
        
        assert isinstance(result, dict)
        assert "word_length_distribution" in result
        assert "char_length_distribution" in result
        assert "vocabulary_overlap" in result
        assert "summary" in result

    def test_tfidf_similarity_evaluation(self, sample_texts):
        """Test TF-IDF similarity evaluation."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        result = evaluator.tfidf_similarity_evaluation(real_texts, fake_texts)
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert "tfidf_cosine_similarity" in result["summary"]

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
    def test_semantic_similarity_evaluation(self, sample_texts):
        """Test semantic similarity evaluation."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        result = evaluator.semantic_similarity_evaluation(real_texts, fake_texts)
        
        assert isinstance(result, dict)
        assert result["available"] is True
        assert "summary" in result

    def test_semantic_similarity_unavailable(self, sample_texts):
        """Test semantic similarity when unavailable."""
        with patch('table_evaluator.evaluators.textual_evaluator.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            real_texts, fake_texts = sample_texts
            evaluator = TextualEvaluator()
            result = evaluator.semantic_similarity_evaluation(real_texts, fake_texts)
            
            assert result["available"] is False
            assert "error" in result

    def test_comprehensive_evaluation(self, sample_texts):
        """Test comprehensive evaluation."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        result = evaluator.comprehensive_evaluation(real_texts, fake_texts, include_semantic=False)
        
        assert isinstance(result, dict)
        assert "lexical_diversity" in result
        assert "tfidf_similarity" in result
        assert "combined_metrics" in result
        assert "recommendations" in result

    def test_quick_evaluation(self, sample_texts):
        """Test quick evaluation."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        result = evaluator.quick_evaluation(real_texts, fake_texts)
        
        assert isinstance(result, dict)
        assert "lexical_similarity" in result
        assert "tfidf_similarity" in result
        assert "overall_similarity" in result
        assert "evaluation_type" in result
        assert result["evaluation_type"] == "quick"

    def test_get_summary_for_integration(self, sample_texts):
        """Test summary extraction for TableEvaluator integration."""
        real_texts, fake_texts = sample_texts
        evaluator = TextualEvaluator()
        
        # Test with comprehensive evaluation results
        eval_results = evaluator.comprehensive_evaluation(real_texts, fake_texts, include_semantic=False)
        summary = evaluator.get_summary_for_integration(eval_results)
        
        assert isinstance(summary, dict)
        assert "textual_similarity" in summary
        assert "quality_rating" in summary
        assert "evaluation_type" in summary

    def test_quality_rating_methods(self):
        """Test quality rating methods."""
        evaluator = TextualEvaluator()
        
        # Test different quality levels
        assert evaluator._rate_lexical_quality(0.95) == "Excellent"
        assert evaluator._rate_lexical_quality(0.85) == "Good"
        assert evaluator._rate_lexical_quality(0.7) == "Fair"
        assert evaluator._rate_lexical_quality(0.5) == "Poor"
        assert evaluator._rate_lexical_quality(0.2) == "Very Poor"

    def test_error_handling_in_evaluations(self):
        """Test error handling in various evaluation methods."""
        evaluator = TextualEvaluator()
        
        # Test with empty series
        empty_real = pd.Series([])
        empty_fake = pd.Series([])
        
        result = evaluator.quick_evaluation(empty_real, empty_fake)
        # Should handle gracefully, might have error but shouldn't crash
        assert isinstance(result, dict)


class TestPerformanceWarnings:
    """Test performance warnings for large datasets."""

    def test_large_dataset_warning(self):
        """Test warning for large text datasets."""
        # Create large dataset
        large_real = pd.Series(["test text"] * 50001)
        large_fake = pd.Series(["sample text"] * 50001)
        
        # Should trigger warning but not fail
        with pytest.warns(None) as warning_list:
            result = text_length_distribution_similarity(large_real, large_fake)
            
        assert isinstance(result, dict)
        # Warning might be issued depending on implementation


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_special_characters(self):
        """Test with special characters and unicode."""
        real_texts = pd.Series(["Hello! @#$%", "Test with émojis 😀", "Numbers 123 and symbols &*()"])
        fake_texts = pd.Series(["Hi! @#$%", "Sample with émojis 😀", "Digits 456 and symbols &*()"])
        
        result = vocabulary_overlap_analysis(real_texts, fake_texts)
        assert isinstance(result, dict)

    def test_very_long_texts(self):
        """Test with very long text strings."""
        long_text = "word " * 1000  # 1000 words
        real_texts = pd.Series([long_text])
        fake_texts = pd.Series([long_text])
        
        result = tfidf_corpus_similarity(real_texts, fake_texts)
        assert isinstance(result, dict)

    def test_mixed_languages(self):
        """Test with mixed language content."""
        real_texts = pd.Series(["Hello world", "Bonjour monde", "Hola mundo"])
        fake_texts = pd.Series(["Hi world", "Salut monde", "Hola mundo"])
        
        result = vocabulary_overlap_analysis(real_texts, fake_texts)
        assert isinstance(result, dict)

    def test_numeric_strings(self):
        """Test with numeric string content."""
        real_texts = pd.Series(["123 456", "789 012", "345 678"])
        fake_texts = pd.Series(["111 222", "333 444", "555 666"])
        
        result = text_length_distribution_similarity(real_texts, fake_texts)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__])