"""Textual evaluation functionality for comparing real and synthetic text corpora."""

import logging

import numpy as np
import pandas as pd

from table_evaluator.advanced_metrics.textual import (
    SENTENCE_TRANSFORMERS_AVAILABLE,
    semantic_similarity_embeddings,
    text_length_distribution_similarity,
    tfidf_corpus_similarity,
    vocabulary_overlap_analysis,
)

logger = logging.getLogger(__name__)


class TextualEvaluator:
    """Advanced textual evaluation for comparing real and synthetic text data."""

    def __init__(self, verbose: bool = False):
        """
        Initialize the textual evaluator.

        Args:
            verbose: Whether to print detailed output during evaluation
        """
        self.verbose = verbose

    def lexical_diversity_evaluation(
        self,
        real_texts: pd.Series,
        fake_texts: pd.Series,
        min_frequency: int = 1,
    ) -> dict:
        """
        Evaluate lexical diversity including length distributions and vocabulary overlap.

        Args:
            real_texts: Series containing real text data
            fake_texts: Series containing synthetic text data
            min_frequency: Minimum word frequency to include in vocabulary analysis

        Returns:
            Dictionary containing lexical diversity analysis results
        """
        if self.verbose:
            print('Computing lexical diversity metrics...')

        results = {}

        try:
            # Length distribution analysis
            results['word_length_distribution'] = text_length_distribution_similarity(
                real_texts, fake_texts, unit='word'
            )
            results['char_length_distribution'] = text_length_distribution_similarity(
                real_texts, fake_texts, unit='char'
            )

            # Vocabulary overlap analysis
            results['vocabulary_overlap'] = vocabulary_overlap_analysis(
                real_texts, fake_texts, min_frequency=min_frequency
            )

            # Summary metrics
            word_similarity = results['word_length_distribution'].get('similarity_score', 0)
            char_similarity = results['char_length_distribution'].get('similarity_score', 0)
            vocab_similarity = results['vocabulary_overlap'].get('jaccard_similarity', 0)

            results['summary'] = {
                'word_length_similarity': float(word_similarity),
                'char_length_similarity': float(char_similarity),
                'vocabulary_jaccard_similarity': float(vocab_similarity),
                'overall_lexical_similarity': float(np.mean([word_similarity, char_similarity, vocab_similarity])),
                'quality_rating': self._rate_lexical_quality(
                    np.mean([word_similarity, char_similarity, vocab_similarity])
                ),
            }

        except Exception as e:
            logger.error(f'Error in lexical diversity evaluation: {e}')
            results['error'] = str(e)

        return results

    def semantic_similarity_evaluation(
        self,
        real_texts: pd.Series,
        fake_texts: pd.Series,
        model_name: str = 'all-MiniLM-L6-v2',
        enable_sampling: bool = False,
        max_samples: int = 1000,
    ) -> dict:
        """
        Evaluate semantic similarity using transformer-based embeddings.

        Args:
            real_texts: Series containing real text data
            fake_texts: Series containing synthetic text data
            model_name: Name of the sentence transformer model to use
            enable_sampling: Whether to sample large datasets for performance
            max_samples: Maximum samples per dataset when sampling is enabled

        Returns:
            Dictionary containing semantic similarity analysis results
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return {
                'error': 'sentence-transformers not available. Install with: pip install sentence-transformers',
                'available': False,
            }

        if self.verbose:
            print(f'Computing semantic similarity using {model_name}...')

        results = {'available': True}

        try:
            semantic_results = semantic_similarity_embeddings(
                real_texts,
                fake_texts,
                model_name=model_name,
                enable_sampling=enable_sampling,
                max_samples=max_samples,
            )

            results.update(semantic_results)

            # Enhanced analysis
            similarity_score = semantic_results.get('semantic_similarity', 0)
            results['summary'] = {
                'semantic_similarity_score': float(similarity_score),
                'embedding_distance': float(semantic_results.get('embedding_distance', 1.0)),
                'quality_rating': self._rate_semantic_quality(similarity_score),
                'model_used': model_name,
            }

        except Exception as e:
            logger.error(f'Error in semantic similarity evaluation: {e}')
            results['error'] = str(e)

        return results

    def tfidf_similarity_evaluation(
        self,
        real_texts: pd.Series,
        fake_texts: pd.Series,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
    ) -> dict:
        """
        Evaluate TF-IDF based corpus similarity.

        Provides a lightweight approach to measure textual similarity without
        requiring heavy transformer models.

        Args:
            real_texts: Series containing real text data
            fake_texts: Series containing synthetic text data
            max_features: Maximum number of TF-IDF features to extract
            ngram_range: Range of n-grams to consider

        Returns:
            Dictionary containing TF-IDF similarity analysis results
        """
        if self.verbose:
            print('Computing TF-IDF corpus similarity...')

        results = {}

        try:
            tfidf_results = tfidf_corpus_similarity(
                real_texts, fake_texts, max_features=max_features, ngram_range=ngram_range
            )

            results.update(tfidf_results)

            # Enhanced analysis
            similarity_score = tfidf_results.get('cosine_similarity', 0)
            results['summary'] = {
                'tfidf_cosine_similarity': float(similarity_score),
                'tfidf_distance': float(tfidf_results.get('tfidf_distance', 1.0)),
                'vocabulary_size': tfidf_results.get('vocabulary_size', 0),
                'quality_rating': self._rate_tfidf_quality(similarity_score),
                'features_used': max_features,
                'ngram_range': ngram_range,
            }

        except Exception as e:
            logger.error(f'Error in TF-IDF similarity evaluation: {e}')
            results['error'] = str(e)

        return results

    def comprehensive_evaluation(
        self,
        real_texts: pd.Series,
        fake_texts: pd.Series,
        include_semantic: bool = True,
        enable_sampling: bool = False,
        max_samples: int = 1000,
        tfidf_config: dict | None = None,
        semantic_config: dict | None = None,
    ) -> dict:
        """
        Run comprehensive textual evaluation combining all available metrics.

        Args:
            real_texts: Series containing real text data
            fake_texts: Series containing synthetic text data
            include_semantic: Whether to include semantic similarity analysis
            enable_sampling: Whether to enable sampling for large datasets
            max_samples: Maximum samples per dataset when sampling is enabled
            tfidf_config: Configuration for TF-IDF evaluation
            semantic_config: Configuration for semantic evaluation

        Returns:
            Dictionary with complete textual analysis
        """
        if tfidf_config is None:
            tfidf_config = {'max_features': 10000, 'ngram_range': (1, 2)}

        if semantic_config is None:
            semantic_config = {
                'model_name': 'all-MiniLM-L6-v2',
                'enable_sampling': enable_sampling,
                'max_samples': max_samples,
            }

        if self.verbose:
            print('Running comprehensive textual evaluation...')

        results = {
            'lexical_diversity': {},
            'tfidf_similarity': {},
            'semantic_similarity': {},
            'combined_metrics': {},
            'recommendations': [],
        }

        # Lexical diversity analysis
        try:
            results['lexical_diversity'] = self.lexical_diversity_evaluation(real_texts, fake_texts)
        except Exception as e:
            logger.error(f'Lexical diversity evaluation failed: {e}')
            results['lexical_diversity'] = {'error': str(e)}

        # TF-IDF similarity analysis
        try:
            results['tfidf_similarity'] = self.tfidf_similarity_evaluation(real_texts, fake_texts, **tfidf_config)
        except Exception as e:
            logger.error(f'TF-IDF similarity evaluation failed: {e}')
            results['tfidf_similarity'] = {'error': str(e)}

        # Semantic similarity analysis (if enabled and available)
        if include_semantic:
            try:
                results['semantic_similarity'] = self.semantic_similarity_evaluation(
                    real_texts, fake_texts, **semantic_config
                )
            except Exception as e:
                logger.error(f'Semantic similarity evaluation failed: {e}')
                results['semantic_similarity'] = {'error': str(e)}

        # Combined analysis
        try:
            results['combined_metrics'] = self._combine_metrics(results, include_semantic)
            results['recommendations'] = self._generate_recommendations(results)
        except Exception as e:
            logger.error(f'Combined analysis failed: {e}')
            results['combined_metrics'] = {'error': str(e)}

        return results

    def quick_evaluation(self, real_texts: pd.Series, fake_texts: pd.Series) -> dict:
        """
        Perform quick textual evaluation using only lightweight metrics.

        Useful for large datasets or when quick feedback is needed.

        Args:
            real_texts: Series containing real text data
            fake_texts: Series containing synthetic text data

        Returns:
            Dictionary with quick evaluation results
        """
        if self.verbose:
            print('Running quick textual evaluation...')

        results = {}

        try:
            # Use only lexical diversity and TF-IDF (fast metrics)
            lexical_results = self.lexical_diversity_evaluation(real_texts, fake_texts)
            tfidf_results = self.tfidf_similarity_evaluation(real_texts, fake_texts)

            # Extract key metrics
            lexical_score = lexical_results.get('summary', {}).get('overall_lexical_similarity', 0)
            tfidf_score = tfidf_results.get('summary', {}).get('tfidf_cosine_similarity', 0)

            results = {
                'lexical_similarity': float(lexical_score),
                'tfidf_similarity': float(tfidf_score),
                'overall_similarity': float((lexical_score + tfidf_score) / 2),
                'quality_rating': self._rate_overall_quality((lexical_score + tfidf_score) / 2),
                'evaluation_type': 'quick',
                'semantic_analysis_included': False,
            }

        except Exception as e:
            logger.error(f'Error in quick evaluation: {e}')
            results['error'] = str(e)

        return results

    def _rate_lexical_quality(self, similarity_score: float) -> str:
        """Rate lexical quality based on similarity score."""
        if similarity_score >= 0.9:
            return 'Excellent'
        if similarity_score >= 0.8:
            return 'Good'
        if similarity_score >= 0.6:
            return 'Fair'
        if similarity_score >= 0.4:
            return 'Poor'
        return 'Very Poor'

    def _rate_semantic_quality(self, similarity_score: float) -> str:
        """Rate semantic quality based on similarity score."""
        if similarity_score >= 0.95:
            return 'Excellent'
        if similarity_score >= 0.85:
            return 'Good'
        if similarity_score >= 0.7:
            return 'Fair'
        if similarity_score >= 0.5:
            return 'Poor'
        return 'Very Poor'

    def _rate_tfidf_quality(self, similarity_score: float) -> str:
        """Rate TF-IDF quality based on cosine similarity score."""
        if similarity_score >= 0.9:
            return 'Excellent'
        if similarity_score >= 0.8:
            return 'Good'
        if similarity_score >= 0.6:
            return 'Fair'
        if similarity_score >= 0.4:
            return 'Poor'
        return 'Very Poor'

    def _rate_overall_quality(self, combined_score: float) -> str:
        """Rate overall textual quality based on combined metrics."""
        if combined_score >= 0.85:
            return 'Excellent'
        if combined_score >= 0.75:
            return 'Good'
        if combined_score >= 0.6:
            return 'Fair'
        if combined_score >= 0.4:
            return 'Poor'
        return 'Very Poor'

    def _combine_metrics(self, results: dict, include_semantic: bool) -> dict:
        """Combine results from different evaluation methods."""
        combined = {}

        # Extract summary scores
        lexical_summary = results.get('lexical_diversity', {}).get('summary', {})
        tfidf_summary = results.get('tfidf_similarity', {}).get('summary', {})
        semantic_summary = results.get('semantic_similarity', {}).get('summary', {})

        # Individual metric scores
        lexical_score = lexical_summary.get('overall_lexical_similarity', 0)
        tfidf_score = tfidf_summary.get('tfidf_cosine_similarity', 0)
        semantic_score = semantic_summary.get('semantic_similarity_score', 0) if include_semantic else None

        # Combined similarity (weighted average)
        if include_semantic and semantic_score is not None and 'error' not in results.get('semantic_similarity', {}):
            # Equal weighting: lexical, TF-IDF, semantic
            overall_similarity = (lexical_score + tfidf_score + semantic_score) / 3
            weights_used = {'lexical': 1 / 3, 'tfidf': 1 / 3, 'semantic': 1 / 3}
        else:
            # Without semantic: lexical and TF-IDF
            overall_similarity = (lexical_score + tfidf_score) / 2
            weights_used = {'lexical': 0.5, 'tfidf': 0.5, 'semantic': 0.0}

        combined['overall_similarity'] = float(overall_similarity)
        combined['weights_used'] = weights_used

        # Quality consensus
        lexical_rating = lexical_summary.get('quality_rating', 'Unknown')
        tfidf_rating = tfidf_summary.get('quality_rating', 'Unknown')
        semantic_rating = semantic_summary.get('quality_rating', 'Unknown') if include_semantic else 'N/A'

        combined['quality_ratings'] = {
            'lexical': lexical_rating,
            'tfidf': tfidf_rating,
            'semantic': semantic_rating,
            'overall': self._rate_overall_quality(overall_similarity),
        }

        # Consistency analysis
        scores = [lexical_score, tfidf_score]
        if include_semantic and semantic_score is not None:
            scores.append(semantic_score)

        combined['consistency_metrics'] = {
            'score_std': float(np.std(scores)),
            'score_range': float(max(scores) - min(scores)),
            'consistency_rating': 'High' if np.std(scores) < 0.1 else 'Medium' if np.std(scores) < 0.2 else 'Low',
        }

        # Text corpus statistics
        combined['corpus_statistics'] = self._extract_corpus_stats(results)

        return combined

    def _extract_corpus_stats(self, results: dict) -> dict:
        """Extract corpus-level statistics from evaluation results."""
        stats = {}

        # From vocabulary analysis
        vocab_results = results.get('lexical_diversity', {}).get('vocabulary_overlap', {})
        if vocab_results and 'error' not in vocab_results:
            stats['vocabulary_diversity_ratio'] = vocab_results.get('vocab_diversity_ratio', 0)
            stats['shared_vocabulary_size'] = vocab_results.get('shared_vocab_size', 0)
            stats['real_vocabulary_size'] = vocab_results.get('real_vocab_size', 0)
            stats['fake_vocabulary_size'] = vocab_results.get('fake_vocab_size', 0)

        # From TF-IDF analysis
        tfidf_results = results.get('tfidf_similarity', {})
        if tfidf_results and 'error' not in tfidf_results:
            stats['tfidf_vocabulary_size'] = tfidf_results.get('vocabulary_size', 0)

        return stats

    def _generate_recommendations(self, results: dict) -> list[str]:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []

        combined = results.get('combined_metrics', {})
        overall_similarity = combined.get('overall_similarity', 0)

        # Overall similarity recommendations
        if overall_similarity < 0.4:
            recommendations.append(
                'Low overall textual similarity detected. Consider significantly improving the text generation model.'
            )
        elif overall_similarity < 0.7:
            recommendations.append('Moderate textual similarity achieved. Fine-tuning the model may improve results.')
        else:
            recommendations.append('Good textual similarity achieved. The synthetic text quality is acceptable.')

        # Specific metric recommendations
        lexical_results = results.get('lexical_diversity', {})
        if lexical_results and 'error' not in lexical_results:
            vocab_overlap = lexical_results.get('vocabulary_overlap', {})
            jaccard_sim = vocab_overlap.get('jaccard_similarity', 0)

            if jaccard_sim < 0.3:
                recommendations.append(
                    'Low vocabulary overlap detected. The synthetic text may not capture the full vocabulary diversity of the real text.'
                )

            vocab_diversity = vocab_overlap.get('vocab_diversity_ratio', 1.0)
            if vocab_diversity < 0.5 or vocab_diversity > 2.0:
                recommendations.append(
                    'Vocabulary diversity ratio is significantly different from real data. '
                    "Consider adjusting the text generation model's vocabulary breadth."
                )

        # TF-IDF specific recommendations
        tfidf_results = results.get('tfidf_similarity', {})
        if tfidf_results and 'error' not in tfidf_results:
            tfidf_sim = tfidf_results.get('cosine_similarity', 0)
            if tfidf_sim < 0.5:
                recommendations.append(
                    'Low TF-IDF similarity suggests differences in word usage patterns. '
                    "Review the model's ability to capture term frequency distributions."
                )

        # Semantic analysis recommendations
        semantic_results = results.get('semantic_similarity', {})
        if semantic_results and 'error' not in semantic_results and semantic_results.get('available', False):
            semantic_sim = semantic_results.get('semantic_similarity', 0)
            if semantic_sim < 0.7:
                recommendations.append(
                    'Low semantic similarity indicates the synthetic text may not capture the meaning and context of the real text effectively.'
                )

        # Consistency recommendations
        consistency = combined.get('consistency_metrics', {})
        if consistency.get('consistency_rating') == 'Low':
            recommendations.append(
                'Inconsistent results across different metrics suggest the synthetic text has mixed quality. '
                'Focus on the lowest-scoring metrics for improvement.'
            )

        return recommendations

    def get_summary_for_integration(self, evaluation_results: dict) -> dict:
        """
        Extract key metrics for integration with TableEvaluator.

        Args:
            evaluation_results: Results from comprehensive_evaluation or quick_evaluation

        Returns:
            Dictionary with key metrics for main evaluation integration
        """
        if 'error' in evaluation_results:
            return {'textual_similarity': 0.0, 'error': evaluation_results['error']}

        # Extract overall similarity score
        if 'combined_metrics' in evaluation_results:
            overall_similarity = evaluation_results['combined_metrics'].get('overall_similarity', 0.0)
            quality_rating = evaluation_results['combined_metrics']['quality_ratings'].get('overall', 'Unknown')
        else:
            # From quick evaluation
            overall_similarity = evaluation_results.get('overall_similarity', 0.0)
            quality_rating = evaluation_results.get('quality_rating', 'Unknown')

        return {
            'textual_similarity': float(overall_similarity),
            'quality_rating': quality_rating,
            'evaluation_type': evaluation_results.get('evaluation_type', 'comprehensive'),
        }
