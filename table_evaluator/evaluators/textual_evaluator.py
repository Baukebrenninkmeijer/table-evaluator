"""Textual evaluation functionality for comparing real and synthetic text corpora."""

import numpy as np
import pandas as pd
from loguru import logger

from table_evaluator.metrics.textual import (
    SENTENCE_TRANSFORMERS_AVAILABLE,
    semantic_similarity_embeddings,
    text_length_distribution_similarity,
    tfidf_corpus_similarity,
    vocabulary_overlap_analysis,
)
from table_evaluator.models.textual_models import (
    CombinedMetrics,
    ComprehensiveTextualResult,
    ConsistencyMetrics,
    CorpusStatistics,
    LexicalDiversityResult,
    LexicalDiversitySummary,
    MetricWeights,
    QualityRatings,
    QuickTextualResult,
    SemanticConfigModel,
    SemanticSimilarityResult,
    SemanticSimilaritySummary,
    TextualEvaluationSummary,
    TfidfConfigModel,
    TfidfSimilarityResult,
    TfidfSimilaritySummary,
)


class TextualEvaluator:
    """Advanced textual evaluation for comparing real and synthetic text data."""

    def __init__(self, *, verbose: bool = False) -> None:
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
    ) -> LexicalDiversityResult:
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

        try:
            # Length distribution analysis
            word_length_distribution = text_length_distribution_similarity(real_texts, fake_texts, unit='word')
            char_length_distribution = text_length_distribution_similarity(real_texts, fake_texts, unit='char')

            # Vocabulary overlap analysis
            vocabulary_overlap = vocabulary_overlap_analysis(real_texts, fake_texts, min_frequency=min_frequency)

            # Summary metrics
            word_similarity = word_length_distribution.similarity_score
            char_similarity = char_length_distribution.similarity_score
            vocab_similarity = vocabulary_overlap.jaccard_similarity
            overall_similarity = float(np.mean([word_similarity, char_similarity, vocab_similarity]))

            summary = LexicalDiversitySummary(
                word_length_similarity=float(word_similarity),
                char_length_similarity=float(char_similarity),
                vocabulary_jaccard_similarity=float(vocab_similarity),
                overall_lexical_similarity=overall_similarity,
                quality_rating=self._rate_lexical_quality(overall_similarity),
            )

            return LexicalDiversityResult(
                word_length_distribution=word_length_distribution,
                char_length_distribution=char_length_distribution,
                vocabulary_overlap=vocabulary_overlap,
                summary=summary,
            )

        except Exception as e:
            logger.error(f'Error in lexical diversity evaluation: {e}')
            # Return a default result with error
            from table_evaluator.models.textual_models import LengthDistributionResult, VocabularyOverlapResult

            default_length = LengthDistributionResult(
                similarity_score=0.0,
                ks_statistic=1.0,
                ks_p_value=0.0,
                mean_real=0.0,
                mean_fake=0.0,
                std_real=0.0,
                std_fake=0.0,
            )
            default_vocab = VocabularyOverlapResult(
                jaccard_similarity=0.0,
                vocab_diversity_ratio=0.0,
                shared_vocab_size=0,
                real_vocab_size=0,
                fake_vocab_size=0,
            )
            default_summary = LexicalDiversitySummary(
                word_length_similarity=0.0,
                char_length_similarity=0.0,
                vocabulary_jaccard_similarity=0.0,
                overall_lexical_similarity=0.0,
                quality_rating='Error',
            )
            return LexicalDiversityResult(
                word_length_distribution=default_length,
                char_length_distribution=default_length,
                vocabulary_overlap=default_vocab,
                summary=default_summary,
                error=str(e),
            )

    def semantic_similarity_evaluation(
        self,
        real_texts: pd.Series,
        fake_texts: pd.Series,
        model_name: str = 'all-MiniLM-L6-v2',
        *,
        enable_sampling: bool = False,
        max_samples: int = 1000,
    ) -> SemanticSimilarityResult:
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
            return SemanticSimilarityResult(
                available=False,
                error='sentence-transformers not available. Install with: pip install sentence-transformers',
            )

        if self.verbose:
            print(f'Computing semantic similarity using {model_name}...')

        try:
            semantic_results = semantic_similarity_embeddings(
                real_texts,
                fake_texts,
                model_name=model_name,
                enable_sampling=enable_sampling,
                max_samples=max_samples,
            )

            # Enhanced analysis
            similarity_score = semantic_results.semantic_similarity
            embedding_distance = semantic_results.embedding_distance
            samples_used = semantic_results.samples_used

            summary = SemanticSimilaritySummary(
                semantic_similarity_score=float(similarity_score),
                embedding_distance=float(embedding_distance),
                quality_rating=self._rate_semantic_quality(similarity_score),
                model_used=model_name,
            )

            return SemanticSimilarityResult(
                available=True,
                semantic_similarity=float(similarity_score),
                embedding_distance=float(embedding_distance),
                model_name=model_name,
                samples_used=samples_used,
                summary=summary,
            )

        except Exception as e:
            logger.error(f'Error in semantic similarity evaluation: {e}')
            return SemanticSimilarityResult(available=True, error=str(e))

    def tfidf_similarity_evaluation(
        self,
        real_texts: pd.Series,
        fake_texts: pd.Series,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> TfidfSimilarityResult:
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

        try:
            tfidf_results = tfidf_corpus_similarity(
                real_texts, fake_texts, max_features=max_features, ngram_range=ngram_range
            )

            # Enhanced analysis
            similarity_score = tfidf_results.cosine_similarity
            tfidf_distance = tfidf_results.tfidf_distance
            vocabulary_size = tfidf_results.vocabulary_size
            real_corpus_norm = tfidf_results.real_corpus_norm
            fake_corpus_norm = tfidf_results.fake_corpus_norm

            summary = TfidfSimilaritySummary(
                tfidf_cosine_similarity=float(similarity_score),
                tfidf_distance=float(tfidf_distance),
                vocabulary_size=vocabulary_size,
                quality_rating=self._rate_tfidf_quality(similarity_score),
                features_used=max_features,
                ngram_range=ngram_range,
            )

            return TfidfSimilarityResult(
                cosine_similarity=float(similarity_score),
                tfidf_distance=float(tfidf_distance),
                vocabulary_size=vocabulary_size,
                real_corpus_norm=real_corpus_norm,
                fake_corpus_norm=fake_corpus_norm,
                summary=summary,
            )

        except Exception as e:
            logger.error(f'Error in TF-IDF similarity evaluation: {e}')
            # Return default result with error
            default_summary = TfidfSimilaritySummary(
                tfidf_cosine_similarity=0.0,
                tfidf_distance=1.0,
                vocabulary_size=0,
                quality_rating='Error',
                features_used=max_features,
                ngram_range=ngram_range,
            )
            return TfidfSimilarityResult(
                cosine_similarity=0.0,
                tfidf_distance=1.0,
                vocabulary_size=0,
                real_corpus_norm=0.0,
                fake_corpus_norm=0.0,
                summary=default_summary,
                error=str(e),
            )

    def comprehensive_evaluation(
        self,
        real_texts: pd.Series,
        fake_texts: pd.Series,
        *,
        include_semantic: bool = True,
        enable_sampling: bool = False,
        max_samples: int = 1000,
        tfidf_config: TfidfConfigModel | None = None,
        semantic_config: SemanticConfigModel | None = None,
    ) -> ComprehensiveTextualResult:
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
            tfidf_config = TfidfConfigModel()

        if semantic_config is None:
            semantic_config = SemanticConfigModel(
                enable_sampling=enable_sampling,
                max_samples=max_samples,
            )

        if self.verbose:
            print('Running comprehensive textual evaluation...')

        # Lexical diversity analysis
        try:
            lexical_diversity = self.lexical_diversity_evaluation(real_texts, fake_texts)
        except Exception as e:
            logger.error(f'Lexical diversity evaluation failed: {e}')
            # Create default result with error
            from table_evaluator.models.textual_models import (
                LengthDistributionResult,
                LexicalDiversitySummary,
                VocabularyOverlapResult,
            )

            default_length = LengthDistributionResult(
                similarity_score=0.0,
                ks_statistic=1.0,
                ks_p_value=0.0,
                mean_real=0.0,
                mean_fake=0.0,
                std_real=0.0,
                std_fake=0.0,
            )
            default_vocab = VocabularyOverlapResult(
                jaccard_similarity=0.0,
                vocab_diversity_ratio=0.0,
                shared_vocab_size=0,
                real_vocab_size=0,
                fake_vocab_size=0,
            )
            default_summary = LexicalDiversitySummary(
                word_length_similarity=0.0,
                char_length_similarity=0.0,
                vocabulary_jaccard_similarity=0.0,
                overall_lexical_similarity=0.0,
                quality_rating='Error',
            )
            lexical_diversity = LexicalDiversityResult(
                word_length_distribution=default_length,
                char_length_distribution=default_length,
                vocabulary_overlap=default_vocab,
                summary=default_summary,
                error=str(e),
            )

        # TF-IDF similarity analysis
        try:
            tfidf_similarity = self.tfidf_similarity_evaluation(
                real_texts, fake_texts, max_features=tfidf_config.max_features, ngram_range=tfidf_config.ngram_range
            )
        except Exception as e:
            logger.error(f'TF-IDF similarity evaluation failed: {e}')
            # Create default result with error
            default_tfidf_summary = TfidfSimilaritySummary(
                tfidf_cosine_similarity=0.0,
                tfidf_distance=1.0,
                vocabulary_size=0,
                quality_rating='Error',
                features_used=tfidf_config.max_features,
                ngram_range=tfidf_config.ngram_range,
            )
            tfidf_similarity = TfidfSimilarityResult(
                cosine_similarity=0.0,
                tfidf_distance=1.0,
                vocabulary_size=0,
                real_corpus_norm=0.0,
                fake_corpus_norm=0.0,
                summary=default_tfidf_summary,
                error=str(e),
            )

        # Semantic similarity analysis (if enabled and available)
        if include_semantic:
            try:
                semantic_similarity = self.semantic_similarity_evaluation(
                    real_texts,
                    fake_texts,
                    model_name=semantic_config.model_name,
                    enable_sampling=semantic_config.enable_sampling,
                    max_samples=semantic_config.max_samples,
                )
            except Exception as e:
                logger.error(f'Semantic similarity evaluation failed: {e}')
                semantic_similarity = SemanticSimilarityResult(available=True, error=str(e))
        else:
            semantic_similarity = SemanticSimilarityResult(available=False)

        # Combined analysis
        try:
            # Convert to dict format for backward compatibility with helper methods
            results_dict = {
                'lexical_diversity': lexical_diversity,
                'tfidf_similarity': tfidf_similarity,
                'semantic_similarity': semantic_similarity,
            }
            combined_metrics = self._combine_metrics(results_dict, include_semantic=include_semantic)
            recommendations = self._generate_recommendations(results_dict)
        except Exception as e:
            logger.error(f'Combined analysis failed: {e}')
            # Create default combined metrics
            combined_metrics = CombinedMetrics(
                overall_similarity=0.0,
                weights_used=MetricWeights(lexical=0.5, tfidf=0.5, semantic=0.0),
                quality_ratings=QualityRatings(lexical='Error', tfidf='Error', semantic='Error', overall='Error'),
                consistency_metrics=ConsistencyMetrics(score_std=0.0, score_range=0.0, consistency_rating='Unknown'),
                corpus_statistics=CorpusStatistics(),
                error=str(e),
            )
            recommendations = [f'Error in analysis: {e!s}']

        return ComprehensiveTextualResult(
            lexical_diversity=lexical_diversity,
            tfidf_similarity=tfidf_similarity,
            semantic_similarity=semantic_similarity,
            combined_metrics=combined_metrics,
            recommendations=recommendations,
        )

    def quick_evaluation(self, real_texts: pd.Series, fake_texts: pd.Series) -> QuickTextualResult:
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

        try:
            # Use only lexical diversity and TF-IDF (fast metrics)
            lexical_results = self.lexical_diversity_evaluation(real_texts, fake_texts)
            tfidf_results = self.tfidf_similarity_evaluation(real_texts, fake_texts)

            # Extract key metrics
            lexical_score = lexical_results.summary.overall_lexical_similarity
            tfidf_score = tfidf_results.summary.tfidf_cosine_similarity
            overall_similarity = float((lexical_score + tfidf_score) / 2)

            return QuickTextualResult(
                lexical_similarity=float(lexical_score),
                tfidf_similarity=float(tfidf_score),
                overall_similarity=overall_similarity,
                quality_rating=self._rate_overall_quality(overall_similarity),
                evaluation_type='quick',
                semantic_analysis_included=False,
            )

        except Exception as e:
            logger.error(f'Error in quick evaluation: {e}')
            return QuickTextualResult(
                lexical_similarity=0.0,
                tfidf_similarity=0.0,
                overall_similarity=0.0,
                quality_rating='Error',
                evaluation_type='quick',
                semantic_analysis_included=False,
                error=str(e),
            )

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

    def _combine_metrics(self, results: dict, *, include_semantic: bool) -> CombinedMetrics:
        """Combine results from different evaluation methods."""
        lexical_result = results.get('lexical_diversity')
        tfidf_result = results.get('tfidf_similarity')
        semantic_result = results.get('semantic_similarity')

        # Extract scores from Pydantic models
        lexical_score = (
            lexical_result.summary.overall_lexical_similarity if lexical_result and not lexical_result.error else 0
        )
        tfidf_score = tfidf_result.summary.tfidf_cosine_similarity if tfidf_result and not tfidf_result.error else 0
        semantic_score = (
            semantic_result.summary.semantic_similarity_score
            if (
                include_semantic
                and semantic_result
                and semantic_result.available
                and semantic_result.summary
                and not semantic_result.error
            )
            else None
        )

        # Combined similarity (weighted average)
        if include_semantic and semantic_score is not None:
            # Equal weighting: lexical, TF-IDF, semantic
            overall_similarity = (lexical_score + tfidf_score + semantic_score) / 3
            weights_used = MetricWeights(lexical=1 / 3, tfidf=1 / 3, semantic=1 / 3)
        else:
            # Without semantic: lexical and TF-IDF
            overall_similarity = (lexical_score + tfidf_score) / 2
            weights_used = MetricWeights(lexical=0.5, tfidf=0.5, semantic=0.0)

        # Quality ratings
        lexical_rating = (
            lexical_result.summary.quality_rating if lexical_result and not lexical_result.error else 'Unknown'
        )
        tfidf_rating = tfidf_result.summary.quality_rating if tfidf_result and not tfidf_result.error else 'Unknown'
        semantic_rating = (
            semantic_result.summary.quality_rating
            if (
                include_semantic
                and semantic_result
                and semantic_result.available
                and semantic_result.summary
                and not semantic_result.error
            )
            else 'N/A'
        )

        quality_ratings = QualityRatings(
            lexical=lexical_rating,
            tfidf=tfidf_rating,
            semantic=semantic_rating,
            overall=self._rate_overall_quality(overall_similarity),
        )

        # Consistency analysis
        scores = [lexical_score, tfidf_score]
        if include_semantic and semantic_score is not None:
            scores.append(semantic_score)

        score_std = float(np.std(scores))
        consistency_metrics = ConsistencyMetrics(
            score_std=score_std,
            score_range=float(max(scores) - min(scores)),
            consistency_rating='High' if score_std < 0.1 else 'Medium' if score_std < 0.2 else 'Low',
        )

        # Text corpus statistics
        corpus_statistics = self._extract_corpus_stats(results)

        return CombinedMetrics(
            overall_similarity=float(overall_similarity),
            weights_used=weights_used,
            quality_ratings=quality_ratings,
            consistency_metrics=consistency_metrics,
            corpus_statistics=corpus_statistics,
        )

    def _extract_corpus_stats(self, results: dict) -> CorpusStatistics:
        """Extract corpus-level statistics from evaluation results."""
        lexical_result = results.get('lexical_diversity')
        tfidf_result = results.get('tfidf_similarity')

        # From vocabulary analysis
        vocabulary_diversity_ratio = None
        shared_vocabulary_size = None
        real_vocabulary_size = None
        fake_vocabulary_size = None

        if lexical_result and not lexical_result.error:
            vocab_overlap = lexical_result.vocabulary_overlap
            vocabulary_diversity_ratio = vocab_overlap.vocab_diversity_ratio
            shared_vocabulary_size = vocab_overlap.shared_vocab_size
            real_vocabulary_size = vocab_overlap.real_vocab_size
            fake_vocabulary_size = vocab_overlap.fake_vocab_size

        # From TF-IDF analysis
        tfidf_vocabulary_size = None
        if tfidf_result and not tfidf_result.error:
            tfidf_vocabulary_size = tfidf_result.vocabulary_size

        return CorpusStatistics(
            vocabulary_diversity_ratio=vocabulary_diversity_ratio,
            shared_vocabulary_size=shared_vocabulary_size,
            real_vocabulary_size=real_vocabulary_size,
            fake_vocabulary_size=fake_vocabulary_size,
            tfidf_vocabulary_size=tfidf_vocabulary_size,
        )

    def _generate_recommendations(self, results: dict) -> list[str]:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []

        lexical_result = results.get('lexical_diversity')
        tfidf_result = results.get('tfidf_similarity')
        semantic_result = results.get('semantic_similarity')

        # Calculate overall similarity for recommendations
        scores = []
        if lexical_result and not lexical_result.error:
            scores.append(lexical_result.summary.overall_lexical_similarity)
        if tfidf_result and not tfidf_result.error:
            scores.append(tfidf_result.summary.tfidf_cosine_similarity)
        if semantic_result and semantic_result.available and semantic_result.summary and not semantic_result.error:
            scores.append(semantic_result.summary.semantic_similarity_score)

        overall_similarity = np.mean(scores) if scores else 0.0

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
        if lexical_result and not lexical_result.error:
            vocab_overlap = lexical_result.vocabulary_overlap
            jaccard_sim = vocab_overlap.jaccard_similarity

            if jaccard_sim < 0.3:
                recommendations.append(
                    'Low vocabulary overlap detected. The synthetic text may not capture '
                    'the full vocabulary diversity of the real text.'
                )

            vocab_diversity = vocab_overlap.vocab_diversity_ratio
            if vocab_diversity < 0.5 or vocab_diversity > 2.0:
                recommendations.append(
                    'Vocabulary diversity ratio is significantly different from real data. '
                    "Consider adjusting the text generation model's vocabulary breadth."
                )

        # TF-IDF specific recommendations
        if tfidf_result and not tfidf_result.error:
            tfidf_sim = tfidf_result.cosine_similarity
            if tfidf_sim < 0.5:
                recommendations.append(
                    'Low TF-IDF similarity suggests differences in word usage patterns. '
                    "Review the model's ability to capture term frequency distributions."
                )

        # Semantic analysis recommendations
        if semantic_result and semantic_result.available and semantic_result.summary and not semantic_result.error:
            semantic_sim = semantic_result.summary.semantic_similarity_score
            if semantic_sim < 0.7:
                recommendations.append(
                    'Low semantic similarity indicates the synthetic text may not capture '
                    'the meaning and context of the real text effectively.'
                )

        # Consistency recommendations based on score variation
        if len(scores) > 1:
            score_std = float(np.std(scores))
            if score_std > 0.2:  # High variance
                recommendations.append(
                    'Inconsistent results across different metrics suggest the synthetic text has mixed quality. '
                    'Focus on the lowest-scoring metrics for improvement.'
                )

        return recommendations

    def get_summary_for_integration(
        self, evaluation_results: ComprehensiveTextualResult | QuickTextualResult
    ) -> TextualEvaluationSummary:
        """
        Extract key metrics for integration with TableEvaluator.

        Args:
            evaluation_results: Results from comprehensive_evaluation or quick_evaluation

        Returns:
            Dictionary with key metrics for main evaluation integration
        """
        if isinstance(evaluation_results, ComprehensiveTextualResult):
            if evaluation_results.combined_metrics.error:
                return TextualEvaluationSummary(
                    textual_similarity=0.0,
                    quality_rating='Error',
                    evaluation_type='comprehensive',
                    error=evaluation_results.combined_metrics.error,
                )

            return TextualEvaluationSummary(
                textual_similarity=evaluation_results.combined_metrics.overall_similarity,
                quality_rating=evaluation_results.combined_metrics.quality_ratings.overall,
                evaluation_type='comprehensive',
            )

        if isinstance(evaluation_results, QuickTextualResult):
            if evaluation_results.error:
                return TextualEvaluationSummary(
                    textual_similarity=0.0,
                    quality_rating='Error',
                    evaluation_type='quick',
                    error=evaluation_results.error,
                )

            return TextualEvaluationSummary(
                textual_similarity=evaluation_results.overall_similarity,
                quality_rating=evaluation_results.quality_rating,
                evaluation_type=evaluation_results.evaluation_type,
            )

        # Fallback for unexpected input
        return TextualEvaluationSummary(
            textual_similarity=0.0,
            quality_rating='Error',
            evaluation_type='unknown',
            error='Unexpected evaluation result type',
        )
