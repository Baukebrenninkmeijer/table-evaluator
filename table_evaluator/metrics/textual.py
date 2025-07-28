"""Textual data comparison metrics for corpus-level analysis."""

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from table_evaluator.models.error_models import ErrorResult, create_error_result
from table_evaluator.models.textual_models import (
    ComprehensiveTextualAnalysisResult,
    LengthDistributionResult,
    SemanticRawResult,
    TfidfRawResult,
    VocabularyOverlapResult,
)

# Optional dependencies for advanced semantic analysis
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def _check_large_dataset_warning(real_texts: pd.Series, fake_texts: pd.Series, threshold: int = 100000) -> None:
    """
    Warn users about performance implications of large text datasets.

    Args:
        real_texts: Real text data series
        fake_texts: Synthetic text data series
        threshold: Row count threshold for warning
    """
    total_rows = len(real_texts) + len(fake_texts)
    if total_rows > threshold:
        logger.warning(
            f'Large text dataset detected ({total_rows:,} total rows). '
            f'Text analysis may be slow. Consider sampling your data for faster processing.'
        )


def _safe_tokenize(text: str, *, use_nltk: bool = True) -> list[str]:
    """
    Safely tokenize text with fallback options.

    Args:
        text: Input text string
        use_nltk: Whether to use NLTK tokenizer if available

    Returns:
        List of tokens
    """
    if pd.isna(text) or text == '':
        return []

    text = str(text).lower()

    if use_nltk and NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except Exception as e:
            # Fallback to simple split if NLTK fails
            logger.debug(f'NLTK tokenization failed for text "{text[:50]}...": {e}')

    # Simple fallback tokenization
    import re

    return re.findall(r'\b\w+\b', text)


def text_length_distribution_similarity(
    real_texts: pd.Series, fake_texts: pd.Series, unit: str = 'word'
) -> LengthDistributionResult:
    """
    Compare character/word length distributions between real and synthetic text.

    Args:
        real_texts: Series containing real text data
        fake_texts: Series containing synthetic text data
        unit: Unit of measurement ("word" or "char")

    Returns:
        Dictionary with distribution similarity metrics

    Raises:
        ValueError: If unit is not "word" or "char"
        TypeError: If inputs are not pandas Series
    """
    # Input validation
    if not isinstance(real_texts, pd.Series):
        raise TypeError('real_texts must be a pandas Series')
    if not isinstance(fake_texts, pd.Series):
        raise TypeError('fake_texts must be a pandas Series')
    if unit not in ['word', 'char']:
        raise ValueError("unit must be 'word' or 'char'")

    # Performance warning
    _check_large_dataset_warning(real_texts, fake_texts)

    # Clean data
    real_clean = real_texts.dropna().astype(str)
    fake_clean = fake_texts.dropna().astype(str)

    if len(real_clean) == 0 or len(fake_clean) == 0:
        logger.warning('Empty text data detected, returning default similarity')
        return LengthDistributionResult(
            similarity_score=0.0,
            ks_statistic=1.0,
            ks_p_value=0.0,
            mean_real=0.0,
            mean_fake=0.0,
            std_real=0.0,
            std_fake=0.0,
        )

    # Calculate lengths
    if unit == 'word':
        real_lengths = real_clean.apply(lambda x: len(_safe_tokenize(x)))
        fake_lengths = fake_clean.apply(lambda x: len(_safe_tokenize(x)))
    else:  # char
        real_lengths = real_clean.str.len()
        fake_lengths = fake_clean.str.len()

    # Remove zero lengths for better statistics
    real_lengths = real_lengths[real_lengths > 0]
    fake_lengths = fake_lengths[fake_lengths > 0]

    if len(real_lengths) == 0 or len(fake_lengths) == 0:
        logger.warning('No valid text lengths found, returning default similarity')
        return LengthDistributionResult(
            similarity_score=0.0,
            ks_statistic=1.0,
            ks_p_value=0.0,
            mean_real=0.0,
            mean_fake=0.0,
            std_real=0.0,
            std_fake=0.0,
        )

    # Kolmogorov-Smirnov test for distribution similarity
    ks_stat, ks_pvalue = stats.ks_2samp(real_lengths.values, fake_lengths.values)

    # Basic distribution statistics
    real_mean = real_lengths.mean()
    fake_mean = fake_lengths.mean()
    real_std = real_lengths.std()
    fake_std = fake_lengths.std()

    return LengthDistributionResult(
        similarity_score=float(1.0 - ks_stat),  # type: ignore
        ks_statistic=float(ks_stat),  # type: ignore
        ks_p_value=float(ks_pvalue),  # type: ignore
        mean_real=float(real_mean),
        mean_fake=float(fake_mean),
        std_real=float(real_std),
        std_fake=float(fake_std),
    )


def vocabulary_overlap_analysis(
    real_texts: pd.Series, fake_texts: pd.Series, min_frequency: int = 1
) -> VocabularyOverlapResult:
    """
    Analyze vocabulary overlap and diversity between real and synthetic text corpora.

    Args:
        real_texts: Series containing real text data
        fake_texts: Series containing synthetic text data
        min_frequency: Minimum word frequency to include in analysis

    Returns:
        Dictionary with vocabulary overlap metrics
    """
    # Input validation
    if not isinstance(real_texts, pd.Series):
        raise TypeError('real_texts must be a pandas Series')
    if not isinstance(fake_texts, pd.Series):
        raise TypeError('fake_texts must be a pandas Series')
    if not isinstance(min_frequency, int) or min_frequency < 1:
        raise ValueError('min_frequency must be a positive integer')

    # Performance warning
    _check_large_dataset_warning(real_texts, fake_texts)

    # Clean and tokenize
    real_clean = real_texts.dropna().astype(str)
    fake_clean = fake_texts.dropna().astype(str)

    if len(real_clean) == 0 or len(fake_clean) == 0:
        logger.warning('Empty text data detected, returning default metrics')
        return VocabularyOverlapResult(
            jaccard_similarity=0.0, vocab_diversity_ratio=0.0, shared_vocab_size=0, real_vocab_size=0, fake_vocab_size=0
        )

    # Build vocabularies
    real_vocab = {}
    fake_vocab = {}

    for text in real_clean:
        tokens = _safe_tokenize(text)
        for token in tokens:
            real_vocab[token] = real_vocab.get(token, 0) + 1

    for text in fake_clean:
        tokens = _safe_tokenize(text)
        for token in tokens:
            fake_vocab[token] = fake_vocab.get(token, 0) + 1

    # Filter by minimum frequency
    real_vocab_filtered = {k: v for k, v in real_vocab.items() if v >= min_frequency}
    fake_vocab_filtered = {k: v for k, v in fake_vocab.items() if v >= min_frequency}

    real_words = set(real_vocab_filtered.keys())
    fake_words = set(fake_vocab_filtered.keys())

    if len(real_words) == 0 and len(fake_words) == 0:
        return VocabularyOverlapResult(
            jaccard_similarity=1.0, vocab_diversity_ratio=1.0, shared_vocab_size=0, real_vocab_size=0, fake_vocab_size=0
        )

    if len(real_words) == 0 or len(fake_words) == 0:
        return VocabularyOverlapResult(
            jaccard_similarity=0.0,
            vocab_diversity_ratio=0.0 if len(real_words) == 0 else float('inf'),
            shared_vocab_size=0,
            real_vocab_size=len(real_words),
            fake_vocab_size=len(fake_words),
        )

    # Calculate metrics
    intersection = real_words.intersection(fake_words)
    union = real_words.union(fake_words)

    jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
    len(intersection) / len(real_words) if len(real_words) > 0 else 0.0
    len(intersection) / len(fake_words) if len(fake_words) > 0 else 0.0

    return VocabularyOverlapResult(
        jaccard_similarity=float(jaccard_similarity),
        vocab_diversity_ratio=float(len(fake_words) / len(real_words)) if len(real_words) > 0 else 0.0,
        shared_vocab_size=len(intersection),
        real_vocab_size=len(real_words),
        fake_vocab_size=len(fake_words),
    )


def tfidf_corpus_similarity(
    real_texts: pd.Series,
    fake_texts: pd.Series,
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfRawResult:
    """
    Calculate TF-IDF based corpus-level similarity using cosine distance.

    Args:
        real_texts: Series containing real text data
        fake_texts: Series containing synthetic text data
        max_features: Maximum number of features to extract
        ngram_range: Range of n-grams to extract
        min_df: Minimum document frequency
        max_df: Maximum document frequency (as fraction)

    Returns:
        Dictionary with TF-IDF similarity metrics
    """
    # Input validation
    if not isinstance(real_texts, pd.Series):
        raise TypeError('real_texts must be a pandas Series')
    if not isinstance(fake_texts, pd.Series):
        raise TypeError('fake_texts must be a pandas Series')

    # Performance warning
    _check_large_dataset_warning(real_texts, fake_texts)

    # Clean data
    real_clean = real_texts.dropna().astype(str).tolist()
    fake_clean = fake_texts.dropna().astype(str).tolist()

    if len(real_clean) == 0 or len(fake_clean) == 0:
        logger.warning('Empty text data detected, returning default similarity')
        return TfidfRawResult(
            cosine_similarity=0.0,
            tfidf_distance=1.0,
            vocabulary_size=0,
            real_corpus_norm=0.0,
            fake_corpus_norm=0.0,
        )

    # Combine corpora for consistent vocabulary
    all_texts = real_clean + fake_clean

    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b\w+\b',  # noqa: S106 # nosec
        )

        # Fit on all texts to ensure consistent vocabulary
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Split back into real and fake matrices
        real_tfidf = tfidf_matrix[: len(real_clean)]  # type: ignore
        fake_tfidf = tfidf_matrix[len(real_clean) :]  # type: ignore

        # Calculate corpus-level representations (mean of all documents)
        real_corpus_vec = np.mean(real_tfidf.toarray(), axis=0).reshape(1, -1)
        fake_corpus_vec = np.mean(fake_tfidf.toarray(), axis=0).reshape(1, -1)

        # Calculate cosine similarity
        cos_sim = cosine_similarity(real_corpus_vec, fake_corpus_vec)[0, 0]
        tfidf_distance = 1.0 - cos_sim

        # Calculate corpus norms for additional metrics
        real_corpus_norm = float(np.linalg.norm(real_corpus_vec))
        fake_corpus_norm = float(np.linalg.norm(fake_corpus_vec))

        return TfidfRawResult(
            cosine_similarity=float(cos_sim),
            tfidf_distance=float(tfidf_distance),
            vocabulary_size=len(vectorizer.vocabulary_),
            real_corpus_norm=real_corpus_norm,
            fake_corpus_norm=fake_corpus_norm,
        )

    except Exception as e:
        logger.error(f'TF-IDF similarity calculation failed: {e!s}')
        return TfidfRawResult(
            cosine_similarity=0.0,
            tfidf_distance=1.0,
            vocabulary_size=0,
            real_corpus_norm=0.0,
            fake_corpus_norm=0.0,
            error=str(e),
        )


def semantic_similarity_embeddings(
    real_texts: pd.Series,
    fake_texts: pd.Series,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    *,
    enable_sampling: bool = False,
    max_samples: int = 1000,
) -> SemanticRawResult:
    """
    Calculate semantic similarity using sentence transformer embeddings.

    This function requires sentence-transformers to be installed.

    Args:
        real_texts: Series containing real text data
        fake_texts: Series containing synthetic text data
        model_name: Name of the sentence transformer model to use
        batch_size: Batch size for embedding computation
        enable_sampling: Whether to sample large datasets for performance
        max_samples: Maximum samples per dataset when sampling is enabled

    Returns:
        Dictionary with semantic similarity metrics

    Raises:
        ImportError: If sentence-transformers is not available
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            'sentence-transformers is required for semantic similarity analysis. '
            'Install it with: pip install sentence-transformers'
        )

    # Input validation
    if not isinstance(real_texts, pd.Series):
        raise TypeError('real_texts must be a pandas Series')
    if not isinstance(fake_texts, pd.Series):
        raise TypeError('fake_texts must be a pandas Series')

    # Performance warning
    _check_large_dataset_warning(real_texts, fake_texts)

    # Clean data
    real_clean = real_texts.dropna().astype(str).tolist()
    fake_clean = fake_texts.dropna().astype(str).tolist()

    if len(real_clean) == 0 or len(fake_clean) == 0:
        logger.warning('Empty text data detected, returning default similarity')
        return SemanticRawResult(
            semantic_similarity=0.0,
            embedding_distance=1.0,
            model_name=model_name,
            samples_used=0,
        )

    # Sample data if enabled and datasets are large
    if enable_sampling and len(real_clean) > max_samples:
        rng = np.random.default_rng()
        real_clean = rng.choice(real_clean, max_samples, replace=False).tolist()
        logger.info(f'Sampled real texts from {len(real_texts)} to {max_samples}')

    if enable_sampling and len(fake_clean) > max_samples:
        rng = np.random.default_rng()
        fake_clean = rng.choice(fake_clean, max_samples, replace=False).tolist()
        logger.info(f'Sampled fake texts from {len(fake_texts)} to {max_samples}')

    try:
        # Load sentence transformer model
        model = SentenceTransformer(model_name)

        # Generate embeddings
        real_embeddings = model.encode(real_clean, batch_size=batch_size, show_progress_bar=False)
        fake_embeddings = model.encode(fake_clean, batch_size=batch_size, show_progress_bar=False)

        # Calculate corpus-level embeddings (mean of all document embeddings)
        real_corpus_embedding = np.mean(real_embeddings, axis=0).reshape(1, -1)
        fake_corpus_embedding = np.mean(fake_embeddings, axis=0).reshape(1, -1)

        # Calculate cosine similarity
        cos_sim = cosine_similarity(real_corpus_embedding, fake_corpus_embedding)[0, 0]
        embedding_distance = 1.0 - cos_sim

        return SemanticRawResult(
            semantic_similarity=float(cos_sim),
            embedding_distance=float(embedding_distance),
            model_name=model_name,
            samples_used=len(real_clean) + len(fake_clean),
        )

    except Exception as e:
        logger.error(f'Semantic similarity calculation failed: {e!s}')
        return SemanticRawResult(
            semantic_similarity=0.0,
            embedding_distance=1.0,
            model_name=model_name,
            samples_used=0,
            error=str(e),
        )


def _validate_textual_analysis_inputs(real_texts: pd.Series, fake_texts: pd.Series) -> None:
    """Validate inputs for textual analysis."""
    if not isinstance(real_texts, pd.Series):
        raise TypeError('real_texts must be a pandas Series')
    if not isinstance(fake_texts, pd.Series):
        raise TypeError('fake_texts must be a pandas Series')


def _run_length_distribution_analysis(
    real_texts: pd.Series, fake_texts: pd.Series
) -> tuple[LengthDistributionResult | ErrorResult, LengthDistributionResult | ErrorResult]:
    """Run word and character length distribution analysis."""
    # Word length distribution analysis
    try:
        word_length_dist = text_length_distribution_similarity(real_texts, fake_texts, unit='word')
    except Exception as e:
        logger.error(f'Word length distribution analysis failed: {e!s}')
        word_length_dist = create_error_result(
            e, 'text_length_distribution_similarity', args=(real_texts, fake_texts), kwargs={'unit': 'word'}
        )

    # Character length distribution analysis
    try:
        char_length_dist = text_length_distribution_similarity(real_texts, fake_texts, unit='char')
    except Exception as e:
        logger.error(f'Char length distribution analysis failed: {e!s}')
        char_length_dist = create_error_result(
            e, 'text_length_distribution_similarity', args=(real_texts, fake_texts), kwargs={'unit': 'char'}
        )

    return word_length_dist, char_length_dist


def _run_vocabulary_analysis(real_texts: pd.Series, fake_texts: pd.Series) -> VocabularyOverlapResult | ErrorResult:
    """Run vocabulary overlap analysis."""
    try:
        vocabulary_overlap = vocabulary_overlap_analysis(real_texts, fake_texts)
    except Exception as e:
        logger.error(f'Vocabulary analysis failed: {e!s}')
        vocabulary_overlap = create_error_result(e, 'vocabulary_overlap_analysis', args=(real_texts, fake_texts))

    return vocabulary_overlap


def _run_tfidf_analysis(real_texts: pd.Series, fake_texts: pd.Series) -> TfidfRawResult | ErrorResult:
    """Run TF-IDF similarity analysis."""
    try:
        tfidf_similarity = tfidf_corpus_similarity(real_texts, fake_texts)
    except Exception as e:
        logger.error(f'TF-IDF analysis failed: {e!s}')
        tfidf_similarity = create_error_result(e, 'tfidf_corpus_similarity', args=(real_texts, fake_texts))

    return tfidf_similarity


def _run_semantic_analysis(
    real_texts: pd.Series, fake_texts: pd.Series, *, include_semantic: bool, enable_sampling: bool, max_samples: int
) -> SemanticRawResult | ErrorResult | None:
    """Run semantic similarity analysis if requested and available."""
    if not include_semantic:
        return None

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return create_error_result(
            ImportError('sentence-transformers not available'),
            'semantic_similarity_embeddings',
            context={'message': 'Install with: pip install sentence-transformers'},
        )

    try:
        semantic_similarity = semantic_similarity_embeddings(
            real_texts, fake_texts, enable_sampling=enable_sampling, max_samples=max_samples
        )
    except Exception as e:
        logger.error(f'Semantic similarity analysis failed: {e!s}')
        semantic_similarity = create_error_result(
            e,
            'semantic_similarity_embeddings',
            args=(real_texts, fake_texts),
            kwargs={'enable_sampling': enable_sampling, 'max_samples': max_samples},
        )

    return semantic_similarity


def _calculate_overall_similarity_metrics(
    word_length_dist: LengthDistributionResult | ErrorResult,
    char_length_dist: LengthDistributionResult | ErrorResult,
    vocabulary_overlap: VocabularyOverlapResult | ErrorResult,
    tfidf_similarity: TfidfRawResult | ErrorResult,
    semantic_similarity: SemanticRawResult | ErrorResult | None,
) -> tuple[float, int]:
    """Calculate overall similarity score from all analysis results."""
    similarity_scores = []

    if not isinstance(word_length_dist, ErrorResult) and word_length_dist.similarity_score is not None:
        similarity_scores.append(word_length_dist.similarity_score)
    if not isinstance(char_length_dist, ErrorResult) and char_length_dist.similarity_score is not None:
        similarity_scores.append(char_length_dist.similarity_score)
    if not isinstance(vocabulary_overlap, ErrorResult) and vocabulary_overlap.jaccard_similarity is not None:
        similarity_scores.append(vocabulary_overlap.jaccard_similarity)
    if not isinstance(tfidf_similarity, ErrorResult) and tfidf_similarity.cosine_similarity is not None:
        similarity_scores.append(tfidf_similarity.cosine_similarity)
    if (
        semantic_similarity
        and not isinstance(semantic_similarity, ErrorResult)
        and semantic_similarity.semantic_similarity is not None
    ):
        similarity_scores.append(semantic_similarity.semantic_similarity)

    overall_similarity = float(np.mean(similarity_scores)) if similarity_scores else 0.0
    num_metrics = len(similarity_scores)

    return overall_similarity, num_metrics


def _determine_analysis_success(
    word_length_dist: LengthDistributionResult | ErrorResult,
    char_length_dist: LengthDistributionResult | ErrorResult,
    vocabulary_overlap: VocabularyOverlapResult | ErrorResult,
    tfidf_similarity: TfidfRawResult | ErrorResult,
) -> bool:
    """Determine if the analysis was successful based on component results."""
    from table_evaluator.models.error_models import ErrorResult

    return (
        not isinstance(word_length_dist, ErrorResult)
        and not isinstance(char_length_dist, ErrorResult)
        and not isinstance(vocabulary_overlap, ErrorResult)
        and not isinstance(tfidf_similarity, ErrorResult)
    )


def comprehensive_textual_analysis(
    real_texts: pd.Series,
    fake_texts: pd.Series,
    *,
    include_semantic: bool = True,
    enable_sampling: bool = False,
    max_samples: int = 1000,
) -> ComprehensiveTextualAnalysisResult:
    """
    Perform comprehensive textual analysis combining all available metrics.

    Args:
        real_texts: Series containing real text data
        fake_texts: Series containing synthetic text data
        include_semantic: Whether to include semantic similarity (requires sentence-transformers)
        enable_sampling: Whether to enable sampling for large datasets
        max_samples: Maximum samples per dataset when sampling is enabled

    Returns:
        ComprehensiveTextualAnalysisResult with comprehensive textual analysis results
    """
    # Input validation
    _validate_textual_analysis_inputs(real_texts, fake_texts)

    # Run all analysis components
    word_length_dist, char_length_dist = _run_length_distribution_analysis(real_texts, fake_texts)
    vocabulary_overlap = _run_vocabulary_analysis(real_texts, fake_texts)
    tfidf_similarity = _run_tfidf_analysis(real_texts, fake_texts)
    semantic_similarity = _run_semantic_analysis(
        real_texts,
        fake_texts,
        include_semantic=include_semantic,
        enable_sampling=enable_sampling,
        max_samples=max_samples,
    )

    # Calculate overall metrics
    overall_similarity, num_metrics = _calculate_overall_similarity_metrics(
        word_length_dist, char_length_dist, vocabulary_overlap, tfidf_similarity, semantic_similarity
    )

    # Determine success status
    success = _determine_analysis_success(word_length_dist, char_length_dist, vocabulary_overlap, tfidf_similarity)

    return ComprehensiveTextualAnalysisResult(
        word_length_dist=word_length_dist,
        char_length_dist=char_length_dist,
        vocabulary_overlap=vocabulary_overlap,
        tfidf_similarity=tfidf_similarity,
        semantic_similarity=semantic_similarity,
        overall_similarity=overall_similarity,
        num_metrics=num_metrics,
        success=success,
    )
