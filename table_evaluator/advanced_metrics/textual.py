"""Textual data comparison metrics for corpus-level analysis."""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from table_evaluator.models.textual_models import (
    LengthDistributionResult,
    VocabularyOverlapResult,
    TfidfSimilarityResult,
)

# Optional dependencies for advanced semantic analysis
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import nltk
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
            f"Large text dataset detected ({total_rows:,} total rows). "
            f"Text analysis may be slow. Consider sampling your data for faster processing."
        )


def _safe_tokenize(text: str, use_nltk: bool = True) -> list[str]:
    """
    Safely tokenize text with fallback options.
    
    Args:
        text: Input text string
        use_nltk: Whether to use NLTK tokenizer if available
        
    Returns:
        List of tokens
    """
    if pd.isna(text) or text == "":
        return []
    
    text = str(text).lower()
    
    if use_nltk and NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except Exception:
            # Fallback to simple split if NLTK fails
            pass
    
    # Simple fallback tokenization
    import re
    return re.findall(r'\b\w+\b', text)


def text_length_distribution_similarity(
    real_texts: pd.Series, fake_texts: pd.Series, unit: str = "word"
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
        raise TypeError("real_texts must be a pandas Series")
    if not isinstance(fake_texts, pd.Series):
        raise TypeError("fake_texts must be a pandas Series")
    if unit not in ["word", "char"]:
        raise ValueError("unit must be 'word' or 'char'")
    
    # Performance warning
    _check_large_dataset_warning(real_texts, fake_texts)
    
    # Clean data
    real_clean = real_texts.dropna().astype(str)
    fake_clean = fake_texts.dropna().astype(str)
    
    if len(real_clean) == 0 or len(fake_clean) == 0:
        logger.warning("Empty text data detected, returning default similarity")
        return LengthDistributionResult(
            similarity_score=0.0,
            ks_statistic=1.0,
            ks_p_value=0.0,
            mean_real=0.0,
            mean_fake=0.0,
            std_real=0.0,
            std_fake=0.0
        )
    
    # Calculate lengths
    if unit == "word":
        real_lengths = real_clean.apply(lambda x: len(_safe_tokenize(x)))
        fake_lengths = fake_clean.apply(lambda x: len(_safe_tokenize(x)))
    else:  # char
        real_lengths = real_clean.str.len()
        fake_lengths = fake_clean.str.len()
    
    # Remove zero lengths for better statistics
    real_lengths = real_lengths[real_lengths > 0]
    fake_lengths = fake_lengths[fake_lengths > 0]
    
    if len(real_lengths) == 0 or len(fake_lengths) == 0:
        logger.warning("No valid text lengths found, returning default similarity")
        return LengthDistributionResult(
            similarity_score=0.0,
            ks_statistic=1.0,
            ks_p_value=0.0,
            mean_real=0.0,
            mean_fake=0.0,
            std_real=0.0,
            std_fake=0.0
        )
    
    # Kolmogorov-Smirnov test for distribution similarity
    ks_stat, ks_pvalue = stats.ks_2samp(real_lengths.values, fake_lengths.values)
    
    # Basic distribution statistics
    real_mean = real_lengths.mean()
    fake_mean = fake_lengths.mean()
    real_std = real_lengths.std()
    fake_std = fake_lengths.std()
    
    return LengthDistributionResult(
        similarity_score=float(1.0 - ks_stat),
        ks_statistic=float(ks_stat),
        ks_p_value=float(ks_pvalue),
        mean_real=float(real_mean),
        mean_fake=float(fake_mean),
        std_real=float(real_std),
        std_fake=float(fake_std)
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
        raise TypeError("real_texts must be a pandas Series")
    if not isinstance(fake_texts, pd.Series):
        raise TypeError("fake_texts must be a pandas Series")
    if not isinstance(min_frequency, int) or min_frequency < 1:
        raise ValueError("min_frequency must be a positive integer")
    
    # Performance warning
    _check_large_dataset_warning(real_texts, fake_texts)
    
    # Clean and tokenize
    real_clean = real_texts.dropna().astype(str)
    fake_clean = fake_texts.dropna().astype(str)
    
    if len(real_clean) == 0 or len(fake_clean) == 0:
        logger.warning("Empty text data detected, returning default metrics")
        return VocabularyOverlapResult(
            jaccard_similarity=0.0,
            vocab_diversity_ratio=0.0,
            shared_vocab_size=0,
            real_vocab_size=0,
            fake_vocab_size=0
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
            jaccard_similarity=1.0,
            vocab_diversity_ratio=1.0,
            shared_vocab_size=0,
            real_vocab_size=0,
            fake_vocab_size=0
        )
    
    if len(real_words) == 0 or len(fake_words) == 0:
        return VocabularyOverlapResult(
            jaccard_similarity=0.0,
            vocab_diversity_ratio=0.0 if len(real_words) == 0 else float('inf'),
            shared_vocab_size=0,
            real_vocab_size=len(real_words),
            fake_vocab_size=len(fake_words)
        )
    
    # Calculate metrics
    intersection = real_words.intersection(fake_words)
    union = real_words.union(fake_words)
    
    jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
    coverage_real_to_fake = len(intersection) / len(real_words) if len(real_words) > 0 else 0.0
    coverage_fake_to_real = len(intersection) / len(fake_words) if len(fake_words) > 0 else 0.0
    
    return VocabularyOverlapResult(
        jaccard_similarity=float(jaccard_similarity),
        vocab_diversity_ratio=float(len(fake_words) / len(real_words)) if len(real_words) > 0 else 0.0,
        shared_vocab_size=len(intersection),
        real_vocab_size=len(real_words),
        fake_vocab_size=len(fake_words)
    )


def tfidf_corpus_similarity(
    real_texts: pd.Series, 
    fake_texts: pd.Series,
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95
) -> dict[str, Union[float, int]]:
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
        raise TypeError("real_texts must be a pandas Series")
    if not isinstance(fake_texts, pd.Series):
        raise TypeError("fake_texts must be a pandas Series")
    
    # Performance warning
    _check_large_dataset_warning(real_texts, fake_texts)
    
    # Clean data
    real_clean = real_texts.dropna().astype(str).tolist()
    fake_clean = fake_texts.dropna().astype(str).tolist()
    
    if len(real_clean) == 0 or len(fake_clean) == 0:
        logger.warning("Empty text data detected, returning default similarity")
        return {"cosine_similarity": 0.0, "tfidf_distance": 1.0}
    
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
            token_pattern=r'\b\w+\b'
        )
        
        # Fit on all texts to ensure consistent vocabulary
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Split back into real and fake matrices
        real_tfidf = tfidf_matrix[:len(real_clean)]
        fake_tfidf = tfidf_matrix[len(real_clean):]
        
        # Calculate corpus-level representations (mean of all documents)
        real_corpus_vec = np.mean(real_tfidf.toarray(), axis=0).reshape(1, -1)
        fake_corpus_vec = np.mean(fake_tfidf.toarray(), axis=0).reshape(1, -1)
        
        # Calculate cosine similarity
        cos_sim = cosine_similarity(real_corpus_vec, fake_corpus_vec)[0, 0]
        tfidf_distance = 1.0 - cos_sim
        
        # Calculate corpus norms for additional metrics
        real_corpus_norm = float(np.linalg.norm(real_corpus_vec))
        fake_corpus_norm = float(np.linalg.norm(fake_corpus_vec))
        
        return {
            "cosine_similarity": float(cos_sim),
            "tfidf_distance": float(tfidf_distance),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "real_corpus_norm": real_corpus_norm,
            "fake_corpus_norm": fake_corpus_norm
        }
        
    except Exception as e:
        logger.error(f"TF-IDF similarity calculation failed: {str(e)}")
        return {"cosine_similarity": 0.0, "tfidf_distance": 1.0, "error": str(e)}


def semantic_similarity_embeddings(
    real_texts: pd.Series,
    fake_texts: pd.Series,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    enable_sampling: bool = False,
    max_samples: int = 1000
) -> dict[str, Union[float, str, int]]:
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
            "sentence-transformers is required for semantic similarity analysis. "
            "Install it with: pip install sentence-transformers"
        )
    
    # Input validation
    if not isinstance(real_texts, pd.Series):
        raise TypeError("real_texts must be a pandas Series")
    if not isinstance(fake_texts, pd.Series):
        raise TypeError("fake_texts must be a pandas Series")
    
    # Performance warning
    _check_large_dataset_warning(real_texts, fake_texts)
    
    # Clean data
    real_clean = real_texts.dropna().astype(str).tolist()
    fake_clean = fake_texts.dropna().astype(str).tolist()
    
    if len(real_clean) == 0 or len(fake_clean) == 0:
        logger.warning("Empty text data detected, returning default similarity")
        return {"semantic_similarity": 0.0, "embedding_distance": 1.0}
    
    # Sample data if enabled and datasets are large
    if enable_sampling and len(real_clean) > max_samples:
        real_clean = np.random.choice(real_clean, max_samples, replace=False).tolist()
        logger.info(f"Sampled real texts from {len(real_texts)} to {max_samples}")
    
    if enable_sampling and len(fake_clean) > max_samples:
        fake_clean = np.random.choice(fake_clean, max_samples, replace=False).tolist()
        logger.info(f"Sampled fake texts from {len(fake_texts)} to {max_samples}")
    
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
        
        return {
            "semantic_similarity": float(cos_sim),
            "embedding_distance": float(embedding_distance),
            "model_name": model_name,
            "samples_used": len(real_clean) + len(fake_clean)
        }
        
    except Exception as e:
        logger.error(f"Semantic similarity calculation failed: {str(e)}")
        return {"semantic_similarity": 0.0, "embedding_distance": 1.0, "error": str(e)}


def comprehensive_textual_analysis(
    real_texts: pd.Series,
    fake_texts: pd.Series,
    include_semantic: bool = True,
    enable_sampling: bool = False,
    max_samples: int = 1000
) -> dict[str, dict]:
    """
    Perform comprehensive textual analysis combining all available metrics.
    
    Args:
        real_texts: Series containing real text data
        fake_texts: Series containing synthetic text data
        include_semantic: Whether to include semantic similarity (requires sentence-transformers)
        enable_sampling: Whether to enable sampling for large datasets
        max_samples: Maximum samples per dataset when sampling is enabled
        
    Returns:
        Dictionary with comprehensive textual analysis results
    """
    # Input validation
    if not isinstance(real_texts, pd.Series):
        raise TypeError("real_texts must be a pandas Series")
    if not isinstance(fake_texts, pd.Series):
        raise TypeError("fake_texts must be a pandas Series")
    
    results = {}
    
    # Length distribution analysis
    try:
        results["word_length_dist"] = text_length_distribution_similarity(
            real_texts, fake_texts, unit="word"
        )
        results["char_length_dist"] = text_length_distribution_similarity(
            real_texts, fake_texts, unit="char"
        )
    except Exception as e:
        logger.error(f"Length distribution analysis failed: {str(e)}")
        results["word_length_dist"] = {"error": str(e)}
        results["char_length_dist"] = {"error": str(e)}
    
    # Vocabulary analysis
    try:
        results["vocabulary_overlap"] = vocabulary_overlap_analysis(real_texts, fake_texts)
    except Exception as e:
        logger.error(f"Vocabulary analysis failed: {str(e)}")
        results["vocabulary_overlap"] = {"error": str(e)}
    
    # TF-IDF similarity
    try:
        results["tfidf_similarity"] = tfidf_corpus_similarity(real_texts, fake_texts)
    except Exception as e:
        logger.error(f"TF-IDF analysis failed: {str(e)}")
        results["tfidf_similarity"] = {"error": str(e)}
    
    # Semantic similarity (optional)
    if include_semantic and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            results["semantic_similarity"] = semantic_similarity_embeddings(
                real_texts, fake_texts, enable_sampling=enable_sampling, max_samples=max_samples
            )
        except Exception as e:
            logger.error(f"Semantic similarity analysis failed: {str(e)}")
            results["semantic_similarity"] = {"error": str(e)}
    elif include_semantic and not SENTENCE_TRANSFORMERS_AVAILABLE:
        results["semantic_similarity"] = {
            "error": "sentence-transformers not available. Install with: pip install sentence-transformers"
        }
    
    # Calculate overall similarity score
    similarity_scores = []
    for metric_name, metric_results in results.items():
        if isinstance(metric_results, dict) and "similarity_score" in metric_results:
            similarity_scores.append(metric_results["similarity_score"])
    
    if similarity_scores:
        results["overall"] = {
            "mean_similarity": float(np.mean(similarity_scores)),
            "min_similarity": float(np.min(similarity_scores)),
            "max_similarity": float(np.max(similarity_scores)),
            "num_metrics": len(similarity_scores)
        }
    
    return results