"""Pydantic models for textual evaluation outputs."""

from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class VocabularyOverlapResult(BaseModel):
    """Model for vocabulary overlap analysis results."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    jaccard_similarity: float = Field(ge=0, le=1, description="Jaccard similarity coefficient")
    vocab_diversity_ratio: float = Field(gt=0, description="Vocabulary diversity ratio")
    shared_vocab_size: int = Field(ge=0, description="Number of shared vocabulary words")
    real_vocab_size: int = Field(ge=0, description="Size of real text vocabulary")
    fake_vocab_size: int = Field(ge=0, description="Size of synthetic text vocabulary")


class LengthDistributionResult(BaseModel):
    """Model for text length distribution analysis results."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    similarity_score: float = Field(ge=0, le=1, description="Distribution similarity score")
    ks_statistic: float = Field(ge=0, description="Kolmogorov-Smirnov test statistic")
    ks_p_value: float = Field(ge=0, le=1, description="KS test p-value")
    mean_real: float = Field(description="Mean length in real texts")
    mean_fake: float = Field(description="Mean length in synthetic texts")
    std_real: float = Field(ge=0, description="Standard deviation of real text lengths")
    std_fake: float = Field(ge=0, description="Standard deviation of synthetic text lengths")


class LexicalDiversitySummary(BaseModel):
    """Model for lexical diversity summary metrics."""
    
    word_length_similarity: float = Field(ge=0, le=1, description="Word length distribution similarity")
    char_length_similarity: float = Field(ge=0, le=1, description="Character length distribution similarity")
    vocabulary_jaccard_similarity: float = Field(ge=0, le=1, description="Vocabulary Jaccard similarity")
    overall_lexical_similarity: float = Field(ge=0, le=1, description="Overall lexical similarity score")
    quality_rating: str = Field(description="Lexical quality rating")


class LexicalDiversityResult(BaseModel):
    """Model for complete lexical diversity evaluation results."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    word_length_distribution: LengthDistributionResult
    char_length_distribution: LengthDistributionResult
    vocabulary_overlap: VocabularyOverlapResult
    summary: LexicalDiversitySummary
    error: Optional[str] = None


class SemanticSimilaritySummary(BaseModel):
    """Model for semantic similarity summary metrics."""
    
    semantic_similarity_score: float = Field(ge=0, le=1, description="Semantic similarity score")
    embedding_distance: float = Field(ge=0, description="Embedding distance")
    quality_rating: str = Field(description="Semantic quality rating")
    model_used: str = Field(description="Model used for semantic analysis")


class SemanticSimilarityResult(BaseModel):
    """Model for semantic similarity evaluation results."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    available: bool = Field(description="Whether semantic analysis is available")
    semantic_similarity: Optional[float] = Field(None, ge=0, le=1, description="Semantic similarity score")
    embedding_distance: Optional[float] = Field(None, ge=0, description="Embedding distance")
    model_name: Optional[str] = Field(None, description="Model used")
    samples_used: Optional[int] = Field(None, ge=0, description="Number of samples used")
    summary: Optional[SemanticSimilaritySummary] = None
    error: Optional[str] = None


class TfidfSimilaritySummary(BaseModel):
    """Model for TF-IDF similarity summary metrics."""
    
    tfidf_cosine_similarity: float = Field(ge=0, le=1, description="TF-IDF cosine similarity")
    tfidf_distance: float = Field(ge=0, description="TF-IDF distance")
    vocabulary_size: int = Field(ge=0, description="TF-IDF vocabulary size")
    quality_rating: str = Field(description="TF-IDF quality rating")
    features_used: int = Field(ge=0, description="Number of features used")
    ngram_range: tuple[int, int] = Field(description="N-gram range used")


class TfidfSimilarityResult(BaseModel):
    """Model for TF-IDF similarity evaluation results."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    cosine_similarity: float = Field(ge=0, le=1, description="Cosine similarity score")
    tfidf_distance: float = Field(ge=0, description="TF-IDF distance")
    vocabulary_size: int = Field(ge=0, description="Vocabulary size")
    real_corpus_norm: float = Field(ge=0, description="Real corpus norm")
    fake_corpus_norm: float = Field(ge=0, description="Synthetic corpus norm")
    summary: TfidfSimilaritySummary
    error: Optional[str] = None


class QualityRatings(BaseModel):
    """Model for quality ratings across different metrics."""
    
    lexical: str = Field(description="Lexical quality rating")
    tfidf: str = Field(description="TF-IDF quality rating")
    semantic: str = Field(description="Semantic quality rating")
    overall: str = Field(description="Overall quality rating")


class ConsistencyMetrics(BaseModel):
    """Model for consistency analysis metrics."""
    
    score_std: float = Field(ge=0, description="Standard deviation of scores")
    score_range: float = Field(ge=0, description="Range of scores")
    consistency_rating: str = Field(description="Consistency rating")


class CorpusStatistics(BaseModel):
    """Model for corpus-level statistics."""
    
    vocabulary_diversity_ratio: Optional[float] = Field(None, gt=0, description="Vocabulary diversity ratio")
    shared_vocabulary_size: Optional[int] = Field(None, ge=0, description="Shared vocabulary size")
    real_vocabulary_size: Optional[int] = Field(None, ge=0, description="Real vocabulary size")
    fake_vocabulary_size: Optional[int] = Field(None, ge=0, description="Synthetic vocabulary size")
    tfidf_vocabulary_size: Optional[int] = Field(None, ge=0, description="TF-IDF vocabulary size")


class CombinedMetrics(BaseModel):
    """Model for combined metrics analysis."""
    
    overall_similarity: float = Field(ge=0, le=1, description="Overall similarity score")
    weights_used: dict[str, float] = Field(description="Weights used in calculation")
    quality_ratings: QualityRatings
    consistency_metrics: ConsistencyMetrics
    corpus_statistics: CorpusStatistics
    error: Optional[str] = None


class ComprehensiveTextualResult(BaseModel):
    """Model for comprehensive textual evaluation results."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    lexical_diversity: LexicalDiversityResult
    tfidf_similarity: TfidfSimilarityResult
    semantic_similarity: SemanticSimilarityResult
    combined_metrics: CombinedMetrics
    recommendations: list[str] = Field(description="Actionable recommendations")


class QuickTextualResult(BaseModel):
    """Model for quick textual evaluation results."""
    
    lexical_similarity: float = Field(ge=0, le=1, description="Lexical similarity score")
    tfidf_similarity: float = Field(ge=0, le=1, description="TF-IDF similarity score")
    overall_similarity: float = Field(ge=0, le=1, description="Overall similarity score")
    quality_rating: str = Field(description="Quality rating")
    evaluation_type: str = Field(default="quick", description="Type of evaluation performed")
    semantic_analysis_included: bool = Field(default=False, description="Whether semantic analysis was included")
    error: Optional[str] = None


class TextualEvaluationSummary(BaseModel):
    """Model for textual evaluation integration summary."""
    
    textual_similarity: float = Field(ge=0, le=1, description="Overall textual similarity")
    quality_rating: str = Field(description="Quality rating")
    evaluation_type: str = Field(description="Type of evaluation performed")
    error: Optional[str] = None