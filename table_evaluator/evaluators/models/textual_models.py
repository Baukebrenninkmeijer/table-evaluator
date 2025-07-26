"""Pydantic models for textual evaluation results."""

from pydantic import BaseModel, Field


class LexicalDiversityResults(BaseModel):
    """Results from lexical diversity analysis."""

    word_length_distribution: dict = Field(description='Word length distribution analysis results')
    char_length_distribution: dict = Field(description='Character length distribution analysis results')
    vocabulary_overlap: dict = Field(description='Vocabulary overlap analysis results')
    summary: dict = Field(description='Summary of lexical diversity metrics')

    class Config:
        extra = 'allow'


class TfidfSimilarityResults(BaseModel):
    """Results from TF-IDF similarity analysis."""

    cosine_similarity: float = Field(description='Cosine similarity between TF-IDF vectors')
    tfidf_distance: float = Field(description='TF-IDF distance between corpora')
    vocabulary_size: int = Field(description='Size of the combined vocabulary')
    similarity_score: float = Field(description='Overall TF-IDF similarity score')
    summary: dict = Field(description='Summary of TF-IDF similarity analysis')
    error: str | None = Field(default=None, description='Error message if TF-IDF analysis failed')

    class Config:
        extra = 'allow'


class SemanticSimilarityResults(BaseModel):
    """Results from semantic similarity analysis."""

    available: bool = Field(description='Whether semantic similarity analysis was available')
    semantic_similarity: float | None = Field(default=None, description='Semantic similarity score')
    embedding_distance: float | None = Field(default=None, description='Embedding distance')
    model_used: str | None = Field(default=None, description='Model used for semantic analysis')
    similarity_score: float | None = Field(default=None, description='Overall semantic similarity score')
    summary: dict | None = Field(default=None, description='Summary of semantic similarity analysis')
    error: str | None = Field(default=None, description='Error message if semantic analysis failed')

    class Config:
        extra = 'allow'


class CombinedTextualMetrics(BaseModel):
    """Combined metrics from textual evaluation."""

    overall_similarity: float = Field(description='Overall textual similarity score')
    lexical_weight: float = Field(description='Weight given to lexical metrics')
    tfidf_weight: float = Field(description='Weight given to TF-IDF metrics')
    semantic_weight: float = Field(description='Weight given to semantic metrics')
    quality_rating: str = Field(description='Quality rating (Excellent/Good/Fair/Poor/Very Poor)')

    class Config:
        extra = 'allow'


class TextualEvaluationResults(BaseModel):
    """Complete results from textual evaluation."""

    lexical_diversity: LexicalDiversityResults = Field(description='Results from lexical diversity analysis')
    tfidf_similarity: TfidfSimilarityResults = Field(description='Results from TF-IDF similarity analysis')
    semantic_similarity: SemanticSimilarityResults | None = Field(
        default=None, description='Results from semantic similarity analysis'
    )
    combined_metrics: CombinedTextualMetrics = Field(description='Combined textual evaluation metrics')
    recommendations: list[str] = Field(
        default_factory=list, description='Actionable recommendations based on textual analysis'
    )

    # Configuration information
    text_columns: list[str] = Field(description='List of text columns that were evaluated')
    included_semantic: bool = Field(default=False, description='Whether semantic analysis was included')
    sampling_enabled: bool = Field(default=False, description='Whether sampling was enabled for large datasets')
    max_samples: int | None = Field(default=None, description='Maximum samples used if sampling was enabled')

    class Config:
        extra = 'allow'
