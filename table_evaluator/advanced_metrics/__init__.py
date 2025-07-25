"""Advanced metrics package for table evaluation."""

# This package contains advanced metrics that don't conflict with the main metrics.py module
# Main metrics are still imported from table_evaluator.metrics (the .py file)

from .wasserstein import (
    wasserstein_distance_1d,
    wasserstein_distance_df,
    earth_movers_distance_summary,
)

from .mmd import (
    mmd_squared,
    mmd_column_wise,
    mmd_multivariate,
    mmd_comprehensive_analysis,
    RBFKernel,
    PolynomialKernel,
    LinearKernel,
)

from .privacy_attacks import (
    identify_quasi_identifiers,
    calculate_k_anonymity,
    simulate_membership_inference_attack,
    comprehensive_privacy_analysis,
)

from .textual import (
    text_length_distribution_similarity,
    vocabulary_overlap_analysis,
    tfidf_corpus_similarity,
    semantic_similarity_embeddings,
    comprehensive_textual_analysis,
)

__all__ = [
    # Wasserstein distance
    "wasserstein_distance_1d",
    "wasserstein_distance_df",
    "earth_movers_distance_summary",
    # Maximum Mean Discrepancy
    "mmd_squared",
    "mmd_column_wise",
    "mmd_multivariate",
    "mmd_comprehensive_analysis",
    "RBFKernel",
    "PolynomialKernel",
    "LinearKernel",
    # Privacy attacks
    "identify_quasi_identifiers",
    "calculate_k_anonymity",
    "simulate_membership_inference_attack",
    "comprehensive_privacy_analysis",
    # Textual analysis
    "text_length_distribution_similarity",
    "vocabulary_overlap_analysis",
    "tfidf_corpus_similarity",
    "semantic_similarity_embeddings",
    "comprehensive_textual_analysis",
]
