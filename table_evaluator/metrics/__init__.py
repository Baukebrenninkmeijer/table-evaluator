"""
Comprehensive metrics module for table evaluator.

This module provides a unified interface for all metrics used in synthetic data evaluation,
organized into logical categories:

- Statistical metrics: Distribution comparison and association measures
- Privacy metrics: Privacy risk assessment and anonymity analysis  
- ML metrics: Machine learning utility evaluation

The module maintains backward compatibility while providing a cleaner, more organized API.

## Migration Guide (v1.9.0+)

### New Module Structure:
```
table_evaluator/metrics/
├── __init__.py          # Central API (backward compatible)
├── statistical.py       # Statistical comparison metrics
├── privacy.py          # Privacy analysis metrics
└── ml.py               # Machine learning evaluation
```

### Import Patterns:
```python
# Backward compatible (recommended for existing code):
from table_evaluator.metrics import mean_absolute_error, cramers_v

# New modular imports (recommended for new code):
from table_evaluator.metrics.statistical import wasserstein_distance_1d
from table_evaluator.metrics.privacy import calculate_k_anonymity
from table_evaluator.metrics.ml import MLEvaluator
```

### What Changed:
- Consolidated scattered metrics into organized modules
- Improved code maintainability and discoverability
- All existing import paths continue to work unchanged
- Enhanced documentation and type hints
"""

# Core statistical metrics - commonly used functions
# ML evaluation metrics
from .ml import (
    MLEvaluator,
    evaluate_ml_utility,
    train_test_on_synthetic,
)

# Privacy metrics
from .privacy import (
    analyze_l_diversity,
    assess_overall_privacy_risk,
    calculate_k_anonymity,
    calculate_privacy_score,
    comprehensive_privacy_analysis,
    generate_comprehensive_recommendations,
    identify_quasi_identifiers,
    simulate_membership_inference_attack,
)
from .statistical import (
    LinearKernel,
    MMDKernel,
    PolynomialKernel,
    RBFKernel,
    associations,
    column_correlations,
    correlation_ratio,
    cosine_similarity,
    # Association metrics
    cramers_v,
    earth_movers_distance_summary,
    euclidean_distance,
    jensenshannon_distance,
    # Distribution comparison
    js_distance_df,
    kolmogorov_smirnov_df,
    kolmogorov_smirnov_test,
    # Basic metrics
    mean_absolute_error,
    mean_absolute_percentage_error,
    mmd_column_wise,
    mmd_comprehensive_analysis,
    mmd_multivariate,
    # MMD metrics
    mmd_squared,
    optimal_transport_cost,
    rmse,
    theils_u,
    # Advanced statistical methods
    wasserstein_distance_1d,
    wasserstein_distance_2d,
    wasserstein_distance_df,
)

# Define what gets imported with "from table_evaluator.metrics import *"
__all__ = [
    # Statistical metrics
    "mean_absolute_error",
    "euclidean_distance",
    "mean_absolute_percentage_error",
    "rmse",
    "cosine_similarity",
    "cramers_v",
    "theils_u",
    "correlation_ratio",
    "associations",
    "column_correlations",
    "js_distance_df",
    "jensenshannon_distance",
    "kolmogorov_smirnov_test",
    "kolmogorov_smirnov_df",
    "wasserstein_distance_1d",
    "wasserstein_distance_2d",
    "wasserstein_distance_df",
    "earth_movers_distance_summary",
    "optimal_transport_cost",
    "mmd_comprehensive_analysis",
    "mmd_squared",
    "mmd_column_wise",
    "mmd_multivariate",
    "MMDKernel",
    "RBFKernel",
    "PolynomialKernel",
    "LinearKernel",

    # Privacy metrics
    "identify_quasi_identifiers",
    "calculate_k_anonymity",
    "analyze_l_diversity",
    "simulate_membership_inference_attack",
    "comprehensive_privacy_analysis",
    "assess_overall_privacy_risk",
    "calculate_privacy_score",
    "generate_comprehensive_recommendations",

    # ML metrics
    "MLEvaluator",
    "evaluate_ml_utility",
    "train_test_on_synthetic",
]

# Version info
__version__ = '1.9.0'
