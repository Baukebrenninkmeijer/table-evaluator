"""Pydantic models for machine learning evaluation results."""

from collections.abc import Iterator

from pydantic import BaseModel, Field


class TargetEvaluationResult(BaseModel):
    """Result for a single target column evaluation."""

    score: float = Field(description='Evaluation score for this target')
    task_type: str = Field(description='Type of ML task (classification or regression)')
    quality_rating: str = Field(description='Quality rating (Excellent/Good/Fair/Poor/Very Poor)')
    error: str | None = Field(default=None, description='Error message if evaluation failed')

    class Config:
        extra = 'allow'


class ClassificationResults(BaseModel):
    """Results from classification target evaluations."""

    targets: dict[str, TargetEvaluationResult] = Field(
        default_factory=dict, description='Results for each classification target column'
    )

    def __getitem__(self, key: str) -> TargetEvaluationResult:
        """Allow dictionary-style access for backward compatibility."""
        return self.targets[key]

    def __setitem__(self, key: str, value: TargetEvaluationResult) -> None:
        """Allow dictionary-style assignment for backward compatibility."""
        self.targets[key] = value

    def items(self) -> dict[str, TargetEvaluationResult]:
        """Allow iteration over items for backward compatibility."""
        return self.targets.items()

    def keys(self) -> list[str]:
        """Allow access to keys for backward compatibility."""
        return self.targets.keys()

    def values(self) -> dict[str, TargetEvaluationResult]:
        """Allow access to values for backward compatibility."""
        return self.targets.values()

    def get(self, key: str, default: TargetEvaluationResult | None = None) -> TargetEvaluationResult | None:
        """Allow .get() method for backward compatibility."""
        return self.targets.get(key, default)

    def __len__(self) -> int:
        """Allow len() function for backward compatibility."""
        return len(self.targets)

    def __iter__(self) -> Iterator[str]:
        """Allow iteration over keys for backward compatibility."""
        return iter(self.targets)

    class Config:
        extra = 'allow'


class RegressionResults(BaseModel):
    """Results from regression target evaluations."""

    targets: dict[str, TargetEvaluationResult] = Field(
        default_factory=dict, description='Results for each regression target column'
    )

    def __getitem__(self, key: str) -> TargetEvaluationResult:
        """Allow dictionary-style access for backward compatibility."""
        return self.targets[key]

    def __setitem__(self, key: str, value: TargetEvaluationResult) -> None:
        """Allow dictionary-style assignment for backward compatibility."""
        self.targets[key] = value

    def items(self) -> dict[str, TargetEvaluationResult]:
        """Allow iteration over items for backward compatibility."""
        return self.targets.items()

    def keys(self) -> list[str]:
        """Allow access to keys for backward compatibility."""
        return self.targets.keys()

    def values(self) -> dict[str, TargetEvaluationResult]:
        """Allow access to values for backward compatibility."""
        return self.targets.values()

    def get(self, key: str, default: TargetEvaluationResult | None = None) -> TargetEvaluationResult | None:
        """Allow .get() method for backward compatibility."""
        return self.targets.get(key, default)

    def __len__(self) -> int:
        """Allow len() function for backward compatibility."""
        return len(self.targets)

    def __iter__(self) -> Iterator[str]:
        """Allow iteration over keys for backward compatibility."""
        return iter(self.targets)

    class Config:
        extra = 'allow'


class TargetsEvaluated(BaseModel):
    """Number of targets evaluated by type."""

    classification: int = Field(default=0, description='Number of classification targets evaluated')
    regression: int = Field(default=0, description='Number of regression targets evaluated')
    total: int = Field(default=0, description='Total number of targets evaluated')

    class Config:
        extra = 'allow'


class ClassificationSummary(BaseModel):
    """Summary statistics for classification targets."""

    best_score: float = Field(description='Best score achieved across classification targets')
    mean_score: float = Field(description='Mean score across classification targets')
    worst_score: float = Field(description='Worst score across classification targets')
    std_score: float = Field(description='Standard deviation of classification scores')

    class Config:
        extra = 'allow'


class RegressionSummary(BaseModel):
    """Summary statistics for regression targets."""

    best_score: float = Field(description='Best score achieved across regression targets')
    mean_score: float = Field(description='Mean score across regression targets')
    worst_score: float = Field(description='Worst score across regression targets')
    std_score: float = Field(description='Standard deviation of regression scores')

    class Config:
        extra = 'allow'


class MLSummary(BaseModel):
    """Summary statistics from ML evaluation."""

    targets_evaluated: TargetsEvaluated = Field(
        default_factory=TargetsEvaluated,
        description='Number of targets evaluated by type (classification, regression, total)',
    )
    classification_summary: ClassificationSummary | None = Field(
        default=None, description='Summary statistics for classification targets'
    )
    regression_summary: RegressionSummary | None = Field(
        default=None, description='Summary statistics for regression targets'
    )
    best_classification_score: float | None = Field(
        default=None, description='Best score achieved across all classification targets'
    )
    best_regression_score: float | None = Field(
        default=None, description='Best score achieved across all regression targets'
    )
    overall_ml_quality: str = Field(
        description='Overall ML quality assessment (Excellent/Good/Fair/Poor/Very Poor/Unknown)'
    )
    error: str | None = Field(default=None, description='Error message if summary generation failed')

    class Config:
        extra = 'allow'


class MLEvaluationResults(BaseModel):
    """Complete results from machine learning evaluation."""

    classification_results: ClassificationResults = Field(
        default_factory=ClassificationResults, description='Results from classification target evaluations'
    )
    regression_results: RegressionResults = Field(
        default_factory=RegressionResults, description='Results from regression target evaluations'
    )
    summary: MLSummary = Field(description='Summary statistics and overall assessment')
    recommendations: list[str] = Field(
        default_factory=list, description='Actionable recommendations based on ML evaluation results'
    )

    # Configuration information
    targets_requested: list[str] | None = Field(
        default=None, description='List of specifically requested target columns'
    )
    auto_detect_enabled: bool = Field(default=True, description='Whether automatic target detection was enabled')
    max_targets_limit: int = Field(default=5, description='Maximum number of targets that were evaluated')

    class Config:
        extra = 'allow'
