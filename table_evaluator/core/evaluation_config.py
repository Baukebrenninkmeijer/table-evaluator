"""Configuration classes for table evaluation."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EvaluationConfig:
    """Configuration for table evaluation process."""

    # Data preprocessing
    unique_thresh: int = 0
    n_samples: Optional[int] = None
    random_seed: int = 1337

    # Privacy evaluation
    n_samples_distance: int = 20000

    # Output control
    verbose: bool = False

    # ML evaluation
    kfold: bool = False
    estimator_configs: Dict = field(default_factory=dict)

    # Validation
    def __post_init__(self):
        """Validate configuration values."""
        if self.unique_thresh < 0:
            raise ValueError("unique_thresh must be non-negative")
        if self.n_samples is not None and self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_samples_distance <= 0:
            raise ValueError("n_samples_distance must be positive")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
