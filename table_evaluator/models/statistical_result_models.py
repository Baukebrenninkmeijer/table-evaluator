"""Result models for statistical metrics functions."""

from pydantic import BaseModel, Field


class WassersteinDistanceSummaryResult(BaseModel):
    """Model for earth_movers_distance_summary results."""

    summary_stats: dict[str, float] = Field(description='Summary statistics across all columns')
    column_distances: dict[str, dict[str, float]] = Field(description='Distance metrics per column')
    overall_metrics: dict[str, float | str] = Field(
        description='Overall Wasserstein metrics (includes worst_column as string)'
    )
    success: bool = True


class MMDAnalysisResult(BaseModel):
    """Model for comprehensive_mmd_analysis results."""

    column_wise: dict[str, dict] = Field(description='Column-wise MMD results by kernel')
    multivariate: dict[str, dict] = Field(description='Multivariate MMD results by kernel')
    summary: dict[str, float] = Field(description='Summary metrics across kernels')
    success: bool = True


class MultivariateMMDResult(BaseModel):
    """Model for multivariate_mmd results."""

    mmd_value: float = Field(description='MMD value')
    kernel_type: str = Field(description='Kernel type used')
    kernel_params: dict = Field(description='Kernel parameters')
    n_samples_real: int = Field(description='Number of real samples')
    n_samples_fake: int = Field(description='Number of fake samples')
    success: bool = True


class JensenShannonResult(BaseModel):
    """Model for Jensen-Shannon distance results."""

    col_name: str = Field(description='Column name')
    js_distance: float = Field(description='Jensen-Shannon distance (0-1, or NaN for invalid cases)')
    success: bool = True


class WassersteinDistanceResult(BaseModel):
    """Model for wasserstein_distance results."""

    col_name: str = Field(description='Column name')
    wasserstein_distance: float = Field(ge=0, description='Wasserstein distance')
    method: str = Field(description='Method used for calculation')
    success: bool = True
