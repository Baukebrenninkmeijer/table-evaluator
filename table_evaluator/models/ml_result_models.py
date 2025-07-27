"""Result models for ML metrics functions."""

from pydantic import BaseModel, Field

from .error_models import ErrorResult


class SingleModelEvaluationResult(BaseModel):
    """Model for _evaluate_single_model results."""

    # Classification metrics
    real_accuracy: float | None = Field(default=None, description='Accuracy of model trained on real data')
    synthetic_accuracy: float | None = Field(default=None, description='Accuracy of model trained on synthetic data')
    real_precision: float | None = Field(default=None, description='Precision of model trained on real data')
    synthetic_precision: float | None = Field(default=None, description='Precision of model trained on synthetic data')
    real_recall: float | None = Field(default=None, description='Recall of model trained on real data')
    synthetic_recall: float | None = Field(default=None, description='Recall of model trained on synthetic data')
    real_f1: float | None = Field(default=None, description='F1 score of model trained on real data')
    synthetic_f1: float | None = Field(default=None, description='F1 score of model trained on synthetic data')

    # Regression metrics
    real_mse: float | None = Field(default=None, description='MSE of model trained on real data')
    synthetic_mse: float | None = Field(default=None, description='MSE of model trained on synthetic data')
    real_mae: float | None = Field(default=None, description='MAE of model trained on real data')
    synthetic_mae: float | None = Field(default=None, description='MAE of model trained on synthetic data')
    real_r2: float | None = Field(default=None, description='R2 score of model trained on real data')
    synthetic_r2: float | None = Field(default=None, description='R2 score of model trained on synthetic data')

    success: bool = True


class MLUtilityEvaluationSummary(BaseModel):
    """Model for ML utility evaluation summary."""

    mean_real_accuracy: float | None = Field(default=None, description='Mean accuracy on real data')
    mean_synthetic_accuracy: float | None = Field(default=None, description='Mean accuracy on synthetic data')
    mean_real_r2_score: float | None = Field(default=None, description='Mean R2 score on real data')
    mean_synthetic_r2_score: float | None = Field(default=None, description='Mean R2 score on synthetic data')
    utility_score: float = Field(description='Overall utility score (synthetic/real performance ratio)')
    score_difference: float = Field(description='Performance difference (real - synthetic)')


class MLUtilityEvaluationResult(BaseModel):
    """Model for evaluate_ml_utility results."""

    task_type: str = Field(description='Type of ML task (classification/regression)')
    model_results: dict[str, SingleModelEvaluationResult | ErrorResult] = Field(
        description='Results for each model type'
    )
    summary: MLUtilityEvaluationSummary | ErrorResult = Field(description='Summary metrics')
    success: bool = True


class TrainTestSyntheticResult(BaseModel):
    """Model for train_test_on_synthetic results."""

    task_type: str = Field(description='Type of ML task (classification/regression)')
    model_type: str = Field(description='Type of model used')

    # Classification metrics
    synthetic_model_accuracy: float | None = Field(
        default=None, description='Accuracy of model trained on synthetic data'
    )
    real_model_accuracy: float | None = Field(
        default=None, description='Accuracy of baseline model trained on real data'
    )
    accuracy_ratio: float | None = Field(default=None, description='Ratio of synthetic to real model accuracy')
    accuracy_difference: float | None = Field(default=None, description='Difference in accuracy (real - synthetic)')
    classification_report_synthetic: dict | None = Field(
        default=None, description='Detailed classification report for synthetic model'
    )
    classification_report_real: dict | None = Field(
        default=None, description='Detailed classification report for real model'
    )

    # Regression metrics
    synthetic_model_mse: float | None = Field(default=None, description='MSE of model trained on synthetic data')
    real_model_mse: float | None = Field(default=None, description='MSE of baseline model trained on real data')
    synthetic_model_r2: float | None = Field(default=None, description='R2 score of model trained on synthetic data')
    real_model_r2: float | None = Field(default=None, description='R2 score of baseline model trained on real data')
    synthetic_model_mae: float | None = Field(default=None, description='MAE of model trained on synthetic data')
    real_model_mae: float | None = Field(default=None, description='MAE of baseline model trained on real data')
    mse_ratio: float | None = Field(default=None, description='Ratio of synthetic to real model MSE')
    r2_ratio: float | None = Field(default=None, description='Ratio of synthetic to real model R2')

    success: bool = True
