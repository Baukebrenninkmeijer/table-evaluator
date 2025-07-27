"""Unified machine learning evaluation functionality with comprehensive target analysis."""

import copy
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from table_evaluator.constants import RANDOM_SEED
from table_evaluator.metrics.statistical import mean_absolute_percentage_error, rmse
from table_evaluator.models.ml_models import (
    ClassificationResults,
    MLEvaluationResults,
    MLSummary,
    RegressionResults,
    TargetEvaluationResult,
)

logger = logging.getLogger(__name__)


class MLEvaluator:
    """Unified machine learning evaluator with comprehensive target analysis."""

    def __init__(
        self,
        comparison_metric: Callable | None = None,
        random_seed: int = RANDOM_SEED,
        *,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the ML evaluator.

        Args:
            comparison_metric: Function to compare two arrays (e.g., stats.pearsonr)
                              If None, defaults to stats.pearsonr
            random_seed: Random seed for reproducibility
            verbose: Whether to print detailed output
        """
        from scipy import stats

        self.comparison_metric = comparison_metric or stats.pearsonr
        self.random_seed = random_seed
        self.verbose = verbose

    def estimator_evaluation(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        target_col: str,
        target_type: str = 'class',
        *,
        kfold: bool = False,
    ) -> float:
        """
        Perform full estimator evaluation comparing real vs synthetic data.

        Creates two sets of estimators: S_r trained on real data and S_f trained on
        fake data. Both are evaluated on their own and the other's test set.

        Args:
            real: Real dataset (must be numerical)
            fake: Synthetic dataset (must be numerical)
            target_col: Column to use as target for prediction
            target_type: Type of task - "class" or "regr"
            kfold: If True, performs 5-fold CV; if False, single train/test split

        Returns:
            float: Correlation value (for regression) or 1 - MAPE (for classification)
        """
        if target_type not in ['class', 'regr']:
            raise ValueError("target_type must be 'regr' or 'class'")

        # Split features and target
        real_x = real.drop([target_col], axis=1)
        fake_x = fake.drop([target_col], axis=1)

        if real_x.columns.tolist() != fake_x.columns.tolist():
            raise ValueError(f'Real and fake columns are different: \n{real_x.columns}\n{fake_x.columns}')

        real_y = real[target_col]
        fake_y = fake[target_col]

        # Initialize estimators based on task type
        estimators = self._get_estimators(target_type)
        estimator_names = [type(clf).__name__ for clf in estimators]

        # Perform K-Fold evaluation
        kf = KFold(n_splits=5)
        results = []

        for train_index, test_index in kf.split(real_y):
            # Split data
            real_x_train, real_x_test = (
                real_x.iloc[train_index],
                real_x.iloc[test_index],
            )
            real_y_train, real_y_test = (
                real_y.iloc[train_index],
                real_y.iloc[test_index],
            )
            fake_x_train, fake_x_test = (
                fake_x.iloc[train_index],
                fake_x.iloc[test_index],
            )
            fake_y_train, fake_y_test = (
                fake_y.iloc[train_index],
                fake_y.iloc[test_index],
            )

            # Create separate estimator copies for real and fake data
            r_estimators = copy.deepcopy(estimators)
            f_estimators = copy.deepcopy(estimators)

            # Fit estimators
            self._fit_estimators(r_estimators, real_x_train, real_y_train, 'real')
            self._fit_estimators(f_estimators, fake_x_train, fake_y_train, 'fake')

            # Score estimators
            fold_results = self._score_estimators(
                r_estimators,
                f_estimators,
                estimator_names,
                real_x_test,
                real_y_test,
                fake_x_test,
                fake_y_test,
                target_type,
            )
            results.append(fold_results)

            # Break if not doing full k-fold
            if not kfold:
                break

        # Aggregate results across folds
        estimators_scores = pd.concat(results).groupby(level=0).mean()

        if self.verbose:
            if target_type == 'class':
                print('\nClassifier F1-scores and their Jaccard similarities:')
            else:
                print('\nRegressor MSE-scores and their Jaccard similarities:')
            print(estimators_scores.to_string())

        # Calculate final metric
        if target_type == 'regr':
            corr, p = self.comparison_metric(estimators_scores['real'], estimators_scores['fake'])
            return corr
        # target_type == "class"
        mape = mean_absolute_percentage_error(estimators_scores['f1_real'], estimators_scores['f1_fake'])
        return 1 - mape

    def _get_estimators(self, target_type: str) -> list:
        """Get appropriate estimators for the task type."""
        if target_type == 'regr':
            return [
                RandomForestRegressor(n_estimators=20, max_depth=5, random_state=RANDOM_SEED),
                Lasso(random_state=RANDOM_SEED),
                Ridge(alpha=1.0, random_state=RANDOM_SEED),
                ElasticNet(random_state=RANDOM_SEED),
            ]
        # target_type == "class"
        return [
            LogisticRegression(
                multi_class='auto',
                solver='lbfgs',
                max_iter=500,
                random_state=RANDOM_SEED,
            ),
            RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED),
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            MLPClassifier(
                [50, 50],
                solver='adam',
                activation='relu',
                learning_rate='adaptive',
                random_state=RANDOM_SEED,
            ),
        ]

    def _fit_estimators(
        self,
        estimators: list,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        data_type: str,
    ) -> None:
        """Fit estimators to training data."""
        if self.verbose:
            print(f'\nFitting {data_type}')

        for i, estimator in enumerate(estimators):
            if self.verbose:
                print(f'{i + 1}: {type(estimator).__name__}')
            estimator.fit(x_train, y_train)

    def _score_estimators(
        self,
        r_estimators: list,
        f_estimators: list,
        estimator_names: list[str],
        real_x_test: pd.DataFrame,
        real_y_test: pd.Series,
        fake_x_test: pd.DataFrame,
        fake_y_test: pd.Series,
        target_type: str,
    ) -> pd.DataFrame:
        """Score estimators on test data."""
        if target_type == 'class':
            return self._score_classification(
                r_estimators,
                f_estimators,
                estimator_names,
                real_x_test,
                real_y_test,
                fake_x_test,
                fake_y_test,
            )
        return self._score_regression(
            r_estimators,
            f_estimators,
            estimator_names,
            real_x_test,
            real_y_test,
            fake_x_test,
            fake_y_test,
        )

    def _score_classification(
        self,
        r_estimators: list,
        f_estimators: list,
        estimator_names: list[str],
        real_x_test: pd.DataFrame,
        real_y_test: pd.Series,
        fake_x_test: pd.DataFrame,
        fake_y_test: pd.Series,
    ) -> pd.DataFrame:
        """Score classification estimators."""
        rows = []

        for r_classifier, f_classifier, estimator_name in zip(
            r_estimators, f_estimators, estimator_names, strict=False
        ):
            for x_test, y_test, dataset_name in [
                (real_x_test, real_y_test, 'real'),
                (fake_x_test, fake_y_test, 'fake'),
            ]:
                pred_real = r_classifier.predict(x_test)
                pred_fake = f_classifier.predict(x_test)

                f1_real = f1_score(y_test, pred_real, average='micro')
                f1_fake = f1_score(y_test, pred_fake, average='micro')
                jaccard_sim = jaccard_score(pred_real, pred_fake, average='micro')

                row = {
                    'index': f'{estimator_name}_{dataset_name}',
                    'f1_real': f1_real,
                    'f1_fake': f1_fake,
                    'jaccard_similarity': jaccard_sim,
                }
                rows.append(row)

        return pd.DataFrame(rows).set_index('index')

    def _score_regression(
        self,
        r_estimators: list,
        f_estimators: list,
        estimator_names: list[str],
        real_x_test: pd.DataFrame,
        real_y_test: pd.Series,
        fake_x_test: pd.DataFrame,
        fake_y_test: pd.Series,
    ) -> pd.DataFrame:
        """Score regression estimators."""
        # Real estimators on real data
        r2r = [rmse(real_y_test, clf.predict(real_x_test)) for clf in r_estimators]
        # Fake estimators on fake data
        f2f = [rmse(fake_y_test, clf.predict(fake_x_test)) for clf in f_estimators]
        # Real estimators on fake data
        r2f = [rmse(fake_y_test, clf.predict(fake_x_test)) for clf in r_estimators]
        # Fake estimators on real data
        f2r = [rmse(real_y_test, clf.predict(real_x_test)) for clf in f_estimators]

        index = [f'real_data_{classifier}' for classifier in estimator_names] + [
            f'fake_data_{classifier}' for classifier in estimator_names
        ]

        return pd.DataFrame({'real': r2r + f2r, 'fake': r2f + f2f}, index=index)

    def _handle_no_suitable_targets(
        self,
        target_columns: list[str] | None,
        *,
        auto_detect_targets: bool,
        max_targets: int,
    ) -> 'MLEvaluationResults':
        """Handle the case when no suitable target columns are found."""
        summary_dict = {'error': 'No suitable target columns found'}
        recommendations = [
            'No suitable target columns detected. Ensure data has categorical or numerical '
            'columns suitable for prediction.'
        ]
        return MLEvaluationResults(
            classification_results=ClassificationResults(),
            regression_results=RegressionResults(),
            summary=MLSummary(**summary_dict),
            recommendations=recommendations,
            targets_requested=target_columns,
            auto_detect_enabled=auto_detect_targets,
            max_targets_limit=max_targets,
        )

    def _evaluate_classification_targets(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        classification_targets: list[str],
    ) -> tuple[dict, list[float]]:
        """Evaluate all classification targets."""

        def _evaluate_classification_target(
            target_col: str,
        ) -> tuple[dict, float | None]:
            try:
                if self.verbose:
                    print(f'Evaluating classification target: {target_col}')

                score = self.estimator_evaluation(real_data, synthetic_data, target_col, target_type='class')
                return {
                    'score': score,
                    'task_type': 'classification',
                    'quality_rating': self._rate_ml_quality(score, 'classification'),
                }, score

            except Exception as e:
                logger.exception('Classification evaluation failed for {target_col}')
                return {'error': str(e)}, None

        classification_results = {}
        classification_scores = []

        for target_col in classification_targets:
            result, score = _evaluate_classification_target(target_col)
            classification_results[target_col] = result
            if score is not None:
                classification_scores.append(score)

        return classification_results, classification_scores

    def _evaluate_regression_targets(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        regression_targets: list[str],
    ) -> tuple[dict, list[float]]:
        """Evaluate all regression targets."""

        def _evaluate_regression_target(target_col: str) -> tuple[dict, float | None]:
            try:
                if self.verbose:
                    print(f'Evaluating regression target: {target_col}')

                score = self.estimator_evaluation(real_data, synthetic_data, target_col, target_type='regr')
                return {
                    'score': score,
                    'task_type': 'regression',
                    'quality_rating': self._rate_ml_quality(score, 'regression'),
                }, score

            except Exception as e:
                logger.exception('Regression evaluation failed for {target_col}')
                return {'error': str(e)}, None

        regression_results = {}
        regression_scores = []

        for target_col in regression_targets:
            result, score = _evaluate_regression_target(target_col)
            regression_results[target_col] = result
            if score is not None:
                regression_scores.append(score)

        return regression_results, regression_scores

    def _build_evaluation_results(
        self,
        classification_results_dict: dict,
        regression_results_dict: dict,
        summary_dict: dict,
        recommendations: list[str],
        target_columns: list[str] | None,
        *,
        auto_detect_targets: bool,
        max_targets: int,
    ) -> 'MLEvaluationResults':
        """Build the final MLEvaluationResults object."""
        try:
            # Create TargetEvaluationResult objects for classification results
            classification_results = ClassificationResults()
            for target, result in classification_results_dict.items():
                if 'error' not in result:
                    classification_results[target] = TargetEvaluationResult(**result)
                else:
                    classification_results[target] = TargetEvaluationResult(
                        score=0.0,
                        task_type='classification',
                        quality_rating='Error',
                        error=result['error'],
                    )

            # Create TargetEvaluationResult objects for regression results
            regression_results = RegressionResults()
            for target, result in regression_results_dict.items():
                if 'error' not in result:
                    regression_results[target] = TargetEvaluationResult(**result)
                else:
                    regression_results[target] = TargetEvaluationResult(
                        score=0.0,
                        task_type='regression',
                        quality_rating='Error',
                        error=result['error'],
                    )

            # ML summary
            summary = MLSummary(**summary_dict)

            # Create comprehensive results
            return MLEvaluationResults(
                classification_results=classification_results,
                regression_results=regression_results,
                summary=summary,
                recommendations=recommendations,
                targets_requested=target_columns,
                auto_detect_enabled=auto_detect_targets,
                max_targets_limit=max_targets,
            )
        except Exception:
            logger.exception('Error building MLEvaluationResults')
            # Fallback: create models with error handling
            return MLEvaluationResults(
                classification_results=ClassificationResults(),
                regression_results=RegressionResults(),
                summary=MLSummary(**summary_dict),
                recommendations=recommendations,
                targets_requested=target_columns,
                auto_detect_enabled=auto_detect_targets,
                max_targets_limit=max_targets,
            )

    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_columns: list[str] | None = None,
        *,
        auto_detect_targets: bool = True,
        classification_targets: list[str] | None = None,
        regression_targets: list[str] | None = None,
        max_targets: int = 5,
        **kwargs: dict,
    ) -> 'MLEvaluationResults':
        """
        Comprehensive ML evaluation for all suitable target columns.

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            target_columns: Specific columns to use as targets (overrides auto-detection)
            auto_detect_targets: Whether to automatically detect suitable target columns
            classification_targets: Specific columns to treat as classification targets
            regression_targets: Specific columns to treat as regression targets
            max_targets: Maximum number of targets to evaluate (to control runtime)
            **kwargs: Additional parameters for estimator_evaluation

        Returns:
            dict: Comprehensive ML evaluation results
        """
        if self.verbose:
            print('Running comprehensive ML evaluation...')

        # Determine target columns
        targets_to_evaluate = self._determine_target_columns(
            real_data,
            target_columns,
            auto_detect=auto_detect_targets,
            classification_targets=classification_targets,
            regression_targets=regression_targets,
            max_targets=max_targets,
        )

        # Handle case when no suitable targets are found
        if not targets_to_evaluate['classification'] and not targets_to_evaluate['regression']:
            return self._handle_no_suitable_targets(
                target_columns,
                auto_detect_targets=auto_detect_targets,
                max_targets=max_targets,
            )

        # Evaluate classification and regression targets
        classification_results, classification_scores = self._evaluate_classification_targets(
            real_data,
            synthetic_data,
            targets_to_evaluate['classification'],
            **kwargs,
        )
        regression_results, regression_scores = self._evaluate_regression_targets(
            real_data, synthetic_data, targets_to_evaluate['regression'], **kwargs
        )

        # Generate summary and recommendations
        summary_dict = self._generate_ml_summary(classification_scores, regression_scores, targets_to_evaluate)
        recommendations = self._generate_ml_recommendations(classification_results, regression_results, summary_dict)

        # Build and return final results
        return self._build_evaluation_results(
            classification_results,
            regression_results,
            summary_dict,
            recommendations,
            target_columns,
            auto_detect_targets,
            max_targets,
        )

    def _process_explicit_target_columns(self, data: pd.DataFrame, target_columns: list[str], max_targets: int) -> dict:
        """Process explicitly specified target columns."""
        targets_to_evaluate = {'classification': [], 'regression': []}

        for col in target_columns[:max_targets]:
            if col in data.columns:
                # Determine task type based on data characteristics
                if self._is_classification_target(data[col]):
                    targets_to_evaluate['classification'].append(col)
                else:
                    targets_to_evaluate['regression'].append(col)

        return targets_to_evaluate

    def _process_task_specific_targets(
        self,
        data: pd.DataFrame,
        classification_targets: list[str] | None,
        regression_targets: list[str] | None,
        max_targets: int,
    ) -> dict:
        """Process task-specific target columns."""
        targets_to_evaluate = {'classification': [], 'regression': []}

        if classification_targets:
            targets_to_evaluate['classification'] = [
                col for col in classification_targets[:max_targets] if col in data.columns
            ]
        if regression_targets:
            targets_to_evaluate['regression'] = [col for col in regression_targets[:max_targets] if col in data.columns]

        return targets_to_evaluate

    def _auto_detect_suitable_targets(self, data: pd.DataFrame, max_targets: int) -> dict:
        """Auto-detect suitable target columns based on data characteristics."""
        targets_to_evaluate = {'classification': [], 'regression': []}
        potential_targets = []

        for col in data.columns:
            col_data = data[col]

            # Skip columns with too many missing values
            if col_data.isna().sum() / len(col_data) > 0.5:
                continue

            # Skip columns with inappropriate unique value ratios
            unique_ratio = len(col_data.unique()) / len(col_data)

            if self._is_classification_target(col_data):
                potential_targets.append((col, 'classification', unique_ratio))
            elif self._is_regression_target(col_data):
                potential_targets.append((col, 'regression', unique_ratio))

        # Sort by suitability and take top candidates
        potential_targets.sort(key=lambda x: x[2])

        for col, task_type, _ in potential_targets[:max_targets]:
            targets_to_evaluate[task_type].append(col)

        return targets_to_evaluate

    def _determine_target_columns(
        self,
        data: pd.DataFrame,
        target_columns: list[str] | None,
        *,
        auto_detect: bool,
        classification_targets: list[str] | None,
        regression_targets: list[str] | None,
        max_targets: int,
    ) -> dict:
        """Determine which columns to use as targets for ML evaluation."""
        # Use explicitly specified targets if provided
        if target_columns:
            return self._process_explicit_target_columns(data, target_columns, max_targets)

        # Use explicitly specified task-specific targets
        if classification_targets or regression_targets:
            return self._process_task_specific_targets(data, classification_targets, regression_targets, max_targets)

        # Auto-detect targets if enabled and no explicit targets provided
        if auto_detect:
            return self._auto_detect_suitable_targets(data, max_targets)

        # Default empty result
        return {'classification': [], 'regression': []}

    def _is_classification_target(self, series: pd.Series) -> bool:
        """Determine if a column is suitable for classification."""
        # Check if it's categorical or has few unique values
        unique_count = len(series.unique())
        total_count = len(series)

        # Consider categorical data types
        if series.dtype == 'object' or series.dtype.name == 'category':
            return unique_count <= min(50, total_count * 0.1)  # At most 50 classes or 10% of data

        # Consider numerical data with few unique values
        if pd.api.types.is_numeric_dtype(series):
            return 2 <= unique_count <= min(20, total_count * 0.05)  # 2-20 classes, max 5% of data

        return False

    def _is_regression_target(self, series: pd.Series) -> bool:
        """Determine if a column is suitable for regression."""
        # Must be numerical
        if not pd.api.types.is_numeric_dtype(series):
            return False

        # Should have reasonable number of unique values
        unique_count = len(series.unique())
        total_count = len(series)

        # At least 10 unique values and at least 5% unique
        return unique_count >= 10 and unique_count >= total_count * 0.05

    def _rate_ml_quality(self, score: float, task_type: str) -> str:
        """Rate the ML quality based on score and task type."""
        if task_type == 'classification':
            # Score is 1 - MAPE (higher is better)
            if score >= 0.9:
                return 'Excellent'
            if score >= 0.8:
                return 'Good'
            if score >= 0.7:
                return 'Fair'
            if score >= 0.6:
                return 'Poor'
            return 'Very Poor'
        # regression
        # Score is correlation coefficient (higher is better)
        if score >= 0.9:
            return 'Excellent'
        if score >= 0.7:
            return 'Good'
        if score >= 0.5:
            return 'Fair'
        if score >= 0.3:
            return 'Poor'
        return 'Very Poor'

    def _generate_ml_summary(
        self,
        classification_scores: list[float],
        regression_scores: list[float],
        targets_evaluated: dict,
    ) -> dict:
        """Generate summary statistics for ML evaluation."""
        summary = {
            'targets_evaluated': {
                'classification': len(targets_evaluated['classification']),
                'regression': len(targets_evaluated['regression']),
                'total': len(targets_evaluated['classification']) + len(targets_evaluated['regression']),
            }
        }

        if classification_scores:
            summary['classification_summary'] = {
                'best_score': max(classification_scores),
                'mean_score': np.mean(classification_scores),
                'worst_score': min(classification_scores),
                'std_score': np.std(classification_scores),
            }
            summary['best_classification_score'] = max(classification_scores)
        else:
            summary['best_classification_score'] = None

        if regression_scores:
            summary['regression_summary'] = {
                'best_score': max(regression_scores),
                'mean_score': np.mean(regression_scores),
                'worst_score': min(regression_scores),
                'std_score': np.std(regression_scores),
            }
            summary['best_regression_score'] = max(regression_scores)
        else:
            summary['best_regression_score'] = None

        # Overall ML quality assessment
        all_scores = classification_scores + regression_scores
        if all_scores:
            mean_score = np.mean(all_scores)
            if mean_score >= 0.8:
                summary['overall_ml_quality'] = 'Excellent'
            elif mean_score >= 0.7:
                summary['overall_ml_quality'] = 'Good'
            elif mean_score >= 0.6:
                summary['overall_ml_quality'] = 'Fair'
            elif mean_score >= 0.5:
                summary['overall_ml_quality'] = 'Poor'
            else:
                summary['overall_ml_quality'] = 'Very Poor'
        else:
            summary['overall_ml_quality'] = 'Unknown'

        return summary

    def _get_overall_quality_recommendations(self, overall_quality: str) -> list[str]:
        """Generate recommendations based on overall ML quality."""
        if overall_quality == 'Very Poor':
            return [
                'Very low ML similarity detected across targets.',
                'Consider fundamental improvements to the data generation model.',
                'Review feature distributions and correlations.',
                'Ensure the synthetic data captures the underlying patterns of the real data.',
            ]
        if overall_quality == 'Poor':
            return [
                'Low ML similarity detected. The synthetic data may not capture complex patterns.',
                'Consider increasing model complexity or training data size.',
                'Review preprocessing steps and feature engineering.',
            ]
        if overall_quality == 'Fair':
            return [
                'Moderate ML similarity achieved. Some patterns are captured.',
                'Fine-tuning the generation model may improve results.',
                'Focus on poorly performing target variables.',
            ]
        if overall_quality in ['Good', 'Excellent']:
            return ['Good ML similarity achieved. The synthetic data captures most important patterns.']
        return []

    def _get_target_specific_recommendations(self, classification_results: dict, regression_results: dict) -> list[str]:
        """Generate recommendations for specific poorly performing targets."""
        poor_targets = []
        for target, result in {**classification_results, **regression_results}.items():
            if (
                isinstance(result, dict)
                and 'quality_rating' in result
                and result['quality_rating'] in ['Poor', 'Very Poor']
            ):
                poor_targets.append(target)

        if poor_targets:
            return [
                f'Targets with poor performance: {", ".join(poor_targets)}. '
                'Focus on improving synthesis for these variables.'
            ]
        return []

    def _get_task_specific_recommendations(self, summary: dict) -> list[str]:
        """Generate recommendations based on task-specific performance."""
        recommendations = []

        if summary.get('best_classification_score') and summary['best_classification_score'] < 0.7:
            recommendations.append(
                'Classification tasks show room for improvement. '
                'Consider balancing class distributions and improving categorical feature synthesis.'
            )

        if summary.get('best_regression_score') and summary['best_regression_score'] < 0.7:
            recommendations.append(
                'Regression tasks show room for improvement. '
                'Focus on capturing numerical distributions and feature correlations.'
            )

        return recommendations

    def _generate_ml_recommendations(
        self, classification_results: dict, regression_results: dict, summary: dict
    ) -> list[str]:
        """Generate actionable recommendations based on ML evaluation results."""
        recommendations = []

        overall_quality = summary.get('overall_ml_quality', 'Unknown')

        # Get different types of recommendations
        recommendations.extend(self._get_overall_quality_recommendations(overall_quality))
        recommendations.extend(self._get_target_specific_recommendations(classification_results, regression_results))
        recommendations.extend(self._get_task_specific_recommendations(summary))

        return recommendations
