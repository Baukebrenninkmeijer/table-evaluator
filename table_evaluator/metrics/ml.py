"""Machine Learning evaluation functionality for synthetic data quality assessment."""

import copy
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
from table_evaluator.utils import set_random_seed


class MLEvaluator:
    """Handles machine learning evaluation of real vs synthetic data."""

    def __init__(
        self,
        comparison_metric: Callable,
        random_seed: int = RANDOM_SEED,
        *,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the ML evaluator.

        Args:
            comparison_metric: Function to compare two arrays (e.g., stats.pearsonr)
            random_seed: Random seed for reproducibility
            verbose: Whether to print detailed output
        """
        self.comparison_metric = comparison_metric
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
            Correlation value (for regression) or 1 - MAPE (for classification)
        """
        if target_type not in ['class', 'regr']:
            raise ValueError('target_type must be "regr" or "class"')

        # Split features and target
        real_x = real.drop([target_col], axis=1)
        fake_x = fake.drop([target_col], axis=1)

        if real_x.columns.tolist() != fake_x.columns.tolist():
            raise ValueError(f'Real and fake columns are different: \n{real_x.columns}\n{fake_x.columns}')

        real_y = real[target_col]
        fake_y = fake[target_col]

        set_random_seed(self.random_seed)

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
            LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500, random_state=RANDOM_SEED),
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


# =============================================================================
# Utility Functions for ML Evaluation
# =============================================================================


def evaluate_ml_utility(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str,
    task_type: str = 'auto',
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    models: list[str] | None = None,
) -> dict:
    """
    Evaluate the machine learning utility of synthetic data.

    Args:
        real_data: Original dataset
        synthetic_data: Synthetic dataset
        target_column: Name of the target column
        task_type: Type of ML task ('classification', 'regression', or 'auto')
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        models: List of model types to use for evaluation

    Returns:
        Dictionary with ML utility evaluation results
    """
    from sklearn.model_selection import train_test_split

    if models is None:
        models = ['random_forest', 'logistic_regression']

    # Auto-detect task type
    if task_type == 'auto':
        if real_data[target_column].dtype in ['object', 'category'] or real_data[target_column].nunique() < 20:
            task_type = 'classification'
        else:
            task_type = 'regression'

    # Prepare data
    X_real = real_data.drop(columns=[target_column])
    y_real = real_data[target_column]
    X_synthetic = synthetic_data.drop(columns=[target_column])
    y_synthetic = synthetic_data[target_column]

    # Split real data for testing
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=test_size, random_state=random_state
    )

    results = {'task_type': task_type, 'model_results': {}, 'summary': {}}

    # Train models and evaluate
    for model_name in models:
        model_results = _evaluate_single_model(
            model_name,
            task_type,
            X_real_train,
            y_real_train,
            X_real_test,
            y_real_test,
            X_synthetic,
            y_synthetic,
            random_state,
        )
        results['model_results'][model_name] = model_results

    # Calculate summary statistics
    if task_type == 'classification':
        real_scores = [results['model_results'][model]['real_accuracy'] for model in models]
        synthetic_scores = [results['model_results'][model]['synthetic_accuracy'] for model in models]
        metric_name = 'accuracy'
    else:
        real_scores = [results['model_results'][model]['real_r2'] for model in models]
        synthetic_scores = [results['model_results'][model]['synthetic_r2'] for model in models]
        metric_name = 'r2_score'

    results['summary'] = {
        f'mean_real_{metric_name}': np.mean(real_scores),
        f'mean_synthetic_{metric_name}': np.mean(synthetic_scores),
        'utility_score': np.mean(synthetic_scores) / np.mean(real_scores) if np.mean(real_scores) > 0 else 0,
        'score_difference': np.mean(real_scores) - np.mean(synthetic_scores),
    }

    return results


def _evaluate_single_model(
    model_name: str,
    task_type: str,
    X_real_train: pd.DataFrame,
    y_real_train: pd.Series,
    X_real_test: pd.DataFrame,
    y_real_test: pd.Series,
    X_synthetic: pd.DataFrame,
    y_synthetic: pd.Series,
    random_state: int,
) -> dict:
    """Evaluate a single model for ML utility assessment."""
    # Get model
    if task_type == 'classification':
        if model_name == 'random_forest':
            model_real = RandomForestClassifier(random_state=random_state)
            model_synthetic = RandomForestClassifier(random_state=random_state)
        elif model_name == 'logistic_regression':
            model_real = LogisticRegression(random_state=random_state, max_iter=1000)
            model_synthetic = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            raise ValueError(f'Unknown classification model: {model_name}')
    else:  # regression
        if model_name == 'random_forest':
            model_real = RandomForestRegressor(random_state=random_state)
            model_synthetic = RandomForestRegressor(random_state=random_state)
        elif model_name == 'linear_regression':
            from sklearn.linear_model import LinearRegression

            model_real = LinearRegression()
            model_synthetic = LinearRegression()
        else:
            raise ValueError(f'Unknown regression model: {model_name}')

    # Train models
    model_real.fit(X_real_train, y_real_train)
    model_synthetic.fit(X_synthetic, y_synthetic)

    # Make predictions on real test set
    pred_real = model_real.predict(X_real_test)
    pred_synthetic = model_synthetic.predict(X_real_test)

    # Calculate metrics
    if task_type == 'classification':
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        real_accuracy = accuracy_score(y_real_test, pred_real)
        synthetic_accuracy = accuracy_score(y_real_test, pred_synthetic)

        try:
            real_precision = precision_score(y_real_test, pred_real, average='weighted')
            synthetic_precision = precision_score(y_real_test, pred_synthetic, average='weighted')
            real_recall = recall_score(y_real_test, pred_real, average='weighted')
            synthetic_recall = recall_score(y_real_test, pred_synthetic, average='weighted')
            real_f1 = f1_score(y_real_test, pred_real, average='weighted')
            synthetic_f1 = f1_score(y_real_test, pred_synthetic, average='weighted')
        except ValueError:
            # Handle cases with single class predictions
            real_precision = real_accuracy
            synthetic_precision = synthetic_accuracy
            real_recall = real_accuracy
            synthetic_recall = synthetic_accuracy
            real_f1 = real_accuracy
            synthetic_f1 = synthetic_accuracy

        return {
            'real_accuracy': real_accuracy,
            'synthetic_accuracy': synthetic_accuracy,
            'real_precision': real_precision,
            'synthetic_precision': synthetic_precision,
            'real_recall': real_recall,
            'synthetic_recall': synthetic_recall,
            'real_f1': real_f1,
            'synthetic_f1': synthetic_f1,
        }
    # regression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    real_mse = mean_squared_error(y_real_test, pred_real)
    synthetic_mse = mean_squared_error(y_real_test, pred_synthetic)
    real_mae = mean_absolute_error(y_real_test, pred_real)
    synthetic_mae = mean_absolute_error(y_real_test, pred_synthetic)
    real_r2 = r2_score(y_real_test, pred_real)
    synthetic_r2 = r2_score(y_real_test, pred_synthetic)

    return {
        'real_mse': real_mse,
        'synthetic_mse': synthetic_mse,
        'real_mae': real_mae,
        'synthetic_mae': synthetic_mae,
        'real_r2': real_r2,
        'synthetic_r2': synthetic_r2,
    }


def train_test_on_synthetic(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str,
    model_type: str = 'random_forest',
    task_type: str = 'auto',
    random_state: int = RANDOM_SEED,
) -> dict:
    """
    Train a model on synthetic data and test on real data.

    This is a common evaluation pattern to assess how well synthetic data
    can replace real data for model training.

    Args:
        real_data: Original dataset for testing
        synthetic_data: Synthetic dataset for training
        target_column: Name of the target column
        model_type: Type of model to use
        task_type: Type of ML task ('classification', 'regression', or 'auto')
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with evaluation results
    """
    from sklearn.model_selection import train_test_split

    # Auto-detect task type
    if task_type == 'auto':
        if real_data[target_column].dtype in ['object', 'category'] or real_data[target_column].nunique() < 20:
            task_type = 'classification'
        else:
            task_type = 'regression'

    # Prepare data
    X_synthetic = synthetic_data.drop(columns=[target_column])
    y_synthetic = synthetic_data[target_column]
    X_real = real_data.drop(columns=[target_column])
    y_real = real_data[target_column]

    # Split real data for evaluation
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=random_state
    )

    # Initialize models
    if task_type == 'classification':
        if model_type == 'random_forest':
            model_synthetic = RandomForestClassifier(random_state=random_state)
            model_real = RandomForestClassifier(random_state=random_state)
        elif model_type == 'logistic_regression':
            model_synthetic = LogisticRegression(random_state=random_state, max_iter=1000)
            model_real = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            raise ValueError(f'Unknown classification model: {model_type}')
    else:  # regression
        if model_type == 'random_forest':
            model_synthetic = RandomForestRegressor(random_state=random_state)
            model_real = RandomForestRegressor(random_state=random_state)
        elif model_type == 'linear_regression':
            from sklearn.linear_model import LinearRegression

            model_synthetic = LinearRegression()
            model_real = LinearRegression()
        else:
            raise ValueError(f'Unknown regression model: {model_type}')

    # Train models
    model_synthetic.fit(X_synthetic, y_synthetic)  # Train on synthetic
    model_real.fit(X_real_train, y_real_train)  # Train on real (baseline)

    # Test both models on real test data
    pred_synthetic = model_synthetic.predict(X_real_test)
    pred_real = model_real.predict(X_real_test)

    # Calculate metrics
    if task_type == 'classification':
        from sklearn.metrics import accuracy_score, classification_report

        synthetic_accuracy = accuracy_score(y_real_test, pred_synthetic)
        real_accuracy = accuracy_score(y_real_test, pred_real)

        return {
            'task_type': task_type,
            'model_type': model_type,
            'synthetic_model_accuracy': synthetic_accuracy,
            'real_model_accuracy': real_accuracy,
            'accuracy_ratio': synthetic_accuracy / real_accuracy if real_accuracy > 0 else 0,
            'accuracy_difference': real_accuracy - synthetic_accuracy,
            'classification_report_synthetic': classification_report(y_real_test, pred_synthetic, output_dict=True),
            'classification_report_real': classification_report(y_real_test, pred_real, output_dict=True),
        }
    # regression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    synthetic_mse = mean_squared_error(y_real_test, pred_synthetic)
    real_mse = mean_squared_error(y_real_test, pred_real)
    synthetic_r2 = r2_score(y_real_test, pred_synthetic)
    real_r2 = r2_score(y_real_test, pred_real)
    synthetic_mae = mean_absolute_error(y_real_test, pred_synthetic)
    real_mae = mean_absolute_error(y_real_test, pred_real)

    return {
        'task_type': task_type,
        'model_type': model_type,
        'synthetic_model_mse': synthetic_mse,
        'real_model_mse': real_mse,
        'synthetic_model_r2': synthetic_r2,
        'real_model_r2': real_r2,
        'synthetic_model_mae': synthetic_mae,
        'real_model_mae': real_mae,
        'mse_ratio': synthetic_mse / real_mse if real_mse > 0 else float('inf'),
        'r2_ratio': synthetic_r2 / real_r2 if real_r2 > 0 else 0,
    }
