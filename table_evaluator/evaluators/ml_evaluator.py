"""Machine Learning evaluation functionality extracted from TableEvaluator."""

import copy
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from table_evaluator.metrics import mean_absolute_percentage_error, rmse


class MLEvaluator:
    """Handles machine learning evaluation of real vs synthetic data."""

    def __init__(
        self,
        comparison_metric: Callable,
        random_seed: int = 1337,
        verbose: bool = False,
    ):
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
        target_type: str = "class",
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
        if target_type not in ["class", "regr"]:
            raise ValueError("target_type must be 'regr' or 'class'")

        # Split features and target
        real_x = real.drop([target_col], axis=1)
        fake_x = fake.drop([target_col], axis=1)

        assert (
            real_x.columns.tolist() == fake_x.columns.tolist()
        ), f"Real and fake columns are different: \n{real_x.columns}\n{fake_x.columns}"

        real_y = real[target_col]
        fake_y = fake[target_col]

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

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
            self._fit_estimators(r_estimators, real_x_train, real_y_train, "real")
            self._fit_estimators(f_estimators, fake_x_train, fake_y_train, "fake")

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
            if target_type == "class":
                print("\nClassifier F1-scores and their Jaccard similarities:")
            else:
                print("\nRegressor MSE-scores and their Jaccard similarities:")
            print(estimators_scores.to_string())

        # Calculate final metric
        if target_type == "regr":
            corr, p = self.comparison_metric(
                estimators_scores["real"], estimators_scores["fake"]
            )
            return corr
        else:  # target_type == "class"
            mape = mean_absolute_percentage_error(
                estimators_scores["f1_real"], estimators_scores["f1_fake"]
            )
            return 1 - mape

    def _get_estimators(self, target_type: str) -> List:
        """Get appropriate estimators for the task type."""
        if target_type == "regr":
            return [
                RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42),
                Lasso(random_state=42),
                Ridge(alpha=1.0, random_state=42),
                ElasticNet(random_state=42),
            ]
        else:  # target_type == "class"
            return [
                LogisticRegression(
                    multi_class="auto", solver="lbfgs", max_iter=500, random_state=42
                ),
                RandomForestClassifier(n_estimators=10, random_state=42),
                DecisionTreeClassifier(random_state=42),
                MLPClassifier(
                    [50, 50],
                    solver="adam",
                    activation="relu",
                    learning_rate="adaptive",
                    random_state=42,
                ),
            ]

    def _fit_estimators(
        self,
        estimators: List,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        data_type: str,
    ):
        """Fit estimators to training data."""
        if self.verbose:
            print(f"\nFitting {data_type}")

        for i, estimator in enumerate(estimators):
            if self.verbose:
                print(f"{i + 1}: {type(estimator).__name__}")
            estimator.fit(x_train, y_train)

    def _score_estimators(
        self,
        r_estimators: List,
        f_estimators: List,
        estimator_names: List[str],
        real_x_test: pd.DataFrame,
        real_y_test: pd.Series,
        fake_x_test: pd.DataFrame,
        fake_y_test: pd.Series,
        target_type: str,
    ) -> pd.DataFrame:
        """Score estimators on test data."""
        if target_type == "class":
            return self._score_classification(
                r_estimators,
                f_estimators,
                estimator_names,
                real_x_test,
                real_y_test,
                fake_x_test,
                fake_y_test,
            )
        else:
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
        r_estimators: List,
        f_estimators: List,
        estimator_names: List[str],
        real_x_test: pd.DataFrame,
        real_y_test: pd.Series,
        fake_x_test: pd.DataFrame,
        fake_y_test: pd.Series,
    ) -> pd.DataFrame:
        """Score classification estimators."""
        rows = []

        for r_classifier, f_classifier, estimator_name in zip(
            r_estimators, f_estimators, estimator_names
        ):
            for x_test, y_test, dataset_name in [
                (real_x_test, real_y_test, "real"),
                (fake_x_test, fake_y_test, "fake"),
            ]:
                pred_real = r_classifier.predict(x_test)
                pred_fake = f_classifier.predict(x_test)

                f1_real = f1_score(y_test, pred_real, average="micro")
                f1_fake = f1_score(y_test, pred_fake, average="micro")
                jaccard_sim = jaccard_score(pred_real, pred_fake, average="micro")

                row = {
                    "index": f"{estimator_name}_{dataset_name}",
                    "f1_real": f1_real,
                    "f1_fake": f1_fake,
                    "jaccard_similarity": jaccard_sim,
                }
                rows.append(row)

        return pd.DataFrame(rows).set_index("index")

    def _score_regression(
        self,
        r_estimators: List,
        f_estimators: List,
        estimator_names: List[str],
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

        index = [f"real_data_{classifier}" for classifier in estimator_names] + [
            f"fake_data_{classifier}" for classifier in estimator_names
        ]

        return pd.DataFrame({"real": r2r + f2r, "fake": r2f + f2f}, index=index)
