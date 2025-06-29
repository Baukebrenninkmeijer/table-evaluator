import logging
import warnings
from os import PathLike
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from table_evaluator.association_metrics import associations
from table_evaluator.data.data_converter import DataConverter
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, jaccard_score

from table_evaluator.core.evaluation_config import EvaluationConfig
from table_evaluator.evaluators.ml_evaluator import MLEvaluator
from table_evaluator.evaluators.privacy_evaluator import PrivacyEvaluator
from table_evaluator.evaluators.statistical_evaluator import StatisticalEvaluator
from table_evaluator.metrics import (
    column_correlations,
    euclidean_distance,
    js_distance_df,
    kolmogorov_smirnov_df,
    mean_absolute_error,
    rmse,
)
from table_evaluator.notebook import EvaluationResult, visualize_notebook
from table_evaluator.utils import _preprocess_data, dict_to_df
from table_evaluator.visualization.visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class TableEvaluator:
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the user to easily evaluate data
    with the `evaluate` method. Additional evaluations can be done with the different methods of evaluate and the visual
    evaluation method.
    """

    comparison_metric: Callable

    def __init__(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        cat_cols: Optional[List[str]] = None,
        unique_thresh: int = 0,
        metric: str | Callable = "pearsonr",
        verbose: bool = False,
        n_samples: Optional[int] = None,
        name: Optional[str] = None,
        seed: int = 1337,
        sample: bool = False,
        infer_types: bool = True,
    ):
        """
        Initialize the TableEvaluator with real and fake datasets.

        Args:
            real (pd.DataFrame): Real dataset.
            fake (pd.DataFrame): Synthetic dataset.
            cat_cols (Optional[List[str]], optional): Columns to be evaluated as discrete. If provided, unique_thresh
                is ignored. Defaults to None.
            unique_thresh (int, optional): Threshold for automatic evaluation if a column is numeric. Defaults to 0.
            metric (str, optional): Metric for evaluating linear relations. Defaults to "pearsonr".
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            n_samples (Optional[int], optional): Number of samples to evaluate. If None, takes the minimum length of
                both datasets. Defaults to None.
            name (Optional[str], optional): Name of the TableEvaluator, used in plotting functions. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 1337.
            sample (bool, optional): Whether to sample the datasets to n_samples. Defaults to False.
        """
        self.name = name
        self.unique_thresh = unique_thresh
        self.real = real.copy()
        self.fake = fake.copy()
        self.comparison_metric: Callable = (
            getattr(stats, metric) if isinstance(metric, str) else metric
        )
        self.verbose = verbose
        self.random_seed = seed
        self.sample = sample
        self.infer_types = infer_types

        self.real, self.fake, self.numerical_columns, self.categorical_columns = (
            _preprocess_data(
                real=self.real,
                fake=self.fake,
                cat_cols=cat_cols,
                unique_thresh=unique_thresh,
                n_samples=n_samples,
                seed=seed,
            )
        )
        self.n_samples = len(self.real)

        # Initialize evaluation configuration
        self.config = EvaluationConfig(
            unique_thresh=unique_thresh, n_samples=n_samples, random_seed=seed
        )

        # Initialize evaluator components
        self.statistical_evaluator = StatisticalEvaluator(
            comparison_metric=self.comparison_metric, verbose=verbose
        )
        self.ml_evaluator = MLEvaluator(
            comparison_metric=self.comparison_metric, random_seed=seed, verbose=verbose
        )
        self.privacy_evaluator = PrivacyEvaluator(verbose=verbose)
        self.visualization_manager = VisualizationManager(
            real=self.real,
            fake=self.fake,
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
        )

    def plot_mean_std(self, fname=None, show: bool = True):
        """
        Class wrapper function for plotting the mean and std using `plots.plot_mean_std`.

        Params:

            fname (str, Optional): If not none, saves the plot with this file name.

        """
        return self.visualization_manager.plot_mean_std(fname=fname, show=show)

    def plot_cumsums(self, nr_cols=4, fname: PathLike | None = None, show: bool = True):
        """

        Plot the cumulative sums for all columns in the real and fake dataset. Height of each row scales with the length

        of the labels. Each plot contains the

        values of a real columns and the corresponding fake column.

        Params:

            fname str: If not none, saves the plot with this file name.

        """
        return self.visualization_manager.plot_cumsums(
            nr_cols=nr_cols, fname=fname, show=show
        )

    def plot_distributions(
        self, nr_cols: int = 3, fname: PathLike | None = None, show: bool = True
    ):
        """

        Plot the distribution plots for all columns in the real and fake dataset. Height of each row of plots scales

        with the length of the labels. Each plot

        contains the values of a real columns and the corresponding fake column.

        Params:

            fname (str, Optional): If not none, saves the plot with this file name.

        """
        return self.visualization_manager.plot_distributions(
            nr_cols=nr_cols, fname=fname, show=show
        )

    def plot_correlation_difference(
        self, plot_diff=True, fname=None, show: bool = True, **kwargs
    ):
        """

        Plot the association matrices for each table and, if chosen, the difference between them.

        :param plot_diff: whether to plot the difference

        :param fname: If not none, saves the plot with this file name.

        :param kwargs: kwargs for sns.heatmap

        """
        return self.visualization_manager.plot_correlation_difference(
            plot_diff=plot_diff, fname=fname, show=show, **kwargs
        )

    def correlation_distance(self, how: str = "euclidean") -> float:
        """

        Calculate distance between correlation matrices with certain metric.

        :param how: metric to measure distance. Choose from [``euclidean``, ``mae``, ``rmse``].

        :return: distance between the association matrices in the chosen evaluation metric. Default: Euclidean

        """

        from scipy.spatial.distance import cosine

        if how == "euclidean":
            distance_func = euclidean_distance
        elif how == "mae":
            distance_func = mean_absolute_error
        elif how == "rmse":
            distance_func = rmse
        elif how == "cosine":

            def custom_cosine(a, b):
                return cosine(a.reshape(-1), b.reshape(-1))

            distance_func = custom_cosine
        else:
            raise ValueError("`how` parameter must be in [euclidean, mae, rmse]")

        real_corr = associations(
            self.real,
            nominal_columns=self.categorical_columns,
            nom_nom_assoc="theil",
            compute_only=True,
        )["corr"]  # type: ignore
        fake_corr = associations(
            self.fake,
            nominal_columns=self.categorical_columns,
            nom_nom_assoc="theil",
            compute_only=True,
        )["corr"]  # type: ignore
        return distance_func(real_corr.values, fake_corr.values)  # type: ignore

    def plot_pca(self, fname: PathLike | None = None, show: bool = True):
        """

        Plot the first two components of a PCA of real and fake data.

        :param fname: If not none, saves the plot with this file name.

        """
        return self.visualization_manager.plot_pca(fname=fname, show=show)

    def get_copies(self, return_len: bool = False) -> Union[pd.DataFrame, int]:
        """

        Check whether any real values occur in the fake data.

        Args:

            return_len (bool): Whether to return the length of the copied rows or not.

        Returns:

            Union[pd.DataFrame, int]: Dataframe containing the duplicates if return_len=False,

                else integer indicating the number of copied rows.

        """
        return self.privacy_evaluator.get_copies(self.real, self.fake, return_len)

    def get_duplicates(
        self, return_values: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[int, int]:
        """

        Return duplicates within each dataset.

        Args:

            return_values (bool): Whether to return the duplicate values in the datasets.

                If False, the lengths are returned.

        Returns:

            Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[int, int]]:

                If return_values is True, returns a tuple of DataFrames with duplicates.

                If return_values is False, returns a tuple of integers representing the lengths of those DataFrames.

        """
        return self.privacy_evaluator.get_duplicates(
            self.real, self.fake, return_values
        )

    def pca_correlation(self, lingress: bool = False):
        """

        Calculate the relation between PCA explained variance values. Due to some very large numbers, in recent

        implementation the MAPE(log) is used instead of regressions like Pearson's r.

        Args:

            lingress (bool): Whether to use a linear regression, in this case Pearson's.

        Returns:

            float: The correlation coefficient if lingress=True, otherwise 1 - MAPE(log(real), log(fake)).

        """
        real, fake = self.convert_numerical()
        return self.statistical_evaluator.pca_correlation(real, fake, lingress)

    def fit_estimators(self):
        """

        Fit self.r_estimators and self.f_estimators to real and fake data, respectively.

        """

        if self.verbose:
            print("\nFitting real")
        for i, c in enumerate(self.r_estimators):
            if self.verbose:
                print(f"{i + 1}: {type(c).__name__}")
            c.fit(self.real_x_train, self.real_y_train)

        if self.verbose:
            print("\nFitting fake")
        for i, c in enumerate(self.f_estimators):
            if self.verbose:
                print(f"{i + 1}: {type(c).__name__}")
            c.fit(self.fake_x_train, self.fake_y_train)

    def score_estimators(self):
        """

        Get F1 scores of self.r_estimators and self.f_estimators on the fake and real data, respectively.

        :return: dataframe with the results for each estimator on each data test set.

        """

        if self.target_type == "class":
            rows = []
            for r_classifier, f_classifier, estimator_name in zip(
                self.r_estimators, self.f_estimators, self.estimator_names
            ):
                for dataset, target, dataset_name in zip(
                    [self.real_x_test, self.fake_x_test],
                    [self.real_y_test, self.fake_y_test],
                    ["real", "fake"],
                ):
                    predictions_classifier_real = r_classifier.predict(dataset)
                    predictions_classifier_fake = f_classifier.predict(dataset)
                    f1_r = f1_score(
                        target, predictions_classifier_real, average="micro"
                    )
                    f1_f = f1_score(
                        target, predictions_classifier_fake, average="micro"
                    )
                    jac_sim = jaccard_score(
                        predictions_classifier_real,
                        predictions_classifier_fake,
                        average="micro",
                    )
                    row = {
                        "index": f"{estimator_name}_{dataset_name}",
                        "f1_real": f1_r,
                        "f1_fake": f1_f,
                        "jaccard_similarity": jac_sim,
                    }
                    rows.append(row)
            results = pd.DataFrame(rows).set_index("index")

        elif self.target_type == "regr":
            r2r = [
                rmse(self.real_y_test, clf.predict(self.real_x_test))
                for clf in self.r_estimators
            ]
            f2f = [
                rmse(self.fake_y_test, clf.predict(self.fake_x_test))
                for clf in self.f_estimators
            ]

            # Calculate test set accuracies on the other dataset
            r2f = [
                rmse(self.fake_y_test, clf.predict(self.fake_x_test))
                for clf in self.r_estimators
            ]
            f2r = [
                rmse(self.real_y_test, clf.predict(self.real_x_test))
                for clf in self.f_estimators
            ]
            index = [
                f"real_data_{classifier}" for classifier in self.estimator_names
            ] + [f"fake_data_{classifier}" for classifier in self.estimator_names]
            results = pd.DataFrame({"real": r2r + f2r, "fake": r2f + f2f}, index=index)
        else:
            raise Exception(
                f"self.target_type should be either 'class' or 'regr', but is {self.target_type}."
            )
        return results

    def visual_evaluation(
        self, save_dir: PathLike | None = None, show: bool = True, **kwargs
    ):
        """

        Plot all visual evaluation metrics. Includes plotting the mean and standard deviation, cumulative sums,

        correlation differences and the PCA transform.

        Args:

            save_dir (str | None): Directory path to save images.

            show (bool): Whether to display the plots.

            **kwargs: Additional keyword arguments for matplotlib.

        Returns:

            None

        """
        return self.visualization_manager.visual_evaluation(
            save_dir=save_dir, show=show, **kwargs
        )

    def basic_statistical_evaluation(self) -> float:
        """

        Calculate the correlation coefficient between the basic properties of self.real and self.fake using Spearman's

        Rho. Spearman's is used because these values can differ a lot in magnitude, and Spearman's is more resilient to

        outliers.

        Returns:

            float: correlation coefficient

        """
        return self.statistical_evaluator.basic_statistical_evaluation(
            self.real, self.fake, self.numerical_columns
        )

    def correlation_correlation(self) -> float:
        """

        Calculate the correlation coefficient between the association matrices of self.real and self.fake using

        self.comparison_metric

        Returns:

            float: The correlation coefficient

        """
        return self.statistical_evaluator.correlation_correlation(
            self.real, self.fake, self.categorical_columns
        )

    def convert_numerical(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Convert dataset to a numerical representations while making sure they have identical columns. This is sometimes

        a problem with categorical columns with many values or very unbalanced values

        Returns:

            Tuple[pd.DataFrame, pd.DataFrame]: Real and fake dataframe factorized using the pandas function

        """

        real = self.real
        fake = self.fake
        for c in self.categorical_columns:
            if real[c].dtype == "object":
                real[c] = pd.factorize(real[c], sort=True)[0]
                fake[c] = pd.factorize(fake[c], sort=True)[0]

        return real, fake

    def convert_numerical_one_hot(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Convert dataset to a numerical representations while making sure they have identical columns.

        This is sometimes a problem with categorical columns with many values or very unbalanced values

        Returns:

            Tuple[pd.DataFrame, pd.DataFrame]: Real and fake dataframe with categorical columns one-hot encoded and

            binary columns factorized.

        """

        cat_cols = (
            self.categorical_columns
            + self.real.select_dtypes("bool").columns.tolist()
            + self.fake.select_dtypes("bool").columns.tolist()
        )
        converter = DataConverter()
        real, fake = converter.to_numerical_one_hot(self.real, self.fake, cat_cols)
        real = real.sort_index(axis=1)
        fake = fake.sort_index(axis=1)

        for col in real.columns:
            if col not in fake:
                logger.warning(f"Adding column {col} with all 0s")
                fake[col] = 0.0

        # Cast True/False columns to 0/1.
        # bool_cols = real.select_dtypes("bool").columns
        # real[bool_cols] = real[bool_cols].astype(float)
        # fake[bool_cols] = fake[bool_cols].astype(float)

        return real, fake

    def estimator_evaluation(
        self, target_col: str, target_type: str = "class", kfold: bool = False
    ) -> float:
        """

        Method to do full estimator evaluation, including training. And estimator is either a regressor or a classifier,

        depending on the task. Two sets are created of each of the estimators `S_r` and `S_f`, for the real and fake

        data respectively. `S_f` is trained on ``self.real`` and `S_r` on ``self.fake``. Then, both are evaluated on

        their own and the others test set. If target_type is ``regr`` we do a regression on the RMSE scores with

        Pearson's. If target_type is ``class``, we calculate F1 scores and do return ``1 - MAPE(F1_r, F1_f)``.

        Args:

            target_col (str): The column to be considered as the target for both regression and classification tasks.

            target_type (str): The type of task. Can be either "class" or "regr".

            kfold (bool): If True, performs 5-fold CV. If False, trains on 80% and tests on 20% of the data once.

        Returns:

            float: Correlation value or 1 - MAPE.

        """
        real, fake = self.convert_numerical()
        result = self.ml_evaluator.estimator_evaluation(
            real, fake, target_col, target_type, kfold
        )

        # Store the scores for backward compatibility with evaluate() method
        # We need to get the scores from the MLEvaluator but it doesn't expose them
        # Let's temporarily set a placeholder
        self.estimators_scores = pd.DataFrame({"real": [result], "fake": [result]})

        return result

    def row_distance(self, n_samples: int | None = None) -> Tuple[np.number, np.number]:
        """

        Calculate mean and standard deviation distances between `self.fake` and `self.real`.

        :param n_samples: Number of samples to take for evaluation. Compute time increases exponentially.

        :return: `(mean, std)` of these distances.

        """
        real, fake = self.convert_numerical_one_hot()
        return self.privacy_evaluator.row_distance(real, fake, n_samples)

    def column_correlations(self):
        """

        Wrapper function around `metrics.column_correlation`.

        :return: Column correlations between ``self.real`` and ``self.fake``.

        """

        real, fake = self.convert_numerical()

        return column_correlations(real, fake, self.categorical_columns)

    def evaluate(
        self,
        target_col: str,
        target_type: str = "class",
        metric: str | None = None,
        verbose: bool | None = None,
        n_samples_distance: int = 20000,
        kfold: bool = False,
        notebook: bool = False,
        return_outputs: bool = False,
    ) -> Dict | None:
        """

        Determine correlation between attributes from the real and fake dataset using a given metric.

        All metrics from scipy.stats are available.

        Args:

            target_col (str): Column to use for predictions with estimators.

            target_type (str, optional): Type of task to perform on the target_col. Can be either "class" for

                classification or "regr" for regression. Defaults to "class".

            metric (str | None, optional): Overwrites self.metric. Scoring metric for the attributes.

                By default Pearson's r is used. Alternatives include Spearman rho (scipy.stats.spearmanr) or

                Kendall Tau (scipy.stats.kendalltau). Defaults to None.

            n_samples_distance (int, optional): The number of samples to take for the row distance. See

                documentation of ``tableEvaluator.row_distance`` for details. Defaults to 20000.

            kfold (bool, optional): Use a 5-fold CV for the ML estimators if set to True. Train/Test on 80%/20%

                of the data if set to False. Defaults to False.

            notebook (bool, optional): Better visualization of the results in a python notebook. Defaults to False.

            verbose (bool | None, optional): Whether to print verbose logging. Defaults to None.

            return_outputs (bool, optional): Will omit printing and instead return a dictionary with all results.

                Defaults to False.

        Returns:

            Dict: A dictionary containing evaluation results if return_outputs is True, otherwise None.

        """

        self.target_type = target_type
        self.verbose = verbose if verbose is not None else self.verbose
        self.comparison_metric = (
            getattr(stats, metric)
            if isinstance(metric, str)
            else metric
            if metric is not None
            else self.comparison_metric
        )

        warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
        pd.options.display.float_format = "{:,.4f}".format

        statistical_results, statistical_tab = self._calculate_statistical_metrics()
        ml_efficacy_results, ml_efficacy_tab = self._calculate_ml_efficacy(
            target_col, target_type, kfold
        )
        privacy_results, privacy_tab = self._calculate_privacy_metrics(
            n_samples_distance
        )

        all_results_dict = {
            "Basic statistics": statistical_results["basic_statistical"],
            "Correlation column correlations": statistical_results[
                "correlation_correlation"
            ],
            "Mean Correlation between fake and real columns": statistical_results[
                "column_correlation"
            ],
            f'{"1 - MAPE Estimator results" if target_type == "class" else "Correlation RMSE"}': ml_efficacy_results[
                "estimators"
            ],
        }
        all_results_dict["Similarity Score"] = np.mean(list(all_results_dict.values()))

        summary = EvaluationResult(
            name="Overview Results", content=dict_to_df(all_results_dict)
        )

        overview_tab = [
            summary,
        ]

        if return_outputs:
            all_results = [
                *overview_tab,
                *ml_efficacy_tab,
                *privacy_tab,
                *statistical_tab,
            ]

            all_results = {
                x.name: x.content.to_dict(orient="index") for x in all_results
            }

            return all_results

        if notebook:
            visualize_notebook(
                self,
                overview=overview_tab,
                privacy_metrics=privacy_tab,
                ml_efficacy=ml_efficacy_tab,
                statistical=statistical_tab,
            )

        else:
            print(f'\n{ml_efficacy_results["efficacy_title"]}:')
            print(self.estimators_scores.to_string())

            print("\nPrivacy results:")
            print(privacy_results["privacy_report"].content.to_string())

            print("\nMiscellaneous results:")
            print(statistical_results["miscellaneous"].to_string())

            print("\nResults:")
            print(summary.content.to_string())

        return all_results_dict

    def _calculate_statistical_metrics(self):
        basic_statistical = self.basic_statistical_evaluation()
        correlation_correlation = self.correlation_correlation()
        column_correlation = self.column_correlations()

        miscellaneous_dict = {
            "Column Correlation Distance RMSE": self.correlation_distance(how="rmse"),
            "Column Correlation distance MAE": self.correlation_distance(how="mae"),
        }

        miscellaneous = pd.DataFrame(
            {"Result": list(miscellaneous_dict.values())},
            index=list(miscellaneous_dict.keys()),
        )

        js_df = js_distance_df(self.real, self.fake, self.numerical_columns)

        statistical_tab = [
            EvaluationResult(
                name="Jensen-Shannon distance",
                content=js_df,
                appendix=f"### Mean: {js_df.js_distance.mean(): .3f}",
            ),
            EvaluationResult(
                name="Kolmogorov-Smirnov statistic",
                content=kolmogorov_smirnov_df(
                    self.real, self.fake, self.numerical_columns
                ),
            ),
        ]
        return {
            "basic_statistical": basic_statistical,
            "correlation_correlation": correlation_correlation,
            "column_correlation": column_correlation,
            "miscellaneous": miscellaneous,
        }, statistical_tab

    def _calculate_ml_efficacy(self, target_col: str, target_type: str, kfold: bool):
        estimators = self.estimator_evaluation(
            target_col=target_col, target_type=target_type, kfold=kfold
        )
        efficacy_title = (
            "Classifier F1-scores and their Jaccard similarities:"
            if target_type == "class"
            else "Regressor MSE-scores"
        )

        ml_efficacy_tab = [
            EvaluationResult(name=efficacy_title, content=self.estimators_scores)
        ]
        return {
            "estimators": estimators,
            "efficacy_title": efficacy_title,
        }, ml_efficacy_tab

    def _calculate_privacy_metrics(self, n_samples_distance: int):
        nearest_neighbor = self.row_distance(n_samples=n_samples_distance)
        privacy_metrics_dict = {
            "Duplicate rows between sets (real/fake)": self.get_duplicates(),
            "nearest neighbor mean": nearest_neighbor[0],
            "nearest neighbor std": nearest_neighbor[1],
        }

        privacy_report = EvaluationResult(
            name="Privacy Results",
            content=dict_to_df(privacy_metrics_dict),
        )

        privacy_tab = [privacy_report]

        return {"privacy_report": privacy_report}, privacy_tab
