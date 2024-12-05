import copy
import logging
import warnings
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dython.nominal import associations, numerical_encoding
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from table_evaluator.metrics import (
    column_correlations,
    euclidean_distance,
    js_distance_df,
    kolmogorov_smirnov_df,
    mean_absolute_error,
    mean_absolute_percentage_error,
    rmse,
)
from table_evaluator.notebook import EvaluationResult, visualize_notebook
from table_evaluator.plots import cdf, plot_correlation_difference, plot_mean_std
from table_evaluator.utils import dict_to_df

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
        metric: str | Callable = 'pearsonr',
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
        self.comparison_metric: Callable = getattr(stats, metric) if isinstance(metric, str) else metric
        self.verbose = verbose
        self.random_seed = seed
        self.sample = sample
        self.infer_types = infer_types

        self._validate_dataframes()
        self.numerical_columns, self.categorical_columns = self._determine_columns(cat_cols)

        self.n_samples = self._set_sample_size(n_samples)
        if sample:
            self.real = self.real.sample(self.n_samples, random_state=self.random_seed)
            self.fake = self.fake.sample(self.n_samples, random_state=self.random_seed)

        self._fill_missing_values()

    def _validate_dataframes(self):
        """Ensure that the real and fake dataframes have the same columns."""
        if len(self.real.columns) != len(self.fake.columns):
            raise ValueError('Columns in real and fake dataframe are not the same')
        self.fake = self.fake[self.real.columns.tolist()]

    def _determine_columns(self, cat_cols: Optional[List[str]]) -> Tuple[List[str], List[str]]:
        """Determine numerical and categorical columns based on the provided data."""
        if cat_cols is None:
            if self.infer_types:
                self.real = self.real.infer_objects()
                self.fake = self.fake.infer_objects()
            numerical_columns = [
                column
                for column in self.real.select_dtypes(include='number').columns
                if self.real[column].nunique() > self.unique_thresh
            ]
            categorical_columns = [column for column in self.real.columns if column not in numerical_columns]
        else:
            categorical_columns = cat_cols
            numerical_columns = [column for column in self.real.columns if column not in cat_cols]
        return numerical_columns, categorical_columns

    def _set_sample_size(self, n_samples: Optional[int]) -> int:
        """Set the number of samples to evaluate."""
        if n_samples is None:
            return min(len(self.real), len(self.fake))
        elif len(self.fake) >= n_samples and len(self.real) >= n_samples:
            return n_samples
        else:
            raise ValueError(
                f'Make sure n_samples < len(fake/real). len(real): {len(self.real)}, len(fake): {len(self.fake)}'
            )

    def _fill_missing_values(self):
        """Fill missing values in the datasets."""
        self.real[self.categorical_columns] = self.real[self.categorical_columns].astype(str).fillna('[NAN]')
        self.fake[self.categorical_columns] = self.fake[self.categorical_columns].astype(str).fillna('[NAN]')

        self.real[self.numerical_columns] = self.real[self.numerical_columns].fillna(
            self.real[self.numerical_columns].mean()
        )
        self.fake[self.numerical_columns] = self.fake[self.numerical_columns].fillna(
            self.fake[self.numerical_columns].mean()
        )

    def plot_mean_std(self, fname=None, show: bool = True):
        """
        Class wrapper function for plotting the mean and std using `plots.plot_mean_std`.

        Params:
            fname (str, Optional): If not none, saves the plot with this file name.
        """
        plot_mean_std(self.real, self.fake, fname=fname, show=show)

    def plot_cumsums(self, nr_cols=4, fname: PathLike | None = None, show: bool = True):
        """
        Plot the cumulative sums for all columns in the real and fake dataset. Height of each row scales with the length
        of the labels. Each plot contains the
        values of a real columns and the corresponding fake column.

        Params:
            fname str: If not none, saves the plot with this file name.
        """
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.real.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.real.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.real[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Cumulative Sums per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.real.columns):
            try:
                r = self.real[col]
                f = self.fake.iloc[:, self.real.columns.tolist().index(col)]
                cdf(r, f, col, 'Cumsum', ax=axes[i])
            except Exception as e:
                print(f'Error while plotting column {col}')
                raise e

        plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))

        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            return fig

    def plot_distributions(self, nr_cols: int = 3, fname: PathLike | None = None, show: bool = True):
        """
        Plot the distribution plots for all columns in the real and fake dataset. Height of each row of plots scales
        with the length of the labels. Each plot
        contains the values of a real columns and the corresponding fake column.

        Params:
            fname (str, Optional): If not none, saves the plot with this file name.
        """
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.real.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.real.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.real[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Distribution per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.real.columns):
            if col not in self.categorical_columns:
                plot_df = pd.DataFrame(
                    {
                        col: pd.concat([self.real[col], self.fake[col]], axis=0),
                        'kind': ['real'] * len(self.real) + ['fake'] * len(self.fake),
                    }
                )
                fig = sns.histplot(
                    plot_df,
                    x=col,
                    hue='kind',
                    ax=axes[i],
                    stat='probability',
                    legend=True,
                    kde=True,
                )
                axes[i].set_autoscaley_on(True)
            else:
                real = self.real.copy()
                fake = self.fake.copy()
                real['kind'] = 'Real'
                fake['kind'] = 'Fake'
                concat = pd.concat([fake, real])
                palette = sns.color_palette(
                    [
                        (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                        (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
                    ]
                )
                x, y, hue = col, 'proportion', 'kind'
                ax = (
                    concat[x]
                    .groupby(concat[hue])
                    .value_counts(normalize=True)
                    .rename(y)
                    .reset_index()
                    .pipe(
                        (sns.barplot, 'data'),
                        x=x,
                        y=y,
                        hue=hue,
                        ax=axes[i],
                        saturation=0.8,
                        palette=palette,
                    )
                )
                ax.set_xticklabels(axes[i].get_xticklabels(), rotation='vertical')
        plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))

        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            return fig

    def plot_correlation_difference(self, plot_diff=True, fname=None, show: bool = True, **kwargs):
        """
        Plot the association matrices for each table and, if chosen, the difference between them.

        :param plot_diff: whether to plot the difference
        :param fname: If not none, saves the plot with this file name.
        :param kwargs: kwargs for sns.heatmap
        """
        plot_correlation_difference(
            self.real,
            self.fake,
            cat_cols=self.categorical_columns,
            plot_diff=plot_diff,
            fname=fname,
            show=show,
            **kwargs,
        )

    def correlation_distance(self, how: str = 'euclidean') -> float:
        """
        Calculate distance between correlation matrices with certain metric.

        :param how: metric to measure distance. Choose from [``euclidean``, ``mae``, ``rmse``].
        :return: distance between the association matrices in the chosen evaluation metric. Default: Euclidean
        """
        from scipy.spatial.distance import cosine

        if how == 'euclidean':
            distance_func = euclidean_distance
        elif how == 'mae':
            distance_func = mean_absolute_error
        elif how == 'rmse':
            distance_func = rmse
        elif how == 'cosine':

            def custom_cosine(a, b):
                return cosine(a.reshape(-1), b.reshape(-1))

            distance_func = custom_cosine
        else:
            raise ValueError('`how` parameter must be in [euclidean, mae, rmse]')

        real_corr = associations(
            self.real,
            nominal_columns=self.categorical_columns,
            nom_nom_assoc='theil',
            compute_only=True,
        )['corr']  # type: ignore
        fake_corr = associations(
            self.fake,
            nominal_columns=self.categorical_columns,
            nom_nom_assoc='theil',
            compute_only=True,
        )['corr']  # type: ignore
        return distance_func(real_corr.values, fake_corr.values)  # type: ignore

    def plot_pca(self, fname: PathLike | None = None, show: bool = True):
        """
        Plot the first two components of a PCA of real and fake data.
        :param fname: If not none, saves the plot with this file name.
        """
        real, fake = self.convert_numerical()

        pca_r = PCA(n_components=2)
        pca_f = PCA(n_components=2)

        real_t = pca_r.fit_transform(real)
        fake_t = pca_f.fit_transform(fake)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('First two components of PCA', fontsize=16)
        sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
        sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
        ax[0].set_title('Real data')
        ax[1].set_title('Fake data')

        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            return fig

    def get_copies(self, return_len: bool = False) -> Union[pd.DataFrame, int]:
        """
        Check whether any real values occur in the fake data.

        Args:
            return_len (bool): Whether to return the length of the copied rows or not.

        Returns:
            Union[pd.DataFrame, int]: Dataframe containing the duplicates if return_len=False,
                else integer indicating the number of copied rows.
        """
        real_hashes = self.real.apply(lambda x: hash(tuple(x)), axis=1)
        fake_hashes = self.fake.apply(lambda x: hash(tuple(x)), axis=1)

        dup_idxs = fake_hashes.isin(real_hashes.values)
        print(dup_idxs)
        dup_idxs = dup_idxs[dup_idxs].sort_index().index.tolist()

        if self.verbose:
            print(f'Nr copied columns: {len(dup_idxs)}')
        copies = self.fake.loc[dup_idxs, :]

        if return_len:
            return len(copies)
        else:
            return copies

    def get_duplicates(self, return_values: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[int, int]:
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
        real_duplicates = self.real[self.real.duplicated(keep=False)]
        fake_duplicates = self.fake[self.fake.duplicated(keep=False)]
        if return_values:
            return real_duplicates, fake_duplicates
        else:
            return len(real_duplicates), len(fake_duplicates)

    def pca_correlation(self, lingress: bool = False):
        """
        Calculate the relation between PCA explained variance values. Due to some very large numbers, in recent
        implementation the MAPE(log) is used instead of regressions like Pearson's r.

        Args:
            lingress (bool): Whether to use a linear regression, in this case Pearson's.

        Returns:
            float: The correlation coefficient if lingress=True, otherwise 1 - MAPE(log(real), log(fake)).
        """
        self.pca_r = PCA(n_components=5)
        self.pca_f = PCA(n_components=5)

        real, fake = self.convert_numerical()

        self.pca_r.fit(real)
        self.pca_f.fit(fake)
        if self.verbose:
            results = pd.DataFrame(
                {
                    'real': self.pca_r.explained_variance_,
                    'fake': self.pca_f.explained_variance_,
                }
            )
            print('\nTop 5 PCA components:')
            print(results.to_string())

        if lingress:
            corr, p, _ = self.comparison_metric(self.pca_r.explained_variance_, self.pca_f.explained_variance_)
            return corr
        else:
            pca_error = mean_absolute_percentage_error(self.pca_r.explained_variance_, self.pca_f.explained_variance_)
            return 1 - pca_error

    def fit_estimators(self):
        """
        Fit self.r_estimators and self.f_estimators to real and fake data, respectively.
        """

        if self.verbose:
            print('\nFitting real')
        for i, c in enumerate(self.r_estimators):
            if self.verbose:
                print(f'{i + 1}: {type(c).__name__}')
            c.fit(self.real_x_train, self.real_y_train)

        if self.verbose:
            print('\nFitting fake')
        for i, c in enumerate(self.f_estimators):
            if self.verbose:
                print(f'{i + 1}: {type(c).__name__}')
            c.fit(self.fake_x_train, self.fake_y_train)

    def score_estimators(self):
        """
        Get F1 scores of self.r_estimators and self.f_estimators on the fake and real data, respectively.

        :return: dataframe with the results for each estimator on each data test set.
        """
        if self.target_type == 'class':
            rows = []
            for r_classifier, f_classifier, estimator_name in zip(
                self.r_estimators, self.f_estimators, self.estimator_names
            ):
                for dataset, target, dataset_name in zip(
                    [self.real_x_test, self.fake_x_test],
                    [self.real_y_test, self.fake_y_test],
                    ['real', 'fake'],
                ):
                    predictions_classifier_real = r_classifier.predict(dataset)
                    predictions_classifier_fake = f_classifier.predict(dataset)
                    f1_r = f1_score(target, predictions_classifier_real, average='micro')
                    f1_f = f1_score(target, predictions_classifier_fake, average='micro')
                    jac_sim = jaccard_score(
                        predictions_classifier_real,
                        predictions_classifier_fake,
                        average='micro',
                    )
                    row = {
                        'index': f'{estimator_name}_{dataset_name}',
                        'f1_real': f1_r,
                        'f1_fake': f1_f,
                        'jaccard_similarity': jac_sim,
                    }
                    rows.append(row)
            results = pd.DataFrame(rows).set_index('index')

        elif self.target_type == 'regr':
            r2r = [rmse(self.real_y_test, clf.predict(self.real_x_test)) for clf in self.r_estimators]
            f2f = [rmse(self.fake_y_test, clf.predict(self.fake_x_test)) for clf in self.f_estimators]

            # Calculate test set accuracies on the other dataset
            r2f = [rmse(self.fake_y_test, clf.predict(self.fake_x_test)) for clf in self.r_estimators]
            f2r = [rmse(self.real_y_test, clf.predict(self.real_x_test)) for clf in self.f_estimators]
            index = [f'real_data_{classifier}' for classifier in self.estimator_names] + [
                f'fake_data_{classifier}' for classifier in self.estimator_names
            ]
            results = pd.DataFrame({'real': r2r + f2r, 'fake': r2f + f2f}, index=index)
        else:
            raise Exception(f"self.target_type should be either 'class' or 'regr', but is {self.target_type}.")
        return results

    def visual_evaluation(self, save_dir: PathLike | None = None, show: bool = True, **kwargs):
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
        if save_dir is None:
            self.plot_mean_std(show=show)
            self.plot_cumsums(show=show)
            self.plot_distributions(show=show)
            self.plot_correlation_difference(show=show, **kwargs)
            self.plot_pca(show=show)
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            self.plot_mean_std(fname=save_dir / 'mean_std.png', show=show)
            self.plot_cumsums(fname=save_dir / 'cumsums.png', show=show)
            self.plot_distributions(fname=save_dir / 'distributions.png', show=show)
            self.plot_correlation_difference(fname=save_dir / 'correlation_difference.png', show=show, **kwargs)
            self.plot_pca(fname=save_dir / 'pca.png', show=show)

    def basic_statistical_evaluation(self) -> float:
        """
        Calculate the correlation coefficient between the basic properties of self.real and self.fake using Spearman's
        Rho. Spearman's is used because these values can differ a lot in magnitude, and Spearman's is more resilient to
        outliers.

        Returns:
            float: correlation coefficient
        """
        total_metrics = pd.DataFrame()
        for ds_name in ['real', 'fake']:
            ds = getattr(self, ds_name)
            metrics = {}
            # TODO: add discrete columns as factors
            num_ds = ds[self.numerical_columns]

            for idx, value in num_ds.mean().items():
                metrics[f'mean_{idx}'] = value
            for idx, value in num_ds.median().items():
                metrics[f'median_{idx}'] = value
            for idx, value in num_ds.std().items():
                metrics[f'std_{idx}'] = value
            for idx, value in num_ds.var().items():
                metrics[f'variance_{idx}'] = value
            total_metrics[ds_name] = metrics.values()

        total_metrics.index = list(metrics.keys())  # type: ignore
        self.statistical_results = total_metrics
        if self.verbose:
            print('\nBasic statistical attributes:')
            print(total_metrics.to_string())
        corr, p = stats.spearmanr(total_metrics['real'], total_metrics['fake'])
        return corr

    def correlation_correlation(self) -> float:
        """
        Calculate the correlation coefficient between the association matrices of self.real and self.fake using
        self.comparison_metric

        Returns:
            float: The correlation coefficient
        """
        total_metrics = pd.DataFrame()
        for ds_name in ['real', 'fake']:
            ds = getattr(self, ds_name)
            corr_df: pd.DataFrame = associations(
                ds,
                nominal_columns=self.categorical_columns,
                nom_nom_assoc='theil',
                compute_only=True,
            )['corr']  # type: ignore
            values = corr_df.values
            # print(values, type(values))
            values = values[~np.eye(values.shape[0], dtype=bool)].reshape(values.shape[0], -1)
            total_metrics[ds_name] = values.flatten()

        self.correlation_correlations = total_metrics
        corr, p = self.comparison_metric(total_metrics['real'], total_metrics['fake'])
        if self.verbose:
            print('\nColumn correlation between datasets:')
            print(total_metrics.to_string())
        return corr

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
            if real[c].dtype == 'object':
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
            + self.real.select_dtypes('bool').columns.tolist()
            + self.fake.select_dtypes('bool').columns.tolist()
        )
        real: pd.DataFrame = numerical_encoding(self.real, nominal_columns=cat_cols).astype(float)  # type: ignore
        real = real.sort_index(axis=1)
        fake: pd.DataFrame = numerical_encoding(self.fake, nominal_columns=cat_cols).astype(float)  # type: ignore
        for col in real.columns:
            if col not in fake:
                logger.warning(f'Adding column {col} with all 0s')
                fake[col] = 0.0
        fake = fake.sort_index(axis=1)

        # Cast True/False columns to 0/1.
        # bool_cols = real.select_dtypes("bool").columns
        # real[bool_cols] = real[bool_cols].astype(float)
        # fake[bool_cols] = fake[bool_cols].astype(float)

        return real, fake

    def estimator_evaluation(self, target_col: str, target_type: str = 'class', kfold: bool = False) -> float:
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
        self.target_col = target_col
        self.target_type = target_type

        # Convert both datasets to numerical representations and split x and  y
        real, fake = self.convert_numerical()

        real_x = real.drop([target_col], axis=1)
        fake_x = fake.drop([target_col], axis=1)

        assert (
            real_x.columns.tolist() == fake_x.columns.tolist()
        ), f'real and fake columns are different: \n{real_x.columns}\n{fake_x.columns}'

        real_y = real[target_col]
        fake_y = fake[target_col]

        # For reproducibilty:
        np.random.seed(self.random_seed)

        if target_type == 'regr':
            self.estimators = [
                RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42),
                Lasso(random_state=42),
                Ridge(alpha=1.0, random_state=42),
                ElasticNet(random_state=42),
            ]
        elif target_type == 'class':
            self.estimators = [
                LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500, random_state=42),
                RandomForestClassifier(n_estimators=10, random_state=42),
                DecisionTreeClassifier(random_state=42),
                MLPClassifier(
                    [50, 50],
                    solver='adam',
                    activation='relu',
                    learning_rate='adaptive',
                    random_state=42,
                ),
            ]
        else:
            raise ValueError("target_type must be 'regr' or 'class'")

        self.estimator_names = [type(clf).__name__ for clf in self.estimators]

        # K Fold
        kf = KFold(n_splits=5)
        res = []
        for train_index, test_index in kf.split(real_y):
            self.real_x_train = real_x.iloc[train_index]
            self.real_x_test = real_x.iloc[test_index]
            self.real_y_train = real_y.iloc[train_index]
            self.real_y_test = real_y.iloc[test_index]
            self.fake_x_train = fake_x.iloc[train_index]
            self.fake_x_test = fake_x.iloc[test_index]
            self.fake_y_train = fake_y.iloc[train_index]
            self.fake_y_test = fake_y.iloc[test_index]

            self.r_estimators = copy.deepcopy(self.estimators)
            self.f_estimators = copy.deepcopy(self.estimators)

            for estimator in self.estimators:
                assert hasattr(estimator, 'fit')
                assert hasattr(estimator, 'score')

            self.fit_estimators()
            res.append(self.score_estimators())

            # Break the loop if we don't want the kfold
            if not kfold:
                break
        print(res)
        self.estimators_scores = pd.concat(res).groupby(level=0).mean()
        if self.verbose:
            print('\nClassifier F1-scores and their Jaccard similarities:') if self.target_type == 'class' else print(
                '\nRegressor MSE-scores and their Jaccard similarities:'
            )
            print(self.estimators_scores.to_string())

        if self.target_type == 'regr':
            corr, p = self.comparison_metric(self.estimators_scores['real'], self.estimators_scores['fake'])
            return corr
        elif self.target_type == 'class':
            mean = mean_absolute_percentage_error(self.estimators_scores['f1_real'], self.estimators_scores['f1_fake'])
            return 1 - mean
        else:
            raise ValueError('`self.target_type` should be `regr` or `class`.')

    def row_distance(self, n_samples: int | None = None) -> Tuple[np.number, np.number]:
        """
        Calculate mean and standard deviation distances between `self.fake` and `self.real`.

        :param n_samples: Number of samples to take for evaluation. Compute time increases exponentially.
        :return: `(mean, std)` of these distances.
        """
        if n_samples is None:
            n_samples = len(self.real)

        real, fake = self.convert_numerical_one_hot()

        columns = sorted(real.columns.tolist())
        real = real[columns]

        for col in columns:
            if col not in fake.columns.tolist():
                fake[col] = 0
        fake = fake[columns]

        for column in real.columns.tolist():
            if len(real[column].unique()) > 2:
                real[column] = (real[column] - real[column].mean()) / real[column].std()
                fake[column] = (fake[column] - fake[column].mean()) / fake[column].std()
        assert real.columns.tolist() == fake.columns.tolist()

        distances = cdist(real[:n_samples], fake[:n_samples])
        min_distances = np.min(distances, axis=1)
        min_mean = np.mean(min_distances)
        min_std = np.std(min_distances)
        return min_mean, min_std

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
        target_type: str = 'class',
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

        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        pd.options.display.float_format = '{:,.4f}'.format

        basic_statistical = self.basic_statistical_evaluation()
        correlation_correlation = self.correlation_correlation()
        column_correlation = self.column_correlations()
        estimators = self.estimator_evaluation(target_col=target_col, target_type=target_type, kfold=kfold)
        nearest_neighbor = self.row_distance(n_samples=n_samples_distance)

        miscellaneous_dict = {
            'Column Correlation Distance RMSE': self.correlation_distance(how='rmse'),
            'Column Correlation distance MAE': self.correlation_distance(how='mae'),
        }

        miscellaneous = pd.DataFrame(
            {'Result': list(miscellaneous_dict.values())},
            index=list(miscellaneous_dict.keys()),
        )

        privacy_metrics_dict = {
            'Duplicate rows between sets (real/fake)': self.get_duplicates(),
            'nearest neighbor mean': nearest_neighbor[0],
            'nearest neighbor std': nearest_neighbor[1],
        }

        privacy_report = EvaluationResult(
            name='Privacy Results',
            content=dict_to_df(privacy_metrics_dict),
        )

        privacy_tab = [privacy_report]

        efficacy_title = (
            'Classifier F1-scores and their Jaccard similarities:'
            if self.target_type == 'class'
            else '\nRegressor MSE-scores'
        )

        ml_efficacy_tab = [EvaluationResult(name=efficacy_title, content=self.estimators_scores)]

        js_df = js_distance_df(self.real, self.fake, self.numerical_columns)

        statistical_tab = [
            EvaluationResult(
                name='Jensen-Shannon distance',
                content=js_df,
                appendix=f'### Mean: {js_df.js_distance.mean(): .3f}',
            ),
            EvaluationResult(
                name='Kolmogorov-Smirnov statistic',
                content=kolmogorov_smirnov_df(self.real, self.fake, self.numerical_columns),
            ),
        ]

        all_results_dict = {
            'Basic statistics': basic_statistical,
            'Correlation column correlations': correlation_correlation,
            'Mean Correlation between fake and real columns': column_correlation,
            f'{"1 - MAPE Estimator results" if self.target_type == "class" else "Correlation RMSE"}': estimators,
        }
        all_results_dict['Similarity Score'] = np.mean(list(all_results_dict.values()))

        summary = EvaluationResult(name='Overview Results', content=dict_to_df(all_results_dict))

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

            all_results = {x.name: x.content.to_dict(orient='index') for x in all_results}

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
            print(f'\n{efficacy_title}:')
            print(self.estimators_scores.to_string())

            print('\nPrivacy results:')
            print(privacy_report.content.to_string())

            print('\nMiscellaneous results:')
            print(miscellaneous.to_string())

            print('\nResults:')
            print(summary.content.to_string())
