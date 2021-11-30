import copy
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from typing import Tuple, Dict, Union
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_squared_error, jaccard_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from dython.nominal import compute_associations, numerical_encoding
from .viz import *
from .metrics import *
from .notebook import visualize_notebook, isnotebook, EvaluationResult
from .utils import dict_to_df


class TableEvaluator:
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the user to easily evaluate data with the `evaluate` method.
    Additional evaluations can be done with the different methods of evaluate and the visual evaluation method.
    """

    def __init__(self, real: pd.DataFrame, fake: pd.DataFrame, cat_cols=None, unique_thresh=0, metric='pearsonr',
                 verbose=False, n_samples=None, name: str = None, seed=1337):
        """
        :param real: Real dataset (pd.DataFrame)
        :param fake: Synthetic dataset (pd.DataFrame)
        :param unique_thresh: Threshold for automatic evaluation if column is numeric
        :param cat_cols: The columns that are to be evaluated as discrete. If passed, unique_thresh is ignored.
        :param metric: the metric to use for evaluation linear relations. Pearson's r by default, but supports all models in scipy.stats
        :param verbose: Whether to print verbose output
        :param n_samples: Number of samples to evaluate. If none, it will take the minimal length of both datasets and cut the larger one off to make sure they
            are the same length.
        :param name: Name of the TableEvaluator. Used in some plotting functions like `viz.plot_correlation_comparison` to indicate your model.
        """
        self.name = name
        self.unique_thresh = unique_thresh
        self.real = real.copy()
        self.fake = fake.copy()
        self.comparison_metric = getattr(stats, metric)
        self.verbose = verbose
        self.random_seed = seed

        # Make sure columns and their order are the same.
        if len(real.columns) == len(fake.columns):
            fake = fake[real.columns.tolist()]
        assert real.columns.tolist() == fake.columns.tolist(), 'Columns in real and fake dataframe are not the same'

        if cat_cols is None:
            real = real.infer_objects()
            fake = fake.infer_objects()
            self.numerical_columns = [column for column in real._get_numeric_data().columns if
                                      len(real[column].unique()) > unique_thresh]
            self.categorical_columns = [column for column in real.columns if column not in self.numerical_columns]
        else:
            self.categorical_columns = cat_cols
            self.numerical_columns = [column for column in real.columns if column not in cat_cols]

        # Make sure the number of samples is equal in both datasets.
        if n_samples is None:
            self.n_samples = min(len(self.real), len(self.fake))
        elif len(fake) >= n_samples and len(real) >= n_samples:
            self.n_samples = n_samples
        else:
            raise Exception(f'Make sure n_samples < len(fake/real). len(real): {len(real)}, len(fake): {len(fake)}')

        self.real = self.real.sample(self.n_samples)
        self.fake = self.fake.sample(self.n_samples)
        assert len(self.real) == len(self.fake), f'len(real) != len(fake)'

        self.real.loc[:, self.categorical_columns] = self.real.loc[:, self.categorical_columns].fillna('[NAN]').astype(
            str)
        self.fake.loc[:, self.categorical_columns] = self.fake.loc[:, self.categorical_columns].fillna('[NAN]').astype(
            str)

        self.real.loc[:, self.numerical_columns] = self.real.loc[:, self.numerical_columns].fillna(
            self.real[self.numerical_columns].mean())
        self.fake.loc[:, self.numerical_columns] = self.fake.loc[:, self.numerical_columns].fillna(
            self.fake[self.numerical_columns].mean())

    def plot_mean_std(self):
        """
        Class wrapper function for plotting the mean and std using `viz.plot_mean_std`.
        """
        plot_mean_std(self.real, self.fake)

    def plot_cumsums(self, nr_cols=4):
        """
        Plot the cumulative sums for all columns in the real and fake dataset. Height of each row scales with the length of the labels. Each plot contains the
        values of a real columns and the corresponding fake column.
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
            r = self.real[col]
            f = self.fake.iloc[:, self.real.columns.tolist().index(col)]
            cdf(r, f, col, 'Cumsum', ax=axes[i])
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.show()

    def plot_distributions(self, nr_cols=3):
        """
        Plot the distribution plots for all columns in the real and fake dataset. Height of each row of plots scales with the length of the labels. Each plot
        contains the values of a real columns and the corresponding fake column.
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
                plot_df = pd.DataFrame({col: self.real[col].append(self.fake[col]), 'kind': ['real'] * self.n_samples + ['fake'] * self.n_samples})
                fig = sns.histplot(plot_df, x=col, hue='kind', ax=axes[i], stat='probability', legend=True)
                axes[i].set_autoscaley_on(True)
            else:
                real = self.real.copy()
                fake = self.fake.copy()
                real['kind'] = 'Real'
                fake['kind'] = 'Fake'
                concat = pd.concat([fake, real])
                palette = sns.color_palette(
                    [(0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                     (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])
                x, y, hue = col, "proportion", "kind"
                ax = (concat[x]
                      .groupby(concat[hue])
                      .value_counts(normalize=True)
                      .rename(y)
                      .reset_index()
                      .pipe((sns.barplot, "data"), x=x, y=y, hue=hue, ax=axes[i], saturation=0.8, palette=palette))
                ax.set_xticklabels(axes[i].get_xticklabels(), rotation='vertical')
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.show()

    def plot_correlation_difference(self, plot_diff=True, **kwargs):
        """
        Plot the association matrices for each table and, if chosen, the difference between them.

        :param plot_diff: whether to plot the difference
        :param kwargs: kwargs for sns.heatmap
        """
        plot_correlation_difference(self.real, self.fake, cat_cols=self.categorical_columns, plot_diff=plot_diff,
                                    **kwargs)

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
            raise ValueError(f'`how` parameter must be in [euclidean, mae, rmse]')

        real_corr = compute_associations(self.real, nominal_columns=self.categorical_columns, theil_u=True)
        fake_corr = compute_associations(self.fake, nominal_columns=self.categorical_columns, theil_u=True)

        return distance_func(
            real_corr.values,
            fake_corr.values
        )

    def plot_pca(self):
        """
        Plot the first two components of a PCA of real and fake data.
        """
        real = numerical_encoding(self.real, nominal_columns=self.categorical_columns)
        fake = numerical_encoding(self.fake, nominal_columns=self.categorical_columns)
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
        plt.show()

    def get_copies(self, return_len: bool = False) -> Union[pd.DataFrame, int]:
        """
        Check whether any real values occur in the fake data.

        :param return_len: whether to return the length of the copied rows or not.
        :return: Dataframe containing the duplicates if return_len=False, else integer indicating the number of copied rows.
        """
        real_hashes = self.real.apply(lambda x: hash(tuple(x)), axis=1)
        fake_hashes = self.fake.apply(lambda x: hash(tuple(x)), axis=1)

        dup_idxs = fake_hashes.isin(real_hashes.values)
        dup_idxs = dup_idxs[dup_idxs == True].sort_index().index.tolist()

        if self.verbose:
            print(f'Nr copied columns: {len(dup_idxs)}')
        copies = self.fake.loc[dup_idxs, :]

        if return_len:
            return len(copies)
        else:
            return copies

    def get_duplicates(self, return_values: bool = False) -> Tuple[Union[pd.DataFrame, int], Union[pd.DataFrame, int]]:
        """
        Return duplicates within each dataset.

        :param return_values: whether to return the duplicate values in the datasets. If false, the lengths are returned.
        :return: dataframe with duplicates or the length of those dataframes if return_values=False.
        """
        real_duplicates = self.real[self.real.duplicated(keep=False)]
        fake_duplicates = self.fake[self.fake.duplicated(keep=False)]
        if return_values:
            return real_duplicates, fake_duplicates
        else:
            return len(real_duplicates), len(fake_duplicates)

    def pca_correlation(self, lingress=False):
        """
        Calculate the relation between PCA explained variance values. Due to some very large numbers, in recent implementation the MAPE(log) is used instead of
        regressions like Pearson's r.

        :param lingress: whether to use a linear regression, in this case Pearson's.
        :return: the correlation coefficient if lingress=True, otherwise 1 - MAPE(log(real), log(fake))
        """
        self.pca_r = PCA(n_components=5)
        self.pca_f = PCA(n_components=5)

        real = self.real
        fake = self.fake

        real = numerical_encoding(real, nominal_columns=self.categorical_columns)
        fake = numerical_encoding(fake, nominal_columns=self.categorical_columns)

        self.pca_r.fit(real)
        self.pca_f.fit(fake)
        if self.verbose:
            results = pd.DataFrame({'real': self.pca_r.explained_variance_, 'fake': self.pca_f.explained_variance_})
            print(f'\nTop 5 PCA components:')
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
            print(f'\nFitting real')
        for i, c in enumerate(self.r_estimators):
            if self.verbose:
                print(f'{i + 1}: {type(c).__name__}')
            c.fit(self.real_x_train, self.real_y_train)

        if self.verbose:
            print(f'\nFitting fake')
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
            for r_classifier, f_classifier, estimator_name in zip(self.r_estimators, self.f_estimators,
                                                                  self.estimator_names):
                for dataset, target, dataset_name in zip([self.real_x_test, self.fake_x_test],
                                                         [self.real_y_test, self.fake_y_test], ['real', 'fake']):
                    predictions_classifier_real = r_classifier.predict(dataset)
                    predictions_classifier_fake = f_classifier.predict(dataset)
                    f1_r = f1_score(target, predictions_classifier_real, average="micro")
                    f1_f = f1_score(target, predictions_classifier_fake, average="micro")
                    jac_sim = jaccard_score(predictions_classifier_real, predictions_classifier_fake, average='micro')
                    row = {'index': f'{estimator_name}_{dataset_name}_testset', 'f1_real': f1_r, 'f1_fake': f1_f,
                           'jaccard_similarity': jac_sim}
                    rows.append(row)
            results = pd.DataFrame(rows).set_index('index')

        elif self.target_type == 'regr':
            r2r = [rmse(self.real_y_test, clf.predict(self.real_x_test)) for clf in self.r_estimators]
            f2f = [rmse(self.fake_y_test, clf.predict(self.fake_x_test)) for clf in self.f_estimators]

            # Calculate test set accuracies on the other dataset
            r2f = [rmse(self.fake_y_test, clf.predict(self.fake_x_test)) for clf in self.r_estimators]
            f2r = [rmse(self.real_y_test, clf.predict(self.real_x_test)) for clf in self.f_estimators]
            index = [f'real_data_{classifier}' for classifier in self.estimator_names] + \
                    [f'fake_data_{classifier}' for classifier in self.estimator_names]
            results = pd.DataFrame({'real': r2r + f2r, 'fake': r2f + f2f}, index=index)
        else:
            raise Exception(f'self.target_type should be either \'class\' or \'regr\', but is {self.target_type}.')
        return results

    def visual_evaluation(self, **kwargs):
        """
        Plot all visual evaluation metrics. Includes plotting the mean and standard deviation, cumulative sums, correlation differences and the PCA transform.

        :param kwargs: any kwargs for matplotlib.
        """
        self.plot_mean_std()
        self.plot_cumsums()
        self.plot_distributions()
        self.plot_correlation_difference(**kwargs)
        self.plot_pca()

    def basic_statistical_evaluation(self) -> float:
        """
        Calculate the correlation coefficient between the basic properties of self.real and self.fake using Spearman's Rho. Spearman's is used because these
        values can differ a lot in magnitude, and Spearman's is more resilient to outliers.

        :return: correlation coefficient
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

        total_metrics.index = metrics.keys()
        self.statistical_results = total_metrics
        if self.verbose:
            print('\nBasic statistical attributes:')
            print(total_metrics.to_string())
        corr, p = stats.spearmanr(total_metrics['real'], total_metrics['fake'])
        return corr

    def distribution_statistical_evaluation(self) -> pd.DataFrame:
        """

        :return:
        """

    def correlation_correlation(self) -> float:
        """
        Calculate the correlation coefficient between the association matrices of self.real and self.fake using self.comparison_metric

        :return: The correlation coefficient
        """
        total_metrics = pd.DataFrame()
        for ds_name in ['real', 'fake']:
            ds = getattr(self, ds_name)
            corr_df = compute_associations(ds, nominal_columns=self.categorical_columns, theil_u=True)
            values = corr_df.values
            values = values[~np.eye(values.shape[0], dtype=bool)].reshape(values.shape[0], -1)
            total_metrics[ds_name] = values.flatten()

        self.correlation_correlations = total_metrics
        corr, p = self.comparison_metric(total_metrics['real'], total_metrics['fake'])
        if self.verbose:
            print('\nColumn correlation between datasets:')
            print(total_metrics.to_string())
        return corr

    # def convert_numerical(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     Special function to convert dataset to a numerical representations while making sure they have identical columns. This is sometimes a problem with
    #     categorical columns with many values or very unbalanced values
    #
    #     :return: Real and fake dataframe with categorical columns one-hot encoded and binary columns factorized.
    #     """
    #     real = numerical_encoding(self.real, nominal_columns=self.categorical_columns)
    #
    #     columns = sorted(real.columns.tolist())
    #     real = real[columns]
    #     fake = numerical_encoding(self.fake, nominal_columns=self.categorical_columns)
    #     for col in columns:
    #         if col not in fake.columns.tolist():
    #             fake[col] = 0
    #     fake = fake[columns]
    #     return real, fake

    def estimator_evaluation(self, target_col: str, target_type: str = 'class') -> float:
        """
        Method to do full estimator evaluation, including training. And estimator is either a regressor or a classifier, depending on the task. Two sets are
        created of each of the estimators `S_r` and `S_f`, for the real and fake data respectively. `S_f` is trained on ``self.real`` and `S_r` on
        ``self.fake``. Then, both are evaluated on their own and the others test set. If target_type is ``regr`` we do a regression on the RMSE scores with
        Pearson's. If target_type is ``class``, we calculate F1 scores and do return ``1 - MAPE(F1_r, F1_f)``.

        :param target_col: which column should be considered the target both both the regression and classification task.
        :param target_type: what kind of task this is. Can be either ``class`` or ``regr``.
        :return: Correlation value or 1 - MAPE
        """
        self.target_col = target_col
        self.target_type = target_type

        # Convert both datasets to numerical representations and split x and  y
        real_x = numerical_encoding(self.real.drop([target_col], axis=1), nominal_columns=self.categorical_columns)

        columns = sorted(real_x.columns.tolist())
        real_x = real_x[columns]
        fake_x = numerical_encoding(self.fake.drop([target_col], axis=1), nominal_columns=self.categorical_columns)
        for col in columns:
            if col not in fake_x.columns.tolist():
                fake_x[col] = 0
        fake_x = fake_x[columns]

        assert real_x.columns.tolist() == fake_x.columns.tolist(), f'real and fake columns are different: \n{real_x.columns}\n{fake_x.columns}'

        if self.target_type == 'class':
            # Encode real and fake target the same
            real_y, uniques = pd.factorize(self.real[target_col])
            mapping = {key: value for value, key in enumerate(uniques)}
            fake_y = [mapping.get(key) for key in self.fake[target_col].tolist()]
        elif self.target_type == 'regr':
            real_y = self.real[target_col]
            fake_y = self.fake[target_col]
        else:
            raise Exception(f'Target Type must be regr or class')

        # For reproducibilty:
        np.random.seed(self.random_seed)

        self.real_x_train, self.real_x_test, self.real_y_train, self.real_y_test = train_test_split(real_x, real_y,
                                                                                                    test_size=0.2)
        self.fake_x_train, self.fake_x_test, self.fake_y_train, self.fake_y_test = train_test_split(fake_x, fake_y,
                                                                                                    test_size=0.2)

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
                MLPClassifier([50, 50], solver='adam', activation='relu', learning_rate='adaptive', random_state=42),
            ]
        else:
            raise ValueError(f'target_type must be \'regr\' or \'class\'')

        self.r_estimators = copy.deepcopy(self.estimators)
        self.f_estimators = copy.deepcopy(self.estimators)
        self.estimator_names = [type(clf).__name__ for clf in self.estimators]

        for estimator in self.estimators:
            assert hasattr(estimator, 'fit')
            assert hasattr(estimator, 'score')

        self.fit_estimators()
        self.estimators_scores = self.score_estimators()
        # print('\nClassifier F1-scores and their Jaccard similarities:') if self.target_type == 'class' \
        #     else print('\nRegressor MSE-scores and their Jaccard similarities:')
        # print(self.estimators_scores.to_string())

        if self.target_type == 'regr':
            corr, p = self.comparison_metric(self.estimators_scores['real'], self.estimators_scores['fake'])
            return corr
        elif self.target_type == 'class':
            mean = mean_absolute_percentage_error(self.estimators_scores['f1_real'], self.estimators_scores['f1_fake'])
            return 1 - mean
        else:
            raise ValueError('`self.target_type` should be `regr` or `class`.')

    def row_distance(self, n_samples: int = None) -> Tuple[np.number, np.number]:
        """
        Calculate mean and standard deviation distances between `self.fake` and `self.real`.

        :param n_samples: Number of samples to take for evaluation. Compute time increases exponentially.
        :return: `(mean, std)` of these distances.
        """
        if n_samples is None:
            n_samples = len(self.real)
        real = numerical_encoding(self.real, nominal_columns=self.categorical_columns)
        fake = numerical_encoding(self.fake, nominal_columns=self.categorical_columns)

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
        return column_correlations(self.real, self.fake, self.categorical_columns)

    def evaluate(self, target_col: str, target_type: str = 'class', metric: str = None, verbose: bool = None,
                 n_samples_distance: int = 20000, notebook: bool = False) -> Dict:
        """
        Determine correlation between attributes from the real and fake dataset using a given metric.
        All metrics from scipy.stats are available.

        :param target_col: column to use for predictions with estimators
        :param target_type: what kind of task to perform on the target_col. Can be either ``class`` for classification or ``regr`` for regression.
        :param metric: overwrites self.metric. Scoring metric for the attributes.
            By default Pearson's r is used. Alternatives include Spearman rho (scipy.stats.spearmanr) or Kendall Tau (scipy.stats.kendalltau).
        :param n_samples_distance: The number of samples to take for the row distance. See documentation of ``tableEvaluator.row_distance`` for details.
        :param verbose: whether to print verbose logging.
        """
        self.verbose = verbose if verbose is not None else self.verbose
        self.comparison_metric = metric if metric is not None else self.comparison_metric

        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        pd.options.display.float_format = '{:,.4f}'.format

        # print(f'\nCorrelation metric: {self.comparison_metric.__name__}')

        basic_statistical = self.basic_statistical_evaluation()
        correlation_correlation = self.correlation_correlation()
        column_correlation = self.column_correlations()
        estimators = self.estimator_evaluation(target_col=target_col, target_type=target_type)
        nearest_neighbor = self.row_distance(n_samples=n_samples_distance)

        miscellaneous_dict = {
            'Column Correlation Distance RMSE': self.correlation_distance(how='rmse'),
            'Column Correlation distance MAE': self.correlation_distance(how='mae'),
        }

        miscellaneous = pd.DataFrame({'Result': list(miscellaneous_dict.values())},
                                     index=list(miscellaneous_dict.keys()))
        # print(f'\nMiscellaneous results:')

        privacy_metrics_dict = {
            'Duplicate rows between sets (real/fake)': self.get_duplicates(),
            'nearest neighbor mean': nearest_neighbor[0],
            'nearest neighbor std': nearest_neighbor[1],
        }
        privacy_report = EvaluationResult(
            name='Privacy Results',
            content=dict_to_df(privacy_metrics_dict),
        )

        all_results_dict = {
            'Basic statistics': basic_statistical,
            'Correlation column correlations': correlation_correlation,
            'Mean Correlation between fake and real columns': column_correlation,
            f'{"1 - MAPE Estimator results" if self.target_type == "class" else "Correlation RMSE"}': estimators,
        }
        similarity_score = np.mean(list(all_results_dict.values()))
        all_results_dict['Similarity Score'] = similarity_score
        # all_results = pd.DataFrame({'Result': list(all_results_dict.values())}, index=list(all_results_dict.keys()))
        all_results = EvaluationResult(
            name='Overview Results',
            content=dict_to_df(all_results_dict)
        )

        efficacy_title = 'Classifier F1-scores and their Jaccard similarities:' if self.target_type == 'class' \
            else '\nRegressor MSE-scores'

        overview_tab = [all_results, ]

        ml_efficacy_tab = [
            EvaluationResult(name='ML Efficacy', content=self.estimators_scores)
        ]

        privacy_tab = [privacy_report]

        js_df = js_distance_df(self.real, self.fake, self.numerical_columns)

        statistical_tab = [
            EvaluationResult(name='Jensen-Shannon distance', content=js_df,
                             appendix=f'### Mean: {js_df.js_distance.mean(): .3f}'),
            EvaluationResult(name='Kolmogorov-Smirnov statistic',
                             content=kolmogorov_smirnov_df(self.real, self.fake, self.numerical_columns)
                             )
        ]

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

            print(f'\nPrivacy results:')
            print(privacy_report.content.to_string())

            print(f'\nMiscellaneous results:')
            print(miscellaneous.to_string())

            print(f'\nResults:')
            print(all_results.content.to_string())
            # return all_results
