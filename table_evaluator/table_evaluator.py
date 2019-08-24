import copy
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import cdist
from dython.nominal import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from helpers import *


class TableEvaluator:
    def __init__(self, real, fake, unique_thresh=55, metric='pearsonr', verbose=False, n_samples=None):
        if isinstance(real, np.ndarray):
            real = pd.DataFrame(real)
            fake = pd.DataFrame(fake)
        assert isinstance(real, pd.DataFrame), f'Make sure you either pass a Pandas DataFrame or Numpy Array'

        self.unique_thresh = unique_thresh
        self.numerical_columns = [column for column in real._get_numeric_data().columns if
                                  len(real[column].unique()) > unique_thresh]
        self.categorical_columns = [column for column in real.columns if column not in self.numerical_columns]
        self.real = real
        self.fake = fake
        self.comparison_metric = getattr(stats, metric)
        self.verbose = verbose

        if n_samples is None:
            self.n_samples = min(len(self.real), len(self.fake))
        elif len(fake) >= n_samples and len(real) >= n_samples:
            self.n_samples = n_samples
        else:
            raise Exception(f'Make sure n_samples < len(fake/real). len(real): {len(real)}, len(fake): {len(fake)}')
        self.real = self.real[:self.n_samples]
        self.fake = self.fake[:self.n_samples]
        assert len(self.real) == len(self.fake), f'len(real) != len(fake)'

    def plot_mean_std(self):
        plot_mean_std(self.real, self.fake)

    def plot_cumsums(self):
        nr_charts = len(self.real.columns)
        nr_cols = 4
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

    def plot_correlation_difference(self, plot_diff=True, *args, **kwargs):
        plot_correlation_difference(self.real, self.fake, cat_cols=self.categorical_columns, plot_diff=plot_diff, *args,
                                    **kwargs)

    def correlation_distance(self, how='euclidean'):
        """
        Calculate distance between correlation matrices with certain metric.
        Metric options are: euclidean, mae (mean absolute error)
        :param how: metric to measure distance
        :return: distance
        """
        distance_func = None
        if how == 'euclidean':
            distance_func = euclidean_distance
        elif how == 'mae':
            distance_func = mean_absolute_error
        elif how == 'rmse':
            distance_func = rmse

        assert distance_func is not None, f'Distance measure was None. Please select a measure from [euclidean, mae]'

        real_corr = associations(self.real, nominal_columns=self.categorical_columns, return_results=True, theil_u=True, plot=False)
        fake_corr = associations(self.fake, nominal_columns=self.categorical_columns, return_results=True, theil_u=True, plot=False)

        return distance_func(
            real_corr.values,
            fake_corr.values
        )

    def plot_2d(self):
        """
        Plot the first two components of a PCA of the numeric columns of real and fake.
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

    def get_copies(self):
        """
        Check whether any real values occur in the fake data
        :return: Dataframe containing the duplicates
        """
        # df = pd.concat([self.real, self.fake])
        # duplicates = df[df.duplicated(keep=False)]
        # return duplicates
        real_hashes = self.real.apply(lambda x: hash(tuple(x)), axis=1)
        fake_hashes = self.fake.apply(lambda x: hash(tuple(x)), axis=1)
        dup_idxs = fake_hashes.isin(real_hashes.values)
        dup_idxs = dup_idxs[dup_idxs == True].sort_index().index.tolist()
        len(dup_idxs)
        print(f'Nr copied columns: {len(dup_idxs)}')

        return self.fake.loc[dup_idxs, :]

    def get_duplicates(self, return_values=False):
        real_duplicates = self.real[self.real.duplicated(keep=False)]
        fake_duplicates = self.fake[self.fake.duplicated(keep=False)]
        if return_values:
            return real_duplicates, fake_duplicates
        return len(real_duplicates), len(fake_duplicates)

    def get_duplicates2(self, return_values=False):
        df = pd.concat([self.real, self.fake])
        duplicates = df[df.duplicated(keep=False)]
        return duplicates

    def pca_correlation(self):
        self.pca_r = PCA(n_components=5)
        self.pca_f = PCA(n_components=5)

        # real = self.real.drop(self.categorical_columns, axis=1)
        # fake = self.fake.drop(self.categorical_columns, axis=1)
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
        # slope, intersect, corr, p, _ = stats.linregress(pca_r.explained_variance_, pca_f.explained_variance_)
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
        :return:
        """
        from sklearn.metrics import mean_squared_error

        if self.target_type == 'class':
            r2r = [f1_score(self.real_y_test, clf.predict(self.real_x_test), average='micro') for clf in self.r_estimators]
            f2f = [f1_score(self.fake_y_test, clf.predict(self.fake_x_test), average='micro') for clf in self.f_estimators]

            # Calculate test set accuracies on the other dataset
            r2f = [f1_score(self.fake_y_test, clf.predict(self.fake_x_test), average='micro') for clf in self.r_estimators]
            f2r = [f1_score(self.real_y_test, clf.predict(self.real_x_test), average='micro') for clf in self.f_estimators]
            index = [f'real_data_{classifier}_F1' for classifier in self.estimator_names] + \
                    [f'fake_data_{classifier}_F1' for classifier in self.estimator_names]
            results = pd.DataFrame({'real': r2r + r2f, 'fake': f2r + f2f}, index=index)

        elif self.target_type == 'regr':
            r2r = [rmse(self.real_y_test, clf.predict(self.real_x_test)) for clf in self.r_estimators]
            f2f = [rmse(self.fake_y_test, clf.predict(self.fake_x_test)) for clf in self.f_estimators]

            # Calculate test set accuracies on the other dataset
            r2f = [rmse(self.fake_y_test, clf.predict(self.fake_x_test)) for clf in self.r_estimators]
            f2r = [rmse(self.real_y_test, clf.predict(self.real_x_test)) for clf in self.f_estimators]
            index = [f'real_data_{classifier}' for classifier in self.estimator_names] + \
                    [f'fake_data_{classifier}' for classifier in self.estimator_names]
            results = pd.DataFrame({'real': r2r + r2f, 'fake': f2r + f2f}, index=index)
        else:
            raise Exception(f'self.target_type should be either \'class\' or \'regr\', but is {self.target_type}.')
        return results

    def visual_evaluation(self, plot=True, **kwargs):
        if plot:
            self.plot_mean_std()
            self.plot_cumsums()
            self.plot_correlation_difference(**kwargs)
            self.plot_2d()

    def statistical_evaluation(self):
        total_metrics = pd.DataFrame()
        for ds_name in ['real', 'fake']:
            ds = getattr(self, ds_name)
            metrics = {}
            num_ds = ds[self.numerical_columns]

            # Basic statistical properties
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

    def correlation_correlation(self):
        total_metrics = pd.DataFrame()
        for ds_name in ['real', 'fake']:
            ds = getattr(self, ds_name)
            corr_df = associations(ds, nominal_columns=self.categorical_columns, return_results=True, theil_u=True, plot=False)
            values = corr_df.values
            values = values[~np.eye(values.shape[0], dtype=bool)].reshape(values.shape[0], -1)
            total_metrics[ds_name] = values.flatten()

        self.correlation_correlations = total_metrics
        corr, p = self.comparison_metric(total_metrics['real'], total_metrics['fake'])
        if self.verbose:
            print('\nColumn correlation between datasets:')
            print(total_metrics.to_string())
        return corr

    def convert_numerical(self):
        real = numerical_encoding(self.real, nominal_columns=self.categorical_columns)

        columns = sorted(real.columns.tolist())
        real = real[columns]
        fake = numerical_encoding(self.fake, nominal_columns=self.categorical_columns)
        for col in columns:
            if col not in fake.columns.tolist():
                fake[col] = 0
        fake = fake[columns]
        return real, fake

    def estimator_evaluation(self, target_col, target_type='class'):
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

        # split real and fake into train and test sets
        self.real_x_train, self.real_x_test, self.real_y_train, self.real_y_test = train_test_split(real_x, real_y, test_size=0.2)
        self.fake_x_train, self.fake_x_test, self.fake_y_train, self.fake_y_test = train_test_split(fake_x, fake_y, test_size=0.2)

        if target_type == 'regr':
            self.estimators = [
                RandomForestRegressor(n_estimators=20, max_depth=5),
                Lasso(),
                Ridge(alpha=1.0),
                ElasticNet(),
            ]
        elif target_type == 'class':
            self.estimators = [
                # SGDClassifier(max_iter=100, tol=1e-3),
                LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500),
                RandomForestClassifier(n_estimators=10),
                DecisionTreeClassifier(),
                MLPClassifier([50, 50], solver='adam', activation='relu', learning_rate='adaptive'),
            ]
        else:
            raise Exception(f'target_type must be \'regr\' or \'class\'')

        self.r_estimators = copy.deepcopy(self.estimators)
        self.f_estimators = copy.deepcopy(self.estimators)
        self.estimator_names = [type(clf).__name__ for clf in self.estimators]

        for estimator in self.estimators:
            assert hasattr(estimator, 'fit')
            assert hasattr(estimator, 'score')

        self.fit_estimators()
        self.estimators_scores = self.score_estimators()
        print('\nClassifier F1-scores:') if self.target_type == 'class' else print('\nRegressor MSE-scores:')
        print(self.estimators_scores.to_string())
        if self.target_type == 'regr':
            corr, p = self.comparison_metric(self.estimators_scores['real'], self.estimators_scores['fake'])
            return corr
        elif self.target_type == 'class':
            mean = mean_absolute_percentage_error(self.estimators_scores['real'], self.estimators_scores['fake'])
            return 1 - mean

    def row_distance(self, n=None):
        if n is None:
            n = len(self.real)
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

        distances = cdist(real[:n], fake[:n])
        min_distances = np.min(distances, axis=1)
        min_mean = np.mean(min_distances)
        min_std = np.std(min_distances)
        return min_mean, min_std

    def evaluate(self, target_col, target_type='class', metric=None, verbose=None):
        """
        Determine correlation between attributes from the real and fake dataset using a given metric.
        All metrics from scipy.stats are available.
        :param target_col: column to use for predictions with estimators
        :param n_samples: the number of samples to use for the estimators. Training time scales mostly linear
        :param metric: scoring metric for the attributes. By default Kendall Tau ranking is used. Alternatives
            include Spearman rho (scipy.stats.spearmanr) ranking.
        """
        if verbose is not None:
            self.verbose = verbose
        if metric is not None:
            self.comparison_metric = metric

        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        pd.options.display.float_format = '{:,.4f}'.format

        print(f'\nCorrelation metric: {self.comparison_metric.__name__}')

        basic_statistical = self.statistical_evaluation()  # 2 columns -> Corr -> correlation coefficient
        correlation_correlation = self.correlation_correlation()  # 2 columns -> Kendall Tau -> Correlation coefficient
        column_correlation = column_correlations(self.real, self.fake, self.categorical_columns)  # 1 column -> Mean
        estimators = self.estimator_evaluation(target_col=target_col, target_type=target_type)  # 1 2 columns -> Kendall Tau -> Correlation coefficient
        pca_variance = self.pca_correlation()  # 1 number
        nearest_neighbor = self.row_distance(n=20000)

        miscellaneous = {}
        miscellaneous['Column Correlation Distance RMSE'] = self.correlation_distance(how='rmse')
        miscellaneous['Column Correlation distance MAE'] = self.correlation_distance(how='mae')

        miscellaneous['Duplicate rows between sets'] = len(self.get_duplicates())
        miscellaneous['nearest neighbor mean'] = nearest_neighbor[0]
        miscellaneous['nearest neighbor std'] = nearest_neighbor[1]
        miscellaneous_df = pd.DataFrame({'Result': list(miscellaneous.values())}, index=list(miscellaneous.keys()))
        print(f'\nMiscellaneous results:')
        print(miscellaneous_df.to_string())

        all_results = {
            'basic statistics': basic_statistical,
            'Correlation column correlations': correlation_correlation,
            'Mean Correlation between fake and real columns': column_correlation,
            f'{"1 - MAPE Estimator results" if self.target_type == "class" else "Correlation RMSE"}': estimators,
            '1 - MAPE 5 PCA components': pca_variance,
        }
        total_result = np.mean(list(all_results.values()))
        all_results['Total Result'] = total_result
        all_results_df = pd.DataFrame({'Result': list(all_results.values())}, index=list(all_results.keys()))

        print(f'\nResults:\nNumber of duplicate rows is ignored for total score.')
        print(all_results_df.to_string())
