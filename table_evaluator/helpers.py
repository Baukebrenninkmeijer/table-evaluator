from dython.nominal import *
from typing import Union, List, Tuple, Optional
from sklearn.metrics import mean_squared_error
import pandas as pd


def load_data(path_real: str, path_fake: str, real_sep: str = ',', fake_sep: str = ',', drop_columns: List = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a real and synthetic data csv. This function makes sure that the loaded data has the same columns with the same data types.

    :param path_real: string path to csv with real data
    :param path_fake: string path to csv with real data
    :param real_sep: separator of the real csv
    :param fake_sep: separator of the fake csv
    :param drop_columns: names of columns to drop.
    :return: Tuple with DataFrame containing the real data and DataFrame containing the synthetic data.
    """
    real = pd.read_csv(path_real, sep=real_sep, low_memory=False)
    fake = pd.read_csv(path_fake, sep=fake_sep, low_memory=False)
    if set(fake.columns.tolist()).issubset(set(real.columns.tolist())):
        real = real[fake.columns]
    elif drop_columns is not None:
        real = real.drop(drop_columns, axis=1)
        try:
            fake = fake.drop(drop_columns, axis=1)
        except:
            print(f'Some of {drop_columns} were not found on fake.index.')
        assert len(fake.columns.tolist()) == len(real.columns.tolist()), \
            f'Real and fake do not have same nr of columns: {len(fake.columns)} and {len(real.columns)}'
        fake.columns = real.columns
    else:
        fake.columns = real.columns

    for col in fake.columns:
        fake[col] = fake[col].astype(real[col].dtype)
    return real, fake


def plot_var_cor(x: Union[pd.DataFrame, np.ndarray], ax=None, return_values: bool = False, **kwargs) -> Optional[np.ndarray]:
    """
    Given a DataFrame, plot the correlation between columns. Function assumes all numeric continuous data. It masks the top half of the correlation matrix,
    since this holds the same values.

    :param x: Dataframe to plot data from
    :param ax: Axis on which to plot the correlations
    :param return_values: return correlation matrix after plotting
    :param kwargs: Keyword arguments that are passed to `sns.heatmap`.
    :return: If return_values=True, returns correlation matrix of `x` as np.ndarray
    """
    if isinstance(x, pd.DataFrame):
        corr = x.corr().values
    elif isinstance(x, np.ndarray):
        corr = np.corrcoef(x, rowvar=False)
    else:
        raise ValueError('Unknown datatype given. Make sure a Pandas DataFrame or Numpy Array is passed for x.')

    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    if ax is None:
        f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, **kwargs)
    if return_values:
        return corr


def plot_correlation_difference(real: pd.DataFrame, fake: pd.DataFrame, plot_diff: bool = True, cat_cols: list = None, **kwargs):
    """
    Plot the association matrices for the `real` dataframe, `fake` dataframe and plot the difference between them. Has support for continuous and Categorical
    (Male, Female) data types. All Object and Category dtypes are considered to be Categorical columns if `dis_cols` is not passed.

    - Continuous - Continuous: Uses Pearson's correlation coefficient
    - Continuous - Categorical: Uses so called correlation ratio (https://en.wikipedia.org/wiki/Correlation_ratio) for both continuous - categorical and categorical - continuous.
    - Categorical - Categorical: Uses Theil's U, an asymmetric correlation metric for Categorical associations

    :param real: DataFrame with real data
    :param fake: DataFrame with synthetic data
    :param plot_diff: Plot difference if True, else not
    :param cat_cols: List of Categorical columns
    :param kwargs: keyword arguments that are passed to ``sns.heatmap``.
    """
    assert isinstance(real, pd.DataFrame), f'`real` parameters must be a Pandas DataFrame'
    assert isinstance(fake, pd.DataFrame), f'`fake` parameters must be a Pandas DataFrame'

    if cat_cols is None:
        cat_cols = real.select_dtypes(['object', 'category'])
    if plot_diff:
        fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    real_corr = associations(real, cat_cols=cat_cols, return_results=True, plot=True, theil_u=True,
                             mark_columns=True, ax=ax[0], **kwargs)
    fake_corr = associations(fake, cat_cols=cat_cols, return_results=True, plot=True, theil_u=True,
                             mark_columns=True, ax=ax[1], **kwargs)

    if plot_diff:
        diff = abs(real_corr - fake_corr)
        sns.set(style="white")
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(diff, ax=ax[2], cmap=cmap, vmax=.3, square=True, annot=kwargs.get('annot', True), center=0,
                    linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')

    titles = ['Real', 'Fake', 'Difference'] if plot_diff else ['Real', 'Fake']
    for i, label in enumerate(titles):
        title_font = {'size': '18'}
        ax[i].set_title(label, **title_font)
    plt.tight_layout()
    plt.show()


def plot_correlation_comparison(evaluators: List, **kwargs):
    """
    Plot the correlation differences of multiple TableEvaluator objects.

    :param evaluators: list of TableEvaluator objects
    :param kwargs: keyword arguments that are passed to ``sns.heatmap``.
    """
    nr_plots = len(evaluators) + 1
    fig, ax = plt.subplots(2, nr_plots, figsize=(4 * nr_plots, 7))
    flat_ax = ax.flatten()
    fake_corr = []
    real_corr = associations(evaluators[0].real, cat_cols=evaluators[0].categorical_columns, return_results=True, plot=True, theil_u=True,
                             mark_columns=True, ax=flat_ax[0], cbar=False, linewidths=0, **kwargs)
    for i in range(1, nr_plots):
        cbar = True if i % (nr_plots - 1) == 0 else False
        fake_corr.append(associations(evaluators[i - 1].fake, cat_cols=evaluators[0].categorical_columns, return_results=True, plot=True, theil_u=True,
                                      mark_columns=True, ax=flat_ax[i], cbar=cbar, linewidths=0, **kwargs))
        if i % (nr_plots - 1) == 0:
            cbar = flat_ax[i].collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)

    for i in range(1, nr_plots):
        cbar = True if i % (nr_plots - 1) == 0 else False
        diff = abs(real_corr - fake_corr[i - 1])
        sns.set(style="white")
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        az = sns.heatmap(diff, ax=flat_ax[i + nr_plots], cmap=cmap, vmax=.3, square=True, annot=kwargs.get('annot', True), center=0,
                         linewidths=0, cbar_kws={"shrink": .8}, cbar=cbar, fmt='.2f')
        if i % (nr_plots - 1) == 0:
            cbar = az.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
    titles = [e.name if e is not None else idx for idx, e in enumerate(evaluators)]
    for i, label in enumerate(titles):
        flat_ax[i].set_yticklabels([])
        flat_ax[i].set_xticklabels([])
        flat_ax[i + nr_plots].set_yticklabels([])
        flat_ax[i + nr_plots].set_xticklabels([])
        title_font = {'size': '28'}
        flat_ax[i].set_title(label, **title_font)
    plt.tight_layout()


def cdf(data_r, data_f, xlabel: str = 'Values', ylabel: str = 'Cumulative Sum', ax=None):
    """
    Plot continous density function on optionally given ax. If no ax, cdf is plotted and shown.

    :param data_r: Series with real data
    :param data_f: Series with fake data
    :param xlabel: Label to put on the x-axis
    :param ylabel: Label to put on the y-axis
    :param ax: The axis to plot on. If ax=None, a new figure is created.
    """
    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    ax = ax if ax else plt.subplots()[1]

    axis_font = {'size': '14'}
    ax.set_xlabel(xlabel, **axis_font)
    ax.set_ylabel(ylabel, **axis_font)

    ax.grid()
    ax.plot(x1, y, marker='o', linestyle='none', label='Real', ms=8)
    ax.plot(x2, y, marker='o', linestyle='none', label='Fake', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    # If labels are strings, rotate them vertical
    if isinstance(data_r, pd.Series) and data_r.dtypes == 'object':
        ax.set_xticklabels(data_r.value_counts().sort_index().index, rotation='vertical')

    if ax is None:
        plt.show()


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the mean absolute error between y_true and y_pred.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: Mean absolute error (float).
    """
    return np.mean(np.abs(np.subtract(y_true, y_pred)))


def euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the euclidean distance between y_true and y_pred.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :returns: Mean absolute error (float).
    """
    return np.sqrt(np.sum(np.power(np.subtract(y_true, y_pred), 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the mean absolute percentage error between y_true and y_pred. Throws ValueError if y_true contains zero values.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: Mean absolute percentage error (float).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the root mean squared error between y_true and y_pred.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: root mean squared error (float).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def column_correlations(dataset_a, dataset_b, categorical_columns, theil_u=True):
    """
    Columnwise correlation calculation between ``dataset_a`` and ``dataset_b``.

    :param dataset_a: First DataFrame
    :param dataset_b: Second DataFrame
    :param categorical_columns: The columns containing categorical values
    :param theil_u: Whether to use Theil's U. If False, use Cramer's V.
    :return: Mean correlation between all columns.
    """
    if categorical_columns is None:
        categorical_columns = list()
    elif categorical_columns == 'all':
        categorical_columns = dataset_a.columns
    assert dataset_a.columns.tolist() == dataset_b.columns.tolist()
    corr = pd.DataFrame(columns=dataset_a.columns, index=['correlation'])

    for column in dataset_a.columns.tolist():
        if column in categorical_columns:
            if theil_u:
                corr[column] = theils_u(dataset_a[column].sort_values(), dataset_b[column].sort_values())
            else:
                corr[column] = cramers_v(dataset_a[column].sort_values(), dataset_b[column].sort_vaues())
        else:
            corr[column], _ = ss.pearsonr(dataset_a[column].sort_values(), dataset_b[column].sort_values())
    corr.fillna(value=np.nan, inplace=True)
    correlation = np.mean(corr.values.flatten())
    return correlation


def associations(dataset: Union[pd.DataFrame, np.ndarray], cat_cols=None, mark_columns=False, theil_u=False, plot=True, return_results=False,
                 ax=None, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE, **kwargs):
    """
    Adapted from: https://github.com/shakedzy/dython

    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:

     - Pearson's R for continuous-continuous cases
     - Correlation Ratio for categorical-continuous cases
     - Cramer's V or Theil's U for categorical-categorical cases

    :param dataset: NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    :param cat_cols: string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    :param mark_columns: Boolean (default: False)
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by cat_cols
    :param theil_u: Boolean (default: False)
        In the case of categorical-categorical features, use Theil's U instead of Cramer's V
    :param plot: Boolean (default: True)
        If True, plot a heat-map of the correlation matrix
    :param return_results: Boolean (default: False)
        If True, the function will return a Pandas DataFrame of the computed associations
    :param kwargs:
        Arguments to be passed to used function and methods
    :return: Pandas DataFrame
        A DataFrame of the correlation/strength-of-association between all features
    """
    if plot is False:
        assert kwargs == {}, f'You have some kwargs that are not needed.\nkwargs: {kwargs}'

    dataset = convert(dataset, 'dataframe')
    columns = dataset.columns
    if cat_cols is None:
        cat_cols = list()
    elif cat_cols == 'all':
        cat_cols = columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0, len(columns)):
        for j in range(i, len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                if columns[i] in cat_cols:
                    if columns[j] in cat_cols:
                        if theil_u:
                            corr[columns[j]][columns[i]] = theils_u(dataset[columns[i]], dataset[columns[j]], nan_strategy=SKIP)
                            corr[columns[i]][columns[j]] = theils_u(dataset[columns[j]], dataset[columns[i]], nan_strategy=SKIP)
                        else:
                            cell = cramers_v(dataset[columns[i]], dataset[columns[j]], nan_strategy=SKIP)
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
                    else:
                        cell = correlation_ratio(dataset[columns[i]], dataset[columns[j]], nan_strategy=SKIP)
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                else:
                    if columns[j] in cat_cols:
                        cell = correlation_ratio(dataset[columns[j]], dataset[columns[i]], nan_strategy=SKIP)
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                    else:
                        cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = ['{} (nom)'.format(col) if col in cat_cols else '{} (con)'.format(col) for col in
                          columns]
        corr.columns = marked_columns
        corr.index = marked_columns
    if plot:
        if ax is None:
            plt.figure(figsize=kwargs.get('figsize', None))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.set(style="white")
        sns.heatmap(corr, annot=kwargs.get('annot', True), fmt=kwargs.get('fmt', '.2f'), cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=kwargs.get('linewidths', 0.5), cbar_kws={"shrink": .8}, cbar=kwargs.get('cbar', True), ax=ax)
        if ax is None:
            plt.show()
    if return_results:
        return corr


def numerical_encoding(dataset, cat_cols: str = 'all', drop_single_label: bool = False, drop_fact_dict: bool = True):
    """
    Adapted from: https://github.com/shakedzy/dython

    Encoding a data-set with mixed data (numerical and categorical) to a numerical-only data-set,
    using the following logic:

    - categorical with only a single value will be marked as zero (or dropped, if requested)
    - categorical with two values will be replaced with the result of Pandas `factorize`
    - categorical with more than two values will be replaced with the result of Pandas `get_dummies`
    - numerical columns will not be modified

    :param dataset: NumPy ndarray / Pandas DataFrame
        The data-set to encode
    :param cat_cols: sequence / string
        A sequence of the nominal (categorical) columns in the dataset. If string, must be 'all' to state that
        all columns are nominal. If None, nothing happens. Default: 'all'
    :param drop_single_label: Boolean, default = False
        If True, nominal columns with a only a single value will be dropped.
    :param drop_fact_dict: Boolean, default = True
        If True, the return value will be the encoded DataFrame alone. If False, it will be a tuple of
        the DataFrame and the dictionary of the binary factorization (originating from pd.factorize)
    :return: DataFrame or (DataFrame, dict). If `drop_fact_dict` is True, returns the encoded DataFrame.
        else, returns a tuple of the encoded DataFrame and dictionary, where each key is a two-value column, and the
        value is the original labels, as supplied by Pandas `factorize`. Will be empty if no two-value columns are
        present in the data-set
    """
    dataset = convert(dataset, 'dataframe')
    if cat_cols is None:
        return dataset
    elif cat_cols == 'all':
        cat_cols = dataset.columns
    converted_dataset = pd.DataFrame()
    binary_columns_dict = dict()
    for col in dataset.columns:
        if col not in cat_cols:
            converted_dataset.loc[:, col] = dataset[col]
        else:
            unique_values = pd.unique(dataset[col])
            if len(unique_values) == 1 and not drop_single_label:
                converted_dataset.loc[:, col] = 0
            elif len(unique_values) == 2:
                converted_dataset.loc[:, col], binary_columns_dict[col] = pd.factorize(dataset[col])
            else:
                dummies = pd.get_dummies(dataset[col], prefix=col)
                converted_dataset = pd.concat([converted_dataset, dummies], axis=1)
    if drop_fact_dict:
        return converted_dataset
    else:
        return converted_dataset, binary_columns_dict


def plot_mean_std_comparison(evaluators: List):
    """
    Plot comparison between the means and standard deviations from each evaluator in evaluators.

    :param evaluators: list of TableEvaluator objects that are to be evaluated.
    """
    nr_plots = len(evaluators)
    fig, ax = plt.subplots(2, nr_plots, figsize=(4 * nr_plots, 7))
    flat_ax = ax.flatten()
    for i in range(nr_plots):
        plot_mean_std(evaluators[i].real, evaluators[i].fake, ax=ax[:, i])

    titles = [e.name if e is not None else idx for idx, e in enumerate(evaluators)]
    for i, label in enumerate(titles):
        title_font = {'size': '24'}
        flat_ax[i].set_title(label, **title_font)
    plt.tight_layout()


def plot_mean_std(real: pd.DataFrame, fake: pd.DataFrame, ax=None):
    """
    Plot the means and standard deviations of each dataset.

    :param real: DataFrame containing the real data
    :param fake: DataFrame containing the fake data
    :param ax: Axis to plot on. If none, a new figure is made.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Absolute Log Mean and STDs of numeric data\n', fontsize=16)

    real = real._get_numeric_data()
    fake = fake._get_numeric_data()
    real_mean = np.log(np.add(abs(real.mean()).values, 1e-5))
    fake_mean = np.log(np.add(abs(fake.mean()).values, 1e-5))
    min_mean = min(real_mean) - 1
    max_mean = max(real_mean) + 1
    line = np.arange(min_mean, max_mean)
    sns.lineplot(x=line, y=line, ax=ax[0])
    sns.scatterplot(x=real_mean,
                    y=fake_mean,
                    ax=ax[0])
    ax[0].set_title('Means of real and fake data')
    ax[0].set_xlabel('real data mean (log)')
    ax[0].set_ylabel('fake data mean (log)')

    real_std = np.log(np.add(real.std().values, 1e-5))
    fake_std = np.log(np.add(fake.std().values, 1e-5))
    min_std = min(real_std) - 1
    max_std = max(real_std) + 1
    line = np.arange(min_std, max_std)
    sns.lineplot(x=line, y=line, ax=ax[1])
    sns.scatterplot(x=real_std,
                    y=fake_std,
                    ax=ax[1])
    ax[1].set_title('Stds of real and fake data')
    ax[1].set_xlabel('real data std (log)')
    ax[1].set_ylabel('fake data std (log)')
    ax[0].grid(True)
    ax[1].grid(True)
    if ax is None:
        plt.show()
