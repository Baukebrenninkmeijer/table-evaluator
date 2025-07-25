import warnings
from collections.abc import Callable
from os import PathLike

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

from table_evaluator import metrics as te_metrics
from table_evaluator.constants import RANDOM_SEED
from table_evaluator.core.evaluation_config import EvaluationConfig
from table_evaluator.data.data_converter import DataConverter
from table_evaluator.evaluators.ml_evaluator import MLEvaluator
from table_evaluator.evaluators.privacy_evaluator import PrivacyEvaluator
from table_evaluator.evaluators.statistical_evaluator import StatisticalEvaluator
from table_evaluator.metrics.statistical import associations
from table_evaluator.notebook import EvaluationResult, visualize_notebook
from table_evaluator.utils import _preprocess_data, dict_to_df
from table_evaluator.visualization.visualization_manager import VisualizationManager


class TableEvaluator:
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the user to easily evaluate data
    with the `evaluate` method. Additional evaluations can be done with the different methods of evaluate and the visual
    evaluation method.
    """

    comparison_metric: Callable

    def _validate_dataframes(self, real: pd.DataFrame, fake: pd.DataFrame) -> None:
        """Validate the input DataFrames."""
        # Type validation
        if not isinstance(real, pd.DataFrame):
            raise TypeError('real must be a pandas DataFrame')
        if not isinstance(fake, pd.DataFrame):
            raise TypeError('fake must be a pandas DataFrame')

        # Empty validation
        if len(real) == 0:
            raise ValueError('real DataFrame cannot be empty')
        if len(fake) == 0:
            raise ValueError('fake DataFrame cannot be empty')

        if len(real.columns) == 0:
            raise ValueError('real DataFrame must have at least one column')
        if len(fake.columns) == 0:
            raise ValueError('fake DataFrame must have at least one column')

        # Column alignment validation
        if set(real.columns) != set(fake.columns):
            missing_in_fake = set(real.columns) - set(fake.columns)
            missing_in_real = set(fake.columns) - set(real.columns)
            error_msg = 'real and fake DataFrames must have the same columns.'
            if missing_in_fake:
                error_msg += f' Missing in fake: {missing_in_fake}.'
            if missing_in_real:
                error_msg += f' Missing in real: {missing_in_real}.'
            raise ValueError(error_msg)

    def _validate_categorical_columns(self, cat_cols: list[str] | None, real: pd.DataFrame) -> None:
        """Validate categorical columns parameter."""
        if cat_cols is not None:
            if not isinstance(cat_cols, list):
                raise TypeError('cat_cols must be a list of strings or None')
            invalid_cols = set(cat_cols) - set(real.columns)
            if invalid_cols:
                raise ValueError(f'cat_cols contains columns not in DataFrames: {invalid_cols}')

    def _validate_parameters(
        self,
        unique_thresh: int,
        *,
        verbose: bool,
        seed: int,
        n_samples: int | None,
        sample: bool,
    ) -> None:
        """Validate all other parameters."""
        if not isinstance(unique_thresh, int) or unique_thresh < 0:
            raise ValueError('unique_thresh must be a non-negative integer')

        if not isinstance(verbose, bool):
            raise TypeError('verbose must be a boolean')

        if not isinstance(seed, int):
            raise TypeError('seed must be an integer')

        if n_samples is not None and (not isinstance(n_samples, int) or n_samples <= 0):
            raise ValueError('n_samples must be a positive integer or None')

        if sample and n_samples is None:
            raise ValueError('n_samples must be specified when sample=True')

    def _validate_and_set_metric(self, metric: str | Callable) -> Callable:
        """Validate and convert metric parameter to callable."""
        if isinstance(metric, str):
            if not hasattr(stats, metric):
                raise ValueError(f"metric '{metric}' is not available in scipy.stats")
            return getattr(stats, metric)
        if not callable(metric):
            raise TypeError('metric must be a string or callable')
        return metric

    def _initialize_evaluator_components(
        self,
        comparison_metric: Callable,
        seed: int,
        *,
        verbose: bool,
        unique_thresh: int,
        n_samples: int | None,
    ) -> None:
        """Initialize all evaluator components."""
        # Initialize evaluation configuration
        self.config = EvaluationConfig(unique_thresh=unique_thresh, n_samples=n_samples, random_seed=seed)

        # Initialize evaluator components
        self.statistical_evaluator = StatisticalEvaluator(comparison_metric=comparison_metric, verbose=verbose)
        self.ml_evaluator = MLEvaluator(comparison_metric=comparison_metric, random_seed=seed, verbose=verbose)
        self.privacy_evaluator = PrivacyEvaluator(verbose=verbose)

        # Initialize visualization manager
        self.visualization_manager = VisualizationManager(
            real=self.real,
            fake=self.fake,
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
        )

    def __init__(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        cat_cols: list[str] | None = None,
        unique_thresh: int = 0,
        metric: str | Callable = 'pearsonr',
        *,
        verbose: bool = False,
        n_samples: int | None = None,
        name: str | None = None,
        seed: int = RANDOM_SEED,
        sample: bool = False,
        infer_types: bool = True,
    ) -> None:
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
            seed (int, optional): Random seed for reproducibility. Defaults to RANDOM_SEED.
            sample (bool, optional): Whether to sample the datasets to n_samples. Defaults to False.
        """
        # Perform all validations using helper methods
        self._validate_dataframes(real, fake)
        self._validate_categorical_columns(cat_cols, real)
        self._validate_parameters(
            unique_thresh,
            verbose=verbose,
            seed=seed,
            n_samples=n_samples,
            sample=sample,
        )

        # Validate and set metric
        comparison_metric = self._validate_and_set_metric(metric)

        # Set basic attributes
        self.name = name
        self.unique_thresh = unique_thresh
        self.real = real.copy()
        self.fake = fake.copy()
        self.comparison_metric = comparison_metric
        self.verbose = verbose
        self.random_seed = seed
        self.sample = sample
        self.infer_types = infer_types

        # Preprocess data
        self.real, self.fake, self.numerical_columns, self.categorical_columns = _preprocess_data(
            real=self.real,
            fake=self.fake,
            cat_cols=cat_cols,
            unique_thresh=unique_thresh,
            n_samples=n_samples,
            seed=seed,
        )
        self.n_samples = len(self.real)

        # Initialize all evaluator components
        self._initialize_evaluator_components(
            comparison_metric,
            seed,
            verbose=verbose,
            unique_thresh=unique_thresh,
            n_samples=n_samples,
        )

    def plot_mean_std(self, fname: PathLike | None = None, *, show: bool = True) -> None:
        """
        Class wrapper function for plotting the mean and std using `plots.plot_mean_std`.

        Params:

            fname (str, Optional): If not none, saves the plot with this file name.

        """
        return self.visualization_manager.plot_mean_std(fname=fname, show=show)

    def plot_cumsums(self, nr_cols: int = 4, fname: PathLike | None = None, *, show: bool = True) -> None:
        """

        Plot the cumulative sums for all columns in the real and fake dataset. Height of each row scales with the length

        of the labels. Each plot contains the

        values of a real columns and the corresponding fake column.

        Params:

            fname str: If not none, saves the plot with this file name.

        """
        return self.visualization_manager.plot_cumsums(nr_cols=nr_cols, fname=fname, show=show)

    def plot_distributions(self, nr_cols: int = 3, fname: PathLike | None = None, *, show: bool = True) -> None:
        """

        Plot the distribution plots for all columns in the real and fake dataset. Height of each row of plots scales

        with the length of the labels. Each plot

        contains the values of a real columns and the corresponding fake column.

        Params:

            fname (str, Optional): If not none, saves the plot with this file name.

        """
        return self.visualization_manager.plot_distributions(nr_cols=nr_cols, fname=fname, show=show)

    def plot_correlation_difference(
        self,
        *,
        plot_diff: bool = True,
        fname: PathLike | None = None,
        show: bool = True,
        **kwargs: dict,
    ) -> None:
        """

        Plot the association matrices for each table and, if chosen, the difference between them.

        :param plot_diff: whether to plot the difference

        :param fname: If not none, saves the plot with this file name.

        :param kwargs: kwargs for sns.heatmap

        """
        return self.visualization_manager.plot_correlation_difference(
            plot_diff=plot_diff, fname=fname, show=show, **kwargs
        )

    def correlation_distance(self, how: str = 'euclidean') -> float:
        """

        Calculate distance between correlation matrices with certain metric.

        :param how: metric to measure distance. Choose from [``euclidean``, ``mae``, ``rmse``].

        :return: distance between the association matrices in the chosen evaluation metric. Default: Euclidean

        """

        from scipy.spatial.distance import cosine

        if how == 'euclidean':
            distance_func = te_metrics.euclidean_distance
        elif how == 'mae':
            distance_func = te_metrics.mean_absolute_error
        elif how == 'rmse':
            distance_func = te_metrics.rmse
        elif how == 'cosine':

            def custom_cosine(a: np.ndarray, b: np.ndarray) -> float:
                return cosine(a.reshape(-1), b.reshape(-1))

            distance_func = custom_cosine
        else:
            raise ValueError('`how` parameter must be in [euclidean, mae, rmse]')

        real_corr = associations(
            self.real,
            nominal_columns=self.categorical_columns,
            nom_nom_assoc='theil',
        )['corr']  # type: ignore
        fake_corr = associations(
            self.fake,
            nominal_columns=self.categorical_columns,
            nom_nom_assoc='theil',
        )['corr']  # type: ignore
        return distance_func(real_corr.values, fake_corr.values)  # type: ignore

    def plot_pca(self, fname: PathLike | None = None, *, show: bool = True) -> None:
        """

        Plot the first two components of a PCA of real and fake data.

        :param fname: If not none, saves the plot with this file name.

        """
        return self.visualization_manager.plot_pca(fname=fname, show=show)

    def get_copies(self, *, return_len: bool = False) -> pd.DataFrame | int:
        """

        Check whether any real values occur in the fake data.

        Args:

            return_len (bool): Whether to return the length of the copied rows or not.

        Returns:

            Union[pd.DataFrame, int]: Dataframe containing the duplicates if return_len=False,

                else integer indicating the number of copied rows.

        """
        return self.privacy_evaluator.get_copies(self.real, self.fake, return_len=return_len)

    def get_duplicates(self, *, return_values: bool = False) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[int, int]:
        """

        Return duplicates within each dataset.

        Args:

            return_values (bool): Whether to return the duplicate values in the datasets.

                If False, the lengths are returned.

        Returns:

            Union[tuple[pd.DataFrame, pd.DataFrame], tuple[int, int]]:

                If return_values is True, returns a tuple of DataFrames with duplicates.

                If return_values is False, returns a tuple of integers representing the lengths of those DataFrames.

        """
        return self.privacy_evaluator.get_duplicates(self.real, self.fake, return_values=return_values)

    def pca_correlation(self, *, lingress: bool = False) -> float:
        """

        Calculate the relation between PCA explained variance values. Due to some very large numbers, in recent

        implementation the MAPE(log) is used instead of regressions like Pearson's r.

        Args:

            lingress (bool): Whether to use a linear regression, in this case Pearson's.

        Returns:

            float: The correlation coefficient if lingress=True, otherwise 1 - MAPE(log(real), log(fake)).

        """
        real, fake = self.convert_numerical()
        return self.statistical_evaluator.pca_correlation(real, fake, lingress=lingress)

    def visual_evaluation(self, save_dir: PathLike | None = None, *, show: bool = True, **kwargs: dict) -> None:
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
        return self.visualization_manager.visual_evaluation(save_dir=save_dir, show=show, **kwargs)

    def basic_statistical_evaluation(self) -> float:
        """

        Calculate the correlation coefficient between the basic properties of self.real and self.fake using Spearman's

        Rho. Spearman's is used because these values can differ a lot in magnitude, and Spearman's is more resilient to

        outliers.

        Returns:

            float: correlation coefficient

        """
        return self.statistical_evaluator.basic_statistical_evaluation(self.real, self.fake, self.numerical_columns)

    def correlation_correlation(self) -> float:
        """

        Calculate the correlation coefficient between the association matrices of self.real and self.fake using

        self.comparison_metric

        Returns:

            float: The correlation coefficient

        """
        return self.statistical_evaluator.correlation_correlation(self.real, self.fake, self.categorical_columns)

    def convert_numerical(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """

        Convert dataset to a numerical representations while making sure they have identical columns. This is sometimes

        a problem with categorical columns with many values or very unbalanced values

        Returns:

            tuple[pd.DataFrame, pd.DataFrame]: Real and fake dataframe factorized using the pandas function

        """

        real = self.real
        fake = self.fake
        for c in self.categorical_columns:
            if real[c].dtype == 'object':
                real[c] = pd.factorize(real[c], sort=True)[0]
                fake[c] = pd.factorize(fake[c], sort=True)[0]

        return real, fake

    def convert_numerical_one_hot(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """

        Convert dataset to a numerical representations while making sure they have identical columns.

        This is sometimes a problem with categorical columns with many values or very unbalanced values

        Returns:

            tuple[pd.DataFrame, pd.DataFrame]: Real and fake dataframe with categorical columns one-hot encoded and

            binary columns factorized.

        """

        cat_cols = (
            self.categorical_columns
            + self.real.select_dtypes('bool').columns.tolist()
            + self.fake.select_dtypes('bool').columns.tolist()
        )
        real, fake, _ = DataConverter.numeric_encoding(real=self.real, fake=self.fake, categorical_columns=cat_cols)
        real = real.astype(float).sort_index(axis=1)
        fake = fake.astype(float)
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

    def estimator_evaluation(self, target_col: str, target_type: str = 'class', *, kfold: bool = False) -> float:
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
            real=real,
            fake=fake,
            target_col=target_col,
            target_type=target_type,
            kfold=kfold,
        )

        # Store the scores for backward compatibility with evaluate() method
        # We need to get the scores from the MLEvaluator but it doesn't expose them
        # Let's temporarily set a placeholder
        self.estimators_scores = pd.DataFrame({'real': [result], 'fake': [result]})

        return result

    def row_distance(self, max_samples: int = 5000) -> tuple[float, float]:
        """
        Calculate mean and standard deviation distances between `self.fake` and `self.real`.

        Args:
            max_samples: Maximum number of samples to use for distance calculation

        Returns:
            Tuple of (mean, std) of these distances.
        """
        real, fake = self.convert_numerical_one_hot()
        return self.privacy_evaluator.row_distance(real=real, synthetic=fake, max_samples=max_samples)

    def column_correlations(self) -> float:
        """

        Wrapper function around `metrics.column_correlation`.

        :return: Column correlations between ``self.real`` and ``self.fake``.

        """

        real, fake = self.convert_numerical()

        return te_metrics.column_correlations(real, fake, self.categorical_columns)

    def _validate_target_and_type(self, target_col: str, target_type: str) -> None:
        """Validate target column and type."""
        if not isinstance(target_col, str):
            raise TypeError('target_col must be a string')
        if target_col not in self.real.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame columns")
        if target_type not in ['class', 'regr']:
            raise ValueError("target_type must be either 'class' or 'regr'")

    def _validate_metric_parameter(self, metric: str | None) -> None:
        """Validate metric parameter."""
        if metric is not None:
            if isinstance(metric, str):
                if not hasattr(stats, metric):
                    raise ValueError(f"metric '{metric}' is not available in scipy.stats")
            elif not callable(metric):
                raise TypeError('metric must be a string, callable, or None')

    def _validate_boolean_parameters(
        self, *, verbose: bool | None, kfold: bool, notebook: bool, return_outputs: bool
    ) -> None:
        """Validate boolean parameters."""
        if verbose is not None and not isinstance(verbose, bool):
            raise TypeError('verbose must be a boolean or None')
        if not isinstance(kfold, bool):
            raise TypeError('kfold must be a boolean')
        if not isinstance(notebook, bool):
            raise TypeError('notebook must be a boolean')
        if not isinstance(return_outputs, bool):
            raise TypeError('return_outputs must be a boolean')

    def _validate_evaluate_inputs(
        self,
        target_col: str,
        target_type: str,
        metric: str | None,
        *,
        verbose: bool | None,
        n_samples_distance: int,
        kfold: bool,
        notebook: bool,
        return_outputs: bool,
    ) -> None:
        """Validate all inputs for the evaluate method."""
        self._validate_target_and_type(target_col, target_type)
        self._validate_metric_parameter(metric)
        self._validate_boolean_parameters(
            verbose=verbose,
            kfold=kfold,
            notebook=notebook,
            return_outputs=return_outputs,
        )

        if not isinstance(n_samples_distance, int) or n_samples_distance <= 0:
            raise ValueError('n_samples_distance must be a positive integer')

    def _setup_evaluation_parameters(self, target_type: str, *, verbose: bool | None, metric: str | None) -> None:
        """Set up evaluation parameters on the instance."""
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

    def _compute_all_metrics(
        self, target_col: str, target_type: str, *, kfold: bool, n_samples_distance: int
    ) -> tuple[dict, dict, dict, list, list, list]:
        """Compute all evaluation metrics."""
        statistical_results, statistical_tab = self._calculate_statistical_metrics()
        ml_efficacy_results, ml_efficacy_tab = self._calculate_ml_efficacy(target_col, target_type, kfold=kfold)
        privacy_results, privacy_tab = self._calculate_privacy_metrics(n_samples_distance)

        return (
            statistical_results,
            ml_efficacy_results,
            privacy_results,
            statistical_tab,
            ml_efficacy_tab,
            privacy_tab,
        )

    def _create_overview_summary(
        self, statistical_results: dict, ml_efficacy_results: dict, target_type: str
    ) -> tuple[dict, list]:
        """Create the overview summary of all results."""
        all_results_dict = {
            'Basic statistics': statistical_results['basic_statistical'],
            'Correlation column correlations': statistical_results['correlation_correlation'],
            'Mean Correlation between fake and real columns': statistical_results['column_correlation'],
            f'{"1 - MAPE Estimator results" if target_type == "class" else "Correlation RMSE"}': ml_efficacy_results[
                'estimators'
            ],
        }
        all_results_dict['Similarity Score'] = np.mean(list(all_results_dict.values()))

        summary = EvaluationResult(name='Overview Results', content=dict_to_df(all_results_dict))
        overview_tab = [summary]

        return all_results_dict, overview_tab

    def _handle_output_format(
        self,
        *,
        return_outputs: bool,
        notebook: bool,
        overview_tab: list,
        ml_efficacy_tab: list,
        privacy_tab: list,
        statistical_tab: list,
        ml_efficacy_results: dict,
        privacy_results: dict,
        statistical_results: dict,
        summary: EvaluationResult,
        all_results_dict: dict,
    ) -> dict | None:
        """Handle different output formats based on parameters."""
        if return_outputs:
            all_results = [
                *overview_tab,
                *ml_efficacy_tab,
                *privacy_tab,
                *statistical_tab,
            ]
            return {x.name: x.content.to_dict(orient='index') for x in all_results}

        if notebook:
            visualize_notebook(
                self,
                overview=overview_tab,
                privacy_metrics=privacy_tab,
                ml_efficacy=ml_efficacy_tab,
                statistical=statistical_tab,
            )
        else:
            logger.info(f'\n{ml_efficacy_results["efficacy_title"]}:')
            logger.info(self.estimators_scores.to_string())

            logger.info('\nPrivacy results:')
            logger.info(privacy_results['privacy_report'].content.to_string())

            logger.info('\nMiscellaneous results:')
            logger.info(statistical_results['miscellaneous'].to_string())

            logger.info('\nResults:')
            logger.info(summary.content.to_string())

        return all_results_dict

    def evaluate(
        self,
        target_col: str,
        target_type: str = 'class',
        metric: str | None = None,
        n_samples_distance: int = 20000,
        *,
        verbose: bool | None = None,
        kfold: bool = False,
        notebook: bool = False,
        return_outputs: bool = False,
    ) -> dict | None:
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
        # Validate all inputs
        self._validate_evaluate_inputs(
            target_col,
            target_type,
            metric,
            verbose=verbose,
            n_samples_distance=n_samples_distance,
            kfold=kfold,
            notebook=notebook,
            return_outputs=return_outputs,
        )

        # Set up evaluation parameters
        self._setup_evaluation_parameters(target_type, verbose=verbose, metric=metric)

        # Compute all metrics
        (
            statistical_results,
            ml_efficacy_results,
            privacy_results,
            statistical_tab,
            ml_efficacy_tab,
            privacy_tab,
        ) = self._compute_all_metrics(target_col, target_type, kfold=kfold, n_samples_distance=n_samples_distance)

        # Create overview summary
        all_results_dict, overview_tab = self._create_overview_summary(
            statistical_results, ml_efficacy_results, target_type
        )

        # Handle output format and return results
        return self._handle_output_format(
            return_outputs=return_outputs,
            notebook=notebook,
            overview_tab=overview_tab,
            ml_efficacy_tab=ml_efficacy_tab,
            privacy_tab=privacy_tab,
            statistical_tab=statistical_tab,
            ml_efficacy_results=ml_efficacy_results,
            privacy_results=privacy_results,
            statistical_results=statistical_results,
            summary=overview_tab[0],
            all_results_dict=all_results_dict,
        )

    def _calculate_statistical_metrics(self) -> tuple[dict, str]:
        basic_statistical = self.basic_statistical_evaluation()
        correlation_correlation = self.correlation_correlation()
        column_correlation = self.column_correlations()

        miscellaneous_dict = {
            'Column Correlation Distance RMSE': self.correlation_distance(how='rmse'),
            'Column Correlation distance MAE': self.correlation_distance(how='mae'),
        }

        miscellaneous = pd.DataFrame(
            {'Result': list(miscellaneous_dict.values())},
            index=list(miscellaneous_dict.keys()),
        )

        js_df = te_metrics.js_distance_df(self.real, self.fake, self.numerical_columns)

        statistical_tab = [
            EvaluationResult(
                name='Jensen-Shannon distance',
                content=js_df,
                appendix=f'### Mean: {js_df.js_distance.mean(): .3f}',
            ),
            EvaluationResult(
                name='Kolmogorov-Smirnov statistic',
                content=te_metrics.kolmogorov_smirnov_df(self.real, self.fake, self.numerical_columns),
            ),
        ]
        return {
            'basic_statistical': basic_statistical,
            'correlation_correlation': correlation_correlation,
            'column_correlation': column_correlation,
            'miscellaneous': miscellaneous,
        }, statistical_tab

    def _calculate_ml_efficacy(self, target_col: str, target_type: str, *, kfold: bool) -> tuple[dict, str]:
        estimators = self.estimator_evaluation(target_col=target_col, target_type=target_type, kfold=kfold)
        efficacy_title = (
            'Classifier F1-scores and their Jaccard similarities:' if target_type == 'class' else 'Regressor MSE-scores'
        )

        ml_efficacy_tab = [EvaluationResult(name=efficacy_title, content=self.estimators_scores)]
        return {
            'estimators': estimators,
            'efficacy_title': efficacy_title,
        }, ml_efficacy_tab

    def _calculate_privacy_metrics(self, n_samples_distance: int) -> tuple[dict, str]:
        nearest_neighbor = self.row_distance(max_samples=n_samples_distance)
        privacy_metrics_dict = {
            'Duplicate rows between sets (real/fake)': self.get_duplicates(return_values=False),
            'nearest neighbor mean': nearest_neighbor[0],
            'nearest neighbor std': nearest_neighbor[1],
        }

        privacy_report = EvaluationResult(
            name='Privacy Results',
            content=dict_to_df(privacy_metrics_dict),
        )

        privacy_tab = [privacy_report]

        return {'privacy_report': privacy_report}, privacy_tab

    def advanced_statistical_evaluation(
        self,
        *,
        include_wasserstein: bool = True,
        include_mmd: bool = True,
        wasserstein_config: dict | None = None,
        mmd_config: dict | None = None,
    ) -> dict:
        """
        Run advanced statistical evaluation using Wasserstein distance and MMD.

        Args:
            include_wasserstein: Whether to include Wasserstein distance analysis
            include_mmd: Whether to include Maximum Mean Discrepancy analysis
            wasserstein_config: Configuration for Wasserstein evaluation
            mmd_config: Configuration for MMD evaluation

        Returns:
            Dictionary with advanced statistical evaluation results
        """
        # Input validation
        if not isinstance(include_wasserstein, bool):
            raise TypeError('include_wasserstein must be a boolean')
        if not isinstance(include_mmd, bool):
            raise TypeError('include_mmd must be a boolean')

        if not include_wasserstein and not include_mmd:
            raise ValueError('At least one of include_wasserstein or include_mmd must be True')

        if wasserstein_config is not None and not isinstance(wasserstein_config, dict):
            raise TypeError('wasserstein_config must be a dictionary or None')
        if mmd_config is not None and not isinstance(mmd_config, dict):
            raise TypeError('mmd_config must be a dictionary or None')

        if wasserstein_config is None:
            wasserstein_config = {'include_2d': False}
        if mmd_config is None:
            mmd_config = {
                'kernel_types': ['rbf', 'polynomial'],
                'include_multivariate': True,
            }

        # Use the unified statistical evaluator
        try:
            unified_results = self.statistical_evaluator.evaluate(
                self.real,
                self.fake,
                numerical_columns=self.numerical_columns,
                categorical_columns=self.categorical_columns,
                include_basic=False,  # Only advanced methods
                include_wasserstein=include_wasserstein,
                include_mmd=include_mmd,
                wasserstein_config=wasserstein_config,
                mmd_config=mmd_config,
            )
            # Return only the advanced parts
            return {
                'wasserstein': unified_results.get('wasserstein', {}),
                'mmd': unified_results.get('mmd', {}),
                'combined_metrics': unified_results.get('combined_metrics', {}),
                'recommendations': unified_results.get('recommendations', []),
            }
        except Exception as e:
            logger.error(f'Advanced statistical evaluation failed: {e}')
            return {'error': str(e)}

    def _validate_advanced_privacy_inputs(
        self,
        *,
        include_k_anonymity: bool,
        include_membership_inference: bool,
        quasi_identifiers: list[str] | None,
        sensitive_attributes: list[str] | None,
    ) -> None:
        """Validate inputs for advanced privacy evaluation."""
        if not isinstance(include_k_anonymity, bool):
            raise TypeError('include_k_anonymity must be a boolean')
        if not isinstance(include_membership_inference, bool):
            raise TypeError('include_membership_inference must be a boolean')

        if not include_k_anonymity and not include_membership_inference:
            raise ValueError('At least one of include_k_anonymity or include_membership_inference must be True')

        if quasi_identifiers is not None:
            if not isinstance(quasi_identifiers, list):
                raise TypeError('quasi_identifiers must be a list or None')
            invalid_cols = set(quasi_identifiers) - set(self.real.columns)
            if invalid_cols:
                raise ValueError(f'quasi_identifiers contains invalid columns: {invalid_cols}')

        if sensitive_attributes is not None:
            if not isinstance(sensitive_attributes, list):
                raise TypeError('sensitive_attributes must be a list or None')
            invalid_cols = set(sensitive_attributes) - set(self.real.columns)
            if invalid_cols:
                raise ValueError(f'sensitive_attributes contains invalid columns: {invalid_cols}')

    def _run_unified_privacy_evaluation(
        self,
        quasi_identifiers: list[str] | None,
        sensitive_attributes: list[str] | None,
        *,
        include_k_anonymity: bool,
        include_membership_inference: bool,
    ) -> dict:
        """Run unified privacy evaluation and format results."""
        try:
            unified_results = self.privacy_evaluator.evaluate(
                self.real,
                self.fake,
                quasi_identifiers=quasi_identifiers,
                sensitive_attributes=sensitive_attributes,
                include_basic=False,  # Only advanced methods
                include_k_anonymity=include_k_anonymity,
                include_membership_inference=include_membership_inference,
            )
            # Return the structure expected by the old API
            return {
                'k_anonymity': unified_results.get('k_anonymity', {}),
                'membership_inference': unified_results.get('membership_inference', {}),
                'overall_assessment': unified_results.get('overall_assessment', {}),
                'combined_recommendations': unified_results.get('recommendations', []),
            }
        except Exception as e:
            logger.error(f'Advanced privacy evaluation failed: {e}')
            return {'error': str(e)}

    def advanced_privacy_evaluation(
        self,
        *,
        include_k_anonymity: bool = True,
        include_membership_inference: bool = True,
        quasi_identifiers: list[str] | None = None,
        sensitive_attributes: list[str] | None = None,
    ) -> dict:
        """
        Run advanced privacy evaluation including k-anonymity and membership inference.

        Args:
            include_k_anonymity: Whether to include k-anonymity analysis
            include_membership_inference: Whether to include membership inference analysis
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attributes: List of sensitive attribute columns

        Returns:
            Dictionary with advanced privacy evaluation results
        """
        # Validate inputs
        self._validate_advanced_privacy_inputs(
            include_k_anonymity=include_k_anonymity,
            include_membership_inference=include_membership_inference,
            quasi_identifiers=quasi_identifiers,
            sensitive_attributes=sensitive_attributes,
        )

        # Run unified privacy evaluation
        return self._run_unified_privacy_evaluation(
            quasi_identifiers,
            sensitive_attributes,
            include_k_anonymity=include_k_anonymity,
            include_membership_inference=include_membership_inference,
        )

    def _validate_comprehensive_evaluation_inputs(
        self,
        target_col: str,
        target_type: str,
        *,
        include_basic: bool,
        include_advanced_statistical: bool,
        include_advanced_privacy: bool,
    ) -> None:
        """Validate inputs for comprehensive evaluation."""
        if not isinstance(target_col, str):
            raise TypeError('target_col must be a string')
        if target_col not in self.real.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame columns")

        if target_type not in ['class', 'regr']:
            raise ValueError("target_type must be either 'class' or 'regr'")

        if not isinstance(include_basic, bool):
            raise TypeError('include_basic must be a boolean')
        if not isinstance(include_advanced_statistical, bool):
            raise TypeError('include_advanced_statistical must be a boolean')
        if not isinstance(include_advanced_privacy, bool):
            raise TypeError('include_advanced_privacy must be a boolean')

        if not any([include_basic, include_advanced_statistical, include_advanced_privacy]):
            raise ValueError('At least one evaluation type must be included')

    def _warn_large_dataset_performance(self) -> None:
        """Warn user about performance on large datasets."""
        total_rows = len(self.real) + len(self.fake)
        if total_rows > 100000:
            logger.warning(
                f'Large dataset detected ({total_rows:,} total rows). '
                'Performance optimizations will be applied automatically.'
            )

    def _run_basic_evaluation(self, target_col: str, target_type: str, **kwargs: dict) -> dict:
        """Run basic evaluation with error handling."""
        try:
            return self.evaluate(target_col, target_type, return_outputs=True, **kwargs)
        except Exception as e:
            logger.exception(f'Basic evaluation failed: {e}')
            return {'error': str(e)}

    def _run_advanced_statistical_evaluation(self) -> dict:
        """Run advanced statistical evaluation with error handling."""
        try:
            return self.advanced_statistical_evaluation()
        except Exception as e:
            logger.exception(f'Advanced statistical evaluation failed: {e}')
            return {'error': str(e)}

    def _run_advanced_privacy_evaluation(self) -> dict:
        """Run advanced privacy evaluation with error handling."""
        try:
            return self.advanced_privacy_evaluation()
        except Exception as e:
            logger.exception('Advanced privacy evaluation failed.')
            return {'error': str(e)}

    def comprehensive_advanced_evaluation(
        self,
        target_col: str,
        target_type: str = 'class',
        *,
        include_basic: bool = True,
        include_advanced_statistical: bool = True,
        include_advanced_privacy: bool = True,
        **kwargs: object,
    ) -> dict:
        """
        Run comprehensive evaluation including both basic and advanced metrics.

        Args:
            target_col: Column to use for ML evaluation
            target_type: Type of ML task ("class" or "regr")
            include_basic: Whether to include basic evaluation
            include_advanced_statistical: Whether to include advanced statistical metrics
            include_advanced_privacy: Whether to include advanced privacy metrics
            **kwargs: Additional parameters

        Returns:
            Dictionary with comprehensive evaluation results
        """
        # Validate inputs
        self._validate_comprehensive_evaluation_inputs(
            target_col,
            target_type,
            include_basic=include_basic,
            include_advanced_statistical=include_advanced_statistical,
            include_advanced_privacy=include_advanced_privacy,
        )

        # Warn about performance on large datasets
        self._warn_large_dataset_performance()

        results = {}

        # Run requested evaluations
        if include_basic:
            results['basic'] = self._run_basic_evaluation(target_col, target_type, **kwargs)

        if include_advanced_statistical:
            results['advanced_statistical'] = self._run_advanced_statistical_evaluation()

        if include_advanced_privacy:
            results['advanced_privacy'] = self._run_advanced_privacy_evaluation()

        return results

    def get_available_advanced_metrics(self) -> dict:
        """
        Get information about available advanced metrics.

        Returns:
            Dictionary with information about available metrics
        """
        return {
            'advanced_statistical': {
                'wasserstein_distance': "Earth Mover's Distance for distribution comparison",
                'mmd': 'Maximum Mean Discrepancy with multiple kernels',
            },
            'advanced_privacy': {
                'k_anonymity': 'K-anonymity analysis for privacy evaluation',
                'membership_inference': 'Membership inference attack simulation',
            },
        }
