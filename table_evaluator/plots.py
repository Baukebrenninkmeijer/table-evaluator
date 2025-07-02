from collections.abc import Sequence
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from table_evaluator.association_metrics import associations


def plot_correlation_difference(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    plot_diff: bool = True,
    cat_cols: Optional[list[str]] = None,
    annot: bool = False,
    fname: str | None = None,
    show: bool = True,
):
    """
    Plot the association matrices for the `real` dataframe, `fake` dataframe and plot the difference between them. Has
    support for continuous and Categorical (Male, Female) data types. All Object and Category dtypes are considered to
    be Categorical columns if `dis_cols` is not passed.

    - Continuous - Continuous: Uses Pearson's correlation coefficient
    - Continuous - Categorical: Uses so called correlation ratio (https://en.wikipedia.org/wiki/Correlation_ratio) for
      both continuous - categorical and categorical - continuous.
    - Categorical - Categorical: Uses Theil's U, an asymmetric correlation metric for Categorical associations

    Args:
        real (pd.DataFrame): DataFrame with real data.
        fake (pd.DataFrame): DataFrame with synthetic data.
        plot_diff (bool): Plot difference if True, else not.
        cat_cols (Optional[list[str]]): List of Categorical columns.
        annot (bool): Whether to annotate the plot with numbers indicating the associations.
    """
    assert isinstance(
        real, pd.DataFrame
    ), "`real` parameters must be a Pandas DataFrame"
    assert isinstance(
        fake, pd.DataFrame
    ), "`fake` parameters must be a Pandas DataFrame"
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.set(style="white")

    if cat_cols is None:
        cat_cols = real.select_dtypes(["object", "category"])
    if plot_diff:
        fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    real_corr = associations(
        real,
        nominal_columns=cat_cols,
        # plot=False,
        nom_nom_assoc="theil",
        # mark_columns=True,
        # annot=annot,
        # ax=ax[0],
        # cmap=cmap,
        compute_only=True,
    )["corr"]
    fake_corr = associations(
        fake,
        nominal_columns=cat_cols,
        # plot=False,
        nom_nom_assoc="theil",
        # mark_columns=True,
        # annot=annot,
        # ax=ax[1],
        # cmap=cmap,
        compute_only=True,
    )["corr"]

    sns.heatmap(
        real_corr,
        ax=ax[0],
        cmap=cmap,
        square=True,
        annot=annot,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        fmt=".2f",
    )
    sns.heatmap(
        fake_corr,
        ax=ax[1],
        cmap=cmap,
        square=True,
        annot=annot,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        fmt=".2f",
    )

    if plot_diff:
        diff = (real_corr - fake_corr).abs()
        sns.heatmap(
            diff,
            ax=ax[2],
            cmap=cmap,
            vmax=0.3,
            square=True,
            annot=annot,
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            fmt=".2f",
        )
    titles = ["Real", "Fake", "Difference"] if plot_diff else ["Real", "Fake"]
    for i, label in enumerate(titles):
        title_font = {"size": "18"}
        ax[i].set_title(label, **title_font)
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        return fig


def plot_distributions(
    real: pd.DataFrame, fake: pd.DataFrame, nr_cols=3, fname=None, show: bool = True
):
    """
    Plot the distribution plots for all columns in the real and fake dataset.
    Height of each row of plots scales with the length of the labels. Each plot
    contains the values of a real columns and the corresponding fake column.
    :param real: Real dataset (pd.DataFrame)
    :param fake: Synthetic dataset (pd.DataFrame)
    :param nr_cols: Number of columns for the subplot grid.
    :param fname: If not none, saves the plot with this file name.
    """
    nr_charts = len(real.columns)
    nr_rows = max(1, nr_charts // nr_cols)
    nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

    max_len = 0
    # Increase the length of plots if the labels are long
    if not real.select_dtypes(include=["object"]).empty:
        lengths = []
        for d in real.select_dtypes(include=["object"]):
            lengths.append(max([len(x.strip()) for x in real[d].unique().tolist()]))
        max_len = max(lengths)

    row_height = 6 + (max_len // 30)
    fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
    fig.suptitle("Distribution per feature", fontsize=16)
    axes = ax.flatten()
    for i, col in enumerate(real.columns):
        if col not in real.select_dtypes(include=["object", "category"]).columns:
            plot_df = pd.DataFrame(
                {
                    col: pd.concat([real[col], fake[col]], axis=0),
                    "kind": ["real"] * len(real) + ["fake"] * len(fake),
                }
            )
            sns.histplot(
                plot_df,
                x=col,
                hue="kind",
                ax=axes[i],
                stat="probability",
                legend=True,
                kde=True,
            )
            axes[i].set_autoscaley_on(True)
        else:
            real_temp = real.copy()
            fake_temp = fake.copy()
            real_temp["kind"] = "Real"
            fake_temp["kind"] = "Fake"
            concat = pd.concat([fake_temp, real_temp])
            palette = sns.color_palette(
                [
                    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
                ]
            )
            x, y, hue = col, "proportion", "kind"
            ax_curr = (
                concat[x]
                .groupby(concat[hue])
                .value_counts(normalize=True)
                .rename(y)
                .reset_index()
                .pipe(
                    (sns.barplot, "data"),
                    x=x,
                    y=y,
                    hue=hue,
                    ax=axes[i],
                    saturation=0.8,
                    palette=palette,
                )
            )
            ax_curr.set_xticklabels(axes[i].get_xticklabels(), rotation="vertical")
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    if fname is not None:
        plt.savefig(fname)
        if not show:
            plt.close(fig)  # Close the figure to prevent it from being displayed
    if show:
        plt.show()
    elif fname is None:
        plt.close(fig)


def plot_correlation_comparison(
    evaluators: Sequence, annot: bool = False, show: bool = False
):
    """
    Plot the correlation differences of multiple TableEvaluator objects.

    Args:
        evaluators (List[TableEvaluator]): List of TableEvaluator objects.
        annot (bool): Whether to annotate the plots with numbers.
    """
    nr_plots = len(evaluators) + 1
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, ax = plt.subplots(2, nr_plots, figsize=(4 * nr_plots, 7))
    flat_ax = ax.flatten()
    flat_ax[nr_plots + 1].clear()
    fake_corr = []
    real_corr = associations(
        evaluators[0].real,
        nominal_columns=evaluators[0].categorical_columns,
        plot=False,
        nom_nom_assoc="theil",
        compute_only=True,
        mark_columns=True,
        annot=False,
        cmap=cmap,
        cbar=False,
        ax=flat_ax[0],
    )["corr"]
    for i in range(1, nr_plots):
        cbar = True if i % (nr_plots - 1) == 0 else False
        fake_corr.append(
            associations(
                evaluators[i - 1].fake,
                nominal_columns=evaluators[0].categorical_columns,
                plot=False,
                nom_nom_assoc="theil",
                compute_only=True,
                mark_columns=True,
                annot=False,
                cmap=cmap,
                cbar=cbar,
                ax=flat_ax[i],
            )["corr"]
        )
        if i % (nr_plots - 1) == 0:
            cbar = flat_ax[i].collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)

    for i in range(1, nr_plots):
        cbar = True if i % (nr_plots - 1) == 0 else False
        diff = abs(real_corr - fake_corr[i - 1])
        sns.set(style="white")
        az = sns.heatmap(
            diff,
            ax=flat_ax[i + nr_plots],
            cmap=cmap,
            vmax=0.3,
            square=True,
            annot=annot,
            center=0,
            linewidths=0,
            cbar=cbar,
            fmt=".2f",
        )
        if i % (nr_plots - 1) == 0:
            cbar = az.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
    titles = ["Real"] + [
        e.name if e.name is not None else idx for idx, e in enumerate(evaluators)
    ]
    for i, label in enumerate(titles):
        flat_ax[i].set_yticklabels([])
        flat_ax[i].set_xticklabels([])
        flat_ax[i + nr_plots].set_yticklabels([])
        flat_ax[i + nr_plots].set_xticklabels([])
        title_font = {"size": "28"}
        flat_ax[i].set_title(label, **title_font)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def plot_cumsums(
    real: pd.DataFrame, fake: pd.DataFrame, fname: str | None = None, show: bool = True
):
    """
    Plot cumulative sum plots for all columns in the dataframes.

    Args:
        real (pd.DataFrame): DataFrame with real data.
        fake (pd.DataFrame): DataFrame with fake data.
        fname (str | None): Optional filename to save the plot to.
        show (bool): Whether to display the plot.
    """
    import math

    numerical_columns = real.select_dtypes(include=[np.number]).columns
    if len(numerical_columns) == 0:
        return

    n_cols = min(3, len(numerical_columns))
    n_rows = math.ceil(len(numerical_columns) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(numerical_columns):
        ax = axes[i] if len(numerical_columns) > 1 else axes[0]
        cdf(real[col], fake[col], xlabel=col, ax=ax, show=False)

    # Hide unused subplots
    for i in range(len(numerical_columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)
        if not show:
            plt.close()
    if show:
        plt.show()


def cdf(
    data_r,
    data_f,
    xlabel: str = "Values",
    ylabel: str = "Cumulative Sum",
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """
    Plot continous density function on optionally given ax. If no ax, cdf is plotted and shown.

    Args:
        data_r (pd.Series): Series with real data.
        data_f (pd.Series): Series with fake data.
        xlabel (str): Label to put on the x-axis.
        ylabel (str): Label to put on the y-axis.
        ax (matplotlib.axes.Axes | None): The axis to plot on. If None, a new figure is created.
        show (bool): Whether to display the plot. Defaults to True.

    Returns:
        matplotlib.axes.Axes | None: The axis with the plot if show is False, otherwise None.
    """
    x1 = data_r.sort_values()
    x2 = data_f.sort_values()
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    local_ax = ax if ax else plt.subplots()[1]

    axis_font = {"size": "14"}
    local_ax.set_xlabel(xlabel, **axis_font)
    local_ax.set_ylabel(ylabel, **axis_font)

    local_ax.grid()
    local_ax.plot(x1, y, marker="o", linestyle="none", label="Real", ms=8)
    local_ax.plot(x2, y, marker="o", linestyle="none", label="Fake", alpha=0.5)
    local_ax.tick_params(axis="both", which="major", labelsize=8)
    local_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
    import matplotlib.ticker as mticker

    # If labels are strings, rotate them vertical
    if isinstance(data_r, pd.Series) and data_r.dtypes == "object":
        all_labels = set(data_r) | set(data_f)
        ticks_loc = local_ax.get_xticks()
        local_ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        local_ax.set_xticklabels(sorted(all_labels), rotation="vertical")

    if ax is None:
        if fname is not None:
            plt.savefig(fname)
            if not show:
                plt.close()
        if show:
            plt.show()
        else:
            return local_ax


def plot_mean_std_comparison(evaluators: List, show: bool = True):
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
        title_font = {"size": "24"}
        flat_ax[i].set_title(label, **title_font)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def plot_mean_std(
    real: pd.DataFrame, fake: pd.DataFrame, ax=None, fname=None, show: bool = True
):
    """
    Plot the means and standard deviations of each dataset.

    :param real: DataFrame containing the real data
    :param fake: DataFrame containing the fake data
    :param ax: Axis to plot on. If none, a new figure is made.
    :param fname: If not none, saves the plot with this file name.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Absolute Log Mean and STDs of numeric data\n", fontsize=16)

    ax[0].grid(True)
    ax[1].grid(True)
    real = real.select_dtypes(include="number")
    fake = fake.select_dtypes(include="number")
    real_mean = np.log(np.add(abs(real.mean()).values, 1e-5))
    fake_mean = np.log(np.add(abs(fake.mean()).values, 1e-5))
    min_mean = min(real_mean) - 1
    max_mean = max(real_mean) + 1
    line = np.arange(min_mean, max_mean)
    sns.lineplot(x=line, y=line, ax=ax[0])
    sns.scatterplot(x=real_mean, y=fake_mean, ax=ax[0])
    ax[0].set_title("Means of real and fake data")
    ax[0].set_xlabel("real data mean (log)")
    ax[0].set_ylabel("fake data mean (log)")

    real_std = np.log(np.add(real.std().values, 1e-5))
    fake_std = np.log(np.add(fake.std().values, 1e-5))
    min_std = min(real_std) - 1
    max_std = max(real_std) + 1
    line = np.arange(min_std, max_std)
    sns.lineplot(x=line, y=line, ax=ax[1])
    sns.scatterplot(x=real_std, y=fake_std, ax=ax[1])
    ax[1].set_title("Stds of real and fake data")
    ax[1].set_xlabel("real data std (log)")
    ax[1].set_ylabel("fake data std (log)")

    if fname is not None:
        plt.savefig(fname)
        plt.close(fig)
    elif ax is None:
        plt.show()
    else:
        return ax
