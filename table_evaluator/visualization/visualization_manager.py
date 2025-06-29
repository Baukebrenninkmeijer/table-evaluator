"""Visualization manager for coordinating all table evaluation plots."""

from os import PathLike
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from table_evaluator.plots import cdf, plot_correlation_difference, plot_mean_std


class VisualizationManager:
    """Manages and coordinates all visualization operations for table evaluation."""

    def __init__(
        self,
        real: pd.DataFrame,
        fake: pd.DataFrame,
        categorical_columns: List[str],
        numerical_columns: List[str],
    ):
        """
        Initialize the visualization manager.

        Args:
            real: Real dataset
            fake: Synthetic dataset
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
        """
        self.real = real
        self.fake = fake
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def plot_mean_std(self, fname: Optional[PathLike] = None, show: bool = True):
        """
        Plot mean and standard deviation comparison between real and fake data.

        Args:
            fname: If not None, saves the plot with this file name
            show: Whether to display the plot
        """
        plot_mean_std(self.real, self.fake, fname=fname, show=show)

    def plot_cumsums(
        self, nr_cols: int = 4, fname: Optional[PathLike] = None, show: bool = True
    ):
        """
        Plot cumulative sums for all columns in real and fake datasets.

        Height of each row scales with the length of the labels. Each plot contains
        the values of a real column and the corresponding fake column.

        Args:
            nr_cols: Number of columns in the subplot grid
            fname: If not None, saves the plot with this file name
            show: Whether to display the plot
        """
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        # Calculate row height based on label lengths
        row_height = self._calculate_label_based_height()

        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle("Cumulative Sums per feature", fontsize=16)
        axes = ax.flatten()

        for i, col in enumerate(self.real.columns):
            try:
                real_col = self.real[col]
                fake_col = self.fake.iloc[:, self.real.columns.tolist().index(col)]
                cdf(real_col, fake_col, col, "Cumsum", ax=axes[i])
            except Exception as e:
                print(f"Error while plotting column {col}")
                raise e

        plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))

        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()

    def plot_distributions(
        self, nr_cols: int = 3, fname: Optional[PathLike] = None, show: bool = True
    ):
        """
        Plot distributions for all columns, using appropriate plot type per column type.

        Args:
            nr_cols: Number of columns in the subplot grid
            fname: If not None, saves the plot with this file name
            show: Whether to display the plot
        """
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        # Calculate row height based on label lengths
        row_height = self._calculate_label_based_height()

        fig, axes = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle("Distribution per feature", fontsize=16)
        axes = axes.flatten()

        for i, col in enumerate(self.real.columns):
            if col in self.numerical_columns:
                self._plot_numerical_distribution(col, axes[i])
            else:
                self._plot_categorical_distribution(col, axes[i])

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))

        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()

    def plot_correlation_difference(
        self,
        plot_diff: bool = True,
        fname: Optional[PathLike] = None,
        show: bool = True,
        **kwargs,
    ):
        """
        Plot correlation matrix difference between real and fake data.

        Args:
            plot_diff: Whether to plot the difference matrix
            fname: If not None, saves the plot with this file name
            show: Whether to display the plot
            **kwargs: Additional arguments for the plotting function
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

    def plot_pca(self, fname: Optional[PathLike] = None, show: bool = True):
        """
        Plot PCA comparison between real and fake data.

        Args:
            fname: If not None, saves the plot with this file name
            show: Whether to display the plot
        """
        # Convert data to numerical for PCA
        from table_evaluator.data.data_converter import DataConverter

        converter = DataConverter()
        real_num, fake_num = converter.to_numerical(
            self.real, self.fake, self.categorical_columns
        )

        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real_num)
        fake_pca = pca.transform(fake_num)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot real data PCA
        sns.scatterplot(x=real_pca[:, 0], y=real_pca[:, 1], ax=ax1, alpha=0.7)
        ax1.set_title("Real Data PCA")
        ax1.set_xlabel("First Principal Component")
        ax1.set_ylabel("Second Principal Component")

        # Plot fake data PCA
        sns.scatterplot(x=fake_pca[:, 0], y=fake_pca[:, 1], ax=ax2, alpha=0.7)
        ax2.set_title("Fake Data PCA")
        ax2.set_xlabel("First Principal Component")
        ax2.set_ylabel("Second Principal Component")

        plt.tight_layout()

        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()

    def visual_evaluation(
        self, save_dir: Optional[PathLike] = None, show: bool = True, **kwargs
    ):
        """
        Generate all visual evaluation plots.

        Includes mean/std, cumulative sums, distributions, correlation differences,
        and PCA comparisons.

        Args:
            save_dir: Directory path to save images (if None, images are not saved)
            show: Whether to display the plots
            **kwargs: Additional keyword arguments for matplotlib
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

            self.plot_mean_std(fname=save_dir / "mean_std.png", show=show)
            self.plot_cumsums(fname=save_dir / "cumsums.png", show=show)
            self.plot_distributions(fname=save_dir / "distributions.png", show=show)
            self.plot_correlation_difference(
                fname=save_dir / "correlation_difference.png", show=show, **kwargs
            )
            self.plot_pca(fname=save_dir / "pca.png", show=show)

    def _calculate_label_based_height(self, base_height: int = 6) -> int:
        """
        Calculate plot height based on the length of categorical labels.

        Args:
            base_height: Base height for the plot

        Returns:
            int: Calculated height based on label lengths
        """
        max_len = 0

        # Find the maximum label length in categorical columns
        categorical_data = self.real.select_dtypes(include=["object"])
        if not categorical_data.empty:
            lengths = []
            for col in categorical_data.columns:
                unique_values = categorical_data[col].unique().tolist()
                col_max_len = max([len(str(x).strip()) for x in unique_values])
                lengths.append(col_max_len)
            max_len = max(lengths) if lengths else 0

        return base_height + (max_len // 30)

    def _plot_numerical_distribution(self, column: str, ax):
        """Plot histogram for numerical column."""
        sns.histplot(
            data=self.real, x=column, alpha=0.7, label="Real", ax=ax, stat="density"
        )
        sns.histplot(
            data=self.fake, x=column, alpha=0.7, label="Fake", ax=ax, stat="density"
        )
        ax.set_title(f"{column} Distribution")
        ax.legend()

    def _plot_categorical_distribution(self, column: str, ax):
        """Plot bar chart for categorical column."""
        # Get value counts for both datasets
        real_counts = self.real[column].value_counts(normalize=True)
        fake_counts = self.fake[column].value_counts(normalize=True)

        # Combine and align the data
        all_categories = list(set(real_counts.index) | set(fake_counts.index))
        real_aligned = [real_counts.get(cat, 0) for cat in all_categories]
        fake_aligned = [fake_counts.get(cat, 0) for cat in all_categories]

        # Create bar plot
        x_pos = range(len(all_categories))
        width = 0.35

        ax.bar(
            [x - width / 2 for x in x_pos], real_aligned, width, label="Real", alpha=0.7
        )
        ax.bar(
            [x + width / 2 for x in x_pos], fake_aligned, width, label="Fake", alpha=0.7
        )

        ax.set_title(f"{column} Distribution")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_categories, rotation=45, ha="right")
        ax.legend()
        ax.set_ylabel("Frequency")
