"""Tests for the VisualizationManager class."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from table_evaluator.visualization.visualization_manager import VisualizationManager


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample data for testing."""
    real_data = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [0.1, 0.2, 0.3, 0.4, 0.5],
            "D": [True, False, True, False, True],
        }
    )
    fake_data = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "f"],
            "C": [0.15, 0.25, 0.35, 0.45, 0.55],
            "D": [True, True, False, False, True],
        }
    )
    return real_data, fake_data


def test_visualization_manager_initialization(sample_data):
    """Test VisualizationManager initialization."""
    real, fake = sample_data
    categorical_columns = ["B", "D"]
    numerical_columns = ["A", "C"]

    manager = VisualizationManager(real, fake, categorical_columns, numerical_columns)

    assert manager.real.equals(real)
    assert manager.fake.equals(fake)
    assert manager.categorical_columns == categorical_columns
    assert manager.numerical_columns == numerical_columns


def test_plot_mean_std(sample_data):
    """Test plot_mean_std method."""
    real, fake = sample_data
    manager = VisualizationManager(real, fake, ["B", "D"], ["A", "C"])

    with patch(
        "table_evaluator.visualization.visualization_manager.plot_mean_std"
    ) as mock_plot:
        manager.plot_mean_std(show=False)
        mock_plot.assert_called_once_with(real, fake, fname=None, show=False)


def test_plot_cumsums(sample_data):
    """Test plot_cumsums method."""
    real, fake = sample_data
    manager = VisualizationManager(real, fake, ["B", "D"], ["A", "C"])

    # Just test that the method exists and can be called without error in a mocked environment
    with patch(
        "table_evaluator.visualization.visualization_manager.plt"
    ) as mock_plt, patch(
        "table_evaluator.visualization.visualization_manager.cdf"
    ) as mock_cdf:
        # Mock plt.subplots to return proper mock objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_axes.flatten.return_value = [MagicMock() for _ in range(len(real.columns))]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Test that the method can be called
        manager.plot_cumsums(show=False)

        # Verify that plotting functions were called
        mock_plt.subplots.assert_called_once()
        assert mock_cdf.call_count == len(real.columns)


def test_plot_distributions(sample_data):
    """Test plot_distributions method."""
    real, fake = sample_data
    manager = VisualizationManager(real, fake, ["B", "D"], ["A", "C"])

    # Test that the method exists and can be called without error in a mocked environment
    with patch(
        "table_evaluator.visualization.visualization_manager.plt"
    ) as mock_plt, patch("table_evaluator.visualization.visualization_manager.sns"):
        # Mock plt.subplots to return proper mock objects
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_axes.flatten.return_value = [MagicMock() for _ in range(len(real.columns))]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Test that the method can be called
        manager.plot_distributions(show=False)

        # Verify that plotting functions were called
        mock_plt.subplots.assert_called_once()


def test_plot_correlation_difference(sample_data):
    """Test plot_correlation_difference method."""
    real, fake = sample_data
    manager = VisualizationManager(real, fake, ["B", "D"], ["A", "C"])

    with patch(
        "table_evaluator.visualization.visualization_manager.plot_correlation_difference"
    ) as mock_plot:
        manager.plot_correlation_difference(show=False)
        mock_plot.assert_called_once_with(
            real, fake, cat_cols=["B", "D"], plot_diff=True, fname=None, show=False
        )


def test_plot_pca(sample_data):
    """Test plot_pca method."""
    real, fake = sample_data
    manager = VisualizationManager(real, fake, ["B", "D"], ["A", "C"])

    with patch("matplotlib.pyplot.subplots") as mock_subplots, patch(
        "matplotlib.pyplot.tight_layout"
    ), patch("matplotlib.pyplot.show"), patch("seaborn.scatterplot"), patch(
        "sklearn.decomposition.PCA"
    ) as mock_pca:
        # Mock the subplots return value
        from unittest.mock import MagicMock

        fig_mock = MagicMock()
        ax1_mock, ax2_mock = MagicMock(), MagicMock()
        mock_subplots.return_value = (fig_mock, (ax1_mock, ax2_mock))

        # Mock PCA
        pca_instance = MagicMock()
        pca_instance.fit_transform.return_value = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ]
        pca_instance.transform.return_value = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        mock_pca.return_value = pca_instance

        manager.plot_pca(show=False)


def test_visual_evaluation_without_save_dir(sample_data):
    """Test visual_evaluation method without save directory."""
    real, fake = sample_data
    manager = VisualizationManager(real, fake, ["B", "D"], ["A", "C"])

    with patch.object(manager, "plot_mean_std") as mock_mean_std, patch.object(
        manager, "plot_cumsums"
    ) as mock_cumsums, patch.object(
        manager, "plot_distributions"
    ) as mock_distributions, patch.object(
        manager, "plot_correlation_difference"
    ) as mock_corr, patch.object(manager, "plot_pca") as mock_pca:
        manager.visual_evaluation(show=False)

        mock_mean_std.assert_called_once_with(show=False)
        mock_cumsums.assert_called_once_with(show=False)
        mock_distributions.assert_called_once_with(show=False)
        mock_corr.assert_called_once_with(show=False)
        mock_pca.assert_called_once_with(show=False)


def test_calculate_label_based_height(sample_data):
    """Test _calculate_label_based_height method."""
    real, fake = sample_data
    manager = VisualizationManager(real, fake, ["B", "D"], ["A", "C"])

    height = manager._calculate_label_based_height()
    assert isinstance(height, int)
    assert height >= 6  # Base height


def test_plot_numerical_distribution(sample_data):
    """Test _plot_numerical_distribution method."""
    real, fake = sample_data
    manager = VisualizationManager(real, fake, ["B", "D"], ["A", "C"])

    with patch("seaborn.histplot") as mock_hist:
        from unittest.mock import MagicMock

        ax = MagicMock()

        manager._plot_numerical_distribution("A", ax)

        assert mock_hist.call_count == 2  # Called for both real and fake data


def test_plot_categorical_distribution(sample_data):
    """Test _plot_categorical_distribution method."""
    real, fake = sample_data
    manager = VisualizationManager(real, fake, ["B", "D"], ["A", "C"])

    from unittest.mock import MagicMock

    ax = MagicMock()

    manager._plot_categorical_distribution("B", ax)

    # Verify that bar plotting methods were called
    assert ax.bar.call_count == 2  # Called for both real and fake data
    ax.set_title.assert_called_once()
    ax.legend.assert_called_once()
