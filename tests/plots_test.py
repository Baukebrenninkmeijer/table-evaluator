import os

import numpy as np
import pandas as pd

from table_evaluator.plots import plot_cumsums, plot_distributions


def test_plot_cumsums(tmp_path):
    real_data = pd.DataFrame(np.random.rand(100, 3), columns=["col1", "col2", "col3"])
    fake_data = pd.DataFrame(np.random.rand(100, 3), columns=["col1", "col2", "col3"])

    # Test with fname (saving to file)
    fname = tmp_path / "cumsums.png"
    plot_cumsums(real_data, fake_data, fname=fname, show=False)
    assert os.path.exists(fname)

    # Test without fname but with show=False to prevent plot windows during testing
    plot_cumsums(real_data, fake_data, show=False)


def test_plot_distributions(tmp_path):
    real_data = pd.DataFrame(np.random.rand(100, 3), columns=["col1", "col2", "col3"])
    fake_data = pd.DataFrame(np.random.rand(100, 3), columns=["col1", "col2", "col3"])

    # Test with fname (saving to file)
    fname = tmp_path / "distributions.png"
    plot_distributions(real_data, fake_data, fname=fname, show=False)
    assert os.path.exists(fname)

    # Test without fname but with show=False to prevent plot windows during testing
    plot_distributions(real_data, fake_data, show=False)
