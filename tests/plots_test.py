import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from table_evaluator.plots import plot_cumsums, plot_distributions

def test_plot_cumsums(tmp_path):
    real_data = pd.DataFrame(np.random.rand(100, 3), columns=['col1', 'col2', 'col3'])
    fake_data = pd.DataFrame(np.random.rand(100, 3), columns=['col1', 'col2', 'col3'])
    
    # Test with fname (saving to file)
    fname = tmp_path / "cumsums.png"
    plot_cumsums(real_data, fake_data, fname=fname)
    assert os.path.exists(fname)

    # Test without fname (displaying plot) - this will open a plot window
    # We can't directly assert if a plot window is opened, but we can check if no error occurs
    plot_cumsums(real_data, fake_data)
    # No assert needed here, just checking for no exceptions

def test_plot_distributions(tmp_path):
    real_data = pd.DataFrame(np.random.rand(100, 3), columns=['col1', 'col2', 'col3'])
    fake_data = pd.DataFrame(np.random.rand(100, 3), columns=['col1', 'col2', 'col3'])
    
    # Test with fname (saving to file)
    fname = tmp_path / "distributions.png"
    plot_distributions(real_data, fake_data, fname=fname)
    assert os.path.exists(fname)

    # Test without fname (displaying plot) - this would open a plot window, but we are testing saving functionality
    # plot_distributions(real_data, fake_data)
