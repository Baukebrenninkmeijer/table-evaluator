"""Layout utilities for plotting functions."""

from typing import Tuple

import pandas as pd


def calculate_subplot_layout(n_items: int, n_cols: int = 3) -> Tuple[int, int]:
    """
    Calculate subplot grid dimensions.

    Args:
        n_items: Number of items to plot
        n_cols: Number of columns in the grid

    Returns:
        Tuple[int, int]: (n_rows, n_cols) for the subplot grid
    """
    n_rows = max(1, n_items // n_cols)
    n_rows = n_rows + 1 if n_items % n_cols != 0 else n_rows
    return n_rows, n_cols


def calculate_label_based_height(dataframe: pd.DataFrame, base_height: int = 6) -> int:
    """
    Calculate plot height based on the length of categorical labels.

    Args:
        dataframe: DataFrame to analyze for label lengths
        base_height: Base height for the plot

    Returns:
        int: Calculated height based on label lengths
    """
    max_len = 0

    # Find the maximum label length in categorical columns
    categorical_data = dataframe.select_dtypes(include=["object"])
    if not categorical_data.empty:
        lengths = []
        for col in categorical_data.columns:
            unique_values = categorical_data[col].unique().tolist()
            col_max_len = max([len(str(x).strip()) for x in unique_values])
            lengths.append(col_max_len)
        max_len = max(lengths) if lengths else 0

    return base_height + (max_len // 30)
