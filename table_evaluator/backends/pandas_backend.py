"""Pandas backend implementation for table-evaluator."""

import logging
from os import PathLike
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class PandasBackend(BaseBackend):
    """Pandas implementation of the BaseBackend protocol.

    This backend wraps pandas operations to provide a consistent interface
    while maintaining all existing functionality and performance characteristics.
    """

    def load_csv(
        self, path: Union[str, PathLike], sep: str = ",", **kwargs: Any
    ) -> pd.DataFrame:
        """Load CSV file using pandas.read_csv."""
        # Remove polars-specific parameters that pandas doesn't support
        polars_params = {"lazy"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in polars_params}
        filtered_kwargs.setdefault("low_memory", False)
        return pd.read_csv(path, sep=sep, **filtered_kwargs)

    def load_parquet(self, path: Union[str, PathLike], **kwargs: Any) -> pd.DataFrame:
        """Load Parquet file using pandas.read_parquet."""
        # Remove parameters that pandas parquet reader doesn't support
        unsupported_params = {"lazy", "sep", "delimiter", "header", "skiprows", "nrows"}
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in unsupported_params
        }
        return pd.read_parquet(path, **filtered_kwargs)

    def save_csv(
        self,
        df: pd.DataFrame,
        path: Union[str, PathLike],
        sep: str = ",",
        **kwargs: Any,
    ) -> None:
        """Save DataFrame to CSV using pandas.to_csv."""
        kwargs.setdefault("index", False)
        df.to_csv(path, sep=sep, **kwargs)

    def save_parquet(
        self, df: pd.DataFrame, path: Union[str, PathLike], **kwargs: Any
    ) -> None:
        """Save DataFrame to Parquet using pandas.to_parquet."""
        kwargs.setdefault("index", False)
        df.to_parquet(path, **kwargs)

    def select_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Select specific columns from pandas DataFrame."""
        return df[columns].copy()

    def filter_rows(
        self, df: pd.DataFrame, condition: Union[str, pd.Series]
    ) -> pd.DataFrame:
        """Filter DataFrame rows using pandas query or boolean mask."""
        if isinstance(condition, str):
            return df.query(condition).copy()
        else:
            return df[condition].copy()

    def sample(
        self, df: pd.DataFrame, n: int, random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """Sample n rows using pandas.sample."""
        return df.sample(n=n, random_state=random_state, replace=False).reset_index(
            drop=True
        )

    def fillna(
        self, df: pd.DataFrame, value: Union[str, float, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Fill missing values using pandas.fillna."""
        return df.fillna(value)

    def get_dtypes(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get data types of DataFrame columns."""
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def get_columns(self, df: pd.DataFrame) -> List[str]:
        """Get column names from DataFrame."""
        return [str(col) for col in df.columns]

    def get_shape(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Get shape (rows, columns) of DataFrame."""
        return df.shape

    def compute_statistics(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute basic statistics for numerical columns."""
        if columns is None:
            # Select only numerical columns
            numerical_df = df.select_dtypes(include=[np.number])
        else:
            # Filter to specified columns that are numerical
            numerical_df = df[columns].select_dtypes(include=[np.number])

        stats_dict = {}
        for col in numerical_df.columns:
            series = numerical_df[col]
            stats_dict[col] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "var": float(series.var()),
                "min": float(series.min()),
                "max": float(series.max()),
                "count": int(series.count()),
                "null_count": int(series.isnull().sum()),
            }

        return stats_dict

    def get_unique_values(self, df: pd.DataFrame, column: str) -> List[Any]:
        """Get unique values from a column."""
        return df[column].unique().tolist()

    def factorize_column(
        self, df: pd.DataFrame, column: str
    ) -> Tuple[pd.DataFrame, Dict[int, Any]]:
        """Convert categorical column to numerical codes."""
        df_copy = df.copy()
        codes, uniques = pd.factorize(df_copy[column], sort=True)
        df_copy[column] = codes

        # Create mapping from codes to original values
        mapping = {i: val for i, val in enumerate(uniques)}

        return df_copy, mapping

    def to_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame as-is since it's already pandas."""
        return df.copy()

    def from_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame as-is since target is pandas."""
        return df.copy()

    def infer_object_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer better dtypes for object columns."""
        return df.infer_objects()

    def get_numerical_columns(
        self, df: pd.DataFrame, unique_thresh: int = 0
    ) -> List[str]:
        """Get numerical columns with unique values above threshold."""
        numerical_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if len(df[col].unique()) > unique_thresh:
                numerical_cols.append(col)
        return numerical_cols

    def get_categorical_columns(
        self, df: pd.DataFrame, numerical_columns: List[str]
    ) -> List[str]:
        """Get categorical columns (all non-numerical columns)."""
        return [col for col in df.columns if col not in numerical_columns]
