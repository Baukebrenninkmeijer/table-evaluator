"""Unified DataFrame wrapper for table-evaluator."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    import polars as pl
    from polars import DataFrame as PolarsDataFrame, LazyFrame as PolarsLazyFrame

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    PolarsDataFrame = None
    PolarsLazyFrame = None
    POLARS_AVAILABLE = False

from .backend_types import BackendType
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend

logger = logging.getLogger(__name__)

if POLARS_AVAILABLE:
    DataFrameType = Union[pd.DataFrame, PolarsDataFrame, PolarsLazyFrame]
    PolarsReturnType = Union[PolarsDataFrame, PolarsLazyFrame]
else:
    DataFrameType = pd.DataFrame
    PolarsReturnType = Any


class DataFrameWrapper:
    """Unified wrapper providing consistent interface over different DataFrame backends.

    This class abstracts the differences between pandas and Polars DataFrames,
    allowing the rest of the codebase to work with a consistent API regardless
    of the underlying backend.
    """

    def __init__(self, df: DataFrameType, backend_type: Optional[BackendType] = None):
        """Initialize DataFrame wrapper.

        Args:
            df: The underlying DataFrame (pandas or Polars)
            backend_type: Explicit backend type (auto-detected if None)
        """
        self._df = df

        if backend_type is None:
            self._backend_type = BackendType.from_dataframe(df)
        else:
            self._backend_type = backend_type

        # Initialize appropriate backend
        if self._backend_type == BackendType.PANDAS:
            self._backend = PandasBackend()
        elif self._backend_type == BackendType.POLARS:
            self._backend = PolarsBackend(
                lazy=isinstance(df, pl.LazyFrame) if POLARS_AVAILABLE else False
            )
        else:
            raise ValueError(f"Unsupported backend type: {self._backend_type}")

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return self._backend_type

    @property
    def is_pandas(self) -> bool:
        """Check if underlying DataFrame is pandas."""
        return self._backend_type == BackendType.PANDAS

    @property
    def is_polars(self) -> bool:
        """Check if underlying DataFrame is Polars."""
        return self._backend_type == BackendType.POLARS

    @property
    def is_lazy(self) -> bool:
        """Check if underlying DataFrame is lazy (Polars LazyFrame)."""
        return POLARS_AVAILABLE and isinstance(self._df, pl.LazyFrame)

    @property
    def underlying_df(self) -> DataFrameType:
        """Get the underlying DataFrame object."""
        return self._df

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape (rows, columns) of DataFrame."""
        return self._backend.get_shape(self._df)

    @property
    def columns(self) -> List[str]:
        """Get column names."""
        return self._backend.get_columns(self._df)

    @property
    def dtypes(self) -> Dict[str, str]:
        """Get data types of columns."""
        return self._backend.get_dtypes(self._df)

    def select(self, columns: List[str]) -> "DataFrameWrapper":
        """Select specific columns."""
        selected_df = self._backend.select_columns(self._df, columns)
        return DataFrameWrapper(selected_df, self._backend_type)

    def filter(self, condition: Union[str, Any]) -> "DataFrameWrapper":
        """Filter rows based on condition."""
        filtered_df = self._backend.filter_rows(self._df, condition)
        return DataFrameWrapper(filtered_df, self._backend_type)

    def sample(self, n: int, random_state: Optional[int] = None) -> "DataFrameWrapper":
        """Sample n rows."""
        sampled_df = self._backend.sample(self._df, n, random_state)
        return DataFrameWrapper(sampled_df, self._backend_type)

    def fillna(self, value: Union[str, float, Dict[str, Any]]) -> "DataFrameWrapper":
        """Fill missing values."""
        filled_df = self._backend.fillna(self._df, value)
        return DataFrameWrapper(filled_df, self._backend_type)

    def get_statistics(
        self, columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute basic statistics."""
        return self._backend.compute_statistics(self._df, columns)

    def get_unique_values(self, column: str) -> List[Any]:
        """Get unique values from column."""
        return self._backend.get_unique_values(self._df, column)

    def factorize_column(
        self, column: str
    ) -> Tuple["DataFrameWrapper", Dict[int, Any]]:
        """Factorize categorical column."""
        factorized_df, mapping = self._backend.factorize_column(self._df, column)
        return DataFrameWrapper(factorized_df, self._backend_type), mapping

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return self._backend.to_pandas(self._df)

    def to_polars(self, lazy: bool = False) -> PolarsReturnType:
        """Convert to Polars DataFrame.

        Args:
            lazy: Whether to return LazyFrame

        Returns:
            Polars DataFrame or LazyFrame

        Raises:
            ImportError: If Polars is not available
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not available")

        if self.is_polars:
            # Already Polars, just convert lazy/eager as needed
            if lazy and not self.is_lazy:
                return self._df.lazy()
            elif not lazy and self.is_lazy:
                return self._df.collect()
            else:
                return self._df
        else:
            # Convert from pandas
            polars_backend = PolarsBackend(lazy=lazy)
            return polars_backend.from_pandas(self._df)

    def collect(self) -> "DataFrameWrapper":
        """Collect lazy DataFrame (no-op for eager DataFrames)."""
        if self.is_lazy:
            collected_df = self._df.collect()
            return DataFrameWrapper(collected_df, self._backend_type)
        else:
            return self

    def copy(self) -> "DataFrameWrapper":
        """Create a copy of the wrapper and underlying DataFrame."""
        if self.is_pandas:
            copied_df = self._df.copy()
        else:
            # For Polars, clone the DataFrame
            copied_df = self._df.clone()

        return DataFrameWrapper(copied_df, self._backend_type)

    def __len__(self) -> int:
        """Get number of rows."""
        return self.shape[0]

    def __str__(self) -> str:
        """String representation."""
        return f"DataFrameWrapper({self._backend_type.value}, shape={self.shape})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"DataFrameWrapper(\n"
            f"  backend={self._backend_type.value},\n"
            f"  shape={self.shape},\n"
            f"  columns={self.columns}\n"
            f")"
        )

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "DataFrameWrapper":
        """Create wrapper from pandas DataFrame."""
        return cls(df, BackendType.PANDAS)

    @classmethod
    def from_polars(cls, df: PolarsReturnType) -> "DataFrameWrapper":
        """Create wrapper from Polars DataFrame."""
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not available")
        return cls(df, BackendType.POLARS)
