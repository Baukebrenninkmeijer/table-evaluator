"""Base backend protocol for table-evaluator."""

from abc import abstractmethod
from os import PathLike
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import pandas as pd

try:
    import polars as pl

    PolarsDataFrame = Union[pl.DataFrame, pl.LazyFrame]
except ImportError:
    PolarsDataFrame = None

# Type aliases
DataFrameType = Union[pd.DataFrame, PolarsDataFrame]
SeriesType = Union[pd.Series, "pl.Series"]


class BaseBackend(Protocol):
    """Protocol defining the interface for DataFrame backends.

    This protocol ensures all backend implementations provide consistent
    APIs for common DataFrame operations used throughout table-evaluator.
    """

    @abstractmethod
    def load_csv(
        self, path: Union[str, PathLike], sep: str = ",", **kwargs: Any
    ) -> DataFrameType:
        """Load CSV file into DataFrame.

        Args:
            path: Path to CSV file
            sep: Column separator
            **kwargs: Additional arguments for the backend's CSV reader

        Returns:
            Loaded DataFrame
        """
        ...

    @abstractmethod
    def load_parquet(self, path: Union[str, PathLike], **kwargs: Any) -> DataFrameType:
        """Load Parquet file into DataFrame.

        Args:
            path: Path to Parquet file
            **kwargs: Additional arguments for the backend's Parquet reader

        Returns:
            Loaded DataFrame
        """
        ...

    @abstractmethod
    def save_csv(
        self,
        df: DataFrameType,
        path: Union[str, PathLike],
        sep: str = ",",
        **kwargs: Any,
    ) -> None:
        """Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            path: Output path
            sep: Column separator
            **kwargs: Additional arguments for the backend's CSV writer
        """
        ...

    @abstractmethod
    def save_parquet(
        self, df: DataFrameType, path: Union[str, PathLike], **kwargs: Any
    ) -> None:
        """Save DataFrame to Parquet file.

        Args:
            df: DataFrame to save
            path: Output path
            **kwargs: Additional arguments for the backend's Parquet writer
        """
        ...

    @abstractmethod
    def select_columns(self, df: DataFrameType, columns: List[str]) -> DataFrameType:
        """Select specific columns from DataFrame.

        Args:
            df: Input DataFrame
            columns: List of column names to select

        Returns:
            DataFrame with selected columns
        """
        ...

    @abstractmethod
    def filter_rows(
        self, df: DataFrameType, condition: Union[str, Any]
    ) -> DataFrameType:
        """Filter DataFrame rows based on condition.

        Args:
            df: Input DataFrame
            condition: Filter condition (implementation-specific)

        Returns:
            Filtered DataFrame
        """
        ...

    @abstractmethod
    def sample(
        self, df: DataFrameType, n: int, random_state: Optional[int] = None
    ) -> DataFrameType:
        """Sample n rows from DataFrame.

        Args:
            df: Input DataFrame
            n: Number of rows to sample
            random_state: Random seed for reproducibility

        Returns:
            Sampled DataFrame
        """
        ...

    @abstractmethod
    def fillna(
        self, df: DataFrameType, value: Union[str, float, Dict[str, Any]]
    ) -> DataFrameType:
        """Fill missing values in DataFrame.

        Args:
            df: Input DataFrame
            value: Value(s) to use for filling NAs

        Returns:
            DataFrame with filled values
        """
        ...

    @abstractmethod
    def get_dtypes(self, df: DataFrameType) -> Dict[str, str]:
        """Get data types of DataFrame columns.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping column names to string representations of dtypes
        """
        ...

    @abstractmethod
    def get_columns(self, df: DataFrameType) -> List[str]:
        """Get column names from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of column names
        """
        ...

    @abstractmethod
    def get_shape(self, df: DataFrameType) -> Tuple[int, int]:
        """Get shape (rows, columns) of DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (n_rows, n_columns)
        """
        ...

    @abstractmethod
    def compute_statistics(
        self, df: DataFrameType, columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute basic statistics for numerical columns.

        Args:
            df: Input DataFrame
            columns: Specific columns to analyze (None for all numerical)

        Returns:
            Dictionary mapping column names to statistics dictionaries
        """
        ...

    @abstractmethod
    def get_unique_values(self, df: DataFrameType, column: str) -> List[Any]:
        """Get unique values from a column.

        Args:
            df: Input DataFrame
            column: Column name

        Returns:
            List of unique values
        """
        ...

    @abstractmethod
    def factorize_column(
        self, df: DataFrameType, column: str
    ) -> Tuple[DataFrameType, Dict[int, Any]]:
        """Convert categorical column to numerical codes.

        Args:
            df: Input DataFrame
            column: Column name to factorize

        Returns:
            Tuple of (DataFrame with factorized column, mapping dict)
        """
        ...

    @abstractmethod
    def to_pandas(self, df: DataFrameType) -> pd.DataFrame:
        """Convert DataFrame to pandas DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            pandas DataFrame
        """
        ...

    @abstractmethod
    def from_pandas(self, df: pd.DataFrame) -> DataFrameType:
        """Create backend DataFrame from pandas DataFrame.

        Args:
            df: pandas DataFrame

        Returns:
            Backend-specific DataFrame
        """
        ...
