"""Polars backend implementation for table-evaluator."""

import logging
from os import PathLike
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from .base_backend import BaseBackend

logger = logging.getLogger(__name__)

if TYPE_CHECKING and POLARS_AVAILABLE:
    PolarsDataFrame = Union[pl.DataFrame, pl.LazyFrame]
    PolarsExpr = pl.Expr
    PolarsDataFrameConcrete = pl.DataFrame
else:
    PolarsDataFrame = Any
    PolarsExpr = Any
    PolarsDataFrameConcrete = Any


class PolarsBackend(BaseBackend):
    """Polars implementation of the BaseBackend protocol.

    This backend provides Polars operations with lazy evaluation capabilities
    while maintaining API compatibility with the pandas backend.
    """

    def __init__(self, lazy: bool = True):
        """Initialize Polars backend.

        Args:
            lazy: Whether to use lazy evaluation by default
        """
        if not POLARS_AVAILABLE:
            raise ImportError(
                "Polars is not available. Install with: pip install polars"
            )
        self.lazy = lazy

    def _ensure_polars_available(self):
        """Ensure polars is available for operations."""
        if not POLARS_AVAILABLE or pl is None:
            raise ImportError(
                "Polars backend operation called but polars is not installed. "
                "Install with: pip install polars"
            )

    def load_csv(
        self, path: Union[str, PathLike], sep: str = ",", **kwargs: Any
    ) -> PolarsDataFrame:
        """Load CSV file using polars.read_csv or scan_csv."""
        self._ensure_polars_available()

        # Remove backend-specific parameters that polars doesn't support
        backend_params = {"lazy"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in backend_params}
        filtered_kwargs.setdefault("separator", sep)

        if self.lazy:
            return pl.scan_csv(path, **filtered_kwargs)
        else:
            return pl.read_csv(path, **filtered_kwargs)

    def load_parquet(
        self, path: Union[str, PathLike], **kwargs: Any
    ) -> PolarsDataFrame:
        """Load Parquet file using polars.read_parquet or scan_parquet."""
        if self.lazy:
            return pl.scan_parquet(path, **kwargs)
        else:
            return pl.read_parquet(path, **kwargs)

    def save_csv(
        self,
        df: PolarsDataFrame,
        path: Union[str, PathLike],
        sep: str = ",",
        **kwargs: Any,
    ) -> None:
        """Save DataFrame to CSV using polars write_csv."""
        kwargs.setdefault("separator", sep)

        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        df.write_csv(path, **kwargs)

    def save_parquet(
        self, df: PolarsDataFrame, path: Union[str, PathLike], **kwargs: Any
    ) -> None:
        """Save DataFrame to Parquet using polars write_parquet."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        df.write_parquet(path, **kwargs)

    def select_columns(
        self, df: PolarsDataFrame, columns: List[str]
    ) -> PolarsDataFrame:
        """Select specific columns from Polars DataFrame."""
        return df.select(columns)

    def filter_rows(
        self, df: PolarsDataFrame, condition: Union[str, PolarsExpr]
    ) -> PolarsDataFrame:
        """Filter DataFrame rows using Polars expressions."""
        if isinstance(condition, str):
            # Convert string condition to Polars expression
            # This is a simplified implementation; more complex parsing may be needed
            return df.filter(pl.expr(condition))
        else:
            return df.filter(condition)

    def sample(
        self, df: PolarsDataFrame, n: int, random_state: Optional[int] = None
    ) -> PolarsDataFrame:
        """Sample n rows using Polars sample."""
        kwargs = {"n": n}
        if random_state is not None:
            kwargs["seed"] = random_state

        # LazyFrame doesn't have sample method, need to collect first
        if isinstance(df, pl.LazyFrame):
            return df.collect().sample(**kwargs)
        else:
            return df.sample(**kwargs)

    def fillna(
        self, df: PolarsDataFrame, value: Union[str, float, Dict[str, Any]]
    ) -> PolarsDataFrame:
        """Fill missing values using Polars fill_null."""
        if isinstance(value, dict):
            # Fill different columns with different values
            result_df = df
            for col, val in value.items():
                result_df = result_df.with_columns(pl.col(col).fill_null(val))
            return result_df
        else:
            # Fill all columns with same value
            return df.fill_null(value)

    def get_dtypes(self, df: PolarsDataFrame) -> Dict[str, str]:
        """Get data types of DataFrame columns."""
        if isinstance(df, pl.LazyFrame):
            schema = df.schema
        else:
            schema = df.schema

        return {col: str(dtype) for col, dtype in schema.items()}

    def get_columns(self, df: PolarsDataFrame) -> List[str]:
        """Get column names from DataFrame."""
        return df.columns

    def get_shape(self, df: PolarsDataFrame) -> Tuple[int, int]:
        """Get shape (rows, columns) of DataFrame."""
        if isinstance(df, pl.LazyFrame):
            # For LazyFrame, we need to collect to get row count
            collected = df.collect()
            return (collected.height, collected.width)
        else:
            return (df.height, df.width)

    def compute_statistics(
        self, df: PolarsDataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute basic statistics for numerical columns."""
        if columns is None:
            # Select only numerical columns
            numerical_cols = [
                col
                for col, dtype in self.get_dtypes(df).items()
                if dtype in ["Int64", "Int32", "Float64", "Float32"]
            ]
        else:
            # Filter to specified columns that are numerical
            dtypes = self.get_dtypes(df)
            numerical_cols = [
                col
                for col in columns
                if dtypes.get(col, "") in ["Int64", "Int32", "Float64", "Float32"]
            ]

        if not numerical_cols:
            return {}

        # Compute statistics using Polars
        stats_exprs = []
        for col in numerical_cols:
            stats_exprs.extend(
                [
                    pl.col(col).mean().alias(f"{col}_mean"),
                    pl.col(col).median().alias(f"{col}_median"),
                    pl.col(col).std().alias(f"{col}_std"),
                    pl.col(col).var().alias(f"{col}_var"),
                    pl.col(col).min().alias(f"{col}_min"),
                    pl.col(col).max().alias(f"{col}_max"),
                    pl.col(col).count().alias(f"{col}_count"),
                    pl.col(col).null_count().alias(f"{col}_null_count"),
                ]
            )

        stats_df = df.select(stats_exprs)
        if isinstance(stats_df, pl.LazyFrame):
            stats_df = stats_df.collect()

        # Convert to dictionary format
        stats_dict = {}
        stats_row = stats_df.row(0)
        stats_names = stats_df.columns

        for col in numerical_cols:
            stats_dict[col] = {
                "mean": stats_row[stats_names.index(f"{col}_mean")],
                "median": stats_row[stats_names.index(f"{col}_median")],
                "std": stats_row[stats_names.index(f"{col}_std")],
                "var": stats_row[stats_names.index(f"{col}_var")],
                "min": stats_row[stats_names.index(f"{col}_min")],
                "max": stats_row[stats_names.index(f"{col}_max")],
                "count": stats_row[stats_names.index(f"{col}_count")],
                "null_count": stats_row[stats_names.index(f"{col}_null_count")],
            }

        return stats_dict

    def get_unique_values(self, df: PolarsDataFrame, column: str) -> List[Any]:
        """Get unique values from a column."""
        unique_df = df.select(pl.col(column).unique())
        if isinstance(unique_df, pl.LazyFrame):
            unique_df = unique_df.collect()

        return unique_df[column].to_list()

    def factorize_column(
        self, df: PolarsDataFrame, column: str
    ) -> Tuple[PolarsDataFrame, Dict[int, Any]]:
        """Convert categorical column to numerical codes."""
        # Get unique values to create mapping
        unique_values = self.get_unique_values(df, column)
        unique_values.sort()  # Sort for consistent ordering

        # Create mapping from codes to original values
        mapping = {i: val for i, val in enumerate(unique_values)}

        # Create reverse mapping for replacement
        reverse_mapping = {val: i for i, val in mapping.items()}

        # Replace values with codes
        df_factorized = df.with_columns(
            pl.col(column).map_elements(
                lambda x: reverse_mapping.get(x, -1), return_dtype=pl.Int64
            )
        )

        return df_factorized, mapping

    def to_pandas(self, df: PolarsDataFrame) -> pd.DataFrame:
        """Convert Polars DataFrame to pandas DataFrame."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        return df.to_pandas()

    def from_pandas(self, df: pd.DataFrame) -> PolarsDataFrame:
        """Create Polars DataFrame from pandas DataFrame."""
        polars_df = pl.from_pandas(df)

        if self.lazy:
            return polars_df.lazy()
        else:
            return polars_df

    def infer_object_types(self, df: PolarsDataFrame) -> PolarsDataFrame:
        """Infer better dtypes for object columns (Polars handles this automatically)."""
        # Polars automatically infers types during loading
        return df

    def get_numerical_columns(
        self, df: PolarsDataFrame, unique_thresh: int = 0
    ) -> List[str]:
        """Get numerical columns with unique values above threshold."""
        dtypes = self.get_dtypes(df)
        numerical_cols = []

        for col, dtype in dtypes.items():
            if dtype in ["Int64", "Int32", "Float64", "Float32"]:
                # Check unique count if threshold is specified
                if unique_thresh > 0:
                    unique_count_df = df.select(pl.col(col).n_unique())
                    if isinstance(unique_count_df, pl.LazyFrame):
                        unique_count_df = unique_count_df.collect()
                    unique_count = unique_count_df.item()

                    if unique_count > unique_thresh:
                        numerical_cols.append(col)
                else:
                    numerical_cols.append(col)

        return numerical_cols

    def get_categorical_columns(
        self, df: PolarsDataFrame, numerical_columns: List[str]
    ) -> List[str]:
        """Get categorical columns (all non-numerical columns)."""
        all_columns = self.get_columns(df)
        return [col for col in all_columns if col not in numerical_columns]

    def collect(self, df: PolarsDataFrame) -> PolarsDataFrameConcrete:
        """Collect LazyFrame into DataFrame."""
        if isinstance(df, pl.LazyFrame):
            return df.collect()
        else:
            return df
