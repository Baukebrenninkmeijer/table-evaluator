"""Backend type definitions for table-evaluator."""

from enum import Enum
from typing import Union

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False


class BackendType(Enum):
    """Enumeration of supported DataFrame backends."""

    PANDAS = "pandas"
    POLARS = "polars"
    AUTO = "auto"

    @classmethod
    def from_dataframe(
        cls, df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"]
    ) -> "BackendType":
        """Detect backend type from DataFrame instance.

        Args:
            df: DataFrame instance to analyze

        Returns:
            BackendType corresponding to the DataFrame type

        Raises:
            ValueError: If DataFrame type is not supported
        """
        if isinstance(df, pd.DataFrame):
            return cls.PANDAS
        elif POLARS_AVAILABLE and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            return cls.POLARS
        else:
            raise ValueError(f"Unsupported DataFrame type: {type(df)}")

    @classmethod
    def get_optimal_backend(
        cls, data_size: int, file_format: str = "csv", prefer_lazy: bool = False
    ) -> "BackendType":
        """Get optimal backend based on data characteristics.

        Args:
            data_size: Number of rows in the dataset
            file_format: File format (csv, parquet, feather)
            prefer_lazy: Whether to prefer lazy evaluation

        Returns:
            Recommended BackendType for the given characteristics
        """
        # Use Polars for large datasets or when lazy evaluation is preferred
        if POLARS_AVAILABLE and (
            data_size > 100_000 or prefer_lazy or file_format in ["parquet", "feather"]
        ):
            return cls.POLARS

        # Default to pandas for smaller datasets or when Polars is not available
        return cls.PANDAS
