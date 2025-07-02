"""Main backend manager for automatic detection and routing."""

import logging
import os
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from .backend_context import BackendContext, GlobalBackendContext
from .backend_factory import BackendFactory
from .backend_types import BackendType
from .dataframe_wrapper import DataFrameWrapper

logger = logging.getLogger(__name__)


class BackendManager:
    """High-level manager for automatic backend detection and routing.

    This class provides the main interface for working with multiple backends,
    handling automatic detection, optimization, and seamless switching between
    pandas and Polars backends.
    """

    def __init__(self, default_backend: Optional[BackendType] = None):
        """Initialize backend manager.

        Args:
            default_backend: Default backend to use (None for AUTO detection)
        """
        self.factory = BackendFactory()
        self.default_backend = default_backend or BackendType.AUTO

        # Performance thresholds for automatic backend selection
        self.large_data_threshold = 100_000  # rows
        self.memory_efficient_formats = {"parquet", "feather", "arrow"}

    def load_data(
        self,
        path: Union[str, PathLike],
        backend: Optional[Union[BackendType, str]] = None,
        auto_detect: bool = True,
        **kwargs,
    ) -> DataFrameWrapper:
        """Load data with automatic backend selection and optimization.

        Args:
            path: Path to data file
            backend: Specific backend to use (None for auto-detection)
            auto_detect: Whether to auto-detect optimal backend
            **kwargs: Additional arguments for the loader

        Returns:
            DataFrameWrapper containing the loaded data

        Example:
            # Auto-detect optimal backend
            data = manager.load_data("data.csv")

            # Force specific backend
            data = manager.load_data("data.parquet", backend=BackendType.POLARS)
        """
        path_obj = Path(path)
        file_format = path_obj.suffix.lower().lstrip(".")

        # Convert string backend to BackendType if needed
        if isinstance(backend, str):
            backend = BackendType.from_string(backend)

        # Determine backend to use
        if backend is None and auto_detect:
            backend = self._detect_optimal_backend_for_file(path_obj, **kwargs)
        elif backend is None:
            backend = self.default_backend

        # Get appropriate backend instance
        backend_instance = self.factory.get_backend(backend)

        # Load data using appropriate method
        if file_format in ["csv", "txt"]:
            df = backend_instance.load_csv(path, **kwargs)
        elif file_format in ["parquet", "pq"]:
            df = backend_instance.load_parquet(path, **kwargs)
        else:
            # Fallback to CSV for unknown formats
            logger.warning(f"Unknown file format '{file_format}', treating as CSV")
            df = backend_instance.load_csv(path, **kwargs)

        logger.info(f"Loaded data from {path} using {backend.value} backend")
        return DataFrameWrapper(df, backend)

    def save_data(
        self,
        wrapper: DataFrameWrapper,
        path: Union[str, PathLike],
        format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Save data using the wrapper's backend.

        Args:
            wrapper: DataFrameWrapper containing data to save
            path: Output path
            format: Output format (auto-detected from path if None)
            **kwargs: Additional arguments for the saver
        """
        path_obj = Path(path)

        if format is None:
            format = path_obj.suffix.lower().lstrip(".")

        # Get the backend instance for this wrapper
        backend_instance = self.factory.get_backend(wrapper.backend_type)

        # Save using appropriate method
        if format in ["csv", "txt"]:
            backend_instance.save_csv(wrapper.underlying_df, path, **kwargs)
        elif format in ["parquet", "pq"]:
            backend_instance.save_parquet(wrapper.underlying_df, path, **kwargs)
        else:
            raise ValueError(f"Unsupported output format: {format}")

        logger.info(f"Saved data to {path} using {wrapper.backend_type.value} backend")

    def convert_backend(
        self, wrapper: DataFrameWrapper, target_backend: BackendType, **kwargs
    ) -> DataFrameWrapper:
        """Convert data between backends.

        Args:
            wrapper: Source DataFrameWrapper
            target_backend: Target backend type
            **kwargs: Additional arguments for backend creation

        Returns:
            New DataFrameWrapper with data in target backend
        """
        if wrapper.backend_type == target_backend:
            return wrapper.copy()

        # Convert via pandas as intermediate format
        if wrapper.backend_type == BackendType.PANDAS:
            # From pandas to Polars
            if target_backend == BackendType.POLARS:
                target_backend_instance = self.factory.get_backend(
                    target_backend, **kwargs
                )
                converted_df = target_backend_instance.from_pandas(
                    wrapper.underlying_df
                )
                return DataFrameWrapper(converted_df, target_backend)

        elif wrapper.backend_type == BackendType.POLARS:
            # From Polars to pandas
            if target_backend == BackendType.PANDAS:
                pandas_df = wrapper.to_pandas()
                return DataFrameWrapper(pandas_df, BackendType.PANDAS)

        raise ValueError(
            f"Cannot convert from {wrapper.backend_type} to {target_backend}"
        )

    def optimize_for_operation(
        self, wrapper: DataFrameWrapper, operation: str, **kwargs
    ) -> DataFrameWrapper:
        """Optimize backend choice for specific operations.

        Args:
            wrapper: Input DataFrameWrapper
            operation: Type of operation ("aggregation", "join", "statistical", etc.)
            **kwargs: Additional context for optimization

        Returns:
            DataFrameWrapper optimized for the operation
        """
        current_backend = wrapper.backend_type
        optimal_backend = self._get_optimal_backend_for_operation(
            wrapper, operation, **kwargs
        )

        if current_backend != optimal_backend:
            logger.info(
                f"Optimizing backend from {current_backend.value} to {optimal_backend.value} for {operation}"
            )
            return self.convert_backend(wrapper, optimal_backend, **kwargs)

        return wrapper

    def _detect_optimal_backend_for_file(self, path: Path, **kwargs) -> BackendType:
        """Detect optimal backend for loading a specific file.

        Args:
            path: Path to the file
            **kwargs: Additional context

        Returns:
            Optimal BackendType for the file
        """
        file_format = path.suffix.lower().lstrip(".")

        # Estimate data size if possible
        try:
            file_size = os.path.getsize(path)
            # Rough heuristic: assume 100 bytes per row for CSV
            estimated_rows = (
                file_size // 100 if file_format == "csv" else file_size // 50
            )
        except OSError:
            estimated_rows = 0

        # Check if lazy evaluation is preferred
        prefer_lazy = kwargs.get("prefer_lazy", False)

        return BackendType.get_optimal_backend(
            data_size=estimated_rows, file_format=file_format, prefer_lazy=prefer_lazy
        )

    def _get_optimal_backend_for_operation(
        self, wrapper: DataFrameWrapper, operation: str, **kwargs
    ) -> BackendType:
        """Get optimal backend for a specific operation.

        Args:
            wrapper: Input data wrapper
            operation: Type of operation
            **kwargs: Additional context

        Returns:
            Optimal BackendType for the operation
        """
        data_size = len(wrapper)

        # Operation-specific heuristics
        if operation in ["aggregation", "groupby", "statistical"]:
            # Polars excels at aggregations on large data
            if POLARS_AVAILABLE and data_size > self.large_data_threshold:
                return BackendType.POLARS

        elif operation in ["join", "merge"]:
            # Polars is very efficient for joins
            if POLARS_AVAILABLE and data_size > 10_000:
                return BackendType.POLARS

        elif operation in ["ml", "sklearn"]:
            # scikit-learn works best with pandas/numpy
            return BackendType.PANDAS

        # Default to current backend if no specific optimization
        return wrapper.backend_type

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends and current configuration.

        Returns:
            Dictionary with backend information
        """
        info = {
            "available_backends": [
                bt.value for bt in self.factory.get_available_backends()
            ],
            "default_backend": self.default_backend.value,
            "polars_available": POLARS_AVAILABLE,
            "current_thread_backend": None,
            "global_backend": None,
        }

        # Thread-local backend
        current_type = BackendContext.get_current_backend_type()
        if current_type:
            info["current_thread_backend"] = current_type.value

        # Global backend
        global_type, global_kwargs = GlobalBackendContext.get_global_backend()
        if global_type:
            info["global_backend"] = {
                "type": global_type.value,
                "kwargs": global_kwargs,
            }

        return info

    def set_performance_thresholds(
        self,
        large_data_threshold: Optional[int] = None,
        memory_efficient_formats: Optional[set] = None,
    ) -> None:
        """Configure performance thresholds for backend selection.

        Args:
            large_data_threshold: Row count threshold for large data
            memory_efficient_formats: Set of formats considered memory efficient
        """
        if large_data_threshold is not None:
            self.large_data_threshold = large_data_threshold

        if memory_efficient_formats is not None:
            self.memory_efficient_formats = memory_efficient_formats

        logger.info(
            f"Updated performance thresholds: large_data={self.large_data_threshold}"
        )

    def create_wrapper(
        self,
        df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        backend_type: Optional[BackendType] = None,
    ) -> DataFrameWrapper:
        """Create DataFrameWrapper from existing DataFrame.

        Args:
            df: Existing DataFrame
            backend_type: Explicit backend type (auto-detected if None)

        Returns:
            DataFrameWrapper for the DataFrame
        """
        return DataFrameWrapper(df, backend_type)
