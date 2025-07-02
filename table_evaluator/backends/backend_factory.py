"""Backend factory for creating and managing backend instances."""

import logging
import threading
from typing import Dict, Optional, Union

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from .backend_types import BackendType
from .base_backend import BaseBackend
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend

logger = logging.getLogger(__name__)


class BackendFactory:
    """Singleton factory for creating and managing backend instances.

    This factory ensures that backend instances are created efficiently
    and provides caching to avoid repeated instantiation overhead.
    """

    _instance: Optional["BackendFactory"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "BackendFactory":
        """Create singleton instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize factory if not already done."""
        if not self._initialized:
            self._backends: Dict[BackendType, BaseBackend] = {}
            self._default_lazy = True
            self._initialized = True

    def get_backend(self, backend_type: BackendType, **kwargs) -> BaseBackend:
        """Get backend instance for the specified type.

        Args:
            backend_type: Type of backend to create
            **kwargs: Additional arguments for backend initialization

        Returns:
            Backend instance

        Raises:
            ValueError: If backend type is not supported
            ImportError: If required dependencies are not available
        """
        # Handle AUTO backend type by detecting best available
        if backend_type == BackendType.AUTO:
            backend_type = self._detect_optimal_backend(**kwargs)

        # Check if backend is already cached
        cache_key = backend_type
        if cache_key in self._backends:
            cached_backend = self._backends[cache_key]

            # For Polars backend, check if lazy setting matches
            if backend_type == BackendType.POLARS and hasattr(cached_backend, "lazy"):
                requested_lazy = kwargs.get("lazy", self._default_lazy)
                if cached_backend.lazy != requested_lazy:
                    # Need to create new instance with different lazy setting
                    pass  # Fall through to creation logic
                else:
                    return cached_backend
            else:
                return cached_backend

        # Create new backend instance
        if backend_type == BackendType.PANDAS:
            backend = PandasBackend()
        elif backend_type == BackendType.POLARS:
            if not POLARS_AVAILABLE:
                raise ImportError(
                    "Polars backend requested but polars is not installed. "
                    "Install with: pip install polars"
                )
            lazy = kwargs.get("lazy", self._default_lazy)
            backend = PolarsBackend(lazy=lazy)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        # Cache the backend instance
        self._backends[cache_key] = backend
        logger.debug(f"Created new {backend_type.value} backend instance")

        return backend

    def _detect_optimal_backend(self, **kwargs) -> BackendType:
        """Detect optimal backend based on context and availability.

        Args:
            **kwargs: Context information (data_size, file_format, etc.)

        Returns:
            Optimal BackendType for the given context
        """
        # If Polars is not available, use pandas
        if not POLARS_AVAILABLE:
            logger.debug("Polars not available, using pandas backend")
            return BackendType.PANDAS

        # Extract context information
        data_size = kwargs.get("data_size", 0)
        file_format = kwargs.get("file_format", "csv")
        prefer_lazy = kwargs.get("prefer_lazy", False)

        # Use heuristics to choose optimal backend
        optimal_type = BackendType.get_optimal_backend(
            data_size=data_size, file_format=file_format, prefer_lazy=prefer_lazy
        )

        logger.debug(
            f"Auto-detected {optimal_type.value} backend "
            f"(size={data_size}, format={file_format}, lazy={prefer_lazy})"
        )

        return optimal_type

    def create_from_dataframe(
        self, df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"]
    ) -> BaseBackend:
        """Create appropriate backend based on DataFrame type.

        Args:
            df: DataFrame instance

        Returns:
            Backend instance matching the DataFrame type
        """
        backend_type = BackendType.from_dataframe(df)

        # For Polars DataFrames, detect if lazy evaluation should be used
        if backend_type == BackendType.POLARS and POLARS_AVAILABLE:
            lazy = isinstance(df, pl.LazyFrame)
            return self.get_backend(backend_type, lazy=lazy)
        else:
            return self.get_backend(backend_type)

    def set_default_lazy(self, lazy: bool) -> None:
        """Set default lazy evaluation setting for Polars backend.

        Args:
            lazy: Whether to use lazy evaluation by default
        """
        self._default_lazy = lazy

        # Clear cached Polars backend to force recreation with new setting
        if BackendType.POLARS in self._backends:
            del self._backends[BackendType.POLARS]

    def clear_cache(self) -> None:
        """Clear all cached backend instances."""
        self._backends.clear()
        logger.debug("Cleared backend cache")

    def get_available_backends(self) -> list[BackendType]:
        """Get list of available backend types.

        Returns:
            List of available BackendType values
        """
        available = [BackendType.PANDAS, BackendType.AUTO]

        if POLARS_AVAILABLE:
            available.append(BackendType.POLARS)

        return available

    def is_backend_available(self, backend_type: BackendType) -> bool:
        """Check if a specific backend is available.

        Args:
            backend_type: Backend type to check

        Returns:
            True if backend is available, False otherwise
        """
        if backend_type in [BackendType.PANDAS, BackendType.AUTO]:
            return True
        elif backend_type == BackendType.POLARS:
            return POLARS_AVAILABLE
        else:
            return False
