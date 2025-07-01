"""Lazy conversion strategy to minimize memory usage and conversion overhead."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import weakref
from functools import wraps

import pandas as pd

try:
    import polars as pl
    from polars import DataFrame as PolarsDataFrame, LazyFrame as PolarsLazyFrame

    POLARS_AVAILABLE = True
    PolarsReturnType = Union[PolarsDataFrame, PolarsLazyFrame]
except ImportError:
    pl = None
    PolarsDataFrame = None
    PolarsLazyFrame = None
    POLARS_AVAILABLE = False
    PolarsReturnType = Any

from .conversion_bridge import DataFrameConverter

logger = logging.getLogger(__name__)


class LazyConversionCache:
    """Cache for lazy conversions with automatic cleanup.

    This cache uses weak references to automatically clean up
    cached conversions when the original DataFrames are garbage collected.
    """

    def __init__(self, max_size: int = 100):
        """Initialize lazy conversion cache.

        Args:
            max_size: Maximum number of cached conversions
        """
        self.max_size = max_size
        self._cache: Dict[int, Tuple[weakref.ref, Any]] = {}
        self._access_order: List[int] = []

    def get(self, df_id: int) -> Optional[Any]:
        """Get cached conversion result.

        Args:
            df_id: ID of the source DataFrame

        Returns:
            Cached result or None if not found
        """
        if df_id in self._cache:
            weak_ref, result = self._cache[df_id]

            # Check if original DataFrame still exists
            if weak_ref() is not None:
                # Move to end of access order (most recently used)
                if df_id in self._access_order:
                    self._access_order.remove(df_id)
                self._access_order.append(df_id)

                return result
            else:
                # Original DataFrame was garbage collected, remove from cache
                del self._cache[df_id]
                if df_id in self._access_order:
                    self._access_order.remove(df_id)

        return None

    def put(self, df: Any, result: Any) -> None:
        """Cache conversion result.

        Args:
            df: Source DataFrame
            result: Conversion result to cache
        """
        df_id = id(df)

        # Create weak reference to source DataFrame
        weak_ref = weakref.ref(df)

        # Add to cache
        self._cache[df_id] = (weak_ref, result)
        self._access_order.append(df_id)

        # Enforce size limit using LRU eviction
        while len(self._cache) > self.max_size:
            oldest_id = self._access_order.pop(0)
            if oldest_id in self._cache:
                del self._cache[oldest_id]

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def cleanup(self) -> int:
        """Remove entries with dead weak references.

        Returns:
            Number of entries removed
        """
        dead_refs = []
        for df_id, (weak_ref, _) in self._cache.items():
            if weak_ref() is None:
                dead_refs.append(df_id)

        for df_id in dead_refs:
            del self._cache[df_id]
            if df_id in self._access_order:
                self._access_order.remove(df_id)

        return len(dead_refs)


class LazyConverter:
    """Lazy converter that defers conversions until absolutely necessary.

    This class provides lazy evaluation for DataFrame conversions,
    only performing expensive operations when the converted data is actually needed.
    """

    def __init__(self, cache_size: int = 50):
        """Initialize lazy converter.

        Args:
            cache_size: Maximum number of cached conversions
        """
        self.converter = DataFrameConverter()
        self.cache = LazyConversionCache(cache_size)
        self._conversion_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "conversions_performed": 0,
            "memory_saved_conversions": 0,
        }

    def create_lazy_pandas_wrapper(
        self, polars_df: Union["pl.DataFrame", "pl.LazyFrame"]
    ) -> "LazyPandasWrapper":
        """Create a lazy wrapper for Polars->Pandas conversion.

        Args:
            polars_df: Source Polars DataFrame

        Returns:
            Lazy wrapper that converts to pandas when accessed
        """
        return LazyPandasWrapper(polars_df, self)

    def create_lazy_polars_wrapper(
        self, pandas_df: pd.DataFrame, lazy: bool = True
    ) -> "LazyPolarsWrapper":
        """Create a lazy wrapper for Pandas->Polars conversion.

        Args:
            pandas_df: Source pandas DataFrame
            lazy: Whether to create LazyFrame

        Returns:
            Lazy wrapper that converts to Polars when accessed
        """
        return LazyPolarsWrapper(pandas_df, self, lazy=lazy)

    def convert_when_needed(
        self,
        source_df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        target_backend: str,
        **kwargs,
    ) -> Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"]:
        """Convert DataFrame only when needed, with caching.

        Args:
            source_df: Source DataFrame
            target_backend: Target backend ("pandas" or "polars")
            **kwargs: Additional conversion arguments

        Returns:
            Converted DataFrame
        """
        df_id = id(source_df)

        # Check cache first
        cached_result = self.cache.get(df_id)
        if cached_result is not None:
            self._conversion_stats["cache_hits"] += 1
            logger.debug(f"Cache hit for conversion to {target_backend}")
            return cached_result

        # Cache miss - perform conversion
        self._conversion_stats["cache_misses"] += 1
        self._conversion_stats["conversions_performed"] += 1

        logger.debug(f"Converting to {target_backend} (cache miss)")

        try:
            if target_backend == "pandas":
                if isinstance(source_df, pd.DataFrame):
                    result = source_df.copy()
                else:
                    result = self.converter.polars_to_pandas(source_df, **kwargs)

            elif target_backend == "polars":
                if POLARS_AVAILABLE and isinstance(
                    source_df, (pl.DataFrame, pl.LazyFrame)
                ):
                    result = source_df.clone()
                else:
                    result = self.converter.pandas_to_polars(source_df, **kwargs)

            else:
                raise ValueError(f"Unknown target backend: {target_backend}")

            # Cache the result
            self.cache.put(source_df, result)

            return result

        except Exception as e:
            logger.error(f"Conversion to {target_backend} failed: {e}")
            raise

    def get_conversion_stats(self) -> Dict[str, int]:
        """Get conversion statistics.

        Returns:
            Dictionary with conversion performance statistics
        """
        stats = self._conversion_stats.copy()
        stats["cache_size"] = self.cache.size()

        # Calculate hit rate
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_requests
        else:
            stats["cache_hit_rate"] = 0.0

        return stats

    def clear_cache(self) -> None:
        """Clear conversion cache."""
        self.cache.clear()
        logger.info("Cleared lazy conversion cache")

    def cleanup_cache(self) -> int:
        """Clean up dead references in cache.

        Returns:
            Number of entries removed
        """
        removed = self.cache.cleanup()
        if removed > 0:
            logger.debug(f"Cleaned up {removed} dead cache entries")
        return removed


class LazyPandasWrapper:
    """Lazy wrapper for Polars->Pandas conversion.

    This wrapper looks like a pandas DataFrame but only performs
    the actual conversion when pandas-specific methods are called.
    """

    def __init__(
        self, polars_df: Union["pl.DataFrame", "pl.LazyFrame"], converter: LazyConverter
    ):
        """Initialize lazy pandas wrapper.

        Args:
            polars_df: Source Polars DataFrame
            converter: Lazy converter instance
        """
        self._polars_df = polars_df
        self._converter = converter
        self._pandas_df: Optional[pd.DataFrame] = None
        self._converted = False

    def _ensure_converted(self) -> pd.DataFrame:
        """Ensure conversion to pandas has been performed."""
        if not self._converted:
            self._pandas_df = self._converter.convert_when_needed(
                self._polars_df, "pandas"
            )
            self._converted = True
            self._converter._conversion_stats["memory_saved_conversions"] += 1

        return self._pandas_df

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to pandas DataFrame after conversion."""
        pandas_df = self._ensure_converted()
        return getattr(pandas_df, name)

    def __getitem__(self, key: Any) -> Any:
        """Delegate item access to pandas DataFrame after conversion."""
        pandas_df = self._ensure_converted()
        return pandas_df[key]

    def __len__(self) -> int:
        """Get length without full conversion if possible."""
        if self._converted:
            return len(self._pandas_df)
        else:
            # Try to get length from Polars without full conversion
            if isinstance(self._polars_df, pl.LazyFrame):
                return len(self._polars_df.collect())
            else:
                return len(self._polars_df)

    def __str__(self) -> str:
        """String representation."""
        if self._converted:
            return str(self._pandas_df)
        else:
            return f"LazyPandasWrapper({type(self._polars_df).__name__})"

    def __repr__(self) -> str:
        """Detailed representation."""
        if self._converted:
            return repr(self._pandas_df)
        else:
            return f"LazyPandasWrapper(source={repr(self._polars_df)}, converted=False)"


class LazyPolarsWrapper:
    """Lazy wrapper for Pandas->Polars conversion.

    This wrapper looks like a Polars DataFrame but only performs
    the actual conversion when Polars-specific methods are called.
    """

    def __init__(
        self, pandas_df: pd.DataFrame, converter: LazyConverter, lazy: bool = True
    ):
        """Initialize lazy Polars wrapper.

        Args:
            pandas_df: Source pandas DataFrame
            converter: Lazy converter instance
            lazy: Whether to create LazyFrame
        """
        self._pandas_df = pandas_df
        self._converter = converter
        self._lazy = lazy
        self._polars_df: Optional[PolarsReturnType] = None
        self._converted = False

    def _ensure_converted(self) -> PolarsReturnType:
        """Ensure conversion to Polars has been performed."""
        if not self._converted:
            self._polars_df = self._converter.convert_when_needed(
                self._pandas_df, "polars", lazy=self._lazy
            )
            self._converted = True
            self._converter._conversion_stats["memory_saved_conversions"] += 1

        return self._polars_df

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to Polars DataFrame after conversion."""
        polars_df = self._ensure_converted()
        return getattr(polars_df, name)

    def __getitem__(self, key: Any) -> Any:
        """Delegate item access to Polars DataFrame after conversion."""
        polars_df = self._ensure_converted()
        return polars_df[key]

    def __len__(self) -> int:
        """Get length without full conversion if possible."""
        if self._converted:
            if isinstance(self._polars_df, pl.LazyFrame):
                return len(self._polars_df.collect())
            else:
                return len(self._polars_df)
        else:
            return len(self._pandas_df)

    def __str__(self) -> str:
        """String representation."""
        if self._converted:
            return str(self._polars_df)
        else:
            return f"LazyPolarsWrapper({type(self._pandas_df).__name__})"

    def __repr__(self) -> str:
        """Detailed representation."""
        if self._converted:
            return repr(self._polars_df)
        else:
            return f"LazyPolarsWrapper(source={repr(self._pandas_df)}, converted=False, lazy={self._lazy})"


def lazy_conversion(cache_size: int = 50):
    """Decorator for methods that should use lazy conversion.

    Args:
        cache_size: Size of conversion cache

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        converter = LazyConverter(cache_size)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inject lazy converter into kwargs
            kwargs["_lazy_converter"] = converter
            return func(*args, **kwargs)

        # Attach converter to function for external access
        wrapper._lazy_converter = converter

        return wrapper

    return decorator


def optimize_for_memory(
    df: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
    target_operations: List[str],
) -> Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"]:
    """Optimize DataFrame representation for memory efficiency.

    Args:
        df: Source DataFrame
        target_operations: List of intended operations

    Returns:
        Optimized DataFrame representation
    """
    if isinstance(df, pd.DataFrame):
        # Optimize pandas DataFrame
        optimized = df.copy()

        # Downcast numeric types
        for col in optimized.select_dtypes(include=["int64"]).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast="integer")

        for col in optimized.select_dtypes(include=["float64"]).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast="float")

        # Convert to categorical if beneficial
        for col in optimized.select_dtypes(include=["object"]).columns:
            unique_ratio = optimized[col].nunique() / len(optimized)
            if unique_ratio < 0.5:  # Less than 50% unique values
                optimized[col] = optimized[col].astype("category")

        return optimized

    elif POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        # For Polars DataFrame, convert to LazyFrame for memory efficiency
        if "aggregation" in target_operations or "filter" in target_operations:
            return df.lazy()
        else:
            return df

    elif POLARS_AVAILABLE and isinstance(df, pl.LazyFrame):
        # Already lazy, return as-is
        return df

    else:
        return df
