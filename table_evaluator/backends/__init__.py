"""Backend abstraction layer for table-evaluator.

This module provides a unified interface for working with different DataFrame backends
(pandas, Polars) while maintaining compatibility and optimal performance.
"""

from .base_backend import BaseBackend
from .backend_context import BackendContext, GlobalBackendContext
from .backend_factory import BackendFactory
from .backend_manager import BackendManager
from .backend_types import BackendType, POLARS_AVAILABLE
from .conversion_bridge import DataFrameConverter, ConversionError
from .data_bridge import DataBridge, convert_to_pandas, convert_to_polars
from .dataframe_wrapper import DataFrameWrapper
from .dtype_bridge import DTypeMapper, DTypeMappingError
from .feature_flags import (
    FeatureFlags,
    get_feature_flags,
    is_feature_enabled,
    is_backend_available,
    get_available_backends,
)
from .lazy_conversion import LazyConverter, optimize_for_memory
from .pandas_backend import PandasBackend
from .schema_validator import SchemaValidator, SchemaIssue

try:
    from .polars_backend import PolarsBackend
except ImportError:
    PolarsBackend = None

__all__ = [
    "BaseBackend",
    "BackendContext",
    "BackendFactory",
    "BackendManager",
    "BackendType",
    "convert_to_pandas",
    "convert_to_polars",
    "ConversionError",
    "DataBridge",
    "DataFrameConverter",
    "DataFrameWrapper",
    "DTypeMapper",
    "DTypeMappingError",
    "FeatureFlags",
    "get_available_backends",
    "get_feature_flags",
    "GlobalBackendContext",
    "is_backend_available",
    "is_feature_enabled",
    "LazyConverter",
    "optimize_for_memory",
    "PandasBackend",
    "PolarsBackend",
    "POLARS_AVAILABLE",
    "SchemaIssue",
    "SchemaValidator",
]
