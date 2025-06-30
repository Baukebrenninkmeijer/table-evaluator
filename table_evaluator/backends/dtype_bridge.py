"""Data type bridge for seamless pandas/Polars conversion."""

import logging
from typing import Any, Dict, List, Set, Union

import numpy as np
import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DTypeMappingError(Exception):
    """Exception raised when dtype mapping fails."""

    pass


class DTypeMapper:
    """Comprehensive dtype mapping between pandas and Polars with validation.

    This class provides bidirectional mapping between pandas and Polars data types,
    ensuring type fidelity during conversions while handling edge cases gracefully.
    """

    def __init__(self):
        """Initialize dtype mapper with comprehensive mapping tables."""
        # Pandas to Polars mapping
        self._pandas_to_polars = self._build_pandas_to_polars_mapping()

        # Polars to Pandas mapping (reverse of above)
        self._polars_to_pandas = self._build_polars_to_pandas_mapping()

        # Supported dtypes for validation
        self._supported_pandas_dtypes = set(self._pandas_to_polars.keys())
        self._supported_polars_dtypes = (
            set(self._polars_to_pandas.keys()) if POLARS_AVAILABLE else set()
        )

    def _build_pandas_to_polars_mapping(self) -> Dict[str, str]:
        """Build comprehensive mapping from pandas dtypes to Polars dtypes."""
        if not POLARS_AVAILABLE:
            return {}

        mapping = {
            # Integer types
            "int8": "Int8",
            "int16": "Int16",
            "int32": "Int32",
            "int64": "Int64",
            "Int8": "Int8",  # Nullable pandas integers
            "Int16": "Int16",
            "Int32": "Int32",
            "Int64": "Int64",
            # Unsigned integer types
            "uint8": "UInt8",
            "uint16": "UInt16",
            "uint32": "UInt32",
            "uint64": "UInt64",
            "UInt8": "UInt8",  # Nullable pandas integers
            "UInt16": "UInt16",
            "UInt32": "UInt32",
            "UInt64": "UInt64",
            # Float types
            "float32": "Float32",
            "float64": "Float64",
            "Float32": "Float32",  # Nullable pandas floats
            "Float64": "Float64",
            # Boolean types
            "bool": "Boolean",
            "boolean": "Boolean",  # Nullable pandas boolean
            # String types
            "object": "String",  # Default for object dtype
            "string": "String",  # Pandas string dtype
            "String": "String",  # Nullable pandas string
            # Datetime types
            "datetime64[ns]": "Datetime",
            "datetime64[us]": "Datetime",
            "datetime64[ms]": "Datetime",
            "datetime64[s]": "Datetime",
            # Timedelta types
            "timedelta64[ns]": "Duration",
            "timedelta64[us]": "Duration",
            "timedelta64[ms]": "Duration",
            "timedelta64[s]": "Duration",
            # Category type
            "category": "Categorical",
        }

        return mapping

    def _build_polars_to_pandas_mapping(self) -> Dict[str, str]:
        """Build comprehensive mapping from Polars dtypes to pandas dtypes."""
        if not POLARS_AVAILABLE:
            return {}

        mapping = {
            # Integer types
            "Int8": "int8",
            "Int16": "int16",
            "Int32": "int32",
            "Int64": "int64",
            # Unsigned integer types
            "UInt8": "uint8",
            "UInt16": "uint16",
            "UInt32": "uint32",
            "UInt64": "uint64",
            # Float types
            "Float32": "float32",
            "Float64": "float64",
            # Boolean type
            "Boolean": "boolean",  # Use nullable boolean
            # String type
            "String": "string",  # Use pandas string dtype
            "Utf8": "string",  # Legacy Polars string type
            # Datetime types
            "Datetime": "datetime64[ns]",
            "Date": "datetime64[ns]",
            # Duration type
            "Duration": "timedelta64[ns]",
            "Time": "timedelta64[ns]",
            # Categorical type
            "Categorical": "category",
        }

        return mapping

    def pandas_to_polars_dtype(
        self, pandas_dtype: Union[str, np.dtype, pd.api.types.CategoricalDtype]
    ) -> str:
        """Convert pandas dtype to Polars dtype string.

        Args:
            pandas_dtype: Pandas dtype to convert

        Returns:
            Corresponding Polars dtype string

        Raises:
            DTypeMappingError: If dtype cannot be mapped
        """
        if not POLARS_AVAILABLE:
            raise DTypeMappingError("Polars is not available")

        # Handle different input types
        if isinstance(pandas_dtype, pd.api.types.CategoricalDtype):
            return "Categorical"

        dtype_str = str(pandas_dtype)

        # Handle datetime with different precisions
        if dtype_str.startswith("datetime64"):
            return "Datetime"

        # Handle timedelta with different precisions
        if dtype_str.startswith("timedelta64"):
            return "Duration"

        # Direct mapping lookup
        if dtype_str in self._pandas_to_polars:
            return self._pandas_to_polars[dtype_str]

        # Fallback for object dtype (assume string)
        if dtype_str == "object":
            return "String"

        raise DTypeMappingError(f"Cannot map pandas dtype '{dtype_str}' to Polars")

    def polars_to_pandas_dtype(self, polars_dtype: Union[str, "pl.DataType"]) -> str:
        """Convert Polars dtype to pandas dtype string.

        Args:
            polars_dtype: Polars dtype to convert

        Returns:
            Corresponding pandas dtype string

        Raises:
            DTypeMappingError: If dtype cannot be mapped
        """
        if not POLARS_AVAILABLE:
            raise DTypeMappingError("Polars is not available")

        # Handle Polars DataType objects
        if hasattr(polars_dtype, "__class__"):
            dtype_str = str(polars_dtype)
        else:
            dtype_str = str(polars_dtype)

        # Clean up dtype string representation
        dtype_str = dtype_str.replace("Polars", "").strip()

        # Direct mapping lookup
        if dtype_str in self._polars_to_pandas:
            return self._polars_to_pandas[dtype_str]

        raise DTypeMappingError(f"Cannot map Polars dtype '{dtype_str}' to pandas")

    def validate_pandas_dtypes(self, dtypes: Dict[str, Any]) -> List[str]:
        """Validate that pandas dtypes can be converted to Polars.

        Args:
            dtypes: Dictionary of column names to pandas dtypes

        Returns:
            List of columns with unsupported dtypes
        """
        unsupported = []

        for column, dtype in dtypes.items():
            try:
                self.pandas_to_polars_dtype(dtype)
            except DTypeMappingError:
                unsupported.append(column)
                logger.warning(
                    f"Column '{column}' has unsupported dtype '{dtype}' for Polars conversion"
                )

        return unsupported

    def validate_polars_dtypes(self, dtypes: Dict[str, Any]) -> List[str]:
        """Validate that Polars dtypes can be converted to pandas.

        Args:
            dtypes: Dictionary of column names to Polars dtypes

        Returns:
            List of columns with unsupported dtypes
        """
        if not POLARS_AVAILABLE:
            return list(dtypes.keys())

        unsupported = []

        for column, dtype in dtypes.items():
            try:
                self.polars_to_pandas_dtype(dtype)
            except DTypeMappingError:
                unsupported.append(column)
                logger.warning(
                    f"Column '{column}' has unsupported dtype '{dtype}' for pandas conversion"
                )

        return unsupported

    def get_conversion_suggestions(
        self, unsupported_dtypes: List[str]
    ) -> Dict[str, str]:
        """Get suggestions for handling unsupported dtypes.

        Args:
            unsupported_dtypes: List of unsupported dtype strings

        Returns:
            Dictionary mapping unsupported dtypes to suggested alternatives
        """
        suggestions = {}

        for dtype in unsupported_dtypes:
            dtype_str = str(dtype)

            if "complex" in dtype_str:
                suggestions[dtype_str] = (
                    "Convert to string representation or split into real/imaginary parts"
                )
            elif "period" in dtype_str:
                suggestions[dtype_str] = "Convert to datetime or string representation"
            elif "interval" in dtype_str:
                suggestions[dtype_str] = (
                    "Convert to string representation or separate start/end columns"
                )
            else:
                suggestions[dtype_str] = (
                    "Convert to string representation or closest supported numeric type"
                )

        return suggestions

    def create_polars_schema(self, pandas_dtypes: Dict[str, Any]) -> Dict[str, str]:
        """Create Polars schema from pandas dtypes.

        Args:
            pandas_dtypes: Dictionary of column names to pandas dtypes

        Returns:
            Dictionary mapping column names to Polars dtype strings

        Raises:
            DTypeMappingError: If any dtypes cannot be mapped
        """
        if not POLARS_AVAILABLE:
            raise DTypeMappingError("Polars is not available")

        schema = {}
        errors = []

        for column, dtype in pandas_dtypes.items():
            try:
                polars_dtype = self.pandas_to_polars_dtype(dtype)
                schema[column] = polars_dtype
            except DTypeMappingError as e:
                errors.append(f"Column '{column}': {str(e)}")

        if errors:
            raise DTypeMappingError(f"Cannot create Polars schema: {'; '.join(errors)}")

        return schema

    def create_pandas_dtypes(self, polars_dtypes: Dict[str, Any]) -> Dict[str, str]:
        """Create pandas dtypes from Polars dtypes.

        Args:
            polars_dtypes: Dictionary of column names to Polars dtypes

        Returns:
            Dictionary mapping column names to pandas dtype strings

        Raises:
            DTypeMappingError: If any dtypes cannot be mapped
        """
        if not POLARS_AVAILABLE:
            raise DTypeMappingError("Polars is not available")

        dtypes = {}
        errors = []

        for column, dtype in polars_dtypes.items():
            try:
                pandas_dtype = self.polars_to_pandas_dtype(dtype)
                dtypes[column] = pandas_dtype
            except DTypeMappingError as e:
                errors.append(f"Column '{column}': {str(e)}")

        if errors:
            raise DTypeMappingError(f"Cannot create pandas dtypes: {'; '.join(errors)}")

        return dtypes

    def get_supported_pandas_dtypes(self) -> Set[str]:
        """Get set of supported pandas dtypes for conversion."""
        return self._supported_pandas_dtypes.copy()

    def get_supported_polars_dtypes(self) -> Set[str]:
        """Get set of supported Polars dtypes for conversion."""
        return self._supported_polars_dtypes.copy()

    def is_dtype_compatible(self, source_dtype: str, target_backend: str) -> bool:
        """Check if a dtype is compatible with target backend.

        Args:
            source_dtype: Source dtype string
            target_backend: Target backend ("pandas" or "polars")

        Returns:
            True if dtype is compatible, False otherwise
        """
        if target_backend == "polars":
            try:
                self.pandas_to_polars_dtype(source_dtype)
                return True
            except DTypeMappingError:
                return False

        elif target_backend == "pandas":
            try:
                self.polars_to_pandas_dtype(source_dtype)
                return True
            except DTypeMappingError:
                return False

        else:
            raise ValueError(f"Unknown target backend: {target_backend}")
