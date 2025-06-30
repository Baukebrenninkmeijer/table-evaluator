"""Backend context manager for thread-safe backend switching."""

import logging
import threading
from contextlib import contextmanager
from typing import Generator, Optional

from .backend_factory import BackendFactory
from .backend_types import BackendType
from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class BackendContext:
    """Thread-safe context manager for backend switching within operations.

    This class allows temporary backend switching within specific code blocks
    while maintaining thread safety and automatic cleanup.
    """

    _local = threading.local()
    _lock = threading.Lock()

    @classmethod
    def get_current_backend(cls) -> Optional[BaseBackend]:
        """Get the current backend for this thread.

        Returns:
            Current backend instance or None if not set
        """
        return getattr(cls._local, "current_backend", None)

    @classmethod
    def get_current_backend_type(cls) -> Optional[BackendType]:
        """Get the current backend type for this thread.

        Returns:
            Current BackendType or None if not set
        """
        return getattr(cls._local, "current_backend_type", None)

    @classmethod
    def set_current_backend(
        cls, backend: BaseBackend, backend_type: BackendType
    ) -> None:
        """Set the current backend for this thread.

        Args:
            backend: Backend instance to set as current
            backend_type: Type of the backend
        """
        cls._local.current_backend = backend
        cls._local.current_backend_type = backend_type

    @classmethod
    def clear_current_backend(cls) -> None:
        """Clear the current backend for this thread."""
        if hasattr(cls._local, "current_backend"):
            delattr(cls._local, "current_backend")
        if hasattr(cls._local, "current_backend_type"):
            delattr(cls._local, "current_backend_type")

    @classmethod
    @contextmanager
    def use_backend(
        cls, backend_type: BackendType, **kwargs
    ) -> Generator[BaseBackend, None, None]:
        """Context manager for temporarily using a specific backend.

        Args:
            backend_type: Type of backend to use
            **kwargs: Additional arguments for backend creation

        Yields:
            Backend instance for the specified type

        Example:
            with BackendContext.use_backend(BackendType.POLARS, lazy=True) as backend:
                df = backend.load_csv("data.csv")
                # Process with Polars backend
        """
        # Store previous backend state
        previous_backend = cls.get_current_backend()
        previous_backend_type = cls.get_current_backend_type()

        try:
            # Get the requested backend
            factory = BackendFactory()
            backend = factory.get_backend(backend_type, **kwargs)

            # Set as current backend
            cls.set_current_backend(backend, backend_type)

            logger.debug(
                f"Switched to {backend_type.value} backend in thread {threading.current_thread().name}"
            )

            yield backend

        finally:
            # Restore previous backend state
            if previous_backend is not None:
                cls.set_current_backend(previous_backend, previous_backend_type)
                logger.debug(
                    f"Restored previous backend in thread {threading.current_thread().name}"
                )
            else:
                cls.clear_current_backend()
                logger.debug(
                    f"Cleared backend context in thread {threading.current_thread().name}"
                )

    @classmethod
    @contextmanager
    def auto_backend(
        cls,
        data_size: int = 0,
        file_format: str = "csv",
        prefer_lazy: bool = False,
        **kwargs,
    ) -> Generator[BaseBackend, None, None]:
        """Context manager for automatically selecting optimal backend.

        Args:
            data_size: Size of data to process
            file_format: Format of data files
            prefer_lazy: Whether to prefer lazy evaluation
            **kwargs: Additional arguments for backend creation

        Yields:
            Automatically selected optimal backend instance

        Example:
            with BackendContext.auto_backend(data_size=1000000, file_format="parquet") as backend:
                df = backend.load_parquet("large_data.parquet")
                # Automatically uses Polars for large parquet files
        """
        context_kwargs = {
            "data_size": data_size,
            "file_format": file_format,
            "prefer_lazy": prefer_lazy,
            **kwargs,
        }

        with cls.use_backend(BackendType.AUTO, **context_kwargs) as backend:
            yield backend

    @classmethod
    def get_or_create_backend(cls, backend_type: BackendType, **kwargs) -> BaseBackend:
        """Get current backend or create new one if not set.

        Args:
            backend_type: Type of backend to get/create
            **kwargs: Additional arguments for backend creation

        Returns:
            Backend instance
        """
        current_backend = cls.get_current_backend()
        current_type = cls.get_current_backend_type()

        # If we have a current backend of the requested type, use it
        if current_backend is not None and current_type == backend_type:
            return current_backend

        # Otherwise create a new backend
        factory = BackendFactory()
        return factory.get_backend(backend_type, **kwargs)


class GlobalBackendContext:
    """Global backend context for setting application-wide defaults.

    This class provides application-wide backend configuration that can be
    overridden locally using BackendContext.
    """

    _global_backend_type: Optional[BackendType] = None
    _global_backend_kwargs: dict = {}
    _lock = threading.Lock()

    @classmethod
    def set_global_backend(cls, backend_type: BackendType, **kwargs) -> None:
        """Set global default backend for the application.

        Args:
            backend_type: Default backend type
            **kwargs: Default arguments for backend creation
        """
        with cls._lock:
            cls._global_backend_type = backend_type
            cls._global_backend_kwargs = kwargs.copy()

        logger.info(f"Set global backend to {backend_type.value}")

    @classmethod
    def get_global_backend(cls) -> tuple[Optional[BackendType], dict]:
        """Get global backend configuration.

        Returns:
            Tuple of (backend_type, kwargs) or (None, {}) if not set
        """
        with cls._lock:
            return cls._global_backend_type, cls._global_backend_kwargs.copy()

    @classmethod
    def clear_global_backend(cls) -> None:
        """Clear global backend configuration."""
        with cls._lock:
            cls._global_backend_type = None
            cls._global_backend_kwargs.clear()

        logger.info("Cleared global backend configuration")

    @classmethod
    def get_effective_backend(
        cls, requested_type: Optional[BackendType] = None
    ) -> BackendType:
        """Get effective backend type considering global and local settings.

        Args:
            requested_type: Explicitly requested backend type

        Returns:
            Effective backend type to use
        """
        # Priority: requested > thread-local > global > auto
        if requested_type is not None:
            return requested_type

        thread_local_type = BackendContext.get_current_backend_type()
        if thread_local_type is not None:
            return thread_local_type

        global_type, _ = cls.get_global_backend()
        if global_type is not None:
            return global_type

        return BackendType.AUTO
