"""Error handling models for standardized error reporting across the package."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ErrorResult(BaseModel):
    """Standardized error result model for consistent error handling."""

    error: str
    error_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    function_name: str | None = None
    args: tuple[Any, ...] | None = None
    kwargs: dict[str, Any] | None = None
    context: dict[str, Any] | None = None

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


def create_error_result(
    error: Exception,
    function_name: str | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> ErrorResult:
    """
    Create a standardized error result from an exception.

    Args:
        error: The exception that occurred
        function_name: Name of the function where error occurred
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function
        context: Additional context information

    Returns:
        ErrorResult with standardized error information
    """
    return ErrorResult(
        error=str(error),
        error_type=type(error).__name__,
        function_name=function_name,
        args=args,
        kwargs=kwargs,
        context=context,
    )
