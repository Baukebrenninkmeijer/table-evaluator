import warnings
from importlib.metadata import PackageNotFoundError, version

from .table_evaluator import TableEvaluator
from .utils import load_data

# Suppress common warnings that appear in dependencies
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*multi_class.*')

__all__ = ['TableEvaluator', 'load_data']


def _get_version() -> str:
    """Get version from setuptools_scm.

    Returns:
        str: The package version string, or 'unknown' if unable to determine.

    Note:
        Uses importlib.metadata for Python 3.8+ with pkg_resources fallback.
        Version is dynamically determined from git tags via setuptools_scm.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version('table-evaluator')
    except (ImportError, PackageNotFoundError):
        try:
            # Fallback for older Python versions
            import pkg_resources

            return pkg_resources.get_distribution('table-evaluator').version
        except Exception:  # Catch any exception from pkg_resources fallback
            return 'unknown'


__version__ = _get_version()
