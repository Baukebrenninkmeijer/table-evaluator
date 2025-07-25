import warnings

from .table_evaluator import TableEvaluator
from .utils import load_data

# Suppress common warnings that appear in dependencies
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*multi_class.*')

__all__ = ['TableEvaluator', 'load_data']


def _get_version() -> str:
    """Get version from setuptools_scm"""
    try:
        from importlib.metadata import version

        return version('table-evaluator')
    except ImportError:
        try:
            # Fallback for older Python versions
            import pkg_resources

            return pkg_resources.get_distribution('table-evaluator').version
        except Exception:
            return 'unknown'


__version__ = _get_version()
