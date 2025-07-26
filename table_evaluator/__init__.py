import warnings
from importlib.metadata import version, PackageNotFoundError

from .table_evaluator import TableEvaluator
from .utils import load_data

# Suppress common warnings that appear in dependencies
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*multi_class.*")

__all__ = ["TableEvaluator", "load_data"]


try:
    __version__ = version('table-evaluator')
except PackageNotFoundError:
    __version__ = 'unknown'
