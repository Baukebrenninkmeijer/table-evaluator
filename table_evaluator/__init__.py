import pkg_resources

from .table_evaluator import TableEvaluator
from .utils import load_data

__all__ = ["TableEvaluator", "load_data"]

__version__ = pkg_resources.get_distribution("table_evaluator").version
