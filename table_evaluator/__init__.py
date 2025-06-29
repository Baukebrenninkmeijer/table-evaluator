try:
    import tomllib
except ImportError:
    import tomli as tomllib
from pathlib import Path

from .table_evaluator import TableEvaluator
from .utils import load_data

__all__ = ["TableEvaluator", "load_data"]


def _get_version():
    """Get version from pyproject.toml"""
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except (FileNotFoundError, KeyError, ImportError):
        return "unknown"


__version__ = _get_version()
