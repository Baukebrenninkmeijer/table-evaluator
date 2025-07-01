import gc
import psutil
import pytest
import pandas as pd
import tempfile
import time
from pathlib import Path

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from table_evaluator.table_evaluator import TableEvaluator
from table_evaluator.utils import load_data
from table_evaluator.data.data_converter import DataConverter

# Skip all tests if polars is not available
pytestmark = pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")


@pytest.fixture(scope="module", params=[1000, 10000, 100000])
def generate_data(request):
    """Generate data for benchmarking with varying sizes."""
    n_rows = request.param
    n_cols = 10
    data = {f"col_{i}": list(range(n_rows)) for i in range(n_cols)}
    pandas_df = pd.DataFrame(data)
    polars_df = pl.DataFrame(data)
    return pandas_df, polars_df


@pytest.fixture(scope="module")
def sample_data_categorical(generate_data):
    """Generate data with categorical columns for DataConverter benchmarks."""
    pandas_df, polars_df = generate_data
    pandas_df["cat_col"] = [f"cat{i % 5}" for i in range(len(pandas_df))]
    polars_df = polars_df.with_columns(
        pl.Series("cat_col", [f"cat{i % 5}" for i in range(len(polars_df))])
    )
    return pandas_df, polars_df


@pytest.fixture(scope="module")
def data_files(generate_data):
    """Create data files for benchmarking."""
    pandas_df, _ = generate_data
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        parquet_path = Path(tmpdir) / "test.parquet"
        pandas_df.to_csv(csv_path, index=False)
        pandas_df.to_parquet(parquet_path, index=False)
        yield csv_path, parquet_path


def test_load_csv_pandas(benchmark, data_files):
    """Benchmark loading CSV with pandas."""
    csv_path, _ = data_files
    benchmark(load_data, csv_path, csv_path, backend="pandas")


def test_load_csv_polars(benchmark, data_files):
    """Benchmark loading CSV with polars."""
    csv_path, _ = data_files
    benchmark(load_data, csv_path, csv_path, backend="polars")


def test_load_parquet_pandas(benchmark, data_files):
    """Benchmark loading parquet with pandas."""
    _, parquet_path = data_files
    benchmark(load_data, parquet_path, parquet_path, backend="pandas")


def test_load_parquet_polars(benchmark, data_files):
    """Benchmark loading parquet with polars."""
    _, parquet_path = data_files
    benchmark(load_data, parquet_path, parquet_path, backend="polars")


def test_pandas_evaluation(benchmark, generate_data):
    """Benchmark pandas evaluation."""
    pandas_df, _ = generate_data
    evaluator = TableEvaluator(pandas_df, pandas_df)
    benchmark(evaluator.evaluate, target_col="col_0")


def test_polars_evaluation(benchmark, generate_data):
    """Benchmark polars evaluation."""
    _, polars_df = generate_data
    evaluator = TableEvaluator(polars_df, polars_df.clone())
    benchmark(evaluator.evaluate, target_col="col_0")


def test_to_numerical_pandas(benchmark, sample_data_categorical):
    """Benchmark DataConverter.to_numerical with pandas."""

    pandas_df, _ = sample_data_categorical
    converter = DataConverter()
    benchmark(converter.to_numerical, pandas_df, pandas_df.copy(), ["cat_col"])


def test_to_numerical_polars(benchmark, sample_data_categorical):
    """Benchmark DataConverter.to_numerical with polars."""

    _, polars_df = sample_data_categorical
    converter = DataConverter(backend="polars")
    benchmark(converter.to_numerical, polars_df, polars_df.clone(), ["cat_col"])


def test_to_one_hot_pandas(benchmark, sample_data_categorical):
    """Benchmark DataConverter.to_one_hot with pandas."""

    pandas_df, _ = sample_data_categorical
    converter = DataConverter()
    benchmark(converter.to_one_hot, pandas_df, pandas_df.copy(), ["cat_col"])


def test_to_one_hot_polars(benchmark, sample_data_categorical):
    """Benchmark DataConverter.to_one_hot with polars."""

    _, polars_df = sample_data_categorical
    converter = DataConverter(backend="polars")
    benchmark(converter.to_one_hot, polars_df, polars_df.clone(), ["cat_col"])


def test_ensure_compatible_columns_pandas(benchmark, generate_data):
    """Benchmark DataConverter.ensure_compatible_columns with pandas."""

    pandas_df, _ = generate_data
    converter = DataConverter()
    benchmark(converter.ensure_compatible_columns, pandas_df, pandas_df.copy())


def test_ensure_compatible_columns_polars(benchmark, generate_data):
    """Benchmark DataConverter.ensure_compatible_columns with polars."""

    _, polars_df = generate_data
    converter = DataConverter(backend="polars")
    benchmark(converter.ensure_compatible_columns, polars_df, polars_df.clone())


# Memory profiling utilities
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def memory_profile_function(func, *args, **kwargs):
    """Profile memory usage of a function."""
    gc.collect()  # Clean up before measurement
    memory_before = get_memory_usage()

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    memory_after = get_memory_usage()
    memory_peak = max(memory_before, memory_after)

    return {
        "result": result,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "memory_used": memory_after - memory_before,
        "memory_peak": memory_peak,
        "execution_time": end_time - start_time,
    }


# Memory profiling tests
@pytest.fixture(params=[10000, 50000, 100000])
def memory_test_data(request):
    """Generate data for memory profiling with varying sizes."""
    n_rows = request.param
    data = {
        "int_col": list(range(n_rows)),
        "float_col": [float(i) * 1.1 for i in range(n_rows)],
        "str_col": [f"str_{i}" for i in range(n_rows)],
        "cat_col": [f"cat_{i % 10}" for i in range(n_rows)],
    }
    pandas_df = pd.DataFrame(data)
    polars_df = pl.DataFrame(data)
    return pandas_df, polars_df, n_rows


def test_memory_usage_pandas_conversion(memory_test_data):
    """Test memory usage of pandas data conversion operations."""

    pandas_df, _, n_rows = memory_test_data
    converter = DataConverter()

    # Test to_numerical
    numerical_profile = memory_profile_function(
        converter.to_numerical, pandas_df, pandas_df.copy(), ["cat_col"]
    )

    # Test to_one_hot
    onehot_profile = memory_profile_function(
        converter.to_one_hot, pandas_df, pandas_df.copy(), ["cat_col"]
    )

    print(f"\nPandas Memory Profile (n_rows={n_rows}):")
    print(
        f"  to_numerical: {numerical_profile['memory_used']:.2f}MB used, {numerical_profile['execution_time']:.4f}s"
    )
    print(
        f"  to_one_hot: {onehot_profile['memory_used']:.2f}MB used, {onehot_profile['execution_time']:.4f}s"
    )


def test_memory_usage_polars_conversion(memory_test_data):
    """Test memory usage of polars data conversion operations."""

    _, polars_df, n_rows = memory_test_data
    converter = DataConverter(backend="polars")

    # Test to_numerical
    numerical_profile = memory_profile_function(
        converter.to_numerical, polars_df, polars_df.clone(), ["cat_col"]
    )

    # Test to_one_hot
    onehot_profile = memory_profile_function(
        converter.to_one_hot, polars_df, polars_df.clone(), ["cat_col"]
    )

    print(f"\nPolars Memory Profile (n_rows={n_rows}):")
    print(
        f"  to_numerical: {numerical_profile['memory_used']:.2f}MB used, {numerical_profile['execution_time']:.4f}s"
    )
    print(
        f"  to_one_hot: {onehot_profile['memory_used']:.2f}MB used, {onehot_profile['execution_time']:.4f}s"
    )


# Scalability tests with varying dataset sizes
@pytest.fixture(params=[1000, 5000, 10000, 50000, 100000])
def scalability_data(request):
    """Generate data for scalability testing with comprehensive size range."""
    n_rows = request.param
    n_cols = 15  # More columns for comprehensive testing

    data = {}
    # Numerical columns
    for i in range(n_cols // 3):
        data[f"num_{i}"] = [float(j + i) for j in range(n_rows)]

    # String columns
    for i in range(n_cols // 3):
        data[f"str_{i}"] = [f"string_{j}_{i}" for j in range(n_rows)]

    # Categorical columns
    for i in range(n_cols // 3):
        data[f"cat_{i}"] = [f"category_{j % (5 + i)}" for j in range(n_rows)]

    pandas_df = pd.DataFrame(data)
    polars_df = pl.DataFrame(data)

    return pandas_df, polars_df, n_rows


def test_scalability_data_loading_csv(benchmark, scalability_data):
    """Test scalability of CSV loading across backends."""
    pandas_df, _, n_rows = scalability_data

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / f"test_{n_rows}.csv"
        pandas_df.to_csv(csv_path, index=False)

        # The benchmark will capture timing automatically
        benchmark(load_data, csv_path, csv_path, backend="pandas")

        # Add custom metrics as extra info
        benchmark.extra_info["n_rows"] = n_rows
        benchmark.extra_info["backend"] = "pandas"


def test_scalability_data_loading_parquet(benchmark, scalability_data):
    """Test scalability of Parquet loading across backends."""
    pandas_df, _, n_rows = scalability_data

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / f"test_{n_rows}.parquet"
        pandas_df.to_parquet(parquet_path, index=False)

        benchmark(load_data, parquet_path, parquet_path, backend="polars")

        benchmark.extra_info["n_rows"] = n_rows
        benchmark.extra_info["backend"] = "polars"


def test_scalability_conversion_operations(scalability_data):
    """Test scalability of conversion operations across different data sizes."""
    pandas_df, polars_df, n_rows = scalability_data

    categorical_cols = [col for col in pandas_df.columns if col.startswith("cat_")]

    # Test pandas conversion scalability

    pandas_converter = DataConverter()
    pandas_profile = memory_profile_function(
        pandas_converter.to_one_hot, pandas_df, pandas_df.copy(), categorical_cols
    )

    # Test polars conversion scalability
    polars_converter = DataConverter(backend="polars")
    polars_profile = memory_profile_function(
        polars_converter.to_one_hot, polars_df, polars_df.clone(), categorical_cols
    )

    print(
        f"\nScalability Test (n_rows={n_rows}, categorical_cols={len(categorical_cols)}):"
    )
    print(
        f"  Pandas: {pandas_profile['memory_used']:.2f}MB, {pandas_profile['execution_time']:.4f}s"
    )
    print(
        f"  Polars: {polars_profile['memory_used']:.2f}MB, {polars_profile['execution_time']:.4f}s"
    )
    print(
        f"  Memory ratio (Polars/Pandas): {polars_profile['memory_used']/max(pandas_profile['memory_used'], 0.001):.2f}"
    )
    print(
        f"  Speed ratio (Pandas/Polars): {pandas_profile['execution_time']/max(polars_profile['execution_time'], 0.001):.2f}"
    )
