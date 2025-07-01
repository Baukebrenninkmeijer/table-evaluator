# To run these benchmarks, you need to install the following packages:
# pip install pytest pytest-benchmark pytest-memray
# Then, you can run the benchmarks using the following command:
# pytest benchmarks/benchmark_performance.py

import pytest
import pandas as pd
import tempfile
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
from table_evaluator.association_metrics import cramers_v
from table_evaluator.plots import plot_distributions

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


@pytest.mark.data_loading
class TestDataLoading:
    def test_load_csv_pandas(self, benchmark, data_files):
        """Benchmark loading CSV with pandas."""
        csv_path, _ = data_files
        benchmark(load_data, csv_path, csv_path, backend="pandas")

    def test_load_csv_polars(self, benchmark, data_files):
        """Benchmark loading CSV with polars."""
        csv_path, _ = data_files
        benchmark(load_data, csv_path, csv_path, backend="polars")

    def test_load_parquet_pandas(self, benchmark, data_files):
        """Benchmark loading parquet with pandas."""
        _, parquet_path = data_files
        benchmark(load_data, parquet_path, parquet_path, backend="pandas")

    def test_load_parquet_polars(self, benchmark, data_files):
        """Benchmark loading parquet with polars."""
        _, parquet_path = data_files
        benchmark(load_data, parquet_path, parquet_path, backend="polars")


@pytest.mark.evaluation
class TestEvaluation:
    def test_pandas_evaluation(self, benchmark, generate_data):
        """Benchmark pandas evaluation."""
        pandas_df, _ = generate_data
        evaluator = TableEvaluator(pandas_df, pandas_df)
        benchmark(evaluator.evaluate, target_col="col_0")

    def test_polars_evaluation(self, benchmark, generate_data):
        """Benchmark polars evaluation."""
        _, polars_df = generate_data
        evaluator = TableEvaluator(polars_df, polars_df.clone())
        benchmark(evaluator.evaluate, target_col="col_0")

    def test_pandas_evaluation_no_target_col(self, benchmark, generate_data):
        """Benchmark pandas evaluation without a target column."""
        pandas_df, _ = generate_data
        evaluator = TableEvaluator(pandas_df, pandas_df)
        benchmark(evaluator.evaluate)

    def test_polars_evaluation_no_target_col(self, benchmark, generate_data):
        """Benchmark polars evaluation without a target column."""
        _, polars_df = generate_data
        evaluator = TableEvaluator(polars_df, polars_df.clone())
        benchmark(evaluator.evaluate)


@pytest.mark.data_conversion
class TestDataConversion:
    def test_to_numerical_pandas(self, benchmark, sample_data_categorical):
        """Benchmark DataConverter.to_numerical with pandas."""
        pandas_df, _ = sample_data_categorical
        converter = DataConverter()
        benchmark(converter.to_numerical, pandas_df, pandas_df.copy(), ["cat_col"])

    def test_to_numerical_polars(self, benchmark, sample_data_categorical):
        """Benchmark DataConverter.to_numerical with polars."""
        _, polars_df = sample_data_categorical
        converter = DataConverter(backend="polars")
        benchmark(converter.to_numerical, polars_df, polars_df.clone(), ["cat_col"])

    def test_to_one_hot_pandas(self, benchmark, sample_data_categorical):
        """Benchmark DataConverter.to_one_hot with pandas."""
        pandas_df, _ = sample_data_categorical
        converter = DataConverter()
        benchmark(converter.to_one_hot, pandas_df, pandas_df.copy(), ["cat_col"])

    def test_to_one_hot_polars(self, benchmark, sample_data_categorical):
        """Benchmark DataConverter.to_one_hot with polars."""
        _, polars_df = sample_data_categorical
        converter = DataConverter(backend="polars")
        benchmark(converter.to_one_hot, polars_df, polars_df.clone(), ["cat_col"])

    def test_ensure_compatible_columns_pandas(self, benchmark, generate_data):
        """Benchmark DataConverter.ensure_compatible_columns with pandas."""
        pandas_df, _ = generate_data
        converter = DataConverter()
        benchmark(converter.ensure_compatible_columns, pandas_df, pandas_df.copy())

    def test_ensure_compatible_columns_polars(self, benchmark, generate_data):
        """Benchmark DataConverter.ensure_compatible_columns with polars."""
        _, polars_df = generate_data
        converter = DataConverter(backend="polars")
        benchmark(converter.ensure_compatible_columns, polars_df, polars_df.clone())


@pytest.mark.plotting
class TestPlotting:
    def test_plot_distributions_pandas(self, benchmark, generate_data):
        """Benchmark plot_distributions with pandas."""
        pandas_df, _ = generate_data
        benchmark(plot_distributions, pandas_df, pandas_df)

    def test_plot_distributions_polars(self, benchmark, generate_data):
        """Benchmark plot_distributions with polars."""
        _, polars_df = generate_data
        benchmark(plot_distributions, polars_df, polars_df)


@pytest.mark.association
class TestAssociation:
    def test_cramers_v_pandas(self, benchmark, sample_data_categorical):
        """Benchmark cramers_v with pandas."""
        pandas_df, _ = sample_data_categorical
        benchmark(cramers_v, pandas_df, "cat_col", "col_0")

    def test_cramers_v_polars(self, benchmark, sample_data_categorical):
        """Benchmark cramers_v with polars."""
        _, polars_df = sample_data_categorical
        benchmark(cramers_v, polars_df, "cat_col", "col_0")


@pytest.mark.memory
@pytest.mark.parametrize("memray", [True], indirect=True)
class TestMemoryProfiling:
    @pytest.fixture(params=[10000, 50000, 100000])
    def memory_test_data(self, request):
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

    def test_memory_usage_pandas_conversion(self, memray, memory_test_data):
        """Test memory usage of pandas data conversion operations."""
        pandas_df, _, n_rows = memory_test_data
        converter = DataConverter()
        converter.to_numerical(pandas_df, pandas_df.copy(), ["cat_col"])
        converter.to_one_hot(pandas_df, pandas_df.copy(), ["cat_col"])

    def test_memory_usage_polars_conversion(self, memray, memory_test_data):
        """Test memory usage of polars data conversion operations."""
        _, polars_df, n_rows = memory_test_data
        converter = DataConverter(backend="polars")
        converter.to_numerical(polars_df, polars_df.clone(), ["cat_col"])
        converter.to_one_hot(polars_df, polars_df.clone(), ["cat_col"])


@pytest.mark.scalability
class TestScalability:
    @pytest.fixture(params=[1000, 5000, 10000, 50000, 100000])
    def scalability_data(self, request):
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

    def test_scalability_data_loading_csv(self, benchmark, scalability_data):
        """Test scalability of CSV loading across backends."""
        pandas_df, _, n_rows = scalability_data

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / f"test_{n_rows}.csv"
            pandas_df.to_csv(csv_path, index=False)

            benchmark(load_data, csv_path, csv_path, backend="pandas")
            benchmark.extra_info["n_rows"] = n_rows
            benchmark.extra_info["backend"] = "pandas"

    def test_scalability_data_loading_parquet(self, benchmark, scalability_data):
        """Test scalability of Parquet loading across backends."""
        pandas_df, _, n_rows = scalability_data

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / f"test_{n_rows}.parquet"
            pandas_df.to_parquet(parquet_path, index=False)

            benchmark(load_data, parquet_path, parquet_path, backend="polars")
            benchmark.extra_info["n_rows"] = n_rows
            benchmark.extra_info["backend"] = "polars"

    def test_scalability_conversion_operations(self, benchmark, scalability_data):
        """Test scalability of conversion operations across different data sizes."""
        pandas_df, polars_df, n_rows = scalability_data
        categorical_cols = [col for col in pandas_df.columns if col.startswith("cat_")]

        # Test pandas conversion scalability
        pandas_converter = DataConverter()
        benchmark.group = "pandas_conversion"
        benchmark(
            pandas_converter.to_one_hot, pandas_df, pandas_df.copy(), categorical_cols
        )

        # Test polars conversion scalability
        polars_converter = DataConverter(backend="polars")
        benchmark.group = "polars_conversion"
        benchmark(
            polars_converter.to_one_hot, polars_df, polars_df.clone(), categorical_cols
        )
