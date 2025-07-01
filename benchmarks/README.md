# Performance Benchmarking Suite

This directory contains comprehensive performance benchmarks comparing pandas and Polars backends for table-evaluator operations.

## Features

- **Operation Benchmarks**: Key functions like data loading, conversion operations
- **Memory Profiling**: Memory usage comparisons between backends
- **Scalability Tests**: Performance across varying dataset sizes (1K to 100K+ rows)
- **Backend Comparisons**: Direct pandas vs Polars performance analysis

## Requirements

Install the development dependencies including benchmarking tools:

```bash
pip install -e ".[dev]"
# or
pip install psutil pytest-benchmark
```

## Running Benchmarks

### Basic Benchmark Run
```bash
# Run all benchmarks
pytest benchmarks/ --benchmark-only

# Run specific benchmark categories
pytest benchmarks/test_performance.py::test_load_csv_pandas --benchmark-only
pytest benchmarks/test_performance.py::test_memory_usage_pandas_conversion -s
```

### Generate Benchmark Reports
```bash
# Generate detailed HTML report
pytest benchmarks/ --benchmark-only --benchmark-html=benchmark_report.html

# Generate JSON results for comparison
pytest benchmarks/ --benchmark-only --benchmark-json=benchmark_results.json

# Compare with previous results
pytest benchmarks/ --benchmark-only --benchmark-compare=benchmark_results.json
```

### Memory Profiling Tests
```bash
# Run memory profiling tests (includes print output)
pytest benchmarks/test_performance.py::test_memory_usage_pandas_conversion -s
pytest benchmarks/test_performance.py::test_memory_usage_polars_conversion -s
pytest benchmarks/test_performance.py::test_scalability_conversion_operations -s
```

## Benchmark Categories

### 1. Data Loading Benchmarks
- CSV loading performance (pandas vs polars)
- Parquet loading performance (pandas vs polars)
- Various dataset sizes (1K, 10K, 100K rows)

### 2. Data Conversion Benchmarks
- `to_numerical()` operation performance
- `to_one_hot()` operation performance
- `ensure_compatible_columns()` performance
- Backend-specific optimizations

### 3. Memory Profiling
- Memory usage during conversion operations
- Memory efficiency comparisons
- Peak memory consumption analysis

### 4. Scalability Tests
- Performance across dataset sizes (1K to 100K+ rows)
- Multi-column conversion operations
- Complex categorical data handling

## Interpreting Results

### Benchmark Output
```
test_load_csv_pandas                    Mean: 45.2ms ± 2.1ms
test_load_csv_polars                    Mean: 32.1ms ± 1.8ms
```

### Memory Profiling Output
```
Pandas Memory Profile (n_rows=50000):
  to_numerical: 12.45MB used, 0.0234s
  to_one_hot: 18.67MB used, 0.0456s

Polars Memory Profile (n_rows=50000):
  to_numerical: 8.23MB used, 0.0187s
  to_one_hot: 11.34MB used, 0.0298s
```

## Adding New Benchmarks

To add new benchmark tests:

1. Use `pytest-benchmark` for timing benchmarks:
```python
def test_my_operation(benchmark):
    result = benchmark(my_function, arg1, arg2)
```

2. Use `memory_profile_function` for memory analysis:
```python
def test_my_memory_usage():
    profile = memory_profile_function(my_function, arg1, arg2)
    print(f"Memory used: {profile['memory_used']:.2f}MB")
```

3. Add fixture parameters for different dataset sizes:
```python
@pytest.fixture(params=[1000, 10000, 100000])
def my_test_data(request):
    # Generate test data based on request.param
```

## Continuous Integration

The benchmark suite can be integrated into CI/CD pipelines to:
- Detect performance regressions
- Compare performance across different environments
- Generate performance reports for releases

Example CI configuration:
```yaml
- name: Run Performance Benchmarks
  run: |
    pytest benchmarks/ --benchmark-only --benchmark-json=ci_benchmarks.json
    # Upload results or compare with baseline
```
