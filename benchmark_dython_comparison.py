"""
Benchmark script to compare native implementations with dython.

This script temporarily installs dython to validate our implementations
produce statistically equivalent results and performance characteristics.
"""

import subprocess  # nosec
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict


def install_dython():
    """Temporarily install dython for comparison."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "dython==0.7.7"])  # nosec
        print("✓ Temporarily installed dython for benchmarking")
        return True
    except subprocess.CalledProcessError:
        print("✗ Could not install dython for comparison")
        return False


def remove_dython():
    """Remove dython after benchmarking."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "dython"]
        )  # nosec
        print("✓ Removed temporary dython installation")
    except subprocess.CalledProcessError:
        print("✗ Could not remove dython")


def generate_test_datasets() -> Dict[str, pd.DataFrame]:
    """Generate various test datasets for comparison."""
    np.random.seed(42)

    datasets = {}

    # Small mixed dataset
    datasets["small_mixed"] = pd.DataFrame(
        {
            "cat_small": ["A", "B"] * 25,
            "cat_medium": [f"Cat_{i%5}" for i in range(50)],
            "num_normal": np.random.normal(50, 10, 50),
            "num_uniform": np.random.uniform(0, 100, 50),
            "bool_col": np.random.choice([True, False], 50),
        }
    )

    # Medium dataset
    datasets["medium_mixed"] = pd.DataFrame(
        {
            "cat1": np.random.choice(["X", "Y", "Z"], 500),
            "cat2": np.random.choice([f"Group_{i}" for i in range(10)], 500),
            "num1": np.random.exponential(2, 500),
            "num2": np.random.gamma(2, 2, 500),
            "bool1": np.random.choice([True, False], 500),
        }
    )

    # Large dataset for performance testing
    datasets["large_mixed"] = pd.DataFrame(
        {
            "cat1": np.random.choice(["A", "B", "C", "D"], 5000),
            "cat2": np.random.choice([f"Type_{i}" for i in range(20)], 5000),
            "num1": np.random.randn(5000),
            "num2": np.random.lognormal(0, 1, 5000),
        }
    )

    return datasets


def benchmark_cramers_v(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Benchmark Cramer's V implementations."""
    print("\n=== Cramer's V Benchmark ===")

    # Import both implementations
    from table_evaluator.association_metrics import cramers_v as native_cramers_v

    try:
        from dython.nominal import cramers_v as dython_cramers_v
    except ImportError:
        print("Dython not available for comparison")
        return {}

    results = {}

    for dataset_name, df in datasets.items():
        print(f"\nTesting dataset: {dataset_name}")

        # Get categorical columns
        cat_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        if len(cat_cols) < 2:
            print(f"Skipping {dataset_name} - insufficient categorical columns")
            continue

        # Test first two categorical columns
        col1, col2 = cat_cols[0], cat_cols[1]
        x, y = df[col1], df[col2]

        # Time native implementation
        start_time = time.time()
        native_result = native_cramers_v(x, y, bias_correction=False)
        native_time = time.time() - start_time

        # Time dython implementation
        start_time = time.time()
        dython_result = dython_cramers_v(x, y, bias_correction=False)
        dython_time = time.time() - start_time

        # Calculate difference
        diff = abs(native_result - dython_result)
        relative_diff = diff / max(dython_result, 1e-10) * 100

        results[dataset_name] = {
            "native_result": native_result,
            "dython_result": dython_result,
            "difference": diff,
            "relative_diff_pct": relative_diff,
            "native_time": native_time,
            "dython_time": dython_time,
            "speedup": dython_time / native_time if native_time > 0 else float("inf"),
        }

        print(f"  Native:  {native_result:.6f} ({native_time:.4f}s)")
        print(f"  Dython:  {dython_result:.6f} ({dython_time:.4f}s)")
        print(f"  Diff:    {diff:.6f} ({relative_diff:.2f}%)")
        print(f"  Speedup: {results[dataset_name]['speedup']:.2f}x")

    return results


def benchmark_theils_u(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Benchmark Theil's U implementations."""
    print("\n=== Theil's U Benchmark ===")

    from table_evaluator.association_metrics import theils_u as native_theils_u

    try:
        from dython.nominal import theils_u as dython_theils_u
    except ImportError:
        print("Dython not available for comparison")
        return {}

    results = {}

    for dataset_name, df in datasets.items():
        print(f"\nTesting dataset: {dataset_name}")

        cat_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        if len(cat_cols) < 2:
            print(f"Skipping {dataset_name} - insufficient categorical columns")
            continue

        col1, col2 = cat_cols[0], cat_cols[1]
        x, y = df[col1], df[col2]

        # Time native implementation
        start_time = time.time()
        native_result = native_theils_u(x, y)
        native_time = time.time() - start_time

        # Time dython implementation
        start_time = time.time()
        dython_result = dython_theils_u(x, y)
        dython_time = time.time() - start_time

        diff = abs(native_result - dython_result)
        relative_diff = diff / max(dython_result, 1e-10) * 100

        results[dataset_name] = {
            "native_result": native_result,
            "dython_result": dython_result,
            "difference": diff,
            "relative_diff_pct": relative_diff,
            "native_time": native_time,
            "dython_time": dython_time,
            "speedup": dython_time / native_time if native_time > 0 else float("inf"),
        }

        print(f"  Native:  {native_result:.6f} ({native_time:.4f}s)")
        print(f"  Dython:  {dython_result:.6f} ({dython_time:.4f}s)")
        print(f"  Diff:    {diff:.6f} ({relative_diff:.2f}%)")
        print(f"  Speedup: {results[dataset_name]['speedup']:.2f}x")

    return results


def benchmark_associations(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Benchmark associations matrix implementations."""
    print("\n=== Associations Matrix Benchmark ===")

    from table_evaluator.association_metrics import associations as native_associations

    try:
        from dython.nominal import associations as dython_associations
    except ImportError:
        print("Dython not available for comparison")
        return {}

    results = {}

    for dataset_name, df in datasets.items():
        print(f"\nTesting dataset: {dataset_name}")

        # Skip large dataset for associations due to O(n²) complexity
        if dataset_name == "large_mixed":
            print("Skipping large dataset for associations benchmark")
            continue

        cat_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # Time native implementation
        start_time = time.time()
        native_result = native_associations(
            df, nominal_columns=cat_cols, compute_only=True
        )["corr"]
        native_time = time.time() - start_time

        # Time dython implementation
        start_time = time.time()
        dython_result = dython_associations(
            df, nominal_columns=cat_cols, compute_only=True
        )["corr"]
        dython_time = time.time() - start_time

        # Calculate matrix difference metrics
        diff_matrix = np.abs(native_result.values - dython_result.values)
        max_diff = np.max(diff_matrix)
        mean_diff = np.mean(diff_matrix)

        results[dataset_name] = {
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "native_time": native_time,
            "dython_time": dython_time,
            "speedup": dython_time / native_time if native_time > 0 else float("inf"),
            "matrix_shape": native_result.shape,
        }

        print(f"  Matrix shape: {native_result.shape}")
        print(f"  Native time:  {native_time:.4f}s")
        print(f"  Dython time:  {dython_time:.4f}s")
        print(f"  Max diff:     {max_diff:.6f}")
        print(f"  Mean diff:    {mean_diff:.6f}")
        print(f"  Speedup:      {results[dataset_name]['speedup']:.2f}x")

    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing native vs dython implementations."""
    print("🚀 Starting Comprehensive Dython vs Native Implementation Benchmark")
    print("=" * 70)

    # Install dython temporarily
    if not install_dython():
        print("Cannot proceed without dython installation")
        return

    try:
        # Generate test datasets
        print("📊 Generating test datasets...")
        datasets = generate_test_datasets()
        print(f"Generated {len(datasets)} test datasets")

        # Run benchmarks
        cramers_results = benchmark_cramers_v(datasets)
        theils_results = benchmark_theils_u(datasets)
        associations_results = benchmark_associations(datasets)

        # Generate summary report
        print("\n" + "=" * 70)
        print("📋 BENCHMARK SUMMARY REPORT")
        print("=" * 70)

        print("\n🎯 ACCURACY ASSESSMENT:")
        print("-" * 30)

        # Check if all differences are within acceptable tolerance
        tolerance = 0.01  # 1% relative difference

        accuracy_issues = []

        for metric_name, results in [
            ("Cramer's V", cramers_results),
            ("Theil's U", theils_results),
        ]:
            max_relative_diff = max(
                (r["relative_diff_pct"] for r in results.values()), default=0
            )
            if max_relative_diff > tolerance:
                accuracy_issues.append(
                    f"{metric_name}: {max_relative_diff:.2f}% max difference"
                )
            print(f"{metric_name:12} max relative difference: {max_relative_diff:.3f}%")

        if associations_results:
            max_assoc_diff = max(
                (r["max_difference"] for r in associations_results.values()), default=0
            )
            if max_assoc_diff > 0.01:
                accuracy_issues.append(
                    f"Associations: {max_assoc_diff:.4f} max absolute difference"
                )
            print(f"{'Associations':12} max absolute difference: {max_assoc_diff:.6f}")

        print("\n⚡ PERFORMANCE ASSESSMENT:")
        print("-" * 30)

        # Calculate average speedups
        all_speedups = []
        for results in [cramers_results, theils_results, associations_results]:
            speedups = [
                r["speedup"] for r in results.values() if r["speedup"] != float("inf")
            ]
            all_speedups.extend(speedups)

        if all_speedups:
            avg_speedup = np.mean(all_speedups)
            print(f"Average speedup: {avg_speedup:.2f}x")

            if avg_speedup < 0.5:
                print("⚠️  WARNING: Native implementation is significantly slower")
            elif avg_speedup > 2.0:
                print("🚀 EXCELLENT: Native implementation is significantly faster")
            else:
                print("✅ GOOD: Performance is comparable")

        print("\n🔍 FINAL ASSESSMENT:")
        print("-" * 30)

        if accuracy_issues:
            print("❌ ACCURACY ISSUES DETECTED:")
            for issue in accuracy_issues:
                print(f"   • {issue}")
            print("\n   Recommendation: Review statistical implementations")
        else:
            print("✅ ACCURACY: All implementations within acceptable tolerance")

        if not all_speedups or np.mean(all_speedups) > 0.1:
            print("✅ PERFORMANCE: Acceptable performance characteristics")
        else:
            print("⚠️  PERFORMANCE: Consider optimization")

        if not accuracy_issues:
            print("\n🎉 OVERALL: Native implementations are ready for production!")
        else:
            print("\n⚠️  OVERALL: Requires fixes before production deployment")

    finally:
        # Clean up dython installation
        remove_dython()


if __name__ == "__main__":
    run_comprehensive_benchmark()
