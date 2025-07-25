"""Performance benchmark for privacy evaluator to ensure refactoring doesn't impact speed."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from table_evaluator.evaluators.privacy_evaluator import PrivacyEvaluator


def create_test_dataset(n_rows: int = 1000, n_cols: int = 10) -> pd.DataFrame:
    """Create a synthetic test dataset for benchmarking."""
    rng = np.random.default_rng(42)

    data = {}
    # Categorical columns (quasi-identifiers)
    data['age_group'] = rng.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_rows)
    data['zipcode'] = rng.choice([f'0{i:04d}' for i in range(1000, 2000)], n_rows)
    data['education'] = rng.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_rows)

    # Sensitive attribute
    data['salary'] = rng.normal(50000, 15000, n_rows)

    # Additional numerical columns
    for i in range(n_cols - 4):
        data[f'feature_{i}'] = rng.normal(0, 1, n_rows)

    return pd.DataFrame(data)


def benchmark_privacy_evaluation(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, iterations: int = 3
) -> dict[str, float]:
    """Benchmark privacy evaluation performance."""

    quasi_identifiers = ['age_group', 'zipcode', 'education']
    sensitive_attributes = ['salary']

    evaluator = PrivacyEvaluator(
        real_data=real_data,
        synthetic_data=synthetic_data,
        quasi_identifiers=quasi_identifiers,
        sensitive_attributes=sensitive_attributes,
    )

    # Warm-up run
    _ = evaluator.evaluate(include_basic=True, include_k_anonymity=True, include_membership_inference=True)

    # Benchmark runs
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()

        _ = evaluator.evaluate(include_basic=True, include_k_anonymity=True, include_membership_inference=True)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times),
        'iterations': iterations,
    }


def benchmark_individual_metrics(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, iterations: int = 5
) -> dict[str, dict[str, float]]:
    """Benchmark individual privacy metrics."""

    from table_evaluator.metrics.privacy import calculate_k_anonymity, simulate_membership_inference_attack

    quasi_identifiers = ['age_group', 'zipcode', 'education']
    sensitive_attributes = ['salary']

    results = {}

    # Benchmark k-anonymity calculation
    k_times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = calculate_k_anonymity(real_data, quasi_identifiers, sensitive_attributes)
        end_time = time.perf_counter()
        k_times.append(end_time - start_time)

    results['k_anonymity'] = {
        'mean_time': np.mean(k_times),
        'std_time': np.std(k_times),
        'min_time': np.min(k_times),
        'max_time': np.max(k_times),
    }

    # Benchmark membership inference attack
    mi_times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = simulate_membership_inference_attack(real_data, synthetic_data)
        end_time = time.perf_counter()
        mi_times.append(end_time - start_time)

    results['membership_inference'] = {
        'mean_time': np.mean(mi_times),
        'std_time': np.std(mi_times),
        'min_time': np.min(mi_times),
        'max_time': np.max(mi_times),
    }

    return results


def run_comprehensive_benchmark() -> dict:
    """Run comprehensive performance benchmark across different dataset sizes."""

    logger.info('🚀 Starting Privacy Evaluator Performance Benchmark')
    logger.info('=' * 60)

    dataset_sizes = [500, 1000, 2000, 5000]
    results = {}

    for size in dataset_sizes:
        logger.info(f'📊 Benchmarking dataset size: {size} rows')

        # Create test datasets
        real_data = create_test_dataset(n_rows=size, n_cols=8)
        # Create synthetic data with some noise
        synthetic_data = real_data.copy()
        # Add noise to numerical columns
        rng = np.random.default_rng(42)
        for col in synthetic_data.select_dtypes(include=[np.number]).columns:
            noise = rng.normal(0, synthetic_data[col].std() * 0.1, len(synthetic_data))
            synthetic_data[col] += noise

        # Benchmark full evaluation
        logger.debug('⏱️  Full evaluation benchmark...')
        full_results = benchmark_privacy_evaluation(real_data, synthetic_data, iterations=3)

        # Benchmark individual metrics
        logger.debug('⏱️  Individual metrics benchmark...')
        metric_results = benchmark_individual_metrics(real_data, synthetic_data, iterations=5)

        results[f'{size}_rows'] = {
            'dataset_size': size,
            'full_evaluation': full_results,
            'individual_metrics': metric_results,
        }

        logger.success(f'Full evaluation: {full_results["mean_time"]:.3f}s ± {full_results["std_time"]:.3f}s')
        logger.success(f'K-anonymity: {metric_results["k_anonymity"]["mean_time"]:.3f}s')
        logger.success(f'Membership inference: {metric_results["membership_inference"]["mean_time"]:.3f}s')

    return results


def print_benchmark_summary(results: dict):
    """Print a summary of benchmark results."""

    logger.info('=' * 60)
    logger.info('📈 PERFORMANCE BENCHMARK SUMMARY')
    logger.info('=' * 60)

    logger.info(f'{"Dataset Size":<12} {"Full Eval (s)":<15} {"K-Anon (s)":<12} {"Mem Inf (s)":<12}')
    logger.info('-' * 55)

    for data in results.values():
        size = data['dataset_size']
        full_time = data['full_evaluation']['mean_time']
        k_anon_time = data['individual_metrics']['k_anonymity']['mean_time']
        mem_inf_time = data['individual_metrics']['membership_inference']['mean_time']

        logger.info(f'{size:<12} {full_time:<15.3f} {k_anon_time:<12.3f} {mem_inf_time:<12.3f}')

    # Performance insights
    logger.info('🔍 Performance Insights:')
    sizes = [data['dataset_size'] for data in results.values()]
    times = [data['full_evaluation']['mean_time'] for data in results.values()]

    if len(sizes) >= 2:
        # Calculate scaling factor
        size_ratio = sizes[-1] / sizes[0]
        time_ratio = times[-1] / times[0]
        scaling_factor = time_ratio / size_ratio

        logger.info(f'• Scaling: {time_ratio:.1f}x time for {size_ratio:.1f}x data')
        logger.info(f'• Efficiency: {scaling_factor:.2f} (closer to 1.0 is more linear)')

        if scaling_factor < 1.5:
            logger.success('✅ Good linear scaling performance')
        elif scaling_factor < 2.0:
            logger.warning('⚠️  Moderate scaling - acceptable for privacy analysis')
        else:
            logger.error('❌ Poor scaling - may need optimization')

    # Memory usage estimation
    largest_size = max(sizes)
    largest_time = max(times)
    logger.info(f'• Estimated time for 10k rows: {largest_time * (10000 / largest_size):.1f}s')

    logger.success('✅ Benchmark completed successfully!')


def save_benchmark_results(results: dict, filename: str = 'privacy_benchmark_results.json'):
    """Save benchmark results to a JSON file."""
    import json
    from datetime import datetime

    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now(tz=datetime.timezone.utc).isoformat(),
        'library_version': '1.9.0',
        'python_version': f'{pd.__version__}',
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__,
    }

    benchmark_dir = Path(__file__).parent
    filepath = benchmark_dir / filename

    with Path(filepath).open('w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.success(f'📁 Results saved to: {filepath}')


if __name__ == '__main__':
    # Run the comprehensive benchmark
    benchmark_results = run_comprehensive_benchmark()

    # Print summary
    print_benchmark_summary(benchmark_results)

    # Save results
    save_benchmark_results(benchmark_results)

    logger.info('🎯 Benchmark completed! Use these results to:')
    logger.info('  1. Monitor performance regressions in future changes')
    logger.info('  2. Identify bottlenecks for optimization')
    logger.info('  3. Set performance expectations for users')
