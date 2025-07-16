Performance Optimization
=======================

This guide covers performance optimization techniques for the Table Evaluator, particularly when working with large datasets.

Overview
--------

The Table Evaluator includes several performance optimization features:

* **Automatic Large Dataset Detection**: Warns when datasets exceed 250,000 rows
* **Sampling Support**: Optional sampling with configurable limits
* **Optimized Algorithms**: Efficient implementations of advanced metrics
* **Memory Management**: Careful memory usage in computationally intensive operations

Large Dataset Handling
----------------------

Automatic Warnings
^^^^^^^^^^^^^^^^^^^

The system automatically detects large datasets and provides helpful warnings:

.. code-block:: python

    # When total rows > 250,000, you'll see warnings like:
    # WARNING: Large dataset detected for MMD calculation (500,000 total rows).
    # This may be slow and memory-intensive. Consider enabling sampling by setting
    # 'enable_sampling=True' and 'max_samples=5000' parameters to improve performance.

Sampling Configuration
^^^^^^^^^^^^^^^^^^^^^^

**Basic Sampling:**

.. code-block:: python

    from table_evaluator.evaluators.advanced_statistical import AdvancedStatisticalEvaluator

    evaluator = AdvancedStatisticalEvaluator(verbose=True)

    # Enable sampling for large datasets
    results = evaluator.comprehensive_evaluation(
        real_df, fake_df, numerical_columns,
        enable_sampling=True,  # Enable sampling
        max_samples=5000       # Maximum samples per dataset
    )

**Advanced Sampling Configuration:**

.. code-block:: python

    # Custom configuration for different methods
    wasserstein_config = {
        'include_2d': False,        # Skip computationally expensive 2D analysis
        'enable_sampling': True,
        'max_samples': 3000
    }

    mmd_config = {
        'kernel_types': ['rbf'],    # Use only RBF kernel for speed
        'include_multivariate': True,
        'enable_sampling': True,
        'max_samples': 3000
    }

    results = evaluator.comprehensive_evaluation(
        real_df, fake_df, numerical_columns,
        wasserstein_config=wasserstein_config,
        mmd_config=mmd_config
    )

**Direct Metric Sampling:**

.. code-block:: python

    from table_evaluator.advanced_metrics.mmd import mmd_squared, RBFKernel
    from table_evaluator.advanced_metrics.wasserstein import wasserstein_distance_2d

    # MMD with sampling
    kernel = RBFKernel()
    mmd_value = mmd_squared(
        real_data, fake_data, kernel,
        enable_sampling=True,
        max_samples=5000
    )

    # 2D Wasserstein with sampling
    distance = wasserstein_distance_2d(
        real_2d_data, fake_2d_data,
        enable_sampling=True,
        max_samples=5000
    )

Performance Recommendations
---------------------------

Dataset Size Guidelines
^^^^^^^^^^^^^^^^^^^^^^^

* **Small datasets** (< 10,000 rows): All methods without sampling
* **Medium datasets** (10,000 - 250,000 rows): All methods, sampling optional
* **Large datasets** (250,000 - 1,000,000 rows): Enable sampling, consider reducing kernel types
* **Very large datasets** (> 1,000,000 rows): Sampling required, use basic methods only

**Recommended Configurations:**

.. code-block:: python

    # Small datasets
    small_config = {
        'enable_sampling': False,
        'wasserstein_config': {'include_2d': True},
        'mmd_config': {'kernel_types': ['rbf', 'polynomial', 'linear']}
    }

    # Medium datasets
    medium_config = {
        'enable_sampling': False,
        'wasserstein_config': {'include_2d': True},
        'mmd_config': {'kernel_types': ['rbf', 'polynomial']}
    }

    # Large datasets
    large_config = {
        'enable_sampling': True,
        'max_samples': 10000,
        'wasserstein_config': {'include_2d': False},
        'mmd_config': {'kernel_types': ['rbf']}
    }

    # Very large datasets
    very_large_config = {
        'enable_sampling': True,
        'max_samples': 5000,
        'wasserstein_config': {'include_2d': False},
        'mmd_config': {'kernel_types': ['rbf'], 'include_multivariate': False}
    }

Memory Management
-----------------

Efficient Memory Usage
^^^^^^^^^^^^^^^^^^^^^^

The Table Evaluator implements several memory optimization techniques:

1. **Incremental Processing**: Large datasets are processed in chunks
2. **Memory-Efficient Algorithms**: Optimized implementations avoid unnecessary copies
3. **Garbage Collection**: Automatic cleanup of intermediate results

**Memory-Conscious Usage:**

.. code-block:: python

    # Process large datasets efficiently
    def process_large_dataset(real_df, fake_df, chunk_size=50000):
        # Process in chunks if necessary
        if len(real_df) > chunk_size:
            # Sample first, then analyze
            sampled_real = real_df.sample(n=chunk_size, random_state=42)
            sampled_fake = fake_df.sample(n=chunk_size, random_state=42)

            evaluator = AdvancedStatisticalEvaluator(verbose=True)
            results = evaluator.comprehensive_evaluation(
                sampled_real, sampled_fake, numerical_columns,
                enable_sampling=False  # Already sampled
            )
        else:
            # Process normally
            evaluator = AdvancedStatisticalEvaluator(verbose=True)
            results = evaluator.comprehensive_evaluation(
                real_df, fake_df, numerical_columns
            )

        return results

Benchmarking and Profiling
---------------------------

Performance Monitoring
^^^^^^^^^^^^^^^^^^^^^^

Monitor performance of your evaluations:

.. code-block:: python

    import time
    from table_evaluator.evaluators.advanced_statistical import AdvancedStatisticalEvaluator

    def benchmark_evaluation(real_df, fake_df, numerical_columns):
        evaluator = AdvancedStatisticalEvaluator(verbose=True)

        # Benchmark different configurations
        configs = [
            {'enable_sampling': False, 'name': 'No sampling'},
            {'enable_sampling': True, 'max_samples': 5000, 'name': 'Sampling 5k'},
            {'enable_sampling': True, 'max_samples': 10000, 'name': 'Sampling 10k'},
        ]

        for config in configs:
            start_time = time.time()

            results = evaluator.comprehensive_evaluation(
                real_df, fake_df, numerical_columns,
                enable_sampling=config.get('enable_sampling', False),
                max_samples=config.get('max_samples', 5000)
            )

            end_time = time.time()
            duration = end_time - start_time

            print(f"{config['name']}: {duration:.2f} seconds")
            print(f"  Similarity: {results['combined_metrics']['overall_similarity']:.4f}")
            print()

Troubleshooting Performance Issues
----------------------------------

Common Performance Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Out of Memory Errors**

   .. code-block:: python

       # Solution: Enable sampling
       results = evaluator.comprehensive_evaluation(
           real_df, fake_df, numerical_columns,
           enable_sampling=True,
           max_samples=1000  # Reduce sample size
       )

2. **Slow Computation**

   .. code-block:: python

       # Solution: Reduce computational complexity
       mmd_config = {
           'kernel_types': ['rbf'],  # Use only one kernel
           'include_multivariate': False  # Skip multivariate analysis
       }

       wasserstein_config = {
           'include_2d': False  # Skip 2D analysis
       }

3. **Convergence Issues**

   .. code-block:: python

       # Solution: Adjust algorithm parameters
       from table_evaluator.advanced_metrics.wasserstein import wasserstein_distance_2d

       distance = wasserstein_distance_2d(
           real_2d, fake_2d,
           max_iter=2000,  # Increase iterations
           convergence_threshold=1e-5  # Adjust threshold
       )

Performance Tuning Tips
-----------------------

1. **Use Appropriate Data Types**

   .. code-block:: python

       # Use float32 instead of float64 for large datasets
       real_df = real_df.astype({'numeric_col': 'float32'})
       fake_df = fake_df.astype({'numeric_col': 'float32'})

2. **Preprocess Data**

   .. code-block:: python

       # Remove missing values beforehand
       real_clean = real_df.dropna()
       fake_clean = fake_df.dropna()

3. **Parallel Processing**

   .. code-block:: python

       # Process multiple column pairs in parallel (for custom implementations)
       from concurrent.futures import ProcessPoolExecutor

       def process_column_pair(args):
           real_col, fake_col = args
           return wasserstein_distance_1d(real_col, fake_col)

       # Use with caution - ensure thread safety
       with ProcessPoolExecutor() as executor:
           results = list(executor.map(process_column_pair, column_pairs))

Best Practices Summary
----------------------

1. **Always enable sampling for datasets > 250,000 rows**
2. **Start with basic analysis, then add advanced metrics**
3. **Use fewer kernels for large datasets**
4. **Monitor memory usage during evaluation**
5. **Adjust sampling size based on available resources**
6. **Consider preprocessing data to remove missing values**
7. **Use appropriate data types for your dataset size**

**Example Production Configuration:**

.. code-block:: python

    def get_optimal_config(dataset_size):
        if dataset_size < 10000:
            return {
                'enable_sampling': False,
                'wasserstein_config': {'include_2d': True},
                'mmd_config': {'kernel_types': ['rbf', 'polynomial']}
            }
        elif dataset_size < 100000:
            return {
                'enable_sampling': False,
                'wasserstein_config': {'include_2d': False},
                'mmd_config': {'kernel_types': ['rbf']}
            }
        else:
            return {
                'enable_sampling': True,
                'max_samples': min(10000, dataset_size // 10),
                'wasserstein_config': {'include_2d': False},
                'mmd_config': {'kernel_types': ['rbf']}
            }

    # Usage
    total_size = len(real_df) + len(fake_df)
    config = get_optimal_config(total_size)

    results = evaluator.comprehensive_evaluation(
        real_df, fake_df, numerical_columns,
        **config
    )
