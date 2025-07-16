Advanced Metrics
================

.. automodule:: table_evaluator.advanced_metrics
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The advanced metrics module provides sophisticated statistical methods for comparing distributions in synthetic data evaluation. These methods go beyond traditional correlation-based metrics to provide deeper insights into distributional differences.

Key Features
------------

* **Wasserstein Distance (Earth Mover's Distance)**: Measures the minimum cost to transform one distribution into another
* **Maximum Mean Discrepancy (MMD)**: Kernel-based method for detecting distribution differences
* **Automatic Performance Optimization**: Sampling for large datasets (>250,000 rows)
* **Multiple Kernel Support**: RBF, Polynomial, and Linear kernels for MMD
* **Comprehensive Analysis**: Combined metrics with quality ratings and recommendations

Quick Start
-----------

Basic usage of advanced statistical evaluation:

.. code-block:: python

    from table_evaluator.evaluators.advanced_statistical import AdvancedStatisticalEvaluator
    import pandas as pd

    # Create evaluator
    evaluator = AdvancedStatisticalEvaluator(verbose=True)

    # Run comprehensive analysis
    results = evaluator.comprehensive_evaluation(
        real_df, fake_df, numerical_columns
    )

    # Access results
    print(f"Overall similarity: {results['combined_metrics']['overall_similarity']:.4f}")
    print(f"Quality consensus: {results['combined_metrics']['quality_consensus']}")

Wasserstein Distance
--------------------

.. automodule:: table_evaluator.advanced_metrics.wasserstein
   :members:
   :undoc-members:
   :show-inheritance:

The Wasserstein distance, also known as Earth Mover's Distance, provides a robust measure of the distance between two probability distributions. It can be interpreted as the minimum cost to transform one distribution into another.

**Key Functions:**

* ``wasserstein_distance_1d()``: Calculate 1D Wasserstein distance between two series
* ``wasserstein_distance_2d()``: Calculate 2D Wasserstein distance using Sinkhorn algorithm
* ``wasserstein_distance_df()``: Calculate distances for all numerical columns
* ``earth_movers_distance_summary()``: Generate comprehensive summary analysis

**Example Usage:**

.. code-block:: python

    from table_evaluator.advanced_metrics.wasserstein import (
        wasserstein_distance_1d,
        wasserstein_distance_df,
        earth_movers_distance_summary
    )

    # Single column analysis
    distance = wasserstein_distance_1d(real_df['column1'], fake_df['column1'])

    # All columns analysis
    distances = wasserstein_distance_df(real_df, fake_df, numerical_columns)

    # Comprehensive summary
    summary = earth_movers_distance_summary(real_df, fake_df, numerical_columns)

Maximum Mean Discrepancy (MMD)
-------------------------------

.. automodule:: table_evaluator.advanced_metrics.mmd
   :members:
   :undoc-members:
   :show-inheritance:

Maximum Mean Discrepancy is a kernel-based method for detecting differences between distributions. It uses kernel functions to map data into a reproducing kernel Hilbert space (RKHS) where mean differences can be computed.

**Key Functions:**

* ``mmd_squared()``: Calculate squared MMD between two samples
* ``mmd_column_wise()``: Calculate MMD for each column independently
* ``mmd_multivariate()``: Calculate multivariate MMD using all columns
* ``mmd_comprehensive_analysis()``: Run comprehensive MMD analysis with multiple kernels

**Kernel Classes:**

* ``RBFKernel``: Radial Basis Function (Gaussian) kernel
* ``PolynomialKernel``: Polynomial kernel
* ``LinearKernel``: Linear kernel

**Example Usage:**

.. code-block:: python

    from table_evaluator.advanced_metrics.mmd import (
        mmd_squared,
        RBFKernel,
        mmd_column_wise,
        mmd_multivariate
    )
    import numpy as np

    # Create kernel
    kernel = RBFKernel(gamma=1.0)

    # Calculate MMD between two arrays
    real_data = real_df[numerical_columns].values
    fake_data = fake_df[numerical_columns].values
    mmd_value = mmd_squared(real_data, fake_data, kernel)

    # Column-wise analysis
    column_results = mmd_column_wise(real_df, fake_df, numerical_columns)

    # Multivariate analysis
    multivariate_results = mmd_multivariate(real_df, fake_df, numerical_columns)

Advanced Statistical Evaluator
-------------------------------

.. automodule:: table_evaluator.evaluators.advanced_statistical
   :members:
   :undoc-members:
   :show-inheritance:

The ``AdvancedStatisticalEvaluator`` class provides a high-level interface for running comprehensive statistical analysis using both Wasserstein distance and MMD.

**Key Methods:**

* ``wasserstein_evaluation()``: Run comprehensive Wasserstein distance analysis
* ``mmd_evaluation()``: Run comprehensive MMD analysis
* ``comprehensive_evaluation()``: Run combined analysis with recommendations

**Example Usage:**

.. code-block:: python

    from table_evaluator.evaluators.advanced_statistical import AdvancedStatisticalEvaluator

    evaluator = AdvancedStatisticalEvaluator(verbose=True)

    # Individual analyses
    wasserstein_results = evaluator.wasserstein_evaluation(real_df, fake_df, numerical_columns)
    mmd_results = evaluator.mmd_evaluation(real_df, fake_df, numerical_columns)

    # Combined analysis
    comprehensive_results = evaluator.comprehensive_evaluation(
        real_df, fake_df, numerical_columns
    )

Performance Optimization
------------------------

Large Dataset Handling
^^^^^^^^^^^^^^^^^^^^^^^

When working with large datasets (>250,000 rows), the advanced metrics can become computationally intensive. The system automatically warns users and provides sampling options:

.. code-block:: python

    # Automatic warnings for large datasets
    results = evaluator.comprehensive_evaluation(
        real_df, fake_df, numerical_columns,
        enable_sampling=True,    # Enable sampling for performance
        max_samples=5000         # Maximum samples per dataset
    )

**Sampling Configuration:**

.. code-block:: python

    # Custom configuration for very large datasets
    wasserstein_config = {
        'include_2d': False,        # Skip 2D analysis for speed
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

Quality Ratings and Interpretation
----------------------------------

The advanced metrics provide quality ratings to help interpret results:

**Wasserstein Distance Quality Ratings:**

* **Excellent**: Mean distance < 0.1
* **Good**: Mean distance < 0.25
* **Fair**: Mean distance < 0.5
* **Poor**: Mean distance < 1.0
* **Very Poor**: Mean distance >= 1.0

**MMD Quality Ratings:**

* **Excellent**: Mean MMD < 0.01
* **Good**: Mean MMD < 0.05
* **Fair**: Mean MMD < 0.1
* **Poor**: Mean MMD < 0.2
* **Very Poor**: Mean MMD >= 0.2

**Combined Analysis:**

The comprehensive evaluation provides:

* Overall similarity score (0-1, higher is better)
* Quality consensus between methods
* Statistical significance assessment
* Actionable recommendations

Best Practices
--------------

1. **Start with Basic Analysis**: Always run basic statistical analysis first
2. **Use Appropriate Sampling**: Enable sampling for datasets >250,000 rows
3. **Choose Kernels Wisely**: RBF kernel is generally most effective for MMD
4. **Interpret Holistically**: Consider both Wasserstein and MMD results together
5. **Follow Recommendations**: Use the automated recommendations for improvement

**Example Workflow:**

.. code-block:: python

    # 1. Basic analysis
    basic_results = table_evaluator.evaluate(target_col='target')

    # 2. Advanced analysis
    advanced_evaluator = AdvancedStatisticalEvaluator(verbose=True)
    advanced_results = advanced_evaluator.comprehensive_evaluation(
        real_df, fake_df, numerical_columns,
        enable_sampling=True  # Enable for large datasets
    )

    # 3. Interpret results
    similarity = advanced_results['combined_metrics']['overall_similarity']
    recommendations = advanced_results['recommendations']

    print(f"Overall similarity: {similarity:.4f}")
    for rec in recommendations:
        print(f"- {rec}")

Troubleshooting
---------------

**Common Issues:**

1. **Memory Issues**: Enable sampling for large datasets
2. **Slow Performance**: Use fewer kernels or reduce max_samples
3. **NaN Results**: Check for missing values in input data
4. **Convergence Warnings**: Increase max_iter for 2D Wasserstein distance

**Error Handling:**

The advanced metrics include comprehensive error handling and input validation:

.. code-block:: python

    try:
        results = evaluator.comprehensive_evaluation(real_df, fake_df, numerical_columns)
    except ValueError as e:
        print(f"Input validation error: {e}")
    except Exception as e:
        print(f"Computation error: {e}")

API Reference
-------------

For detailed API documentation, see the individual module documentation:

* :py:mod:`table_evaluator.advanced_metrics.wasserstein`
* :py:mod:`table_evaluator.advanced_metrics.mmd`
* :py:mod:`table_evaluator.evaluators.advanced_statistical`
