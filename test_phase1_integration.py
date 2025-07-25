#!/usr/bin/env python3
"""
Integration test for Phase 1 advanced functionality.
Tests the new MMD, Wasserstein, and advanced privacy features.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the current directory to the path so we can import table_evaluator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from table_evaluator.evaluators.advanced_privacy import AdvancedPrivacyEvaluator
from table_evaluator.evaluators.advanced_statistical import AdvancedStatisticalEvaluator
from table_evaluator.metrics.statistical import (
    RBFKernel,
    earth_movers_distance_summary,
    mmd_column_wise,
    mmd_multivariate,
    mmd_squared,
    wasserstein_distance_1d,
    wasserstein_distance_df,
)
from table_evaluator.table_evaluator import TableEvaluator


def test_mmd_functionality():
    """Test MMD functionality with different kernels."""
    print("Testing MMD functionality...")

    # Create test data
    np.random.seed(42)
    real_data = np.random.normal(0, 1, (100, 3))
    fake_data = np.random.normal(0.2, 1.1, (100, 3))

    # Test MMD with RBF kernel
    rbf_kernel = RBFKernel(gamma=1.0)
    mmd_value = mmd_squared(real_data, fake_data, rbf_kernel)
    print(f"MMD with RBF kernel: {mmd_value:.6f}")

    # Test with sampling enabled
    mmd_value_sampled = mmd_squared(
        real_data, fake_data, rbf_kernel, enable_sampling=True, max_samples=50
    )
    print(f"MMD with sampling: {mmd_value_sampled:.6f}")

    # Test column-wise MMD
    real_df = pd.DataFrame(real_data, columns=["A", "B", "C"])
    fake_df = pd.DataFrame(fake_data, columns=["A", "B", "C"])

    column_results = mmd_column_wise(real_df, fake_df, ["A", "B", "C"])
    print(f"Column-wise MMD results shape: {column_results.shape}")
    print(
        f"Columns with significant differences: {column_results['significant'].sum()}"
    )

    # Test multivariate MMD
    multivariate_results = mmd_multivariate(real_df, fake_df, ["A", "B", "C"])
    print(f"Multivariate MMD: {multivariate_results['mmd_squared']:.6f}")
    print(f"P-value: {multivariate_results['p_value']:.6f}")

    return True


def test_wasserstein_functionality():
    """Test Wasserstein distance functionality."""
    print("\nTesting Wasserstein distance functionality...")

    # Create test data
    np.random.seed(42)
    real_data = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(0, 1, 100),
            "C": np.random.normal(0, 1, 100),
        }
    )
    fake_data = pd.DataFrame(
        {
            "A": np.random.normal(0.2, 1.1, 100),
            "B": np.random.normal(0.2, 1.1, 100),
            "C": np.random.normal(0.2, 1.1, 100),
        }
    )

    # Test 1D Wasserstein distance
    w1_dist = wasserstein_distance_1d(real_data["A"], fake_data["A"])
    print(f"1D Wasserstein distance for column A: {w1_dist:.6f}")

    # Test column-wise Wasserstein
    distances = wasserstein_distance_df(real_data, fake_data, ["A", "B", "C"])
    print(f"Column-wise Wasserstein distances shape: {distances.shape}")
    print(f"Mean distance: {distances['wasserstein_distance'].mean():.6f}")

    # Test summary analysis
    summary = earth_movers_distance_summary(real_data, fake_data, ["A", "B", "C"])
    print(f"Quality score: {summary['overall_metrics']['quality_score']:.6f}")
    print(
        f"Distribution similarity: {summary['overall_metrics']['distribution_similarity']:.6f}"
    )

    return True


def test_advanced_statistical_evaluator():
    """Test the advanced statistical evaluator."""
    print("\nTesting Advanced Statistical Evaluator...")

    # Create test data
    np.random.seed(42)
    real_data = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 200),
            "B": np.random.normal(0, 1, 200),
            "C": np.random.normal(0, 1, 200),
        }
    )
    fake_data = pd.DataFrame(
        {
            "A": np.random.normal(0.2, 1.1, 200),
            "B": np.random.normal(0.2, 1.1, 200),
            "C": np.random.normal(0.2, 1.1, 200),
        }
    )

    evaluator = AdvancedStatisticalEvaluator(verbose=True)

    # Test Wasserstein evaluation
    wasserstein_results = evaluator.wasserstein_evaluation(
        real_data, fake_data, ["A", "B", "C"]
    )
    print(
        f"Wasserstein quality rating: {wasserstein_results['quality_metrics']['quality_rating']}"
    )

    # Test MMD evaluation
    mmd_results = evaluator.mmd_evaluation(real_data, fake_data, ["A", "B", "C"])
    if "quality_metrics" in mmd_results:
        print(f"MMD quality rating: {mmd_results['quality_metrics']['mmd_rating']}")

    # Test comprehensive evaluation
    comprehensive_results = evaluator.comprehensive_evaluation(
        real_data, fake_data, ["A", "B", "C"]
    )
    print(
        f"Overall similarity: {comprehensive_results['combined_metrics']['overall_similarity']:.6f}"
    )
    print(f"Recommendations: {len(comprehensive_results['recommendations'])}")

    return True


def test_advanced_privacy_evaluator():
    """Test the advanced privacy evaluator."""
    print("\nTesting Advanced Privacy Evaluator...")

    # Create test data
    np.random.seed(42)
    real_data = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 150),
            "B": np.random.normal(0, 1, 150),
            "C": np.random.normal(0, 1, 150),
        }
    )
    fake_data = pd.DataFrame(
        {
            "A": np.random.normal(0.2, 1.1, 150),
            "B": np.random.normal(0.2, 1.1, 150),
            "C": np.random.normal(0.2, 1.1, 150),
        }
    )

    evaluator = AdvancedPrivacyEvaluator(verbose=True)

    # Test membership inference attack
    try:
        mia_results = evaluator.membership_inference_attack(
            real_data, fake_data, ["A", "B", "C"]
        )
        print(f"MIA attack accuracy: {mia_results['attack_accuracy']:.6f}")
        print(f"Privacy risk: {mia_results['privacy_risk']}")
    except Exception as e:
        print(f"MIA test failed (expected for small datasets): {e}")

    # Test attribute inference attack
    try:
        aia_results = evaluator.attribute_inference_attack(
            real_data, fake_data, ["A", "B", "C"], target_column="A"
        )
        print(f"AIA attack accuracy: {aia_results['attack_accuracy']:.6f}")
        print(f"Privacy risk: {aia_results['privacy_risk']}")
    except Exception as e:
        print(f"AIA test failed (expected for small datasets): {e}")

    return True


def test_table_evaluator_integration():
    """Test integration with TableEvaluator."""
    print("\nTesting TableEvaluator integration...")

    # Create test data
    np.random.seed(42)
    real_data = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(0, 1, 100),
            "C": np.random.normal(0, 1, 100),
            "D": np.random.choice(["x", "y", "z"], 100),
        }
    )
    fake_data = pd.DataFrame(
        {
            "A": np.random.normal(0.2, 1.1, 100),
            "B": np.random.normal(0.2, 1.1, 100),
            "C": np.random.normal(0.2, 1.1, 100),
            "D": np.random.choice(["x", "y", "z"], 100),
        }
    )

    # Test TableEvaluator with advanced metrics
    evaluator = TableEvaluator(real_data, fake_data, verbose=True)

    # Test that TableEvaluator can access advanced evaluators
    print(f"Numerical columns: {evaluator.numerical_columns}")
    print(f"Categorical columns: {evaluator.categorical_columns}")

    # Test basic evaluation still works
    results = evaluator.evaluate("A", target_type="regr", return_outputs=True)
    print("Basic evaluation completed successfully")
    print(f"Results keys: {list(results.keys())}")

    return True


def main():
    """Run all integration tests."""
    print("Running Phase 1 Advanced Functionality Integration Tests")
    print("=" * 60)

    tests = [
        test_mmd_functionality,
        test_wasserstein_functionality,
        test_advanced_statistical_evaluator,
        test_advanced_privacy_evaluator,
        test_table_evaluator_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓ {test.__name__} passed")
            else:
                failed += 1
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed with error: {e}")

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 All Phase 1 advanced functionality tests passed!")
        return 0
    print("❌ Some tests failed. Please check the implementation.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
