"""Quick test to verify Phase 1 advanced functionality integration."""

import numpy as np
import pandas as pd
from table_evaluator import TableEvaluator


def test_advanced_functionality():
    """Test the new advanced functionality."""

    # Create sample data
    np.random.seed(42)

    # Real data
    real_data = pd.DataFrame(
        {
            "numerical_1": np.random.normal(0, 1, 1000),
            "numerical_2": np.random.exponential(2, 1000),
            "categorical_1": np.random.choice(["A", "B", "C"], 1000),
            "categorical_2": np.random.choice(["X", "Y"], 1000),
            "target": np.random.choice([0, 1], 1000),
        }
    )

    # Synthetic data (slightly different distribution)
    synthetic_data = pd.DataFrame(
        {
            "numerical_1": np.random.normal(0.1, 1.1, 1000),
            "numerical_2": np.random.exponential(2.2, 1000),
            "categorical_1": np.random.choice(["A", "B", "C"], 1000, p=[0.4, 0.3, 0.3]),
            "categorical_2": np.random.choice(["X", "Y"], 1000, p=[0.6, 0.4]),
            "target": np.random.choice([0, 1], 1000),
        }
    )

    print("Creating TableEvaluator...")
    evaluator = TableEvaluator(
        real_data,
        synthetic_data,
        cat_cols=["categorical_1", "categorical_2"],
        verbose=True,
    )

    print("\n1. Testing advanced statistical evaluation...")
    try:
        advanced_stats = evaluator.advanced_statistical_evaluation()
        print("✅ Advanced statistical evaluation successful")

        # Check Wasserstein results
        if "wasserstein" in advanced_stats:
            wass_quality = advanced_stats["wasserstein"].get("quality_metrics", {})
            print(
                f"   - Mean Wasserstein distance: {wass_quality.get('mean_wasserstein_p1', 'N/A')}"
            )

        # Check MMD results
        if "mmd" in advanced_stats:
            mmd_quality = advanced_stats["mmd"].get("quality_metrics", {})
            print(
                f"   - MMD quality score: {mmd_quality.get('overall_quality_score', 'N/A')}"
            )

    except Exception as e:
        print(f"❌ Advanced statistical evaluation failed: {e}")

    print("\n2. Testing advanced privacy evaluation...")
    try:
        advanced_privacy = evaluator.advanced_privacy_evaluation()
        print("✅ Advanced privacy evaluation successful")

        # Check results
        overall = advanced_privacy.get("overall_assessment", {})
        print(f"   - Overall risk level: {overall.get('overall_risk_level', 'N/A')}")
        print(f"   - Privacy score: {overall.get('privacy_score', 'N/A')}")

    except Exception as e:
        print(f"❌ Advanced privacy evaluation failed: {e}")

    print("\n3. Testing plugin system...")
    try:
        available_metrics = evaluator.get_available_advanced_metrics()
        print("✅ Plugin system accessible")
        print(
            f"   - Built-in advanced metrics: {len(available_metrics['built_in_advanced'])}"
        )
        print(f"   - Plugin metrics: {len(available_metrics['plugins'])}")

    except Exception as e:
        print(f"❌ Plugin system failed: {e}")

    print("\n4. Testing comprehensive evaluation...")
    try:
        comprehensive = evaluator.comprehensive_advanced_evaluation(
            target_col="target",
            include_basic=False,  # Skip basic for speed
            include_advanced_statistical=True,
            include_advanced_privacy=True,
            include_plugins=False,  # Skip plugins for speed
        )
        print("✅ Comprehensive advanced evaluation successful")
        print(f"   - Evaluation components: {list(comprehensive.keys())}")

    except Exception as e:
        print(f"❌ Comprehensive evaluation failed: {e}")

    print("\nPhase 1 integration test completed!")


if __name__ == "__main__":
    test_advanced_functionality()
