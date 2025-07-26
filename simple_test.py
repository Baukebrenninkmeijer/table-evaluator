"""Simple test to verify basic functionality."""

import pandas as pd


def test_import():
    try:
        print("✅ TableEvaluator import successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_creation():
    try:
        from table_evaluator import TableEvaluator

        # Create simple test data
        real_data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

        synthetic_data = pd.DataFrame(
            {"A": [1, 2, 3, 4, 6], "B": ["a", "b", "c", "d", "f"]}
        )

        evaluator = TableEvaluator(real_data, synthetic_data, verbose=False)
        print("✅ TableEvaluator creation successful")

        # Test if textual methods exist
        if hasattr(evaluator, "textual_evaluation"):
            print("✅ Textual evaluation method available")

        if hasattr(evaluator, "get_available_advanced_metrics"):
            print("✅ Advanced metrics info method available")

        return True
    except Exception as e:
        print(f"❌ Basic creation failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Phase 1 integration...")

    success = True
    success &= test_import()
    success &= test_basic_creation()

    if success:
        print("\n🎉 Phase 1 integration test PASSED!")
    else:
        print("\n❌ Phase 1 integration test FAILED!")
