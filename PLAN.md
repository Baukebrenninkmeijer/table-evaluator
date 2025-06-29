# Table-Evaluator Improvement Plan

This document outlines the plan for improving the `table-evaluator` library.

## Phase 1: Refactoring and Code Cleanup

The goal of this phase is to improve the structure and maintainability of the existing codebase.

*   [ ] **Decouple Plotting from `TableEvaluator`:**
    *   [ ] Move all plotting logic from `TableEvaluator` into standalone functions in `table_evaluator/plots.py`.
    *   [ ] Restore plotting wrapper methods to `TableEvaluator` for separate access, but ensure `visual_evaluation` uses standalone functions.
    *   [ ] Ensure plots do not show automatically when `fname` is provided.
    *   [ ] Create tests for the new standalone plotting functions.

*   [ ] **Separate Data Preprocessing from `TableEvaluator`:**
    *   [ ] Create a new function or class in `table_evaluator/utils.py` to handle the data preprocessing steps currently in `TableEvaluator.__init__` (sampling, NaN filling, type inference).
    *   [ ] Update `TableEvaluator` to use this new utility.
    *   [ ] Create tests for the new preprocessing utility.

*   [ ] **Refactor `TableEvaluator.evaluate()`:**
    *   [ ] Break down the `evaluate()` method into smaller, private methods, each responsible for a specific part of the evaluation (e.g., `_calculate_statistical_metrics`, `_calculate_ml_efficacy`, `_calculate_privacy_metrics`).
    *   [ ] The main `evaluate()` method will then call these smaller methods and aggregate the results.
    *   [ ] Update existing tests for `evaluate()` to reflect the new structure.

*   [ ] **Code Cleanup:**
    *   [ ] Remove the decommissioned `plot_var_cor` function from `table_evaluator/plots.py`.
    *   [ ] Move the `if __name__ == "__main__":` block from `table_evaluator/table_evaluator.py` to a new `example.py` script in the root directory.

## Phase 2: Modernize Tooling

*   [ ] **Integrate Ruff for Linting and Formatting:**
    *   [ ] Add `ruff` to the `pyproject.toml` development dependencies.
    *   [ ] Configure `ruff` in `pyproject.toml` or a `ruff.toml` file.
    *   [ ] Run `ruff check --fix .` and `ruff format .` to apply initial linting and formatting.
    *   [ ] Update the `.github/workflows/run-tests.yml` to include a `ruff` check step.

*   [ ] **Switch to `uv` for Dependency Management (Optional but Recommended):**
    *   [ ] This is a manual step for the user, but I can provide instructions. The project already uses `pyproject.toml`, so `uv` can be used as a faster installer (`uv pip sync`).

## Phase 3: Expand Functionality

*   [ ] **Add Advanced Privacy Metrics:**
    *   [ ] Implement a Membership Inference Attack evaluation.
    *   [ ] Add a new module `privacy.py` for these metrics.
    *   [ ] Write tests for the new privacy metrics.

*   [ ] **Improve Scalability with Polars:**
    *   [ ] Refactor the core logic in `metrics.py` and `table_evaluator.py` to accept either pandas or Polars DataFrames.
    *   [ ] Use dispatching to call the appropriate library's functions.
    *   [ ] Add `polars` as a dependency.
    *   [ ] Update tests to run on both pandas and Polars DataFrames.

*   [ ] **Add Report Exporting:**
    *   [ ] Add a new method to `TableEvaluator` to export the evaluation results to an HTML file.
    *   [ ] The report should include plots and tables.
    *   [ ] Write tests for the report generation.
