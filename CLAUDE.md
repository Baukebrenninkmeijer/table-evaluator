# AI Assistant Development Workflow Guidelines

This document defines the standard workflow for AI-assisted development tasks within this project.

## Project Overview: Table Evaluator

This is a Python package called "table-evaluator" (v1.9.0) for evaluating synthetic tabular data quality against real data. It provides comprehensive metrics and analysis tools for comparing real and synthetic datasets.

### Key Features
- **Advanced statistical metrics**: Wasserstein Distance, Maximum Mean Discrepancy (MMD)
- **Privacy analysis**: k-anonymity, l-diversity, membership inference attacks
- **Plugin architecture**: Extensible framework for custom metrics
- **Comprehensive visualization**: Statistical plots and comparison charts

### Project Structure
```
table_evaluator/
├── table_evaluator.py       # Main API entry point
├── metrics/                  # Core mathematical functions
│   ├── wasserstein.py       # Distribution comparison
│   ├── mmd.py               # Kernel-based analysis
│   └── privacy_attacks.py   # Privacy risk assessment
├── evaluators/              # Business logic layer
│   ├── advanced_statistical.py
│   ├── advanced_privacy.py
│   └── ml_evaluator.py
└── plugins/                 # Extensibility framework
```

### Development Environment Setup
- **Python 3.10+** required
- **Dependencies**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Testing**: pytest with comprehensive test suite
- **Code Quality**: ruff (linting), black (formatting), bandit (security)
- **Build System**: pyproject.toml with modern Python packaging
- **Package Management**: uv for dependency management
- **Version Control**: semantic versioning with pre-commit hooks

## Development Workflow

### Code Changes and Testing
- **Always run tests** after implementing code changes
- If no tests exist for modified code, ask whether to create them before proceeding
- **Test commands**:
  - Full test suite: `python -m pytest` or `make test`
  - Specific test file: `python -m pytest tests/test_specific.py`
  - Coverage report: `python -m pytest --cov=table_evaluator --cov-report=html`
- **Code quality checks**:
  - Linting: `python -m ruff check table_evaluator/`
  - Formatting: `python -m black table_evaluator/ tests/`
  - Security: `python -m bandit -r table_evaluator/`
  - All quality checks: `make lint` or `pre-commit run --all-files`

### Branch and Worktree Management
- Verify you're in the correct worktree: `git worktree list`.
- Verify you're in the correct branch: `git branch`.
- If no worktree and branch exists for this task, create them.
- Operate within the appropriate branch worktree for all development tasks
- Ensure branch isolation for different features/tasks

### Collaboration and Integration
1. **Branch Creation**: Check if task branch exists; create if needed
2. **Code Push**: Push changes to feature branch upon completion
3. **Pull Request**: Create PR using GitHub CLI (`gh pr create`)
4. **CI/CD Monitoring**:
   - Monitor pipeline status: `gh pr checks`
   - Address failures promptly
   - Run local equivalents when possible (pre-commit, build, install)
5. **Quality Gates**: Ensure all checks pass before requesting review

## Local Development Best Practices

### Python-Specific Guidelines
- **Type hints**: Use type annotations for all new functions and methods
- **Docstrings**: Follow NumPy/SciPy docstring conventions
- **Error handling**: Use appropriate exceptions with descriptive messages
- **Performance**: Consider memory usage for large datasets (>100k rows)
- **Compatibility**: Ensure code works with pandas backend

### Code Quality Standards
- **PEP 8 compliance**: Follow Python style guidelines
- **Import organization**: Use ruff for sorting and organizing imports
- **Test coverage**: Aim for >90% coverage on new code
- **Documentation**: Include usage examples in docstrings
- **Backward compatibility**: Maintain API compatibility when possible

### Pre-commit Workflow
- Run pre-commit hooks before committing: `pre-commit run --all-files`
- Verify builds/installations locally to catch CI/CD issues early
- Use project's Makefile targets for standardized operations

## Recent Major Achievement: Dython Dependency Replacement (v1.9.0)

### Background
Successfully replaced the `dython` library with native implementations for statistical association metrics, achieving 0.000% accuracy difference while improving performance and reducing external dependencies.

### Key Implementation Insights

#### Statistical Functions Implemented:
1. **Cramer's V**: Must use `correction=False` in scipy's chi2_contingency to match dython behavior
2. **Theil's U**: Shannon entropy-based uncertainty coefficient (asymmetric)
3. **Correlation Ratio**: ANOVA-based categorical-numerical association
4. **Associations Matrix**: Preserves Pearson correlation signs (allows negative values)

#### Critical Implementation Details:
- **Error Handling**: Use specific exception types (`ValueError`, `RuntimeError`, `np.linalg.LinAlgError`, `TypeError`)
- **Input Validation**: Convert to pandas Series, handle NaN values, validate lengths
- **Numerical Stability**: Add epsilon constants, proper clamping for edge cases
- **Statistical Accuracy**: Comprehensive validation against reference implementations

#### Files Created/Modified:
- `table_evaluator/association_metrics.py` (149 lines) - Native implementations
- `tests/test_dython_compatibility.py` (375 lines) - Comprehensive test suite
- `benchmark_dython_comparison.py` (372 lines) - Validation benchmark
- `table_evaluator/data/data_converter.py` - Eliminated code duplication

### Validation Results:
- **Statistical Accuracy**: 0.000% difference vs original dython
- **Performance**: 0.72x average speedup
- **Test Coverage**: 81% with 22 comprehensive test cases
- **Production Ready**: Extensive error handling and edge case testing

### Testing Strategy Used:
- Edge cases: empty data, single categories, NaN values, large datasets
- Statistical accuracy validation against manual calculations
- Performance benchmarking with multiple dataset sizes
- Symmetry/asymmetry property testing where applicable

This achievement demonstrates successful dependency reduction while maintaining statistical accuracy through careful native implementation and comprehensive validation.

## Gemini CLI Integration

### Overview
You can offload individual tasks to the Gemini CLI, which you can call with the `gemini` command.
Always use YOLO mode (`-y`) for trusted development environments to streamline the workflow.

### Command Reference
```bash
gemini -p "your prompt here" -y
```

### Proven Use Cases & Best Practices

#### ✅ **Highly Effective Tasks**
- **Project Status Analysis**: Comprehensive project reports and progress summaries
- **CI/CD Pipeline Optimization**: Performance analysis and configuration improvements
- **Code Analysis & Optimization**: Benchmark suite enhancements and refactoring suggestions
- **Documentation Generation**: Technical documentation and README sections

#### ⚠️ **Tasks with Limitations**
- **Large Codebase Analysis**: Tasks with `--all_files` may timeout after 2 minutes
- **Complex Multi-step Operations**: Break into smaller, focused tasks
- **Import Path Dependencies**: May require manual corrections for project-specific imports

### Recommended Workflow

1. **Start with Focused Tasks** (< 500 words prompts)
   ```bash
   gemini -p "Analyze CI pipeline and suggest 3 specific optimizations" -y
   ```

2. **Use YOLO Mode for Trusted Environments**
   ```bash
   gemini -p "Optimize benchmark performance tests" -y
   ```

3. **Validate and Test Gemini Suggestions**
   - Always run local tests after Gemini modifications
   - Check import paths and dependencies
   - Verify configuration changes

4. **Combine with Manual Implementation**
   - Use Gemini for analysis and recommendations
   - Implement complex changes manually with Gemini insights
   - Leverage Gemini's optimization suggestions

### Options Reference
```
Options:
  -m, --model                    Model      [string] [default: "gemini-2.5-pro"]
  -p, --prompt                   Prompt. Appended to input on stdin (if any).
                                                                        [string]
  -s, --sandbox                  Run in sandbox?                       [boolean]
      --sandbox-image            Sandbox image URI.                     [string]
  -d, --debug                    Run in debug mode?   [boolean] [default: false]
  -a, --all_files                Include ALL files in context?
                                                      [boolean] [default: false]
      --show_memory_usage        Show memory usage in status bar
                                                      [boolean] [default: false]
  -y, --yolo                     Automatically accept all actions (aka YOLO
                                 mode, see
                                 https://www.youtube.com/watch?v=xvFZjo5PgG0 for
                                 more details)?       [boolean] [default: false]
      --telemetry                Enable telemetry? This flag specifically
                                 controls if telemetry is sent. Other
                                 --telemetry-* flags set specific values but do
                                 not enable telemetry on their own.    [boolean]
      --telemetry-target         Set the telemetry target (local or gcp).
                                 Overrides settings files.
                                              [string] [choices: "local", "gcp"]
      --telemetry-otlp-endpoint  Set the OTLP endpoint for telemetry. Overrides
                                 environment variables and settings files.
                                                                        [string]
      --telemetry-log-prompts    Enable or disable logging of user prompts for
                                 telemetry. Overrides settings files.  [boolean]
  -c, --checkpointing            Enables checkpointing of file edits
                                                      [boolean] [default: false]
  -v, --version                  Show version number                   [boolean]
  -h, --help                     Show help                             [boolean]
  ```
