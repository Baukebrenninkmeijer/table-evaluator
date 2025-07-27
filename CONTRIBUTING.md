# Contributing to Table Evaluator

We welcome contributions to the Table Evaluator project! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- uv for dependency management
- Git

### Setting up the Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/table-evaluator.git
   cd table-evaluator
   ```

2. **Set up the development environment:**
   ```bash
   make setup-dev
   ```

   This will:
   - Install all dependencies including development tools
   - Set up pre-commit hooks
   - Configure the environment

3. **Verify the setup:**
   ```bash
   make test
   ```

## Development Workflow

### Making Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards (see below).

3. **Run the development cycle:**
   ```bash
   make dev
   ```

   This will format, lint, and test your code.

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

### Coding Standards

We use several tools to maintain code quality:

- **Ruff**: For linting and formatting
- **MyPy**: For type checking
- **Bandit**: For security scanning
- **Pytest**: For testing

#### Code Formatting

- We use Ruff for code formatting
- Line length is set to 88 characters
- Use double quotes for strings
- Follow PEP 8 guidelines

#### Type Hints

- Add type hints to all new functions and methods
- Use modern typing syntax (e.g., `list[str]` instead of `List[str]`)

#### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md if your changes affect the public API

### Testing

- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Follow the existing test structure in the `tests/` directory

#### Running Tests

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/test_specific.py

# Run with coverage report
poetry run pytest --cov=table_evaluator --cov-report=html
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. They run automatically on `git commit` and include:

- Code formatting with Ruff
- Linting checks
- Security scanning with Bandit
- Type checking with MyPy
- Basic file checks (trailing whitespace, etc.)

You can run all hooks manually:
```bash
make pre-commit
```

## Pull Request Process

1. **Ensure your code passes all checks:**
   ```bash
   make ci
   ```

2. **Update documentation** if needed.

3. **Create a pull request** with:
   - Clear title and description
   - Reference any related issues
   - Include tests for new functionality
   - Update CHANGELOG.md if applicable

4. **Wait for review** and address any feedback.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment for all contributors

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Table Evaluator version
- Minimal code example to reproduce the issue
- Full error traceback

## Security Issues

For security vulnerabilities, please email the maintainers directly rather than creating a public issue.

## Questions?

If you have questions about contributing, feel free to:

- Open a discussion on GitHub
- Reach out to the maintainers
- Check existing issues and discussions

Thank you for contributing to Table Evaluator!
