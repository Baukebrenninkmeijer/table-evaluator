.PHONY: help install test lint format security clean docs build publish

# Default target
help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run tests with coverage"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with ruff"
	@echo "  security    Run security scans"
	@echo "  pre-commit  Run all pre-commit hooks"
	@echo "  clean       Clean build artifacts"
	@echo "  docs        Build documentation"
	@echo "  build       Build package"
	@echo "  publish     Publish to PyPI"
	@echo "  setup-dev   Set up development environment"

install:
	poetry install --with development

test:
	poetry run pytest tests/ --cov=table_evaluator --cov-report=term --cov-report=html

lint:
	poetry run ruff check .
	poetry run mypy table_evaluator/

format:
	poetry run ruff format .
	poetry run ruff check --fix .

security:
	poetry run bandit -r table_evaluator/
	poetry run safety check

pre-commit:
	poetry run pre-commit run --all-files

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs:
	poetry run sphinx-build -M html docs/source docs/build

build: clean
	poetry build

publish: build
	poetry publish

setup-dev: install
	poetry run pre-commit install
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

# Quick development cycle
dev: format lint test
	@echo "Development cycle complete!"

# CI/CD simulation
ci: lint security test
	@echo "CI pipeline simulation complete!"
