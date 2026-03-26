# Makefile for latence-python SDK
# Usage: make <target>

.PHONY: install dev test test-cov test-integration lint format check clean build publish-test

# Install package in editable mode
install:
	pip install -e .

# Install with dev dependencies and pre-commit hooks
dev:
	pip install -e ".[dev]"
	pip install pre-commit pytest-cov
	pre-commit install
	@echo ""
	@echo "Development environment ready!"
	@echo "Pre-commit hooks installed."

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=latence --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated at htmlcov/index.html"

# Run integration tests against staging API
test-integration:
	@echo "Running integration tests against staging..."
	LATENCE_BASE_URL=https://staging.api.latence.ai pytest tests/integration/ -v

# Run linter and type checker
lint:
	ruff check src/latence tests/
	mypy src/latence

# Format code
format:
	ruff format src/latence tests/
	ruff check --fix src/latence tests/

# Run all checks (lint + test) - use before pushing
check: lint test
	@echo ""
	@echo "All checks passed!"

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Build package
build: clean
	pip install build
	python -m build

# Publish to Test PyPI (for testing releases)
publish-test: build
	pip install twine
	twine upload --repository testpypi dist/*
	@echo ""
	@echo "Published to Test PyPI!"
	@echo "Install with: pip install --index-url https://test.pypi.org/simple/ latence"

# Show help
help:
	@echo "Available targets:"
	@echo "  make dev            - Install dev dependencies and pre-commit hooks"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make test-integration - Run integration tests against staging"
	@echo "  make lint           - Run linter and type checker"
	@echo "  make format         - Format code with ruff"
	@echo "  make check          - Run all checks (lint + test)"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make build          - Build package"
	@echo "  make publish-test   - Publish to Test PyPI"
