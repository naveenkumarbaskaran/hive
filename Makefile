.PHONY: help install dev test lint fmt clean run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Hive (production)
	pip install -e .

dev: ## Install with dev + anthropic extras
	pip install -e ".[all]"

test: ## Run test suite
	python3 -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	python3 -m pytest tests/ --cov=ept --cov-report=term-missing

lint: ## Run ruff linter
	ruff check ept/ tests/ run_ept.py

fmt: ## Auto-format with ruff
	ruff format ept/ tests/ run_ept.py

clean: ## Remove build artifacts
	rm -rf __pycache__ ept/__pycache__ tests/__pycache__ .pytest_cache
	rm -rf dist/ build/ *.egg-info

run: ## Run Hive (use FEATURE="..." to pass a feature request)
	python3 run_ept.py $(FEATURE)

version: ## Show version
	python3 -c "from ept import __version__; print(__version__)"
