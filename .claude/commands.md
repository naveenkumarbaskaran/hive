# Hive — Claude Code Commands
#
# These are pre-approved commands Claude can run in this project.
# They correspond to Makefile targets and common dev workflows.

# Run tests
/test: make test

# Run tests with coverage
/test-cov: make test-cov

# Lint check
/lint: make lint

# Auto-format
/fmt: make fmt

# Show version
/version: make version

# Clean build artifacts
/clean: make clean

# Install in dev mode
/dev-setup: python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[all]"

# Run a specific test file
/test-file: python3 -m pytest $1 -v --tb=short

# Run a specific test by name
/test-name: python3 -m pytest tests/ -v --tb=short -k "$1"
