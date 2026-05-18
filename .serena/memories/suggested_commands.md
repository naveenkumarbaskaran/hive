# Suggested Commands

```bash
# Install
pip install -e .           # production
make dev                   # with dev + anthropic extras (pip install -e ".[all]")

# Run
make run FEATURE="Build a URL shortener REST API with rate limiting"
# or directly:
python3 run_hive.py "your feature request"
# or via CLI after install:
hive "your feature request"

# Test
make test                  # pytest tests/ -v --tb=short
make test-cov              # with coverage report

# Lint / format
make lint                  # ruff check
make fmt                   # ruff format

# Clean
make clean
```
