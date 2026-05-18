# Code Style and Conventions

## Python
- Python 3.12+
- ruff linter: E, F, W, I, UP, B, SIM rules; line-length 100; E501 ignored
- ruff formatter: double quotes
- No external runtime deps beyond httpx (by design — keep it lean)
- anthropic SDK is an optional extra, not a hard dependency

## Project structure
- `hive/` — core framework code
- `tests/` — pytest test suite (715+ tests)
- `run_hive.py` — CLI entry point
- `projects/` — generated output (gitignored)

## Key design principle
Single runtime dependency (httpx) is intentional. Don't add deps without strong justification.
