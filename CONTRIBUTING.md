# Contributing to Hive

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/naveenkumarbaskaran/hive.git
cd hive

# Install in dev mode with all extras
make dev

# Verify everything works
make test
```

## Project Structure

```
hive/
├── hive/                  # Core package (12 modules + plugins subpackage)
│   ├── __init__.py       #   Package exports + version
│   ├── llm_client.py     #   Pluggable LLM connector
│   ├── agents.py         #   Agent definitions + personalities
│   ├── state.py          #   Blackboard, events, checkpoints
│   ├── crew.py           #   13-phase orchestrator
│   ├── prompts.py        #   All LLM prompt templates
│   ├── ui.py             #   Terminal UI (ANSI)
│   ├── connectors.py     #   External knowledge ingestion
│   ├── memory.py         #   3-tier learning memory
│   ├── hardening.py      #   Production utilities
│   ├── sandbox.py        #   Code execution loop (syntax, import, test)
│   ├── telemetry.py      #   Cost tracking, budget enforcement
│   └── plugins/          #   Optional plugin system (protocol-based)
│       ├── base.py       #     5 plugin protocols
│       ├── registry.py   #     Plugin discovery, loading, lifecycle
│       └── examples/     #     Example plugins (SAP, guidelines, GitHub, lifecycle)
├── tests/                # Test suite (~531 tests)
│   ├── test_hive.py       #   Core functionality tests
│   ├── test_hardening.py #   Hardening & safety tests
│   └── test_plugins.py   #   Plugin system tests
├── run_hive.py            # CLI entry point
├── ARCHITECTURE.md       # Detailed architecture docs
└── pyproject.toml        # Package configuration
```

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Write code** following existing patterns:
   - Place agent logic in `hive/agents.py`
   - Place prompt templates in `hive/prompts.py`
   - Place state/data structures in `hive/state.py`
   - Use `hive/hardening.py` utilities for file I/O
   - Use `hive/sandbox.py` for code execution validation
   - Use `hive/telemetry.py` for cost tracking integration
   - Use `on_token` callback in `LLMClient.chat()` for streaming output
   - Use `_dependency_context()` in `hive/crew.py` when building dev prompts
   - Use `ConnectorRegistry.ingest_url()` for URL-based attachments
   - Use `hive/plugins/` for optional plugin-based extensions (never modify core for plugins)

3. **Add tests** in `tests/`:
   - Tests must not make real LLM API calls
   - Mock the LLM client for agent behavior tests
   - Use `pytest` fixtures and `tmp_path`

4. **Run checks**:
   ```bash
   make test        # All tests pass
   make lint        # No lint errors
   ```

5. **Submit a PR** against `main`

## Code Style

- **Python 3.12+** — use modern syntax (match/case, `X | Y` unions, etc.)
- **Type hints** on all function signatures
- **Docstrings** on all public classes and functions
- **No heavy dependencies** — the core has only `httpx` as a required dep
- Follow patterns in existing code (dataclasses, f-strings, etc.)

## Adding a New Agent

1. Define the agent in `hive/agents.py` (add to `AgentRoster`)
2. Add system + task prompts in `hive/prompts.py`
3. Add the phase logic in `hive/crew.py`
4. Add color + emoji in `hive/ui.py`
5. Write tests covering the new phase

## Adding a New Connector

1. Add the connector type to `ConnectorType` enum in `hive/connectors.py`
2. Implement the ingestion function
3. Register it in `ConnectorRegistry`
4. Add tests

**URL-based connectors:** `--attach https://...` fetches remote URLs automatically.
See `is_url()`, `fetch_url()`, and `ConnectorRegistry.ingest_url()` in `hive/connectors.py`.
New URL logic auto-detects type from extension or Content-Type header.

## Adding a Plugin

1. Create a Python file with a class that has a `meta = PluginMeta(name=...)` attribute
2. Implement one or more protocol methods:
   - `get_knowledge(ctx)` — return domain knowledge items
   - `get_guidelines(ctx)` — return coding rules as text
   - `connect(ctx)` / `execute(action, params)` / `disconnect()` — system connector
   - `get_test_data(ctx, schema)` — return test fixtures
   - `on_phase_start(phase, ctx)` / `on_phase_end(phase, ctx)` — lifecycle hooks
3. Load via `--plugin ./your_plugin.py` or place in `HIVE_PLUGINS_DIR` for auto-discovery
4. See `hive/plugins/examples/` for working examples
5. Write tests in `tests/test_plugins.py`

## Commit Messages

Use conventional commits:

```
feat: add new agent for security review
fix: handle empty PRD in architecture phase
test: add coverage for checkpoint resume
docs: update ARCHITECTURE.md with memory system
```

## Reporting Issues

- Use GitHub Issues
- Include: Python version, OS, error traceback, steps to reproduce
- For security issues, see [SECURITY.md](SECURITY.md)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
