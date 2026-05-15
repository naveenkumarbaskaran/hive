# CLAUDE.md — Project Intelligence for Claude

> This file is automatically read by Claude Code / Claude in VS Code when
> working in this repository. It provides the context needed to help
> effectively without asking redundant questions.

## What is Hive?

Hive is a **multi-agent SDLC framework** where named AI agents (Scout, Penny,
Archie, Quinn, Judge, Pixel, Flow, Alex, Morgan, Devs, Reviewers) collaborate
through a shared Blackboard to turn a feature request into production-ready code.

**EPT = Empowered Product Team** — the conceptual name for the agent crew.
The Python package is `hive`, the CLI command is `hive`.

No LangChain. No CrewAI. Just Python 3.12+, httpx, and structured prompts.

## Quick Reference

| Item | Value |
|------|-------|
| Package name | `hive` (PyPI: `hive-ept`) |
| CLI command | `hive` |
| Entry point | `run_hive.py` → `main()` |
| Core package | `hive/` (12 modules + plugins subpackage) |
| Tests | `tests/test_hive.py` (~397), `tests/test_hardening.py` (~88), `tests/test_plugins.py` (~92) — 547 total |
| Python | ≥ 3.12 |
| Build system | Hatchling |
| Only runtime dep | `httpx` |
| License | Apache-2.0 |

## Repository Layout

```
hive/                     ← repo root
├── hive/                 ← core Python package
│   ├── __init__.py       ← exports + __version__
│   ├── agents.py         ← Agent dataclass, AgentRoster, DEV_POOL
│   ├── connectors.py     ← KnowledgeItem, ConnectorRegistry, git repo ingest
│   ├── crew.py           ← EPTCrew: 13-phase orchestrator (largest file ~2230 lines)
│   ├── hardening.py      ← atomic_write, file_lock, sanitize, budget, disk checks
│   ├── llm_client.py     ← LLMClient, ModelTier, auto-detect backend, retry+escalate
│   ├── memory.py         ← 3-tier memory: Agent → Team → Global
│   ├── prompts.py        ← system prompts + task templates for all agents
│   ├── sandbox.py        ← Code execution loop: syntax check, import check, test runner
│   ├── state.py          ← Blackboard, Events, checkpoints, save/load
│   ├── telemetry.py      ← CostTracker, BudgetExceeded, estimate_cost, model_context_window
│   ├── ui.py             ← ANSI terminal UI, sign-off prompts, progress dashboard
│   └── plugins/          ← Optional plugin system (protocol-based)
│       ├── __init__.py   ← Plugin exports
│       ├── base.py       ← 5 plugin protocols (Knowledge, Guidelines, System, TestData, Lifecycle)
│       ├── registry.py   ← PluginRegistry: discovery, loading, invocation helpers
│       └── examples/     ← Example plugins (SAP, company guidelines, GitHub, lifecycle)
├── run_hive.py           ← CLI entry point (argparse)
├── llm_client.py         ← backward-compat shim → hive/llm_client.py
├── tests/
│   ├── test_hive.py      ← ~397 unit tests (NO real LLM calls)
│   ├── test_hardening.py ← hardening + integration tests
│   └── test_plugins.py   ← plugin system tests (~92)
├── projects/             ← runtime output (gitignored)
├── pyproject.toml        ← package config, scripts, tool settings
├── Makefile              ← dev shortcuts
├── ARCHITECTURE.md       ← detailed design doc (READ THIS for deep understanding)
└── CONTRIBUTING.md       ← contribution guide
```

## Development Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or: make dev

# Daily workflow
make test                        # run all tests (must pass, no API calls)
make lint                        # ruff check
make fmt                         # ruff format
make run FEATURE="Build a ..."   # run hive with a feature request

# Individual commands
python3 -m pytest tests/ -v --tb=short
ruff check hive/ tests/ run_hive.py
ruff format hive/ tests/ run_hive.py
```

## Architecture at a Glance

**13-phase pipeline** in `hive/crew.py`:
1. Welcome/Intake → 2. Knowledge Ingest → 3. Research (Scout) →
4. Interview (Penny+Flow) → 5. PRD (Penny) → 6. Feasibility (Archie) →
7. Architecture+Contract (Archie) → 8. Ratification (Penny) →
9. Crew Assembly → 10. Build (Devs, layered) → 11. Integration (Quinn) →
12. Test Docs (UAT+SIT) → 13. Release (Penny+Morgan)

**Key patterns:**
- **Blackboard** (`hive/state.py`): single shared state, all agents read/write
- **ModelTier** (`hive/llm_client.py`): FAST / BALANCED / POWERFUL — agents request capability, not model names
- **Resilient LLM** (`hive/llm_client.py`): 5-attempt retry with tier escalation + 429 model rotation
- **Code Execution Sandbox** (`hive/sandbox.py`): syntax check, import check, test runner in isolated temp dir
- **Context-Aware Sandbox** (`hive/sandbox.py`): `check_file_in_context()` stages all sibling registry files alongside the target so cross-module imports resolve correctly; distinguishes internal vs external `ModuleNotFoundError`
- **Cost Tracking** (`hive/telemetry.py`): per-call metering, per-phase metrics, budget enforcement
- **Self-Reflection**: dev agents self-critique code against contract before review (FAST tier)
- **Adaptive Context**: model-aware context window budgeting (70% context / 30% task+output)
- **Project DNA**: post-run LLM extraction of lessons → global memory for future projects
- **Event bus**: agents emit Events, UI renders them (decoupled)
- **Dep-layered build**: topological sort of file deps → parallel layers
- **Memory** (`hive/memory.py`): 3-tier learning (agent → team → global)
- **Rate-limit retry**: 429-cascaded files queued for retry after cooldown
- **Request Pacing** (`hive/llm_client.py`): thread-safe `_RequestPacer` enforces min interval between LLM calls; respects server `Retry-After` headers on 429s
- **Streaming LLM** (`hive/llm_client.py`): `on_token` callback for real-time token streaming across all backends
- **URL Attachment** (`hive/connectors.py`): `--attach https://...` fetches remote URLs, auto-detects type
- **Registry-Aware Dev Context** (`hive/crew.py`): devs get full code of declared dependencies via `_dependency_context()`
- **Plugin System** (`hive/plugins/`): optional protocol-based plugins for knowledge, guidelines, systems, test data, lifecycle hooks

## Key Design Decisions — DO NOT VIOLATE

1. **No heavy frameworks** — no LangChain, CrewAI, vector stores. Only `httpx`.
2. **Tests must never call real LLMs** — mock `LLMClient` in all tests.
3. **All file I/O through `hardening.py`** — use `atomic_write()`, never raw `open().write()`.
4. **Blackboard is the single source of truth** — no agent-to-agent messaging.
5. **Environment variables for config** — never hardcode API keys or model names.
6. **Python 3.12+ required** — use modern syntax: `X | Y` unions, match/case, etc.
7. **Type hints on all function signatures**. Docstrings on all public APIs.
8. **Conventional commits**: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `ci:`.
9. **Keep docs in sync with code** — after ANY codebase change (new module, new feature,
   new env var, changed architecture, updated line counts/test counts), update ALL of:
   - `ARCHITECTURE.md` — module map, design patterns, data flow diagrams
   - `README.md` — test badge count, feature descriptions, env var tables, output structure
   - `CLAUDE.md` — module count, test count, repo layout, key patterns, env vars
   - `.github/copilot-instructions.md` — mirrors CLAUDE.md (module count, patterns, env vars)
   - `CONTRIBUTING.md` — project structure, coding guidance
   Commit doc updates as a separate `docs:` commit or include in the `feat:`/`fix:` commit.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_API_KEY` | (required) | API key for LLM backend |
| `LLM_BASE_URL` | `https://api.anthropic.com` | LLM endpoint |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `LLM_MODEL_BIG` | same as `LLM_MODEL` | For POWERFUL tier |
| `LLM_MODEL_SMALL` | same as `LLM_MODEL` | For FAST tier |
| `LLM_FORMAT` | `auto` | Force: `anthropic`, `openai`, or `auto` |
| `LLM_FALLBACK_MODELS` | (none) | Comma-separated fallback models for 429 rotation |
| `HIVE_LOG_LEVEL` | `WARNING` | Log level: DEBUG, INFO, WARNING, ERROR |
| `HIVE_MIN_DISK_MB` | `50` | Min free disk before saves |
| `HIVE_PROJECTS_DIR` | `./projects` | Where projects are saved |
| `HIVE_LLM_TIMEOUT` | `120` | HTTP timeout (seconds) for LLM requests |
| `HIVE_MAX_BUILD_WORKERS` | `2` | Max parallel file-build threads per dep layer |
| `HIVE_MAX_REVISIONS` | `3` | Max code revision cycles per file |
| `HIVE_MAX_EVENTS` | `1000` | Max events kept in Blackboard memory |
| `HIVE_MAX_GLOBAL_MEMORY` | `100` | Max global memory entries retained |
| `HIVE_BUDGET_USD` | `0` (unlimited) | Max USD spend per run |
| `HIVE_COST_PER_1K_INPUT` | model-based | Override $/1K input tokens |
| `HIVE_COST_PER_1K_OUTPUT` | model-based | Override $/1K output tokens |
| `HIVE_SANDBOX_TIMEOUT` | `30` | Max seconds per sandbox execution |
| `HIVE_SANDBOX_ENABLED` | `1` | Set `0` to disable code execution sandbox |
| `HIVE_RATE_LIMIT_COOLDOWN` | `30` | Seconds to wait before retrying rate-limited files |
| `HIVE_REQUEST_PACE_MS` | `200` | Minimum milliseconds between LLM requests (0 to disable) |
| `HIVE_PLUGINS_DIR` | `./plugins` | Directory to scan for plugin modules |

## Common Tasks for Claude

### Adding a new agent
1. Define in `hive/agents.py` (add to `AgentRoster`)
2. Add system + task prompts in `hive/prompts.py`
3. Add phase logic in `hive/crew.py`
4. Add color + emoji in `hive/ui.py`
5. Write tests (mock LLM)

### Adding a new connector type
1. Add to `ConnectorType` enum in `hive/connectors.py`
2. Implement ingestion function
3. Register in `ConnectorRegistry`
4. Add agent routing rules
5. Write tests

### Adding a new pipeline phase
1. Add phase name to the phase flow in `hive/crew.py`
2. Implement the phase method on `EPTCrew`
3. Add to `completed_phases` tracking
4. Add checkpoint save after the phase
5. Add UI rendering for the phase events
6. Write tests

### Adding a plugin
1. Create a Python file with a class that has a `meta = PluginMeta(name=...)` attribute
2. Implement one or more protocol methods: `get_knowledge()`, `get_guidelines()`, `connect()/execute()/disconnect()`, `get_test_data()`, `on_phase_start()/on_phase_end()`
3. Load via `--plugin ./your_plugin.py` or place in `HIVE_PLUGINS_DIR`
4. See `hive/plugins/examples/` for working examples
5. Write tests in `tests/test_plugins.py`

### Debugging a failed run
1. Check `projects/<slug>/docs/logbook.json` — every LLM call is logged
2. Check `projects/<slug>/checkpoints/board_latest.json` — full state
3. Resume with `hive --resume projects/<slug>/checkpoints/board_latest.json`

## Testing Conventions

- All tests in `tests/` directory
- Test files: `test_*.py`
- Run: `make test` or `python3 -m pytest tests/ -v --tb=short`
- **Never make real LLM API calls in tests** — use `MagicMock` for `LLMClient`
- Use `tmp_path` fixture for filesystem tests
- Use `monkeypatch` for env vars and module-level constants

## Code Style

- Ruff for linting and formatting (config in `pyproject.toml`)
- Line length: 100 (enforced by formatter, not linter)
- Quote style: double quotes
- Imports: isort-compatible (via ruff `I` rules)
- Dataclasses preferred over dicts for structured data
- f-strings preferred over `.format()` or `%`
