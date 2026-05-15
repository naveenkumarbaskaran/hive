# Copilot Instructions — Hive Project

## What is Hive?

Hive is a **multi-agent SDLC framework** where named AI agents (Scout, Penny,
Archie, Quinn, Judge, Pixel, Flow, Alex, Morgan, Devs, Reviewers) collaborate
through a shared Blackboard to turn a feature request into production-ready code.

- **Package:** `hive` (PyPI: `hive-ept`)
- **CLI:** `hive`
- **Python:** ≥ 3.12
- **Only runtime dep:** `httpx`
- **No LangChain. No CrewAI.** Just Python, httpx, and structured prompts.

## Repository Layout

```
hive/                     ← core Python package (13 modules)
  __init__.py             ← exports + __version__
  agents.py               ← Agent dataclass, AgentRoster, DEV_POOL
  connectors.py           ← KnowledgeItem, ConnectorRegistry, git repo ingest, brownfield codebase_index
  crew.py                 ← EPTCrew: 13-phase orchestrator (~2572 lines)
  dashboard.py            ← SSE-based web dashboard (DashboardServer), real-time progress
  hardening.py            ← atomic_write, file_lock, sanitize, budget, disk checks
  llm_client.py           ← LLMClient, ModelTier, auto-detect backend, retry+escalate, multi-provider
  memory.py               ← 3-tier memory: Agent → Team → Global
  prompts.py              ← system prompts + task templates for all agents
  sandbox.py              ← Code execution loop: syntax check, import check, test runner, multi-file test execution
  state.py                ← Blackboard, Events, checkpoints, save/load, approved_signatures()
  telemetry.py            ← CostTracker, BudgetExceeded, estimate_cost, model_context_window
  ui.py                   ← ANSI terminal UI, sign-off prompts, progress dashboard, build_preview()
  plugins/                ← Optional plugin system (protocol-based)
    base.py               ← 5 plugin protocols (Knowledge, Guidelines, System, TestData, Lifecycle)
    registry.py           ← PluginRegistry: discovery, loading, invocation helpers
    examples/             ← Example plugins (SAP, company guidelines, GitHub, lifecycle)
run_hive.py               ← CLI entry point (argparse)
llm_client.py             ← backward-compat shim → hive/llm_client.py
tests/
  test_hive.py            ← ~536 unit tests (NO real LLM calls)
  test_hardening.py       ← ~88 hardening + integration tests
  test_plugins.py         ← ~92 plugin system tests (715 total)
```

## Architecture

**13-phase pipeline** in `hive/crew.py`:
1. Welcome/Intake → 2. Knowledge Ingest → 3. Research (Scout) →
4. Interview (Penny+Flow) → 5. PRD (Penny) → 6. Feasibility (Archie) →
7. Architecture+Contract (Archie) → 8. Ratification (Penny) →
9. Crew Assembly → 10. Build (Devs, layered) → 11. Integration (Quinn) →
12. Test Docs (UAT+SIT) → 13. Release (Penny+Morgan)

**Key patterns:**
- **Blackboard** (`state.py`): single shared state, all agents read/write
- **ModelTier** (`llm_client.py`): FAST / BALANCED / POWERFUL — agents request capability, not model names
- **Resilient LLM** (`llm_client.py`): 5-attempt retry with tier escalation + 429 model rotation
- **Code Execution Sandbox** (`sandbox.py`): syntax check, import check, test runner in isolated temp dir
- **Context-Aware Sandbox** (`sandbox.py`): `check_file_in_context()` stages all sibling registry files alongside the target so cross-module imports resolve correctly; distinguishes internal vs external `ModuleNotFoundError`
- **Cost Tracking** (`telemetry.py`): per-call metering, per-phase metrics, budget enforcement
- **Self-Reflection**: dev agents self-critique code against contract before review (FAST tier)
- **Adaptive Context**: model-aware context window budgeting (70% context / 30% task+output)
- **Project DNA**: post-run LLM extraction of lessons → global memory for future projects
- **Event bus**: agents emit Events, UI renders them (decoupled)
- **Dep-layered build**: topological sort of file deps → parallel layers
- **Memory** (`memory.py`): 3-tier learning (agent → team → global)
- **Rate-limit retry**: 429-cascaded files queued for retry after cooldown
- **Request Pacing** (`llm_client.py`): thread-safe `_RequestPacer` enforces min interval between LLM calls; respects server `Retry-After` headers on 429s
- **Streaming LLM** (`llm_client.py`): `on_token` callback for real-time token streaming across all backends
- **URL Attachment** (`connectors.py`): `--attach https://...` fetches remote URLs, auto-detects type
- **Registry-Aware Dev Context** (`crew.py`): devs get full code of declared dependencies via `_dependency_context()`
- **Contract Amendment Rebuild** (`crew.py`): Judge's AMEND_CONTRACT verdict applies amendment, refreshes cache, and triggers a full rebuild of the file
- **Contract-Aware Review** (`crew.py` + `prompts.py`): Quinn receives contract specs for the file under review, including dependency interfaces and amendments
- **Dep-Blocker Guard** (`crew.py`): `_downgrade_dep_blockers()` auto-downgrades FAIL verdicts from reviewers when the only blockers reference unapproved contract dependencies guaranteed to exist in later build layers
- **Quality Playbook** (`prompts.py`): OWASP security checklist, SOLID principles, DPP/PII checks, input validation, and secret hygiene injected into all agent prompts
- **PII Scanner** (`sandbox.py`): `scan_pii()` regex-based static analysis detects hardcoded secrets, PII in logs, eval/exec, unsafe deserialization, and shell injection
- **Regression Test Generation** (`prompts.py` + `crew.py`): Quinn auto-generates executable regression test suite including boundary, negative, security, and PII tests
- **Test Execution Feedback Loop** (`sandbox.py` + `crew.py`): `run_test_in_context()` stages all project files and runs real pytest during build; `_test_execution_check()` feeds failures back to dev for fixing (up to `MAX_TEST_FIX_ATTEMPTS` rounds)
- **Integration Test Fix Loop** (`crew.py`): `_integration_test_fix_loop()` isolates per-file test failures after `run_code_checks`, routes to responsible devs, and re-runs up to `MAX_INTEGRATION_FIXES` rounds
- **Plugin System** (`plugins/`): optional protocol-based plugins for knowledge, guidelines, systems, test data, lifecycle hooks
- **Context Compression** (`state.py`): `Blackboard.approved_signatures()` extracts only function/class/import/assignment signatures from approved files (70-80% smaller); used in self-reflection, test-fix, and integration-fix prompts for token efficiency
- **Multi-Provider LLM** (`llm_client.py`): per-tier provider routing — different base URLs, API keys, and format hints per capability tier via `resolve_endpoint(tier)`; thread-safe via thread-local storage for parallel builds
- **Interactive Build Mode** (`ui.py` + `crew.py`): `--interactive` CLI flag enables `build_preview()` — user can [a]pprove / [f]eedback / [s]kip each file during build; feedback loops back to dev agent
- **Live Dashboard** (`dashboard.py`): SSE-based web dashboard (`DashboardServer`); `--dashboard [PORT]` flag (default 8765); real-time phase progress, file status, cost tracking, event log; stdlib only (no new deps)
- **Brownfield Mode** (`connectors.py` + `crew.py`): `--modify PATH` for existing codebase modification; `codebase_index(root)` does AST-based Python signature extraction + regex-based JS/TS/Java/Go analysis; auto-attaches structure to architecture prompt

## Coding Rules — ALWAYS FOLLOW

1. **No heavy frameworks** — no LangChain, CrewAI, vector stores. Only `httpx`.
2. **Tests must never call real LLMs** — mock `LLMClient` with `MagicMock` in all tests.
3. **All file I/O through `hardening.py`** — use `atomic_write()`, never raw `open().write()`.
4. **Blackboard is the single source of truth** — no agent-to-agent messaging.
5. **Environment variables for config** — never hardcode API keys or model names.
6. **Python 3.12+** — use modern syntax: `X | Y` unions, `match/case`, etc.
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

## Code Style

- **Formatter/Linter:** Ruff (config in `pyproject.toml`)
- **Line length:** 100
- **Quotes:** double quotes
- **Imports:** isort-compatible (ruff `I` rules)
- **Data structures:** dataclasses preferred over dicts
- **Strings:** f-strings preferred over `.format()` or `%`

## Testing

```bash
make test          # or: python3 -m pytest tests/ -v --tb=short
make lint          # ruff check
make fmt           # ruff format
```

- All tests in `tests/` directory, files named `test_*.py`
- **Never make real LLM API calls** — use `MagicMock` for `LLMClient`
- Use `tmp_path` fixture for filesystem tests
- Use `monkeypatch` for env vars and module-level constants

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
| `HIVE_MAX_INTEGRATION_FIXES` | `2` | Max rounds of integration test fix loop |
| `HIVE_MAX_TEST_FIX_ATTEMPTS` | `2` | Max attempts to fix test failures during build |
| `LLM_BASE_URL_FAST` | (none) | Per-tier LLM endpoint for FAST tier |
| `LLM_BASE_URL_BALANCED` | (none) | Per-tier LLM endpoint for BALANCED tier |
| `LLM_BASE_URL_POWERFUL` | (none) | Per-tier LLM endpoint for POWERFUL tier |
| `LLM_API_KEY_FAST` | (none) | Per-tier API key for FAST tier |
| `LLM_API_KEY_BALANCED` | (none) | Per-tier API key for BALANCED tier |
| `LLM_API_KEY_POWERFUL` | (none) | Per-tier API key for POWERFUL tier |
| `LLM_FORMAT_FAST` | (none) | Per-tier format hint for FAST tier |
| `LLM_FORMAT_BALANCED` | (none) | Per-tier format hint for BALANCED tier |
| `LLM_FORMAT_POWERFUL` | (none) | Per-tier format hint for POWERFUL tier |

## Common Tasks

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
