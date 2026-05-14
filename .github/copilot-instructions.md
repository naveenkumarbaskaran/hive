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
hive/                     ← core Python package (10 modules)
  __init__.py             ← exports + __version__
  agents.py               ← Agent dataclass, AgentRoster, DEV_POOL
  connectors.py           ← KnowledgeItem, ConnectorRegistry, git repo ingest
  crew.py                 ← EPTCrew: 13-phase orchestrator (~1740 lines)
  hardening.py            ← atomic_write, file_lock, sanitize, budget, disk checks
  llm_client.py           ← LLMClient, ModelTier, auto-detect backend, retry+escalate
  memory.py               ← 3-tier memory: Agent → Team → Global
  prompts.py              ← system prompts + task templates for all agents
  state.py                ← Blackboard, Events, checkpoints, save/load
  ui.py                   ← ANSI terminal UI, sign-off prompts, progress display
run_hive.py               ← CLI entry point (argparse)
llm_client.py             ← backward-compat shim → hive/llm_client.py
tests/
  test_hive.py            ← ~300 unit tests (NO real LLM calls)
  test_hardening.py       ← hardening + integration tests
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
- **Event bus**: agents emit Events, UI renders them (decoupled)
- **Dep-layered build**: topological sort of file deps → parallel layers
- **Memory** (`memory.py`): 3-tier learning (agent → team → global)

## Coding Rules — ALWAYS FOLLOW

1. **No heavy frameworks** — no LangChain, CrewAI, vector stores. Only `httpx`.
2. **Tests must never call real LLMs** — mock `LLMClient` with `MagicMock` in all tests.
3. **All file I/O through `hardening.py`** — use `atomic_write()`, never raw `open().write()`.
4. **Blackboard is the single source of truth** — no agent-to-agent messaging.
5. **Environment variables for config** — never hardcode API keys or model names.
6. **Python 3.12+** — use modern syntax: `X | Y` unions, `match/case`, etc.
7. **Type hints on all function signatures**. Docstrings on all public APIs.
8. **Conventional commits**: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `ci:`.

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
| `HIVE_MAX_REVISIONS` | `3` | Max code revision cycles per file |
| `HIVE_MAX_EVENTS` | `1000` | Max events kept in Blackboard memory |
| `HIVE_MAX_GLOBAL_MEMORY` | `100` | Max global memory entries retained |

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

### Debugging a failed run
1. Check `projects/<slug>/docs/logbook.json` — every LLM call is logged
2. Check `projects/<slug>/checkpoints/board_latest.json` — full state
3. Resume with `hive --resume projects/<slug>/checkpoints/board_latest.json`
