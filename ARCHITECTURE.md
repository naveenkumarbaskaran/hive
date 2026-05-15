# EPT — Empowered Product Team: Architecture

## Overview

EPT is a lightweight multi-agent SDLC framework. Named AI agents with distinct
personalities collaborate through a shared Blackboard to take a feature request
from idea to production-ready code — with the user in the loop at every gate.

No heavy frameworks. No LangChain. No CrewAI dependency. Just Python, httpx,
and structured prompts.

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_hive.py (CLI)                         │
│                     argparse + entry point                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     hive/crew.py (Orchestrator)                  │
│                                                                 │
│  Phase 0: Welcome/Intake (name, role, end-user, as-is)          │
│      │                                                          │
│  Phase 0.5: Knowledge Ingest (files, folders, API specs)        │
│      │                                                          │
│  Phase 1: Research ──► Phase 2: Interview ──► Phase 3: PRD     │
│      │                                           │  ↕ signoff  │
│      ▼                                           ▼              │
│  Phase 4: Feasibility ──► Phase 5: Architecture ──► Phase 6    │
│      ↕ signoff                  ↕ signoff         Ratification  │
│                                                      │          │
│  Phase 7: Crew Assembly ──► Phase 8: Build (layered) │          │
│      │                         │ ┌────────────────┐  │          │
│      │                         │ │ Dev → Review    │  │          │
│      │                         │ │  (parallel/layer│  │          │
│      │                         │ │   → Revise      │  │          │
│      │                         │ │   → Judge?)     │  │          │
│      │                         │ └────────────────┘  │          │
│      ▼                         ▼                     ▼          │
│  Phase 9: Integration ──► Phase 10: Test Docs ──► Phase 11:    │
│                              (UAT.md + SIT.md)      Release     │
│                                                   (with Handover│
│                                                    + Packaging  │
│                                                    + Delivery)  │
└────────┬───────────────────────┬─────────────────┬──────────────┘
         │                       │                 │
         ▼                       ▼                 ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│ hive/agents  │    │   hive/state      │    │  hive/ui      │
│             │    │  (Blackboard)    │    │ (Terminal)   │
│ Scout  🔍   │    │                  │    │              │
│ Penny  📋   │◄──►│  Research ctx    │───►│  Events      │
│ Archie 🏗️   │    │  PRD / Arch      │    │  Colors      │
│ Quinn  🧪   │    │  Registry        │    │  Progress    │
│ Judge  ⚖️   │    │  Sign-offs       │    │  Sign-off UI │
│ Pixel  🎨   │    │  Events bus      │    │              │
│ Flow   🧭   │    │  Checkpoints     │    └──────────────┘
│ Alex   👤   │    │  uat_doc         │
│ Morgan 📬   │    │  sit_doc         │
│ Dev×N  🔨   │    │  handover_doc    │
│ Rev×N  🔎   │    └────────┬─────────┘
└─────────────┘             │
                            ▼
                 ┌──────────────────┐
                 │  llm_client.py   │
                 │                  │
                 │  ModelTier enum  │
                 │  Auto-detect     │
                 │  Model pool +    │
                 │  429 rotation    │
                 │  Anthropic SDK   │
                 │  OpenAI compat   │
                 └──────────────────┘
                            │
         ┌───────────────┼───────────────┐
         │                │               │
         ▼                ▼               ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐
│ sandbox.py     │  │ telemetry.py  │  │ hardening.py  │
│               │  │              │  │              │
│ Sandbox class │  │ CostTracker  │  │ atomic_write │
│ syntax check  │  │ BudgetGuard  │  │ file_lock    │
│ import check  │  │ PhaseMetrics │  │ sanitize     │
│ test runner   │  │ pricing tbl  │  │ disk checks  │
│ safe env      │  │ context win  │  │ budget ctx   │
└───────────────┘  └──────────────┘  └──────────────┘
                            │
                            ▼
                    LLM Backend
              (Hyperspace / Anthropic /
               OpenAI / Ollama / ...)
```

## Module Map

| File | Lines | Purpose |
|------|-------|---------|
| `hive/crew.py` | ~2572 | 13-phase orchestrator: parallel build, sandbox loop, self-reflection, cost tracking, contract amendment rebuild, PII scanning, regression test generation, project DNA extraction |
| `hive/prompts.py` | ~1342 | System prompts + task templates for all agent roles including OWASP checklists, SOLID/PII checks, self-reflection, regression tests, project DNA |
| `hive/ui.py` | ~1130 | ANSI terminal rendering, sign-off prompts, progress dashboard with live cost, delivery summary, `build_preview()` for interactive mode |
| `hive/state.py` | ~740 | Blackboard, UserProfile, LogEntry, Events, adaptive context header, checkpoint save/load, `approved_signatures()` for context compression |
| `hive/llm_client.py` | ~635 | Pluggable LLM connector. Auto-detects backend. Tier→model. Resilient retry + 429 model-pool rotation. Streaming support (on_token callback). Multi-provider per-tier routing via `resolve_endpoint(tier)`. |
| `hive/connectors.py` | ~580 | Connector system: ConnectorType, KnowledgeItem, ConnectorRegistry, agent routing, git repo clone & ingest, URL fetch, brownfield `codebase_index()` |
| `hive/hardening.py` | ~478 | atomic_write, file_lock, sanitize, budget, disk checks |
| `hive/memory.py` | ~456 | Memory system: MemoryEntry, AgentMemory, TeamMemory, GlobalMemory, MemoryManager (3-tier learning) |
| `hive/sandbox.py` | ~600 | Code execution loop: Sandbox, syntax check, import check, test runner, PII scanner, coverage runner, safe subprocess, context-aware imports, multi-file test execution (`run_test_in_context`, `run_all_tests_in_context`) |
| `hive/agents.py` | ~338 | Agent dataclass with logbook+memory-wired think(), AgentRoster (10 named agents), DEV_POOL, REVIEWER_POOL |
| `hive/telemetry.py` | ~317 | **NEW** CostTracker, BudgetExceeded, estimate_cost, model_context_window, per-phase PhaseMetrics |
| `hive/dashboard.py` | ~350 | **NEW** SSE-based web dashboard: DashboardServer, real-time phase progress, file status, cost tracking, event log |
| `hive/plugins/` | ~660 | **NEW** Optional plugin system: protocols (base.py), discovery+registry (registry.py), 4 example plugins |
| `hive/__init__.py` | ~44 | Package exports |
| `run_hive.py` | ~147 | CLI entry point with --resume, --list-projects, --auto, --interactive, --dashboard, --modify, --attach, --repo, --plugin |
| `tests/test_hive.py` | ~4796 | ~536 unit tests (no API calls) |
| `tests/test_hardening.py` | ~669 | ~88 hardening + integration tests |
| `tests/test_plugins.py` | ~760 | ~92 plugin system tests |

## Key Design Patterns

### 1. Blackboard Pattern
All agents read/write to a single `Blackboard` dataclass. No agent-to-agent
message passing — the board IS the shared memory. Events are emitted as
side effects for the UI to render.

```python
board = Blackboard(feature="...")
board.research = ResearchContext(...)   # Scout writes
board.prd = "..."                       # Penny reads research, writes PRD
board.architecture = "..."              # Archie reads PRD, writes arch
```

### 2. Model Tier System
Agents request capability, not model names. The `LLMClient` resolves:

| Tier | Role | Env Var | Default |
|------|------|---------|---------|
| FAST | Scout, Quinn, Alex | `LLM_MODEL_SMALL` | same as LLM_MODEL |
| BALANCED | Penny, Pixel, Flow | `LLM_MODEL` | claude-sonnet-4-20250514 |
| POWERFUL | Archie, Judge, Devs | `LLM_MODEL_BIG` | same as LLM_MODEL |

Tiers can be escalated on retry: `tier.escalate()` bumps FAST→BALANCED→POWERFUL.

### 3. Resilient LLM Calls & Dynamic Model Routing
Every agent call goes through a **4-layer resilience system**:

```
Agent.think()  ──►  LLMClient.chat()  ──►  Backend
     │                     │                      │
     │  LogEntry created   │  Retry loop:         │
     │  on success/fail    │    1. Try original    │
     │                     │    2. On 429: rotate  │
     │                     │       model in pool   │
     │                     │    3. Strip thinking  │
     │                     │    4. Escalate tier   │
     │                     │    5. Raise if exhaust│
     ▼                     ▼                      ▼
  Logbook entry         Resilience metadata     LLM Response
  + LLM_INCIDENT event  on LLMResponse          (text, tokens)
```

**Retry strategy** (configurable, default 5 attempts):
- **429 rate-limit**: immediately rotate to the next model in the pool
  (`LLM_FALLBACK_MODELS` + tier models), with short 0.5–2s backoff.
  When the entire pool is exhausted, it resets and waits longer before retrying.
  Escalated-tier models that are also rate-limited are skipped.
- **Other failures**: exponential backoff.
  - After first failure: strip `thinking` param (proxy may not support it)
  - After second failure: escalate tier (FAST→BALANCED→POWERFUL)
  - All retries exhausted: raise with full error history attached

Each `LLMResponse` carries resilience metadata:
```python
LLMResponse(
    text="...",
    tier_requested="fast", tier_used="balanced",  # was escalated
    retries=2, tier_escalated=True, thinking_stripped=True,
    model_switched=True,                           # rotated due to 429
    errors=["HTTPStatusError: 429", "HTTPError: 503"],
    duration_s=12.3,
)
```

Different agents **automatically use different models** based on their tier —
no configuration needed. If a lighter model fails or is rate-limited, the system
transparently escalates or rotates without losing context.

### 4. Event Bus
Every agent action emits an `Event(type, agent, content)`. The `TerminalUI`
subscribes by calling `flush_events()` after each phase/step. This decouples
agent logic from display logic.

Event types: `THINKING`, `SPEAKING`, `HANDSHAKE`, `AGREEMENT`, `DISAGREEMENT`,
`WRITING`, `REVIEWING`, `VERDICT`, `CHECKPOINT`, `USER_SIGNOFF`, `LLM_INCIDENT`, etc.

### 5. Conditional Crew Composition
Not every project needs every agent:

```python
agents = AgentRoster.compose(has_frontend=True, dev_count=3)
# → Scout, Penny, Archie, Quinn, Judge, Morgan (always)
# → Pixel, Flow, Alex (only if has_frontend)
# → Dexter, Devi, Dale (dev pool, count from dep graph)
# → Remy, River, Robin (sub-reviewers, spawned by Quinn on >8 files)
```

### 6. Dependency-Layered Parallel Build
Archie's contract defines a file dependency graph. `dep_layers()` does a
topological sort into parallel layers:

```
Layer 1: [models.py, config.py]     ← no deps, built in parallel
Layer 2: [routes.py, middleware.py]  ← depend on layer 1, built in parallel
Layer 3: [app.py]                    ← depends on layer 2
```

Each layer runs with `ThreadPoolExecutor(max_workers=min(len(layer), 4))`.
A `threading.Lock` protects writes to `board.registry` and `board.all_deferred`.
`_save()` (checkpoint) is called once per layer after all futures complete.

Each file goes through: **Generate → Sandbox → Self-Reflect → Review → (Revise?) → Approve/Escalate**

### 6b. Code Execution Sandbox
Before any human or agent review, every generated Python file passes through
a secure sandbox (`hive/sandbox.py`):

```
Dev generates code
      ↓
Sandbox: syntax check (py_compile)
      ↓ pass?
Sandbox: import check (can module load?)
      ↓ pass?
Sandbox: test execution (if test files exist)
      ↓ fail?
Dev receives sandbox feedback → revises (up to 2 rounds)
      ↓ still failing?
Proceed to review anyway (sandbox is advisory, not blocking)
```

**Sandbox safety:**
- Runs in an isolated temp directory (auto-cleaned)
- API keys and secrets stripped from environment
- Process timeout enforced (default 30s, `HIVE_SANDBOX_TIMEOUT`)
- Output truncated to prevent prompt bloat
- Path traversal in filenames blocked
- Can be disabled entirely: `HIVE_SANDBOX_ENABLED=0`
- **Context-aware imports**: `check_file_in_context()` stages all sibling registry files alongside the target so cross-module imports resolve correctly; distinguishes internal vs external `ModuleNotFoundError`

### 6b-ii. Test Execution Feedback Loop
After sandbox syntax/import checks and self-reflection, test files go through
a **real pytest execution** loop (`hive/sandbox.py` + `hive/crew.py`):

```
Dev code passes syntax + import check
      ↓
Self-Reflection (FAST tier)
      ↓
run_test_in_context(): stage ALL project files + run pytest
      ↓ tests pass?
      │   YES → proceed to review
      │   NO  → feed real pytest output back to dev
      ↓
Dev rewrites based on actual test failures
      ↓ (up to MAX_TEST_FIX_ATTEMPTS rounds, default 2)
Proceed to review
```

**Multi-file test execution** (`run_test_in_context()`, `run_all_tests_in_context()`):
- Stages ALL approved files from the registry into a temp directory
- Cross-module imports resolve correctly since all siblings are present
- Runs real `pytest` with output capture
- Returns pass/fail status + captured output for LLM feedback

### 6b-iii. Integration Test Fix Loop
During the integration phase, after `run_code_checks` identifies test failures,
`_integration_test_fix_loop()` (`hive/crew.py`) routes failures back to devs:

```
Quinn runs run_code_checks (full integration test suite)
      ↓ failures found?
      │   NO  → integration complete
      │   YES → isolate which test files fail
      ↓
For each failing test:
  - Identify responsible dev (who built the source file)
  - Send real pytest output + contract to dev
  - Dev fixes the source file
      ↓
Re-run integration tests
      ↓ (up to MAX_INTEGRATION_FIXES rounds, default 2)
Proceed with remaining integration results
```

Configurable via `HIVE_MAX_INTEGRATION_FIXES` (default 2) and
`HIVE_MAX_TEST_FIX_ATTEMPTS` (default 2).

### 6c. Self-Reflection Loop
After sandbox (before review), each Python file goes through a **self-reflection**
step where the dev agent critiques its own code against the contract:

```
Dev code passes sandbox
      ↓
Dev (FAST tier): "Self-critique this code against the contract.
                  What's missing? What could break?"
      ↓ suggests improvements?
Dev rewrites (single pass) → validated before accepting
      ↓
Proceeds to reviewer
```

This catches low-hanging issues cheaply (FAST tier) without consuming reviewer
budget. The reflection prompt includes the contract spec, expected exports,
and dependency context.

### 6d. Cost Tracking & Budget Enforcement
Every LLM call is metered by `CostTracker` (`hive/telemetry.py`), which
accumulates tokens × model-specific pricing:

```
EPTCrew.run()
  ├── cost_tracker.start_phase("research")
  │     ├── Agent.think() → LLMResponse (input_tokens, output_tokens)
  │     └── cost_tracker.record_call(model, in_tok, out_tok)
  ├── cost_tracker.end_phase() → PhaseMetrics
  │
  └── After all phases:
        ├── board._cost_tracker → ui.final_summary() shows cost breakdown
        └── project_dna.json includes cost summary
```

**Budget guard:** if `HIVE_BUDGET_USD` is set (e.g., `5.0`), the tracker
raises `BudgetExceeded` when the running total exceeds the limit. The pipeline
saves a checkpoint and exits gracefully — resume picks up where it left off.

**Model pricing:** built-in pricing table covers 15+ models (Claude, GPT-4,
Gemini, Llama, Mistral). Unknown models use a safe default. Users can
override with `HIVE_COST_PER_1K_INPUT` / `HIVE_COST_PER_1K_OUTPUT`.

### 6e. Adaptive Context Window
`full_context_header()` in `state.py` now accepts a `max_tokens` parameter
derived from the model's known context window. The method allocates 70% of
the budget for context (PRD, architecture, contract, research) and reserves
30% for task prompt + output. This prevents context overflow when using
smaller-window models.

`model_context_window()` in `telemetry.py` resolves the model name to its
known window size (200K for Claude, 128K for GPT-4o, etc.) using substring
matching to handle prefixed model names.

### 6f. Project DNA Extraction
After the pipeline completes, a post-run step extracts **Project DNA** —
structured lessons learned:

```python
# hive/crew.py → _extract_project_dna()
LLM prompt: "Analyze the build logbook and extract:
  - stack_patterns: what worked
  - common_mistakes: what failed repeatedly
  - architecture_lessons: design insights
  - review_insights: quality patterns"

→ projects/<slug>/docs/project_dna.json
→ fed into global memory for future projects
```

This makes each run contribute to the system's collective intelligence.

### 6g. Streaming LLM Output
`LLMClient.chat()` accepts an optional `on_token: Callable[[str], None] | None`
callback. When provided, tokens stream in real-time rather than waiting for
the full response. All three backends are supported:

- **Anthropic SDK** — native streaming via `client.messages.stream()`
- **Anthropic HTTP SSE** — `_stream_anthropic_http()` parses Server-Sent Events
- **OpenAI SSE** — `_stream_openai()` parses `data:` lines from the SSE stream

`Agent.think()` also accepts `on_token`, passing it through to the LLM client.
This enables real-time progress display in the terminal UI during long generations.

### 6h. URL-based Knowledge Attachment
`--attach https://...` now fetches remote URLs via httpx. New functions in
`hive/connectors.py`:

- `is_url(path)` — detects `http://` or `https://` strings
- `fetch_url(url)` — GETs the URL content via httpx (respects `HIVE_LLM_TIMEOUT`)
- `_content_type_to_connector(ct)` — maps Content-Type header to ConnectorType
- `_url_label(url)` — extracts a short label from the URL path
- `ConnectorRegistry.ingest_url(url)` — orchestrates fetch + type detection + item creation

Auto-detects connector type from URL file extension or response Content-Type
header. Binary content (images, archives, etc.) is rejected with a warning.

### 6i. Registry-Aware Dev Context
During the build phase, developers now receive the **full source code of their
declared dependencies** rather than generic 40-line previews of all approved files.

`EPTCrew._dependency_context(file_entry)` in `hive/crew.py`:
1. Reads the file's `depends_on` list from the contract
2. Looks up each dependency in `board.registry`
3. Assembles their full approved code into a context block

The `DEV_TASK`, `DEV_REVISION_TASK`, and `DEV_SANDBOX_REVISION_TASK` prompt
templates include a `{dependency_context}` placeholder that receives this
targeted context. This dramatically improves code quality for files with
inter-module dependencies.

### 6j. Request Pacing
Thread-safe `_RequestPacer` in `hive/llm_client.py` enforces a minimum interval
(`HIVE_REQUEST_PACE_MS`, default 200 ms) between LLM calls. On 429 responses the
pacer respects the server's `Retry-After` header, preventing thundering-herd retries.

### 6k. Rate-Limit Retry & Dropped File Recovery
Files that fail due to 429 rate-limit cascades during build are not silently
dropped. Instead, they're queued for retry after a cooldown:

```
Build phase:
  Layer N files built in parallel → some hit 429 cascade
      ↓
  _retry_rate_limited_files(): wait HIVE_RATE_LIMIT_COOLDOWN sec → retry
      ↓ still failing?
  File marked as DROPPED with clear skip_reason
  Delivery summary shows resume command
```

**Single-model warning:** if `LLM_FALLBACK_MODELS` is not configured and
only one model is available, the build phase emits a warning and increases
backoff base (3.0s instead of default) since model rotation is not possible.

**Quinn sub-reviewer delegation:** on builds with more than 8 files, Quinn
spawns ephemeral FAST-tier sub-reviewer agents (Remy, River, Robin, Riley) —
one per file batch. Quinn only re-reviews files that a sub-reviewer FAILed,
avoiding a single-agent bottleneck on large builds.

### 6l. Plugin System (Optional)
Hive has a protocol-based plugin architecture (`hive/plugins/`) that allows
injecting domain knowledge, coding guidelines, external system connectors,
test data, and lifecycle hooks — all without modifying core modules.

```
Plugin Discovery (3 sources):
  1. --plugin ./path/to/plugin.py     (explicit CLI paths)
  2. HIVE_PLUGINS_DIR directory scan   (default: ./plugins/)
  3. Python entry points               (hive.plugins group)
      ↓
  PluginRegistry ─► auto-detect categories via duck typing
      ↓
  During knowledge ingest:
    KnowledgePlugins → items added to board.knowledge_base
    GuidelinesPlugins → rules stored in board.plugin_guidelines
      ↓
  During each phase:
    LifecyclePlugins → on_phase_start() / on_phase_end() hooks
      ↓
  On demand:
    SystemPlugins → connect/execute/disconnect (GitHub, Docker, SAP, etc.)
    TestDataPlugins → generate fixtures, mock data
```

**Five plugin protocols** (all `@runtime_checkable`, structural typing):
- `KnowledgePlugin` — domain docs (SAP, Salesforce, industry knowledge)
- `GuidelinesPlugin` — coding rules, linting config, company standards
- `SystemPlugin` — external system connectors (GitHub, Docker, JIRA, SAP)
- `TestDataPlugin` — test fixtures, mock data, seed data generators
- `LifecyclePlugin` — pre/post hooks for any pipeline phase

**Zero-impact design:** if no plugins are loaded, all plugin code paths are
no-ops guarded by `if self.plugin_registry`. The core pipeline remains
completely unchanged.

### 6m. Context Compression
`Blackboard.approved_signatures()` in `hive/state.py` extracts only the
structural signatures (function defs, class defs, imports, top-level assignments)
from all approved files — reducing context size by 70-80% compared to full code.

Used in:
- **Self-reflection** prompts: dev sees compressed sibling context
- **Test-fix** prompts: dev gets focused signatures instead of bloated full code
- **Integration-fix** prompts: targeted context for fixing cross-module failures

This allows feeding more approved-file context without exceeding model windows.

### 6n. Multi-Provider LLM Routing
`hive/llm_client.py` supports per-tier provider routing — each capability tier
(FAST, BALANCED, POWERFUL) can point to a different LLM backend:

```
resolve_endpoint(tier) → (base_url, api_key, format_hint)

FAST     → LLM_BASE_URL_FAST     / LLM_API_KEY_FAST     / LLM_FORMAT_FAST
BALANCED → LLM_BASE_URL_BALANCED / LLM_API_KEY_BALANCED / LLM_FORMAT_BALANCED
POWERFUL → LLM_BASE_URL_POWERFUL / LLM_API_KEY_POWERFUL / LLM_FORMAT_POWERFUL
```

Falls back to the default `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_FORMAT` if tier-
specific vars are not set. Thread-safe via thread-local storage — critical for
the parallel dep-layer build where multiple threads call different tiers.

Per-URL format detection is cached so auto-detection only happens once per
unique base URL.

### 6o. Interactive Build Mode
`--interactive` CLI flag (or `EPTCrew(interactive=True)`) enables user-in-the-loop
file review during the build phase:

```
Dev generates code → sandbox passes → self-reflection
      ↓
build_preview(): display code in terminal
      ↓
User: [a]pprove / [f]eedback / [s]kip
      ↓ feedback?
User's feedback sent back to dev agent → re-generation
      ↓ approve?
Proceeds to Quinn review (normal flow)
```

`build_preview()` in `hive/ui.py` renders syntax-highlighted code with a 3-option
prompt. Feedback strings are injected as additional constraints for the dev.

### 6p. Live Web Dashboard
`hive/dashboard.py` provides an SSE-based web dashboard (stdlib only, no deps):

```
--dashboard [PORT]  (default: 8765)
      ↓
DashboardServer.start()
  ├── HTTP GET /          → Auto-refreshing HTML page (EventSource/SSE)
  ├── HTTP GET /events    → SSE stream (phase progress, file status, cost, events)
  └── DashboardServer.stop() on pipeline exit
```

The dashboard shows:
- Current phase progress (with elapsed time)
- File build status (pending / building / approved / failed)
- Running cost breakdown
- Event log (LLM calls, verdicts, sign-offs)

No new dependencies — uses `http.server` and manual SSE framing.

### 6q. Brownfield Mode — Existing Codebase Modification
`--modify PATH` (or `EPTCrew(modify_path=PATH)`) enables modification of an
existing codebase rather than generating greenfield code:

```
--modify ./my-project "Add rate limiting to the API"
      ↓
codebase_index(root):
  - Python files: AST-based signature extraction (_python_signatures)
  - JS/TS/Java/Go: regex-based signature extraction (_generic_signatures)
      ↓
Knowledge ingest: codebase structure auto-attached as knowledge items
      ↓
Architecture prompt: existing code structure injected for context
      ↓
Build phase: devs modify existing files (not just create new ones)
```

`codebase_index(root)` in `hive/connectors.py` produces a structured summary
of the existing codebase suitable for LLM consumption — function signatures,
class definitions, and module structure.

### 7. User Welcome & Profile
Before any AI agent runs, the pipeline collects the user's identity:

```
Welcome & Intake → name, role, company (all optional)
  → Is this request for you or someone else?
  → If someone else: end user name, role, description
  → How do you currently do this? (as-is process)
  → Any extra context?
```

This `UserProfile` is injected into every downstream prompt:
- Scout's research uses it for domain analysis
- Penny's interview questions are tailored to the user context
- PRD includes Stakeholders, Current State (As-Is), and user-perspective stories
- Release notes address the requester by name

### 8. User Sign-off Gates with Attribution
The pipeline pauses for user approval at:
- **PRD** — after Penny writes it (up to 3 revision rounds)
- **Feasibility** — after Archie's assessment
- **Architecture + Contract** — before any code is written

Every sign-off records **who produced and reviewed** the artifact:

```python
SignOff(
    artifact="prd", version=1, approved=True,
    produced_by="Penny 📋 (Product Manager)",
    reviewed_by=["Scout 🔍 (Research Analyst)"],
)
```

Sign-off prompts display attribution visually:
```
┌──────────────────────────────────────────────────────┐
│  SIGN-OFF REQUIRED: PRD                              │
└──────────────────────────────────────────────────────┘
    Produced by : Penny 📋 (Product Manager)
    Reviewed by : Scout 🔍 (Research Analyst)
```

The final delivery summary includes a complete sign-off log with all parties.
Release notes include a Parties & Attribution table.

### 9. Judge Escalation
If a file fails review `MAX_REVISIONS` (3) times, it's escalated to Judge, who
can: **APPROVE** (defer issues), **REJECT** (skip file), or **AMEND_CONTRACT**
(change the spec and rebuild).

### 10. Logbook — Persistent Thinking Record
Every LLM call is recorded in a `Logbook`—a list of `LogEntry` dataclasses
stored per-project at `docs/logbook.json`. Each entry captures:

| Field | Description |
|-------|-------------|
| `agent_id`, `agent_name` | Who made the call |
| `phase` | Which pipeline phase |
| `task_summary` | First 120 chars of the prompt |
| `model_requested` / `model_used` | Intended vs actual model |
| `tier_requested` / `tier_used` | Intended vs actual tier |
| `input_tokens` / `output_tokens` | Token usage |
| `duration_s` | Wall-clock time |
| `retries`, `tier_escalated`, `thinking_stripped` | Resilience metadata |
| `errors` | Error messages from failed attempts |
| `response_preview` | First 200 chars of response |
| `success` | Did the call ultimately succeed? |

The logbook enables:
- **Post-mortem debugging**: see exactly which calls failed and why
- **Cost analysis**: total tokens per agent, per phase
- **Performance profiling**: which agents are slow, which models are used
- **Audit trail**: full record of every AI decision

The delivery summary prints a logbook digest:
```
📓 Logbook Summary:
    LLM calls     : 23
    Total tokens  : 45,200 in / 18,900 out
    Total time    : 142.3s
    Retries       : 2
    Tier escalated: 1x
    Model switches: 3x (429 rotation)
    Models used   : claude-sonnet-4-20250514 (20x), claude-haiku-4-20250414 (3x)
```

### 11. Connector System — External Knowledge Ingest
Users can attach files or entire folder paths to feed domain knowledge,
reference code, API specs, test cases, and data into the pipeline.

**Connector Types:**

| Type | Extensions | When Used |
|------|-----------|-----------|
| DOCUMENT | .md .txt .rst | Business docs, domain knowledge |
| CODEBASE | .py .js .ts .java .go .rs .rb .cs | Reference/legacy code |
| TEST_CASE | test_*.* *_test.* *.spec.* *.test.* | Existing test suites |
| DATA_FILE | .csv .tsv .json .yaml .xml | Sample/live data |
| API_SPEC | openapi.* swagger.* .graphql .proto | API specifications |
| SCHEMA | .sql .ddl .prisma | Database schemas |
| GIT_REPO | (full repo clone) | Reference repos — "build something similar" |

**Size management:**
- **< 8 KB**: injected in full into agent prompts
- **8–50 KB**: truncated (head + tail lines)
- **> 50 KB**: Scout auto-summarizes; only summary flows to agents

**Agent routing** — each agent receives only its relevant knowledge:

| Agent | Receives |
|-------|----------|
| Scout 🔍 | ALL items |
| Penny 📋 | DOCUMENT, DATA_FILE |
| Archie 🏗️ | API_SPEC, SCHEMA, CODEBASE |
| Quinn 🧪 | TEST_CASE, API_SPEC |
| Judge ⚖️ | TEST_CASE, API_SPEC |
| Devs 🔨 | CODEBASE, API_SPEC, SCHEMA, GIT_REPO |

**CLI usage:**
```bash
python3 run_hive.py --attach ./docs/ --attach ./api/swagger.yaml "Build X"
python3 run_hive.py --repo https://github.com/org/project "Build similar for our use case"
```
Interactive mode also asks for paths during the ingest phase.

### 12. Git Repository Support — "Build Something Similar"

Users can provide a git repo URL (via `--repo` or interactively) and the crew will:

1. **Ingest phase**: Shallow-clone the repo, build a file tree, ingest source files as `GIT_REPO` knowledge items
2. **Research phase**: Scout performs a deep reverse-engineering analysis:
   - Tech stack & framework identification
   - Architecture patterns (layering, data flow, error handling)
   - File structure mapping
   - Data model & API surface catalog
   - What to replicate vs. adapt for the new use case
   - Per-agent takeaways (Penny, Archie, Devs, Quinn)
3. **Downstream flow**: The repo analysis is injected into:
   - Scout's research (full context)
   - Penny's interview questions and PRD generation
   - All other agents via `full_context_header()` (Archie, Devs, Quinn, Judge)

**Git URL detection** recognizes: `github.com`, `gitlab.com`, `bitbucket.org`, `git@` SSH URLs, `.git` suffixed URLs.

**File prioritization**: README, main entry points, and config files are prioritized when building the repo context for LLM prompts (capped at ~20K chars).

### 13. Memory System — Individual + Team + Global Learning

Agents learn from mistakes, adapt over time, and share knowledge. Memories persist within a project and distill across projects.

**Three memory tiers:**

| Tier | Scope | Contents | Lifetime |
|------|-------|----------|----------|
| **Agent Memory** | Per-agent, per-project | Mistakes, patterns, lessons | Full detail within project |
| **Team Memory** | Shared board, per-project | Insights any agent pushes for others | Full detail within project |
| **Global Memory** | Cross-project | Distilled lessons from completed projects | Persists forever (capped at 100) |

**Memory entry kinds:**

| Kind | Prefix | When Created |
|------|--------|-------------|
| `mistake` | `[AVOID]` | Review failure, parse error, user rejection |
| `pattern` | `[DO]` | First-try PASS, successful approaches |
| `lesson` | `[KNOW]` | Judge decisions, deferred issues, general learnings |
| `insight` | `[NOTE]` | Team pushes — domain findings, architecture decisions |

**How memories flow into prompts:**
- Before each `Agent.think()` call, the crew sets `board.memory_context`
- `MemoryManager.context_for_agent()` combines Global + Personal + Team blocks
- The memory block is prepended to the task string in `think()`
- Entries are sorted: phase-relevant first, then mistakes > lessons > patterns

**When memories are recorded:**
- **Research**: Scout records JSON parse failures; pushes domain/stack to team
- **PRD**: Penny records rejections; pushes req scope to team
- **Architecture**: Archie records parse failures/rejections; pushes file plan to team
- **Build**: Devs record each review FAIL with issue details; PASS patterns tracked; Quinn pushes common blockers to all devs
- **Escalation**: Judge decisions become lessons; contract amendments pushed to team
- **Release**: All memories distilled → global; global saved to `projects/.global_memory.json`

**Storage:**
```
projects/
  .global_memory.json           ← cross-project distilled lessons
  <slug>/
    memory/
      agent_scout.json          ← Scout's personal memories
      agent_dev_1.json          ← Dev_1's personal memories
      agent_quinn.json          ← Quinn's personal memories
      team.json                 ← shared team insights
```

**Cross-project flow:**
1. New project starts → `MemoryManager.load_global()` loads past lessons
2. Agents see relevant global lessons in their prompts
3. During the project, agents accumulate personal + team memories
4. Project completes → `distill_to_global()` extracts lessons + mistakes + team insights
5. Global memory saved (capped at 100 most recent entries)

## Data Flow

```
User Feature Request
        │
        ▼
   Welcome / Intake ───► UserProfile (name, role, end-user, as-is)
        │
        ▼
   Knowledge Ingest ───► KnowledgeItems (docs, code, specs, schemas, tests)
        │                  ├── Large files auto-summarized by Scout
        │                  └── Git repos: shallow-clone → file tree + source items
        │
        ▼
   Scout (FAST) ──────► Repo Analysis (if git repo attached) → ResearchContext JSON
        │
        ▼
   Penny (BALANCED) ───► Interview Questions (with Red Flag pushback) ──► User Answers
        │
        ▼
   Penny (BALANCED) ───► PRD (Markdown) ──► [User Sign-off]
        │
        ▼
   Archie (POWERFUL) ──► Feasibility JSON ──► [User Sign-off]
        │
        ▼
   Archie (POWERFUL) ──► Architecture + Contract ──► [User Sign-off]
        │
        ▼
   Penny (BALANCED) ───► Ratification check
        │
        ▼
   AgentRoster.compose() ──► Active crew + dev pool
        │
        ▼
   For each dep layer (parallel within layer):
     Dev (POWERFUL) ─► Code ─► Sandbox (syntax+import check)
                                  ├── FAIL → Dev revises from sandbox feedback (up to 2x)
                                  └── PASS → Self-Reflect (FAST tier self-critique)
                                              │
                                              ▼
                              Quinn/sub-reviewer (FAST) review
                                  ├── PASS → save to src/
                                  ├── PASS_WITH_NOTES → save + defer
                                  ├── FAIL → revise (up to 3x)
                                  └── FAIL 3x → Judge (POWERFUL)
        │
        ▼
   Quinn (FAST) ─► Integration review (all files + sandbox results)
        │
        ▼
   Alex (FAST) ──► UAT.md  (pseudo-user scenarios, copy-paste ready)
   Quinn (FAST) ──► SIT.md  (system integration test plan)
        │
        ▼
   Penny (BALANCED) ──► release_notes.md
   Penny (BALANCED) ──► Handover.md  (arch summary, how-to-run, attribution, backlog)
   Penny (BALANCED) ──► Packaging artifacts  (pyproject.toml / package.json / go.mod …)
   Morgan (BALANCED) ──► delivery_checklist.md  (final checklist + crew sign-offs)
        │
        ▼   _extract_project_dna() ─► project_dna.json  (lessons → global memory)
   _sync_costs() ─► cost summary in logbook + final_summary
        │
        ▼   projects/<slug>/
     ├── docs/     (PRD, arch, contract, research, interviews, sign-offs,
     │              release notes, UAT, SIT, Handover, delivery checklist,
     │              project_dna.json, logbook.json)
     ├── src/      (generated source + packaging artifacts)
     ├── memory/   (agent learnings + team insights)
     └── checkpoints/ (board snapshots for resume)
```

## Project Output Structure

```
projects/
  <project-slug>/
    docs/
      user_profile.json        # Welcome/intake: name, role, end-user, as-is
      research_context.json    # Scout's structured analysis
      interviews.json          # All Q&A from interview phase
      prd.md                   # Penny's PRD (user-approved, with stakeholders)
      architecture.md          # Archie's design narrative
      contract.md              # Ratified file contract
      crew.json                # Active crew composition
      signoffs.json            # All sign-offs with attribution (who produced/reviewed)
      knowledge_base.json      # Ingested external knowledge items
      logbook.json             # Every LLM call: agent, model, tokens, retries, errors
      project_dna.json         # Post-run extracted lessons (stack patterns, mistakes, insights)
      release_notes.md         # Final summary with parties & attribution table
      UAT.md                   # Alex: pseudo-user acceptance test scenarios
      SIT.md                   # Quinn: system integration test plan
      Handover.md              # Penny: arch summary, how-to-run, backlog, attribution
      delivery_checklist.md    # Morgan: final delivery checklist + crew sign-offs
    src/
      <files defined in contract>
      pyproject.toml           # Python: hatchling config + extracted deps (or package.json / go.mod)
      requirements.txt         # Python: pip-installable deps
      Makefile                 # install / test / lint / run targets
      README.md                # What it does, install, usage, env vars
    checkpoints/
      board_<timestamp>.json   # Full blackboard snapshots
      board_latest.json        # Quick-resume pointer
```

## Resume Support

Checkpoints serialize the full Blackboard (minus events) to JSON. Resume
skips completed phases by checking `board.completed_phases`:

```bash
python3 run_hive.py --resume projects/my_api/checkpoints/board_latest.json
python3 run_hive.py --attach ./docs/ --attach ./api/swagger.yaml "Build a payment gateway"
```

## Environment Configuration

```bash
export LLM_BASE_URL="http://localhost:6655"
export LLM_API_KEY="your-key"
export LLM_MODEL="claude-sonnet-4-20250514"
export LLM_MODEL_BIG="claude-sonnet-4-20250514"
export LLM_MODEL_SMALL="claude-haiku-4-20250414"
export LLM_FORMAT="auto"                          # auto | anthropic | openai
export LLM_FALLBACK_MODELS="model-b,model-c"      # optional: 429 rotation pool

# Per-tier provider routing (optional — mix providers per capability tier)
export LLM_BASE_URL_FAST="http://localhost:11434/v1"
export LLM_API_KEY_FAST="unused"
export LLM_FORMAT_FAST="openai"
export LLM_BASE_URL_POWERFUL="https://api.anthropic.com"
export LLM_API_KEY_POWERFUL="sk-ant-..."
export LLM_FORMAT_POWERFUL="anthropic"
```

## Dependencies

- Python 3.12+
- `httpx` — HTTP client (all backends)
- `anthropic` — SDK (optional, only for native Anthropic endpoints)
- `pytest` — testing

That's it. No LangChain, no CrewAI, no vector stores, no agent frameworks.
