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
│      │                         │ ┌──────────────┐    │          │
│      │                         │ │ Dev → Review  │    │          │
│      │                         │ │   → Revise    │    │          │
│      │                         │ │   → Judge?    │    │          │
│      │                         │ └──────────────┘    │          │
│      ▼                         ▼                     ▼          │
│  Phase 9: Integration ──► Phase 10: Release                     │
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
│ Alex   👤   │    │                  │
│ Dev×N  🔨   │    └────────┬─────────┘
└─────────────┘             │
                            ▼
                 ┌──────────────────┐
                 │  llm_client.py   │
                 │                  │
                 │  ModelTier enum  │
                 │  Auto-detect     │
                 │  Anthropic SDK   │
                 │  Anthropic proxy │
                 │  OpenAI compat   │
                 └──────────────────┘
                            │
                            ▼
                    LLM Backend
              (Hyperspace / Anthropic /
               OpenAI / Ollama / ...)
```

## Module Map

| File | Lines | Purpose |
|------|-------|---------|
| `llm_client.py` | ~490 | Pluggable LLM connector. Auto-detects backend. Tier→model. Resilient retry + tier escalation. |
| `hive/__init__.py` | ~20 | Package exports |
| `hive/connectors.py` | ~570 | Connector system: ConnectorType, KnowledgeItem, ConnectorRegistry, agent routing, git repo clone & ingest |
| `hive/memory.py` | ~440 | Memory system: MemoryEntry, AgentMemory, TeamMemory, GlobalMemory, MemoryManager (3-tier learning) |
| `hive/state.py` | ~690 | Blackboard, UserProfile, LogEntry, Events, knowledge_base, repo_analysis, memory_context, checkpoint save/load |
| `hive/agents.py` | ~310 | Agent dataclass with logbook+memory-wired think(), AgentRoster, DEV_POOL |
| `hive/prompts.py` | ~700 | System prompts + task templates for all 10+ agent roles (with user/knowledge/repo context) |
| `hive/ui.py` | ~820 | ANSI terminal rendering, welcome intake, knowledge intake, repo clone summary, memory stats, sign-off prompts, logbook summary |
| `hive/crew.py` | ~1440 | 12-phase orchestrator (welcome→ingest→release), parsers, repo analysis, memory recording, build/review/revise loop |
| `run_hive.py` | ~100 | CLI entry point with --resume, --list-projects, --auto, --attach, --repo |
| `test_hive.py` | ~1700 | 206 unit tests (no API calls) |

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
Every agent call goes through a **3-layer resilience system**:

```
Agent.think()  ──►  LLMClient.chat()  ──►  Backend
     │                     │                      │
     │  LogEntry created   │  Retry loop:         │
     │  on success/fail    │    1. Try original    │
     │                     │    2. Strip thinking  │
     │                     │    3. Escalate tier   │
     │                     │    4. Raise if exhaust │
     ▼                     ▼                      ▼
  Logbook entry         Resilience metadata     LLM Response
  + LLM_INCIDENT event  on LLMResponse          (text, tokens)
```

**Retry strategy** (configurable, default 3 attempts):
- Attempt 1: try with requested tier + thinking
- Attempt 2 on fail: strip `thinking` param (proxy may not support it)
- Attempt 3 on fail: escalate tier (FAST→BALANCED→POWERFUL), use bigger model
- All failures: raise with full error history attached

Each `LLMResponse` carries resilience metadata:
```python
LLMResponse(
    text="...",
    tier_requested="fast", tier_used="balanced",  # was escalated
    retries=2, tier_escalated=True, thinking_stripped=True,
    errors=["TimeoutError: ...", "HTTPError: 503"],
    duration_s=12.3,
)
```

Different agents **automatically use different models** based on their tier—
no configuration needed. If a lighter model fails, the system transparently
escalates to a heavier one without losing context.

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
# → Scout, Penny, Archie, Quinn, Judge (always)
# → Pixel, Flow, Alex (only if has_frontend)
# → Dexter, Devi, Dale (dev pool, count from dep graph)
```

### 6. Dependency-Layered Build
Archie's contract defines a file dependency graph. `dep_layers()` does a
topological sort into parallel layers:

```
Layer 1: [models.py, config.py]     ← no deps, build in parallel
Layer 2: [routes.py, middleware.py]  ← depend on layer 1
Layer 3: [app.py]                    ← depends on layer 2
```

Each file goes through: **Generate → Review → (Revise?) → Approve/Escalate**

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
    Models used   : claude-sonnet-4-20250514 (23x)
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
   Penny (BALANCED) ───► Interview Questions ──► User Answers
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
   For each dep layer:
     Dev (POWERFUL) ──► Code ──► Quinn (FAST) review
                                  ├── PASS → save to src/
                                  ├── PASS_WITH_NOTES → save + defer
                                  ├── FAIL → revise (up to 3x)
                                  └── FAIL 3x → Judge (POWERFUL)
        │
        ▼
   Quinn (FAST) ──► Integration review (all files together)
        │
        ▼
   Penny (BALANCED) ──► Release notes
        │
        ▼
   projects/<slug>/
     ├── docs/     (PRD, architecture, contract, research, interviews, signoffs, release notes)
     ├── src/      (generated source code)
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
      release_notes.md         # Final summary with parties & attribution table
    src/
      <files defined in contract>
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
export LLM_FORMAT="auto"           # auto | anthropic | openai
```

## Dependencies

- Python 3.12+
- `httpx` — HTTP client (all backends)
- `anthropic` — SDK (optional, only for native Anthropic endpoints)
- `pytest` — testing

That's it. No LangChain, no CrewAI, no vector stores, no agent frameworks.
