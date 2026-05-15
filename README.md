# 🐝 Hive

**Collective Intelligence Building Software**

[Quickstart](#quickstart) • [How It Works](#how-it-works) • [The Crew](#the-crew) • [CLI Reference](#cli-reference) • [Configuration](#configuration) • [Architecture](ARCHITECTURE.md) • [Contributing](CONTRIBUTING.md)

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Tests](https://img.shields.io/badge/tests-571%20passing-brightgreen)
![Dependencies](https://img.shields.io/badge/deps-1%20(httpx)-orange)

---

Hive assembles a team of AI agents that collaborate through a 13-phase software development lifecycle to turn your feature request into a complete, tested, documented, and shippable project — with checkpoints, memory, and human oversight built in.

```
You: "Build a URL shortener REST API with rate limiting"

Hive:  🔍 Scout     researches the problem space
       📋 Penny     interviews you, then writes the PRD
       🏗️ Archie    designs the architecture
       ⚖️ Judge     ratifies the plan
       👩‍💻 Dev team  builds all source files (parallel, by dependency layer)
       🧪 Quinn     quality-reviews every file (delegates to sub-reviewers on large builds)
       👤 Alex      writes UAT scenarios as a named pseudo-user
       🧪 Quinn     writes the SIT integration test plan
       📋 Penny     writes Handover.md + stack-aware packaging artifacts
       📬 Morgan    runs the final delivery checklist

Output: projects/url_shortener_rest_api_with_rate_limiting/
        ├── src/          ← source code + pyproject.toml, Makefile, README.md
        ├── docs/         ← PRD, architecture, UAT.md, SIT.md, Handover.md,
        │                    release_notes.md, delivery_checklist.md
        ├── memory/       ← agent learnings for next time
        └── checkpoints/  ← resume from any phase
```

## Quickstart

### 1. Install

```bash
# Clone
git clone https://github.com/naveenkumarbaskaran/hive.git
cd hive

# Install (requires Python 3.12+)
pip install -e .

# Or with dev tools
make dev
```

### 2. Configure your LLM

```bash
# Anthropic (direct)
export LLM_API_KEY=sk-ant-...
export LLM_MODEL=claude-sonnet-4-20250514

# Or OpenAI-compatible (LiteLLM, Ollama, vLLM, Azure, etc.)
export LLM_BASE_URL=http://localhost:8000
export LLM_API_KEY=your-key
export LLM_MODEL=gpt-4o

# Optional: fallback models for 429 rate-limit rotation
export LLM_FALLBACK_MODELS=claude-haiku-4-5,claude-sonnet-4-5
```

### 3. Build something

```bash
# Interactive mode (with welcome intake + sign-offs)
hive "Build a URL shortener REST API with rate limiting"

# Auto mode (skip sign-offs — great for testing/CI)
hive --auto "Build a CLI todo app with SQLite"

# Attach existing docs as context
hive --attach ./api-spec.yaml --attach ./design-doc.md "Build the payment service"

# Study an existing repo and build something similar
hive --repo https://github.com/user/project "Build something similar for healthcare"

# Resume from where you left off
hive --resume projects/my_project/checkpoints/board_latest.json

# List all projects
hive --list-projects
```

## How It Works

Hive runs a **13-phase pipeline** inspired by real-world product development:

| # | Phase | Agent | What Happens |
|---|-------|-------|-------------|
| 1 | **Welcome** | System | Collect user identity, context, as-is process |
| 2 | **Ingest** | System | Import attached files, repos, external knowledge |
| 3 | **Research** | 🔍 Scout | Analyze problem space, tech landscape, constraints |
| 4 | **Interview** | 📋 Penny | Ask clarifying questions — including architectural red flags (REPL vs CLI, auth, cross-platform) |
| 5 | **PRD** | 📋 Penny | Write product requirements document → user sign-off |
| 6 | **Feasibility** | 🏗️ Archie | Assess technical feasibility → user sign-off |
| 7 | **Architecture** | 🏗️ Archie | Design system architecture, file plan, dep graph → user sign-off |
| 8 | **Ratification** | 📋 Penny | Cross-check arch vs PRD |
| 9 | **Crew** | System | Assemble dev team based on project needs |
| 10 | **Build** | 👩‍💻 Devs | Write all files in parallel by dep layer; Quinn + sub-reviewers QA each file |
| 11 | **Integration** | 🧪 Quinn | Full cross-file integration review |
| 12 | **Test Docs** | 👤 Alex + 🧪 Quinn | UAT.md (pseudo-user scenarios) + SIT.md (integration test plan) |
| 13 | **Release** | 📋 Penny + 📬 Morgan | Handover.md, packaging artifacts, delivery checklist |

**Every phase saves a checkpoint.** Crash, rate-limit, or pause — resume exactly where you left off.

### The Quality Loop

During the Build phase, every file goes through a multi-agent quality pipeline:

```
Dev writes code
      ↓
Sandbox: syntax check + import check (automated, no LLM cost)
      ↓ pass?
Self-Reflection: dev critiques own code against contract (FAST tier)
      ↓
Quinn reviews (or sub-reviewer on builds > 8 files)
      ↓ issues found?
Dev rewrites (up to 3 rounds)
      ↓ still failing?
Judge arbitrates: APPROVE / REJECT / AMEND CONTRACT
```

Frontend files additionally get reviewed by Pixel (style) and Alex (UX/user perspective).

### Code Execution Sandbox

Every generated Python file is **actually executed** before review:
- **Syntax check** via `py_compile` (zero cost, instant)
- **Import check** verifies the module loads without runtime errors
- **Context-aware imports**: `check_file_in_context()` stages sibling registry files so cross-module imports resolve correctly; distinguishes internal vs external `ModuleNotFoundError`
- **Test execution** if test files are generated
- Runs in an isolated temp directory with API keys stripped
- Sandbox feedback loops back to the dev for self-correction
- Configurable: `HIVE_SANDBOX_TIMEOUT=30`, disable with `HIVE_SANDBOX_ENABLED=0`

### Streaming LLM Output

`LLMClient.chat()` accepts an optional `on_token` callback. When provided, tokens stream in real-time from all 3 backends (Anthropic SDK, Anthropic HTTP SSE, OpenAI SSE). `Agent.think()` also supports `on_token` — enabling live progress display during long generations.

### URL-based Knowledge Attachment

`--attach https://...` now fetches remote URLs via httpx:
- Auto-detects document type from URL extension or Content-Type header
- Rejects binary content (images, archives)
- Supports any text resource: API specs, docs, raw source files

```bash
hive --attach https://example.com/openapi.yaml "Build a client for this API"
```

### Registry-Aware Dev Context

During build, developers receive the **full source code of their declared dependencies** (not just generic file previews). `_dependency_context()` looks up each file's `depends_on` entries in the registry and assembles targeted context — dramatically improving code quality for inter-module dependencies.

### Cost Tracking & Budget Guard

Every LLM call is metered with model-specific pricing (15+ models built in):
- **Live cost display** in the progress dashboard during build
- **Per-phase cost breakdown** in the delivery summary
- **Budget enforcement**: set `HIVE_BUDGET_USD=5.0` to cap spend per run
- Graceful checkpoint + exit on budget exceeded (resume picks up)
- Override pricing: `HIVE_COST_PER_1K_INPUT`, `HIVE_COST_PER_1K_OUTPUT`

### Project DNA — Cross-Project Learning

After each run, Hive extracts structured lessons from the build:
- Stack patterns, common mistakes, architecture insights, review patterns
- Saved as `project_dna.json` and fed into global memory
- Future projects benefit from past experience automatically

### Penny's Interview — Architectural Red Flags

Before writing the PRD, Penny scans the feature request for patterns that commonly lead to bad requirements and forces the right questions:

| Detected Pattern | Question Asked |
|---|---|
| CLI with navigation (cd, history) | REPL session or one-shot CLI? |
| Destructive operations (delete, overwrite) | Confirmation by default, or --yes flag? |
| Cross-platform claim | Which OSes are required? |
| REST API, no auth mentioned | What auth method? |
| Search/scan feature | Recursive by default? Depth/timeout limit? |
| Sends to external systems | Fail silently or surface errors? |

### Parallel Build

Files within the same dependency layer build concurrently (up to 4 workers). The dep graph is a DAG — no file starts until all its dependencies are approved. Checkpoints save after each layer.

### Memory System

Hive has a **3-tier memory system** that makes agents smarter over time:

| Layer | Scope | Purpose |
|-------|-------|---------|
| **Agent Memory** | Per agent, per project | Mistakes, patterns, lessons learned |
| **Team Memory** | Shared across agents | Cross-agent insights and warnings |
| **Global Memory** | Across ALL projects | Distilled lessons that load into every future run |

After each project, memories are automatically distilled into compact global lessons.

## Plugin System (Optional)

Extend Hive with domain-specific knowledge, coding guidelines, external system connectors, test data generators, and lifecycle hooks — all without modifying core code:

```bash
# Load a plugin
hive --plugin ./plugins/sap_knowledge.py "Build an SAP integration"

# Multiple plugins
hive --plugin ./sap.py --plugin ./company_rules.py "Build a REST API"
```

**Five plugin types** (protocol-based, no inheritance required):
- **KnowledgePlugin** — inject domain docs (SAP modules, Salesforce objects, industry knowledge)
- **GuidelinesPlugin** — inject coding rules, linting configs, company standards
- **SystemPlugin** — connect to GitHub, Docker, JIRA, SAP, databases
- **TestDataPlugin** — generate fixtures, mock data, seed data
- **LifecyclePlugin** — run custom logic before/after any pipeline phase

Plugins are auto-discovered from `--plugin` paths, `HIVE_PLUGINS_DIR`, or Python entry points.
See `hive/plugins/examples/` for working examples.

## The Crew

| Agent | Role | When Active |
|-------|------|-------------|
| 🔍 **Scout** | Research Analyst | Always |
| 📋 **Penny** | Product Manager | Always |
| 🏗️ **Archie** | Technical Architect | Always |
| ⚖️ **Judge** | Arbitrator | Always |
| 🧪 **Quinn** | Quality Engineer | Always |
| 📬 **Morgan** | Delivery Manager | Always — runs final delivery checklist |
| 🎨 **Pixel** | UI Designer | Frontend projects only |
| 🧭 **Flow** | UX Designer | Frontend projects only |
| 👤 **Alex** | User Advocate | Frontend projects + UAT doc for all |
| 👩‍💻 **Dev 1–N** | Developers | Build phase (N scales with project size) |
| 🔎 **Remy/River/Robin/Riley** | Sub-reviewers | Build phase on projects > 8 files |

## What Gets Generated

Every project produces a complete, shippable directory:

```
projects/<slug>/
├── src/
│   ├── *.py / *.ts / *.go    ← source files
│   ├── pyproject.toml         ← packaging (Python) — deps extracted from imports
│   ├── requirements.txt
│   ├── Makefile               ← install / test / lint / run / clean
│   └── README.md              ← usage, install, env vars, examples
│
├── docs/
│   ├── prd.md                 ← product requirements
│   ├── architecture.md        ← system design
│   ├── contract.md            ← file-by-file build contract
│   ├── UAT.md                 ← user acceptance tests (named pseudo-user, copy-paste ready)
│   ├── SIT.md                 ← system integration test plan
│   ├── Handover.md            ← full project handover (arch, how-to-run, attribution, backlog)
│   ├── release_notes.md       ← delivery summary with attribution table
│   ├── delivery_checklist.md  ← Morgan's final checklist + project summary + crew sign-offs
│   ├── project_dna.json       ← extracted lessons for future projects
│   ├── logbook.json           ← every LLM call (model, tokens, cost, retries)
│   └── signoffs.json          ← agent sign-off history
│
├── memory/
│   ├── agent_*.json           ← per-agent learnings
│   └── team.json              ← cross-agent team insights
│
└── checkpoints/
    ├── board_*.json           ← timestamped snapshots
    └── board_latest.json      ← most recent state
```

## CLI Reference

```
usage: hive [-h] [--resume PATH] [--list-projects] [-v] [--auto]
           [--attach PATH] [--repo URL] [--log-level LEVEL] [--version]
           [feature]

Hive — Your AI Dev Crew, Assembled

positional arguments:
  feature               Feature request to build

options:
  -h, --help            show this help message and exit
  --resume PATH         Resume from checkpoint JSON
  --list-projects       List existing projects
  -v, --verbose         Verbose output
  --auto                Auto-approve all sign-offs (for testing / CI)
  --attach PATH         Attach knowledge files/folders (repeatable)
  --repo URL            Clone & study a git repo as reference (repeatable)
  --plugin PATH         Load a plugin module or package (repeatable)
  --log-level LEVEL     Log level: DEBUG, INFO, WARNING, ERROR
  --version             show program's version number and exit
```

### Examples

```bash
# Simple feature
hive "Build a markdown parser in Rust"

# With external context
hive --attach ./openapi.yaml "Build a Python SDK for this API"

# Study a repo + build
hive --repo https://github.com/pallets/flask "Build a similar microframework for Go"

# Debug mode
hive --log-level DEBUG --verbose "Build a chat server"

# CI/CD friendly (no interactive prompts)
hive --auto "Build a REST API for widgets"

# Resume after a crash or rate-limit interruption
hive --resume projects/my_project/checkpoints/board_latest.json
```

## Configuration

All configuration is via **environment variables** — no config files to manage.

### Required

| Variable | Description |
|----------|------------|
| `LLM_API_KEY` | API key for your LLM provider |
| `LLM_MODEL` | Default model name (e.g., `claude-sonnet-4-20250514`) |

### Optional

| Variable | Default | Description |
|----------|---------|------------|
| `LLM_BASE_URL` | `https://api.anthropic.com` | LLM endpoint URL |
| `LLM_MODEL_BIG` | Same as `LLM_MODEL` | Model for heavy reasoning tasks (POWERFUL tier) |
| `LLM_MODEL_SMALL` | Same as `LLM_MODEL` | Model for light tasks (FAST tier) |
| `LLM_FORMAT` | `auto` | Force format: `anthropic`, `openai`, or `auto` |
| `LLM_FALLBACK_MODELS` | — | Comma-separated fallback models for 429 rotation |
| `HIVE_PROJECTS_DIR` | `./projects` | Where projects are saved |
| `HIVE_MIN_DISK_MB` | `50` | Minimum free disk space before saves |
| `HIVE_LOG_LEVEL` | `WARNING` | Default log level |
| `HIVE_MAX_REVISIONS` | `3` | Max code revision cycles per file |
| `HIVE_LLM_TIMEOUT` | `120` | HTTP timeout (seconds) for LLM requests |
| `HIVE_BUDGET_USD` | `0` (unlimited) | Max USD spend per run; 0 = no limit |
| `HIVE_COST_PER_1K_INPUT` | model-based | Override $/1K input tokens |
| `HIVE_COST_PER_1K_OUTPUT` | model-based | Override $/1K output tokens |
| `HIVE_SANDBOX_TIMEOUT` | `30` | Max seconds per sandbox execution |
| `HIVE_SANDBOX_ENABLED` | `1` | Set to `0` to disable code execution sandbox |
| `HIVE_RATE_LIMIT_COOLDOWN` | `30` | Seconds to wait before retrying rate-limited files |
| `HIVE_REQUEST_PACE_MS` | `200` | Minimum milliseconds between LLM requests (0 to disable) |
| `HIVE_MAX_BUILD_WORKERS` | `2` | Max parallel file-build threads per dep layer |
| `HIVE_MAX_EVENTS` | `1000` | Max events kept in Blackboard memory |
| `HIVE_MAX_GLOBAL_MEMORY` | `100` | Max global memory entries retained |
| `HIVE_PLUGINS_DIR` | `./plugins` | Directory to scan for plugin modules |
| `NO_COLOR` | — | Disable ANSI colors (any value) |

### Rate Limit Handling

When a model returns a 429, Hive immediately rotates to the next available model in the pool rather than stalling:

```bash
export LLM_FALLBACK_MODELS=claude-haiku-4-5,claude-sonnet-4-5-20251001
```

Pool order: primary model → `LLM_MODEL_BIG` → `LLM_MODEL_SMALL` → `LLM_FALLBACK_MODELS`. If all models are rate-limited, Hive waits with exponential backoff then resets.

### LLM Provider Examples

```bash
# Anthropic (direct)
export LLM_BASE_URL=https://api.anthropic.com
export LLM_API_KEY=sk-ant-...
export LLM_MODEL=claude-sonnet-4-20250514

# OpenAI
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o

# Ollama (local)
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_API_KEY=unused
export LLM_MODEL=llama3

# Azure OpenAI
export LLM_BASE_URL=https://myresource.openai.azure.com/openai/deployments/gpt-4o
export LLM_API_KEY=your-azure-key
export LLM_MODEL=gpt-4o

# LiteLLM proxy
export LLM_BASE_URL=http://localhost:4000
export LLM_API_KEY=your-key
export LLM_MODEL=claude-sonnet-4-20250514
```

## Testing

```bash
# Run all tests
make test

# With coverage
make test-cov

# Lint
make lint
```

571 tests cover state management, agent logic, prompt parsing, UI rendering, connectors, memory, checkpoints, hardening utilities, parallel build, sandbox execution, cost tracking, streaming, URL ingestion, dependency context, model fallback, contract amendment rebuild, and the plugin system — all without making real API calls.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation including:

- Module map with responsibilities
- Design patterns used
- Data flow diagrams
- Checkpoint/resume mechanics
- Memory distillation process

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
