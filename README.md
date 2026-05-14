<p align="center">
  <strong>🐝 Hive</strong><br>
  <em>Collective Intelligence Building Software</em>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#the-crew">The Crew</a> •
  <a href="#cli-reference">CLI Reference</a> •
  <a href="#configuration">Configuration</a> •
  <a href="ARCHITECTURE.md">Architecture</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-286%20passing-brightgreen">
  <img alt="Dependencies" src="https://img.shields.io/badge/deps-1%20(httpx)-orange">
</p>

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

During the Build phase, every file goes through a multi-agent review chain:

```
Dev writes code
      ↓
Quinn reviews (or sub-reviewer on builds > 8 files)
      ↓ issues found?
Dev rewrites (up to 2 rounds)
      ↓ still failing?
Judge arbitrates: APPROVE / REJECT / AMEND CONTRACT
```

Frontend files additionally get reviewed by Pixel (style) and Alex (UX/user perspective).

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
│   ├── logbook.json           ← every LLM call (model, tokens, latency, retries)
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

286 tests cover state management, agent logic, prompt parsing, UI rendering, connectors, memory, checkpoints, hardening utilities, parallel build, and model fallback — all without making real API calls.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation including:

- Module map with responsibilities
- Design patterns used
- Data flow diagrams
- Checkpoint/resume mechanics
- Memory distillation process

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.


---

Hive assembles a team of AI agents that collaborate through a 12-phase software development lifecycle to turn your feature request into a complete, tested, documented project — with checkpoints, memory, and human oversight built in.

```
You: "Build a URL shortener REST API with rate limiting"

Hive:  🔍 Scout researches the problem space
       📋 Penny writes a product requirements doc
       👤 Alex interviews you for clarification
       🏗️ Archie designs the architecture
       ⚖️ Judge ratifies the plan
       👩‍💻 Dev team builds 16 files across 4 modules
       🧪 Quinn runs quality review
       🎨 Pixel polishes code style
       📦 Release packages everything with docs

Output: projects/build_a_url_shortener_rest_api_with_rate/
        ├── src/          (16 production files)
        ├── docs/         (PRD, architecture, contract, release notes)
        ├── memory/       (agent learnings for next time)
        └── checkpoints/  (resume from any phase)
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
```

### 3. Build something

```bash
# Interactive mode (with welcome intake + sign-offs)
hive "Build a URL shortener REST API with rate limiting"

# Auto mode (skip sign-offs — great for testing)
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

Hive runs a **12-phase pipeline** inspired by real-world product development:

| # | Phase | Agent | What Happens |
|---|-------|-------|-------------|
| 1 | **Welcome** | System | Collect user identity, context, as-is process |
| 2 | **Ingest** | System | Import attached files, repos, external knowledge |
| 3 | **Research** | 🔍 Scout | Analyze problem space, tech landscape, constraints |
| 4 | **Interview** | 👤 Alex | Ask clarifying questions (auto-answered or interactive) |
| 5 | **PRD** | 📋 Penny | Write product requirements document |
| 6 | **Feasibility** | 🔍 Scout | Assess technical feasibility, estimate effort |
| 7 | **Architecture** | 🏗️ Archie | Design system architecture, file plan, dep graph |
| 8 | **Ratification** | ⚖️ Judge | Review & approve (or reject) the plan |
| 9 | **Crew** | System | Assemble dev team based on project needs |
| 10 | **Build** | 👩‍💻 Devs | Write all source files (with Quinn & Pixel review) |
| 11 | **Integration** | 🧪 Quinn | Full integration review of all code together |
| 12 | **Release** | 📋 Penny | Generate release notes, final packaging |

**Every phase saves a checkpoint.** If the process crashes, hits an API error, or you just want to pause — resume from where you left off.

### The Quality Loop

During the Build phase, every file goes through a multi-agent review chain:

```
Dev writes code → Quinn reviews (QA) → Pixel reviews (style) → Alex reviews (UX)
                       ↓ issues found?
                  Dev rewrites (up to 2 rounds)
```

### Memory System

Hive has a **3-tier memory system** that makes agents smarter over time:

| Layer | Scope | Purpose |
|-------|-------|---------|
| **Agent Memory** | Per agent, per project | Mistakes, patterns, lessons learned |
| **Team Memory** | Shared across agents | Cross-agent insights and warnings |
| **Global Memory** | Across ALL projects | Distilled lessons that load into future runs |

After each project, memories are automatically distilled into compact global lessons.

## The Crew

| Agent | Role | Personality |
|-------|------|-------------|
| 🔍 **Scout** | Research Analyst | Thorough, data-driven, identifies risks early |
| 📋 **Penny** | Product Manager | User-focused, writes clear specs and docs |
| 👤 **Alex** | UX Lead | Empathetic, asks the right questions |
| 🏗️ **Archie** | Architect | Pragmatic, designs for maintainability |
| ⚖️ **Judge** | Technical Director | Experienced, ratifies or blocks plans |
| 🧪 **Quinn** | QA Engineer | Meticulous, catches edge cases |
| 🎨 **Pixel** | Code Stylist | Clean code advocate, consistency enforcer |
| 🧭 **Flow** | Scrum Master | Tracks progress, manages handoffs |
| 👩‍💻 **Dev 1-N** | Developers | Specialists with distinct personalities |

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
  --auto                Auto-approve all sign-offs (for testing)
  --attach PATH         Attach knowledge files/folders (repeatable)
  --repo URL            Clone & study a git repo as reference (repeatable)
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
| `LLM_MODEL_BIG` | Same as `LLM_MODEL` | Model for heavy reasoning tasks |
| `LLM_MODEL_SMALL` | Same as `LLM_MODEL` | Model for light classification tasks |
| `LLM_FORMAT` | `auto` | Force format: `anthropic`, `openai`, or `auto` |
| `PROJECTS_DIR` | `./projects` | Where projects are saved |
| `HIVE_MIN_DISK_MB` | `50` | Minimum free disk space before saves |
| `HIVE_LOG_LEVEL` | `WARNING` | Default log level |
| `NO_COLOR` | — | Disable ANSI colors (any value) |

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

## Project Output

Each run produces a complete project directory:

```
projects/<slug>/
├── src/              # Generated source code
├── docs/
│   ├── prd.md        # Product requirements
│   ├── architecture.md
│   ├── contract.md   # File-by-file build contract
│   ├── release_notes.md
│   ├── signoffs.json # Agent sign-off history
│   ├── logbook.json  # Full activity log
│   └── ...
├── memory/           # Agent learning data
│   ├── agent_*.json
│   └── team.json
└── checkpoints/      # Resumable state snapshots
    ├── board_*.json
    └── board_latest.json
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

278 tests cover state management, agent logic, prompt parsing, UI rendering, connectors, memory, checkpoints, hardening utilities, and more — all without making real API calls.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation including:

- Module map with responsibilities
- 13 design patterns used
- Data flow diagrams
- Checkpoint/resume mechanics
- Memory distillation process

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
