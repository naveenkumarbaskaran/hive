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
  <img alt="Tests" src="https://img.shields.io/badge/tests-278%20passing-brightgreen">
  <img alt="Dependencies" src="https://img.shields.io/badge/deps-1%20(httpx)-orange">
</p>

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
ept "Build a URL shortener REST API with rate limiting"

# Auto mode (skip sign-offs — great for testing)
ept --auto "Build a CLI todo app with SQLite"

# Attach existing docs as context
ept --attach ./api-spec.yaml --attach ./design-doc.md "Build the payment service"

# Study an existing repo and build something similar
ept --repo https://github.com/user/project "Build something similar for healthcare"

# Resume from where you left off
ept --resume projects/my_project/checkpoints/board_latest.json

# List all projects
ept --list-projects
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
usage: ept [-h] [--resume PATH] [--list-projects] [-v] [--auto]
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
ept "Build a markdown parser in Rust"

# With external context
ept --attach ./openapi.yaml "Build a Python SDK for this API"

# Study a repo + build
ept --repo https://github.com/pallets/flask "Build a similar microframework for Go"

# Debug mode
ept --log-level DEBUG --verbose "Build a chat server"

# CI/CD friendly (no interactive prompts)
ept --auto "Build a REST API for widgets"
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
| `EPT_MIN_DISK_MB` | `50` | Minimum free disk space before saves |
| `EPT_LOG_LEVEL` | `WARNING` | Default log level |
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
