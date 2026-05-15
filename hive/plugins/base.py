"""
Plugin protocols — structural typing for all plugin categories.

Uses Python Protocols so plugin authors don't need to import or inherit
from Hive. Any object that has the right attributes/methods is a valid
plugin (duck typing).

Plugin categories:
  KnowledgePlugin   — Domain knowledge (SAP modules, Salesforce objects, ...)
  GuidelinesPlugin   — Coding/project rules (linting, style, company standards)
  SystemPlugin       — External system connectors (GitHub, Docker, JIRA, ...)
  TestDataPlugin     — Test fixtures, mock data, seed data generators
  LifecyclePlugin    — Pre/post hooks for any pipeline phase
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ─────────────────────────────────────────────────────────────────────────────
#  Plugin metadata — every plugin must provide this
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PluginMeta:
    """Metadata that every plugin must declare."""
    name: str                                 # unique id, e.g. "sap-knowledge"
    version: str = "0.1.0"                    # semver
    description: str = ""                     # one-liner
    author: str = ""                          # optional
    category: str = ""                        # auto-detected from protocol
    config_schema: dict[str, Any] = field(default_factory=dict)  # optional JSON-schema for config


# ─────────────────────────────────────────────────────────────────────────────
#  Plugin context — passed to plugins at invocation time
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PluginContext:
    """Context provided to plugins when they are invoked.

    Contains the information plugins need without exposing internal
    Blackboard internals. Plugins should not depend on hive.state.
    """
    feature: str = ""                         # the user's feature request
    stack: list[str] = field(default_factory=list)  # detected tech stack
    phase: str = ""                           # current pipeline phase
    project_slug: str = ""                    # project directory name
    config: dict[str, Any] = field(default_factory=dict)  # per-plugin config
    extra: dict[str, Any] = field(default_factory=dict)  # open-ended extras


# ─────────────────────────────────────────────────────────────────────────────
#  Knowledge items — reuse the core connector type
# ─────────────────────────────────────────────────────────────────────────────

# Plugins produce the same KnowledgeItem structure so items flow through
# the existing Blackboard routing.  Import deferred to avoid circular deps:
#   from hive.connectors import KnowledgeItem


# ─────────────────────────────────────────────────────────────────────────────
#  Protocol definitions
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class KnowledgePlugin(Protocol):
    """Provides domain-specific knowledge items.

    Examples:
      - SAP module documentation for MM/SD/FI
      - Salesforce object schemas
      - Industry-specific regulatory docs
      - Internal wiki / Confluence pages
    """

    meta: PluginMeta

    def get_knowledge(self, ctx: PluginContext) -> list[Any]:
        """Return a list of KnowledgeItem-compatible objects.

        Each item should have at minimum:
          source_type, source_path, label, content, raw_size
        """
        ...


@runtime_checkable
class GuidelinesPlugin(Protocol):
    """Provides coding guidelines, linting rules, or project standards.

    Examples:
      - Company coding standards
      - Linting configuration (ESLint, Ruff, etc.)
      - Architecture decision records (ADRs)
      - API design guidelines
    """

    meta: PluginMeta

    def get_guidelines(self, ctx: PluginContext) -> str:
        """Return guidelines as a formatted string.

        The returned text is injected into agent context so the crew
        follows your project/company rules automatically.
        """
        ...


@runtime_checkable
class SystemPlugin(Protocol):
    """Connects to an external system for read/write operations.

    Examples:
      - GitHub: create repos, PRs, issues
      - Docker: build images, run containers
      - JIRA/Azure DevOps: create work items
      - SAP: read/write business objects via RFC/OData
      - Databases: seed data, run migrations
    """

    meta: PluginMeta

    def connect(self, ctx: PluginContext) -> bool:
        """Establish connection. Returns True on success."""
        ...

    def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a named action with parameters.

        Returns a result dict. The schema is plugin-defined.
        Actions are plugin-specific, e.g.:
          - github.create_pr(title=..., body=...)
          - docker.build(dockerfile=..., tag=...)
          - sap.read_bapi(name=..., params=...)
        """
        ...

    def disconnect(self) -> None:
        """Clean up connection resources."""
        ...


@runtime_checkable
class TestDataPlugin(Protocol):
    """Provides test data, fixtures, or mock generators.

    Examples:
      - Generate realistic user data for testing
      - Provide SAP test master data (materials, vendors, etc.)
      - Load seed data from CSV/JSON/YAML
      - Generate API response mocks
    """

    meta: PluginMeta

    def get_test_data(self, ctx: PluginContext, schema: dict[str, Any] | None = None) -> list[Any]:
        """Return test data items.

        Args:
            ctx: Plugin context with feature and stack info.
            schema: Optional schema hint describing what data is needed.

        Returns:
            List of KnowledgeItem-compatible objects with source_type="data_file".
        """
        ...


@runtime_checkable
class LifecyclePlugin(Protocol):
    """Hooks into pipeline phases for custom pre/post actions.

    Examples:
      - Notify Slack/Teams when a phase starts
      - Run linting after build phase
      - Deploy to staging after integration
      - Post metrics to a dashboard
    """

    meta: PluginMeta

    def on_phase_start(self, phase: str, ctx: PluginContext) -> None:
        """Called before a pipeline phase begins."""
        ...

    def on_phase_end(self, phase: str, ctx: PluginContext) -> None:
        """Called after a pipeline phase completes."""
        ...
