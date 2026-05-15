"""
Hive Plugin System — Optional, extensible connectors.

Provides a protocol-based plugin architecture for injecting:
  - Domain knowledge   (SAP, Salesforce, industry docs)
  - Coding guidelines  (linting rules, style guides, company standards)
  - External systems   (GitHub, Docker, CI/CD, SAP, JIRA, etc.)
  - Test data          (fixtures, mocks, seed data in any format)
  - Lifecycle hooks    (pre/post phase callbacks)

Plugins are **totally optional** — the core Hive pipeline works without them.

Discovery order:
  1. Explicit paths via ``--plugin ./my_plugin.py``
  2. ``HIVE_PLUGINS_DIR`` directory (default: ``./plugins/``)
  3. Python entry points under ``hive.plugins`` group

Usage (plugin author):
  from hive.plugins import KnowledgePlugin, PluginMeta

  class SAPKnowledgePlugin:
      meta = PluginMeta(name="sap-knowledge", version="1.0", ...)
      def get_knowledge(self, feature, context):
          return [KnowledgeItem(...)]

  # Register via entry point in pyproject.toml:
  [project.entry-points."hive.plugins"]
  sap-knowledge = "my_package:SAPKnowledgePlugin"
"""

from hive.plugins.base import (
    GuidelinesPlugin,
    KnowledgePlugin,
    LifecyclePlugin,
    PluginMeta,
    SystemPlugin,
    TestDataPlugin,
)
from hive.plugins.registry import PluginRegistry

__all__ = [
    "PluginMeta",
    "KnowledgePlugin",
    "GuidelinesPlugin",
    "SystemPlugin",
    "TestDataPlugin",
    "LifecyclePlugin",
    "PluginRegistry",
]
