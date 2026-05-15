"""
Plugin Registry — Discovery, loading, and lifecycle management.

Discovers plugins from three sources (in order):
  1. Explicit paths passed via ``--plugin ./path/to/plugin.py``
  2. Directory scan of ``HIVE_PLUGINS_DIR`` (default: ``./plugins/``)
  3. Python entry points under the ``hive.plugins`` group

All discovery is lazy and optional. If no plugins are found, the
registry is empty and the core pipeline is unaffected.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hive.connectors import KnowledgeItem
from hive.plugins.base import (
    PluginContext,
    PluginMeta,
)

logger = logging.getLogger("hive.plugins")


# ─────────────────────────────────────────────────────────────────────────────
#  Plugin wrapper — holds a loaded plugin + its resolved category
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedPlugin:
    """A successfully loaded and categorized plugin instance."""
    instance: Any
    meta: PluginMeta
    categories: list[str] = field(default_factory=list)  # e.g. ["knowledge", "lifecycle"]


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────

class PluginRegistry:
    """Discovers, loads, and manages Hive plugins.

    Usage:
        registry = PluginRegistry()
        registry.discover()                    # auto-discover from all sources
        registry.load_from_path("./my.py")     # explicit load

        # Gather knowledge from all KnowledgePlugins
        items = registry.gather_knowledge(ctx)

        # Get guidelines text from all GuidelinesPlugins
        text = registry.gather_guidelines(ctx)

        # Fire lifecycle hooks
        registry.on_phase_start("build", ctx)
    """

    def __init__(self) -> None:
        self._plugins: dict[str, LoadedPlugin] = {}
        self._discovered = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def names(self) -> list[str]:
        """All loaded plugin names."""
        return list(self._plugins.keys())

    @property
    def count(self) -> int:
        return len(self._plugins)

    def get(self, name: str) -> LoadedPlugin | None:
        return self._plugins.get(name)

    def __bool__(self) -> bool:
        return len(self._plugins) > 0

    def __len__(self) -> int:
        return len(self._plugins)

    # ── Category queries ──────────────────────────────────────────────────────

    def _by_category(self, category: str) -> list[LoadedPlugin]:
        return [p for p in self._plugins.values() if category in p.categories]

    @property
    def knowledge_plugins(self) -> list[LoadedPlugin]:
        return self._by_category("knowledge")

    @property
    def guidelines_plugins(self) -> list[LoadedPlugin]:
        return self._by_category("guidelines")

    @property
    def system_plugins(self) -> list[LoadedPlugin]:
        return self._by_category("system")

    @property
    def testdata_plugins(self) -> list[LoadedPlugin]:
        return self._by_category("testdata")

    @property
    def lifecycle_plugins(self) -> list[LoadedPlugin]:
        return self._by_category("lifecycle")

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover(
        self,
        explicit_paths: list[str] | None = None,
        plugins_dir: str | None = None,
    ) -> int:
        """Auto-discover plugins from all sources.

        Args:
            explicit_paths: Paths from ``--plugin`` CLI args.
            plugins_dir: Override ``HIVE_PLUGINS_DIR`` env var.

        Returns:
            Number of plugins loaded.
        """
        count_before = self.count

        # 1. Explicit paths (highest priority)
        for path_str in (explicit_paths or []):
            try:
                self.load_from_path(path_str)
            except Exception as exc:
                logger.warning("Failed to load plugin from %s: %s", path_str, exc)

        # 2. Plugins directory
        pdir = plugins_dir or os.getenv("HIVE_PLUGINS_DIR", "./plugins")
        pdir_path = Path(pdir)
        if pdir_path.is_dir():
            self._discover_directory(pdir_path)

        # 3. Entry points (pip-installed plugins)
        self._discover_entry_points()

        self._discovered = True
        loaded = self.count - count_before
        if loaded:
            logger.info("Discovered %d plugin(s): %s", loaded, ", ".join(self.names))
        return loaded

    def _discover_directory(self, directory: Path) -> None:
        """Scan a directory for Python plugin files."""
        for child in sorted(directory.iterdir()):
            if child.suffix == ".py" and not child.name.startswith("_"):
                try:
                    self.load_from_path(str(child))
                except Exception as exc:
                    logger.warning("Failed to load plugin %s: %s", child.name, exc)
            elif child.is_dir() and (child / "__init__.py").exists():
                # It's a package — try to import it
                try:
                    self.load_from_path(str(child / "__init__.py"))
                except Exception as exc:
                    logger.warning("Failed to load plugin package %s: %s", child.name, exc)

    def _discover_entry_points(self) -> None:
        """Load plugins registered via Python entry points."""
        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            # Python 3.12+ returns a SelectableGroups or dict
            group = eps.select(group="hive.plugins") if hasattr(eps, "select") else eps.get("hive.plugins", [])
            for ep in group:
                if ep.name in self._plugins:
                    continue  # already loaded
                try:
                    plugin_cls = ep.load()
                    self._register_instance(plugin_cls(), source=f"entrypoint:{ep.name}")
                except Exception as exc:
                    logger.warning("Failed to load entry point %s: %s", ep.name, exc)
        except Exception:
            pass  # entry_points not available or failed — that's fine

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_from_path(self, path_str: str) -> str | None:
        """Load a plugin from a Python file path.

        Looks for any class with a ``meta`` attribute of type ``PluginMeta``
        and methods matching one of the plugin protocols.

        Returns the plugin name on success, None on failure.
        """
        path = Path(path_str).resolve()
        if not path.exists():
            logger.warning("Plugin path does not exist: %s", path)
            return None

        # Import the module
        module_name = f"hive_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.warning("Cannot create module spec for %s", path)
            return None

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("Error executing plugin module %s: %s", path.name, exc)
            return None

        # Scan for plugin classes
        loaded_name = None
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, type) and hasattr(obj, "meta"):
                try:
                    instance = obj()
                    name = self._register_instance(instance, source=str(path))
                    if name:
                        loaded_name = name
                except Exception as exc:
                    logger.warning("Failed to instantiate %s from %s: %s",
                                   attr_name, path.name, exc)
        return loaded_name

    def load_instance(self, instance: Any) -> str | None:
        """Register a pre-instantiated plugin object.

        Useful for testing or programmatic plugin loading.
        """
        return self._register_instance(instance, source="direct")

    def _register_instance(self, instance: Any, source: str = "") -> str | None:
        """Register a plugin instance, auto-detecting its categories."""
        meta = getattr(instance, "meta", None)
        if not isinstance(meta, PluginMeta):
            logger.warning("Plugin from %s has no valid PluginMeta", source)
            return None

        if meta.name in self._plugins:
            logger.debug("Plugin %s already loaded, skipping", meta.name)
            return meta.name

        categories = _detect_categories(instance)
        if not categories:
            logger.warning("Plugin %s (%s) doesn't implement any known protocol",
                          meta.name, source)
            return None

        self._plugins[meta.name] = LoadedPlugin(
            instance=instance,
            meta=meta,
            categories=categories,
        )
        logger.info("Loaded plugin: %s (%s) categories=%s source=%s",
                    meta.name, meta.description, categories, source)
        return meta.name

    # ── Invocation helpers ────────────────────────────────────────────────────

    def gather_knowledge(self, ctx: PluginContext) -> list[KnowledgeItem]:
        """Collect KnowledgeItems from all KnowledgePlugins.

        Returns a combined, deduplicated list of items.
        """
        items: list[KnowledgeItem] = []
        seen: set[str] = set()

        for lp in self.knowledge_plugins:
            try:
                raw = lp.instance.get_knowledge(ctx)
                for item in raw:
                    # Accept dicts or KnowledgeItem-like objects
                    ki = _to_knowledge_item(item, plugin_name=lp.meta.name)
                    if ki and ki.source_path not in seen:
                        seen.add(ki.source_path)
                        items.append(ki)
            except Exception as exc:
                logger.warning("Plugin %s.get_knowledge() failed: %s",
                              lp.meta.name, exc)
        return items

    def gather_guidelines(self, ctx: PluginContext) -> str:
        """Collect guidelines text from all GuidelinesPlugins.

        Returns concatenated guidelines with headers per plugin.
        """
        parts: list[str] = []
        for lp in self.guidelines_plugins:
            try:
                text = lp.instance.get_guidelines(ctx)
                if text and text.strip():
                    parts.append(
                        f"## Guidelines: {lp.meta.name}\n{text.strip()}"
                    )
            except Exception as exc:
                logger.warning("Plugin %s.get_guidelines() failed: %s",
                              lp.meta.name, exc)
        return "\n\n".join(parts)

    def gather_test_data(self, ctx: PluginContext,
                         schema: dict[str, Any] | None = None) -> list[KnowledgeItem]:
        """Collect test data items from all TestDataPlugins."""
        items: list[KnowledgeItem] = []
        for lp in self.testdata_plugins:
            try:
                raw = lp.instance.get_test_data(ctx, schema)
                for item in raw:
                    ki = _to_knowledge_item(item, plugin_name=lp.meta.name)
                    if ki:
                        items.append(ki)
            except Exception as exc:
                logger.warning("Plugin %s.get_test_data() failed: %s",
                              lp.meta.name, exc)
        return items

    def on_phase_start(self, phase: str, ctx: PluginContext) -> None:
        """Fire on_phase_start for all LifecyclePlugins."""
        for lp in self.lifecycle_plugins:
            try:
                lp.instance.on_phase_start(phase, ctx)
            except Exception as exc:
                logger.warning("Plugin %s.on_phase_start(%s) failed: %s",
                              lp.meta.name, phase, exc)

    def on_phase_end(self, phase: str, ctx: PluginContext) -> None:
        """Fire on_phase_end for all LifecyclePlugins."""
        for lp in self.lifecycle_plugins:
            try:
                lp.instance.on_phase_end(phase, ctx)
            except Exception as exc:
                logger.warning("Plugin %s.on_phase_end(%s) failed: %s",
                              lp.meta.name, phase, exc)

    def connect_systems(self, ctx: PluginContext) -> dict[str, bool]:
        """Connect all SystemPlugins. Returns {name: success}."""
        results: dict[str, bool] = {}
        for lp in self.system_plugins:
            try:
                ok = lp.instance.connect(ctx)
                results[lp.meta.name] = bool(ok)
            except Exception as exc:
                logger.warning("Plugin %s.connect() failed: %s",
                              lp.meta.name, exc)
                results[lp.meta.name] = False
        return results

    def disconnect_systems(self) -> None:
        """Disconnect all SystemPlugins."""
        for lp in self.system_plugins:
            try:
                lp.instance.disconnect()
            except Exception as exc:
                logger.warning("Plugin %s.disconnect() failed: %s",
                              lp.meta.name, exc)

    def execute_system(self, plugin_name: str, action: str,
                       params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute an action on a specific SystemPlugin."""
        lp = self._plugins.get(plugin_name)
        if not lp or "system" not in lp.categories:
            return {"error": f"System plugin '{plugin_name}' not found"}
        try:
            return lp.instance.execute(action, params or {})
        except Exception as exc:
            return {"error": str(exc)}

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of loaded plugins."""
        if not self._plugins:
            return "No plugins loaded."
        lines = [f"Loaded {self.count} plugin(s):"]
        for name, lp in self._plugins.items():
            cats = ", ".join(lp.categories)
            desc = lp.meta.description or "(no description)"
            lines.append(f"  • {name} [{cats}] — {desc}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Category detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_categories(instance: Any) -> list[str]:
    """Detect which plugin protocols an instance satisfies."""
    cats: list[str] = []
    if _has_method(instance, "get_knowledge"):
        cats.append("knowledge")
    if _has_method(instance, "get_guidelines"):
        cats.append("guidelines")
    if _has_method(instance, "connect") and _has_method(instance, "execute"):
        cats.append("system")
    if _has_method(instance, "get_test_data"):
        cats.append("testdata")
    if _has_method(instance, "on_phase_start") and _has_method(instance, "on_phase_end"):
        cats.append("lifecycle")
    return cats


def _has_method(obj: Any, name: str) -> bool:
    """Check if an object has a callable attribute with the given name."""
    attr = getattr(obj, name, None)
    return callable(attr)


# ─────────────────────────────────────────────────────────────────────────────
#  KnowledgeItem coercion
# ─────────────────────────────────────────────────────────────────────────────

def _to_knowledge_item(obj: Any, plugin_name: str = "") -> KnowledgeItem | None:
    """Coerce a plugin-returned object into a KnowledgeItem.

    Accepts:
      - KnowledgeItem instances (pass-through)
      - Dicts with the right keys
      - Objects with the right attributes
    """
    if isinstance(obj, KnowledgeItem):
        if plugin_name:
            obj.tags.append(f"plugin:{plugin_name}")
        return obj

    if isinstance(obj, dict):
        try:
            ki = KnowledgeItem(
                source_type=obj.get("source_type", "document"),
                source_path=obj.get("source_path", f"plugin://{plugin_name}"),
                label=obj.get("label", "plugin-item"),
                content=obj.get("content", ""),
                raw_size=obj.get("raw_size", len(obj.get("content", ""))),
                was_summarized=obj.get("was_summarized", False),
                summary=obj.get("summary", ""),
                tags=obj.get("tags", []) + [f"plugin:{plugin_name}"],
                metadata=obj.get("metadata", {}),
            )
            return ki
        except Exception:
            return None

    # Object with attributes
    try:
        ki = KnowledgeItem(
            source_type=getattr(obj, "source_type", "document"),
            source_path=getattr(obj, "source_path", f"plugin://{plugin_name}"),
            label=getattr(obj, "label", "plugin-item"),
            content=getattr(obj, "content", ""),
            raw_size=getattr(obj, "raw_size", 0),
            was_summarized=getattr(obj, "was_summarized", False),
            summary=getattr(obj, "summary", ""),
            tags=list(getattr(obj, "tags", [])) + [f"plugin:{plugin_name}"],
            metadata=dict(getattr(obj, "metadata", {})),
        )
        return ki
    except Exception:
        return None
