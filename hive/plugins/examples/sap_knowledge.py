"""
Example: SAP Domain Knowledge Plugin

Demonstrates how to provide domain-specific knowledge to the Hive crew.
This example shows SAP module knowledge — replace with your own domain.

Usage:
  hive --plugin ./hive/plugins/examples/sap_knowledge.py "Build an SAP integration..."

Or place in ./plugins/ directory for auto-discovery.
"""

from __future__ import annotations

from hive.plugins.base import PluginContext, PluginMeta


class SAPKnowledgePlugin:
    """Provides SAP domain knowledge to the crew.

    A real implementation would:
      - Read from SAP documentation repos
      - Query SAP systems via RFC / OData
      - Load ABAP naming conventions
      - Provide BAPI/RFC/IDoc reference data
    """

    meta = PluginMeta(
        name="sap-knowledge",
        version="0.1.0",
        description="SAP domain knowledge — modules, BAPIs, naming conventions",
        author="Hive Examples",
        category="knowledge",
    )

    def get_knowledge(self, ctx: PluginContext) -> list[dict]:
        """Return SAP knowledge items relevant to the feature."""
        items: list[dict] = []

        # Example: inject SAP naming conventions if feature mentions SAP
        feature_lower = ctx.feature.lower()
        if "sap" in feature_lower or "abap" in feature_lower:
            items.append({
                "source_type": "document",
                "source_path": "plugin://sap-knowledge/naming-conventions",
                "label": "SAP Naming Conventions",
                "content": (
                    "## SAP/ABAP Naming Conventions\n\n"
                    "- Custom objects: prefix with Z or Y (e.g., ZTABLE, YBAPI)\n"
                    "- Function modules: Z_<MODULE>_<VERB>_<OBJECT>\n"
                    "- Classes: ZCL_<MODULE>_<PURPOSE>\n"
                    "- Interfaces: ZIF_<MODULE>_<PURPOSE>\n"
                    "- Table types: ZTT_<DESCRIPTION>\n"
                    "- Data elements: ZDE_<DESCRIPTION>\n"
                    "- Domains: ZDOM_<DESCRIPTION>\n"
                ),
                "raw_size": 400,
                "tags": ["sap", "naming", "conventions"],
                "metadata": {"domain": "SAP", "module": "basis"},
            })

        if "mm" in feature_lower or "material" in feature_lower:
            items.append({
                "source_type": "document",
                "source_path": "plugin://sap-knowledge/mm-overview",
                "label": "SAP MM Module Overview",
                "content": (
                    "## SAP Materials Management (MM)\n\n"
                    "Key BAPIs:\n"
                    "- BAPI_MATERIAL_GETLIST — search materials\n"
                    "- BAPI_MATERIAL_GET_DETAIL — read material master\n"
                    "- BAPI_PO_CREATE1 — create purchase orders\n"
                    "- BAPI_PR_CREATE — create purchase requisitions\n\n"
                    "Key tables: MARA, MARC, MARD, EKKO, EKPO, EBAN\n"
                ),
                "raw_size": 350,
                "tags": ["sap", "mm", "materials"],
                "metadata": {"domain": "SAP", "module": "MM"},
            })

        return items
