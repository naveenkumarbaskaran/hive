"""
Example: Company Coding Guidelines Plugin

Demonstrates how to inject project/company-specific coding rules
that the crew will follow automatically.

Usage:
  hive --plugin ./hive/plugins/examples/company_guidelines.py "Build a REST API..."
"""

from __future__ import annotations

from hive.plugins.base import PluginContext, PluginMeta


class CompanyGuidelinesPlugin:
    """Injects company-specific coding guidelines.

    A real implementation would:
      - Load from a shared repo / wiki
      - Filter by detected tech stack
      - Include linting configs
      - Include architecture decision records (ADRs)
    """

    meta = PluginMeta(
        name="company-guidelines",
        version="0.1.0",
        description="Company coding standards and project rules",
        author="Hive Examples",
        category="guidelines",
    )

    def get_guidelines(self, ctx: PluginContext) -> str:
        """Return coding guidelines tailored to the tech stack."""
        sections: list[str] = []

        # General rules (always apply)
        sections.append(
            "## General Rules\n"
            "- All code must be reviewed before merge\n"
            "- No hardcoded secrets — use environment variables\n"
            "- Write tests for all business logic (minimum 80% coverage)\n"
            "- Use conventional commits: feat:, fix:, test:, docs:\n"
            "- All public APIs must have docstrings\n"
        )

        # Stack-specific rules
        stack = {s.lower() for s in ctx.stack}
        if "python" in stack or any("py" in s for s in stack):
            sections.append(
                "## Python Standards\n"
                "- Use Ruff for linting and formatting\n"
                "- Type hints on all function signatures\n"
                "- Prefer dataclasses over plain dicts\n"
                "- Use f-strings over .format()\n"
                "- Line length: 100 characters\n"
            )

        if "typescript" in stack or "javascript" in stack:
            sections.append(
                "## TypeScript/JavaScript Standards\n"
                "- Use ESLint + Prettier\n"
                "- Prefer TypeScript over JavaScript\n"
                "- Use strict mode\n"
                "- No any types — use proper generics\n"
            )

        if "docker" in stack or "container" in ctx.feature.lower():
            sections.append(
                "## Container Standards\n"
                "- Use multi-stage builds\n"
                "- Run as non-root user\n"
                "- Pin base image versions\n"
                "- Use .dockerignore\n"
            )

        return "\n".join(sections)
