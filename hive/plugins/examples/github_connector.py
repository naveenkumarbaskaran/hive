"""
Example: GitHub System Connector Plugin

Demonstrates how to connect to external systems.
This example shows GitHub — replace with your own system (JIRA, SAP, Docker, etc.)

Usage:
  hive --plugin ./hive/plugins/examples/github_connector.py "Build a ..."
"""

from __future__ import annotations

from typing import Any

from hive.plugins.base import PluginContext, PluginMeta


class GitHubConnectorPlugin:
    """Connects to GitHub for repository operations.

    A real implementation would:
      - Use PyGithub or httpx to call the GitHub API
      - Create repos, branches, PRs, issues
      - Read repo metadata and CI status
      - Post generated code as PRs

    Supported actions:
      - create_repo(name, description, private)
      - create_pr(title, body, head, base)
      - create_issue(title, body, labels)
      - get_repo_info()
    """

    meta = PluginMeta(
        name="github-connector",
        version="0.1.0",
        description="GitHub system connector — repos, PRs, issues",
        author="Hive Examples",
        category="system",
    )

    def __init__(self) -> None:
        self._connected = False
        self._token: str = ""
        self._repo: str = ""

    def connect(self, ctx: PluginContext) -> bool:
        """Connect to GitHub using token from config."""
        import os
        self._token = ctx.config.get("github_token", "") or os.getenv("GITHUB_TOKEN", "")
        self._repo = ctx.config.get("github_repo", "")

        if not self._token:
            return False

        self._connected = True
        return True

    def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a GitHub action."""
        if not self._connected:
            return {"error": "Not connected. Call connect() first."}

        # In a real implementation, these would call the GitHub API
        match action:
            case "create_repo":
                return {"status": "ok", "action": "create_repo",
                        "message": f"Would create repo: {params.get('name', '?')}"}
            case "create_pr":
                return {"status": "ok", "action": "create_pr",
                        "message": f"Would create PR: {params.get('title', '?')}"}
            case "create_issue":
                return {"status": "ok", "action": "create_issue",
                        "message": f"Would create issue: {params.get('title', '?')}"}
            case "get_repo_info":
                return {"status": "ok", "repo": self._repo,
                        "connected": self._connected}
            case _:
                return {"error": f"Unknown action: {action}"}

    def disconnect(self) -> None:
        """Clean up."""
        self._connected = False
        self._token = ""
