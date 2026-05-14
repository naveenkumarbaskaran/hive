#!/usr/bin/env python3
"""
Hive — Your AI Dev Crew, Assembled

CLI entry point. Run an AI-powered product team against a feature request.

Usage:
  hive "Build a rate-limited REST API for user registration"
  hive --attach ./docs/ --attach ./api/swagger.yaml "Build a payment gateway"
  hive --repo https://github.com/user/project "Build something similar for X"
  hive --resume projects/a_rate_limited_rest_api/checkpoints/board_latest.json
  hive --list-projects
  hive --verbose "..."
  hive --auto "..."   # skip sign-offs (for testing)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env file from project root if it exists (no dependencies)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not os.environ.get(key):  # don't override existing env vars
                os.environ[key] = value


_load_dotenv()

from hive.llm_client import llm
from hive.crew import EPTCrew
from hive.state import list_projects, load_checkpoint
from hive.memory import MemoryManager
from hive.hardening import setup_logging


def main() -> None:
    from hive import __version__

    parser = argparse.ArgumentParser(
        prog="hive",
        description="Hive — Your AI Dev Crew, Assembled",
        epilog="Documentation: https://github.com/naveenkumarbaskaran/hive",
    )
    parser.add_argument("feature", nargs="?", help="Feature request to build")
    parser.add_argument("--resume", metavar="PATH", help="Resume from checkpoint JSON")
    parser.add_argument("--list-projects", action="store_true", help="List existing projects")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-approve all sign-offs (for testing)")
    parser.add_argument("--attach", action="append", default=[],
                        metavar="PATH",
                        help="Attach knowledge files/folders (repeatable)")
    parser.add_argument("--repo", action="append", default=[],
                        metavar="URL",
                        help="Clone & study a git repo as reference (repeatable)")
    parser.add_argument("--log-level", metavar="LEVEL", default=None,
                        help="Log level: DEBUG, INFO, WARNING, ERROR (default: WARNING)")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__} (Hive)")

    args = parser.parse_args()

    # ── Initialize structured logging ──
    setup_logging(args.log_level)

    # ── List projects ──
    if args.list_projects:
        projects = list_projects()
        if not projects:
            print("No projects found.")
        else:
            print(f"\n{'Slug':30} {'Feature':40} {'Phase':15}")
            print("─" * 85)
            for p in projects:
                print(f"{p['slug']:30} {p.get('feature', '?'):40} {p.get('current_phase', '?'):15}")
        return

    # ── Resume ──
    if args.resume:
        board = load_checkpoint(args.resume)
        crew = EPTCrew(
            feature=board.feature,
            client=llm,
            verbose=args.verbose,
            auto_approve=args.auto,
            attach_paths=args.attach,
            repo_urls=args.repo,
        )
        crew.board = board
        crew.ui.board = board  # sync UI with restored board
        # Re-initialize memory system for the resumed project
        crew.memory = MemoryManager(
            project_slug=board.project_slug,
            memory_dir=board.memory_dir,
        )
        crew.memory.load_global()
        crew.memory.load()
        # Rehydrate crew agents if the crew phase was already completed
        if "crew" in board.completed_phases:
            from hive.agents import AgentRoster
            crew.agents = AgentRoster.compose(
                has_frontend=board.research.has_frontend,
                dev_count=board.dev_count or 1,
            )
        print(f"Resuming project: {board.feature}")
        print(f"Last phase: {board.current_phase}")
        print(f"Completed: {', '.join(board.completed_phases)}")
        crew.run()
        return

    # ── New project ──
    if not args.feature:
        parser.print_help()
        print("\n  Error: provide a feature request or --resume/--list-projects")
        sys.exit(1)

    crew = EPTCrew(
        feature=args.feature,
        client=llm,
        verbose=args.verbose,
        auto_approve=args.auto,
        attach_paths=args.attach,
        repo_urls=args.repo,
    )
    crew.run()


if __name__ == "__main__":
    main()
