"""
Hive — Collective Intelligence Building Software

A lightweight multi-agent SDLC framework where named AI agents
collaborate to build software. Your AI dev crew, assembled.

Internal package name: hive (EPT — Empowered Product Team)
"""

__version__ = "1.0.0"

from hive.agents import DEV_POOL, Agent, AgentRoster, make_dev_agent
from hive.connectors import ConnectorRegistry, KnowledgeItem
from hive.crew import EPTCrew
from hive.memory import MemoryEntry, MemoryManager
from hive.sandbox import Sandbox, SandboxResult, run_code_checks, syntax_check_file
from hive.state import (
    Amendment,
    Blackboard,
    Event,
    EventType,
    FileEntry,
    Issue,
    ResearchContext,
    SignOff,
    list_projects,
    load_checkpoint,
    save_checkpoint,
)
from hive.telemetry import BudgetExceeded, CostTracker, estimate_cost, model_context_window
from hive.ui import TerminalUI

__all__ = [
    "__version__",
    "EPTCrew", "Agent", "AgentRoster", "DEV_POOL", "make_dev_agent",
    "Blackboard", "Event", "EventType", "ResearchContext", "FileEntry",
    "Issue", "Amendment", "SignOff",
    "save_checkpoint", "load_checkpoint", "list_projects",
    "TerminalUI",
    "MemoryManager", "MemoryEntry",
    "ConnectorRegistry", "KnowledgeItem",
    "Sandbox", "SandboxResult", "run_code_checks", "syntax_check_file",
    "CostTracker", "BudgetExceeded", "estimate_cost", "model_context_window",
]
