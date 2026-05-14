"""
Hive — Collective Intelligence Building Software

A lightweight multi-agent SDLC framework where named AI agents
collaborate to build software. Your AI dev crew, assembled.

Internal package name: hive (EPT — Empowered Product Team)
"""

__version__ = "1.0.0"

from hive.agents import Agent, AgentRoster, DEV_POOL, make_dev_agent
from hive.state import (
    Blackboard, Event, EventType, ResearchContext, FileEntry,
    Issue, Amendment, SignOff, save_checkpoint, load_checkpoint, list_projects,
)
from hive.crew import EPTCrew
from hive.ui import TerminalUI
from hive.memory import MemoryManager, MemoryEntry
from hive.connectors import ConnectorRegistry, KnowledgeItem

__all__ = [
    "__version__",
    "EPTCrew", "Agent", "AgentRoster", "DEV_POOL", "make_dev_agent",
    "Blackboard", "Event", "EventType", "ResearchContext", "FileEntry",
    "Issue", "Amendment", "SignOff",
    "save_checkpoint", "load_checkpoint", "list_projects",
    "TerminalUI",
    "MemoryManager", "MemoryEntry",
    "ConnectorRegistry", "KnowledgeItem",
]
