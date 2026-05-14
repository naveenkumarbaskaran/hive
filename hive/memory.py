"""
EPT Memory System — Individual + Team + Global memory for learning agents.

Each agent has a personal memory that tracks mistakes, patterns, and
lessons learned. Agents can also push insights to a shared Team Memory
(visible to all agents within the same project).

At project completion, all memories are distilled into compact lessons
and saved to Global Memory, which loads into future projects so the
crew gets smarter over time.

Storage layout:
  projects/<slug>/memory/
    agent_<id>.json    — per-agent memories
    team.json          — team-shared memories
  projects/.global_memory.json  — cross-project distilled lessons
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from hive.hardening import atomic_write, file_lock

logger = logging.getLogger("hive.memory")


# ─────────────────────────────────────────────────────────────────────────────
#  Data models
# ─────────────────────────────────────────────────────────────────────────────

MemoryKind = Literal["mistake", "pattern", "lesson", "insight"]


@dataclass
class MemoryEntry:
    """A single memory — something an agent learned."""

    kind: str                        # MemoryKind: "mistake", "pattern", "lesson", "insight"
    content: str                     # what was learned
    context: str = ""                # what situation triggered it
    phase: str = ""                  # which phase it originated from
    agent_id: str = ""               # who created this memory
    tags: list[str] = field(default_factory=list)          # routing / lookup tags
    source_project: str = ""         # project slug where it originated
    timestamp: float = field(default_factory=time.time)

    @property
    def one_liner(self) -> str:
        """Compact string for display."""
        icon = {"mistake": "❌", "pattern": "✅", "lesson": "💡", "insight": "🔍"}.get(
            self.kind, "📝")
        return f"{icon} [{self.kind}] {self.content[:120]}"


@dataclass
class AgentMemory:
    """Personal memory store for one agent."""

    agent_id: str
    entries: list[MemoryEntry] = field(default_factory=list)

    # ── Queries ──

    @property
    def mistakes(self) -> list[MemoryEntry]:
        return [e for e in self.entries if e.kind == "mistake"]

    @property
    def patterns(self) -> list[MemoryEntry]:
        return [e for e in self.entries if e.kind == "pattern"]

    @property
    def lessons(self) -> list[MemoryEntry]:
        return [e for e in self.entries if e.kind == "lesson"]

    def for_phase(self, phase: str) -> list[MemoryEntry]:
        """Memories relevant to a specific phase."""
        return [e for e in self.entries if e.phase == phase or not e.phase]

    # ── Mutations ──

    def remember(
        self,
        kind: str,
        content: str,
        context: str = "",
        phase: str = "",
        tags: list[str] | None = None,
        source_project: str = "",
    ) -> MemoryEntry:
        """Record a new memory. Returns the entry."""
        entry = MemoryEntry(
            kind=kind,
            content=content,
            context=context,
            phase=phase,
            agent_id=self.agent_id,
            tags=tags or [],
            source_project=source_project,
        )
        self.entries.append(entry)
        return entry

    # ── Context rendering ──

    def context_block(self, phase: str = "", max_entries: int = 15) -> str:
        """Render memories as a context string for LLM prompts.

        Prioritizes: phase-relevant entries, then mistakes, then recent.
        """
        relevant = self.for_phase(phase) if phase else list(self.entries)
        if not relevant:
            return ""

        # Sort: phase-relevant mistakes first, then patterns, then by recency
        def _sort_key(e: MemoryEntry) -> tuple:
            phase_match = 0 if (e.phase == phase and phase) else 1
            kind_rank = {"mistake": 0, "lesson": 1, "pattern": 2, "insight": 3}.get(
                e.kind, 4)
            return (phase_match, kind_rank, -e.timestamp)

        relevant.sort(key=_sort_key)
        relevant = relevant[:max_entries]

        lines = [f"YOUR MEMORY ({self.agent_id}) — lessons from past work:"]
        for e in relevant:
            prefix = {"mistake": "AVOID", "pattern": "DO", "lesson": "KNOW", "insight": "NOTE"}.get(
                e.kind, "NOTE")
            line = f"  [{prefix}] {e.content}"
            if e.context:
                line += f" (context: {e.context})"
            lines.append(line)

        return "\n".join(lines)


@dataclass
class TeamMemory:
    """Shared memory board — agents push insights for the whole crew."""

    entries: list[MemoryEntry] = field(default_factory=list)

    def push(
        self,
        agent_id: str,
        content: str,
        kind: str = "insight",
        for_agents: list[str] | None = None,
        phase: str = "",
        tags: list[str] | None = None,
        source_project: str = "",
    ) -> MemoryEntry:
        """An agent pushes an insight to the team board."""
        entry = MemoryEntry(
            kind=kind,
            content=content,
            phase=phase,
            agent_id=agent_id,
            tags=(tags or []) + ([f"for:{a}" for a in for_agents] if for_agents else []),
            source_project=source_project,
        )
        self.entries.append(entry)
        return entry

    def for_agent(self, agent_id: str, phase: str = "", max_entries: int = 10) -> list[MemoryEntry]:
        """Get team memories relevant to a specific agent."""
        relevant = []
        for e in self.entries:
            # Skip entries the agent wrote themselves
            if e.agent_id == agent_id:
                continue
            # Check if explicitly targeted
            for_tags = [t for t in e.tags if t.startswith("for:")]
            if for_tags:
                # Entry is targeted — include only if this agent is named
                if f"for:{agent_id}" not in e.tags:
                    continue
            # Phase filter (include general entries too)
            if phase and e.phase and e.phase != phase:
                continue
            relevant.append(e)

        # Sort by recency
        relevant.sort(key=lambda e: -e.timestamp)
        return relevant[:max_entries]

    def context_block(self, agent_id: str, phase: str = "", max_entries: int = 10) -> str:
        """Render team memories as context for an agent's prompt."""
        entries = self.for_agent(agent_id, phase, max_entries)
        if not entries:
            return ""

        lines = ["TEAM MEMORY — insights from your colleagues:"]
        for e in entries:
            who = e.agent_id or "system"
            lines.append(f"  [{who}] {e.content}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Global Memory — cross-project distilled lessons
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalMemory:
    """Distilled lessons that carry across projects.

    Loaded at project start, updated at project end.
    Lives at projects/.global_memory.json.
    """

    lessons: list[MemoryEntry] = field(default_factory=list)

    def add_lesson(
        self,
        content: str,
        agent_id: str = "",
        tags: list[str] | None = None,
        source_project: str = "",
    ) -> MemoryEntry:
        """Add a distilled lesson from a completed project."""
        entry = MemoryEntry(
            kind="lesson",
            content=content,
            agent_id=agent_id,
            tags=tags or [],
            source_project=source_project,
        )
        self.lessons.append(entry)
        return entry

    def context_block(self, agent_id: str = "", max_lessons: int = 10) -> str:
        """Render global lessons as context for prompt injection.

        If agent_id is given, prioritizes lessons from that agent.
        """
        if not self.lessons:
            return ""

        entries = list(self.lessons)

        # Prioritize lessons from the same agent, then most recent
        def _sort_key(e: MemoryEntry) -> tuple:
            agent_match = 0 if e.agent_id == agent_id else 1
            return (agent_match, -e.timestamp)

        entries.sort(key=_sort_key)
        entries = entries[:max_lessons]

        lines = ["GLOBAL LESSONS — wisdom from past projects:"]
        for e in entries:
            source = f" (from: {e.source_project})" if e.source_project else ""
            lines.append(f"  💡 {e.content}{source}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Memory Manager — wires everything together
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_MEMORY_PATH = Path("./projects/.global_memory.json")


class MemoryManager:
    """Manages all memory tiers for a project.

    Usage:
        mm = MemoryManager(project_slug="my_project")
        mm.load_global()  # at start
        mm.get_agent("scout").remember("mistake", "Missed API pagination")
        mm.team.push("scout", "API uses cursor pagination, not offset")
        mm.save()         # after each phase
        mm.distill_and_save_global()  # at project end
    """

    def __init__(self, project_slug: str = "", memory_dir: Path | None = None):
        self.project_slug = project_slug
        self.memory_dir = memory_dir or (Path("./projects") / project_slug / "memory")
        self.agents: dict[str, AgentMemory] = {}
        self.team = TeamMemory()
        self.global_memory = GlobalMemory()

    # ── Agent memory access ──

    def get_agent(self, agent_id: str) -> AgentMemory:
        """Get or create an agent's personal memory."""
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentMemory(agent_id=agent_id)
        return self.agents[agent_id]

    # ── Context for a specific agent + phase ──

    def context_for_agent(self, agent_id: str, phase: str = "") -> str:
        """Build the full memory context string for an agent.

        Combines: personal memory + team memory + global lessons.
        Injected into agent prompts.
        """
        parts: list[str] = []

        # Global lessons
        global_ctx = self.global_memory.context_block(agent_id)
        if global_ctx:
            parts.append(global_ctx)

        # Personal memory
        agent_mem = self.get_agent(agent_id)
        personal_ctx = agent_mem.context_block(phase)
        if personal_ctx:
            parts.append(personal_ctx)

        # Team memory
        team_ctx = self.team.context_block(agent_id, phase)
        if team_ctx:
            parts.append(team_ctx)

        if not parts:
            return ""

        return "\n\n".join(parts)

    # ── Persistence ──

    def save(self) -> None:
        """Save all project-level memories to disk (atomic writes)."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Agent memories
        for agent_id, mem in self.agents.items():
            path = self.memory_dir / f"agent_{agent_id}.json"
            atomic_write(path, json.dumps(
                [asdict(e) for e in mem.entries], indent=2, default=str,
            ))

        # Team memory
        team_path = self.memory_dir / "team.json"
        atomic_write(team_path, json.dumps(
            [asdict(e) for e in self.team.entries], indent=2, default=str,
        ))
        logger.debug("Saved project memories: %d agents, %d team entries",
                    len(self.agents), len(self.team.entries))

    def load(self) -> None:
        """Load project-level memories from disk."""
        if not self.memory_dir.exists():
            return

        # Agent memories
        for path in self.memory_dir.glob("agent_*.json"):
            agent_id = path.stem.removeprefix("agent_")
            try:
                data = json.loads(path.read_text())
                entries = [MemoryEntry(**e) for e in data]
                self.agents[agent_id] = AgentMemory(agent_id=agent_id, entries=entries)
            except Exception:
                pass  # graceful degradation

        # Team memory
        team_path = self.memory_dir / "team.json"
        if team_path.exists():
            try:
                data = json.loads(team_path.read_text())
                self.team = TeamMemory(entries=[MemoryEntry(**e) for e in data])
            except Exception:
                pass

    def load_global(self, path: Path | None = None) -> None:
        """Load global (cross-project) memory with file locking."""
        gpath = path or GLOBAL_MEMORY_PATH
        if not gpath.exists():
            return
        try:
            with file_lock(gpath):
                data = json.loads(gpath.read_text())
            self.global_memory = GlobalMemory(
                lessons=[MemoryEntry(**e) for e in data],
            )
            logger.debug("Loaded %d global lessons", len(self.global_memory.lessons))
        except Exception as exc:
            logger.warning("Failed to load global memory: %s", exc)

    def save_global(self, path: Path | None = None) -> None:
        """Save global memory to disk with file locking (atomic write)."""
        gpath = path or GLOBAL_MEMORY_PATH
        gpath.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            [asdict(e) for e in self.global_memory.lessons], indent=2, default=str,
        )
        with file_lock(gpath):
            atomic_write(gpath, content)
        logger.debug("Saved %d global lessons", len(self.global_memory.lessons))

    # ── Distillation — project → global ──

    def distill_to_global(self) -> list[MemoryEntry]:
        """Distill project memories into compact global lessons.

        Called at project completion. Extracts the most valuable
        learnings and adds them to global memory.

        Returns the new global entries created.
        """
        new_lessons: list[MemoryEntry] = []

        # 1. Promote all agent "lesson" and high-value "mistake" entries
        for agent_id, mem in self.agents.items():
            for e in mem.entries:
                if e.kind == "lesson" or (e.kind == "mistake" and len(e.content) > 20):
                    lesson = self.global_memory.add_lesson(
                        content=f"[{agent_id}] {e.content}",
                        agent_id=agent_id,
                        tags=e.tags,
                        source_project=self.project_slug,
                    )
                    new_lessons.append(lesson)

        # 2. Promote team insights
        for e in self.team.entries:
            lesson = self.global_memory.add_lesson(
                content=f"[team/{e.agent_id}] {e.content}",
                agent_id=e.agent_id,
                tags=e.tags,
                source_project=self.project_slug,
            )
            new_lessons.append(lesson)

        # 3. Cap global memory to avoid unbounded growth
        _max_global = int(os.environ.get("HIVE_MAX_GLOBAL_MEMORY", "100"))
        if len(self.global_memory.lessons) > _max_global:
            # Keep the most recent
            self.global_memory.lessons = self.global_memory.lessons[-_max_global:]

        return new_lessons

    # ── Stats ──

    def stats(self) -> dict:
        """Return memory stats for display."""
        agent_counts = {
            aid: len(m.entries) for aid, m in self.agents.items()
        }
        return {
            "agent_memories": agent_counts,
            "team_entries": len(self.team.entries),
            "global_lessons": len(self.global_memory.lessons),
            "total": sum(agent_counts.values()) + len(self.team.entries),
        }
