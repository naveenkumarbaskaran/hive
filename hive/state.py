"""
EPT Shared State — Blackboard pattern for multi-agent collaboration.

All agents read/write to a single Blackboard. Every action produces an Event
that the UI renders. Checkpoints serialize the full board for resume.

Project folder layout:
  projects/
    .global_memory.json           — cross-project distilled lessons
    <project-slug>/
      docs/
        research_context.json     — Scout's analysis
        interviews.json           — all interview Q&A
        prd.md                    — Penny's PRD (user signed-off)
        architecture.md           — Archie's design narrative
        contract.md               — ratified contract
        crew.json                 — crew composition snapshot
        signoffs.json             — feature-level sign-off log
        knowledge_base.json       — ingested external knowledge items
      src/
        <generated source files>
      checkpoints/
        board_<timestamp>.json    — full blackboard snapshot
      memory/
        agent_<id>.json           — per-agent memories
        team.json                 — team shared insights
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from hive.connectors import KnowledgeItem
from hive.connectors import knowledge_for_agent as _knowledge_for_agent
from hive.hardening import (
    CHECKPOINT_SCHEMA_VERSION,
    atomic_write,
    check_disk_space,
    sanitize_filename,
    validate_checkpoint_data,
)

logger = logging.getLogger("hive.state")


# ─────────────────────────────────────────────────────────────────────────────
#  Events — the communication bus
# ─────────────────────────────────────────────────────────────────────────────


class EventType(str, Enum):
    WELCOME = "welcome"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HANDSHAKE = "handshake"  # agent-to-agent negotiation
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    WRITING = "writing"  # agent producing an artifact
    REVIEWING = "reviewing"
    VERDICT = "verdict"
    CHECKPOINT = "checkpoint"
    USER_INPUT = "user_input"
    USER_SIGNOFF = "user_signoff"  # user approved an artifact
    CREW_FORMED = "crew_formed"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    ESCALATION = "escalation"
    LLM_INCIDENT = "llm_incident"  # retry, fallback, model switch
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    agent: str  # agent id (e.g. "penny", "archie")
    content: str  # main text
    target: str = ""  # target agent or file
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
#  Research Context
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ResearchContext:
    domain: str = "unknown"
    product_type: str = "unknown"
    has_frontend: bool = False
    stack: dict = field(default_factory=dict)
    compliance: list[str] = field(default_factory=list)
    scale_tier: str = "startup"
    unknowns: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    raw_summary: str = ""

    def as_block(self) -> str:
        stack_str = ", ".join(f"{k}={v}" for k, v in self.stack.items()) or "not specified"
        compliance_str = ", ".join(self.compliance) or "none"
        unknowns_str = "\n".join(f"  - {u}" for u in self.unknowns) or "  (none)"
        return (
            f"RESEARCH CONTEXT:\n"
            f"  Domain       : {self.domain}\n"
            f"  Product type : {self.product_type}\n"
            f"  Has frontend : {'Yes' if self.has_frontend else 'No'}\n"
            f"  Stack        : {stack_str}\n"
            f"  Compliance   : {compliance_str}\n"
            f"  Scale tier   : {self.scale_tier}\n"
            f"  Open unknowns:\n{unknowns_str}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  User Profile — captured at welcome/intake
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class UserProfile:
    """User identity and context gathered during the welcome/intake phase."""

    name: str = ""  # user's display name
    role: str = ""  # e.g. "Product Owner", "Developer"
    company: str = ""  # optional company/org
    is_request_for_self: bool = True  # is the feature for this user or someone else?
    end_user_name: str = ""  # if for someone else — who?
    end_user_role: str = ""  # e.g. "Customer Service Agent"
    end_user_description: str = ""  # additional context about the end user
    as_is_process: str = ""  # how the user currently does things (before this feature)
    additional_context: str = ""  # any extra notes the user offered

    def as_block(self) -> str:
        """Render as a context block for prompts."""
        parts = ["USER PROFILE:"]
        if self.name:
            parts.append(f"  Requester     : {self.name}")
        if self.role:
            parts.append(f"  Role          : {self.role}")
        if self.company:
            parts.append(f"  Company       : {self.company}")
        if self.is_request_for_self:
            parts.append("  End user      : (requester themselves)")
        else:
            eu = self.end_user_name or "(unnamed)"
            parts.append(f"  End user      : {eu}")
            if self.end_user_role:
                parts.append(f"  End user role : {self.end_user_role}")
            if self.end_user_description:
                parts.append(f"  End user desc : {self.end_user_description}")
        if self.as_is_process:
            parts.append(f"  As-is process : {self.as_is_process}")
        if self.additional_context:
            parts.append(f"  Extra context : {self.additional_context}")
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Artifacts and tracking
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Logbook — persistent record of every agent interaction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LogEntry:
    """One agent↔LLM interaction, recorded in the project Logbook."""

    agent_id: str  # "scout", "penny", "dev_1"
    agent_name: str  # "Scout", "Penny"
    phase: str  # current_phase at call time
    task_summary: str  # first 120 chars of the task
    model_requested: str  # tier or explicit model name
    model_used: str  # actual model that responded
    tier_requested: str  # "fast" | "balanced" | "powerful"
    tier_used: str  # may differ if escalated
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    duration_s: float = 0.0  # wall-clock seconds
    retries: int = 0  # how many retries before success
    tier_escalated: bool = False  # was the tier bumped up?
    thinking_stripped: bool = False  # was thinking param removed?
    errors: list[str] = field(default_factory=list)  # error msgs from failed attempts
    response_preview: str = ""  # first 200 chars of response
    timestamp: float = field(default_factory=time.time)
    success: bool = True


# ─────────────────────────────────────────────────────────────────────────────
#  Artifacts and tracking
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Issue:
    severity: str
    code: str
    description: str
    from_agent: str = ""


@dataclass
class FileEntry:
    name: str
    code: str = ""
    approved: bool = False
    revision: int = 0
    skip_reason: str = ""
    is_frontend: bool = False
    deferred_issues: list[Issue] = field(default_factory=list)
    assigned_dev: str = ""  # which dev agent built this
    test_output: str = ""  # last pytest output (from test execution feedback loop)


@dataclass
class Amendment:
    requested_by: str
    description: str
    outcome: str = ""


@dataclass
class SignOff:
    """Feature-level user sign-off on an artifact."""

    artifact: str  # "prd" | "architecture" | "contract" | "crew"
    version: int  # increments on changes
    approved: bool
    feedback: str = ""  # user's comments
    produced_by: str = ""  # "Scout 🔍 (Research Analyst)" — who created it
    reviewed_by: list[str] = field(default_factory=list)  # agents who reviewed/contributed
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
#  Blackboard — the shared state
# ─────────────────────────────────────────────────────────────────────────────

PROJECTS_DIR = Path(os.environ.get("HIVE_PROJECTS_DIR", "./projects"))

# Maximum events kept in memory (older events are trimmed)
_MAX_EVENTS = int(os.environ.get("HIVE_MAX_EVENTS", "1000"))


@dataclass
class Blackboard:
    """Central shared state for all agents. The single source of truth."""

    # ── Identity ──
    feature: str = ""
    crew_name: str = "EPT — Empowered Product Team"
    project_slug: str = ""

    # ── Research ──
    research: ResearchContext = field(default_factory=ResearchContext)

    # ── Reference repo analysis ──
    repo_analysis: str = ""  # Scout's deep study of a reference repo
    repo_urls: list[str] = field(default_factory=list)  # git URLs the user provided

    # ── Interviews ──
    interviews: dict[str, dict[str, str]] = field(default_factory=dict)

    # ── Documents ──
    prd: str = ""
    architecture: str = ""
    contract: str = ""

    # ── Build plan ──
    file_plan: list[str] = field(default_factory=list)
    dep_graph: dict[str, list[str]] = field(default_factory=dict)

    # ── File registry ──
    registry: dict[str, FileEntry] = field(default_factory=dict)
    amendments: list[Amendment] = field(default_factory=list)
    all_deferred: list[tuple[str, Issue]] = field(default_factory=list)

    # ── User interaction ──
    user_profile: UserProfile | None = None
    user_interjections: list[str] = field(default_factory=list)
    signoffs: list[SignOff] = field(default_factory=list)

    # ── Verdicts ──
    integration_verdict: str = ""
    integration_notes: str = ""  # Quinn's detailed findings
    release_verdict: str = ""
    pii_report: str = ""  # PII/security scan results

    # ── Test documentation ──
    uat_doc: str = ""
    sit_doc: str = ""
    handover_doc: str = ""

    # ── Active crew ──
    active_agents: list[str] = field(default_factory=list)
    dev_count: int = 1

    # ── Event log ──
    events: list[Event] = field(default_factory=list)

    # ── Knowledge base — ingested external context ──
    knowledge_base: list[KnowledgeItem] = field(default_factory=list)

    # ── Plugin guidelines — optional, injected by plugin system ──
    plugin_guidelines: str = ""

    # ── Logbook — persistent LLM interaction log ──
    logbook: list[LogEntry] = field(default_factory=list)

    # ── Memory context (injected by MemoryManager, not serialized) ──
    memory_context: str = ""  # current agent's memory block (set before each think())

    # ── Phase tracking ──
    current_phase: str = ""
    completed_phases: list[str] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────────────────────
    #  Project folder management
    # ─────────────────────────────────────────────────────────────────────────

    def init_project(self) -> Path:
        """Create the project folder structure. Returns project root."""
        if not self.project_slug:
            self.project_slug = re.sub(r"[^\w]+", "_", self.feature)[:40].strip("_").lower()
        root = PROJECTS_DIR / self.project_slug
        (root / "docs").mkdir(parents=True, exist_ok=True)
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (root / "memory").mkdir(parents=True, exist_ok=True)
        return root

    @property
    def project_root(self) -> Path:
        return PROJECTS_DIR / self.project_slug

    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"

    @property
    def src_dir(self) -> Path:
        return self.project_root / "src"

    @property
    def checkpoints_dir(self) -> Path:
        return self.project_root / "checkpoints"

    @property
    def memory_dir(self) -> Path:
        return self.project_root / "memory"

    # ─────────────────────────────────────────────────────────────────────────
    #  Artifact persistence — save docs as user-readable files
    # ─────────────────────────────────────────────────────────────────────────

    def save_research(self) -> Path:
        path = self.docs_dir / "research_context.json"
        data = asdict(self.research)
        atomic_write(path, json.dumps(data, indent=2))
        return path

    def save_interviews(self) -> Path:
        path = self.docs_dir / "interviews.json"
        atomic_write(path, json.dumps(self.interviews, indent=2))
        return path

    def save_prd(self) -> Path:
        path = self.docs_dir / "prd.md"
        atomic_write(path, f"# PRD — {self.feature}\n\n{self.prd}")
        return path

    def save_architecture(self) -> Path:
        path = self.docs_dir / "architecture.md"
        atomic_write(path, f"# Architecture — {self.feature}\n\n{self.architecture}")
        return path

    def save_contract(self) -> Path:
        path = self.docs_dir / "contract.md"
        atomic_write(path, f"# Contract — {self.feature}\n\n```contract\n{self.contract}\n```")
        return path

    def save_crew(self, agents: dict) -> Path:
        path = self.docs_dir / "crew.json"
        crew_data = {
            "crew_name": self.crew_name,
            "feature": self.feature,
            "agents": {
                aid: {
                    "name": a.name,
                    "role": a.role,
                    "active": a.active,
                    "emoji": a.emoji,
                    "tagline": a.tagline,
                }
                for aid, a in agents.items()
            },
        }
        atomic_write(path, json.dumps(crew_data, indent=2))
        return path

    def save_source_file(self, entry: FileEntry) -> Path:
        safe_name = sanitize_filename(entry.name)
        if safe_name != entry.name:
            logger.warning("Sanitized filename: %r → %r", entry.name, safe_name)
        path = self.src_dir / safe_name
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, entry.code)
        return path

    def save_signoffs(self) -> Path:
        path = self.docs_dir / "signoffs.json"
        data = [asdict(s) for s in self.signoffs]
        atomic_write(path, json.dumps(data, indent=2))
        return path

    def save_user_profile(self) -> Path:
        """Save user profile to docs folder."""
        path = self.docs_dir / "user_profile.json"
        data = asdict(self.user_profile) if self.user_profile else {}
        atomic_write(path, json.dumps(data, indent=2))
        return path

    def save_knowledge_base(self) -> Path:
        """Save the knowledge base (all ingested items) to docs folder."""
        path = self.docs_dir / "knowledge_base.json"
        data = [asdict(item) for item in self.knowledge_base]
        atomic_write(path, json.dumps(data, indent=2))
        return path

    def save_repo_analysis(self) -> Path:
        """Save Scout's reference repo analysis to docs folder."""
        path = self.docs_dir / "repo_analysis.md"
        header = "# Reference Repository Analysis\n\n"
        if self.repo_urls:
            header += "Repos: " + ", ".join(self.repo_urls) + "\n\n"
        atomic_write(path, header + self.repo_analysis)
        return path

    def repo_context(self) -> str:
        """Return repo analysis context for prompts."""
        if not self.repo_analysis:
            return ""
        return (
            "REFERENCE REPOSITORY ANALYSIS (Scout studied an existing repo the user provided):\n"
            + self.repo_analysis
        )

    def knowledge_for_agent(self, agent_role: str, max_chars: int = 12_000) -> str:
        """Return knowledge items relevant to a specific agent role."""
        return _knowledge_for_agent(self.knowledge_base, agent_role, max_chars)

    def save_logbook(self) -> Path:
        """Save the logbook (all agent↔LLM interactions) to docs folder."""
        path = self.docs_dir / "logbook.json"
        data = [asdict(entry) for entry in self.logbook]
        atomic_write(path, json.dumps(data, indent=2))
        return path

    def log_llm_call(self, entry: LogEntry) -> None:
        """Append a logbook entry and emit incident events if needed."""
        self.logbook.append(entry)
        # Emit incidents so the UI can show them
        if entry.retries > 0:
            self.emit(
                EventType.LLM_INCIDENT,
                entry.agent_id,
                f"⚡ {entry.retries} retries"
                + (
                    f" (tier escalated {entry.tier_requested}→{entry.tier_used})"
                    if entry.tier_escalated
                    else ""
                )
                + (" (thinking stripped)" if entry.thinking_stripped else "")
                + f" | model={entry.model_used} | {entry.duration_s:.1f}s",
            )
        if not entry.success:
            self.emit(
                EventType.LLM_INCIDENT,
                entry.agent_id,
                f"💥 LLM call FAILED after {entry.retries} retries: "
                + (entry.errors[-1][:120] if entry.errors else "unknown"),
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  Sign-off management
    # ─────────────────────────────────────────────────────────────────────────

    def record_signoff(
        self,
        artifact: str,
        approved: bool,
        feedback: str = "",
        produced_by: str = "",
        reviewed_by: list[str] | None = None,
    ) -> SignOff:
        """Record a user sign-off on an artifact with attribution."""
        version = sum(1 for s in self.signoffs if s.artifact == artifact) + 1
        so = SignOff(
            artifact=artifact,
            version=version,
            approved=approved,
            feedback=feedback,
            produced_by=produced_by,
            reviewed_by=reviewed_by or [],
        )
        self.signoffs.append(so)

        # Build attribution string for the event
        attr_parts = []
        if produced_by:
            attr_parts.append(f"Produced by: {produced_by}")
        if reviewed_by:
            attr_parts.append(f"Reviewed by: {', '.join(reviewed_by)}")
        attr_str = f" | {' | '.join(attr_parts)}" if attr_parts else ""

        self.emit(
            EventType.USER_SIGNOFF,
            "user",
            f"{'✅ Approved' if approved else '❌ Rejected'} {artifact} v{version}"
            + (f": {feedback}" if feedback else "")
            + attr_str,
        )
        self.save_signoffs()
        return so

    def is_signed_off(self, artifact: str) -> bool:
        """Check if the latest sign-off for an artifact is approved."""
        matching = [s for s in self.signoffs if s.artifact == artifact]
        return matching[-1].approved if matching else False

    # ─────────────────────────────────────────────────────────────────────────
    #  Event helpers
    # ─────────────────────────────────────────────────────────────────────────

    def emit(
        self, event_type: EventType, agent: str, content: str, target: str = "", **metadata
    ) -> Event:
        ev = Event(type=event_type, agent=agent, content=content, target=target, metadata=metadata)
        self.events.append(ev)
        # Cap event log to prevent unbounded memory growth
        if len(self.events) > _MAX_EVENTS:
            self.events = self.events[-_MAX_EVENTS:]
        return ev

    # ─────────────────────────────────────────────────────────────────────────
    #  Context builders
    # ─────────────────────────────────────────────────────────────────────────

    def interview_context(self) -> str:
        parts = []
        for agent_id, answers in self.interviews.items():
            if answers:
                lines = "\n".join(f"  Q: {q}\n  A: {a}" for q, a in answers.items())
                parts.append(f"### {agent_id} interview:\n{lines}")
        return "\n\n".join(parts) if parts else "(no answers)"

    def approved_summary(self) -> str:
        approved = [e for e in self.registry.values() if e.approved]
        if not approved:
            return "(no files approved yet)"
        return "\n".join(
            f"  ✅ {e.name}"
            + (f" [{len(e.deferred_issues)} deferred]" if e.deferred_issues else "")
            + (f" (by {e.assigned_dev})" if e.assigned_dev else "")
            for e in approved
        )

    def approved_interfaces(self, max_lines: int = 40) -> str:
        approved = [e for e in self.registry.values() if e.approved]
        if not approved:
            return "(none)"
        parts = []
        for e in approved:
            lines = e.code.splitlines()
            preview = "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                preview += f"\n# ... ({len(lines)} lines total)"
            parts.append(f"### {e.name}\n```\n{preview}\n```")
        return "\n\n".join(parts)

    def approved_signatures(self) -> str:
        """Compact interface view — only classes, functions, and exports.

        Extracts `def`, `class`, `@`, top-level assignments, and docstrings
        from each approved file.  Typically 70-80% smaller than full code,
        saving significant tokens on review/reflection prompts.
        """
        approved = [e for e in self.registry.values() if e.approved]
        if not approved:
            return "(none)"
        parts: list[str] = []
        for e in approved:
            sig_lines: list[str] = []
            in_docstring = False
            for line in e.code.splitlines():
                stripped = line.strip()
                # Track docstrings
                if '"""' in stripped or "'''" in stripped:
                    if in_docstring:
                        in_docstring = False
                        continue
                    in_docstring = stripped.count('"""') == 1 or stripped.count("'''") == 1
                if in_docstring:
                    continue
                # Keep structural lines
                if (
                    stripped.startswith(("def ", "class ", "async def "))
                    or stripped.startswith("@")
                    or (
                        not line.startswith((" ", "\t"))
                        and "=" in stripped
                        and not stripped.startswith("#")
                    )
                    or stripped.startswith(("import ", "from "))
                ):
                    sig_lines.append(line)
            code_summary = "\n".join(sig_lines) if sig_lines else "(empty)"
            parts.append(f"### {e.name}\n```\n{code_summary}\n```")
        return "\n\n".join(parts)

    def approved_full(self) -> str:
        approved = [e for e in self.registry.values() if e.approved]
        if not approved:
            return "(none)"
        return "\n\n".join(f"### {e.name}\n```\n{e.code}\n```" for e in approved)

    def interjections_context(self) -> str:
        if not self.user_interjections:
            return ""
        msgs = "\n".join(f"  - {m}" for m in self.user_interjections)
        return f"\n\nUSER INTERJECTIONS (must be incorporated):\n{msgs}"

    def user_context(self) -> str:
        """Return user profile context for prompts."""
        if self.user_profile:
            return self.user_profile.as_block()
        return ""

    def full_context_header(self, max_tokens: int = 100_000) -> str:
        """Build the full context string for agent prompts.

        Args:
            max_tokens: token budget for context. Callers can pass a
                model-aware limit from telemetry.model_context_window().
                Defaults to 100K (safe for most models).
        """
        from hive.hardening import budget_context

        sections = []
        if self.user_profile:
            sections.append(("user_profile", self.user_profile.as_block(), 1))
        sections.append(("research", self.research.as_block(), 2))
        if self.repo_analysis:
            sections.append(("repo", self.repo_context(), 5))
        sections.append(("prd", f"PRD:\n{self.prd}", 3))
        sections.append(("architecture", f"Architecture:\n{self.architecture}", 4))
        sections.append(("contract", f"CONTRACT:\n```contract\n{self.contract}\n```", 2))
        if self.amendments:
            amend_text = "\n".join(
                f"- [{a.requested_by}] {a.description[:300]}" for a in self.amendments
            )
            sections.append(
                ("amendments", f"CONTRACT AMENDMENTS (applied during build):\n{amend_text}", 1)
            )
        ij = self.interjections_context()
        if ij:
            sections.append(("interjections", ij, 1))
        if self.plugin_guidelines:
            sections.append(
                ("plugin_guidelines", f"Plugin Guidelines:\n{self.plugin_guidelines}", 2)
            )
        # Reserve 30% of window for the actual task prompt + output
        context_budget = int(max_tokens * 0.7)
        return budget_context(sections, max_tokens=context_budget)

    def dep_layers(self) -> list[list[str]]:
        """Topological sort of dep_graph into parallel layers."""
        if not self.dep_graph:
            return [self.file_plan] if self.file_plan else []

        remaining = set(self.file_plan)
        resolved: set[str] = set()
        layers: list[list[str]] = []

        for _ in range(len(self.file_plan) + 1):
            if not remaining:
                break
            layer = []
            for f in list(remaining):
                deps = set(self.dep_graph.get(f, []))
                if deps <= resolved:
                    layer.append(f)
            if not layer:
                layers.append(sorted(remaining))
                break
            layers.append(sorted(layer))
            resolved.update(layer)
            remaining -= set(layer)

        return layers


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint persistence
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint(board: Blackboard) -> Path:
    """Save full blackboard to the project's checkpoint folder (atomic writes)."""
    cdir = board.checkpoints_dir
    cdir.mkdir(parents=True, exist_ok=True)
    check_disk_space(cdir)  # fail-fast if disk is nearly full
    ts = int(time.time())
    path = cdir / f"board_{ts}.json"

    d = asdict(board)
    events_raw = d.pop("events", None)  # saved separately as events.json sidecar
    d.pop("logbook", None)  # logbook is saved separately in docs/
    # knowledge_base content can be large — save separately
    d.pop("knowledge_base", None)
    d.pop("memory_context", None)  # transient — managed by MemoryManager
    d["_schema_version"] = CHECKPOINT_SCHEMA_VERSION

    content = json.dumps(d, indent=2, default=str)
    atomic_write(path, content)

    # Save events sidecar so the dashboard can replay them on resume
    if events_raw:
        events_path = cdir / "events.json"
        atomic_write(events_path, json.dumps(events_raw, indent=2, default=str))

    # Also save a "latest" copy for quick resume (atomic, no read-back)
    latest = cdir / "board_latest.json"
    atomic_write(latest, content)
    logger.info("Checkpoint saved: %s", path.name)

    return path


def load_checkpoint(path: str) -> Blackboard:
    """Load a blackboard from a checkpoint file with schema validation."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))

    # Validate and add defaults for missing fields
    d.pop("_schema_version", None)
    d = validate_checkpoint_data(d)

    research = ResearchContext(**d.pop("research", {}))

    # Load user profile if present
    user_profile_data = d.pop("user_profile", None)
    user_profile = UserProfile(**user_profile_data) if user_profile_data else None

    registry: dict[str, FileEntry] = {}
    for name, entry_d in d.pop("registry", {}).items():
        deferred = [Issue(**i) for i in entry_d.pop("deferred_issues", [])]
        registry[name] = FileEntry(**entry_d, deferred_issues=deferred)

    amendments = [Amendment(**a) for a in d.pop("amendments", [])]
    all_deferred = [(item[0], Issue(**item[1])) for item in d.pop("all_deferred", [])]
    signoffs_raw = d.pop("signoffs", [])
    signoffs = []
    for s in signoffs_raw:
        # Handle older checkpoints without produced_by / reviewed_by
        s.setdefault("produced_by", "")
        s.setdefault("reviewed_by", [])
        signoffs.append(SignOff(**s))

    d.pop("events", None)
    d.pop("logbook", None)  # logbook is not in checkpoints
    d.pop("knowledge_base", None)  # loaded from docs/knowledge_base.json
    d.pop("memory_context", None)  # transient — managed by MemoryManager

    board = Blackboard(
        **d,
        research=research,
        user_profile=user_profile,
        registry=registry,
        amendments=amendments,
        all_deferred=all_deferred,
        signoffs=signoffs,
    )

    # Rehydrate knowledge base from docs folder if available
    kb_path = board.docs_dir / "knowledge_base.json"
    if kb_path.exists():
        try:
            kb_data = json.loads(kb_path.read_text())
            board.knowledge_base = [KnowledgeItem(**item) for item in kb_data]
        except Exception:
            pass  # graceful degradation — knowledge can be re-ingested

    # Rehydrate events from sidecar (saved alongside checkpoint)
    events_path = p.parent / "events.json"
    if events_path.exists():
        try:
            raw_events = json.loads(events_path.read_text(encoding="utf-8"))
            for ev in raw_events:
                try:
                    ev["type"] = EventType(ev["type"])
                    board.events.append(Event(**ev))
                except Exception:
                    pass
        except Exception:
            pass  # graceful degradation — events can be lost without breaking resume

    return board


def list_projects() -> list[dict]:
    """List all existing projects with metadata."""
    projects = []
    if not PROJECTS_DIR.exists():
        return projects
    for pdir in sorted(PROJECTS_DIR.iterdir()):
        if pdir.is_dir():
            latest = pdir / "checkpoints" / "board_latest.json"
            info = {"slug": pdir.name, "path": str(pdir)}
            if latest.exists():
                try:
                    d = json.loads(latest.read_text())
                    info["feature"] = d.get("feature", "")
                    info["current_phase"] = d.get("current_phase", "")
                    info["release_verdict"] = d.get("release_verdict", "")
                except Exception:
                    pass
            projects.append(info)
    return projects
