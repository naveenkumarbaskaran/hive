"""
EPT Agents — Named AI team members with personalities.

Each agent is a lightweight object: a name, a role, a personality,
and a tier that determines which LLM model it uses. Agents don't own
conversation history — the Blackboard is the single source of truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from hive.llm_client import LLMClient, LLMResponse, ModelTier, llm as _default_llm
from hive.state import Blackboard, EventType, LogEntry


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    """A named team member in the EPT crew."""

    id: str                          # "scout", "penny", "dev_1"
    name: str                        # "Scout", "Penny"
    role: str                        # "Research Analyst", "Product Manager"
    emoji: str                       # "🔍"
    tagline: str                     # witty one-liner
    tier: ModelTier = ModelTier.BALANCED
    thinking: bool = False           # extended thinking (Anthropic)
    conditional: str = "always"      # "always" | "frontend" | "dev"
    active: bool = True

    @property
    def label(self) -> str:
        return f"{self.emoji} {self.name}"

    @property
    def card(self) -> str:
        status = "active" if self.active else "benched"
        return f"{self.emoji} {self.name:10} {self.role:20} \"{self.tagline}\""

    def think(
        self,
        board: Blackboard,
        task: str,
        system: str,
        client: LLMClient | None = None,
        max_tokens: int = 8192,
        retries: int = 5,
    ) -> str:
        """Ask this agent to reason about a task. Logs to the Logbook. Returns response text."""
        client = client or _default_llm
        thinking_cfg = {"type": "adaptive"} if self.thinking else None

        # Inject memory context if available
        memory_block = board.memory_context
        if memory_block:
            task = f"{memory_block}\n\n{task}"

        board.emit(EventType.THINKING, self.id, f"Working on: {task[:80]}...")

        try:
            resp = client.chat(
                system=system,
                messages=[{"role": "user", "content": task}],
                tier=self.tier,
                max_tokens=max_tokens,
                thinking=thinking_cfg,
                retries=retries,
            )
            # Record in logbook
            board.log_llm_call(LogEntry(
                agent_id=self.id, agent_name=self.name,
                phase=board.current_phase,
                task_summary=task[:120],
                model_requested=resp.model_requested or client.resolve_model(self.tier),
                model_used=resp.model,
                tier_requested=resp.tier_requested or self.tier.value,
                tier_used=resp.tier_used or self.tier.value,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cache_read_tokens=resp.cache_read_tokens,
                duration_s=resp.duration_s,
                retries=resp.retries,
                tier_escalated=resp.tier_escalated,
                thinking_stripped=resp.thinking_stripped,
                errors=resp.errors,
                response_preview=resp.text[:200],
                success=True,
            ))
        except Exception as exc:
            # Log the failure
            board.log_llm_call(LogEntry(
                agent_id=self.id, agent_name=self.name,
                phase=board.current_phase,
                task_summary=task[:120],
                model_requested=client.resolve_model(self.tier),
                model_used="(failed)",
                tier_requested=self.tier.value,
                tier_used=self.tier.value,
                retries=retries,
                errors=[f"{exc.__class__.__name__}: {str(exc)[:200]}"],
                success=False,
            ))
            raise

        board.emit(EventType.SPEAKING, self.id, resp.text[:200] + "..." if len(resp.text) > 200 else resp.text)

        return resp.text

    def think_with_prefix(
        self,
        board: Blackboard,
        prefix: list[dict],
        task: str,
        system: str,
        client: LLMClient | None = None,
        max_tokens: int = 8192,
    ) -> str:
        """Think with cached prefix messages (for reviews). Logs to the Logbook."""
        client = client or _default_llm
        thinking_cfg = {"type": "adaptive"} if self.thinking else None

        # Inject memory context if available
        memory_block = board.memory_context
        if memory_block:
            task = f"{memory_block}\n\n{task}"

        messages = prefix + [{"role": "user", "content": task}]

        board.emit(EventType.THINKING, self.id, f"Reviewing: {task[:80]}...")

        try:
            resp = client.chat(
                system=system,
                messages=messages,
                tier=self.tier,
                max_tokens=max_tokens,
                thinking=thinking_cfg,
                cache_control_msgs=True,
            )
            board.log_llm_call(LogEntry(
                agent_id=self.id, agent_name=self.name,
                phase=board.current_phase,
                task_summary=f"[prefix+review] {task[:100]}",
                model_requested=resp.model_requested or client.resolve_model(self.tier),
                model_used=resp.model,
                tier_requested=resp.tier_requested or self.tier.value,
                tier_used=resp.tier_used or self.tier.value,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cache_read_tokens=resp.cache_read_tokens,
                duration_s=resp.duration_s,
                retries=resp.retries,
                tier_escalated=resp.tier_escalated,
                thinking_stripped=resp.thinking_stripped,
                errors=resp.errors,
                response_preview=resp.text[:200],
                success=True,
            ))
        except Exception as exc:
            board.log_llm_call(LogEntry(
                agent_id=self.id, agent_name=self.name,
                phase=board.current_phase,
                task_summary=f"[prefix+review] {task[:100]}",
                model_requested=client.resolve_model(self.tier),
                model_used="(failed)",
                tier_requested=self.tier.value,
                tier_used=self.tier.value,
                errors=[f"{exc.__class__.__name__}: {str(exc)[:200]}"],
                success=False,
            ))
            raise

        board.emit(EventType.SPEAKING, self.id, resp.text[:200] + "..." if len(resp.text) > 200 else resp.text)

        return resp.text

    def say(self, board: Blackboard, message: str, to: Agent | None = None) -> None:
        """Agent speaks — emits event for UI."""
        target = to.id if to else ""
        board.emit(EventType.SPEAKING, self.id, message, target=target)


# ─────────────────────────────────────────────────────────────────────────────
#  Dev tagline pool
# ─────────────────────────────────────────────────────────────────────────────

DEV_POOL = [
    ("Dexter",  "Hold my coffee and watch this deploy."),
    ("Devi",    "It works on my machine. Ship the machine."),
    ("Dale",    "First, let me refactor..."),
    ("Dana",    "Comments are love letters to your future self."),
    ("Dylan",   "sudo make it work"),
    ("Drew",    "One does not simply write bug-free code."),
]


def make_dev_agent(index: int) -> Agent:
    """Create a named Dev sub-agent."""
    name, tagline = DEV_POOL[index % len(DEV_POOL)]
    return Agent(
        id=f"dev_{index + 1}",
        name=name,
        role="Developer",
        emoji="🔨",
        tagline=tagline,
        tier=ModelTier.POWERFUL,
        thinking=True,
        conditional="dev",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Agent Roster — all named agents
# ─────────────────────────────────────────────────────────────────────────────

class AgentRoster:
    """Factory for the full EPT roster. Call .compose() to get active agents."""

    # ── Core team (always active) ──

    SCOUT = Agent(
        id="scout", name="Scout", role="Research Analyst", emoji="🔍",
        tagline="I read between the lines so you don't have to.",
        tier=ModelTier.FAST, thinking=False, conditional="always",
    )

    PENNY = Agent(
        id="penny", name="Penny", role="Product Manager", emoji="📋",
        tagline="Requirements are just wishes with deadlines.",
        tier=ModelTier.BALANCED, thinking=False, conditional="always",
    )

    ARCHIE = Agent(
        id="archie", name="Archie", role="Technical Architect", emoji="🏗️",
        tagline="I design systems that outlive sprints.",
        tier=ModelTier.POWERFUL, thinking=True, conditional="always",
    )

    QUINN = Agent(
        id="quinn", name="Quinn", role="Quality Engineer", emoji="🧪",
        tagline="I break things so users don't have to.",
        tier=ModelTier.FAST, thinking=False, conditional="always",
    )

    JUDGE = Agent(
        id="judge", name="Judge", role="Arbitrator", emoji="⚖️",
        tagline="The verdict is in: one more revision.",
        tier=ModelTier.POWERFUL, thinking=True, conditional="always",
    )

    # ── Frontend specialists (conditional) ──

    PIXEL = Agent(
        id="pixel", name="Pixel", role="UI Designer", emoji="🎨",
        tagline="Every pixel reports to me.",
        tier=ModelTier.BALANCED, thinking=False, conditional="frontend",
    )

    FLOW = Agent(
        id="flow", name="Flow", role="UX Designer", emoji="🧭",
        tagline="I map the journey before you take the first step.",
        tier=ModelTier.BALANCED, thinking=False, conditional="frontend",
    )

    ALEX = Agent(
        id="alex", name="Alex", role="User Advocate", emoji="👤",
        tagline="I'm the voice of the confused, angry, delighted user.",
        tier=ModelTier.FAST, thinking=False, conditional="frontend",
    )

    @classmethod
    def all_agents(cls) -> list[Agent]:
        return [cls.SCOUT, cls.PENNY, cls.ARCHIE, cls.QUINN, cls.JUDGE,
                cls.PIXEL, cls.FLOW, cls.ALEX]

    @classmethod
    def compose(cls, has_frontend: bool, dev_count: int = 1) -> dict[str, Agent]:
        """
        Build the active crew based on project characteristics.
        Returns {agent_id: Agent} with .active set appropriately.
        """
        agents: dict[str, Agent] = {}

        for agent in cls.all_agents():
            a = Agent(**{k: getattr(agent, k) for k in agent.__dataclass_fields__})
            if a.conditional == "frontend":
                a.active = has_frontend
            else:
                a.active = True
            agents[a.id] = a

        # Spawn dev sub-agents
        for i in range(dev_count):
            dev = make_dev_agent(i)
            dev.active = True
            agents[dev.id] = dev

        return agents

    @classmethod
    def get(cls, agents: dict[str, Agent], agent_id: str) -> Agent:
        """Get an agent, raising if not found."""
        if agent_id not in agents:
            raise KeyError(f"Agent '{agent_id}' not in crew. Active: {list(agents.keys())}")
        return agents[agent_id]
