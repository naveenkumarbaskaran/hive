"""
EPT Crew Orchestrator — Runs the full Empowered Product Team pipeline.

Phase Flow:
  1. RESEARCH    — Scout analyzes the feature request
  2. INTERVIEW   — Penny + Flow ask the user clarifying questions
  3. PRD         — Penny writes the PRD → user sign-off
  4. FEASIBILITY — Archie checks feasibility → user sign-off
  5. ARCHITECTURE— Archie designs architecture + contract → user sign-off
  6. RATIFICATION— Penny cross-checks arch vs PRD
  7. CREW        — Compose the active crew based on findings
  8. BUILD       — Dev agents implement files (parallel by dep layers)
  9. INTEGRATION — Quinn reviews the full codebase together
  10. RELEASE     — Penny writes release notes, artifacts saved

User sign-offs happen after PRD, Architecture, and optionally at any point.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from hive.agents import Agent, AgentRoster, make_dev_agent, make_reviewer_agent
from hive.connectors import (
    ConnectorRegistry,
    KnowledgeItem,
    format_size,
    ingest_repo,
    is_git_url,
)
from hive.hardening import (
    atomic_write,
    check_disk_space,
    clean_code_fences,
    get_cleanup_registry,
    validate_code_output,
)
from hive.llm_client import LLMClient
from hive.llm_client import llm as _default_llm
from hive.memory import MemoryManager
from hive.prompts import (
    ALEX_REVIEW_TASK,
    ALEX_SYSTEM,
    ARCHIE_FEASIBILITY_SYSTEM,
    ARCHIE_SYSTEM,
    ARCHIE_TASK,
    DEV_REVISION_TASK,
    DEV_SANDBOX_REVISION_TASK,
    DEV_SELF_REFLECT_TASK,
    DEV_SYSTEM,
    DEV_TASK,
    DM_SYSTEM,
    DM_TASK,
    HANDOVER_SYSTEM,
    HANDOVER_TASK,
    INTEGRATION_SYSTEM,
    INTEGRATION_TASK,
    JUDGE_SYSTEM,
    JUDGE_TASK,
    PACKAGING_SYSTEM,
    PACKAGING_TASK,
    PENNY_INTERVIEW_SYSTEM,
    PENNY_INTERVIEW_TASK,
    PENNY_PRD_SYSTEM,
    PENNY_PRD_TASK,
    PENNY_RATIFY_SYSTEM,
    PIXEL_REVIEW_TASK,
    PIXEL_SYSTEM,
    PROJECT_DNA_SYSTEM,
    PROJECT_DNA_TASK,
    QUINN_REVIEW_TASK,
    QUINN_SYSTEM,
    RELEASE_SYSTEM,
    RELEASE_TASK,
    SCOUT_REPO_ANALYSIS_SYSTEM,
    SCOUT_REPO_ANALYSIS_TASK,
    SCOUT_SYSTEM,
    SCOUT_TASK,
    SIT_SYSTEM,
    SIT_TASK,
    UAT_SYSTEM,
    UAT_TASK,
)
from hive.sandbox import (
    SANDBOX_ENABLED,
    run_code_checks,
    syntax_check_file,
)
from hive.state import (
    Amendment,
    Blackboard,
    EventType,
    FileEntry,
    Issue,
    ResearchContext,
    UserProfile,
    save_checkpoint,
)
from hive.telemetry import (
    BudgetExceeded,
    CostTracker,
)
from hive.ui import TerminalUI

logger = logging.getLogger("hive.crew")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Any:
    """Extract and parse JSON from LLM response (handles markdown fences)."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fence
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { or [
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        s = text.find(start_char)
        if s != -1:
            e = text.rfind(end_char)
            if e > s:
                try:
                    return json.loads(text[s:e + 1])
                except json.JSONDecodeError:
                    pass

    raise ValueError(f"Could not parse JSON from response:\n{text[:300]}")


def _parse_contract(text: str) -> dict[str, dict]:
    """Parse the contract block from Archie's response."""
    m = re.search(r"```contract\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        raise ValueError("No ```contract``` block found in architecture response")

    contract_text = m.group(1)
    files: dict[str, dict] = {}
    current_file: str | None = None

    for line in contract_text.splitlines():
        line = line.strip()
        if not line or line == "FILES:":
            continue

        # File entry: "  filename.py:"
        file_match = re.match(r"^(\S+\.?\w*):$", line)
        if file_match:
            current_file = file_match.group(1)
            files[current_file] = {
                "purpose": "", "deps": [], "exports": [],
                "patterns": [], "is_frontend": False,
            }
            continue

        if current_file and ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            if key in ("deps", "exports", "patterns"):
                # Parse list: [a, b, c] or []
                val_clean = val.strip("[]")
                if val_clean:
                    files[current_file][key] = [
                        v.strip().strip("'\"") for v in val_clean.split(",")
                    ]
                else:
                    files[current_file][key] = []
            elif key == "is_frontend":
                files[current_file]["is_frontend"] = val.lower() in ("true", "yes", "1")
            elif key == "purpose":
                files[current_file]["purpose"] = val

    if not files:
        raise ValueError("Contract block contained no file definitions")

    return files


def _parse_verdict(text: str) -> tuple[str, list[Issue]]:
    """Parse a reviewer's verdict response."""
    verdict = "FAIL"
    if "VERDICT:" in text.upper():
        m = re.search(r"VERDICT:\s*(PASS_WITH_NOTES|PASS|FAIL)", text, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()

    issues: list[Issue] = []
    in_issues = False
    in_deferred = False

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("ISSUES:"):
            in_issues = True
            in_deferred = False
            continue
        if line.upper().startswith("DEFERRED:"):
            in_deferred = True
            in_issues = False
            continue
        if line.upper().startswith("NOTES:"):
            in_issues = False
            in_deferred = False
            continue

        if (in_issues or in_deferred) and line.startswith("- "):
            sev_match = re.match(r"-\s*\[(\w+)\]\s*(.*)", line)
            if sev_match:
                severity = sev_match.group(1).lower()
                desc = sev_match.group(2).strip()
            else:
                severity = "warning" if in_deferred else "blocker"
                desc = line[2:].strip()
            issues.append(Issue(severity=severity, description=desc, code=""))

    return verdict, issues


def _extract_architecture_text(text: str) -> str:
    """Extract the architecture section (before the contract block)."""
    contract_start = text.find("```contract")
    if contract_start > 0:
        return text[:contract_start].strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  EPT Crew
# ─────────────────────────────────────────────────────────────────────────────

class EPTCrew:
    """
    The Empowered Product Team orchestrator.

    Usage:
        crew = EPTCrew(feature="A rate-limited REST API for user registration")
        crew.run()
    """

    MAX_REVISIONS = int(os.environ.get("HIVE_MAX_REVISIONS", "3"))

    def __init__(
        self,
        feature: str,
        client: LLMClient | None = None,
        verbose: bool = False,
        auto_approve: bool = False,
        attach_paths: list[str] | None = None,
        repo_urls: list[str] | None = None,
        plugin_paths: list[str] | None = None,
    ):
        self.feature = feature
        self.client = client or _default_llm
        self.verbose = verbose
        self.auto_approve = auto_approve  # skip user sign-offs (for testing)
        self.attach_paths = attach_paths or []  # external knowledge paths
        self.repo_urls = repo_urls or []        # git repo URLs to clone & study

        # State
        self.board = Blackboard(feature=feature)
        self.agents: dict[str, Agent] = {}
        self._registry_lock = threading.Lock()  # protects concurrent registry + deferred writes

        # Memory
        self.memory = MemoryManager()  # initialized properly after project_slug is known

        # Telemetry — cost tracking + budget enforcement
        self.cost_tracker = CostTracker()

        # UI
        self.ui = TerminalUI(self.board, verbose=verbose)

        # Plugins — totally optional, zero impact when empty
        self.plugin_registry = self._init_plugins(plugin_paths)

    # ── Plugin helpers (zero-impact when no plugins) ─────────────────────────

    @staticmethod
    def _init_plugins(plugin_paths: list[str] | None) -> Any:
        """Initialize the plugin registry. Returns None if no plugins found."""
        try:
            from hive.plugins.registry import PluginRegistry
            registry = PluginRegistry()
            count = registry.discover(explicit_paths=plugin_paths)
            if count:
                logger.info("Plugin system active: %s", registry.summary())
            return registry if registry else None
        except Exception as exc:
            logger.debug("Plugin system not available: %s", exc)
            return None

    def _plugin_context(self) -> Any:
        """Build a PluginContext from current board state."""
        if not self.plugin_registry:
            return None
        from hive.plugins.base import PluginContext
        return PluginContext(
            feature=self.feature,
            stack=self.board.research.languages if self.board.research else [],
            phase=self.board.current_phase,
            project_slug=self.board.project_slug,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Main pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> Blackboard:
        """Execute the full EPT pipeline. Returns the final Blackboard."""
        self.ui.banner()

        # Initialize project folder
        self.board.init_project()
        self.board.emit(EventType.PHASE_START, "system",
                        f"Project initialized: {self.board.project_root}")

        # Initialize memory system
        self.memory = MemoryManager(
            project_slug=self.board.project_slug,
            memory_dir=self.board.memory_dir,
        )
        self.memory.load_global()   # cross-project lessons
        self.memory.load()          # resume project-level memories if any

        # Ordered phase list — resume skips completed ones
        phases = [
            ("welcome",       self._phase_welcome),
            ("ingest",        self._phase_ingest),
            ("research",      self._phase_research),
            ("interview",     self._phase_interview),
            ("prd",           self._phase_prd),
            ("feasibility",   self._phase_feasibility),
            ("architecture",  self._phase_architecture),
            ("ratification",  self._phase_ratification),
            ("crew",          self._phase_crew),
            ("build",         self._phase_build),
            ("integration",   self._phase_integration),
            ("test_docs",     self._phase_test_docs),
            ("release",       self._phase_release),
        ]

        done = set(self.board.completed_phases)

        # ── Resilient phase execution ──
        # Critical phases that must succeed (cannot degrade gracefully)
        critical_phases = {"welcome", "prd", "architecture", "crew", "build"}

        try:
            for phase_idx, (name, fn) in enumerate(phases):
                if name in done:
                    self.board.emit(EventType.PHASE_END, "system",
                                    f"Skipping {name} (already completed)")
                    self.ui.flush_events()
                    continue
                # Phase progress: "Phase 5/13 — 38%"
                self.ui.overall_progress(phase_idx, len(phases), name)
                self.cost_tracker.start_phase(name)
                # Plugin lifecycle: phase start
                pctx = self._plugin_context()
                if self.plugin_registry and pctx:
                    self.plugin_registry.on_phase_start(name, pctx)
                try:
                    fn()
                except KeyboardInterrupt:
                    self.cost_tracker.end_phase()
                    raise  # always honour user interrupts
                except BudgetExceeded as exc:
                    self.cost_tracker.end_phase()
                    self.board.emit(EventType.ERROR, "system",
                                    f"💰 {exc}")
                    self.ui.flush_events()
                    self._save()
                    raise
                except Exception as exc:
                    self.cost_tracker.end_phase()
                    if name in critical_phases:
                        raise  # critical phase — can't recover
                    # Non-critical phase: save checkpoint, log, continue
                    logger.error("Phase %s failed (non-critical): %s", name, exc)
                    self.board.emit(
                        EventType.ERROR, "system",
                        f"Phase '{name}' failed: {exc} — continuing (non-critical)",
                    )
                    self.ui.flush_events()
                    self._save()
                    self.board.completed_phases.append(name)
                else:
                    self.cost_tracker.end_phase()
                    # Plugin lifecycle: phase end
                    if self.plugin_registry and pctx:
                        self.plugin_registry.on_phase_end(name, pctx)
        except KeyboardInterrupt:
            print("\n\n  ⚠️  Pipeline interrupted by user. Saving checkpoint...")
            self._save()
            self.board._memory_stats = self.memory.stats()
            self.board._cost_tracker = self.cost_tracker
            self.ui.final_summary()
            return self.board
        except BudgetExceeded:
            self.board._memory_stats = self.memory.stats()
            self.board._cost_tracker = self.cost_tracker
            self.ui.final_summary()
            return self.board
        except Exception as exc:
            self.board.emit(EventType.ERROR, "system", f"Pipeline error: {exc}")
            self.ui.flush_events()
            self._save()
            raise

        # ── Post-pipeline: Project DNA extraction + memory distill ──
        try:
            if self.board.logbook and any(e.success for e in self.board.logbook):
                self._extract_project_dna()
        except Exception as exc:
            logger.warning("Project DNA extraction failed: %s", exc)

        self.board._memory_stats = self.memory.stats()
        self.board._cost_tracker = self.cost_tracker
        self.ui.final_summary()
        return self.board

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 0: Welcome / Intake
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_welcome(self) -> None:
        self.board.current_phase = "welcome"
        self.ui.phase_header("Welcome & Intake")
        self.ui.flush_events()

        if self.auto_approve:
            # Auto mode: create a default profile
            self.board.user_profile = UserProfile(
                name="AutoUser",
                role="Tester",
                is_request_for_self=True,
            )
        else:
            self.board.user_profile = self.ui.welcome_intake()

        self.board.save_user_profile()

        greeting = self.board.user_profile.name or "there"
        self.board.emit(EventType.WELCOME, "system",
                        f"Welcome aboard, {greeting}! Let's build something great.")
        self.ui.flush_events()

        self.board.completed_phases.append("welcome")
        self.ui.phase_footer("Welcome & Intake")

    # ─────────────────────────────────────────────────────────────────────────────
    #  Phase 0.5: Knowledge Ingest
    # ─────────────────────────────────────────────────────────────────────────────

    def _phase_ingest(self) -> None:
        """Ingest external knowledge from CLI paths, repos, or interactive input."""
        self.board.current_phase = "ingest"
        self.ui.phase_header("Knowledge Ingest")
        self.ui.flush_events()

        paths: list[str] = list(self.attach_paths)  # CLI-provided paths
        git_urls: list[str] = list(self.repo_urls)   # CLI-provided --repo URLs

        # Interactive: ask user for additional paths / repos (unless auto mode)
        if not self.auto_approve:
            extra = self.ui.knowledge_intake()
            paths.extend(extra)

        # Separate git URLs from regular paths (user might pass a git URL via --attach too)
        regular_paths: list[str] = []
        for p in paths:
            if is_git_url(p):
                git_urls.append(p)
            else:
                regular_paths.append(p)

        # ── Clone & ingest git repos ──
        repo_trees: list[str] = []
        repo_items: list[KnowledgeItem] = []
        for url in git_urls:
            self.board.emit(EventType.THINKING, "system",
                            f"🔄 Cloning repo: {url}")
            self.ui.flush_events()

            try:
                items, repo_path, tree = ingest_repo(url, max_files=200)
                repo_items.extend(items)
                repo_trees.append(tree)
                self.board.repo_urls.append(url)
                self.ui.repo_clone_summary(url, len(items), tree)

                self.board.emit(EventType.AGREEMENT, "system",
                                f"✅ Cloned {url} → {len(items)} files ingested")
                self.ui.flush_events()
            except RuntimeError as e:
                self.board.emit(EventType.ERROR, "system",
                                f"Failed to clone {url}: {e}")
                self.ui.flush_events()

        # Store repo trees for Scout's deep analysis later
        self._repo_trees = repo_trees

        # ── Ingest regular paths ──
        file_items = ConnectorRegistry.ingest_all(regular_paths) if regular_paths else []

        # Combine
        items = repo_items + file_items

        if not items and not git_urls:
            self.board.emit(EventType.PHASE_END, "system",
                            "No external knowledge attached — skipping ingest.")
            self.ui.flush_events()
            self.board.completed_phases.append("ingest")
            self.ui.phase_footer("Knowledge Ingest")
            return

        if items:
            # Show what was ingested
            self.ui.ingest_summary(items)

            # Summarize large items using Scout
            large_items = [i for i in items if i.was_summarized]
            if large_items:
                scout = AgentRoster.SCOUT
                for item in large_items:
                    self.board.emit(EventType.THINKING, "scout",
                                    f"Summarizing large file: {item.label} ({format_size(item.raw_size)})")
                    self.ui.flush_events()

                    summary_task = (
                        f"Summarize the following {item.source_type} file for use by a software team.\n"
                        f"File: {item.label}\n"
                        f"Original size: {format_size(item.raw_size)}\n\n"
                        f"Content (truncated):\n{item.content[:6000]}\n\n"
                        f"Provide a concise, information-dense summary (max 500 words) covering:\n"
                        f"- What this file is / does\n"
                        f"- Key entities, structures, or interfaces\n"
                        f"- Important business rules or constraints\n"
                        f"- Anything a developer would need to know"
                    )
                    item.summary = scout.think(
                        self.board, summary_task, SCOUT_SYSTEM, self.client,
                    )
                    self.ui.flush_events()

            # Store on board (deduplicate by source_path)
            existing_paths = {i.source_path for i in self.board.knowledge_base}
            for item in items:
                if item.source_path not in existing_paths:
                    self.board.knowledge_base.append(item)
                    existing_paths.add(item.source_path)

            self.board.save_knowledge_base()

        total_size = sum(i.raw_size for i in self.board.knowledge_base)
        self.board.emit(
            EventType.AGREEMENT, "system",
            f"✅ {len(self.board.knowledge_base)} knowledge items loaded "
            f"({format_size(total_size)} total)"
            + (f" from {len(git_urls)} repo(s)" if git_urls else ""),
        )
        self.ui.flush_events()

        # ── Plugin injection: knowledge + guidelines ──
        self._ingest_plugin_knowledge()

        self.board.completed_phases.append("ingest")
        self.ui.phase_footer("Knowledge Ingest")

    def _ingest_plugin_knowledge(self) -> None:
        """Inject knowledge and guidelines from plugins (if any loaded)."""
        if not self.plugin_registry:
            return
        pctx = self._plugin_context()
        if not pctx:
            return

        # Knowledge plugins → KnowledgeItems on the board
        plugin_items = self.plugin_registry.gather_knowledge(pctx)
        if plugin_items:
            existing_paths = {i.source_path for i in self.board.knowledge_base}
            added = 0
            for item in plugin_items:
                if item.source_path not in existing_paths:
                    self.board.knowledge_base.append(item)
                    existing_paths.add(item.source_path)
                    added += 1
            if added:
                self.board.emit(
                    EventType.AGREEMENT, "system",
                    f"🔌 {added} knowledge item(s) from plugins",
                )
                self.ui.flush_events()

        # Guidelines plugins → stored on board for injection into prompts
        guidelines = self.plugin_registry.gather_guidelines(pctx)
        if guidelines:
            self.board.plugin_guidelines = guidelines
            self.board.emit(
                EventType.AGREEMENT, "system",
                f"🔌 Guidelines loaded from plugins "
                f"({len(guidelines)} chars)",
            )
            self.ui.flush_events()

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 1: Research
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_research(self) -> None:
        self.board.current_phase = "research"
        self.ui.phase_header("Research")
        self.ui.flush_events()

        scout = AgentRoster.SCOUT

        # ── Step 1: Deep repo analysis (if repos were attached) ──
        repo_trees = getattr(self, "_repo_trees", [])
        if self.board.repo_urls and not self.board.repo_analysis:
            self.board.emit(EventType.THINKING, "scout",
                            "📖 Deep-studying reference repo(s)...")
            self.ui.flush_events()

            # Gather the most important files from the repo as context
            repo_files_text = self._build_repo_files_context()
            repo_tree_text = "\n\n".join(repo_trees) if repo_trees else "(tree not available)"

            analysis_task = SCOUT_REPO_ANALYSIS_TASK.format(
                repo_tree=repo_tree_text,
                repo_files=repo_files_text,
                feature=self.feature,
                user_context=self.board.user_context(),
            )
            self._set_memory("scout")
            self.board.repo_analysis = scout.think(
                self.board, analysis_task,
                SCOUT_REPO_ANALYSIS_SYSTEM, self.client,
                max_tokens=4096,
            )
            self._clear_memory()
            self.board.save_repo_analysis()

            self.board.emit(EventType.AGREEMENT, "scout",
                            "🔍 Reference repo analysis complete — findings will guide the crew")
            self.ui.flush_events()

        # ── Step 2: Normal research (enriched with repo context) ──
        task = SCOUT_TASK.format(
            feature=self.feature,
            user_context=self.board.user_context(),
            knowledge_context=self.board.knowledge_for_agent("scout"),
            repo_context=self.board.repo_context(),
        )
        self._set_memory("scout")
        resp = scout.think(self.board, task, SCOUT_SYSTEM, self.client)
        self._clear_memory()

        try:
            data = _parse_json(resp)
            self.board.research = ResearchContext(**{
                k: v for k, v in data.items()
                if k in ResearchContext.__dataclass_fields__
            })
        except (ValueError, TypeError) as e:
            self.board.emit(EventType.ERROR, "scout",
                            f"Failed to parse research: {e}. Using defaults.")
            self.board.research.raw_summary = resp
            self._record_mistake("scout",
                                 "Research response wasn't valid JSON — used defaults",
                                 context=f"parse error: {e}")

        # Push research findings to team memory
        r = self.board.research
        self._push_team_insight(
            "scout",
            f"Domain={r.domain}, type={r.product_type}, stack={r.stack}, "
            f"frontend={'yes' if r.has_frontend else 'no'}, scale={r.scale_tier}",
        )

        self.board.save_research()
        self.board.emit(EventType.AGREEMENT, "scout",
                        f"Research complete: {self.board.research.domain} / "
                        f"{self.board.research.product_type}")
        self.ui.flush_events()

        self.board.completed_phases.append("research")
        self.ui.phase_footer("Research")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 2: Interview
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_interview(self) -> None:
        self.board.current_phase = "interview"
        self.ui.phase_header("Interview")
        self.ui.flush_events()

        penny = AgentRoster.PENNY

        # Penny generates questions
        task = PENNY_INTERVIEW_TASK.format(
            research_context=self.board.research.as_block(),
            feature=self.feature,
            user_context=self.board.user_context(),
            knowledge_context=self.board.knowledge_for_agent("penny"),
            repo_context=self.board.repo_context(),
        )
        self._set_memory("penny")
        resp = penny.think(self.board, task, PENNY_INTERVIEW_SYSTEM, self.client)
        self._clear_memory()
        self.ui.flush_events()

        try:
            questions = _parse_json(resp)
            if not isinstance(questions, list):
                questions = [str(questions)]
        except ValueError:
            # Fall back: split by newline, strip bullets
            questions = [
                line.strip().lstrip("-•*0123456789.) ")
                for line in resp.splitlines()
                if line.strip() and "?" in line
            ][:5]

        if not questions:
            questions = ["Could you describe any specific requirements or constraints?"]

        # Ask the user
        if self.auto_approve:
            answers = {q: "(auto-approved)" for q in questions}
        else:
            answers = self.ui.answer_questions(questions)

        self.board.interviews["penny"] = answers
        self.board.save_interviews()

        self.board.emit(EventType.AGREEMENT, "penny",
                        f"Interview complete: {len(answers)} answers collected.")
        self.ui.flush_events()

        self.board.completed_phases.append("interview")
        self.ui.phase_footer("Interview")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 3: PRD
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_prd(self) -> None:
        self.board.current_phase = "prd"
        self.ui.phase_header("PRD")
        self.ui.flush_events()

        penny = AgentRoster.PENNY
        prev_prd = ""  # track previous version for diff

        for attempt in range(3):  # max 3 revisions
            task = PENNY_PRD_TASK.format(
                research_context=self.board.research.as_block(),
                interview_context=self.board.interview_context(),
                feature=self.feature,
                user_context=self.board.user_context(),
                knowledge_context=self.board.knowledge_for_agent("penny"),
                repo_context=self.board.repo_context(),
            )

            if attempt > 0 and self.board.signoffs:
                last_so = self.board.signoffs[-1]
                if not last_so.approved:
                    task += f"\n\nUSER FEEDBACK (must address):\n{last_so.feedback}"

            self._set_memory("penny")
            resp = penny.think(self.board, task, PENNY_PRD_SYSTEM, self.client)
            self._clear_memory()
            self.board.prd = resp
            self.board.save_prd()

            self.board.emit(EventType.WRITING, "penny",
                            f"PRD draft v{attempt + 1} written ({len(resp.splitlines())} lines)")
            self.ui.flush_events()

            # Show diff on revision so the user can see what changed
            if attempt > 0 and prev_prd:
                self.ui.revision_diff("PRD", prev_prd, resp, attempt + 1)

            # Sign-off
            if self.auto_approve:
                self.board.record_signoff("prd", True,
                                          produced_by="Penny 📋 (Product Manager)",
                                          reviewed_by=["Scout 🔍 (Research Analyst)"])
                break
            else:
                approved, feedback = self.ui.signoff_prompt(
                    "PRD", resp,
                    produced_by="Penny 📋 (Product Manager)",
                    reviewed_by=["Scout 🔍 (Research Analyst)"],
                )
                self.board.record_signoff("prd", approved, feedback,
                                          produced_by="Penny 📋 (Product Manager)",
                                          reviewed_by=["Scout 🔍 (Research Analyst)"])
                if approved:
                    break
                self.board.emit(EventType.DISAGREEMENT, "user",
                                f"PRD rejected: {feedback}")
                self.ui.flush_events()
                # Record the rejection as a lesson for Penny
                self._record_mistake("penny",
                                     f"PRD was rejected: {feedback[:200]}",
                                     context=f"PRD draft v{attempt + 1}")
            prev_prd = resp

        # Push PRD scope insight to the team
        self._push_team_insight(
            "penny",
            f"PRD covers {len(self.board.prd.splitlines())} lines of requirements",
            for_agents=["archie", "quinn"],
        )

        self.board.completed_phases.append("prd")
        self.ui.phase_footer("PRD")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 4: Feasibility Check
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_feasibility(self) -> None:
        self.board.current_phase = "feasibility"
        self.ui.phase_header("Feasibility Check")
        self.ui.flush_events()

        archie = AgentRoster.ARCHIE

        task = (
            f"{self.board.research.as_block()}\n\n"
            f"PRD:\n{self.board.prd}\n\n"
            f"{self.board.knowledge_for_agent('archie')}\n\n"
            "Evaluate feasibility of every requirement."
        )
        self._set_memory("archie")
        resp = archie.think(self.board, task, ARCHIE_FEASIBILITY_SYSTEM, self.client)
        self._clear_memory()
        self.ui.flush_events()

        try:
            feas = _parse_json(resp)
            feasible = feas.get("feasible", True)
            concerns = feas.get("concerns", [])
            suggestions = feas.get("suggestions", [])
        except ValueError:
            feasible = "infeasible" not in resp.lower()
            concerns = []
            suggestions = []

        if not feasible:
            self.board.emit(EventType.DISAGREEMENT, "archie",
                            f"Feasibility concerns: {len(concerns)} issues found")
            for c in concerns:
                self.board.emit(EventType.ESCALATION, "archie",
                                f"[{c.get('severity', '?')}] {c.get('requirement', '?')}: "
                                f"{c.get('detail', '?')}")
        else:
            self.board.emit(EventType.AGREEMENT, "archie", "All requirements are feasible ✓")

        self.ui.flush_events()

        # Sign-off on feasibility
        if not self.auto_approve:
            summary = f"Feasible: {feasible}\n"
            if concerns:
                summary += "Concerns:\n" + "\n".join(
                    f"  - [{c.get('severity', '?')}] {c.get('requirement', '?')}: {c.get('detail', '?')}"
                    for c in concerns
                )
            if suggestions:
                summary += "\nSuggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)

            approved, feedback = self.ui.signoff_prompt(
                "Feasibility", summary,
                produced_by="Archie 🏗️ (Tech Architect)",
                reviewed_by=["Penny 📋 (Product Manager)"],
            )
            self.board.record_signoff("feasibility", approved, feedback,
                                      produced_by="Archie 🏗️ (Tech Architect)",
                                      reviewed_by=["Penny 📋 (Product Manager)"])

            if not approved:
                self.board.emit(EventType.DISAGREEMENT, "user",
                                f"Feasibility concerns noted: {feedback}")
                self.ui.flush_events()
        else:
            self.board.record_signoff("feasibility", True,
                                      produced_by="Archie 🏗️ (Tech Architect)",
                                      reviewed_by=["Penny 📋 (Product Manager)"])

        self.board.completed_phases.append("feasibility")
        self.ui.phase_footer("Feasibility Check")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 5: Architecture + Contract
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_architecture(self) -> None:
        self.board.current_phase = "architecture"
        self.ui.phase_header("Architecture")
        self.ui.flush_events()

        archie = AgentRoster.ARCHIE
        prev_arch = ""  # track previous version for diff

        for attempt in range(3):
            context = self.board.full_context_header() if attempt == 0 else (
                self.board.full_context_header() +
                f"\n\nPREVIOUS FEEDBACK:\n{self.board.signoffs[-1].feedback}"
            )
            task = ARCHIE_TASK.format(full_context=context)
            self._set_memory("archie")
            resp = archie.think(self.board, task, ARCHIE_SYSTEM, self.client)
            self._clear_memory()
            self.ui.flush_events()

            # Parse architecture + contract
            try:
                arch_text = _extract_architecture_text(resp)
                contract_data = _parse_contract(resp)
            except ValueError as e:
                self.board.emit(EventType.ERROR, "archie", f"Parse error: {e}")
                self.ui.flush_events()
                self._record_mistake("archie",
                                     f"Architecture response failed to parse: {e}",
                                     context=f"attempt {attempt + 1}")
                if attempt < 2:
                    continue
                raise

            self.board.architecture = arch_text
            self.board.contract = resp[resp.find("```contract"):]
            self.board.file_plan = list(contract_data.keys())
            self.board.dep_graph = {
                f: meta.get("deps", []) for f, meta in contract_data.items()
            }

            # Purge old registry entries from previous attempts (avoid orphans)
            if attempt > 0:
                old_files = set(self.board.registry.keys()) - set(contract_data.keys())
                for old_f in old_files:
                    del self.board.registry[old_f]
                    logger.info("Purged orphan registry entry: %s", old_f)

            # Save registry entries from contract
            for fname, meta in contract_data.items():
                self.board.registry[fname] = FileEntry(
                    name=fname,
                    is_frontend=meta.get("is_frontend", False),
                )

            self.board.save_architecture()
            self.board.save_contract()

            self.board.emit(EventType.WRITING, "archie",
                            f"Architecture + contract: {len(contract_data)} files defined")
            self.ui.flush_events()

            # Show diff on revision so the user can see what changed
            if attempt > 0 and prev_arch:
                self.ui.revision_diff("Architecture", prev_arch, resp, attempt + 1)

            # Sign-off
            preview = (
                f"Architecture ({len(arch_text.splitlines())} lines)\n\n"
                f"Contract files: {', '.join(contract_data.keys())}\n\n"
                f"Dependency layers: {len(self.board.dep_layers())}"
            )
            if self.auto_approve:
                self.board.record_signoff("architecture", True,
                                          produced_by="Archie 🏗️ (Tech Architect)",
                                          reviewed_by=["Penny 📋 (Product Manager)",
                                                        "Quinn 🧪 (Quality Engineer)"])
                break
            else:
                approved, feedback = self.ui.signoff_prompt(
                    "Architecture + Contract", preview,
                    produced_by="Archie 🏗️ (Tech Architect)",
                    reviewed_by=["Penny 📋 (Product Manager)"],
                )
                self.board.record_signoff("architecture", approved, feedback,
                                          produced_by="Archie 🏗️ (Tech Architect)",
                                          reviewed_by=["Penny 📋 (Product Manager)"])
                if approved:
                    break
                self.board.emit(EventType.DISAGREEMENT, "user",
                                f"Architecture rejected: {feedback}")
                self.ui.flush_events()
                self._record_mistake("archie",
                                     f"Architecture was rejected: {feedback[:200]}",
                                     context=f"Architecture draft v{attempt + 1}")
            prev_arch = resp

        # Push architecture decisions to team memory
        self._push_team_insight(
            "archie",
            f"Architecture defines {len(self.board.file_plan)} files "
            f"in {len(self.board.dep_layers())} dep layers",
            for_agents=["quinn"],
        )

        self.board.completed_phases.append("architecture")
        self.ui.phase_footer("Architecture")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 6: Ratification
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_ratification(self) -> None:
        self.board.current_phase = "ratification"
        self.ui.phase_header("Ratification")
        self.ui.flush_events()

        penny = AgentRoster.PENNY
        task = (
            f"PRD:\n{self.board.prd}\n\n"
            f"Architecture:\n{self.board.architecture}\n\n"
            f"Contract:\n{self.board.contract}\n\n"
            "Cross-check the architecture against the PRD."
        )
        resp = penny.think(self.board, task, PENNY_RATIFY_SYSTEM, self.client)
        self.ui.flush_events()

        if "NEEDS_CHANGES" in resp.upper():
            self.board.emit(EventType.DISAGREEMENT, "penny",
                            "Architecture has gaps vs PRD. Noting for Archie.")
            self.board.amendments.append(Amendment(
                requested_by="penny",
                description="Ratification found gaps — see review.",
                outcome=resp,
            ))
        else:
            self.board.emit(EventType.AGREEMENT, "penny",
                            "Architecture aligns with PRD ✓")

        self.ui.flush_events()
        self.board.completed_phases.append("ratification")
        self.ui.phase_footer("Ratification")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 7: Crew Composition
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_crew(self) -> None:
        self.board.current_phase = "crew"
        self.ui.phase_header("Crew Assembly")
        self.ui.flush_events()

        has_frontend = self.board.research.has_frontend

        # Determine dev count from dep layers
        layers = self.board.dep_layers()
        max_parallel = max(len(layer) for layer in layers) if layers else 1
        dev_count = min(max_parallel, len(self.board.file_plan), 6)

        self.agents = AgentRoster.compose(
            has_frontend=has_frontend,
            dev_count=dev_count,
        )
        self.board.active_agents = [a.id for a in self.agents.values() if a.active]
        self.board.dev_count = dev_count

        # Save crew snapshot
        self.board.save_crew(self.agents)

        self.board.emit(EventType.CREW_FORMED, "system",
                        f"Crew assembled: {len(self.board.active_agents)} active agents, "
                        f"{dev_count} devs")
        self.ui.flush_events()

        # Show the crew
        self.ui.crew_intro(self.agents)

        self.board.completed_phases.append("crew")
        self.ui.phase_footer("Crew Assembly")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 8: Build — parallel dev agents by dependency layers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_rate_limit_cascade(exc: Exception) -> bool:
        """Check if an exception is a 429 rate-limit cascade from the LLM client."""
        # httpx.HTTPStatusError with 429 status code
        if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
            return exc.response.status_code == 429
        # Stringified 429 from proxies (word boundary to avoid false positives)
        return bool(re.search(r'\b429\b', str(exc)[:200]))

    def _retry_rate_limited_files(
        self, rate_limited_files: list[str], total_files: int, completed: int,
    ) -> None:
        """Retry files that failed due to 429 rate-limit cascades.

        Waits for a cooldown period, then retries each file sequentially
        (gentle on the API). Files that still fail get a clear skip_reason
        so the delivery summary surfaces them prominently.
        """
        cooldown = int(os.environ.get("HIVE_RATE_LIMIT_COOLDOWN", "30"))
        count = len(rate_limited_files)
        self.board.emit(
            EventType.LLM_INCIDENT, "system",
            f"⚠️ {count} file(s) hit rate-limit cascade — "
            f"waiting {cooldown}s before retry: {', '.join(rate_limited_files)}",
        )
        self.ui.flush_events()
        print(f"  ⏳ Rate-limit cooldown: waiting {cooldown}s before retrying "
              f"{count} file(s)...")
        time.sleep(cooldown)

        still_failed: list[str] = []
        for fname in rate_limited_files:
            self.ui.file_status(fname, "retrying", "rate-limit recovery")
            try:
                success = self._build_file(fname)
            except Exception as exc:
                logger.error("Retry failed for %s: %s", fname, exc)
                success = False

            if success:
                self.ui.file_status(fname, "approved", "(recovered after cooldown)")
                self.board.emit(
                    EventType.VERDICT, "system",
                    f"✅ {fname} recovered after rate-limit retry",
                    target=fname,
                )
            else:
                still_failed.append(fname)
                entry = self.board.registry.get(fname)
                if entry:
                    entry.skip_reason = (
                        "Rate-limit cascade: LLM unavailable after retry "
                        "(recoverable — resume with --resume)"
                    )
                self.ui.file_status(fname, "dropped", "rate-limit — LLM unavailable")
                self.board.emit(
                    EventType.ERROR, "system",
                    f"❌ {fname} dropped: rate-limit cascade persisted after retry",
                    target=fname,
                )

        if still_failed:
            print(f"  ⚠️  {len(still_failed)} file(s) could not be recovered: "
                  f"{', '.join(still_failed)}")
            print(f"  💡 Resume later with: hive --resume "
                  f"{self.board.project_root}/checkpoints/board_latest.json")
        self.ui.flush_events()

    def _phase_build(self) -> None:
        self.board.current_phase = "build"
        self.ui.phase_header("Build")
        self.ui.flush_events()

        # Cache parsed contract once for the whole build phase
        if "```contract" in self.board.contract:
            self._contract_cache = _parse_contract(self.board.contract)
        else:
            self._contract_cache = {}

        layers = self.board.dep_layers()
        total_files = len(self.board.file_plan)
        completed = 0
        rate_limited_files: list[str] = []  # files that failed due to 429 cascade

        for layer_idx, layer in enumerate(layers):
            self.board.emit(EventType.PHASE_START, "system",
                            f"Build layer {layer_idx + 1}/{len(layers)}: {', '.join(layer)}")
            self.ui.flush_events()

            # Files in the same dep layer have no inter-dependencies — build in parallel
            # Adaptive: scale workers to CPU count, capped at layer size
            cpu_limit = os.cpu_count() or 4
            max_workers = min(len(layer), cpu_limit)
            if max_workers == 1:
                # Single file in layer — skip thread overhead
                for fname in layer:
                    completed += 1
                    self.ui.progress(completed, total_files, fname)
                    existing = self.board.registry.get(fname)
                    if existing and existing.approved:
                        self.ui.file_status(fname, "approved", "(from checkpoint)")
                        continue
                    try:
                        success = self._build_file(fname)
                    except Exception as exc:
                        logger.error("Error building %s: %s", fname, exc)
                        if self._is_rate_limit_cascade(exc):
                            rate_limited_files.append(fname)
                            self.ui.file_status(fname, "rate-limited",
                                                "429 cascade — queued for retry")
                            self.board.emit(
                                EventType.LLM_INCIDENT, "system",
                                f"⚠️ {fname}: rate-limit cascade, queued for retry")
                            continue
                        success = False
                    if success:
                        self.ui.file_status(fname, "approved")
                    else:
                        self.ui.file_status(fname, "failed", "exceeded max revisions")
            else:
                futures: dict = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for fname in layer:
                        existing = self.board.registry.get(fname)
                        if existing and existing.approved:
                            completed += 1
                            self.ui.progress(completed, total_files, fname)
                            self.ui.file_status(fname, "approved", "(from checkpoint)")
                            continue
                        futures[executor.submit(self._build_file, fname)] = fname

                    for future in as_completed(futures):
                        fname = futures[future]
                        completed += 1
                        self.ui.progress(completed, total_files, fname)
                        try:
                            success = future.result()
                        except Exception as exc:
                            if self._is_rate_limit_cascade(exc):
                                rate_limited_files.append(fname)
                                self.ui.file_status(fname, "rate-limited",
                                                    "429 cascade — queued for retry")
                                self.board.emit(
                                    EventType.LLM_INCIDENT, "system",
                                    f"⚠️ {fname}: rate-limit cascade, queued for retry")
                                continue
                            logger.error("Unhandled error building %s: %s", fname, exc)
                            success = False
                        if success:
                            self.ui.file_status(fname, "approved")
                        else:
                            self.ui.file_status(fname, "failed", "exceeded max revisions")

            self._save()  # checkpoint after each layer

        # ── Retry pass for rate-limited files ──────────────────────────────
        if rate_limited_files:
            self._retry_rate_limited_files(rate_limited_files, total_files, completed)

        self._contract_cache = {}  # free memory
        self.board.completed_phases.append("build")
        self.ui.phase_footer("Build")

    def _dependency_context(self, fname: str, meta: dict) -> str:
        """Build targeted dependency context — full code of declared deps.

        Instead of showing the dev 40-line previews of ALL approved files,
        this gives the FULL code of the specific files this file depends on.
        This dramatically improves import accuracy and interface alignment.
        """
        deps = meta.get("deps", [])
        if not deps:
            return ""

        parts: list[str] = [
            "\nDEPENDENCY FILES (full code of your declared dependencies):"
        ]
        total_chars = 0
        max_chars = 30_000  # budget to avoid context explosion

        for dep_name in deps:
            entry = self.board.registry.get(dep_name)
            if not entry or not entry.code:
                continue
            code = entry.code
            entry_text = f"\n### {dep_name}\n```\n{code}\n```"
            if total_chars + len(entry_text) > max_chars:
                # Truncate this dep to stay within budget
                remaining = max_chars - total_chars
                if remaining > 200:
                    truncated = code[: remaining - 100]
                    entry_text = (
                        f"\n### {dep_name}\n```\n{truncated}\n"
                        f"# ... (truncated, {len(code)} chars total)\n```"
                    )
                    parts.append(entry_text)
                break
            parts.append(entry_text)
            total_chars += len(entry_text)

        if len(parts) == 1:
            return ""  # no deps found in registry
        return "\n".join(parts)

    def _build_file(self, fname: str) -> bool:
        """Build a single file: generate → review → revise loop."""
        entry = self.board.registry.get(fname)
        if not entry:
            entry = FileEntry(name=fname)
            with self._registry_lock:
                self.board.registry[fname] = entry

        # Get contract metadata (use cached parse from _phase_build)
        contract_data = getattr(self, '_contract_cache', None)
        if contract_data is None:
            contract_data = (
                _parse_contract(self.board.contract)
                if "```contract" in self.board.contract else {}
            )
        meta = contract_data.get(fname, {})

        # Select a dev agent (round-robin, lock for consistent key ordering)
        dev_agents = [a for a in self.agents.values() if a.id.startswith("dev_") and a.active]
        if not dev_agents:
            dev_agents = [make_dev_agent(0)]
        with self._registry_lock:
            dev_idx = list(self.board.registry.keys()).index(fname) % len(dev_agents)
        dev = dev_agents[dev_idx]
        entry.assigned_dev = dev.name

        self.ui.file_status(fname, "building", f"→ {dev.name}")

        # Generate
        system = DEV_SYSTEM.format(dev_name=dev.name, dev_tagline=dev.tagline)
        dep_ctx = self._dependency_context(fname, meta)
        task = DEV_TASK.format(
            full_context=self.board.full_context_header(),
            approved_interfaces=self.board.approved_interfaces(),
            dependency_context=dep_ctx,
            filename=fname,
            purpose=meta.get("purpose", "see contract"),
            deps=meta.get("deps", []),
            exports=meta.get("exports", []),
            patterns=meta.get("patterns", []),
            revision_notes="",
        )

        self._set_memory(dev.id)
        code = dev.think(self.board, task, system, self.client)
        self._clear_memory()
        self.ui.flush_events()

        # Clean code (strip markdown fences if present) + validate
        code = self._clean_code(code)
        try:
            code = validate_code_output(code, fname)
        except ValueError as e:
            logger.warning("Code validation failed for %s: %s", fname, e)
            self.board.emit(EventType.ERROR, dev.id,
                            f"Code validation: {e}")
            self._record_mistake(dev.id,
                                 f"Generated invalid code for {fname}: {e}",
                                 context="initial generation")
        entry.code = code
        entry.revision = 1

        # ── Sandbox pre-check: catch syntax/import errors before review ──
        code = self._sandbox_check(fname, entry, dev, system, dep_ctx=dep_ctx)
        entry.code = code

        # ── Self-reflection: dev critiques own code before review ──
        code = self._self_reflect(fname, entry, dev, system, meta)
        entry.code = code

        # Review loop
        for attempt in range(1, self.MAX_REVISIONS + 1):
            self.ui.file_status(fname, "reviewing", f"attempt {attempt}")

            verdict, issues = self._review_file(fname, entry)

            if verdict == "PASS":
                entry.approved = True
                self.board.emit(EventType.VERDICT, "quinn",
                                f"PASS: {fname}", target=fname)
                self.board.save_source_file(entry)
                self.ui.flush_events()
                # Record success pattern
                if entry.revision > 1:
                    self._record_pattern(dev.id,
                                         f"File {fname} passed after {entry.revision} revisions",
                                         context="multiple revisions needed — pay attention to reviews")
                else:
                    self._record_pattern(dev.id,
                                         f"File {fname} passed first try",
                                         context="clean implementation approach worked")
                return True

            elif verdict == "PASS_WITH_NOTES":
                entry.approved = True
                deferred = [i for i in issues if i.severity != "blocker"]
                entry.deferred_issues = deferred
                with self._registry_lock:
                    self.board.all_deferred.extend((fname, i) for i in deferred)
                self.board.emit(EventType.VERDICT, "quinn",
                                f"PASS_WITH_NOTES: {fname} ({len(deferred)} deferred)",
                                target=fname)
                self.board.save_source_file(entry)
                self.ui.flush_events()
                # Record deferred issues as lessons for the dev
                for issue in deferred:
                    self._record_lesson(dev.id,
                                        f"Deferred issue in {fname}: {issue.description[:120]}",
                                        context=f"severity={issue.severity}")
                return True

            else:  # FAIL
                self.board.emit(EventType.VERDICT, "quinn",
                                f"FAIL: {fname} ({len(issues)} issues)",
                                target=fname)
                self.ui.flush_events()

                # Record each failure reason as a mistake for the dev
                for issue in issues:
                    self._record_mistake(dev.id,
                                         f"Review failed {fname}: {issue.description[:120]}",
                                         context=f"severity={issue.severity}, attempt={attempt}")
                # Push blocker issues to team memory so other devs learn
                blockers = [i for i in issues if i.severity == "blocker"]
                if blockers:
                    self._push_team_insight(
                        "quinn",
                        f"Common blockers in {fname}: "
                        + "; ".join(b.description[:80] for b in blockers[:3]),
                        for_agents=[a.id for a in dev_agents],
                    )

                if attempt >= self.MAX_REVISIONS:
                    # Escalate to Judge
                    return self._escalate_to_judge(fname, entry, issues, attempt)

                # Revise
                self.ui.file_status(fname, "revising", f"→ {dev.name}")
                review_text = "\n".join(
                    f"- [{i.severity}] {i.description}" for i in issues
                )
                rev_task = DEV_REVISION_TASK.format(
                    full_context=self.board.full_context_header(),
                    approved_interfaces=self.board.approved_interfaces(),
                    dependency_context=dep_ctx,
                    filename=fname,
                    current_code=entry.code,
                    review_issues=review_text,
                )
                self._set_memory(dev.id)
                code = dev.think(self.board, rev_task, system, self.client)
                self._clear_memory()
                self.ui.flush_events()

                code = self._clean_code(code)
                try:
                    code = validate_code_output(code, fname)
                except ValueError as e:
                    logger.warning("Revision validation failed for %s: %s", fname, e)
                    self.board.emit(EventType.ERROR, dev.id,
                                    f"Revision validation: {e}")
                entry.code = code
                entry.revision += 1

                # Sandbox re-check after revision
                code = self._sandbox_check(fname, entry, dev, system, dep_ctx=dep_ctx)
                entry.code = code

        return False

    def _sandbox_check(
        self, fname: str, entry: FileEntry, dev: Agent, system: str,
        max_sandbox_retries: int = 2,
        dep_ctx: str = "",
    ) -> str:
        """Run the sandbox on generated code; let the dev self-correct on failure.

        If the sandbox catches syntax/import errors, the dev gets immediate
        feedback and a chance to fix without wasting a reviewer LLM call.
        Returns the (possibly revised) code.
        """
        if not SANDBOX_ENABLED or not fname.endswith(".py"):
            return entry.code

        # Build the file set: all approved files + this file
        file_set: dict[str, str] = {}
        for name, fe in self.board.registry.items():
            if fe.approved and fe.code:
                file_set[name] = fe.code
        file_set[fname] = entry.code

        for sandbox_attempt in range(1, max_sandbox_retries + 1):
            self.ui.file_status(fname, "sandbox", f"check #{sandbox_attempt}")
            result = syntax_check_file(fname, entry.code)

            if result.success:
                self.board.emit(EventType.SPEAKING, "system",
                                f"🧪 Sandbox: {fname} — {result.feedback}")
                self.ui.flush_events()
                return entry.code

            # Sandbox found errors — let the dev self-correct
            self.board.emit(EventType.ERROR, "system",
                            f"🧪 Sandbox: {fname} — {result.feedback}")
            self.ui.flush_events()

            if sandbox_attempt >= max_sandbox_retries:
                # Give up on sandbox self-correction, let the reviewer handle it
                self.board.emit(EventType.SPEAKING, "system",
                                f"🧪 Sandbox: {fname} — exceeded retries, proceeding to review")
                self._record_mistake(dev.id,
                                     f"Sandbox failed for {fname}: {result.output[:200]}",
                                     context="sandbox self-correction exhausted")
                return entry.code

            # Ask the dev to fix based on real execution output
            self.ui.file_status(fname, "sandbox-fix", f"→ {dev.name}")
            sandbox_task = DEV_SANDBOX_REVISION_TASK.format(
                full_context=self.board.full_context_header(),
                approved_interfaces=self.board.approved_interfaces(),
                dependency_context=dep_ctx,
                filename=fname,
                current_code=entry.code,
                sandbox_output=result.output,
            )
            self._set_memory(dev.id)
            code = dev.think(self.board, sandbox_task, system, self.client)
            self._clear_memory()
            self.ui.flush_events()

            code = self._clean_code(code)
            try:
                code = validate_code_output(code, fname)
            except ValueError as e:
                logger.warning("Sandbox revision validation failed for %s: %s", fname, e)
            entry.code = code

        return entry.code

    def _self_reflect(
        self, fname: str, entry: FileEntry, dev: Agent, system: str,
        meta: dict,
    ) -> str:
        """Dev agent self-critiques its own code before sending to reviewers.

        This catches contract mismatches, missing exports, and obvious bugs
        that would waste a reviewer LLM call. Uses FAST tier to keep costs low.
        Returns the (possibly improved) code.
        """
        if not fname.endswith(".py"):
            return entry.code

        self.ui.file_status(fname, "reflecting", f"→ {dev.name}")

        reflect_task = DEV_SELF_REFLECT_TASK.format(
            filename=fname,
            purpose=meta.get("purpose", "see contract"),
            deps=meta.get("deps", []),
            exports=meta.get("exports", []),
            patterns=meta.get("patterns", []),
            code=entry.code,
            approved_interfaces=self.board.approved_interfaces(),
        )
        self._set_memory(dev.id)
        try:
            reflected = dev.think(self.board, reflect_task, system, self.client)
        except Exception as exc:
            logger.warning("Self-reflection failed for %s: %s", fname, exc)
            self._clear_memory()
            return entry.code
        self._clear_memory()
        self.ui.flush_events()

        reflected = self._clean_code(reflected)
        try:
            reflected = validate_code_output(reflected, fname)
        except ValueError:
            # Reflection produced invalid output — keep original
            return entry.code

        # Only accept if reflection changed something
        if reflected.strip() != entry.code.strip():
            self.board.emit(EventType.SPEAKING, dev.id,
                            f"🔍 Self-reflection improved {fname}")
            self._record_pattern(dev.id,
                                 f"Self-reflection caught issues in {fname}",
                                 context="pre-review self-correction")
        return reflected

    def _review_file(self, fname: str, entry: FileEntry) -> tuple[str, list[Issue]]:
        """Run all applicable reviewers on a file."""
        all_issues: list[Issue] = []
        worst_verdict = "PASS"

        # On large builds (>8 files), delegate to a sub-reviewer so Quinn isn't the
        # sole bottleneck. Sub-reviewers use FAST tier; Quinn does final integration pass.
        large_build = len(self.board.file_plan) > 8
        if large_build:
            reviewer_idx = list(self.board.file_plan).index(fname) % 4
            reviewer = make_reviewer_agent(reviewer_idx)
            self.board.emit(EventType.SPEAKING, reviewer.id,
                            f"Sub-review: {fname}")
        else:
            reviewer = AgentRoster.QUINN

        # Primary review (Quinn or sub-reviewer)
        quinn = AgentRoster.QUINN
        task = QUINN_REVIEW_TASK.format(
            full_context=self.board.full_context_header(),
            approved_interfaces=self.board.approved_interfaces(),
            filename=fname,
            code=entry.code,
        )
        self._set_memory("quinn")
        resp = reviewer.think(self.board, task, QUINN_SYSTEM, self.client)
        self._clear_memory()
        verdict, issues = _parse_verdict(resp)
        for i in issues:
            i.from_agent = reviewer.id
        all_issues.extend(issues)
        if verdict == "FAIL":
            worst_verdict = "FAIL"
            # On a FAIL from a sub-reviewer, escalate to Quinn for second opinion
            if large_build:
                self.board.emit(EventType.SPEAKING, "quinn",
                                f"Sub-reviewer flagged FAIL on {fname} — re-reviewing")
                self._set_memory("quinn")
                resp2 = quinn.think(self.board, task, QUINN_SYSTEM, self.client)
                self._clear_memory()
                v2, iss2 = _parse_verdict(resp2)
                for i in iss2:
                    i.from_agent = "quinn"
                all_issues.extend(iss2)
                if v2 != "FAIL":
                    worst_verdict = v2  # Quinn overrides sub-reviewer on pass
        elif verdict == "PASS_WITH_NOTES" and worst_verdict != "FAIL":
            worst_verdict = "PASS_WITH_NOTES"

        # Frontend files get Pixel + Alex review
        if entry.is_frontend and self.agents:
            pixel = self.agents.get("pixel")
            if pixel and pixel.active:
                task = PIXEL_REVIEW_TASK.format(
                    full_context=self.board.full_context_header(),
                    filename=fname,
                    code=entry.code,
                )
                self._set_memory("pixel")
                resp = pixel.think(self.board, task, PIXEL_SYSTEM, self.client)
                self._clear_memory()
                v, iss = _parse_verdict(resp)
                for i in iss:
                    i.from_agent = "pixel"
                all_issues.extend(iss)
                if v == "FAIL":
                    worst_verdict = "FAIL"

            alex = self.agents.get("alex")
            if alex and alex.active:
                task = ALEX_REVIEW_TASK.format(
                    full_context=self.board.full_context_header(),
                    filename=fname,
                    code=entry.code,
                )
                self._set_memory("alex")
                resp = alex.think(self.board, task, ALEX_SYSTEM, self.client)
                self._clear_memory()
                v, iss = _parse_verdict(resp)
                for i in iss:
                    i.from_agent = "alex"
                all_issues.extend(iss)
                if v == "FAIL":
                    worst_verdict = "FAIL"

        self.ui.flush_events()
        return worst_verdict, all_issues

    def _escalate_to_judge(
        self, fname: str, entry: FileEntry, issues: list[Issue], attempt: int,
    ) -> bool:
        """Let Judge decide the fate of a file that keeps failing."""
        self.board.emit(EventType.ESCALATION, "system",
                        f"Escalating {fname} to Judge after {attempt} failures")
        self.ui.flush_events()

        judge = AgentRoster.JUDGE
        review_hist = "\n".join(f"- [{i.severity}] {i.description}" for i in issues)

        task = JUDGE_TASK.format(
            full_context=self.board.full_context_header(),
            filename=fname,
            code=entry.code,
            review_history=review_hist,
            attempt=attempt,
        )
        self._set_memory("judge")
        resp = judge.think(self.board, task, JUDGE_SYSTEM, self.client)
        self._clear_memory()
        self.ui.flush_events()

        if "APPROVE" in resp.upper() and "REJECT" not in resp.upper():
            entry.approved = True
            deferred = [Issue(severity="deferred", description=i.description, code="")
                        for i in issues]
            entry.deferred_issues = deferred
            self.board.all_deferred.extend((fname, i) for i in deferred)
            self.board.emit(EventType.VERDICT, "judge",
                            f"APPROVED (with deferred): {fname}")
            self.board.save_source_file(entry)
            self.ui.flush_events()
            # Record the escalation outcome as a lesson
            self._record_lesson("judge",
                                f"Approved {fname} with {len(deferred)} deferred issues after {attempt} dev attempts",
                                context="escalation resolved by accepting with notes")
            return True

        elif "AMEND_CONTRACT" in resp.upper():
            # Contract amendment
            amend_match = re.search(r"AMENDMENT:\s*(.*?)(?:RATIONALE:|$)",
                                    resp, re.DOTALL | re.IGNORECASE)
            amend_text = amend_match.group(1).strip() if amend_match else resp
            self.board.amendments.append(Amendment(
                requested_by="judge",
                description=amend_text,
                outcome="contract_amended",
            ))
            entry.skip_reason = "contract amended by Judge — rebuild needed"
            self.board.emit(EventType.VERDICT, "judge",
                            f"AMEND CONTRACT for {fname}")
            self.ui.flush_events()
            self._record_lesson("judge",
                                f"Contract amendment needed for {fname}: {amend_text[:120]}",
                                context="original contract was insufficient")
            self._push_team_insight("judge",
                                    f"Contract was amended for {fname} — check your deps",
                                    for_agents=["archie"])
            return False

        else:
            entry.skip_reason = "Rejected by Judge after max revisions"
            self.board.emit(EventType.VERDICT, "judge", f"REJECTED: {fname}")
            self.ui.flush_events()
            self._record_lesson("judge",
                                f"Rejected {fname} after {attempt} attempts — file couldn't meet quality bar",
                                context="max revisions exceeded, fundamental issues")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 9: Integration
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_integration(self) -> None:
        self.board.current_phase = "integration"
        self.ui.phase_header("Integration Testing")
        self.ui.flush_events()

        # ── Sandbox: run all approved files together ──────────────────────
        sandbox_section = ""
        if SANDBOX_ENABLED:
            all_files = {
                name: fe.code for name, fe in self.board.registry.items()
                if fe.approved and fe.code
            }
            if all_files:
                self.board.emit(EventType.SPEAKING, "system",
                                f"🧪 Running sandbox on {len(all_files)} approved files...")
                self.ui.flush_events()
                sb_result = run_code_checks(all_files)
                status = "PASS ✓" if sb_result.success else "FAIL ✗"
                self.board.emit(EventType.SPEAKING, "system",
                                f"🧪 Sandbox integration: {status}")
                self.ui.flush_events()
                sandbox_section = (
                    "CODE EXECUTION RESULTS (files were actually run in a sandbox):\n"
                    f"Status: {status}\n{sb_result.output}\n"
                    "Use these real results to inform your review.\n"
                )

        quinn = AgentRoster.QUINN
        task = INTEGRATION_TASK.format(
            full_context=self.board.full_context_header(),
            approved_full=self.board.approved_full(),
            sandbox_section=sandbox_section,
        )
        self._set_memory("quinn")
        resp = quinn.think(self.board, task, INTEGRATION_SYSTEM, self.client,
                           max_tokens=4096)
        self._clear_memory()
        self.ui.flush_events()

        # Store Quinn's detailed findings for transparency
        self.board.integration_notes = resp

        if "PASS" in resp.upper():
            self.board.integration_verdict = "PASS"
            self.board.emit(EventType.VERDICT, "quinn",
                            "Integration: PASS ✓")
        else:
            self.board.integration_verdict = "FAIL"
            self.board.emit(EventType.VERDICT, "quinn",
                            "Integration: FAIL — see notes")

            # ── Integration Gate: FAIL requires explicit override ──
            if self.auto_approve:
                self.board.emit(
                    EventType.ESCALATION, "quinn",
                    "⚠️  Integration FAILED in auto mode — proceeding with warning. "
                    "Review integration notes in delivery summary.",
                )
            else:
                override = self.ui.integration_gate(resp)
                if override:
                    self.board.emit(EventType.AGREEMENT, "user",
                                    "User overrode integration FAIL — proceeding to release")
                    self.board.integration_verdict = "FAIL_OVERRIDDEN"
                else:
                    self.board.emit(EventType.DISAGREEMENT, "user",
                                    "User declined to override — pipeline halted at integration")
                    self._save()
                    self.board.completed_phases.append("integration")
                    self.ui.phase_footer("Integration Testing")
                    raise KeyboardInterrupt("Integration gate: user declined override")

        self.ui.flush_events()
        self.board.completed_phases.append("integration")
        self.ui.phase_footer("Integration Testing")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 10: Test Docs — UAT + SIT
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_test_docs(self) -> None:
        self.board.current_phase = "test_docs"
        self.ui.phase_header("Test Documentation")
        self.ui.flush_events()

        alex = AgentRoster.ALEX
        quinn = AgentRoster.QUINN

        user_info = ""
        if self.board.user_profile and self.board.user_profile.name:
            up = self.board.user_profile
            user_info = f"Requested by: {up.name}"
            if up.role:
                user_info += f" ({up.role})"

        deferred_text = "\n".join(
            f"- [{i.severity}] {fname}: {i.description}"
            for fname, i in self.board.all_deferred
        ) or "(none)"

        prd_lines = self.board.prd.splitlines()
        prd_summary = "\n".join(
            l for l in prd_lines if l.strip().startswith("FR-") or l.strip().startswith("##")
        ) or self.board.prd[:2000]

        # ── UAT — Alex writes from pseudo-user perspective ──
        uat_task = UAT_TASK.format(
            feature=self.feature,
            user_info=user_info,
            full_context=self.board.full_context_header(),
            prd_summary=prd_summary,
            approved_summary=self.board.approved_summary(),
            deferred_issues=deferred_text,
        )
        self.board.emit(EventType.WRITING, "alex", "Writing UAT scenarios...")
        self.ui.flush_events()
        self._set_memory("alex")
        uat_resp = alex.think(self.board, uat_task, UAT_SYSTEM, self.client, max_tokens=4096)
        self._clear_memory()
        self.board.uat_doc = uat_resp
        uat_path = self.board.docs_dir / "UAT.md"
        atomic_write(uat_path, uat_resp)
        self.board.emit(EventType.WRITING, "alex", f"UAT saved: {uat_path}")
        self.ui.flush_events()

        # ── SIT — Quinn writes the integration test plan ──
        sit_task = SIT_TASK.format(
            feature=self.feature,
            full_context=self.board.full_context_header(),
            contract=self.board.contract,
            approved_summary=self.board.approved_summary(),
            deferred_issues=deferred_text,
            integration_verdict=self.board.integration_verdict,
        )
        self.board.emit(EventType.WRITING, "quinn", "Writing SIT plan...")
        self.ui.flush_events()
        self._set_memory("quinn")
        sit_resp = quinn.think(self.board, sit_task, SIT_SYSTEM, self.client, max_tokens=4096)
        self._clear_memory()
        self.board.sit_doc = sit_resp
        sit_path = self.board.docs_dir / "SIT.md"
        atomic_write(sit_path, sit_resp)
        self.board.emit(EventType.WRITING, "quinn", f"SIT saved: {sit_path}")
        self.ui.flush_events()

        self._save()
        self.board.completed_phases.append("test_docs")
        self.ui.phase_footer("Test Documentation")

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 10: Release
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_release(self) -> None:
        self.board.current_phase = "release"
        self.ui.phase_header("Release")
        self.ui.flush_events()

        penny = AgentRoster.PENNY

        deferred_text = "\n".join(
            f"- [{i.severity}] {fname}: {i.description}"
            for fname, i in self.board.all_deferred
        ) or "(none)"

        amend_text = "\n".join(
            f"- [{a.requested_by}] {a.description}" for a in self.board.amendments
        ) or "(none)"

        # Build signoff attribution summary for release notes
        signoff_log = "\n".join(
            f"- {so.artifact} v{so.version}: {'Approved' if so.approved else 'Rejected'}"
            + (f" | Produced by: {so.produced_by}" if so.produced_by else "")
            + (f" | Reviewed by: {', '.join(so.reviewed_by)}" if so.reviewed_by else "")
            for so in self.board.signoffs
        ) or "(none)"

        # User context for release notes
        user_info = ""
        if self.board.user_profile and self.board.user_profile.name:
            up = self.board.user_profile
            user_info = f"Requested by: {up.name}"
            if up.role:
                user_info += f" ({up.role})"
            if not up.is_request_for_self and up.end_user_name:
                user_info += f"\nEnd user: {up.end_user_name}"
                if up.end_user_role:
                    user_info += f" ({up.end_user_role})"

        task = RELEASE_TASK.format(
            feature=self.feature,
            full_context=self.board.full_context_header(),
            approved_summary=self.board.approved_summary(),
            deferred_issues=deferred_text,
            amendments=amend_text,
            signoff_log=signoff_log,
            user_info=user_info,
        )
        self._set_memory("penny")
        resp = penny.think(self.board, task, RELEASE_SYSTEM, self.client)
        self._clear_memory()
        self.board.release_verdict = resp
        self.ui.flush_events()

        # Save release notes (atomic write)
        release_path = self.board.docs_dir / "release_notes.md"
        atomic_write(release_path, resp)
        self.board.emit(EventType.WRITING, "penny",
                        f"Release notes saved: {release_path}")

        # ── Handover document ──
        self._generate_handover(deferred_text, amend_text, signoff_log, user_info)

        # ── Stack-aware packaging artifacts ──
        self._generate_packaging_artifacts()

        # ── Delivery Manager final checklist ──
        self._generate_delivery_checklist(deferred_text, user_info)

        # ── Distill memories to global ──
        new_lessons = self.memory.distill_to_global()
        self.memory.save_global()
        if new_lessons:
            self.board.emit(EventType.AGREEMENT, "system",
                            f"💡 {len(new_lessons)} lessons distilled to global memory")

        # ── Clean up temp dirs (cloned repos) ──
        cleanup = get_cleanup_registry()
        if cleanup.registered:
            logger.info("Cleaning up %d temp directories", len(cleanup.registered))
            cleanup._cleanup()

        # Attach memory stats for UI summary
        self.board._memory_stats = self.memory.stats()

        # Final save
        self._save()

        self.ui.flush_events()
        self.board.completed_phases.append("release")
        self.ui.phase_footer("Release")

    # ─────────────────────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_handover(
        self, deferred_text: str, amend_text: str, signoff_log: str, user_info: str
    ) -> None:
        """Generate Handover.md — comprehensive project delivery document."""
        penny = AgentRoster.PENNY
        task = HANDOVER_TASK.format(
            feature=self.feature,
            user_info=user_info,
            full_context=self.board.full_context_header(),
            approved_summary=self.board.approved_summary(),
            signoff_log=signoff_log,
            deferred_issues=deferred_text,
            amendments=amend_text,
            integration_verdict=self.board.integration_verdict,
        )
        self.board.emit(EventType.WRITING, "penny", "Writing Handover document...")
        self.ui.flush_events()
        self._set_memory("penny")
        resp = penny.think(self.board, task, HANDOVER_SYSTEM, self.client, max_tokens=6000)
        self._clear_memory()
        self.board.handover_doc = resp
        handover_path = self.board.docs_dir / "Handover.md"
        atomic_write(handover_path, resp)
        self.board.emit(EventType.WRITING, "penny", f"Handover saved: {handover_path}")
        self.ui.flush_events()

    def _generate_packaging_artifacts(self) -> None:
        """Generate stack-aware packaging files (pyproject.toml, README, Makefile, etc.)."""
        penny = AgentRoster.PENNY

        # Detect stack from research context
        stack = "generic"
        if self.board.research:
            lang = getattr(self.board.research, "language", "") or ""
            framework = getattr(self.board.research, "framework", "") or ""
            combined = f"{lang} {framework}".lower()
            if "python" in combined:
                stack = "Python"
            elif "node" in combined or "javascript" in combined or "typescript" in combined:
                stack = "Node.js"
            elif "go" in combined or "golang" in combined:
                stack = "Go"

        # Extract import lines from source files for dep detection
        import_lines: list[str] = []
        for entry in self.board.registry.values():
            if entry.approved and entry.code:
                for line in entry.code.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("from ") or \
                       stripped.startswith("require(") or stripped.startswith("import {"):
                        import_lines.append(stripped)
        source_imports = "\n".join(sorted(set(import_lines)))[:3000]

        # Derive kebab-case project name from feature
        project_name = re.sub(r"[^\w\s-]", "", self.feature.lower())
        project_name = re.sub(r"[\s_]+", "-", project_name.strip())[:40].strip("-")

        arch_lines = self.board.architecture.splitlines()
        architecture_summary = "\n".join(arch_lines[:30])

        task = PACKAGING_TASK.format(
            feature=self.feature,
            stack=stack,
            project_name=project_name,
            source_imports=source_imports,
            approved_summary=self.board.approved_summary(),
            architecture_summary=architecture_summary,
        )
        self.board.emit(EventType.WRITING, "penny",
                        f"Generating {stack} packaging artifacts...")
        self.ui.flush_events()
        self._set_memory("penny")
        resp = penny.think(self.board, task, PACKAGING_SYSTEM, self.client, max_tokens=6000)
        self._clear_memory()

        # Parse and write each artifact from fenced blocks
        artifact_count = 0
        for match in re.finditer(
            r"```filename:\s*(\S+)\s*\n(.*?)```", resp, re.DOTALL
        ):
            filename = match.group(1).strip()
            content = match.group(2)
            artifact_path = self.board.src_dir / filename
            atomic_write(artifact_path, content)
            self.board.emit(EventType.WRITING, "penny", f"Artifact saved: {artifact_path}")
            artifact_count += 1

        if artifact_count == 0:
            # Fallback: save the raw response as packaging_notes.md
            fallback = self.board.docs_dir / "packaging_notes.md"
            atomic_write(fallback, resp)
            self.board.emit(EventType.WRITING, "penny", f"Packaging notes saved: {fallback}")

        self.ui.flush_events()

    def _generate_delivery_checklist(self, deferred_text: str, user_info: str) -> None:
        """Morgan runs the final delivery checklist and project summary."""
        dm = AgentRoster.DM

        # Build list of docs that were generated
        docs_list = "\n".join(
            f"- {p.name} ({p.stat().st_size // 1024} KB)"
            for p in sorted(self.board.docs_dir.glob("*.md"))
        ) or "(none)"

        task = DM_TASK.format(
            feature=self.feature,
            user_info=user_info,
            full_context=self.board.full_context_header(),
            approved_summary=self.board.approved_summary(),
            deferred_issues=deferred_text,
            integration_verdict=self.board.integration_verdict,
            docs_list=docs_list,
        )
        self.board.emit(EventType.WRITING, "dm", "Running final delivery checklist...")
        self.ui.flush_events()
        resp = dm.think(self.board, task, DM_SYSTEM, self.client, max_tokens=4096)
        checklist_path = self.board.docs_dir / "delivery_checklist.md"
        atomic_write(checklist_path, resp)
        self.board.emit(EventType.WRITING, "dm", f"Delivery checklist saved: {checklist_path}")
        self.ui.flush_events()

    def _save(self) -> None:
        """Save a checkpoint + logbook + knowledge base + memories."""
        check_disk_space(self.board.checkpoints_dir)
        path = save_checkpoint(self.board)
        self.board.save_logbook()
        if self.board.knowledge_base:
            self.board.save_knowledge_base()
        if self.board.repo_analysis:
            self.board.save_repo_analysis()
        self.memory.save()
        self.board.emit(EventType.CHECKPOINT, "system", f"Saved: {path}")

    def _set_memory(self, agent_id: str) -> None:
        """Set the memory context on the board for the next agent.think() call."""
        self.board.memory_context = self.memory.context_for_agent(
            agent_id, phase=self.board.current_phase,
        )

    def _clear_memory(self) -> None:
        """Clear transient memory context after a think() call."""
        self.board.memory_context = ""

    def _record_mistake(self, agent_id: str, content: str, context: str = "") -> None:
        """Record a mistake in an agent's personal memory."""
        self.memory.get_agent(agent_id).remember(
            kind="mistake",
            content=content,
            context=context,
            phase=self.board.current_phase,
            source_project=self.board.project_slug,
        )

    def _record_pattern(self, agent_id: str, content: str, context: str = "") -> None:
        """Record a successful pattern in an agent's personal memory."""
        self.memory.get_agent(agent_id).remember(
            kind="pattern",
            content=content,
            context=context,
            phase=self.board.current_phase,
            source_project=self.board.project_slug,
        )

    def _record_lesson(self, agent_id: str, content: str, context: str = "") -> None:
        """Record a lesson learned in an agent's personal memory."""
        self.memory.get_agent(agent_id).remember(
            kind="lesson",
            content=content,
            context=context,
            phase=self.board.current_phase,
            source_project=self.board.project_slug,
        )

    def _push_team_insight(self, agent_id: str, content: str,
                           for_agents: list[str] | None = None) -> None:
        """Push an insight to the team memory board."""
        self.memory.team.push(
            agent_id=agent_id,
            content=content,
            for_agents=for_agents,
            phase=self.board.current_phase,
            source_project=self.board.project_slug,
        )

    def _build_repo_files_context(self, max_chars: int = 20_000) -> str:
        """Build a context string from the key files of ingested repo(s).

        Prioritises: README, config/main entry points, schemas, API routes.
        Falls back to knowledge_base items tagged 'git_repo'.
        """
        repo_items = [
            i for i in self.board.knowledge_base
            if "git_repo" in i.tags
        ]
        if not repo_items:
            return "(no repo files ingested)"

        # Sort by priority: docs first, then code, then data
        priority = {"document": 0, "api_spec": 1, "schema": 2,
                    "codebase": 3, "test_case": 4, "data_file": 5}
        repo_items.sort(key=lambda i: priority.get(i.source_type, 9))

        # Boost README / main / index / config files to the top
        def _boost_key(item: KnowledgeItem) -> int:
            name = item.label.lower()
            for i, kw in enumerate(["readme", "main", "index", "app", "config",
                                     "package.json", "setup.py", "pyproject"]):
                if kw in name:
                    return i
            return 99

        repo_items.sort(key=_boost_key)

        parts: list[str] = []
        budget = max_chars
        for item in repo_items:
            text = item.summary if item.was_summarized and item.summary else item.content
            entry = f"\n### {item.label} ({item.source_type})\n{text}"
            if len(entry) > budget:
                entry = entry[:budget] + "\n(... truncated ...)"
                parts.append(entry)
                break
            parts.append(entry)
            budget -= len(entry)

        return "\n".join(parts) if parts else "(no readable repo files)"

    @staticmethod
    def _clean_code(code: str) -> str:
        """Strip markdown code fences from LLM output (uses hardening module)."""
        return clean_code_fences(code)

    # ─────────────────────────────────────────────────────────────────────────
    #  Cost tracking — sync logbook entries to cost tracker
    # ─────────────────────────────────────────────────────────────────────────

    def _sync_costs(self) -> None:
        """Sync recent logbook entries to the cost tracker.

        Called periodically (after each phase) to keep cost/budget up to date.
        Only processes entries not yet tracked (idempotent via _last_cost_idx).
        """
        start_idx = getattr(self, '_last_cost_idx', 0)
        for entry in self.board.logbook[start_idx:]:
            self.cost_tracker.record_call(
                model=entry.model_used,
                input_tokens=entry.input_tokens,
                output_tokens=entry.output_tokens,
                cache_read_tokens=entry.cache_read_tokens,
                retries=entry.retries,
                success=entry.success,
            )
        self._last_cost_idx = len(self.board.logbook)

    # ─────────────────────────────────────────────────────────────────────────
    #  Project DNA — extract reusable knowledge after run
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_project_dna(self) -> None:
        """Extract structured lessons from the completed run.

        Uses an LLM call to analyze build outcomes and extract reusable
        patterns for future projects. Saves as project_dna.json and
        feeds into global memory.
        """
        approved = [e for e in self.board.registry.values() if e.approved]
        skipped = [e for e in self.board.registry.values() if e.skip_reason]

        # Build outcomes summary
        build_outcomes: list[str] = []
        for entry in self.board.registry.values():
            status = "✅ approved" if entry.approved else f"❌ {entry.skip_reason or 'failed'}"
            rev_note = f" ({entry.revision} revisions)" if entry.revision > 1 else ""
            deferred = f" [{len(entry.deferred_issues)} deferred]" if entry.deferred_issues else ""
            build_outcomes.append(f"  {entry.name}: {status}{rev_note}{deferred}")

        deferred_text = "\n".join(
            f"  - [{i.severity}] {fname}: {i.description}"
            for fname, i in self.board.all_deferred[:20]
        ) or "(none)"

        # Stack detection
        stack = "unknown"
        if self.board.research:
            s = self.board.research.stack
            if isinstance(s, dict):
                stack = f"{s.get('language', '?')} / {s.get('framework', '?')}"
            elif isinstance(s, str):
                stack = s

        arch_lines = self.board.architecture.splitlines()[:30]

        task = PROJECT_DNA_TASK.format(
            feature=self.feature,
            stack=stack,
            file_count=len(self.board.registry),
            approved_count=len(approved),
            skipped_count=len(skipped),
            llm_calls=len(self.board.logbook),
            retries=sum(e.retries for e in self.board.logbook),
            architecture_summary="\n".join(arch_lines),
            build_outcomes="\n".join(build_outcomes),
            deferred_issues=deferred_text,
            integration_verdict=self.board.integration_verdict,
        )

        self.board.emit(EventType.WRITING, "system",
                        "🧬 Extracting Project DNA...")
        self.ui.flush_events()

        scout = AgentRoster.SCOUT
        try:
            resp = scout.think(
                self.board, task, PROJECT_DNA_SYSTEM, self.client, max_tokens=2048,
            )
            dna = _parse_json(resp)
        except Exception as exc:
            logger.warning("Project DNA parse failed: %s", exc)
            dna = {"error": str(exc), "raw": resp if 'resp' in dir() else ""}

        # Save to docs
        dna_path = self.board.docs_dir / "project_dna.json"
        atomic_write(dna_path, json.dumps(dna, indent=2, default=str))
        self.board.emit(EventType.WRITING, "system",
                        f"🧬 Project DNA saved: {dna_path}")
        self.ui.flush_events()

        # Feed structured lessons into global memory
        for category in ("stack_patterns", "common_mistakes",
                         "architecture_lessons", "review_insights"):
            items = dna.get(category, [])
            if isinstance(items, list):
                for item in items[:5]:
                    self._record_lesson(
                        "scout",
                        f"[{category}] {item}",
                        context=f"project={self.board.project_slug}",
                    )

        # Distill to global memory
        new_lessons = self.memory.distill_to_global()
        self.memory.save_global()
        self.memory.save()
        if new_lessons:
            self.board.emit(EventType.SPEAKING, "system",
                            f"🧠 {len(new_lessons)} lessons distilled to global memory")
            self.ui.flush_events()
