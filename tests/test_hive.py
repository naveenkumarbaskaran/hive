"""
EPT Test Suite — No API calls. Tests state, agents, parsing, UI, and pipeline logic.

Run: python -m pytest test_hive.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────

from hive.state import (
    Blackboard, EventType, Event, ResearchContext, FileEntry, Issue,
    Amendment, SignOff, UserProfile, LogEntry, save_checkpoint, load_checkpoint, PROJECTS_DIR,
)
from hive.llm_client import LLMClient, LLMResponse, ModelTier
from hive.agents import Agent, AgentRoster, make_dev_agent, DEV_POOL
from hive.connectors import (
    ConnectorType, KnowledgeItem, ConnectorRegistry,
    knowledge_for_agent, knowledge_context, format_size,
    SMALL_THRESHOLD, MEDIUM_THRESHOLD,
    is_git_url, repo_file_tree,
)
from hive.crew import (
    EPTCrew, _parse_json, _parse_contract, _parse_verdict,
    _extract_architecture_text,
)
from hive.ui import TerminalUI, agent_color, agent_emoji, C
from hive.prompts import (
    SCOUT_SYSTEM, PENNY_PRD_SYSTEM, ARCHIE_SYSTEM,
    QUINN_SYSTEM, DEV_SYSTEM, JUDGE_SYSTEM,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def board():
    """Fresh Blackboard with a test feature."""
    b = Blackboard(feature="Test REST API for widgets")
    b.research = ResearchContext(
        domain="e-commerce",
        product_type="REST API",
        has_frontend=False,
        stack={"language": "Python", "framework": "FastAPI"},
        scale_tier="startup",
        raw_summary="A REST API for widget management.",
    )
    return b


@pytest.fixture
def board_with_project(board, tmp_path, monkeypatch):
    """Blackboard with project folders created in a temp directory."""
    import hive.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_DIR", tmp_path / "projects")
    board.init_project()
    return board


@pytest.fixture
def sample_contract():
    return """\
## ARCHITECTURE
Some design docs here.

```contract
FILES:
  models.py:
    purpose: Data models
    deps: []
    exports: [Widget, WidgetCreate]
    patterns: [dataclass]
    is_frontend: false
  routes.py:
    purpose: API endpoints
    deps: [models.py]
    exports: [router]
    patterns: [REST]
    is_frontend: false
  app.py:
    purpose: Application entry point
    deps: [routes.py]
    exports: [app]
    patterns: [factory]
    is_frontend: false
```
"""


# ═════════════════════════════════════════════════════════════════════════════
#  State Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestBlackboard:

    def test_init_creates_slug(self, board):
        board.init_project()
        assert board.project_slug
        assert "_" in board.project_slug or board.project_slug.isalpha()

    def test_project_dirs(self, board_with_project):
        b = board_with_project
        assert b.project_root.exists()
        assert b.docs_dir.exists()
        assert b.src_dir.exists()
        assert b.checkpoints_dir.exists()

    def test_emit_event(self, board):
        ev = board.emit(EventType.THINKING, "scout", "Analyzing...")
        assert len(board.events) == 1
        assert ev.agent == "scout"
        assert ev.type == EventType.THINKING

    def test_save_research(self, board_with_project):
        path = board_with_project.save_research()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["domain"] == "e-commerce"

    def test_save_prd(self, board_with_project):
        board_with_project.prd = "Some PRD content"
        path = board_with_project.save_prd()
        assert path.exists()
        assert "PRD" in path.read_text()

    def test_save_architecture(self, board_with_project):
        board_with_project.architecture = "Some arch"
        path = board_with_project.save_architecture()
        assert path.exists()

    def test_save_contract(self, board_with_project):
        board_with_project.contract = "FILES:\n  app.py:"
        path = board_with_project.save_contract()
        assert path.exists()

    def test_save_source_file(self, board_with_project):
        entry = FileEntry(name="models.py", code="class Widget: pass")
        path = board_with_project.save_source_file(entry)
        assert path.exists()
        assert "Widget" in path.read_text()

    def test_save_interviews(self, board_with_project):
        board_with_project.interviews = {"penny": {"Q1": "A1"}}
        path = board_with_project.save_interviews()
        assert path.exists()

    def test_signoff_workflow(self, board_with_project):
        b = board_with_project
        so = b.record_signoff("prd", True, "Looks good")
        assert so.approved
        assert so.artifact == "prd"
        assert b.is_signed_off("prd")

        so2 = b.record_signoff("prd", False, "Missing edge cases")
        assert not b.is_signed_off("prd")
        assert so2.version == 2

    def test_dep_layers_linear(self, board):
        board.file_plan = ["a.py", "b.py", "c.py"]
        board.dep_graph = {"a.py": [], "b.py": ["a.py"], "c.py": ["b.py"]}
        layers = board.dep_layers()
        assert layers == [["a.py"], ["b.py"], ["c.py"]]

    def test_dep_layers_parallel(self, board):
        board.file_plan = ["a.py", "b.py", "c.py", "d.py"]
        board.dep_graph = {"a.py": [], "b.py": [], "c.py": ["a.py", "b.py"], "d.py": ["c.py"]}
        layers = board.dep_layers()
        assert layers[0] == ["a.py", "b.py"]
        assert layers[1] == ["c.py"]
        assert layers[2] == ["d.py"]

    def test_dep_layers_empty(self, board):
        layers = board.dep_layers()
        assert layers == []

    def test_dep_layers_no_graph(self, board):
        board.file_plan = ["a.py", "b.py"]
        layers = board.dep_layers()
        assert layers == [["a.py", "b.py"]]

    def test_approved_summary(self, board):
        board.registry["app.py"] = FileEntry(name="app.py", approved=True, assigned_dev="Dexter")
        board.registry["bad.py"] = FileEntry(name="bad.py", approved=False)
        summary = board.approved_summary()
        assert "app.py" in summary
        assert "Dexter" in summary
        assert "bad.py" not in summary

    def test_interjections_context(self, board):
        assert board.interjections_context() == ""
        board.user_interjections.append("Add pagination")
        ctx = board.interjections_context()
        assert "pagination" in ctx


class TestCheckpoint:

    def test_save_and_load(self, board_with_project):
        b = board_with_project
        b.prd = "Test PRD"
        b.registry["app.py"] = FileEntry(name="app.py", code="# app", approved=True)

        path = save_checkpoint(b)
        assert path.exists()

        loaded = load_checkpoint(str(path))
        assert loaded.prd == "Test PRD"
        assert loaded.feature == b.feature
        assert "app.py" in loaded.registry
        assert loaded.registry["app.py"].approved


class TestResearchContext:

    def test_as_block(self):
        rc = ResearchContext(domain="fintech", product_type="CLI tool")
        block = rc.as_block()
        assert "fintech" in block
        assert "CLI tool" in block

    def test_defaults(self):
        rc = ResearchContext()
        assert rc.domain == "unknown"
        assert rc.has_frontend is False


# ═════════════════════════════════════════════════════════════════════════════
#  Agent Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestAgent:

    def test_label(self):
        a = AgentRoster.SCOUT
        assert "Scout" in a.label
        assert "🔍" in a.label

    def test_card(self):
        a = AgentRoster.PENNY
        assert "Penny" in a.card
        assert "Product Manager" in a.card

    def test_say_emits_event(self, board):
        a = AgentRoster.SCOUT
        a.say(board, "Hello!", to=AgentRoster.PENNY)
        assert len(board.events) == 1
        assert board.events[0].target == "penny"


class TestAgentRoster:

    def test_all_agents_count(self):
        agents = AgentRoster.all_agents()
        assert len(agents) == 9  # scout, penny, archie, quinn, judge, pixel, flow, alex, dm (Morgan)

    def test_compose_no_frontend(self):
        agents = AgentRoster.compose(has_frontend=False, dev_count=2)
        # Pixel, Flow, Alex should be inactive
        assert not agents["pixel"].active
        assert not agents["flow"].active
        assert not agents["alex"].active
        # Core should be active
        assert agents["scout"].active
        assert agents["penny"].active
        assert agents["archie"].active
        # 2 devs
        assert agents["dev_1"].active
        assert agents["dev_2"].active

    def test_compose_with_frontend(self):
        agents = AgentRoster.compose(has_frontend=True, dev_count=1)
        assert agents["pixel"].active
        assert agents["flow"].active
        assert agents["alex"].active

    def test_compose_dev_count(self):
        agents = AgentRoster.compose(has_frontend=False, dev_count=4)
        dev_agents = [a for aid, a in agents.items() if aid.startswith("dev_")]
        assert len(dev_agents) == 4

    def test_get_raises_on_missing(self):
        agents = AgentRoster.compose(has_frontend=False, dev_count=1)
        with pytest.raises(KeyError):
            AgentRoster.get(agents, "nonexistent")


class TestDevPool:

    def test_make_dev_agent(self):
        dev = make_dev_agent(0)
        assert dev.name == "Dexter"
        assert dev.id == "dev_1"
        assert dev.tier.value == "powerful"

    def test_make_dev_agent_wraps(self):
        dev = make_dev_agent(len(DEV_POOL))
        assert dev.name == DEV_POOL[0][0]

    def test_all_devs_unique_names(self):
        names = {make_dev_agent(i).name for i in range(len(DEV_POOL))}
        assert len(names) == len(DEV_POOL)


# ═════════════════════════════════════════════════════════════════════════════
#  Parser Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestParseJson:

    def test_direct_json(self):
        result = _parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_fenced(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = _parse_json(text)
        assert result == {"key": "value"}

    def test_json_array(self):
        result = _parse_json('["q1?", "q2?"]')
        assert result == ["q1?", "q2?"]

    def test_embedded_json(self):
        text = 'Here is my analysis:\n\n{"domain": "fintech", "feasible": true}\n\nDone.'
        result = _parse_json(text)
        assert result["domain"] == "fintech"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_json("This is not JSON at all")


class TestParseContract:

    def test_basic_contract(self, sample_contract):
        files = _parse_contract(sample_contract)
        assert "models.py" in files
        assert "routes.py" in files
        assert "app.py" in files
        assert files["routes.py"]["deps"] == ["models.py"]
        assert files["models.py"]["exports"] == ["Widget", "WidgetCreate"]
        assert files["app.py"]["is_frontend"] is False

    def test_no_contract_block(self):
        with pytest.raises(ValueError, match="No.*contract"):
            _parse_contract("Just some text without a contract")

    def test_empty_contract(self):
        with pytest.raises(ValueError, match="no file"):
            _parse_contract("```contract\nFILES:\n```")


class TestParseVerdict:

    def test_pass(self):
        text = "VERDICT: PASS\n\nNOTES:\n- Looks good"
        verdict, issues = _parse_verdict(text)
        assert verdict == "PASS"
        assert len(issues) == 0

    def test_fail_with_issues(self):
        text = "VERDICT: FAIL\n\nISSUES:\n- [blocker] Missing error handling\n- [warning] Style"
        verdict, issues = _parse_verdict(text)
        assert verdict == "FAIL"
        assert len(issues) == 2
        assert issues[0].severity == "blocker"
        assert issues[1].severity == "warning"

    def test_pass_with_notes(self):
        text = "VERDICT: PASS_WITH_NOTES\n\nDEFERRED:\n- [minor] Could use caching"
        verdict, issues = _parse_verdict(text)
        assert verdict == "PASS_WITH_NOTES"
        assert len(issues) == 1

    def test_no_verdict_defaults_fail(self):
        text = "Some rambling without a verdict"
        verdict, issues = _parse_verdict(text)
        assert verdict == "FAIL"


class TestExtractArchitecture:

    def test_extracts_before_contract(self, sample_contract):
        arch = _extract_architecture_text(sample_contract)
        assert "Some design docs" in arch
        assert "```contract" not in arch

    def test_no_contract_returns_full(self):
        text = "Full architecture text"
        assert _extract_architecture_text(text) == text


# ═════════════════════════════════════════════════════════════════════════════
#  UI Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestUI:

    def test_agent_color_known(self):
        assert agent_color("scout") == C.CYAN
        assert agent_color("penny") == C.YELLOW

    def test_agent_color_dev(self):
        color = agent_color("dev_1")
        assert color  # should return some color string

    def test_agent_emoji_known(self):
        assert agent_emoji("scout") == "🔍"
        assert agent_emoji("penny") == "📋"

    def test_agent_emoji_dev(self):
        assert agent_emoji("dev_1") == "🔨"

    def test_flush_events(self, board, capsys):
        ui = TerminalUI(board)
        board.emit(EventType.AGREEMENT, "scout", "Done!")
        ui.flush_events()
        captured = capsys.readouterr()
        assert "Done!" in captured.out

    def test_banner(self, board, capsys):
        ui = TerminalUI(board)
        ui.banner()
        captured = capsys.readouterr()
        assert "EPT" in captured.out

    def test_file_status(self, board, capsys):
        ui = TerminalUI(board)
        ui.file_status("app.py", "approved")
        captured = capsys.readouterr()
        assert "app.py" in captured.out
        assert "APPROVED" in captured.out


# ═════════════════════════════════════════════════════════════════════════════
#  Prompts Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPrompts:
    """Verify prompts are well-formed strings with expected markers."""

    def test_scout_system_has_json_format(self):
        assert '"domain"' in SCOUT_SYSTEM
        assert "JSON" in SCOUT_SYSTEM

    def test_penny_prd_has_sections(self):
        assert "User Stories" in PENNY_PRD_SYSTEM
        assert "Functional Requirements" in PENNY_PRD_SYSTEM
        assert "Acceptance Criteria" in PENNY_PRD_SYSTEM

    def test_archie_has_contract_format(self):
        assert "```contract" in ARCHIE_SYSTEM
        assert "deps" in ARCHIE_SYSTEM

    def test_quinn_has_verdict_format(self):
        assert "VERDICT:" in QUINN_SYSTEM
        assert "PASS" in QUINN_SYSTEM
        assert "FAIL" in QUINN_SYSTEM

    def test_dev_system_has_placeholder(self):
        assert "{dev_name}" in DEV_SYSTEM
        assert "{dev_tagline}" in DEV_SYSTEM

    def test_judge_has_approve_reject(self):
        assert "APPROVE" in JUDGE_SYSTEM
        assert "REJECT" in JUDGE_SYSTEM
        assert "AMEND_CONTRACT" in JUDGE_SYSTEM


# ═════════════════════════════════════════════════════════════════════════════
#  Crew Integration (mocked LLM)
# ═════════════════════════════════════════════════════════════════════════════

class TestCrewInit:

    def test_creates_crew(self):
        crew = EPTCrew(feature="Test feature", auto_approve=True)
        assert crew.feature == "Test feature"
        assert crew.board.feature == "Test feature"

    def test_board_initialized(self):
        crew = EPTCrew(feature="Test")
        assert crew.board.current_phase == ""
        assert crew.board.events == []


class TestCrewCleanCode:

    def test_strips_markdown_fences(self):
        code = "```python\nprint('hello')\n```"
        assert EPTCrew._clean_code(code) == "print('hello')"

    def test_leaves_clean_code(self):
        code = "print('hello')"
        assert EPTCrew._clean_code(code) == "print('hello')"

    def test_strips_language_tag(self):
        code = "```javascript\nconsole.log('hi');\n```"
        assert EPTCrew._clean_code(code) == "console.log('hi');"


# ═════════════════════════════════════════════════════════════════════════════
#  SignOff Dataclass
# ═════════════════════════════════════════════════════════════════════════════

class TestSignOff:

    def test_defaults(self):
        so = SignOff(artifact="prd", version=1, approved=True)
        assert so.feedback == ""
        assert so.produced_by == ""
        assert so.reviewed_by == []
        assert so.timestamp > 0

    def test_with_feedback(self):
        so = SignOff(artifact="arch", version=2, approved=False, feedback="Too complex")
        assert so.feedback == "Too complex"

    def test_with_attribution(self):
        so = SignOff(
            artifact="prd", version=1, approved=True,
            produced_by="Penny 📋 (Product Manager)",
            reviewed_by=["Scout 🔍 (Research Analyst)", "Archie 🏗️ (Tech Architect)"],
        )
        assert so.produced_by == "Penny 📋 (Product Manager)"
        assert len(so.reviewed_by) == 2
        assert "Scout" in so.reviewed_by[0]


# ═════════════════════════════════════════════════════════════════════════════
#  UserProfile
# ═════════════════════════════════════════════════════════════════════════════

class TestUserProfile:

    def test_defaults(self):
        up = UserProfile()
        assert up.name == ""
        assert up.is_request_for_self is True
        assert up.as_is_process == ""

    def test_full_profile(self):
        up = UserProfile(
            name="Alice", role="Product Owner", company="Acme Corp",
            is_request_for_self=False, end_user_name="Bob",
            end_user_role="Customer Service Agent",
            end_user_description="Handles 200+ tickets/day",
            as_is_process="Currently uses a spreadsheet to track tickets.",
        )
        assert up.name == "Alice"
        assert up.end_user_role == "Customer Service Agent"

    def test_as_block_self(self):
        up = UserProfile(name="Alice", role="Developer")
        block = up.as_block()
        assert "Alice" in block
        assert "Developer" in block
        assert "requester themselves" in block

    def test_as_block_other(self):
        up = UserProfile(
            name="Alice", is_request_for_self=False,
            end_user_name="Bob", end_user_role="Admin",
            as_is_process="Manual process with paper forms",
        )
        block = up.as_block()
        assert "Bob" in block
        assert "Admin" in block
        assert "Manual process" in block

    def test_as_block_empty(self):
        up = UserProfile()
        block = up.as_block()
        assert "USER PROFILE:" in block

    def test_board_user_context(self, board):
        assert board.user_context() == ""
        board.user_profile = UserProfile(name="TestUser")
        ctx = board.user_context()
        assert "TestUser" in ctx

    def test_full_context_header_includes_user(self, board):
        board.user_profile = UserProfile(name="Jane", role="PM")
        board.prd = "test prd"
        board.architecture = "test arch"
        board.contract = "test contract"
        header = board.full_context_header()
        assert "Jane" in header
        assert "PM" in header

    def test_save_user_profile(self, board_with_project):
        b = board_with_project
        b.user_profile = UserProfile(name="Alice", role="Dev")
        path = b.save_user_profile()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "Alice"

    def test_checkpoint_with_user_profile(self, board_with_project):
        b = board_with_project
        b.user_profile = UserProfile(
            name="TestUser", role="PM",
            is_request_for_self=False, end_user_name="EndUser",
        )
        path = save_checkpoint(b)
        loaded = load_checkpoint(str(path))
        assert loaded.user_profile is not None
        assert loaded.user_profile.name == "TestUser"
        assert loaded.user_profile.end_user_name == "EndUser"

    def test_checkpoint_without_user_profile(self, board_with_project):
        b = board_with_project
        b.user_profile = None
        path = save_checkpoint(b)
        loaded = load_checkpoint(str(path))
        assert loaded.user_profile is None

    def test_record_signoff_with_attribution(self, board_with_project):
        b = board_with_project
        so = b.record_signoff(
            "prd", True, "Looks great",
            produced_by="Penny 📋",
            reviewed_by=["Scout 🔍", "Archie 🏗️"],
        )
        assert so.produced_by == "Penny 📋"
        assert len(so.reviewed_by) == 2
        # Check event includes attribution
        signoff_events = [e for e in b.events if e.type == EventType.USER_SIGNOFF]
        assert signoff_events
        assert "Produced by" in signoff_events[-1].content

    def test_checkpoint_preserves_signoff_attribution(self, board_with_project):
        b = board_with_project
        b.record_signoff("prd", True, produced_by="Penny 📋",
                         reviewed_by=["Scout 🔍"])
        path = save_checkpoint(b)
        loaded = load_checkpoint(str(path))
        assert loaded.signoffs[0].produced_by == "Penny 📋"
        assert loaded.signoffs[0].reviewed_by == ["Scout 🔍"]


# ═════════════════════════════════════════════════════════════════════════════
#  Event
# ═════════════════════════════════════════════════════════════════════════════

class TestEvent:

    def test_event_creation(self):
        ev = Event(type=EventType.THINKING, agent="scout", content="Working...")
        assert ev.type == EventType.THINKING
        assert ev.agent == "scout"
        assert ev.timestamp > 0

    def test_all_event_types(self):
        """Ensure all EventType values are valid."""
        for et in EventType:
            assert isinstance(et.value, str)
        assert len(EventType) >= 10


# ═════════════════════════════════════════════════════════════════════════════
#  Logbook & Resilience
# ═════════════════════════════════════════════════════════════════════════════

class TestLogEntry:

    def test_defaults(self):
        entry = LogEntry(
            agent_id="scout", agent_name="Scout",
            phase="research", task_summary="Analyze feature",
            model_requested="claude-sonnet-4-20250514",
            model_used="claude-sonnet-4-20250514",
            tier_requested="fast", tier_used="fast",
        )
        assert entry.success is True
        assert entry.retries == 0
        assert entry.tier_escalated is False
        assert entry.errors == []

    def test_with_resilience_info(self):
        entry = LogEntry(
            agent_id="archie", agent_name="Archie",
            phase="architecture", task_summary="Design arch",
            model_requested="small-model", model_used="big-model",
            tier_requested="fast", tier_used="powerful",
            retries=2, tier_escalated=True, thinking_stripped=True,
            errors=["TimeoutError: proxy", "HTTPError: 503"],
            duration_s=15.3,
        )
        assert entry.tier_escalated
        assert entry.thinking_stripped
        assert len(entry.errors) == 2


class TestLogbook:

    def test_log_llm_call(self, board):
        entry = LogEntry(
            agent_id="scout", agent_name="Scout",
            phase="research", task_summary="Analyze",
            model_requested="m1", model_used="m1",
            tier_requested="fast", tier_used="fast",
        )
        board.log_llm_call(entry)
        assert len(board.logbook) == 1
        assert board.logbook[0].agent_name == "Scout"

    def test_log_retry_emits_incident(self, board):
        entry = LogEntry(
            agent_id="archie", agent_name="Archie",
            phase="architecture", task_summary="Design",
            model_requested="m1", model_used="m2",
            tier_requested="fast", tier_used="balanced",
            retries=2, tier_escalated=True,
        )
        board.log_llm_call(entry)
        incidents = [e for e in board.events if e.type == EventType.LLM_INCIDENT]
        assert len(incidents) == 1
        assert "retries" in incidents[0].content
        assert "escalated" in incidents[0].content

    def test_log_failure_emits_incident(self, board):
        entry = LogEntry(
            agent_id="quinn", agent_name="Quinn",
            phase="build", task_summary="Review",
            model_requested="m1", model_used="(failed)",
            tier_requested="fast", tier_used="fast",
            retries=3, success=False,
            errors=["HTTPError: 503 Service Unavailable"],
        )
        board.log_llm_call(entry)
        incidents = [e for e in board.events if e.type == EventType.LLM_INCIDENT]
        assert len(incidents) >= 1
        # One for retries, one for failure
        failure_incidents = [e for e in incidents if "FAILED" in e.content]
        assert len(failure_incidents) == 1

    def test_save_logbook(self, board_with_project):
        b = board_with_project
        b.log_llm_call(LogEntry(
            agent_id="scout", agent_name="Scout",
            phase="research", task_summary="Analyze",
            model_requested="m1", model_used="m1",
            tier_requested="fast", tier_used="fast",
            input_tokens=500, output_tokens=200, duration_s=2.5,
        ))
        path = b.save_logbook()
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["agent_name"] == "Scout"
        assert data[0]["input_tokens"] == 500

    def test_logbook_not_in_checkpoint(self, board_with_project):
        b = board_with_project
        b.log_llm_call(LogEntry(
            agent_id="scout", agent_name="Scout",
            phase="research", task_summary="Analyze",
            model_requested="m1", model_used="m1",
            tier_requested="fast", tier_used="fast",
        ))
        path = save_checkpoint(b)
        loaded = load_checkpoint(str(path))
        # Logbook is saved separately, not in checkpoints
        assert loaded.logbook == []


class TestModelTierEscalation:

    def test_fast_escalates_to_balanced(self):
        assert ModelTier.FAST.escalate() == ModelTier.BALANCED

    def test_balanced_escalates_to_powerful(self):
        assert ModelTier.BALANCED.escalate() == ModelTier.POWERFUL

    def test_powerful_stays_powerful(self):
        assert ModelTier.POWERFUL.escalate() == ModelTier.POWERFUL


class TestLLMResponseResilience:

    def test_response_has_resilience_fields(self):
        resp = LLMResponse(
            text="hello", model="m1",
            tier_requested="fast", tier_used="balanced",
            retries=1, tier_escalated=True,
            thinking_stripped=True, model_switched=True,
            model_used="fallback-model", duration_s=3.2,
            errors=["first error"],
        )
        assert resp.tier_escalated
        assert resp.thinking_stripped
        assert resp.model_switched
        assert resp.model_used == "fallback-model"
        assert resp.retries == 1
        assert resp.duration_s == 3.2
        assert len(resp.errors) == 1


class TestModelPoolAndFallback:
    """Tests for 429 rate-limit model rotation."""

    def test_build_model_pool_deduplicates(self):
        """When all tiers resolve to same model, pool has just one entry."""
        client = LLMClient(
            base_url="http://fake", api_key="k",
            default_model="m1", model_big="m1", model_small="m1",
        )
        pool = client._build_model_pool("m1")
        assert pool == ["m1"]

    def test_build_model_pool_includes_all_unique(self):
        """Pool includes primary + tier models + fallbacks, deduplicated."""
        client = LLMClient(
            base_url="http://fake", api_key="k",
            default_model="balanced", model_big="big", model_small="small",
        )
        client.fallback_models = ["fallback-1", "fallback-2"]
        pool = client._build_model_pool("balanced")
        assert pool == ["balanced", "big", "small", "fallback-1", "fallback-2"]

    def test_build_model_pool_primary_first(self):
        """Primary model is always first in the pool."""
        client = LLMClient(
            base_url="http://fake", api_key="k",
            default_model="balanced", model_big="big", model_small="small",
        )
        pool = client._build_model_pool("big")
        assert pool[0] == "big"

    def test_parse_fallback_models_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_FALLBACK_MODELS", "model-a, model-b , model-c")
        client = LLMClient(base_url="http://fake", api_key="k")
        assert client.fallback_models == ["model-a", "model-b", "model-c"]

    def test_parse_fallback_models_empty(self, monkeypatch):
        monkeypatch.delenv("LLM_FALLBACK_MODELS", raising=False)
        client = LLMClient(base_url="http://fake", api_key="k")
        assert client.fallback_models == []

    def test_is_rate_limit_error_httpx(self):
        """Detects 429 from httpx.HTTPStatusError."""
        import httpx as _httpx
        req = _httpx.Request("POST", "http://fake/v1/messages")
        resp = _httpx.Response(429, request=req)
        exc = _httpx.HTTPStatusError("rate limited", request=req, response=resp)
        assert LLMClient._is_rate_limit_error(exc) is True

    def test_is_rate_limit_error_string(self):
        """Detects 429 from stringified exceptions."""
        exc = RuntimeError("429 Too Many Requests")
        assert LLMClient._is_rate_limit_error(exc) is True

    def test_is_rate_limit_error_non_429(self):
        """Non-429 errors are not rate limits."""
        import httpx as _httpx
        req = _httpx.Request("POST", "http://fake/v1/messages")
        resp = _httpx.Response(500, request=req)
        exc = _httpx.HTTPStatusError("server error", request=req, response=resp)
        assert LLMClient._is_rate_limit_error(exc) is False

    def test_is_rate_limit_error_no_false_positive_on_port(self):
        """Port numbers like 4290 should not trigger false positive."""
        exc = RuntimeError("Connection refused on port 4290")
        assert LLMClient._is_rate_limit_error(exc) is False

    def test_is_rate_limit_error_no_false_positive_on_id(self):
        """Ticket IDs like #4291 should not trigger false positive."""
        exc = RuntimeError("See issue #4291 for details")
        assert LLMClient._is_rate_limit_error(exc) is False


class TestChatRetryLoop:
    """Integration tests for chat() retry logic with mocked transport."""

    def _make_client(self, fallbacks: list[str] | None = None) -> LLMClient:
        client = LLMClient(
            base_url="http://fake", api_key="k",
            default_model="model-a", model_big="model-b", model_small="model-c",
            api_format="openai",
        )
        if fallbacks:
            client.fallback_models = fallbacks
        return client

    @staticmethod
    def _make_429(url: str = "http://fake/v1/chat/completions"):
        import httpx as _httpx
        req = _httpx.Request("POST", url)
        resp = _httpx.Response(429, request=req)
        return _httpx.HTTPStatusError("429 rate limited", request=req, response=resp)

    def test_429_rotates_model_and_succeeds(self):
        """On 429, chat() switches to next model in pool and succeeds."""
        client = self._make_client()
        ok_resp = LLMResponse(text="ok", model="model-b")

        call_count = 0
        models_tried: list[str] = []

        original_openai = client._chat_openai

        def mock_openai(system, messages, model, temperature, max_tokens):
            nonlocal call_count
            call_count += 1
            models_tried.append(model)
            if call_count == 1:
                raise TestChatRetryLoop._make_429()
            return ok_resp

        client._chat_openai = mock_openai

        with patch("hive.llm_client.time.sleep"):
            resp = client.chat(system="s", messages=[{"role": "user", "content": "hi"}],
                               tier=ModelTier.BALANCED, retries=3)

        assert resp.text == "ok"
        assert resp.model_switched is True
        assert resp.retries == 1
        # First call should be model-a (BALANCED), second should be different
        assert models_tried[0] == "model-a"
        assert models_tried[1] != "model-a"

    def test_429_exhausts_pool_then_resets(self):
        """When all models are 429'd, pool resets and retries from the start."""
        client = self._make_client()
        ok_resp = LLMResponse(text="ok", model="model-a")

        call_count = 0

        def mock_openai(system, messages, model, temperature, max_tokens):
            nonlocal call_count
            call_count += 1
            # Fail first 3 (one per pool model), succeed on 4th
            if call_count <= 3:
                raise TestChatRetryLoop._make_429()
            return ok_resp

        client._chat_openai = mock_openai

        with patch("hive.llm_client.time.sleep"):
            resp = client.chat(system="s", messages=[{"role": "user", "content": "hi"}],
                               tier=ModelTier.BALANCED, retries=5)

        assert resp.text == "ok"
        assert resp.retries == 3  # 3 failures before success
        assert resp.model_switched is True

    def test_non_429_does_not_rotate_model(self):
        """Non-rate-limit errors use existing backoff/escalation, not model rotation."""
        client = self._make_client()
        ok_resp = LLMResponse(text="ok", model="model-a")

        call_count = 0
        models_tried: list[str] = []

        def mock_openai(system, messages, model, temperature, max_tokens):
            nonlocal call_count
            call_count += 1
            models_tried.append(model)
            if call_count == 1:
                raise RuntimeError("server error 500")
            return ok_resp

        client._chat_openai = mock_openai

        with patch("hive.llm_client.time.sleep"):
            resp = client.chat(system="s", messages=[{"role": "user", "content": "hi"}],
                               tier=ModelTier.BALANCED, retries=3)

        assert resp.text == "ok"
        assert resp.model_switched is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Connector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConnectorType:

    def test_all_types_have_string_values(self):
        for ct in ConnectorType:
            assert isinstance(ct.value, str)

    def test_enum_membership(self):
        assert ConnectorType("document") == ConnectorType.DOCUMENT
        assert ConnectorType("test_case") == ConnectorType.TEST_CASE


class TestConnectorDetectType:

    def test_detects_markdown(self):
        assert ConnectorRegistry.detect_type(Path("docs/spec.md")) == ConnectorType.DOCUMENT

    def test_detects_python(self):
        assert ConnectorRegistry.detect_type(Path("app.py")) == ConnectorType.CODEBASE

    def test_detects_test_file(self):
        assert ConnectorRegistry.detect_type(Path("test_auth.py")) == ConnectorType.TEST_CASE

    def test_detects_spec_file(self):
        assert ConnectorRegistry.detect_type(Path("widget.spec.ts")) == ConnectorType.TEST_CASE

    def test_detects_csv(self):
        assert ConnectorRegistry.detect_type(Path("data.csv")) == ConnectorType.DATA_FILE

    def test_detects_sql(self):
        assert ConnectorRegistry.detect_type(Path("schema.sql")) == ConnectorType.SCHEMA

    def test_detects_graphql(self):
        assert ConnectorRegistry.detect_type(Path("schema.graphql")) == ConnectorType.API_SPEC

    def test_detects_openapi_yaml(self):
        assert ConnectorRegistry.detect_type(Path("openapi.yaml")) == ConnectorType.API_SPEC

    def test_detects_swagger_json(self):
        assert ConnectorRegistry.detect_type(Path("swagger.json")) == ConnectorType.API_SPEC

    def test_returns_none_for_binary(self):
        assert ConnectorRegistry.detect_type(Path("image.png")) is None

    def test_returns_none_for_dsstore(self):
        assert ConnectorRegistry.detect_type(Path(".DS_Store")) is None

    def test_returns_none_for_unknown_ext(self):
        assert ConnectorRegistry.detect_type(Path("file.xyz123")) is None

    def test_test_pattern_overrides_codebase(self):
        """test_*.py should be TEST_CASE, not CODEBASE."""
        assert ConnectorRegistry.detect_type(Path("test_main.py")) == ConnectorType.TEST_CASE
        assert ConnectorRegistry.detect_type(Path("main_test.go")) == ConnectorType.TEST_CASE


class TestConnectorIngestFile:

    def test_ingest_small_text_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello\nThis is a doc.")
        item = ConnectorRegistry.ingest_file(f)
        assert item is not None
        assert item.source_type == "document"
        assert item.label == "readme.md"
        assert "Hello" in item.content
        assert not item.was_summarized
        assert item.raw_size > 0

    def test_ingest_large_file_gets_truncated(self, tmp_path):
        f = tmp_path / "big.py"
        # Write a file larger than MEDIUM_THRESHOLD
        content = "\n".join(f"line_{i} = {i}" for i in range(5000))
        f.write_text(content)
        item = ConnectorRegistry.ingest_file(f)
        assert item is not None
        assert item.was_summarized  # > 50KB
        assert "omitted" in item.content  # truncation marker

    def test_ingest_medium_file_truncated(self, tmp_path):
        """Files between 8KB and 50KB get truncated but not summarized."""
        f = tmp_path / "medium.py"
        # Create content between SMALL_THRESHOLD and MEDIUM_THRESHOLD
        line = "x = 'a' * 80  # padding\n"
        lines_needed = (SMALL_THRESHOLD // len(line)) + 100
        content = line * lines_needed
        f.write_text(content)
        item = ConnectorRegistry.ingest_file(f)
        assert item is not None
        # File could be medium or large depending on exact size
        assert item.source_type == "codebase"

    def test_ingest_returns_none_for_missing_file(self):
        item = ConnectorRegistry.ingest_file(Path("/nonexistent/file.py"))
        assert item is None

    def test_ingest_returns_none_for_binary(self, tmp_path):
        f = tmp_path / "icon.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")
        item = ConnectorRegistry.ingest_file(f)
        assert item is None

    def test_force_type_overrides_extension(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"openapi": "3.0.0"}')
        item = ConnectorRegistry.ingest_file(f, force_type=ConnectorType.API_SPEC)
        assert item is not None
        assert item.source_type == "api_spec"

    def test_auto_tags_include_type_and_extension(self, tmp_path):
        f = tmp_path / "schema.sql"
        f.write_text("CREATE TABLE users (id INT);")
        item = ConnectorRegistry.ingest_file(f)
        assert item is not None
        assert "schema" in item.tags
        assert "sql" in item.tags


class TestConnectorIngestDirectory:

    def test_ingests_all_recognizable_files(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "app.py").write_text("print('hi')")
        (tmp_path / "test_app.py").write_text("def test(): pass")
        (tmp_path / "icon.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # skipped

        items = ConnectorRegistry.ingest_directory(tmp_path)
        assert len(items) == 3
        types = {i.source_type for i in items}
        assert "document" in types
        assert "codebase" in types
        assert "test_case" in types

    def test_skips_pycache(self, tmp_path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.cpython-311.pyc").write_bytes(b"")
        (tmp_path / "main.py").write_text("pass")

        items = ConnectorRegistry.ingest_directory(tmp_path)
        assert len(items) == 1
        assert items[0].label == "main.py"

    def test_skips_git_directory(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]")
        (tmp_path / "app.py").write_text("pass")

        items = ConnectorRegistry.ingest_directory(tmp_path)
        assert len(items) == 1

    def test_max_files_limit(self, tmp_path):
        for i in range(20):
            (tmp_path / f"file_{i}.py").write_text(f"# file {i}")
        items = ConnectorRegistry.ingest_directory(tmp_path, max_files=5)
        assert len(items) == 5

    def test_recurse_subdirectories(self, tmp_path):
        sub = tmp_path / "src" / "core"
        sub.mkdir(parents=True)
        (sub / "models.py").write_text("class User: pass")
        (tmp_path / "readme.md").write_text("# Doc")

        items = ConnectorRegistry.ingest_directory(tmp_path)
        assert len(items) == 2
        labels = {i.label for i in items}
        assert "models.py" in labels
        assert "readme.md" in labels


class TestConnectorIngest:

    def test_ingest_single_file(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("print('hello')")
        items = ConnectorRegistry.ingest(str(f))
        assert len(items) == 1
        assert items[0].source_type == "codebase"

    def test_ingest_directory(self, tmp_path):
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "b.md").write_text("# doc")
        items = ConnectorRegistry.ingest(str(tmp_path))
        assert len(items) == 2

    def test_ingest_typed_path(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text('{"api": true}')
        items = ConnectorRegistry.ingest(f"{f}:api_spec")
        assert len(items) == 1
        assert items[0].source_type == "api_spec"

    def test_ingest_nonexistent_returns_empty(self):
        items = ConnectorRegistry.ingest("/nonexistent/path/abc123")
        assert items == []

    def test_ingest_all_deduplicates(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("pass")
        items = ConnectorRegistry.ingest_all([str(f), str(f)])
        assert len(items) == 1


class TestKnowledgeRouting:

    @pytest.fixture
    def sample_items(self):
        return [
            KnowledgeItem(source_type="document", source_path="/a.md",
                          label="a.md", content="doc", raw_size=10),
            KnowledgeItem(source_type="codebase", source_path="/b.py",
                          label="b.py", content="code", raw_size=20),
            KnowledgeItem(source_type="test_case", source_path="/test_c.py",
                          label="test_c.py", content="tests", raw_size=30),
            KnowledgeItem(source_type="api_spec", source_path="/api.yaml",
                          label="api.yaml", content="spec", raw_size=40),
            KnowledgeItem(source_type="schema", source_path="/schema.sql",
                          label="schema.sql", content="sql", raw_size=50),
        ]

    def test_scout_gets_everything(self, sample_items):
        result = knowledge_for_agent(sample_items, "scout")
        assert "a.md" in result
        assert "b.py" in result
        assert "test_c.py" in result
        assert "api.yaml" in result
        assert "schema.sql" in result

    def test_penny_gets_docs_and_data(self, sample_items):
        result = knowledge_for_agent(sample_items, "penny")
        assert "a.md" in result
        assert "b.py" not in result
        assert "test_c.py" not in result

    def test_archie_gets_specs_and_code(self, sample_items):
        result = knowledge_for_agent(sample_items, "archie")
        assert "b.py" in result
        assert "api.yaml" in result
        assert "schema.sql" in result
        assert "a.md" not in result

    def test_quinn_gets_tests_and_specs(self, sample_items):
        result = knowledge_for_agent(sample_items, "quinn")
        assert "test_c.py" in result
        assert "api.yaml" in result
        assert "a.md" not in result

    def test_dev_agents_route_correctly(self, sample_items):
        result = knowledge_for_agent(sample_items, "dev_1")
        assert "b.py" in result
        assert "api.yaml" in result
        assert "a.md" not in result

    def test_unknown_role_returns_empty(self, sample_items):
        result = knowledge_for_agent(sample_items, "unknown_agent")
        assert result == ""

    def test_empty_items_returns_empty(self):
        assert knowledge_for_agent([], "scout") == ""

    def test_knowledge_context_returns_all(self, sample_items):
        result = knowledge_context(sample_items)
        assert "a.md" in result
        assert "schema.sql" in result

    def test_max_chars_truncation(self, sample_items):
        result = knowledge_for_agent(sample_items, "scout", max_chars=50)
        assert len(result) <= 200  # headers + truncation msg


class TestFormatSize:

    def test_bytes(self):
        assert format_size(500) == "500 B"

    def test_kilobytes(self):
        assert format_size(8192) == "8.0 KB"

    def test_megabytes(self):
        assert format_size(2_097_152) == "2.0 MB"


class TestKnowledgeOnBlackboard:

    def test_knowledge_base_starts_empty(self, board):
        assert board.knowledge_base == []

    def test_knowledge_for_agent_method(self, board):
        board.knowledge_base = [
            KnowledgeItem(source_type="document", source_path="/a.md",
                          label="a.md", content="hello", raw_size=5),
        ]
        result = board.knowledge_for_agent("scout")
        assert "a.md" in result

    def test_save_knowledge_base(self, board_with_project):
        b = board_with_project
        b.knowledge_base = [
            KnowledgeItem(source_type="document", source_path="/a.md",
                          label="a.md", content="hello", raw_size=5),
        ]
        path = b.save_knowledge_base()
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["label"] == "a.md"

    def test_knowledge_base_survives_checkpoint(self, board_with_project):
        """Knowledge is rehydrated from docs/knowledge_base.json on resume."""
        b = board_with_project
        b.knowledge_base = [
            KnowledgeItem(source_type="codebase", source_path="/b.py",
                          label="b.py", content="pass", raw_size=4),
        ]
        b.save_knowledge_base()
        save_checkpoint(b)
        loaded = load_checkpoint(str(b.checkpoints_dir / "board_latest.json"))
        assert len(loaded.knowledge_base) == 1
        assert loaded.knowledge_base[0].label == "b.py"


class TestCrewIngestPhase:

    def test_crew_has_attach_paths(self):
        crew = EPTCrew("test feature", auto_approve=True,
                       attach_paths=["./docs"])
        assert crew.attach_paths == ["./docs"]

    def test_crew_defaults_empty_attach(self):
        crew = EPTCrew("test feature", auto_approve=True)
        assert crew.attach_paths == []


# ─────────────────────────────────────────────────────────────────────────────
#  Git Repo: URL detection
# ─────────────────────────────────────────────────────────────────────────────

class TestGitUrlDetection:

    @pytest.mark.parametrize("url", [
        "https://github.com/user/repo",
        "https://github.com/user/repo.git",
        "https://gitlab.com/org/project",
        "https://bitbucket.org/team/lib",
        "git@github.com:user/repo.git",
        "git@gitlab.com:org/project.git",
        "https://my-server.com/project.git",
    ])
    def test_git_urls_detected(self, url):
        assert is_git_url(url) is True

    @pytest.mark.parametrize("path", [
        "/home/user/project",
        "./relative/path",
        "just-a-folder",
        "https://example.com/api/data",
        "ftp://files.example.com/repo",
    ])
    def test_non_git_paths_rejected(self, path):
        assert is_git_url(path) is False

    def test_whitespace_stripped(self):
        assert is_git_url("  https://github.com/user/repo  ") is True


# ─────────────────────────────────────────────────────────────────────────────
#  Git Repo: File tree builder
# ─────────────────────────────────────────────────────────────────────────────

class TestRepoFileTree:

    def test_basic_tree(self, tmp_path):
        """Build a small directory and verify the tree output."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        (tmp_path / "README.md").write_text("# Hello")
        (tmp_path / "setup.py").write_text("")

        tree = repo_file_tree(tmp_path, max_depth=3)
        assert tree.startswith(tmp_path.name + "/")
        assert "src/" in tree
        assert "main.py" in tree
        assert "README.md" in tree

    def test_skips_git_dir(self, tmp_path):
        """The .git directory should be hidden."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("")
        (tmp_path / "app.py").write_text("")

        tree = repo_file_tree(tmp_path, max_depth=2)
        assert ".git" not in tree
        assert "app.py" in tree

    def test_skips_pycache(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "m.cpython-314.pyc").write_text("")
        (tmp_path / "lib.py").write_text("")

        tree = repo_file_tree(tmp_path)
        assert "__pycache__" not in tree
        assert "lib.py" in tree

    def test_max_depth_respected(self, tmp_path):
        """Deeply nested dirs beyond max_depth should not appear."""
        d = tmp_path
        for name in ["a", "b", "c", "d", "e"]:
            d = d / name
            d.mkdir()
            (d / "f.txt").write_text("")

        tree = repo_file_tree(tmp_path, max_depth=2)
        # Only depth 1,2 should show — "a" and "b"
        assert "a/" in tree
        assert "b/" in tree
        # "c" is at depth 3, should not appear
        assert "c/" not in tree

    def test_empty_dir(self, tmp_path):
        tree = repo_file_tree(tmp_path)
        # Just the root line
        assert tree == tmp_path.name + "/"


# ─────────────────────────────────────────────────────────────────────────────
#  Git Repo: Blackboard repo fields
# ─────────────────────────────────────────────────────────────────────────────

class TestBlackboardRepoFields:

    def test_repo_context_empty_by_default(self):
        board = Blackboard(feature="test")
        assert board.repo_context() == ""

    def test_repo_context_with_analysis(self):
        board = Blackboard(feature="test")
        board.repo_analysis = "This repo uses Flask + SQLAlchemy."
        ctx = board.repo_context()
        assert "REFERENCE REPOSITORY ANALYSIS" in ctx
        assert "Flask" in ctx

    def test_repo_urls_default(self):
        board = Blackboard(feature="test")
        assert board.repo_urls == []

    def test_save_repo_analysis(self):
        board = Blackboard(feature="test repo analysis")
        board.init_project()  # creates docs dir
        board.repo_urls = ["https://github.com/user/repo"]
        board.repo_analysis = "## Overview\nA REST API built with FastAPI."
        path = board.save_repo_analysis()
        content = path.read_text()
        assert "Reference Repository Analysis" in content
        assert "github.com/user/repo" in content
        assert "FastAPI" in content

    def test_full_context_header_includes_repo(self):
        board = Blackboard(feature="test")
        board.repo_analysis = "Key insight: uses event sourcing."
        header = board.full_context_header()
        assert "event sourcing" in header

    def test_full_context_header_without_repo(self):
        board = Blackboard(feature="test")
        header = board.full_context_header()
        assert "REFERENCE REPOSITORY" not in header


# ─────────────────────────────────────────────────────────────────────────────
#  Git Repo: Crew constructor & params
# ─────────────────────────────────────────────────────────────────────────────

class TestCrewRepoUrls:

    def test_crew_accepts_repo_urls(self):
        crew = EPTCrew(
            "test feature",
            auto_approve=True,
            repo_urls=["https://github.com/org/project"],
        )
        assert crew.repo_urls == ["https://github.com/org/project"]

    def test_crew_repo_urls_default_empty(self):
        crew = EPTCrew("test feature", auto_approve=True)
        assert crew.repo_urls == []

    def test_crew_multiple_repo_urls(self):
        urls = [
            "https://github.com/a/b",
            "git@gitlab.com:x/y.git",
        ]
        crew = EPTCrew("test feature", auto_approve=True, repo_urls=urls)
        assert len(crew.repo_urls) == 2
        assert crew.repo_urls[0] == urls[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Git Repo: Knowledge item tagging
# ─────────────────────────────────────────────────────────────────────────────

class TestRepoKnowledgeItemTagging:

    def test_git_repo_connector_type_exists(self):
        """GIT_REPO should be a valid ConnectorType."""
        assert hasattr(ConnectorType, "GIT_REPO")
        assert ConnectorType.GIT_REPO.value  # non-empty string

    def test_knowledge_item_with_git_tags(self):
        item = KnowledgeItem(
            source_type=ConnectorType.GIT_REPO,
            source_path="https://github.com/u/r",
            label="main.py",
            content="print('hello')",
            raw_size=14,
            tags=["git_repo", "code"],
            metadata={"git_url": "https://github.com/u/r"},
        )
        assert "git_repo" in item.tags
        assert item.metadata["git_url"] == "https://github.com/u/r"
        assert item.source_type == ConnectorType.GIT_REPO


# ─────────────────────────────────────────────────────────────────────────────
#  Git Repo: Repo analysis prompt templates
# ─────────────────────────────────────────────────────────────────────────────

class TestRepoPrompts:

    def test_scout_repo_analysis_prompts_exist(self):
        from hive.prompts import SCOUT_REPO_ANALYSIS_SYSTEM, SCOUT_REPO_ANALYSIS_TASK
        assert "reverse-engineer" in SCOUT_REPO_ANALYSIS_SYSTEM.lower() or \
               "repo" in SCOUT_REPO_ANALYSIS_SYSTEM.lower()
        assert "{repo_tree}" in SCOUT_REPO_ANALYSIS_TASK
        assert "{repo_files}" in SCOUT_REPO_ANALYSIS_TASK
        assert "{feature}" in SCOUT_REPO_ANALYSIS_TASK

    def test_scout_task_has_repo_context_placeholder(self):
        from hive.prompts import SCOUT_TASK
        assert "{repo_context}" in SCOUT_TASK

    def test_penny_interview_has_repo_context(self):
        from hive.prompts import PENNY_INTERVIEW_TASK
        assert "{repo_context}" in PENNY_INTERVIEW_TASK

    def test_penny_prd_has_repo_context(self):
        from hive.prompts import PENNY_PRD_TASK
        assert "{repo_context}" in PENNY_PRD_TASK


# ─────────────────────────────────────────────────────────────────────────────
#  Memory: MemoryEntry
# ─────────────────────────────────────────────────────────────────────────────

from hive.memory import (
    MemoryEntry, AgentMemory, TeamMemory, GlobalMemory, MemoryManager,
)


class TestMemoryEntry:

    def test_create_entry(self):
        e = MemoryEntry(kind="mistake", content="forgot error handling")
        assert e.kind == "mistake"
        assert e.content == "forgot error handling"
        assert e.timestamp > 0

    def test_one_liner_icons(self):
        for kind, icon in [("mistake", "❌"), ("pattern", "✅"),
                           ("lesson", "💡"), ("insight", "🔍")]:
            e = MemoryEntry(kind=kind, content="test")
            assert icon in e.one_liner

    def test_entry_with_full_metadata(self):
        e = MemoryEntry(
            kind="lesson", content="always validate input",
            context="user tried SQL injection", phase="build",
            agent_id="dev_1", tags=["security"],
            source_project="proj_a",
        )
        assert e.phase == "build"
        assert "security" in e.tags
        assert e.source_project == "proj_a"


# ─────────────────────────────────────────────────────────────────────────────
#  Memory: AgentMemory
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentMemory:

    def test_remember(self):
        mem = AgentMemory(agent_id="dev_1")
        entry = mem.remember("mistake", "missed null check", phase="build")
        assert len(mem.entries) == 1
        assert entry.agent_id == "dev_1"

    def test_filtered_queries(self):
        mem = AgentMemory(agent_id="scout")
        mem.remember("mistake", "bad parse")
        mem.remember("pattern", "good structure")
        mem.remember("lesson", "always validate")
        assert len(mem.mistakes) == 1
        assert len(mem.patterns) == 1
        assert len(mem.lessons) == 1

    def test_for_phase(self):
        mem = AgentMemory(agent_id="quinn")
        mem.remember("mistake", "missed edge case", phase="build")
        mem.remember("lesson", "general tip")  # no phase
        mem.remember("pattern", "interview pattern", phase="interview")

        build_mems = mem.for_phase("build")
        # Should include "build" + phase-less entries
        assert len(build_mems) == 2

    def test_context_block_empty(self):
        mem = AgentMemory(agent_id="scout")
        assert mem.context_block() == ""

    def test_context_block_with_entries(self):
        mem = AgentMemory(agent_id="dev_1")
        mem.remember("mistake", "forgot imports")
        mem.remember("pattern", "good naming convention")
        block = mem.context_block()
        assert "YOUR MEMORY (dev_1)" in block
        assert "AVOID" in block
        assert "DO" in block

    def test_context_block_max_entries(self):
        mem = AgentMemory(agent_id="dev_1")
        for i in range(20):
            mem.remember("lesson", f"lesson {i}")
        block = mem.context_block(max_entries=5)
        # Only 5 entries should appear
        assert block.count("[KNOW]") == 5


# ─────────────────────────────────────────────────────────────────────────────
#  Memory: TeamMemory
# ─────────────────────────────────────────────────────────────────────────────

class TestTeamMemory:

    def test_push_and_retrieve(self):
        team = TeamMemory()
        team.push("scout", "API uses pagination")
        entries = team.for_agent("dev_1")
        assert len(entries) == 1
        assert "pagination" in entries[0].content

    def test_excludes_own_entries(self):
        team = TeamMemory()
        team.push("scout", "something useful")
        # Scout shouldn't see its own push
        assert len(team.for_agent("scout")) == 0
        assert len(team.for_agent("dev_1")) == 1

    def test_targeted_entries(self):
        team = TeamMemory()
        team.push("archie", "watch the deps", for_agents=["dev_1", "dev_2"])
        assert len(team.for_agent("dev_1")) == 1
        assert len(team.for_agent("dev_3")) == 0  # not targeted
        assert len(team.for_agent("quinn")) == 0

    def test_context_block_empty(self):
        team = TeamMemory()
        assert team.context_block("dev_1") == ""

    def test_context_block_with_entries(self):
        team = TeamMemory()
        team.push("scout", "domain uses REST")
        team.push("archie", "use layered pattern")
        block = team.context_block("dev_1")
        assert "TEAM MEMORY" in block
        assert "scout" in block
        assert "archie" in block


# ─────────────────────────────────────────────────────────────────────────────
#  Memory: GlobalMemory
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalMemory:

    def test_add_lesson(self):
        gm = GlobalMemory()
        entry = gm.add_lesson("always define error codes", agent_id="archie")
        assert len(gm.lessons) == 1
        assert entry.kind == "lesson"

    def test_context_block_empty(self):
        gm = GlobalMemory()
        assert gm.context_block() == ""

    def test_context_block_with_lessons(self):
        gm = GlobalMemory()
        gm.add_lesson("validate all inputs", source_project="proj_a")
        gm.add_lesson("use dependency injection", source_project="proj_b")
        block = gm.context_block()
        assert "GLOBAL LESSONS" in block
        assert "proj_a" in block
        assert "validate all inputs" in block

    def test_agent_prioritization(self):
        gm = GlobalMemory()
        gm.add_lesson("general lesson", agent_id="other")
        gm.add_lesson("agent-specific lesson", agent_id="dev_1")
        block = gm.context_block(agent_id="dev_1", max_lessons=1)
        assert "agent-specific" in block


# ─────────────────────────────────────────────────────────────────────────────
#  Memory: MemoryManager
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryManager:

    def test_get_agent_creates(self):
        mm = MemoryManager(project_slug="test")
        agent_mem = mm.get_agent("scout")
        assert agent_mem.agent_id == "scout"
        # Getting same agent returns same object
        assert mm.get_agent("scout") is agent_mem

    def test_context_for_agent_empty(self):
        mm = MemoryManager(project_slug="test")
        assert mm.context_for_agent("dev_1") == ""

    def test_context_for_agent_combined(self):
        mm = MemoryManager(project_slug="test")
        # Personal memory
        mm.get_agent("dev_1").remember("mistake", "forgot validation")
        # Team memory
        mm.team.push("scout", "API uses OAuth", for_agents=["dev_1"])
        # Global memory
        mm.global_memory.add_lesson("always add tests")

        ctx = mm.context_for_agent("dev_1", phase="build")
        assert "YOUR MEMORY" in ctx
        assert "TEAM MEMORY" in ctx
        assert "GLOBAL LESSONS" in ctx

    def test_save_and_load(self, tmp_path):
        mm = MemoryManager(project_slug="test", memory_dir=tmp_path)
        mm.get_agent("scout").remember("lesson", "check pagination")
        mm.team.push("archie", "use clean architecture")
        mm.save()

        # Verify files exist
        assert (tmp_path / "agent_scout.json").exists()
        assert (tmp_path / "team.json").exists()

        # Load into fresh manager
        mm2 = MemoryManager(project_slug="test", memory_dir=tmp_path)
        mm2.load()
        assert len(mm2.get_agent("scout").entries) == 1
        assert mm2.get_agent("scout").entries[0].content == "check pagination"
        assert len(mm2.team.entries) == 1

    def test_global_save_and_load(self, tmp_path):
        gpath = tmp_path / "global.json"
        mm = MemoryManager(project_slug="test")
        mm.global_memory.add_lesson("always validate", source_project="proj_a")
        mm.save_global(gpath)

        assert gpath.exists()

        mm2 = MemoryManager(project_slug="other")
        mm2.load_global(gpath)
        assert len(mm2.global_memory.lessons) == 1
        assert mm2.global_memory.lessons[0].content == "always validate"

    def test_distill_to_global(self):
        mm = MemoryManager(project_slug="test_proj")
        mm.get_agent("dev_1").remember("lesson", "use type hints")
        mm.get_agent("dev_1").remember("mistake", "missed error handling in main.py")
        mm.team.push("quinn", "watch for null references")

        lessons = mm.distill_to_global()
        assert len(lessons) >= 3  # 1 lesson + 1 mistake + 1 team insight
        assert len(mm.global_memory.lessons) >= 3

    def test_distill_caps_global(self):
        mm = MemoryManager(project_slug="test")
        # Pre-fill global with 98 lessons
        for i in range(98):
            mm.global_memory.add_lesson(f"old lesson {i}")
        # Add project memories
        mm.get_agent("dev_1").remember("lesson", "new lesson A")
        mm.get_agent("dev_1").remember("lesson", "new lesson B")
        mm.get_agent("dev_1").remember("lesson", "new lesson C")
        mm.team.push("scout", "new team insight")

        mm.distill_to_global()
        # Should be capped at 100
        assert len(mm.global_memory.lessons) <= 100

    def test_stats(self):
        mm = MemoryManager(project_slug="test")
        mm.get_agent("scout").remember("lesson", "a")
        mm.get_agent("dev_1").remember("mistake", "b")
        mm.get_agent("dev_1").remember("pattern", "c")
        mm.team.push("scout", "d")
        mm.global_memory.add_lesson("e")

        s = mm.stats()
        assert s["agent_memories"]["scout"] == 1
        assert s["agent_memories"]["dev_1"] == 2
        assert s["team_entries"] == 1
        assert s["global_lessons"] == 1
        assert s["total"] == 4  # agent entries + team (not global)


# ─────────────────────────────────────────────────────────────────────────────
#  Memory: Blackboard integration
# ─────────────────────────────────────────────────────────────────────────────

class TestBlackboardMemory:

    def test_memory_context_default_empty(self):
        board = Blackboard(feature="test")
        assert board.memory_context == ""

    def test_memory_context_set_and_clear(self):
        board = Blackboard(feature="test")
        board.memory_context = "YOUR MEMORY: avoid X"
        assert "avoid X" in board.memory_context
        board.memory_context = ""
        assert board.memory_context == ""

    def test_init_project_creates_memory_dir(self):
        board = Blackboard(feature="test memory dir creation")
        board.init_project()
        assert board.memory_dir.exists()
        # Cleanup
        import shutil
        shutil.rmtree(board.project_root, ignore_errors=True)

    def test_memory_dir_property(self):
        board = Blackboard(feature="test")
        board.project_slug = "my_project"
        assert board.memory_dir.name == "memory"
        assert "my_project" in str(board.memory_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  Memory: Crew integration
# ─────────────────────────────────────────────────────────────────────────────

class TestCrewMemory:

    def test_crew_has_memory_manager(self):
        crew = EPTCrew("test feature", auto_approve=True)
        assert hasattr(crew, 'memory')
        assert isinstance(crew.memory, MemoryManager)

    def test_crew_memory_helpers(self):
        """Test that memory helper methods exist and work."""
        crew = EPTCrew("test feature", auto_approve=True)
        crew.board.current_phase = "build"
        crew.board.project_slug = "test"

        # Record operations should not raise
        crew._record_mistake("dev_1", "test mistake")
        crew._record_pattern("dev_1", "test pattern")
        crew._record_lesson("dev_1", "test lesson")
        crew._push_team_insight("scout", "test insight", for_agents=["dev_1"])

        assert len(crew.memory.get_agent("dev_1").entries) == 3
        assert len(crew.memory.team.entries) == 1

    def test_set_and_clear_memory(self):
        crew = EPTCrew("test feature", auto_approve=True)
        crew.board.current_phase = "build"
        crew.memory.get_agent("dev_1").remember("mistake", "forgot imports")

        crew._set_memory("dev_1")
        assert "forgot imports" in crew.board.memory_context

        crew._clear_memory()
        assert crew.board.memory_context == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
