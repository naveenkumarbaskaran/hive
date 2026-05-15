"""
Hardening Test Suite — Tests production-grade utilities and safety mechanisms.

Covers:
  - Atomic file writes (crash safety)
  - Path sanitization (traversal prevention)
  - Token estimation & budget management
  - File locking
  - LLM output validation
  - Exponential backoff with jitter
  - Cleanup registry
  - Checkpoint schema validation
  - Event capping
  - Context window budgeting
  - Code fence cleaning

Run: python -m pytest test_hardening.py -v
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from hive.hardening import (
    CleanupRegistry,
    atomic_write,
    backoff_wait,
    budget_context,
    clean_code_fences,
    estimate_tokens,
    file_lock,
    sanitize_filename,
    setup_logging,
    truncate_to_budget,
    validate_checkpoint_data,
    validate_code_output,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Atomic writes
# ─────────────────────────────────────────────────────────────────────────────

class TestAtomicWrite:
    """Test atomic_write for crash-safe file operations."""

    def test_basic_write(self, tmp_path):
        """Writes content correctly."""
        p = tmp_path / "test.txt"
        atomic_write(p, "hello world")
        assert p.read_text() == "hello world"

    def test_overwrite(self, tmp_path):
        """Overwrites existing content atomically."""
        p = tmp_path / "test.txt"
        p.write_text("old")
        atomic_write(p, "new")
        assert p.read_text() == "new"

    def test_creates_parent_dirs(self, tmp_path):
        """Creates missing parent directories."""
        p = tmp_path / "a" / "b" / "c" / "test.txt"
        atomic_write(p, "deep")
        assert p.read_text() == "deep"

    def test_no_temp_file_on_success(self, tmp_path):
        """No .tmp files left after successful write."""
        p = tmp_path / "test.txt"
        atomic_write(p, "content")
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_handles_unicode(self, tmp_path):
        """Correctly writes Unicode content."""
        p = tmp_path / "test.txt"
        content = "Hello 🌍 — café résumé 日本語"
        atomic_write(p, content)
        assert p.read_text() == content

    def test_empty_content(self, tmp_path):
        """Writes empty file without error."""
        p = tmp_path / "test.txt"
        atomic_write(p, "")
        assert p.read_text() == ""

    def test_large_content(self, tmp_path):
        """Handles large files correctly."""
        p = tmp_path / "large.txt"
        content = "x" * 1_000_000
        atomic_write(p, content)
        assert p.read_text() == content

    def test_accepts_string_path(self, tmp_path):
        """Works with string paths, not just Path objects."""
        p = str(tmp_path / "test.txt")
        atomic_write(p, "string path")
        assert Path(p).read_text() == "string path"


# ─────────────────────────────────────────────────────────────────────────────
#  File locking
# ─────────────────────────────────────────────────────────────────────────────

class TestFileLock:
    """Test advisory file locking."""

    def test_basic_lock_unlock(self, tmp_path):
        """Lock can be acquired and released."""
        p = tmp_path / "locktest.json"
        p.write_text("{}")
        with file_lock(p):
            # We can do work inside the lock
            p.write_text('{"locked": true}')
        assert json.loads(p.read_text()) == {"locked": True}

    def test_lock_creates_lock_file(self, tmp_path):
        """Lock file is created adjacent to target."""
        p = tmp_path / "data.json"
        p.write_text("{}")
        with file_lock(p):
            lock_file = tmp_path / "data.json.lock"
            assert lock_file.exists()

    def test_concurrent_lock(self, tmp_path):
        """Two threads can't write simultaneously (serial execution)."""
        p = tmp_path / "counter.txt"
        p.write_text("0")
        results = []

        def increment(thread_id):
            with file_lock(p):
                val = int(p.read_text())
                time.sleep(0.01)  # simulate work
                p.write_text(str(val + 1))
                results.append(thread_id)

        threads = [threading.Thread(target=increment, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert int(p.read_text()) == 5
        assert len(results) == 5


# ─────────────────────────────────────────────────────────────────────────────
#  Path sanitization
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizeFilename:
    """Test path traversal prevention."""

    def test_normal_path(self):
        assert sanitize_filename("src/main.py") == "src/main.py"

    def test_dot_dot_traversal(self):
        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert result == "etc/passwd"

    def test_absolute_path(self):
        result = sanitize_filename("/etc/passwd")
        assert not result.startswith("/")

    def test_empty_string(self):
        assert sanitize_filename("") == "unnamed_file"

    def test_just_dots(self):
        assert sanitize_filename("../..") == "unnamed_file"

    def test_null_bytes(self):
        result = sanitize_filename("test\x00.py")
        assert "\x00" not in result

    def test_control_characters(self):
        result = sanitize_filename("test\x01\x02\x03.py")
        assert "\x01" not in result

    def test_windows_backslash(self):
        result = sanitize_filename("src\\main.py")
        assert result == "src/main.py"

    def test_mixed_traversal(self):
        result = sanitize_filename("../src/../../etc/passwd")
        assert ".." not in result
        assert result == "src/etc/passwd"

    def test_dotfiles_preserved(self):
        """Dotfiles like .gitignore should be kept."""
        assert sanitize_filename(".gitignore") == ".gitignore"

    def test_nested_safe_path(self):
        assert sanitize_filename("src/utils/helpers.py") == "src/utils/helpers.py"


# ─────────────────────────────────────────────────────────────────────────────
#  Token estimation & budget management
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenEstimation:
    """Test token counting heuristics."""

    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        tokens = estimate_tokens("hello world")
        assert tokens >= 1

    def test_proportional(self):
        """Longer text = more tokens."""
        short = estimate_tokens("short")
        long = estimate_tokens("a" * 1000)
        assert long > short

    def test_code_estimation(self):
        code = "def hello():\n    print('hello world')\n"
        tokens = estimate_tokens(code)
        assert 5 < tokens < 100

    def test_truncation_no_op(self):
        """Short text should not be truncated."""
        text = "short text"
        result = truncate_to_budget(text, max_tokens=10000)
        assert result == text

    def test_truncation_applied(self):
        """Long text should be truncated."""
        text = "x" * 1_000_000
        result = truncate_to_budget(text, max_tokens=1000)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_budget_context_respects_limit(self):
        """budget_context stays within token budget."""
        sections = [
            ("high", "A" * 3000, 1),
            ("medium", "B" * 3000, 2),
            ("low", "C" * 3000, 3),
        ]
        result = budget_context(sections, max_tokens=2000)
        assert estimate_tokens(result) <= 2500  # some slack for marker

    def test_budget_context_priority_order(self):
        """Higher priority sections included first."""
        sections = [
            ("low", "LOW_CONTENT", 10),
            ("high", "HIGH_CONTENT", 1),
        ]
        result = budget_context(sections, max_tokens=100000)
        # Both should be present since budget is large
        assert "HIGH_CONTENT" in result
        assert "LOW_CONTENT" in result

    def test_budget_context_empty_sections(self):
        """Empty sections are skipped."""
        sections = [
            ("a", "", 1),
            ("b", "content", 2),
        ]
        result = budget_context(sections, max_tokens=10000)
        assert result == "content"


# ─────────────────────────────────────────────────────────────────────────────
#  LLM output validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeValidation:
    """Test LLM output validation for code generation."""

    def test_valid_code(self):
        code = "def hello():\n    print('world')"
        result = validate_code_output(code, "hello.py")
        assert result == code

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_code_output("", "test.py")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_code_output("   \n\n  ", "test.py")

    def test_refusal_short_raises(self):
        with pytest.raises(ValueError, match="refused"):
            validate_code_output("I cannot generate that code.", "test.py")

    def test_refusal_long_not_raised(self):
        """Long output with refusal words is likely real code, not a refusal."""
        code = "# I cannot believe how complex this is\n" + "x = 1\n" * 50
        result = validate_code_output(code, "test.py")
        assert "x = 1" in result

    def test_code_fences_stripped(self):
        raw = "```python\ndef hello():\n    pass\n```"
        result = validate_code_output(raw, "hello.py")
        assert "```" not in result
        assert "def hello" in result


class TestCleanCodeFences:
    """Test markdown code fence stripping."""

    def test_no_fences(self):
        assert clean_code_fences("plain code") == "plain code"

    def test_simple_fence(self):
        code = "```python\ndef f():\n    pass\n```"
        result = clean_code_fences(code)
        assert "```" not in result
        assert "def f" in result

    def test_multiple_blocks(self):
        """If multiple blocks, returns the largest."""
        code = "```js\nconsole.log('hi')\n```\n\n```python\ndef f():\n    x = 1\n    y = 2\n    return x + y\n```"
        result = clean_code_fences(code)
        assert "def f" in result

    def test_fence_with_preamble(self):
        code = "Here is the code:\n\n```python\ndef f():\n    pass\n```"
        result = clean_code_fences(code)
        assert "def f" in result
        assert "Here is" not in result

    def test_no_language_tag(self):
        code = "```\ndef f():\n    pass\n```"
        result = clean_code_fences(code)
        assert "def f" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Exponential backoff
# ─────────────────────────────────────────────────────────────────────────────

class TestBackoffWait:
    """Test exponential backoff with jitter."""

    def test_returns_float(self):
        result = backoff_wait(1)
        assert isinstance(result, float)

    def test_grows_with_attempts(self):
        """Average backoff grows with more attempts."""
        samples_1 = [backoff_wait(1) for _ in range(100)]
        samples_5 = [backoff_wait(5) for _ in range(100)]
        avg_1 = sum(samples_1) / len(samples_1)
        avg_5 = sum(samples_5) / len(samples_5)
        assert avg_5 > avg_1

    def test_respects_max(self):
        """Never exceeds max_wait."""
        for _ in range(100):
            result = backoff_wait(100, max_wait=10.0)
            assert result <= 10.0

    def test_has_jitter(self):
        """Different calls produce different values (jitter)."""
        results = {backoff_wait(3) for _ in range(10)}
        assert len(results) > 1  # not all the same

    def test_minimum_wait(self):
        """Always waits at least 0.5s."""
        for _ in range(50):
            assert backoff_wait(1) >= 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  Cleanup registry
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanupRegistry:
    """Test temp directory cleanup management."""

    def test_register_and_cleanup(self, tmp_path):
        registry = CleanupRegistry()
        d = tmp_path / "temp_repo"
        d.mkdir()
        (d / "file.txt").write_text("test")
        registry.register(d)

        assert d in registry.registered
        registry._cleanup()
        assert not d.exists()

    def test_cleanup_missing_dir(self, tmp_path):
        """Doesn't raise if registered dir already deleted."""
        registry = CleanupRegistry()
        d = tmp_path / "already_gone"
        registry.register(d)
        # Should not raise
        registry._cleanup()

    def test_multiple_registrations(self, tmp_path):
        registry = CleanupRegistry()
        dirs = []
        for i in range(3):
            d = tmp_path / f"temp_{i}"
            d.mkdir()
            registry.register(d)
            dirs.append(d)

        assert len(registry.registered) == 3
        registry._cleanup()
        for d in dirs:
            assert not d.exists()


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointValidation:
    """Test checkpoint schema validation."""

    def test_valid_data(self):
        data = {"feature": "Test feature", "current_phase": "build"}
        result = validate_checkpoint_data(data)
        assert result["feature"] == "Test feature"

    def test_missing_feature_raises(self):
        with pytest.raises(ValueError, match="feature"):
            validate_checkpoint_data({"current_phase": "build"})

    def test_not_a_dict_raises(self):
        with pytest.raises(ValueError, match="not a dict"):
            validate_checkpoint_data("not a dict")

    def test_defaults_added(self):
        data = {"feature": "Test"}
        result = validate_checkpoint_data(data)
        assert result["project_slug"] == ""
        assert result["file_plan"] == []
        assert result["dep_graph"] == {}
        assert result["dev_count"] == 1

    def test_type_coercion(self):
        """Non-string string fields get coerced."""
        data = {"feature": "Test", "prd": 12345}
        result = validate_checkpoint_data(data)
        assert result["prd"] == "12345"

    def test_invalid_list_reset(self):
        """Non-list list fields get reset to empty list."""
        data = {"feature": "Test", "file_plan": "not a list"}
        result = validate_checkpoint_data(data)
        assert result["file_plan"] == []


# ─────────────────────────────────────────────────────────────────────────────
#  Event capping (integration with state.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestEventCapping:
    """Test that events list is capped to prevent OOM."""

    def test_events_capped(self):
        from hive.state import _MAX_EVENTS, Blackboard, EventType
        board = Blackboard(feature="test")
        # Emit more than max
        for i in range(_MAX_EVENTS + 500):
            board.emit(EventType.THINKING, "test", f"event {i}")
        assert len(board.events) == _MAX_EVENTS

    def test_cap_preserves_recent(self):
        from hive.state import _MAX_EVENTS, Blackboard, EventType
        board = Blackboard(feature="test")
        for i in range(_MAX_EVENTS + 100):
            board.emit(EventType.THINKING, "test", f"event {i}")
        # The last event should be the most recent one
        assert board.events[-1].content == f"event {_MAX_EVENTS + 99}"


# ─────────────────────────────────────────────────────────────────────────────
#  Context budgeting (integration with state.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBudgeting:
    """Test that full_context_header respects token budget."""

    def test_context_header_produces_string(self):
        from hive.state import Blackboard, ResearchContext
        board = Blackboard(feature="test")
        board.research = ResearchContext(domain="test")
        board.prd = "PRD content"
        board.architecture = "Arch content"
        board.contract = "Contract content"
        result = board.full_context_header()
        assert isinstance(result, str)
        assert "PRD content" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Atomic writes in state.py (integration test)
# ─────────────────────────────────────────────────────────────────────────────

class TestAtomicWriteIntegration:
    """Test that state.py persistence uses atomic writes."""

    def test_save_checkpoint_atomic(self, tmp_path):
        """save_checkpoint should create files atomically (no .tmp leftovers)."""
        from hive.state import PROJECTS_DIR, Blackboard, save_checkpoint
        with patch.object(
            type(PROJECTS_DIR), '__fspath__',
            return_value=str(tmp_path),
        ):
            pass  # This is complex to mock; test the function directly instead

        board = Blackboard(feature="Test Atomic")
        board.project_slug = "test_atomic"
        # Override the project dir
        board_root = tmp_path / "test_atomic"
        (board_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (board_root / "docs").mkdir(parents=True, exist_ok=True)

        # Monkey-patch the board's PROJECTS_DIR for this test
        import hive.state as state_mod
        old_dir = state_mod.PROJECTS_DIR
        state_mod.PROJECTS_DIR = tmp_path
        try:
            path = save_checkpoint(board)
            assert path.exists()
            latest = board.checkpoints_dir / "board_latest.json"
            assert latest.exists()

            # Verify valid JSON
            data = json.loads(path.read_text())
            assert data["feature"] == "Test Atomic"
            assert "_schema_version" in data

            # No temp files
            tmp_files = list(board.checkpoints_dir.glob("*.tmp"))
            assert tmp_files == []
        finally:
            state_mod.PROJECTS_DIR = old_dir


# ─────────────────────────────────────────────────────────────────────────────
#  Filename sanitization in state.py (integration test)
# ─────────────────────────────────────────────────────────────────────────────

class TestFilenameSanitizationIntegration:
    """Test that save_source_file sanitizes LLM-generated filenames."""

    def test_traversal_blocked(self, tmp_path):
        import hive.state as state_mod
        from hive.state import Blackboard, FileEntry
        old_dir = state_mod.PROJECTS_DIR
        state_mod.PROJECTS_DIR = tmp_path
        try:
            board = Blackboard(feature="Test Sanitize")
            board.project_slug = "test_sanitize"
            board.init_project()

            entry = FileEntry(name="../../etc/passwd", code="malicious content")
            path = board.save_source_file(entry)

            # Should NOT escape the src directory — the path must stay within project
            assert str(path).startswith(str(tmp_path))
            # The '..' traversal components are stripped
            assert ".." not in str(path)
            assert path.read_text() == "malicious content"
        finally:
            state_mod.PROJECTS_DIR = old_dir


# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

class TestLoggingSetup:
    """Test structured logging configuration."""

    def test_setup_does_not_raise(self):
        setup_logging("WARNING")

    def test_setup_with_env_var(self):
        with patch.dict(os.environ, {"HIVE_LOG_LEVEL": "DEBUG"}):
            setup_logging()

    def test_setup_invalid_level(self):
        """Invalid level falls back to WARNING."""
        setup_logging("NONEXISTENT")


# ─────────────────────────────────────────────────────────────────────────────
#  LLM client backoff (unit test)
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMClientBackoff:
    """Test that LLM client uses exponential backoff."""

    def test_backoff_imported(self):
        """LLM client has its own backoff function."""
        from hive.llm_client import _backoff_wait
        assert callable(_backoff_wait)

    def test_default_retries_is_five(self):
        """Default retries should be 5 (not 3)."""
        import inspect

        from hive.llm_client import LLMClient
        sig = inspect.signature(LLMClient.chat)
        default = sig.parameters["retries"].default
        assert default == 5


# ─────────────────────────────────────────────────────────────────────────────
#  Disk-space pre-check tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckDiskSpace:
    """Validate the check_disk_space utility."""

    def test_returns_positive_free_mb(self, tmp_path):
        """Running on a real filesystem should report positive free space."""
        from hive.hardening import check_disk_space
        free = check_disk_space(tmp_path, min_mb=1)
        assert isinstance(free, int) and free > 0

    def test_raises_when_below_threshold(self, tmp_path):
        """Should raise DiskSpaceError when free < min_mb."""
        from hive.hardening import DiskSpaceError, check_disk_space
        # Request an absurdly large threshold so any disk will fail
        with pytest.raises(DiskSpaceError, match="Insufficient disk space"):
            check_disk_space(tmp_path, min_mb=999_999_999)

    def test_walks_up_to_existing_parent(self, tmp_path):
        """If path doesn't exist, the check uses the nearest existing parent."""
        from hive.hardening import check_disk_space
        deep = tmp_path / "a" / "b" / "c" / "d.json"
        free = check_disk_space(deep, min_mb=1)
        assert free > 0

    def test_env_var_default(self, monkeypatch, tmp_path):
        """MIN_DISK_MB should be overridable via HIVE_MIN_DISK_MB env var."""
        # Importing again inside a fresh monkeypatch scope
        monkeypatch.setenv("HIVE_MIN_DISK_MB", "999999999")
        # Need to reload the module so the module-level constant picks it up
        import importlib

        import hive.hardening
        importlib.reload(hive.hardening)
        from hive.hardening import DiskSpaceError, check_disk_space
        with pytest.raises(DiskSpaceError):
            check_disk_space(tmp_path)
        # Restore
        monkeypatch.delenv("HIVE_MIN_DISK_MB")
        importlib.reload(hive.hardening)

    def test_fail_open_on_os_error(self, tmp_path, monkeypatch):
        """If shutil.disk_usage raises, check_disk_space should return -1."""
        import shutil as _shutil

        from hive.hardening import check_disk_space
        monkeypatch.setattr(_shutil, "disk_usage", lambda _: (_ for _ in ()).throw(OSError("mocked")))
        result = check_disk_space(tmp_path, min_mb=1)
        assert result == -1

    def test_disk_space_error_is_os_error(self):
        """DiskSpaceError should be a subclass of OSError."""
        from hive.hardening import DiskSpaceError
        assert issubclass(DiskSpaceError, OSError)
