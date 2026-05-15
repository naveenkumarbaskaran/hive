"""
EPT Hardening — Production-grade utilities for data integrity and safety.

Provides:
  - atomic_write()       — crash-safe file writes (temp → rename)
  - file_lock()          — advisory file locking for concurrent access
  - sanitize_filename()  — prevent path traversal from LLM output
  - estimate_tokens()    — rough token counting for budget management
  - truncate_to_budget() — smart truncation respecting token limits
  - validate_code_output() — reject LLM garbage/refusals
  - CleanupRegistry      — track temp dirs for cleanup on exit
"""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import random
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("hive")


# ─────────────────────────────────────────────────────────────────────────────
#  Atomic file writes
# ─────────────────────────────────────────────────────────────────────────────

def atomic_write(path: Path | str, content: str) -> None:
    """Write content to a file atomically using temp-file + rename.

    On POSIX, os.replace() is atomic. This ensures that a crash mid-write
    never leaves a corrupt file — either the old content or the new content
    is present, never a partial file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=".hive_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        # Clean up the temp file on any failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  File locking — advisory locks for concurrent access safety
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def file_lock(path: Path | str, *, timeout: float = 30.0):
    """Advisory file lock using fcntl.flock.

    Uses a .lock file adjacent to the target. Blocks up to `timeout`
    seconds, then raises TimeoutError. Automatically released on exit.
    """
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fp = open(lock_path, "w")
    try:
        # Try non-blocking first; if that fails, poll up to timeout
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            deadline = time.time() + timeout
            acquired = False
            while time.time() < deadline:
                try:
                    fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    time.sleep(0.1)
            if not acquired:
                fp.close()
                raise TimeoutError(
                    f"Could not acquire lock on {lock_path} within {timeout}s"
                ) from None
        yield
    finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        fp.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Path sanitisation — prevent LLM-generated path traversal
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    """Remove path traversal and dangerous characters from LLM-generated filenames.

    Ensures the result is a safe relative path within the project directory.
    >>> sanitize_filename("../../etc/passwd")
    'etc/passwd'
    >>> sanitize_filename("src/main.py")
    'src/main.py'
    >>> sanitize_filename("")
    'unnamed_file'
    """
    # Normalise separators
    name = name.replace("\\", "/")
    # Remove null bytes and control characters
    name = re.sub(r"[\x00-\x1f]", "", name)
    # Split and drop any traversal components
    parts = name.split("/")
    safe_parts: list[str] = []
    for p in parts:
        p = p.strip()
        if p in ("", ".", ".."):
            continue
        # Strip leading dots that could be hidden files (keep .gitignore etc.)
        safe_parts.append(p)
    result = "/".join(safe_parts)
    if not result:
        return "unnamed_file"
    # Ensure no absolute path
    return result.lstrip("/")


# ─────────────────────────────────────────────────────────────────────────────
#  Token estimation & budget management
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/code.

    This is intentionally conservative (over-estimates slightly)
    to avoid blowing past model context windows.
    """
    if not text:
        return 0
    return max(1, len(text) // 3)


# Maximum input tokens for the model (leaving room for output)
DEFAULT_MAX_INPUT_TOKENS = 150_000  # conservative for Claude 200K context


def truncate_to_budget(
    text: str,
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    keep_head_ratio: float = 0.7,
) -> str:
    """Truncate text to fit within a token budget.

    Keeps the first `keep_head_ratio` of the budget from the beginning
    and the rest from the end, with a truncation marker in the middle.
    """
    current = estimate_tokens(text)
    if current <= max_tokens:
        return text

    # Convert token budget to approximate char budget
    char_budget = max_tokens * 3
    head_chars = int(char_budget * keep_head_ratio)
    tail_chars = char_budget - head_chars - 100  # room for marker

    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars > 0 else ""
    marker = f"\n\n... (truncated {current - max_tokens:,} tokens to fit context window) ...\n\n"
    return head + marker + tail


def budget_context(
    sections: list[tuple[str, str, int]],
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> str:
    """Build a context string from prioritized sections within a token budget.

    Args:
        sections: list of (label, content, priority) tuples.
                  Lower priority number = higher importance (kept first).
        max_tokens: total budget for all sections combined.

    Returns:
        Combined context string, with lower-priority sections truncated
        or dropped to fit.
    """
    # Sort by priority (lower = more important)
    sorted_sections = sorted(sections, key=lambda s: s[2])

    parts: list[str] = []
    remaining = max_tokens

    for label, content, _prio in sorted_sections:
        if not content or not content.strip():
            continue
        tokens = estimate_tokens(content)
        if tokens <= remaining:
            parts.append(content)
            remaining -= tokens
        elif remaining > 500:
            # Truncate to fit remaining budget
            truncated = truncate_to_budget(content, max_tokens=remaining)
            parts.append(truncated)
            remaining = 0
        # else: skip this section entirely

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  LLM output validation
# ─────────────────────────────────────────────────────────────────────────────

_REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm sorry", "as an ai",
    "i'm not able", "i am unable", "i apologize",
]


def validate_code_output(text: str, filename: str) -> str:
    """Validate that LLM output looks like actual code, not prose/refusal.

    Returns the cleaned code string if valid, raises ValueError if not.
    """
    cleaned = clean_code_fences(text)

    if not cleaned.strip():
        raise ValueError(f"LLM returned empty output for {filename}")

    lower = cleaned.lower()[:300]
    # Short responses that look like refusals
    if len(cleaned) < 200 and any(p in lower for p in _REFUSAL_PATTERNS):
        raise ValueError(f"LLM refused to generate code for {filename}: {cleaned[:100]}")

    return cleaned


def clean_code_fences(code: str) -> str:
    """Strip markdown code fences from LLM output.

    Handles:
    - Single code block with language tag
    - Multiple code blocks (takes the largest one)
    - Preamble text before the code block
    """
    code = code.strip()

    # Try to find code blocks
    blocks = re.findall(r"```(?:\w*)\s*\n(.*?)```", code, re.DOTALL)
    if blocks:
        # Return the largest block (likely the main code)
        return max(blocks, key=len).strip()

    # Simple fence strip (start/end)
    if code.startswith("```"):
        first_nl = code.find("\n")
        code = code[first_nl + 1:]
    if code.endswith("```"):
        code = code[:-3].rstrip()

    return code


# ─────────────────────────────────────────────────────────────────────────────
#  Exponential backoff with jitter
# ─────────────────────────────────────────────────────────────────────────────

def backoff_wait(attempt: int, base: float = 1.0, max_wait: float = 60.0) -> float:
    """Compute exponential backoff with full jitter.

    attempt starts at 1:
      attempt 1 → uniform(0, 2)
      attempt 2 → uniform(0, 4)
      attempt 3 → uniform(0, 8)
      ...
      capped at max_wait
    """
    ceiling = min(base * (2 ** attempt), max_wait)
    return random.uniform(0.5, ceiling)


# ─────────────────────────────────────────────────────────────────────────────
#  Temp directory cleanup registry
# ─────────────────────────────────────────────────────────────────────────────

class CleanupRegistry:
    """Track temporary directories and clean them up on exit.

    Usage:
        cleanup = CleanupRegistry()
        cleanup.register(some_temp_path)
        # At process exit, all registered paths are deleted
    """

    def __init__(self) -> None:
        self._paths: list[Path] = []
        atexit.register(self._cleanup)

    def register(self, path: Path | str) -> Path:
        """Register a temp path for cleanup. Returns the path."""
        p = Path(path)
        self._paths.append(p)
        return p

    def _cleanup(self) -> None:
        """Remove all registered temp directories."""
        for p in self._paths:
            try:
                if p.exists():
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink(missing_ok=True)
            except Exception:
                pass  # best-effort cleanup

    @property
    def registered(self) -> list[Path]:
        return list(self._paths)


# Module-level singleton for temp cleanup
_cleanup_registry = CleanupRegistry()


def register_temp_path(path: Path | str) -> Path:
    """Register a temp path for automatic cleanup on exit."""
    return _cleanup_registry.register(path)


def get_cleanup_registry() -> CleanupRegistry:
    """Get the module-level cleanup registry."""
    return _cleanup_registry


# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str | None = None) -> None:
    """Configure structured logging for EPT.

    Level is read from HIVE_LOG_LEVEL env var, or passed directly.
    Logs go to stderr so they don't mix with UI output on stdout.
    """
    import sys

    level_str = (level or os.environ.get("HIVE_LOG_LEVEL", "WARNING")).upper()
    numeric_level = getattr(logging, level_str, logging.WARNING)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    root_logger = logging.getLogger("hive")
    root_logger.setLevel(numeric_level)
    if not root_logger.handlers:
        root_logger.addHandler(handler)


# ─────────────────────────────────────────────────────────────────────────────
#  Disk space pre-check
# ─────────────────────────────────────────────────────────────────────────────

MIN_DISK_MB = int(os.environ.get("HIVE_MIN_DISK_MB", "50"))


class DiskSpaceError(OSError):
    """Raised when free disk space is below the required minimum."""


def check_disk_space(path: Path | str, min_mb: int | None = None) -> int:
    """Check that *path*'s filesystem has at least *min_mb* MB free.

    Parameters
    ----------
    path : Path or str
        Any file or directory on the target filesystem.  If it doesn't exist
        yet, its nearest existing parent is used.
    min_mb : int, optional
        Minimum free megabytes required.  Defaults to ``MIN_DISK_MB`` (env
        ``HIVE_MIN_DISK_MB``, default 50).

    Returns
    -------
    int
        Available megabytes on the filesystem.

    Raises
    ------
    DiskSpaceError
        If free space is below the threshold.
    """
    if min_mb is None:
        min_mb = MIN_DISK_MB

    target = Path(path)
    # Walk up to find an existing directory for disk_usage()
    check_path = target if target.is_dir() else target.parent
    while not check_path.exists():
        check_path = check_path.parent

    try:
        usage = shutil.disk_usage(check_path)
    except OSError as exc:
        logger.warning("Could not check disk space on %s: %s", check_path, exc)
        return -1  # fail-open: don't block saves if we can't stat FS

    free_mb = usage.free // (1024 * 1024)
    if free_mb < min_mb:
        raise DiskSpaceError(
            f"Insufficient disk space: {free_mb} MB free, "
            f"need at least {min_mb} MB (on {check_path})"
        )

    logger.debug("Disk space OK: %d MB free (min %d MB) on %s",
                 free_mb, min_mb, check_path)
    return free_mb


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint schema version
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_SCHEMA_VERSION = 2

def validate_checkpoint_data(data: dict) -> dict:
    """Validate checkpoint data structure. Adds defaults for missing fields.

    Returns the (possibly modified) data dict ready for Blackboard construction.
    Raises ValueError if the data is fundamentally corrupt.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint data is not a dict: {type(data)}")

    # Required field
    if "feature" not in data or not data["feature"]:
        raise ValueError("Checkpoint missing required 'feature' field")

    # Ensure string fields
    for field_name in ("feature", "project_slug", "crew_name", "prd",
                       "architecture", "contract", "current_phase",
                       "integration_verdict", "integration_notes",
                       "release_verdict",
                       "repo_analysis"):
        data.setdefault(field_name, "")
        if not isinstance(data[field_name], str):
            data[field_name] = str(data[field_name])

    # Ensure list fields
    for field_name in ("file_plan", "completed_phases", "active_agents",
                       "user_interjections", "repo_urls"):
        data.setdefault(field_name, [])
        if not isinstance(data[field_name], list):
            data[field_name] = []

    # Ensure dict fields
    for field_name in ("dep_graph", "interviews"):
        data.setdefault(field_name, {})
        if not isinstance(data[field_name], dict):
            data[field_name] = {}

    # Ensure numeric fields
    data.setdefault("dev_count", 1)
    if not isinstance(data["dev_count"], int):
        data["dev_count"] = 1

    return data
