"""
EPT Connectors — Ingest external knowledge into the Blackboard.

Users can attach:
  - Business / domain documents  (.md .txt .rst)
  - API specs                    (openapi.yaml, swagger.json, .graphql, .proto)
  - Database schemas             (.sql .ddl .prisma)
  - Sample / live data files     (.csv .tsv .json .yaml)
  - Reference code               (.py .js .ts .java .go .rs .rb .cs)
  - Test cases                   (test_*.py, *_test.*, *.spec.*)
  - Full directories             (recursed & classified per-file)
  - Git repositories             (cloned, analyzed by Scout, used as reference)
  - URLs                         (https://... — fetched via httpx, auto-typed)

Each ingested item becomes a KnowledgeItem stored on the Blackboard.
Large files are auto-summarized by Scout; small ones are injected in full.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import httpx

from hive.hardening import register_temp_path

logger = logging.getLogger("hive.connectors")


# ─────────────────────────────────────────────────────────────────────────────
#  Connector types
# ─────────────────────────────────────────────────────────────────────────────

class ConnectorType(Enum):
    DOCUMENT  = "document"       # .md .txt .rst — business / domain docs
    CODEBASE  = "codebase"       # .py .js .ts .java — reference code
    TEST_CASE = "test_case"      # test_*.py, *.spec.ts — existing tests
    DATA_FILE = "data_file"      # .csv .json .yaml — sample / live data
    API_SPEC  = "api_spec"       # openapi.yaml, swagger.json, .graphql
    SCHEMA    = "schema"         # .sql .ddl .prisma — database schemas
    GIT_REPO  = "git_repo"       # cloned git repository (reference impl)


# ─────────────────────────────────────────────────────────────────────────────
#  Knowledge item — one ingested artefact
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeItem:
    """One ingested knowledge artefact from the user."""
    source_type: str              # ConnectorType.value
    source_path: str              # original path provided by user
    label: str                    # human-readable name
    content: str                  # processed text (full or truncated)
    raw_size: int                 # original size in bytes
    was_summarized: bool = False  # True if Scout compressed it
    summary: str = ""             # Scout's summary (for large items)
    tags: list[str] = field(default_factory=list)      # routing tags
    metadata: dict = field(default_factory=dict)        # extra info


# ─────────────────────────────────────────────────────────────────────────────
#  Size tiers
# ─────────────────────────────────────────────────────────────────────────────

SMALL_THRESHOLD  = 8_192      # 8 KB  → inject in full
MEDIUM_THRESHOLD = 51_200     # 50 KB → truncate (first+last N lines)
# > 50 KB  → summarize via Scout

TRUNCATE_HEAD = 120           # lines to keep from start
TRUNCATE_TAIL = 40            # lines to keep from end


# ─────────────────────────────────────────────────────────────────────────────
#  Extension → ConnectorType mapping
# ─────────────────────────────────────────────────────────────────────────────

# Default extension map; register() can extend it
_EXT_MAP: dict[str, ConnectorType] = {
    # Documents
    ".md": ConnectorType.DOCUMENT,
    ".txt": ConnectorType.DOCUMENT,
    ".rst": ConnectorType.DOCUMENT,
    # Code
    ".py": ConnectorType.CODEBASE,
    ".js": ConnectorType.CODEBASE,
    ".ts": ConnectorType.CODEBASE,
    ".tsx": ConnectorType.CODEBASE,
    ".jsx": ConnectorType.CODEBASE,
    ".java": ConnectorType.CODEBASE,
    ".go": ConnectorType.CODEBASE,
    ".rs": ConnectorType.CODEBASE,
    ".rb": ConnectorType.CODEBASE,
    ".cs": ConnectorType.CODEBASE,
    ".c": ConnectorType.CODEBASE,
    ".cpp": ConnectorType.CODEBASE,
    ".h": ConnectorType.CODEBASE,
    ".hpp": ConnectorType.CODEBASE,
    ".swift": ConnectorType.CODEBASE,
    ".kt": ConnectorType.CODEBASE,
    # Data
    ".csv": ConnectorType.DATA_FILE,
    ".tsv": ConnectorType.DATA_FILE,
    ".json": ConnectorType.DATA_FILE,
    ".yaml": ConnectorType.DATA_FILE,
    ".yml": ConnectorType.DATA_FILE,
    ".xml": ConnectorType.DATA_FILE,
    # API specs (detected by name too — see _detect_type)
    ".graphql": ConnectorType.API_SPEC,
    ".gql": ConnectorType.API_SPEC,
    ".proto": ConnectorType.API_SPEC,
    # Schema
    ".sql": ConnectorType.SCHEMA,
    ".ddl": ConnectorType.SCHEMA,
    ".prisma": ConnectorType.SCHEMA,
}

# Filenames / patterns that override the extension-based type
_NAME_OVERRIDES: list[tuple[re.Pattern, ConnectorType]] = [
    (re.compile(r"openapi", re.IGNORECASE), ConnectorType.API_SPEC),
    (re.compile(r"swagger", re.IGNORECASE), ConnectorType.API_SPEC),
]

# Test-case name patterns — override CODEBASE → TEST_CASE
_TEST_PATTERNS: list[re.Pattern] = [
    re.compile(r"^test_"),
    re.compile(r"_test\."),
    re.compile(r"\.test\."),
    re.compile(r"\.spec\."),
]

# Directories / files to skip
_SKIP_DIRS = {
    "__pycache__", ".git", ".svn", "node_modules", ".venv", "venv",
    ".pytest_cache", ".mypy_cache", ".tox", "dist", "build", ".next",
    ".idea", ".vscode",
}
_SKIP_FILES = {".DS_Store", "Thumbs.db", ".gitignore", ".gitkeep"}

# Extensions to skip (binary / non-readable)
_SKIP_EXTS = {
    ".pyc", ".pyo", ".class", ".o", ".so", ".dylib", ".dll",
    ".exe", ".bin", ".wasm", ".png", ".jpg", ".jpeg", ".gif",
    ".bmp", ".ico", ".svg", ".webp", ".mp3", ".mp4", ".wav",
    ".avi", ".zip", ".tar", ".gz", ".rar", ".7z", ".jar",
    ".whl", ".egg", ".pdf", ".docx", ".pptx", ".xlsx",
}


# ─────────────────────────────────────────────────────────────────────────────
#  ConnectorRegistry — detection, reading, registration
# ─────────────────────────────────────────────────────────────────────────────

class ConnectorRegistry:
    """Detects file types, reads content, and creates KnowledgeItems."""

    # ── Type detection ──────────────────────────────────────────────────

    @staticmethod
    def detect_type(path: Path) -> ConnectorType | None:
        """Auto-detect connector type from file extension / name.

        Returns None if the file is not a recognizable type.
        """
        name = path.name
        ext = path.suffix.lower()

        # Skip known binary / noise
        if ext in _SKIP_EXTS or name in _SKIP_FILES:
            return None

        # Name-based overrides (openapi.yaml, swagger.json, etc.)
        for pattern, ctype in _NAME_OVERRIDES:
            if pattern.search(name):
                return ctype

        # Test-case detection (overrides CODEBASE)
        base = ext and name.replace(ext, "")  # filename without ext
        for pattern in _TEST_PATTERNS:
            if pattern.search(name) or (base and pattern.search(base)):
                return ConnectorType.TEST_CASE

        # Extension-based
        return _EXT_MAP.get(ext)

    # ── Reading ─────────────────────────────────────────────────────────

    @staticmethod
    def _read_text(path: Path) -> tuple[str, int]:
        """Read a text file, return (content, raw_size).

        Falls back gracefully on encoding errors.
        """
        raw_size = path.stat().st_size
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="latin-1")
            except Exception:
                text = f"(could not read {path.name}: encoding error)"
        return text, raw_size

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate a medium-sized file to head + tail lines."""
        lines = text.splitlines()
        total = len(lines)
        if total <= TRUNCATE_HEAD + TRUNCATE_TAIL:
            return text  # already fits
        head = lines[:TRUNCATE_HEAD]
        tail = lines[-TRUNCATE_TAIL:]
        omitted = total - TRUNCATE_HEAD - TRUNCATE_TAIL
        return "\n".join(
            head
            + [f"\n# ... ({omitted} lines omitted from middle) ...\n"]
            + tail
        )

    # ── Tagging ─────────────────────────────────────────────────────────

    @staticmethod
    def _auto_tags(ctype: ConnectorType, path: Path) -> list[str]:
        """Generate routing tags from type and path."""
        tags = [ctype.value]
        ext = path.suffix.lower().lstrip(".")
        if ext:
            tags.append(ext)
        # Add parent directory name as a tag (useful for routing)
        parent = path.parent.name
        if parent and parent != ".":
            tags.append(parent)
        return tags

    # ── Single-file ingest ──────────────────────────────────────────────

    @classmethod
    def ingest_file(cls, path: Path, force_type: ConnectorType | None = None) -> KnowledgeItem | None:
        """Ingest a single file into a KnowledgeItem.

        Returns None if the file type is unrecognized or unreadable.
        The `was_summarized` flag is set for large files, but the actual
        summarization is done later by Scout in the ingest phase.
        """
        if not path.is_file():
            return None

        ctype = force_type or cls.detect_type(path)
        if ctype is None:
            return None

        text, raw_size = cls._read_text(path)

        # Size-based processing
        if raw_size <= SMALL_THRESHOLD:
            content = text
            was_summarized = False
        elif raw_size <= MEDIUM_THRESHOLD:
            content = cls._truncate(text)
            was_summarized = False
        else:
            # Large: store truncated version, mark for summarization
            content = cls._truncate(text)
            was_summarized = True

        tags = cls._auto_tags(ctype, path)
        line_count = len(text.splitlines())

        return KnowledgeItem(
            source_type=ctype.value,
            source_path=str(path),
            label=path.name,
            content=content,
            raw_size=raw_size,
            was_summarized=was_summarized,
            summary="",
            tags=tags,
            metadata={"lines": line_count, "extension": path.suffix},
        )

    # ── Directory ingest ────────────────────────────────────────────────

    @classmethod
    def ingest_directory(
        cls,
        directory: Path,
        force_type: ConnectorType | None = None,
        max_files: int = 200,
    ) -> list[KnowledgeItem]:
        """Recursively ingest a directory into KnowledgeItems.

        Skips hidden dirs, __pycache__, node_modules, etc.
        Limits to max_files to prevent overwhelming the context window.
        """
        items: list[KnowledgeItem] = []
        count = 0

        for child in sorted(directory.rglob("*")):
            if count >= max_files:
                break

            # Skip unwanted directories
            parts = child.relative_to(directory).parts
            if any(p in _SKIP_DIRS or p.startswith(".") for p in parts):
                continue

            if child.is_file():
                item = cls.ingest_file(child, force_type)
                if item is not None:
                    count += 1
                    items.append(item)

        return items

    # ── Main entry point ────────────────────────────────────────────────

    @classmethod
    def ingest(
        cls,
        path_str: str,
        force_type: ConnectorType | None = None,
    ) -> list[KnowledgeItem]:
        """Ingest a path, URL, or directory into KnowledgeItems.

        Accepts:
          - A file path:     "./docs/spec.md"
          - A directory:     "./docs/"
          - A typed path:    "./api.yaml:api_spec" (colon-separated override)
          - A URL:           "https://example.com/spec.yaml" (fetched via httpx)
        """
        # ── URL handling: fetch remote content ──────────────────────────
        if is_url(path_str):
            return cls.ingest_url(path_str, force_type)

        # Check for type override suffix  "path:type_name"
        # E.g. "./file.json:api_spec" or "/abs/path.json:api_spec"
        if ":" in path_str:
            parts = path_str.rsplit(":", 1)
            if len(parts) == 2 and parts[1] in {ct.value for ct in ConnectorType}:
                path_str = parts[0]
                force_type = ConnectorType(parts[1])

        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            return []

        if path.is_dir():
            return cls.ingest_directory(path, force_type)
        else:
            item = cls.ingest_file(path, force_type)
            return [item] if item else []

    @classmethod
    def ingest_url(
        cls,
        url: str,
        force_type: ConnectorType | None = None,
    ) -> list[KnowledgeItem]:
        """Fetch a URL and ingest its content as a KnowledgeItem.

        Auto-detects type from URL path extension or Content-Type header.
        Returns empty list on fetch failure.
        """
        text, raw_size, content_type = fetch_url(url)
        if text is None:
            return []

        # Detect type from URL path extension, then Content-Type header
        ctype = force_type
        if ctype is None:
            url_path = urlparse(url).path
            ext = Path(url_path).suffix.lower()
            if ext:
                ctype = _EXT_MAP.get(ext)
            if ctype is None:
                ctype = _content_type_to_connector(content_type)

        label = _url_label(url)

        # Size-based processing (same logic as ingest_file)
        if raw_size <= SMALL_THRESHOLD:
            content = text
            was_summarized = False
        elif raw_size <= MEDIUM_THRESHOLD:
            content = cls._truncate(text)
            was_summarized = False
        else:
            content = cls._truncate(text)
            was_summarized = True

        tags = [ctype.value, "url"]
        # Add domain as a tag for routing context
        domain = urlparse(url).netloc
        if domain:
            tags.append(domain)

        item = KnowledgeItem(
            source_type=ctype.value,
            source_path=url,
            label=label,
            content=content,
            raw_size=raw_size,
            was_summarized=was_summarized,
            summary="",
            tags=tags,
            metadata={"url": url, "content_type": content_type},
        )
        return [item]

    @classmethod
    def ingest_all(cls, paths: list[str]) -> list[KnowledgeItem]:
        """Batch-ingest multiple paths, deduplicating by resolved source_path."""
        seen: set[str] = set()
        items: list[KnowledgeItem] = []
        for p in paths:
            for item in cls.ingest(p.strip()):
                if item.source_path not in seen:
                    seen.add(item.source_path)
                    items.append(item)
        return items

    # ── Extension API ───────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        ctype: ConnectorType,
        extensions: list[str],
        processor: Callable[[Path], str] | None = None,
    ) -> None:
        """Register new extensions for a connector type.

        If processor is given, it replaces the default read_text for those
        extensions (future: PDF extraction, protobuf parsing, etc.).
        """
        for ext in extensions:
            _EXT_MAP[ext] = ctype
        # Future: store processor for custom file parsing (PDF, protobuf, etc.)
        if processor is not None:
            logger.info("Custom processor registered for %s (not yet wired)", extensions)


# ─────────────────────────────────────────────────────────────────────────────
#  Agent routing — which agents consume which types
# ─────────────────────────────────────────────────────────────────────────────

# Maps agent role keywords → the connector types they should receive
AGENT_ROUTING: dict[str, set[str]] = {
    "scout":         {ct.value for ct in ConnectorType},  # everything
    "penny":         {ConnectorType.DOCUMENT.value, ConnectorType.DATA_FILE.value,
                      ConnectorType.GIT_REPO.value},
    "archie":        {ConnectorType.API_SPEC.value, ConnectorType.SCHEMA.value,
                      ConnectorType.CODEBASE.value, ConnectorType.GIT_REPO.value},
    "quinn":         {ConnectorType.TEST_CASE.value, ConnectorType.API_SPEC.value,
                      ConnectorType.GIT_REPO.value},
    "pixel":         {ConnectorType.DOCUMENT.value, ConnectorType.CODEBASE.value},
    "alex":          {ConnectorType.DOCUMENT.value, ConnectorType.CODEBASE.value},
    "flow":          {ConnectorType.DOCUMENT.value, ConnectorType.CODEBASE.value},
    "judge":         {ConnectorType.TEST_CASE.value, ConnectorType.API_SPEC.value},
    "dev":           {ConnectorType.CODEBASE.value, ConnectorType.API_SPEC.value,
                      ConnectorType.SCHEMA.value, ConnectorType.GIT_REPO.value},
    "integration":   {ConnectorType.TEST_CASE.value, ConnectorType.API_SPEC.value},
    "release":       {ConnectorType.DOCUMENT.value},
}


def knowledge_for_agent(
    items: list[KnowledgeItem],
    agent_role: str,
    max_chars: int = 12_000,
) -> str:
    """Return a formatted knowledge block for a specific agent role.

    Filters items by the agent's routing rules and truncates the
    total output to max_chars.
    """
    role_key = agent_role.lower().split()[0]  # "Scout 🔍" → "scout"
    # Devs: "dev_1", "dev_2" etc.
    if role_key.startswith("dev"):
        role_key = "dev"

    allowed = AGENT_ROUTING.get(role_key, set())
    if not allowed:
        return ""

    relevant = [i for i in items if i.source_type in allowed]
    if not relevant:
        return ""

    parts = ["ATTACHED KNOWLEDGE:"]
    budget = max_chars
    for item in relevant:
        # Prefer summary for summarized items
        text = item.summary if item.was_summarized and item.summary else item.content
        header = f"\n### [{item.source_type.upper()}] {item.label}"
        entry = f"{header}\n{text}"
        if len(entry) > budget:
            entry = entry[:budget] + "\n(... truncated ...)"
            parts.append(entry)
            break
        parts.append(entry)
        budget -= len(entry)

    return "\n".join(parts)


def knowledge_context(
    items: list[KnowledgeItem],
    max_chars: int = 12_000,
) -> str:
    """Return a formatted knowledge block with ALL items (used for Scout)."""
    if not items:
        return ""
    return knowledge_for_agent(items, "scout", max_chars)


def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ─────────────────────────────────────────────────────────────────────────────
#  URL support — fetch remote content for --attach https://...
# ─────────────────────────────────────────────────────────────────────────────

_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)

# Map Content-Type header values to ConnectorType
_CONTENT_TYPE_MAP: dict[str, ConnectorType] = {
    "text/markdown": ConnectorType.DOCUMENT,
    "text/plain": ConnectorType.DOCUMENT,
    "text/html": ConnectorType.DOCUMENT,
    "text/csv": ConnectorType.DATA_FILE,
    "application/json": ConnectorType.DATA_FILE,
    "application/yaml": ConnectorType.DATA_FILE,
    "application/x-yaml": ConnectorType.DATA_FILE,
    "text/yaml": ConnectorType.DATA_FILE,
    "application/xml": ConnectorType.DATA_FILE,
    "text/xml": ConnectorType.DATA_FILE,
    "application/sql": ConnectorType.SCHEMA,
    "application/graphql": ConnectorType.API_SPEC,
}


def is_url(path_str: str) -> bool:
    """Check if a string is an HTTP(S) URL (not a git repo URL)."""
    stripped = path_str.strip()
    # Git URLs are handled separately by is_git_url()
    if is_git_url(stripped):
        return False
    return bool(_URL_PATTERN.match(stripped))


def fetch_url(url: str, timeout: int = 30) -> tuple[str | None, int, str]:
    """Fetch a URL and return (text, size_bytes, content_type).

    Returns (None, 0, "") on failure. Handles redirects, timeouts,
    and non-text responses gracefully.
    """
    try:
        resp = httpx.get(
            url.strip(),
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Hive-EPT/1.0 (knowledge-ingest)"},
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch URL %s: %s", url, exc)
        return None, 0, ""

    content_type = resp.headers.get("content-type", "")
    # Extract mime type (strip charset and params)
    mime = content_type.split(";")[0].strip().lower()

    # Reject binary content types
    if mime.startswith(("image/", "audio/", "video/", "application/octet-stream",
                        "application/zip", "application/pdf")):
        logger.warning("Skipping binary URL %s (content-type: %s)", url, mime)
        return None, 0, ""

    text = resp.text
    raw_size = len(resp.content)
    return text, raw_size, mime


def _content_type_to_connector(content_type: str) -> ConnectorType:
    """Map a MIME content-type to a ConnectorType. Defaults to DOCUMENT."""
    ct = content_type.split(";")[0].strip().lower()
    return _CONTENT_TYPE_MAP.get(ct, ConnectorType.DOCUMENT)


def _url_label(url: str) -> str:
    """Extract a human-readable label from a URL."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path:
        name = path.rsplit("/", 1)[-1]
        if name:
            return name
    return parsed.netloc or url[:50]


# ─────────────────────────────────────────────────────────────────────────────
#  Git repository support
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that identify a string as a git repo URL
_GIT_URL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^https?://github\.com/", re.IGNORECASE),
    re.compile(r"^https?://gitlab\.com/", re.IGNORECASE),
    re.compile(r"^https?://bitbucket\.org/", re.IGNORECASE),
    re.compile(r"^git@"),
    re.compile(r"\.git$"),
    re.compile(r"^https?://.*\.git$", re.IGNORECASE),
]


def is_git_url(path_str: str) -> bool:
    """Check if a string looks like a git repository URL."""
    return any(p.search(path_str.strip()) for p in _GIT_URL_PATTERNS)


def clone_repo(url: str, target_dir: Path | None = None, depth: int = 1) -> Path:
    """Shallow-clone a git repo and return the local path.

    Args:
        url:        Git clone URL (https or ssh)
        target_dir: Where to clone. If None, uses a temp directory.
        depth:      Clone depth (1 = latest commit only)

    Returns:
        Path to the cloned repo root.

    Raises:
        RuntimeError: if `git clone` fails.
    """
    if target_dir is None:
        # Extract a name from the URL for the temp folder
        name = url.rstrip("/").rsplit("/", 1)[-1]
        name = name.removesuffix(".git")
        target_dir = Path(tempfile.mkdtemp(prefix=f"hive_repo_{name}_"))
        # Register for cleanup on process exit
        register_temp_path(target_dir)
        logger.debug("Clone temp dir registered for cleanup: %s", target_dir)

    cmd = ["git", "clone", "--depth", str(depth), url, str(target_dir)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return target_dir


def repo_file_tree(repo_dir: Path, max_depth: int = 4) -> str:
    """Build a compact tree string of a repo's file structure.

    Like `tree -L 4 --dirsfirst` but in pure Python.
    Skips .git, __pycache__, node_modules, etc.
    """
    lines: list[str] = [repo_dir.name + "/"]

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        # Filter out skippable dirs/files
        children = [
            c for c in children
            if c.name not in _SKIP_DIRS
            and not c.name.startswith(".")
            and c.name not in _SKIP_FILES
            and c.suffix not in _SKIP_EXTS
        ]
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            if child.is_dir():
                lines.append(f"{prefix}{connector}{child.name}/")
                extension = "    " if is_last else "│   "
                _walk(child, prefix + extension, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{child.name}")

    _walk(repo_dir, "", 1)
    return "\n".join(lines[:300])  # cap at 300 lines


def ingest_repo(
    url: str,
    target_dir: Path | None = None,
    max_files: int = 200,
) -> tuple[list[KnowledgeItem], Path, str]:
    """Clone a git repo and ingest it.

    Returns:
        (items, repo_path, file_tree) — the ingested items, the local
        clone path, and a compact file tree string.
    """
    repo_path = clone_repo(url, target_dir)
    tree = repo_file_tree(repo_path)
    items = ConnectorRegistry.ingest_directory(repo_path, max_files=max_files)

    # Tag every item as coming from a git repo
    for item in items:
        item.tags.append("git_repo")
        item.metadata["git_url"] = url

    return items, repo_path, tree
