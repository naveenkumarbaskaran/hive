"""
Hive Sandbox — Safe code execution for the build feedback loop.

Runs generated code in a subprocess with:
  - Timeout protection (default 30s)
  - No network access (best-effort via env scrubbing)
  - Isolated temp directory (cleaned up after)
  - Captured stdout/stderr for feedback to agents
  - Exit code tracking for pass/fail determination

The sandbox supports:
  1. Syntax check  — compile + import without running
  2. Test run      — run pytest/unittest on test files
  3. Lint check    — basic pyflakes/compile check
  4. Full run      — execute a script (for CLI-type projects)

Environment variable:
  HIVE_SANDBOX_TIMEOUT  — max seconds per execution (default: 30)
  HIVE_SANDBOX_ENABLED  — "0" to disable sandbox entirely (default: "1")
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("hive.sandbox")

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

SANDBOX_TIMEOUT = int(os.environ.get("HIVE_SANDBOX_TIMEOUT", "30"))
SANDBOX_ENABLED = os.environ.get("HIVE_SANDBOX_ENABLED", "1") != "0"


# ─────────────────────────────────────────────────────────────────────────────
#  Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    """Result of a sandbox execution."""
    success: bool                      # True if exit code == 0
    exit_code: int = -1                # process exit code
    stdout: str = ""                   # captured stdout (truncated)
    stderr: str = ""                   # captured stderr (truncated)
    command: str = ""                  # what was executed
    timeout: bool = False              # True if killed by timeout
    error: str = ""                    # internal error message (not from subprocess)
    files_written: list[str] = field(default_factory=list)  # files placed in sandbox dir

    @property
    def output(self) -> str:
        """Combined stdout + stderr for prompt injection."""
        parts = []
        if self.stdout.strip():
            parts.append(f"STDOUT:\n{self.stdout.strip()}")
        if self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr.strip()}")
        if self.timeout:
            parts.append(f"TIMEOUT: Process killed after {SANDBOX_TIMEOUT}s")
        if self.error:
            parts.append(f"SANDBOX ERROR: {self.error}")
        return "\n\n".join(parts) or "(no output)"

    @property
    def feedback(self) -> str:
        """Human-readable single-line summary."""
        if self.timeout:
            return f"⏱️ Timeout after {SANDBOX_TIMEOUT}s"
        if self.success:
            return "✅ Execution passed"
        return f"❌ Exit code {self.exit_code}"


# ─────────────────────────────────────────────────────────────────────────────
#  Sandbox execution engine
# ─────────────────────────────────────────────────────────────────────────────

_MAX_OUTPUT = 4000  # max chars of stdout/stderr to capture (avoid prompt bloat)


def _extract_missing_module(stderr: str) -> str | None:
    """Extract the top-level module name from a ModuleNotFoundError message.

    Examples:
        "ModuleNotFoundError: No module named 'click'"  → "click"
        "ModuleNotFoundError: No module named 'todo.models'"  → "todo"
    """
    import re
    m = re.search(r"No module named ['\"]([\w.]+)", stderr)
    if m:
        return m.group(1).split(".")[0]
    return None


def _is_internal_module(module_name: str, staged_files: dict[str, str]) -> bool:
    """Check whether a module name corresponds to a file staged in the sandbox.

    Handles both flat files (``todo.py`` → module ``todo``) and packages
    (``models/user.py`` → module ``models``).
    """
    staged_modules: set[str] = set()
    for f in staged_files:
        if f.endswith(".py"):
            # "todo.py" → "todo", "models/user.py" → "models"
            parts = f.replace(".py", "").split("/")
            staged_modules.add(parts[0])
    return module_name in staged_modules


def _safe_env() -> dict[str, str]:
    """Build a restricted environment for subprocess execution.

    Strips API keys, tokens, and sensitive variables. Keeps PATH, HOME,
    and Python-related vars so the interpreter works.
    """
    keep_prefixes = ("PATH", "HOME", "TMPDIR", "LANG", "LC_", "PYTHON", "VIRTUAL_ENV")
    env: dict[str, str] = {}
    for key, val in os.environ.items():
        if any(key.startswith(p) for p in keep_prefixes):
            env[key] = val
    # Ensure Python can find its stdlib
    env["PYTHONPATH"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _truncate(text: str, max_chars: int = _MAX_OUTPUT) -> str:
    """Truncate text to max_chars, adding a note if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text)} chars total)"


class Sandbox:
    """Isolated execution environment for generated code.

    Usage:
        sandbox = Sandbox()
        sandbox.add_file("calculator.py", code)
        sandbox.add_file("test_calculator.py", test_code)
        result = sandbox.run_tests()
        # or: result = sandbox.syntax_check("calculator.py")
        # or: result = sandbox.run_script("calculator.py", args=["--help"])
    """

    def __init__(self, timeout: int = SANDBOX_TIMEOUT):
        self.timeout = timeout
        self._tmpdir: Path | None = None
        self._files: dict[str, str] = {}

    @property
    def workdir(self) -> Path:
        """Lazy-create the temp directory."""
        if self._tmpdir is None:
            self._tmpdir = Path(tempfile.mkdtemp(prefix="hive_sandbox_"))
        return self._tmpdir

    def add_file(self, name: str, content: str) -> Path:
        """Write a file into the sandbox directory."""
        # Prevent path traversal
        safe_name = name.replace("..", "").lstrip("/")
        path = self.workdir / safe_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self._files[safe_name] = content
        return path

    def add_files(self, files: dict[str, str]) -> None:
        """Write multiple files into the sandbox."""
        for name, content in files.items():
            self.add_file(name, content)

    def cleanup(self) -> None:
        """Remove the sandbox temp directory."""
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    # ─────────────────────────────────────────────────────────────────────
    #  Execution methods
    # ─────────────────────────────────────────────────────────────────────

    def _run(self, cmd: list[str], cwd: Path | None = None) -> SandboxResult:
        """Execute a command in the sandbox with timeout and capture."""
        cwd = cwd or self.workdir
        cmd_str = " ".join(cmd)
        logger.debug("Sandbox exec: %s (cwd=%s, timeout=%ds)", cmd_str, cwd, self.timeout)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=_safe_env(),
            )
            return SandboxResult(
                success=proc.returncode == 0,
                exit_code=proc.returncode,
                stdout=_truncate(proc.stdout),
                stderr=_truncate(proc.stderr),
                command=cmd_str,
                files_written=list(self._files.keys()),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Process timed out after {self.timeout}s",
                command=cmd_str,
                timeout=True,
                files_written=list(self._files.keys()),
            )
        except Exception as exc:
            return SandboxResult(
                success=False,
                exit_code=-1,
                command=cmd_str,
                error=f"Sandbox error: {exc}",
                files_written=list(self._files.keys()),
            )

    def syntax_check(self, filename: str) -> SandboxResult:
        """Check a Python file for syntax errors (compile only, no execution)."""
        filepath = self.workdir / filename
        if not filepath.exists():
            return SandboxResult(
                success=False, error=f"File not found: {filename}",
                command=f"python3 -m py_compile {filename}",
            )
        return self._run([sys.executable, "-m", "py_compile", str(filepath)])

    def syntax_check_all(self) -> SandboxResult:
        """Check all Python files in the sandbox for syntax errors."""
        py_files = [f for f in self._files if f.endswith(".py")]
        if not py_files:
            return SandboxResult(success=True, stdout="No Python files to check")

        errors: list[str] = []
        for f in py_files:
            result = self.syntax_check(f)
            if not result.success:
                errors.append(f"{f}: {result.stderr or result.error}")

        if errors:
            return SandboxResult(
                success=False,
                exit_code=1,
                stderr="\n".join(errors),
                command=f"py_compile [{', '.join(py_files)}]",
            )
        return SandboxResult(
            success=True,
            exit_code=0,
            stdout=f"All {len(py_files)} files compile OK",
            command=f"py_compile [{', '.join(py_files)}]",
        )

    def run_tests(self, test_files: list[str] | None = None) -> SandboxResult:
        """Run pytest (or unittest) on test files in the sandbox.

        If test_files is None, discovers all test_*.py / *_test.py files.
        """
        if test_files is None:
            test_files = [
                f for f in self._files
                if f.startswith("test_") or f.endswith("_test.py")
            ]
        if not test_files:
            return SandboxResult(
                success=True, stdout="No test files found — skipping",
                command="(no tests)",
            )

        # Try pytest first, fall back to unittest
        cmd = [sys.executable, "-m", "pytest", "-x", "-v", "--tb=short",
               "--no-header", "-q"] + test_files
        result = self._run(cmd, cwd=self.workdir)

        # If pytest isn't installed, fall back to unittest
        if not result.success and "No module named" in result.stderr:
            cmd = [sys.executable, "-m", "unittest", "discover",
                   "-s", ".", "-p", "test_*.py", "-v"]
            result = self._run(cmd, cwd=self.workdir)

        return result

    def run_script(self, filename: str, args: list[str] | None = None) -> SandboxResult:
        """Run a Python script with optional arguments."""
        filepath = self.workdir / filename
        if not filepath.exists():
            return SandboxResult(
                success=False, error=f"File not found: {filename}",
                command=f"python3 {filename}",
            )
        cmd = [sys.executable, str(filepath)] + (args or [])
        return self._run(cmd)

    def import_check(self, filename: str) -> SandboxResult:
        """Try to import a module to check for import-time errors.

        This catches issues like missing dependencies, circular imports at
        module level, and runtime errors in top-level code.
        """
        module_name = filename.replace(".py", "").replace("/", ".")
        # Use -c to import the module in a subprocess
        cmd = [sys.executable, "-c", f"import {module_name}"]
        return self._run(cmd, cwd=self.workdir)

    def run_coverage(self, test_files: list[str] | None = None) -> SandboxResult:
        """Run tests with coverage measurement using coverage.py.

        Returns a SandboxResult where stdout contains the coverage report.
        If coverage.py is not installed, falls back to a plain test run.
        """
        if test_files is None:
            test_files = [
                f for f in self._files
                if f.startswith("test_") or f.endswith("_test.py")
            ]
        if not test_files:
            return SandboxResult(
                success=True, stdout="No test files — coverage skipped",
                command="(no tests for coverage)",
            )

        # Determine which source files to measure
        source_files = [f for f in self._files if not f.startswith("test_")]
        source_arg = ",".join(source_files) if source_files else "."

        # Try running with coverage
        cmd = [
            sys.executable, "-m", "coverage", "run",
            "--source", source_arg,
            "-m", "pytest", "-x", "-q", "--tb=short", "--no-header",
        ] + test_files
        result = self._run(cmd, cwd=self.workdir)

        if not result.success and "No module named" in result.stderr:
            # coverage not installed — fall back to plain pytest
            return self.run_tests(test_files)

        if result.success:
            # Generate the report
            report_cmd = [sys.executable, "-m", "coverage", "report", "--show-missing"]
            report = self._run(report_cmd, cwd=self.workdir)
            return SandboxResult(
                success=True,
                exit_code=0,
                stdout=f"Tests passed.\n\nCOVERAGE REPORT:\n{report.stdout}",
                stderr=report.stderr,
                command="coverage run + report",
                files_written=list(self._files.keys()),
            )

        return result


# ─────────────────────────────────────────────────────────────────────────────
#  PII Scanner — regex-based static analysis
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that may indicate PII leakage in code
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Hardcoded email", re.compile(
        r"""(?:"|')[\w.+-]+@[\w-]+\.[\w.-]+(?:"|')""", re.IGNORECASE)),
    ("Hardcoded IP address", re.compile(
        r"""(?:"|')\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:"|')""")),
    ("Hardcoded password/secret", re.compile(
        r"""(?:password|secret|api_key|apikey|token|auth)\s*=\s*(?:"|')[^"']{4,}(?:"|')""",
        re.IGNORECASE)),
    ("PII in log/print statement", re.compile(
        r"""(?:log(?:ger)?\.(?:info|debug|warning|error|critical)|print)\s*\(.*"""
        r"""(?:password|email|ssn|social.?security|phone|address|credit.?card"""
        r"""|date.?of.?birth|dob)""",
        re.IGNORECASE)),
    ("eval/exec with variable", re.compile(
        r"""(?:eval|exec)\s*\([^)"']*[a-zA-Z_]""")),
    ("pickle.loads (unsafe deserialization)", re.compile(
        r"""pickle\.loads?\s*\(""")),
    ("yaml.load without SafeLoader", re.compile(
        r"""yaml\.load\s*\([^)]*\)(?!.*(?:Safe|Full)Loader)""")),
    ("subprocess with shell=True", re.compile(
        r"""subprocess\.(?:run|call|Popen)\s*\([^)]*shell\s*=\s*True""")),
]


@dataclass
class PIIFinding:
    """A single PII/security finding from the static scanner."""
    filename: str
    line_number: int
    rule: str
    snippet: str


def scan_pii(files: dict[str, str]) -> list[PIIFinding]:
    """Scan source code files for PII leakage and security anti-patterns.

    Returns a list of findings. Each finding includes the file, line, rule name,
    and the offending code snippet.
    """
    findings: list[PIIFinding] = []

    for filename, code in files.items():
        # Skip test files — they may intentionally use fake data
        if filename.startswith("test_"):
            continue
        for line_num, line in enumerate(code.splitlines(), start=1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            for rule_name, pattern in _PII_PATTERNS:
                if pattern.search(line):
                    findings.append(PIIFinding(
                        filename=filename,
                        line_number=line_num,
                        rule=rule_name,
                        snippet=stripped[:120],
                    ))

    return findings


def format_pii_findings(findings: list[PIIFinding]) -> str:
    """Format PII findings as a human-readable report."""
    if not findings:
        return "PII/Security scan: CLEAN — no issues found."

    lines = [f"PII/Security scan: {len(findings)} finding(s):\n"]
    for f in findings:
        lines.append(f"  [{f.rule}] {f.filename}:{f.line_number} — {f.snippet}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def run_code_checks(
    files: dict[str, str],
    timeout: int = SANDBOX_TIMEOUT,
) -> SandboxResult:
    """Run syntax check + tests on a set of files. Returns combined result.

    This is the main entry point for the build phase feedback loop.
    """
    if not SANDBOX_ENABLED:
        return SandboxResult(success=True, stdout="Sandbox disabled (HIVE_SANDBOX_ENABLED=0)")

    with Sandbox(timeout=timeout) as sb:
        sb.add_files(files)

        # Phase 1: Syntax check all files
        syntax = sb.syntax_check_all()
        if not syntax.success:
            return SandboxResult(
                success=False,
                exit_code=1,
                stdout=syntax.stdout,
                stderr=f"SYNTAX ERRORS:\n{syntax.stderr}",
                command=syntax.command,
                files_written=list(files.keys()),
            )

        # Phase 2: Import check non-test files
        source_files = [f for f in files if not f.startswith("test_")]
        import_errors: list[str] = []
        for f in source_files:
            if f.endswith(".py"):
                result = sb.import_check(f)
                if not result.success:
                    stderr = result.stderr
                    if "ModuleNotFoundError" in stderr:
                        missing = _extract_missing_module(stderr)
                        if missing and not _is_internal_module(missing, files):
                            # Truly external — not staged in the sandbox
                            logger.info(
                                "Import check for %s: external dep '%s' (OK)",
                                f, missing,
                            )
                            continue
                        # Internal module is missing → real problem
                        logger.warning(
                            "Import check for %s: internal module '%s' not found",
                            f, missing,
                        )
                    import_errors.append(f"{f}: {stderr}")

        if import_errors:
            return SandboxResult(
                success=False,
                exit_code=1,
                stderr="IMPORT ERRORS:\n" + "\n".join(import_errors),
                command="import check",
                files_written=list(files.keys()),
            )

        # Phase 3: Run tests if any exist
        test_result = sb.run_tests()
        if test_result.command == "(no tests)":
            # No tests to run — syntax + import passed, that's a win
            return SandboxResult(
                success=True,
                exit_code=0,
                stdout=f"Syntax OK ({len(files)} files). No tests to run.",
                command="syntax + import check",
                files_written=list(files.keys()),
            )

        return test_result


def syntax_check_file(filename: str, code: str) -> SandboxResult:
    """Quick syntax check for a single file."""
    if not SANDBOX_ENABLED:
        return SandboxResult(success=True, stdout="Sandbox disabled")

    with Sandbox(timeout=10) as sb:
        sb.add_file(filename, code)
        return sb.syntax_check(filename)


def check_file_in_context(
    target: str,
    target_code: str,
    context_files: dict[str, str],
    timeout: int = SANDBOX_TIMEOUT,
) -> SandboxResult:
    """Check a single file with supporting context files staged alongside.

    Writes *all* ``context_files`` plus the ``target`` into the sandbox, then
    runs syntax + import checks on the **target only**.  Sibling files are
    present so that cross-module imports resolve correctly.

    ModuleNotFoundError for truly-external packages (not in ``context_files``)
    are tolerated; errors for modules that *are* staged are treated as real
    failures.
    """
    if not SANDBOX_ENABLED:
        return SandboxResult(success=True, stdout="Sandbox disabled (HIVE_SANDBOX_ENABLED=0)")

    all_files = {**context_files, target: target_code}

    with Sandbox(timeout=timeout) as sb:
        sb.add_files(all_files)

        # 1. Syntax check the target file
        result = sb.syntax_check(target)
        if not result.success:
            return result

        # 2. Import check (Python source files only, skip test files)
        if target.endswith(".py") and not target.startswith("test_"):
            result = sb.import_check(target)
            if not result.success:
                stderr = result.stderr
                if "ModuleNotFoundError" in stderr:
                    missing = _extract_missing_module(stderr)
                    if missing and not _is_internal_module(missing, all_files):
                        # Truly external dep — not our problem
                        return SandboxResult(
                            success=True,
                            exit_code=0,
                            stdout=(
                                f"Syntax OK: {target}. "
                                f"Import: external dep '{missing}' — skipped."
                            ),
                            command=f"syntax + import check: {target}",
                            files_written=list(all_files.keys()),
                        )
                # Internal module missing or non-import error → real failure
                return result

        return SandboxResult(
            success=True,
            exit_code=0,
            stdout=f"Syntax + import OK: {target}",
            command=f"check_file_in_context: {target}",
            files_written=list(all_files.keys()),
        )
