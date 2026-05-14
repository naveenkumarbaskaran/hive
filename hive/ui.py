""" 
Hive Terminal UI — Rich terminal display for agent activity.

Renders Events from the Blackboard into a beautiful terminal experience.
Shows agent thought processes, handshakes, reviews, and progress.

Uses ANSI escape codes directly — no heavy dependencies.

Respects NO_COLOR (https://no-color.org/) and TERM=dumb for plain output.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from typing import Callable

from hive.state import Blackboard, Event, EventType, UserProfile


# ─────────────────────────────────────────────────────────────────────────────
#  NO_COLOR / dumb terminal detection
# ─────────────────────────────────────────────────────────────────────────────

def _use_color() -> bool:
    """Return True if ANSI color output should be used."""
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


USE_COLOR = _use_color()


# ─────────────────────────────────────────────────────────────────────────────
#  ANSI color helpers
# ─────────────────────────────────────────────────────────────────────────────

class C:
    """ANSI color codes."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    UNDER   = "\033[4m"

    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    BG_BLACK   = "\033[40m"
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN    = "\033[46m"
    BG_WHITE   = "\033[47m"

    # 256-color
    @staticmethod
    def fg(n: int) -> str:
        return f"\033[38;5;{n}m"

    @staticmethod
    def bg(n: int) -> str:
        return f"\033[48;5;{n}m"


def colored(text: str, *codes: str) -> str:
    if not USE_COLOR:
        return text
    return "".join(codes) + text + C.RESET


# ─────────────────────────────────────────────────────────────────────────────
#  Agent color map
# ─────────────────────────────────────────────────────────────────────────────

AGENT_COLORS: dict[str, str] = {
    "scout":  C.CYAN,
    "penny":  C.YELLOW,
    "archie": C.BLUE,
    "quinn":  C.GREEN,
    "judge":  C.MAGENTA,
    "pixel":  C.fg(208),   # orange
    "flow":   C.fg(141),   # lavender
    "alex":   C.fg(219),   # pink
    "user":   C.WHITE,
    "system": C.DIM,
}

def agent_color(agent_id: str) -> str:
    """Get color for an agent. Dev agents get a cycling color."""
    if agent_id.startswith("dev_"):
        dev_colors = [C.RED, C.fg(202), C.fg(226), C.fg(118), C.fg(51), C.fg(135)]
        try:
            idx = int(agent_id.split("_")[1]) - 1
        except (ValueError, IndexError):
            idx = 0
        return dev_colors[idx % len(dev_colors)]
    return AGENT_COLORS.get(agent_id, C.WHITE)


# ─────────────────────────────────────────────────────────────────────────────
#  Emoji lookup for agents
# ─────────────────────────────────────────────────────────────────────────────

AGENT_EMOJI: dict[str, str] = {
    "scout": "🔍", "penny": "📋", "archie": "🏗️", "quinn": "🧪",
    "judge": "⚖️", "pixel": "🎨", "flow": "🧭", "alex": "👤",
    "user": "👨‍💻", "system": "⚙️",
}

def agent_emoji(agent_id: str) -> str:
    if agent_id.startswith("dev_"):
        return "🔨"
    return AGENT_EMOJI.get(agent_id, "🤖")


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal width
# ─────────────────────────────────────────────────────────────────────────────

def term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except (AttributeError, ValueError, OSError):
        return 80


# ─────────────────────────────────────────────────────────────────────────────
#  UI Renderer
# ─────────────────────────────────────────────────────────────────────────────

class TerminalUI:
    """Renders EPT events to the terminal with flair."""

    def __init__(self, board: Blackboard, verbose: bool = False):
        self.board = board
        self.verbose = verbose
        self._rendered_count = 0                     # track what we've shown
        self._phase_start_time: float | None = None
        self._thinking_shown: set[str] = set()       # avoid flooding thinking msgs

    # ─────────────────────────────────────────────────────────────────────────
    #  High-level prints
    # ─────────────────────────────────────────────────────────────────────────

    def banner(self) -> None:
        w = term_width()
        title = "EPT — Empowered Product Team: The Crew"
        box_w = max(len(title) + 4, 50)
        pad = (w - box_w) // 2

        print()
        print(" " * pad + colored("╔" + "═" * (box_w - 2) + "╗", C.CYAN, C.BOLD))
        inner = title.center(box_w - 2)
        print(" " * pad + colored("║", C.CYAN, C.BOLD) + colored(inner, C.WHITE, C.BOLD) + colored("║", C.CYAN, C.BOLD))
        print(" " * pad + colored("╚" + "═" * (box_w - 2) + "╝", C.CYAN, C.BOLD))
        print()

    def crew_intro(self, agents: dict) -> None:
        """Show the active agents with their cards."""
        print(colored("  The Crew:", C.BOLD))
        print()
        for aid, a in agents.items():
            if not a.active:
                continue
            color = agent_color(aid)
            status = colored("ACTIVE", C.GREEN, C.BOLD)
            print(f"    {colored(a.emoji + ' ' + a.name, color, C.BOLD):30}  "
                  f"{a.role:22}  {status}")
            print(f"    {colored('  ' + '\"' + a.tagline + '\"', C.DIM, C.ITALIC)}")
        print()

    def welcome_intake(self) -> UserProfile:
        """Welcome the user and collect identity & context. Returns UserProfile."""
        w = term_width()
        border = "─" * (w - 4)

        print()
        print(colored(f"  {border}", C.CYAN))
        print(colored("  👋 Welcome to Hive — Your AI Dev Crew, Assembled!", C.BOLD, C.CYAN))
        print(colored(f"  {border}", C.CYAN))
        print()
        print(colored("  Before we begin, let's get to know you a little.", C.DIM))
        print(colored("  (All fields are optional — press Enter to skip any)", C.DIM))
        print()

        # Name
        print(colored("  What's your name?", C.WHITE, C.BOLD))
        print(colored("    > ", C.DIM), end="")
        try:
            name = input().strip()
        except (EOFError, KeyboardInterrupt):
            name = ""

        # Greeting
        if name:
            print(colored(f"\n  Nice to meet you, {name}! 🤝", C.GREEN, C.BOLD))
        else:
            print(colored("\n  No worries — let's get building! 🤝", C.GREEN))

        # Role
        print()
        print(colored("  What's your role? (e.g., Product Owner, Developer, Founder)", C.WHITE, C.BOLD))
        print(colored("    > ", C.DIM), end="")
        try:
            role = input().strip()
        except (EOFError, KeyboardInterrupt):
            role = ""

        # Company (optional)
        print()
        print(colored("  Company / organization? (optional)", C.WHITE, C.BOLD))
        print(colored("    > ", C.DIM), end="")
        try:
            company = input().strip()
        except (EOFError, KeyboardInterrupt):
            company = ""

        # Is this for yourself or someone else?
        print()
        print(colored("  Is this feature request for yourself, or for another user/team?", C.WHITE, C.BOLD))
        print(colored("    [1] For myself", C.DIM))
        print(colored("    [2] For someone else / another team", C.DIM))
        print(colored("    > ", C.DIM), end="")
        try:
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            choice = "1"

        is_for_self = choice != "2"
        end_user_name = ""
        end_user_role = ""
        end_user_desc = ""

        if not is_for_self:
            print()
            print(colored("  Who is the end user? (name or team)", C.WHITE, C.BOLD))
            print(colored("    > ", C.DIM), end="")
            try:
                end_user_name = input().strip()
            except (EOFError, KeyboardInterrupt):
                end_user_name = ""

            print()
            print(colored("  What is the end user's role? (e.g., Customer Service Agent, Admin)", C.WHITE, C.BOLD))
            print(colored("    > ", C.DIM), end="")
            try:
                end_user_role = input().strip()
            except (EOFError, KeyboardInterrupt):
                end_user_role = ""

            print()
            print(colored("  Any other details about the end user? (optional)", C.WHITE, C.BOLD))
            print(colored("    > ", C.DIM), end="")
            try:
                end_user_desc = input().strip()
            except (EOFError, KeyboardInterrupt):
                end_user_desc = ""

        # As-is process
        print()
        print(colored("  How do you (or the end user) currently handle this?", C.WHITE, C.BOLD))
        print(colored("  Describe the as-is process, if any. (Type END on a new line when done)", C.DIM))
        as_is_lines = []
        while True:
            try:
                line = input(colored("    > ", C.DIM))
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip().upper() == "END":
                break
            as_is_lines.append(line)
        as_is_process = "\n".join(as_is_lines)

        # Any additional context
        print()
        print(colored("  Anything else we should know? (optional, Enter to skip)", C.WHITE, C.BOLD))
        print(colored("    > ", C.DIM), end="")
        try:
            additional = input().strip()
        except (EOFError, KeyboardInterrupt):
            additional = ""

        profile = UserProfile(
            name=name,
            role=role,
            company=company,
            is_request_for_self=is_for_self,
            end_user_name=end_user_name,
            end_user_role=end_user_role,
            end_user_description=end_user_desc,
            as_is_process=as_is_process,
            additional_context=additional,
        )

        # Confirm
        print()
        print(colored(f"  {border}", C.CYAN))
        requester = name or "Anonymous"
        if is_for_self:
            print(colored(f"  ✓ Got it, {requester}. You'll be the end user.", C.GREEN))
        else:
            eu = end_user_name or "the end user"
            print(colored(f"  ✓ Got it, {requester}. Building for {eu} ({end_user_role or 'role TBD'}).", C.GREEN))
        if as_is_process:
            print(colored("  ✓ As-is process captured.", C.GREEN))
        print(colored(f"  {border}", C.CYAN))
        print()

        return profile

    def knowledge_intake(self) -> list[str]:
        """Ask the user for external knowledge paths. Returns list of paths."""
        w = term_width()
        line = "─" * (w - 4)

        print()
        print(colored(f"  {line}", C.CYAN))
        print(colored("  📎 Knowledge & Reference Materials", C.BOLD, C.CYAN))
        print(colored(f"  {line}", C.CYAN))
        print()
        print(colored("  Do you have any reference materials to share?", C.WHITE, C.BOLD))
        print(colored("  These help the crew understand your domain, constraints,", C.DIM))
        print(colored("  and expectations better.", C.DIM))
        print()
        print(colored("  Examples:", C.DIM))
        print(colored("    • Business docs      (.md, .txt)", C.DIM))
        print(colored("    • API specs           (openapi.yaml, swagger.json)", C.DIM))
        print(colored("    • Database schemas    (.sql, .prisma)", C.DIM))
        print(colored("    • Sample data         (.csv, .json, .yaml)", C.DIM))
        print(colored("    • Test cases          (test_*.py)", C.DIM))
        print(colored("    • Reference code      (.py, .ts, .go, ...)", C.DIM))
        print(colored("    • Full folders        (/path/to/project/)", C.DIM))
        print(colored("    • Git repos           (https://github.com/user/repo)", C.DIM))
        print()
        print(colored("  Enter paths (comma-separated), or press Enter to skip:", C.WHITE, C.BOLD))
        print(colored("    > ", C.DIM), end="")

        try:
            raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""

        if not raw:
            print(colored("\n  ↳ No materials attached — that's fine!", C.DIM))
            return []

        # Split by comma, strip whitespace
        paths = [p.strip() for p in raw.split(",") if p.strip()]
        print(colored(f"\n  ↳ Got {len(paths)} path(s) to process.", C.GREEN))
        return paths

    def ingest_summary(self, items: list) -> None:
        """Display a summary of ingested knowledge items."""
        from hive.connectors import format_size

        w = term_width()
        line = "─" * (w - 4)
        print()
        print(colored(f"  {line}", C.DIM))
        print(colored("  📂 Ingested Knowledge:", C.BOLD, C.WHITE))

        # Group by source type
        type_icons = {
            "document":  "📄", "codebase":  "💻", "test_case": "🧪",
            "data_file": "📊", "api_spec":  "🔌", "schema":    "🗃️",
            "git_repo":  "📦",
        }
        for item in items:
            icon = type_icons.get(item.source_type, "📦")
            size = format_size(item.raw_size)
            status = "summarized" if item.was_summarized else "full"
            kind = item.source_type.upper().replace("_", " ")
            print(
                colored(f"    {icon} ", C.WHITE)
                + colored(f"{item.label:30}", C.BOLD)
                + colored(f" → {kind:12}", C.CYAN)
                + colored(f" ({size}, {status})", C.DIM)
            )

        total_size = sum(i.raw_size for i in items)
        large_count = sum(1 for i in items if i.was_summarized)
        print()
        print(colored(
            f"  ✅ {len(items)} item(s) loaded ({format_size(total_size)} total)"
            + (f" — {large_count} to be summarized by Scout" if large_count else ""),
            C.GREEN, C.BOLD,
        ))
        print(colored(f"  {line}", C.DIM))
        print()

    def repo_clone_summary(self, url: str, file_count: int, tree: str) -> None:
        """Display a summary after cloning a git repo."""
        w = term_width()
        line = "─" * (w - 4)
        print()
        print(colored(f"  {line}", C.DIM))
        print(colored(f"  📦 Reference Repo: {url}", C.BOLD, C.CYAN))
        print(colored(f"     {file_count} files ingested", C.GREEN))
        print()
        # Show first 20 lines of tree
        tree_lines = tree.splitlines()[:20]
        for tl in tree_lines:
            print(colored(f"     {tl}", C.DIM))
        if len(tree.splitlines()) > 20:
            print(colored(f"     ... ({len(tree.splitlines()) - 20} more entries)", C.DIM))
        print(colored(f"  {line}", C.DIM))
        print()

    def phase_header(self, phase: str) -> None:
        w = term_width()
        line = "─" * (w - 4)
        print()
        print(colored(f"  {line}", C.DIM))
        label = f"  ▸ PHASE: {phase.upper()}"
        print(colored(label, C.BOLD, C.CYAN))
        print(colored(f"  {line}", C.DIM))
        print()
        self._phase_start_time = time.time()

    def phase_footer(self, phase: str) -> None:
        elapsed = ""
        if self._phase_start_time:
            dt = time.time() - self._phase_start_time
            elapsed = f" ({dt:.1f}s)"
        print(colored(f"  ✓ {phase} complete{elapsed}", C.GREEN, C.BOLD))

    def signoff_prompt(
        self, artifact: str, content_preview: str,
        produced_by: str = "", reviewed_by: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Ask the user for sign-off on an artifact. Returns (approved, feedback)."""
        w = term_width()
        print()
        print(colored("  ┌" + "─" * (w - 6) + "┐", C.YELLOW))
        print(colored("  │", C.YELLOW) + colored(f"  SIGN-OFF REQUIRED: {artifact.upper()}", C.BOLD, C.YELLOW) + " " * max(0, w - 30 - len(artifact)) + colored("│", C.YELLOW))
        print(colored("  └" + "─" * (w - 6) + "┘", C.YELLOW))

        # Attribution
        if produced_by or reviewed_by:
            print()
            if produced_by:
                print(colored(f"    Produced by : {produced_by}", C.CYAN))
            if reviewed_by:
                print(colored(f"    Reviewed by : {', '.join(reviewed_by)}", C.CYAN))
        print()

        # Show preview (wrapped)
        preview_lines = content_preview.splitlines()[:20]
        for line in preview_lines:
            wrapped = textwrap.fill(line, width=w - 8)
            for wl in wrapped.splitlines():
                print(colored("    │ ", C.DIM) + wl)
        if len(content_preview.splitlines()) > 20:
            print(colored(f"    │ ... ({len(content_preview.splitlines())} total lines)", C.DIM))
        print()

        while True:
            print(colored("  👨‍💻 ", C.WHITE) + colored("Approve? ", C.BOLD) + colored("[y]es / [n]o + feedback: ", C.DIM), end="")
            try:
                resp = input().strip()
            except (EOFError, KeyboardInterrupt):
                return False, "Session interrupted"

            if resp.lower() in ("y", "yes", ""):
                return True, ""
            elif resp.lower().startswith(("n", "no")):
                feedback = resp[resp.find(" ") + 1:] if " " in resp else ""
                if not feedback:
                    print(colored("    Feedback (what should change): ", C.DIM), end="")
                    try:
                        feedback = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        feedback = "rejected"
                return False, feedback
            else:
                # Treat as feedback / change request
                return False, resp

    def user_input(self, prompt: str, multiline: bool = False) -> str:
        """Ask the user for freeform input."""
        print()
        print(colored(f"  👨‍💻 {prompt}", C.WHITE, C.BOLD))
        if multiline:
            print(colored("    (Enter your response. Type END on a new line when done)", C.DIM))
            lines = []
            while True:
                try:
                    line = input(colored("    > ", C.DIM))
                except (EOFError, KeyboardInterrupt):
                    break
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            return "\n".join(lines)
        else:
            print(colored("    > ", C.DIM), end="")
            try:
                return input().strip()
            except (EOFError, KeyboardInterrupt):
                return ""

    def answer_questions(self, questions: list[str]) -> dict[str, str]:
        """Present interview questions and collect answers."""
        print()
        print(colored("  📋 Penny has questions for you:", C.YELLOW, C.BOLD))
        print(colored("    (Answer each question, or type 'skip' to leave blank)", C.DIM))
        print()

        answers = {}
        for i, q in enumerate(questions, 1):
            print(colored(f"  Q{i}: ", C.YELLOW, C.BOLD) + q)
            print(colored("    > ", C.DIM), end="")
            try:
                ans = input().strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            if ans.lower() == "skip":
                ans = "(skipped)"
            answers[q] = ans
            print()

        return answers

    # ─────────────────────────────────────────────────────────────────────────
    #  Event rendering — process new events
    # ─────────────────────────────────────────────────────────────────────────

    def flush_events(self) -> None:
        """Render all unprocessed events."""
        while self._rendered_count < len(self.board.events):
            ev = self.board.events[self._rendered_count]
            self._render_event(ev)
            self._rendered_count += 1

    def _render_event(self, ev: Event) -> None:
        """Render a single event."""
        color = agent_color(ev.agent)
        emoji = agent_emoji(ev.agent)
        label = f"{emoji} {ev.agent:8}"

        if ev.type == EventType.WELCOME:
            print(colored(f"  {label}  {ev.content}", color, C.BOLD))

        elif ev.type == EventType.THINKING:
            # Only show first thinking msg per agent per phase (avoid spam)
            key = f"{ev.agent}:{self.board.current_phase}"
            if key not in self._thinking_shown or self.verbose:
                self._thinking_shown.add(key)
                print(colored(f"  {label}  💭 {ev.content}", color, C.DIM))

        elif ev.type == EventType.SPEAKING:
            # Truncate long speech for the terminal
            text = ev.content
            if len(text) > 200 and not self.verbose:
                text = text[:197] + "..."
            target_str = f" → {ev.target}" if ev.target else ""
            print(colored(f"  {label}{target_str}  {text}", color))

        elif ev.type == EventType.HANDSHAKE:
            print(colored(f"  {label}  🤝 {ev.content}", color, C.BOLD))

        elif ev.type == EventType.AGREEMENT:
            print(colored(f"  {label}  ✅ {ev.content}", C.GREEN))

        elif ev.type == EventType.DISAGREEMENT:
            print(colored(f"  {label}  ❌ {ev.content}", C.RED))

        elif ev.type == EventType.WRITING:
            print(colored(f"  {label}  ✍️  {ev.content}", color))

        elif ev.type == EventType.REVIEWING:
            print(colored(f"  {label}  🔎 {ev.content}", color))

        elif ev.type == EventType.VERDICT:
            verdict_color = C.GREEN if "PASS" in ev.content.upper() else C.RED
            print(colored(f"  {label}  📜 {ev.content}", verdict_color, C.BOLD))

        elif ev.type == EventType.CHECKPOINT:
            print(colored(f"  ⚙️  checkpoint  💾 {ev.content}", C.DIM))

        elif ev.type == EventType.USER_SIGNOFF:
            if "✅" in ev.content:
                print(colored(f"  👨‍💻 user      {ev.content}", C.GREEN, C.BOLD))
            else:
                print(colored(f"  👨‍💻 user      {ev.content}", C.RED, C.BOLD))

        elif ev.type == EventType.CREW_FORMED:
            print(colored(f"  ⚙️  system    {ev.content}", C.CYAN))

        elif ev.type == EventType.PHASE_START:
            pass  # handled by phase_header()

        elif ev.type == EventType.PHASE_END:
            pass  # handled by phase_footer()

        elif ev.type == EventType.ESCALATION:
            print(colored(f"  {label}  ⚡ {ev.content}", C.RED, C.BOLD))

        elif ev.type == EventType.LLM_INCIDENT:
            print(colored(f"  {label}  🔄 {ev.content}", C.YELLOW, C.DIM))

        elif ev.type == EventType.ERROR:
            print(colored(f"  {label}  💥 {ev.content}", C.RED))

        else:
            print(colored(f"  {label}  {ev.content}", color))

    # ─────────────────────────────────────────────────────────────────────────
    #  Progress indicators
    # ─────────────────────────────────────────────────────────────────────────

    def progress(self, current: int, total: int, label: str = "") -> None:
        w = max(term_width() - 20, 20)
        bar_w = min(40, w)
        filled = int(bar_w * current / max(total, 1))
        bar = "█" * filled + "░" * (bar_w - filled)
        pct = current / max(total, 1) * 100
        print(f"\r  [{bar}] {pct:5.1f}% {label}", end="", flush=True)
        if current >= total:
            print()  # newline when complete

    def file_status(self, filename: str, status: str, detail: str = "") -> None:
        """Show file-level status during build."""
        status_colors = {
            "building":  C.YELLOW,
            "reviewing": C.CYAN,
            "approved":  C.GREEN,
            "failed":    C.RED,
            "skipped":   C.DIM,
            "revising":  C.MAGENTA,
        }
        color = status_colors.get(status, C.WHITE)
        icon = {"building": "🔨", "reviewing": "🔎", "approved": "✅",
                "failed": "❌", "skipped": "⏭️", "revising": "🔄"}.get(status, "•")
        line = f"  {icon} {filename:30} {colored(status.upper(), color, C.BOLD)}"
        if detail:
            line += f"  {colored(detail, C.DIM)}"
        print(line)

    # ─────────────────────────────────────────────────────────────────────────
    #  Final summary
    # ─────────────────────────────────────────────────────────────────────────

    def final_summary(self) -> None:
        """Print the end-of-run summary."""
        w = term_width()
        b = self.board

        print()
        print(colored("═" * w, C.CYAN))
        print(colored("  DELIVERY SUMMARY", C.BOLD, C.CYAN))
        print(colored("═" * w, C.CYAN))
        print()

        # User
        if b.user_profile and b.user_profile.name:
            print(colored(f"  👤 Requester: {b.user_profile.name}"
                          + (f" ({b.user_profile.role})" if b.user_profile.role else ""),
                          C.WHITE, C.BOLD))
            if not b.user_profile.is_request_for_self and b.user_profile.end_user_name:
                print(colored(f"     End user : {b.user_profile.end_user_name}"
                              + (f" ({b.user_profile.end_user_role})" if b.user_profile.end_user_role else ""),
                              C.DIM))
            print()

        # Files
        approved = [e for e in b.registry.values() if e.approved]
        skipped = [e for e in b.registry.values() if e.skip_reason]
        total = len(b.registry)
        print(colored(f"  📦 Files: {len(approved)}/{total} approved", C.GREEN, C.BOLD)
              + (f", {len(skipped)} skipped" if skipped else ""))

        for e in approved:
            dev_note = f" (by {e.assigned_dev})" if e.assigned_dev else ""
            deferred_note = f" [{len(e.deferred_issues)} deferred]" if e.deferred_issues else ""
            print(f"    ✅ {e.name}{dev_note}{deferred_note}")
        for e in skipped:
            print(colored(f"    ⏭️  {e.name}: {e.skip_reason}", C.DIM))

        # Knowledge base
        if b.knowledge_base:
            from hive.connectors import format_size
            total_kb_size = sum(i.raw_size for i in b.knowledge_base)
            types = set(i.source_type for i in b.knowledge_base)
            print()
            print(colored(
                f"  📎 Knowledge: {len(b.knowledge_base)} items "
                f"({format_size(total_kb_size)}) — "
                f"types: {', '.join(sorted(types))}",
                C.CYAN,
            ))

        # Reference repos
        if b.repo_urls:
            print()
            print(colored(f"  📦 Reference repos: {len(b.repo_urls)}", C.CYAN))
            for url in b.repo_urls:
                print(colored(f"    → {url}", C.DIM))
            if b.repo_analysis:
                lines = len(b.repo_analysis.splitlines())
                print(colored(f"    Scout analysis: {lines} lines", C.DIM))

        # Sign-off log with attribution
        if b.signoffs:
            print()
            print(colored("  📝 Sign-off Log:", C.YELLOW, C.BOLD))
            for so in b.signoffs:
                status = "✅ Approved" if so.approved else "❌ Rejected"
                line = f"    {status}: {so.artifact} v{so.version}"
                if so.produced_by:
                    line += f"  (by {so.produced_by})"
                if so.reviewed_by:
                    line += f"  [reviewed: {', '.join(so.reviewed_by)}]"
                print(colored(line, C.GREEN if so.approved else C.RED))

        # Deferred
        if b.all_deferred:
            print()
            print(colored(f"  ⚠️  Deferred issues: {len(b.all_deferred)}", C.YELLOW))
            for fname, issue in b.all_deferred[:5]:
                print(colored(f"    [{issue.severity}] {fname}: {issue.description}", C.DIM))
            if len(b.all_deferred) > 5:
                print(colored(f"    ... and {len(b.all_deferred) - 5} more", C.DIM))

        # Amendments
        if b.amendments:
            print()
            print(colored(f"  📝 Contract amendments: {len(b.amendments)}", C.MAGENTA))
            for a in b.amendments:
                print(colored(f"    • [{a.requested_by}] {a.description}", C.DIM))

        # Verdicts
        print()
        if b.integration_verdict:
            vcolor = C.GREEN if "PASS" in b.integration_verdict.upper() else C.RED
            print(colored(f"  🧪 Integration: {b.integration_verdict}", vcolor, C.BOLD))
        if b.release_verdict:
            print(colored(f"  🚀 Release: {b.release_verdict[:100]}", C.GREEN, C.BOLD))

        # Project location
        if b.project_slug:
            print()
            print(colored(f"  📂 Project files: {b.project_root}", C.CYAN))

        # Memory summary (if memory_stats are set by the crew)
        memory_stats = getattr(b, '_memory_stats', None)
        if memory_stats:
            print()
            print(colored("  🧠 Memory:", C.MAGENTA, C.BOLD))
            agent_mems = memory_stats.get("agent_memories", {})
            if agent_mems:
                agents_with = [(a, c) for a, c in agent_mems.items() if c > 0]
                if agents_with:
                    print(f"    Agent memories: " + ", ".join(
                        f"{a} ({c})" for a, c in sorted(agents_with)
                    ))
            team_count = memory_stats.get("team_entries", 0)
            if team_count:
                print(f"    Team insights : {team_count}")
            global_count = memory_stats.get("global_lessons", 0)
            if global_count:
                print(f"    Global lessons: {global_count}")
            total = memory_stats.get("total", 0)
            if total:
                print(colored(f"    Total entries : {total}", C.DIM))

        # Logbook summary
        if b.logbook:
            print()
            total_calls = len(b.logbook)
            total_retries = sum(e.retries for e in b.logbook)
            total_escalations = sum(1 for e in b.logbook if e.tier_escalated)
            total_thinking_strips = sum(1 for e in b.logbook if e.thinking_stripped)
            total_failures = sum(1 for e in b.logbook if not e.success)
            total_input = sum(e.input_tokens for e in b.logbook)
            total_output = sum(e.output_tokens for e in b.logbook)
            total_duration = sum(e.duration_s for e in b.logbook)

            print(colored("  📓 Logbook Summary:", C.CYAN, C.BOLD))
            print(f"    LLM calls     : {total_calls}")
            print(f"    Total tokens  : {total_input:,} in / {total_output:,} out")
            print(f"    Total time    : {total_duration:.1f}s")
            if total_retries:
                print(colored(f"    Retries       : {total_retries}", C.YELLOW))
            if total_escalations:
                print(colored(f"    Tier escalated: {total_escalations}x", C.YELLOW))
            if total_thinking_strips:
                print(colored(f"    Thinking strip: {total_thinking_strips}x", C.YELLOW))
            if total_failures:
                print(colored(f"    Failed calls  : {total_failures}", C.RED))

            # Model usage breakdown
            models_used: dict[str, int] = {}
            for e in b.logbook:
                if e.success:
                    models_used[e.model_used] = models_used.get(e.model_used, 0) + 1
            if models_used:
                print("    Models used   : " + ", ".join(
                    f"{m} ({c}x)" for m, c in sorted(models_used.items(), key=lambda x: -x[1])
                ))

            # Per-agent breakdown
            agent_calls: dict[str, list] = {}
            for e in b.logbook:
                agent_calls.setdefault(e.agent_name, []).append(e)
            print()
            print(colored("    Per-agent breakdown:", C.DIM))
            for agent_name, entries in sorted(agent_calls.items()):
                calls = len(entries)
                tokens = sum(e.input_tokens + e.output_tokens for e in entries)
                dur = sum(e.duration_s for e in entries)
                retries = sum(e.retries for e in entries)
                tier_set = {e.tier_used for e in entries if e.success}
                tier_str = "/".join(sorted(tier_set)) if tier_set else "?"
                line = f"      {agent_name:12} {calls:3} calls  {tokens:>8,} tok  {dur:6.1f}s  tier={tier_str}"
                if retries:
                    line += colored(f"  ({retries} retries)", C.YELLOW)
                print(line)

            print(colored(f"\n    Full logbook: {b.docs_dir / 'logbook.json'}", C.DIM))

        print()
        print(colored("═" * w, C.CYAN))
        print()
