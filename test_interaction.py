"""
Simulate a user interacting with hive — including mid-flow change requests.

Uses pattern-matching against hive's stdout to send the right response at
the right moment, rather than pre-queuing a fixed number of answers.

Scenario
────────
Feature      : "Build a simple note-taking CLI"
Welcome      : name=Alex, role=Developer, for myself
Interview    : answer every question with sensible defaults
PRD sign-off : REJECT first draft (add search command)
PRD v2       : APPROVE
Feasibility  : APPROVE
Architecture : REJECT (split storage into its own module)
Arch v2      : APPROVE
Build        : runs unattended

Run:
    python3 test_interaction.py
    python3 test_interaction.py --fast   # no delays
"""

import argparse
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


# ── Response rules ─────────────────────────────────────────────────────────
# Each rule: if any pattern matches a line just printed, send `reply`.
# Rules are checked in order; first match wins.
# `once=True` means the rule fires only once (prevent re-triggering).

@dataclass
class Rule:
    patterns: list[str]          # substrings to match in recent output
    reply: str                   # what to send
    label: str                   # human-readable description
    once: bool = True            # fire only once
    fired: int = 0               # how many times this rule has fired


RULES: list[Rule] = [
    # ── Welcome intake ────────────────────────────────────────────────────
    Rule(["What's your name?"],                         "Alex",         "name"),
    Rule(["What's your role?"],                         "Developer",    "role"),
    Rule(["Company / organization?"],                   "",             "company (skip)"),
    Rule(["For myself", "For someone else"],            "1",            "for myself"),
    Rule(["as-is process", "Type END"],                 "END",          "as-is (skip)"),
    Rule(["Anything else we should know"],              "",             "extra context (skip)"),

    # ── Knowledge ingest ─────────────────────────────────────────────────
    Rule(["Enter paths (comma-separated)"],             "",             "no attachments"),

    # ── Interview — answer every question with a default ─────────────────
    # Fires once per "Q<n>:" line (up to 8 questions)
    Rule(["Q1:"],   "one-shot CLI, each command exits",     "Q1", once=False),
    Rule(["Q2:"],   "title and body, creation date auto",   "Q2", once=False),
    Rule(["Q3:"],   "plain text files in ~/.notes/",        "Q3", once=False),
    Rule(["Q4:"],   "case-insensitive search",              "Q4", once=False),
    Rule(["Q5:"],   "error message if note not found",      "Q5", once=False),
    Rule(["Q6:"],   "skip",                                 "Q6", once=False),
    Rule(["Q7:"],   "skip",                                 "Q7", once=False),
    Rule(["Q8:"],   "skip",                                 "Q8", once=False),

    # ── PRD sign-offs ────────────────────────────────────────────────────
    # The "Approve?" prompt uses print(..., end="") so readline never sees it.
    # Match on the sign-off box header line (which does end with \n), then
    # the driver waits briefly and sends the response.
    Rule(
        ["SIGN-OFF REQUIRED: PRD"],
        "n add a 'search' command that finds notes by keyword in title or body",
        "PRD v1 REJECT (add search)",
    ),
    Rule(
        ["SIGN-OFF REQUIRED: PRD"],
        "y",
        "PRD v2 APPROVE",
        once=False,
    ),

    # ── Feasibility ──────────────────────────────────────────────────────
    Rule(
        ["SIGN-OFF REQUIRED: FEASIBILITY"],
        "y",
        "Feasibility APPROVE",
    ),

    # ── Architecture sign-offs ───────────────────────────────────────────
    Rule(
        ["SIGN-OFF REQUIRED: ARCHITECTURE"],
        "n please put all file I/O in a dedicated storage.py module, keep commands thin",
        "Arch v1 REJECT (storage module)",
    ),
    Rule(
        ["SIGN-OFF REQUIRED: ARCHITECTURE"],
        "y",
        "Arch v2 APPROVE",
        once=False,
    ),
]


# ── Driver ─────────────────────────────────────────────────────────────────

class SimDriver:
    def __init__(self, proc: subprocess.Popen, fast: bool = False) -> None:
        self.proc = proc
        self.fast = fast
        self.window: list[str] = []   # rolling window of recent output lines
        self.lock = threading.Lock()

    def _matches(self, rule: Rule) -> bool:
        """All patterns must appear somewhere in the recent window."""
        combined = "\n".join(self.window[-30:])
        return all(p in combined for p in rule.patterns)

    def _try_fire(self, line: str) -> None:
        """Check rules against updated window; send response if matched."""
        with self.lock:
            self.window.append(line)

        for rule in RULES:
            if rule.once and rule.fired >= 1:
                continue
            if not self._matches(rule):
                continue

            # Matched — fire
            rule.fired += 1
            # Sign-off headers need extra time: hive prints the box, then
            # the Approve? prompt (no newline), then blocks on input.
            is_signoff = "SIGN-OFF" in " ".join(rule.patterns)
            delay = (1.5 if is_signoff else 0.3) if self.fast else (3.0 if is_signoff else 1.5)
            time.sleep(delay)

            tag = f"\033[33m  [sim] → {rule.label}: {repr(rule.reply) if rule.reply else '(enter)'}\033[0m"
            print(tag, flush=True)

            assert self.proc.stdin is not None
            try:
                self.proc.stdin.write((rule.reply + "\n").encode())
                self.proc.stdin.flush()
            except BrokenPipeError:
                pass

            # Clear window after firing so next prompt doesn't re-trigger
            with self.lock:
                self.window.clear()
            break

    def run(self) -> int:
        assert self.proc.stdout is not None

        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip("\n")
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            self._try_fire(line)

        self.proc.wait(timeout=30)
        return self.proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Minimal delays")
    args = parser.parse_args()

    print("=" * 70)
    print("  HIVE INTERACTION SIMULATION")
    print("  Rejecting PRD v1 (add search) + Arch v1 (add storage module)")
    print(f"  Mode: {'fast' if args.fast else 'realistic'}")
    print("=" * 70)
    print()

    proc = subprocess.Popen(
        [
            sys.executable, "run_hive.py",
            "Build a simple note-taking CLI — create, list, view, and delete "
            "notes stored as local text files",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    driver = SimDriver(proc, fast=args.fast)
    rc = driver.run()

    print()
    print("=" * 70)
    print(f"  [sim] finished — exit code {rc}")
    print("=" * 70)
    return rc


if __name__ == "__main__":
    sys.exit(main())
