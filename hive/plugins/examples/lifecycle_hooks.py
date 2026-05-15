"""
Example: Lifecycle Hooks Plugin

Demonstrates how to hook into pipeline phases for custom actions.
This example logs phase timing — replace with your own hooks
(Slack notifications, CI triggers, metric dashboards, etc.)

Usage:
  hive --plugin ./hive/plugins/examples/lifecycle_hooks.py "Build a ..."
"""

from __future__ import annotations

import time

from hive.plugins.base import PluginContext, PluginMeta


class TimingPlugin:
    """Tracks phase timing as a lifecycle hook example.

    A real implementation might:
      - Post to Slack/Teams when a phase starts/ends
      - Trigger CI/CD pipelines after build
      - Report metrics to Datadog/Grafana
      - Send email notifications on completion
    """

    meta = PluginMeta(
        name="phase-timer",
        version="0.1.0",
        description="Tracks pipeline phase durations (lifecycle hook example)",
        author="Hive Examples",
        category="lifecycle",
    )

    def __init__(self) -> None:
        self._phase_starts: dict[str, float] = {}
        self.timings: dict[str, float] = {}

    def on_phase_start(self, phase: str, ctx: PluginContext) -> None:
        """Record phase start time."""
        self._phase_starts[phase] = time.time()

    def on_phase_end(self, phase: str, ctx: PluginContext) -> None:
        """Record phase duration."""
        start = self._phase_starts.pop(phase, None)
        if start:
            self.timings[phase] = time.time() - start
