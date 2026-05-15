"""
Hive Telemetry — Cost tracking, token budgets, and run analytics.

Tracks every LLM call's estimated cost, enforces budget limits, and provides
per-phase / per-agent cost rollups for the delivery summary.

Cost estimation uses published token pricing. Because pricing changes,
users can override via HIVE_COST_PER_1K_INPUT / HIVE_COST_PER_1K_OUTPUT.

Environment variables:
  HIVE_BUDGET_USD       — max USD spend per run (default: 0 = unlimited)
  HIVE_COST_PER_1K_INPUT   — override $/1K input tokens (default: model-based)
  HIVE_COST_PER_1K_OUTPUT  — override $/1K output tokens (default: model-based)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("hive.telemetry")

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

BUDGET_USD = float(os.environ.get("HIVE_BUDGET_USD", "0"))  # 0 = unlimited

# Override pricing ($/1K tokens). 0 = use model-based lookup.
COST_OVERRIDE_INPUT = float(os.environ.get("HIVE_COST_PER_1K_INPUT", "0"))
COST_OVERRIDE_OUTPUT = float(os.environ.get("HIVE_COST_PER_1K_OUTPUT", "0"))

# ─────────────────────────────────────────────────────────────────────────────
#  Model pricing table ($/1K tokens) — sourced from published pricing as of 2025
# ─────────────────────────────────────────────────────────────────────────────

# (input_per_1k, output_per_1k, cache_read_per_1k)
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # Anthropic Claude
    "claude-sonnet-4-20250514":     (0.003,  0.015,  0.0003),
    "claude-4-sonnet":              (0.003,  0.015,  0.0003),
    "claude-3-5-sonnet-20241022":   (0.003,  0.015,  0.0003),
    "claude-3-5-sonnet-20240620":   (0.003,  0.015,  0.0003),
    "claude-3-5-haiku-20241022":    (0.0008, 0.004,  0.00008),
    "claude-3-haiku-20240307":      (0.00025, 0.00125, 0.000025),
    "claude-3-opus-20240229":       (0.015,  0.075,  0.0015),
    "claude-opus-4-20250514":       (0.015,  0.075,  0.0015),
    # OpenAI GPT
    "gpt-4o":                       (0.0025, 0.01,   0.00125),
    "gpt-4o-mini":                  (0.00015, 0.0006, 0.000075),
    "gpt-4o-2024-11-20":            (0.0025, 0.01,   0.00125),
    "gpt-4-turbo":                  (0.01,   0.03,   0.005),
    "gpt-4":                        (0.03,   0.06,   0.03),
    "gpt-3.5-turbo":                (0.0005, 0.0015, 0.00025),
    "o1":                           (0.015,  0.06,   0.0075),
    "o1-mini":                      (0.003,  0.012,  0.0015),
    "o3-mini":                      (0.0011, 0.0044, 0.00055),
    # DeepSeek
    "deepseek-chat":                (0.00014, 0.00028, 0.00007),
    "deepseek-coder":               (0.00014, 0.00028, 0.00007),
    # Google Gemini
    "gemini-1.5-pro":               (0.00125, 0.005, 0.000315),
    "gemini-1.5-flash":             (0.000075, 0.0003, 0.0000375),
    "gemini-2.0-flash":             (0.0001, 0.0004, 0.00005),
}

# Fallback pricing when model isn't in the table
DEFAULT_PRICING = (0.003, 0.015, 0.0003)  # conservative Claude Sonnet pricing


# ─────────────────────────────────────────────────────────────────────────────
#  Model context windows
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic
    "claude-sonnet-4-20250514":     200_000,
    "claude-4-sonnet":              200_000,
    "claude-3-5-sonnet-20241022":   200_000,
    "claude-3-5-sonnet-20240620":   200_000,
    "claude-3-5-haiku-20241022":    200_000,
    "claude-3-haiku-20240307":      200_000,
    "claude-3-opus-20240229":       200_000,
    "claude-opus-4-20250514":       200_000,
    # OpenAI
    "gpt-4o":                       128_000,
    "gpt-4o-mini":                  128_000,
    "gpt-4o-2024-11-20":            128_000,
    "gpt-4-turbo":                  128_000,
    "gpt-4":                        8_192,
    "gpt-3.5-turbo":                16_385,
    "o1":                           200_000,
    "o1-mini":                      128_000,
    "o3-mini":                      200_000,
    # DeepSeek
    "deepseek-chat":                64_000,
    "deepseek-coder":               64_000,
    # Google Gemini
    "gemini-1.5-pro":               2_000_000,
    "gemini-1.5-flash":             1_000_000,
    "gemini-2.0-flash":             1_000_000,
}

DEFAULT_CONTEXT_WINDOW = 128_000


def model_context_window(model: str) -> int:
    """Get the context window size (in tokens) for a model.

    Uses substring matching to handle model versions and prefixes
    (e.g., 'anthropic--claude-4-sonnet' matches 'claude-4-sonnet').
    """
    # Direct match
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    # Substring match (handles proxy prefixes like 'anthropic--')
    model_lower = model.lower()
    for key, window in MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower:
            return window
    return DEFAULT_CONTEXT_WINDOW


# ─────────────────────────────────────────────────────────────────────────────
#  Cost calculation
# ─────────────────────────────────────────────────────────────────────────────

def _get_pricing(model: str) -> tuple[float, float, float]:
    """Get (input_per_1k, output_per_1k, cache_read_per_1k) for a model."""
    if COST_OVERRIDE_INPUT > 0:
        return (
            COST_OVERRIDE_INPUT,
            COST_OVERRIDE_OUTPUT or COST_OVERRIDE_INPUT * 5,
            COST_OVERRIDE_INPUT * 0.1,
        )
    # Direct match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Substring match
    model_lower = model.lower()
    for key, pricing in MODEL_PRICING.items():
        if key in model_lower:
            return pricing
    return DEFAULT_PRICING


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate USD cost for a single LLM call."""
    inp_rate, out_rate, cache_rate = _get_pricing(model)
    cost = (
        (input_tokens / 1000) * inp_rate
        + (output_tokens / 1000) * out_rate
        + (cache_read_tokens / 1000) * cache_rate
    )
    return round(cost, 6)


# ─────────────────────────────────────────────────────────────────────────────
#  Budget enforcement
# ─────────────────────────────────────────────────────────────────────────────

class BudgetExceeded(RuntimeError):
    """Raised when the run's estimated cost exceeds the configured budget."""

    def __init__(self, spent: float, budget: float):
        self.spent = spent
        self.budget = budget
        super().__init__(
            f"Budget exceeded: ${spent:.4f} spent of ${budget:.2f} limit. "
            f"Set HIVE_BUDGET_USD=0 to disable budget enforcement."
        )


@dataclass
class PhaseMetrics:
    """Metrics for a single pipeline phase."""
    phase: str
    start_time: float = 0.0
    end_time: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    errors: int = 0

    @property
    def duration_s(self) -> float:
        """Wall-clock duration in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class CostTracker:
    """Tracks cumulative cost and per-phase metrics for a run.

    Usage:
        tracker = CostTracker()
        tracker.start_phase("research")
        # ... make LLM calls ...
        tracker.record_call(model, input_tokens, output_tokens, cache_read_tokens)
        tracker.end_phase()
        print(tracker.summary())
    """
    budget_usd: float = BUDGET_USD
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_calls: int = 0
    run_start: float = field(default_factory=time.time)
    phase_metrics: list[PhaseMetrics] = field(default_factory=list)
    _current_phase: PhaseMetrics | None = field(default=None, repr=False)

    def start_phase(self, phase: str) -> None:
        """Begin tracking a new phase."""
        self._current_phase = PhaseMetrics(phase=phase, start_time=time.time())

    def end_phase(self) -> PhaseMetrics | None:
        """Finish the current phase and archive its metrics."""
        if self._current_phase:
            self._current_phase.end_time = time.time()
            self.phase_metrics.append(self._current_phase)
            result = self._current_phase
            self._current_phase = None
            return result
        return None

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        retries: int = 0,
        success: bool = True,
    ) -> float:
        """Record a single LLM call. Returns estimated cost.

        Raises BudgetExceeded if the cumulative cost exceeds budget_usd.
        """
        cost = estimate_cost(model, input_tokens, output_tokens, cache_read_tokens)
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_read_tokens += cache_read_tokens
        self.total_calls += 1

        # Update current phase metrics
        if self._current_phase:
            self._current_phase.llm_calls += 1
            self._current_phase.input_tokens += input_tokens
            self._current_phase.output_tokens += output_tokens
            self._current_phase.cache_read_tokens += cache_read_tokens
            self._current_phase.cost_usd += cost
            self._current_phase.retries += retries
            if not success:
                self._current_phase.errors += 1

        # Budget enforcement
        if self.budget_usd > 0 and self.total_cost > self.budget_usd:
            raise BudgetExceeded(self.total_cost, self.budget_usd)

        return cost

    @property
    def run_duration(self) -> float:
        """Elapsed run time in seconds."""
        return time.time() - self.run_start

    @property
    def cost_per_minute(self) -> float:
        """Current burn rate in USD/minute."""
        elapsed = self.run_duration
        return (self.total_cost / elapsed * 60) if elapsed > 0 else 0.0

    def phase_summary(self) -> list[dict]:
        """Return per-phase cost breakdown for display."""
        results = []
        for pm in self.phase_metrics:
            results.append({
                "phase": pm.phase,
                "calls": pm.llm_calls,
                "tokens": pm.total_tokens,
                "cost": pm.cost_usd,
                "duration": pm.duration_s,
                "retries": pm.retries,
                "errors": pm.errors,
            })
        return results

    def budget_remaining(self) -> float | None:
        """USD remaining before budget cap. None if unlimited."""
        if self.budget_usd <= 0:
            return None
        return max(0.0, self.budget_usd - self.total_cost)

    def budget_pct(self) -> float | None:
        """Percentage of budget consumed. None if unlimited."""
        if self.budget_usd <= 0:
            return None
        return min(100.0, (self.total_cost / self.budget_usd) * 100)
