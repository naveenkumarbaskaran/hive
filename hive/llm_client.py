"""
Pluggable LLM Client — drop-in connector for any LLM backend.

Supports:
  - Anthropic native (SDK with streaming, extended thinking, prompt caching)
  - OpenAI-compatible (LiteLLM, Ollama, vLLM, Azure OpenAI, any /v1/chat/completions)
  - SAP Hyperspace / LiteLLM proxy (auto-detected via /anthropic/v1/messages)

Auto-detects API format on first call. Override with LLM_FORMAT env var.

Model tiers:
  Agents request a capability tier, not a specific model name.
  The client resolves tiers to concrete models from env/config:
    FAST      → LLM_MODEL_SMALL  (classification, short JSON, quick checks)
    BALANCED  → LLM_MODEL        (PRD writing, moderate reasoning)
    POWERFUL  → LLM_MODEL_BIG    (architecture, code gen, deep reasoning)

Usage:
  from llm_client import llm, ModelTier   # singleton + tier enum

  # Agents request by tier:
  resp = llm.chat(system="...", messages=[...], tier=ModelTier.POWERFUL)

  # Or override with an explicit model name:
  resp = llm.chat(system="...", messages=[...], model="gpt-4o")

  # Or construct your own client:
  from llm_client import LLMClient
  client = LLMClient(base_url="...", api_key="...", default_model="...")

Environment variables:
  LLM_BASE_URL    — endpoint (default: https://api.anthropic.com)
  LLM_API_KEY     — API key / token
  LLM_MODEL       — default / balanced model name
  LLM_MODEL_BIG   — model for heavy reasoning (default: same as LLM_MODEL)
  LLM_MODEL_SMALL — model for light tasks (default: same as LLM_MODEL)
  LLM_FORMAT      — force format: "anthropic" | "openai" | "auto" (default: auto)
  LLM_FALLBACK_MODELS — comma-separated fallback models for 429 rotation

  Per-tier provider overrides (multi-provider routing):
  LLM_BASE_URL_FAST     — endpoint for FAST tier (e.g. OpenAI for cheap tasks)
  LLM_BASE_URL_POWERFUL — endpoint for POWERFUL tier (e.g. Anthropic for reasoning)
  LLM_API_KEY_FAST      — API key for FAST tier endpoint
  LLM_API_KEY_POWERFUL  — API key for POWERFUL tier endpoint
  LLM_FORMAT_FAST       — force format for FAST tier endpoint
  LLM_FORMAT_POWERFUL   — force format for POWERFUL tier endpoint
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from hive.hardening import backoff_wait as _backoff_wait

logger = logging.getLogger("hive.llm")


# ─────────────────────────────────────────────────────────────────────────────
#  Thread-safe request pacer — prevents concurrent threads from flooding
# ─────────────────────────────────────────────────────────────────────────────


class _RequestPacer:
    """Thread-safe pacer that enforces a minimum interval between LLM requests.

    When multiple threads call ``pace()`` concurrently, each one waits its turn
    so requests are serialized with at least ``min_interval_ms`` between them.
    This prevents N parallel build threads from sending N simultaneous requests
    to a rate-limited endpoint.
    """

    def __init__(self, min_interval_ms: int = 0) -> None:
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = min_interval_ms / 1000.0

    @property
    def interval_ms(self) -> int:
        return int(self._min_interval * 1000)

    @interval_ms.setter
    def interval_ms(self, value: int) -> None:
        self._min_interval = value / 1000.0

    def pace(self) -> float:
        """Block until enough time has passed since the last call.

        Returns the actual wait time in seconds (0 if no wait was needed).
        """
        if self._min_interval <= 0:
            return 0.0
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                wait = self._min_interval - elapsed
                time.sleep(wait)
            else:
                wait = 0.0
            self._last_call = time.time()
            return wait

    def backoff(self, seconds: float) -> None:
        """Temporarily widen the pacing interval after a rate-limit hit.

        Sets ``_last_call`` into the future so the *next* ``pace()`` call by
        any thread will honour the server-requested ``Retry-After`` delay.
        The base ``min_interval`` is unchanged — once the extra pause is
        consumed the pacer returns to its normal cadence.
        """
        with self._lock:
            future = time.time() + seconds - self._min_interval
            if future > self._last_call:
                self._last_call = future


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After seconds from a 429 response, if available.

    Checks the exception's response for the ``Retry-After`` header
    and the Anthropic-specific ``retry-after`` body field.
    Returns seconds to wait, or None if not found.
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return None

    # Check standard Retry-After header
    header_val = None
    if hasattr(resp, "headers"):
        header_val = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if header_val:
        try:
            return float(header_val)
        except (ValueError, TypeError):
            pass

    # Check Anthropic-style JSON body: {"error": {"type": "rate_limit_error", ...}}
    try:
        body = resp.json() if hasattr(resp, "json") else {}
        err = body.get("error", {})
        if isinstance(err, dict):
            # Some proxies include retry_after in the error body
            ra = err.get("retry_after") or err.get("retry-after")
            if ra is not None:
                return float(ra)
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Model tiers — agents request capability, not a specific model
# ─────────────────────────────────────────────────────────────────────────────


class ModelTier(str, Enum):
    """Capability tiers that agents request. Resolved to concrete models at runtime."""

    FAST = "fast"  # classification, short JSON, quick checks
    BALANCED = "balanced"  # PRD writing, moderate reasoning, summaries
    POWERFUL = "powerful"  # architecture, code generation, deep reasoning

    def escalate(self) -> ModelTier:
        """Bump to the next tier (e.g., after a failed attempt)."""
        order = [ModelTier.FAST, ModelTier.BALANCED, ModelTier.POWERFUL]
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]


# ─────────────────────────────────────────────────────────────────────────────
#  Response wrapper
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Unified response from any backend."""

    text: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str = ""
    raw: Any = None
    # ── Resilience metadata ──
    tier_requested: str = ""  # tier the caller asked for
    tier_used: str = ""  # tier actually used (may be escalated)
    model_requested: str = ""  # model resolved from requested tier
    retries: int = 0  # how many retries before success
    tier_escalated: bool = False  # was tier bumped?
    thinking_stripped: bool = False  # was thinking removed on fallback?
    model_switched: bool = False  # was model switched due to rate limit?
    model_used: str = ""  # actual model used (may differ from requested)
    errors: list[str] = field(default_factory=list)  # error msgs from failed attempts
    duration_s: float = 0.0  # wall-clock seconds


# ─────────────────────────────────────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────────────────────────────────────


class LLMClient:
    """
    Pluggable LLM connector. Users can swap backends by changing env vars —
    no code changes needed.
    """

    ANTHROPIC_NATIVE = "anthropic"  # direct Anthropic SDK (streaming, thinking, cache)
    ANTHROPIC_PROXY = "anthropic_proxy"  # /anthropic/v1/messages (Hyperspace / LiteLLM)
    OPENAI_COMPAT = "openai"  # /v1/chat/completions

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        model_big: str | None = None,
        model_small: str | None = None,
        api_format: str | None = None,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.anthropic.com")).rstrip(
            "/"
        )
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.default_model = default_model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
        self.model_big = model_big or os.getenv("LLM_MODEL_BIG", "") or self.default_model
        self.model_small = model_small or os.getenv("LLM_MODEL_SMALL", "") or self.default_model
        self.fallback_models: list[str] = self._parse_fallback_models()
        self.http_timeout: int = int(os.getenv("HIVE_LLM_TIMEOUT", "120"))
        self._format: str | None = api_format or os.getenv("LLM_FORMAT", "").lower() or None
        if self._format == "auto":
            self._format = None  # auto = run detection
        self._anthropic_client: Any = None  # lazy-loaded SDK client
        self._pacer = _RequestPacer(int(os.getenv("HIVE_REQUEST_PACE_MS", "200")))

        # ── Per-tier provider overrides ──────────────────────────────────
        # Allow different providers per tier: LLM_BASE_URL_FAST, LLM_API_KEY_BIG, etc.
        self._tier_config: dict[str, dict[str, str]] = {}
        for tier_name in ("fast", "balanced", "powerful"):
            suffix = f"_{tier_name.upper()}"
            tier_url = os.getenv(f"LLM_BASE_URL{suffix}", "")
            tier_key = os.getenv(f"LLM_API_KEY{suffix}", "")
            tier_fmt = os.getenv(f"LLM_FORMAT{suffix}", "")
            if tier_url or tier_key:
                self._tier_config[tier_name] = {
                    "base_url": tier_url.rstrip("/") if tier_url else self.base_url,
                    "api_key": tier_key or self.api_key,
                    "format": tier_fmt.lower() if tier_fmt else "",
                }

        # Thread-local for per-call endpoint overrides (thread-safe multi-provider)
        self._local = threading.local()
        # Cache detected formats per-URL so multi-provider doesn't re-probe
        self._format_cache: dict[str, str] = {}

    # ── Fallback model helpers ─────────────────────────────────────────────────

    def _parse_fallback_models(self) -> list[str]:
        """Parse LLM_FALLBACK_MODELS env var (comma-separated) into a list."""
        env_val = os.getenv("LLM_FALLBACK_MODELS", "")
        if env_val:
            return [m.strip() for m in env_val.split(",") if m.strip()]
        return []

    def _build_model_pool(self, primary_model: str) -> list[str]:
        """Build a de-duplicated ordered list of models for 429 rotation.

        Order: primary → other tier models → explicit fallbacks.
        """
        candidates = [
            primary_model,
            self.default_model,
            self.model_big,
            self.model_small,
            *self.fallback_models,
        ]
        seen: set[str] = set()
        pool: list[str] = []
        for m in candidates:
            if m and m not in seen:
                seen.add(m)
                pool.append(m)
        return pool

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check if an exception is a 429 rate-limit error."""
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 429
        # Also catch stringified 429 from proxies — word boundary avoids
        # false positives on port numbers like 4290 or ticket IDs like #4291.
        return bool(re.search(r"\b429\b", str(exc)[:80]))

    @property
    def _effective_url(self) -> str:
        """Thread-local base URL override, or the default."""
        return getattr(self._local, "url", None) or self.base_url

    @property
    def _effective_key(self) -> str:
        """Thread-local API key override, or the default."""
        return getattr(self._local, "key", None) or self.api_key

    def _set_effective_endpoint(self, url: str, key: str) -> None:
        """Set thread-local endpoint for the current call."""
        self._local.url = url
        self._local.key = key

    def _clear_effective_endpoint(self) -> None:
        """Clear thread-local endpoint after a call."""
        self._local.url = None
        self._local.key = None

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve_model(self, tier: ModelTier) -> str:
        """Resolve a capability tier to a concrete model name."""
        return {
            ModelTier.FAST: self.model_small,
            ModelTier.BALANCED: self.default_model,
            ModelTier.POWERFUL: self.model_big,
        }[tier]

    def resolve_endpoint(self, tier: ModelTier) -> tuple[str, str, str | None]:
        """Resolve a tier to its ``(base_url, api_key, format_hint)``.

        Per-tier overrides are read from ``LLM_BASE_URL_FAST``,
        ``LLM_API_KEY_POWERFUL``, ``LLM_FORMAT_BALANCED``, etc.
        Falls back to the default endpoint when no override is set.
        """
        cfg = self._tier_config.get(tier.value)
        if cfg:
            fmt = cfg["format"] or None
            if fmt == "auto":
                fmt = None
            return cfg["base_url"], cfg["api_key"], fmt
        return self.base_url, self.api_key, None

    def chat(
        self,
        system: str,
        messages: list[dict],
        model: str | None = None,
        tier: ModelTier | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        thinking: dict | None = None,
        cache_control_msgs: bool = False,
        retries: int = 5,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """
        Send a chat request with resilient retry + tier escalation.

        Args:
            on_token: Optional callback invoked with each text token as it
                streams from the LLM. Works with all backends (Anthropic SDK,
                Anthropic HTTP, OpenAI-compatible). Pass ``None`` to disable
                streaming and collect the full response at once (default).

        Retry strategy (up to `retries` attempts):
          1. Try with the requested tier/model.
          2. On 429 rate-limit: immediately rotate to the next model in the
             pool (tier models + LLM_FALLBACK_MODELS) with minimal backoff.
             When all pool models are exhausted, reset and wait longer.
          3. On other failures: log the error, wait with exponential backoff.
          4. After first non-429 failure: if thinking was enabled, retry without it.
          5. After second non-429 failure: escalate tier (FAST→BALANCED→POWERFUL).
          6. If all retries exhausted: raise with full error history.

        The returned LLMResponse includes resilience metadata:
          retries, tier_escalated, thinking_stripped, model_switched,
          model_used, errors, duration_s.
        """
        t0 = time.time()

        requested_tier = tier or ModelTier.BALANCED
        current_tier = requested_tier
        original_model = model or self.resolve_model(requested_tier)
        current_model = original_model
        current_thinking = thinking

        # Resolve per-tier endpoint (may differ from default)
        tier_url, tier_key, tier_fmt_hint = self.resolve_endpoint(current_tier)
        self._set_effective_endpoint(tier_url, tier_key)
        try:
            fmt = self._detect_format(url_override=tier_url, fmt_override=tier_fmt_hint)
        finally:
            self._clear_effective_endpoint()

        if cache_control_msgs and fmt in (self.ANTHROPIC_NATIVE, self.ANTHROPIC_PROXY):
            messages = self._inject_cache_control(messages)

        errors: list[str] = []
        tier_escalated = False
        thinking_stripped = False
        model_switched = False

        # Build a rotation pool for 429 rate-limit scenarios
        model_pool = self._build_model_pool(original_model)
        rate_limited_models: set[str] = set()
        if len(model_pool) <= 1 and not getattr(self, "_single_model_warned", False):
            self._single_model_warned = True
            logger.warning(
                "Model pool has only 1 model (%s). Set LLM_FALLBACK_MODELS for 429 rotation.",
                model_pool[0] if model_pool else "none",
            )
            print(
                "  [LLM] ⚠️ Single-model pool — 429 rotation disabled. "
                "Set LLM_FALLBACK_MODELS for resilience."
            )

        for attempt in range(1, retries + 1):
            self._pacer.pace()
            # Set thread-local endpoint for this attempt (supports multi-provider)
            self._set_effective_endpoint(tier_url, tier_key)
            try:
                if fmt == self.ANTHROPIC_NATIVE:
                    resp = self._chat_anthropic_sdk(
                        system,
                        messages,
                        current_model,
                        temperature,
                        max_tokens,
                        current_thinking,
                        on_token=on_token,
                    )
                elif fmt == self.ANTHROPIC_PROXY:
                    resp = self._chat_anthropic_http(
                        system,
                        messages,
                        current_model,
                        temperature,
                        max_tokens,
                        current_thinking,
                        path_prefix="/anthropic",
                        on_token=on_token,
                    )
                else:
                    resp = self._chat_openai(
                        system, messages, current_model, temperature, max_tokens, on_token=on_token
                    )

                # Success — attach resilience metadata
                resp.tier_requested = requested_tier.value
                resp.tier_used = current_tier.value
                resp.model_requested = original_model
                resp.retries = attempt - 1
                resp.tier_escalated = tier_escalated
                resp.thinking_stripped = thinking_stripped
                resp.model_switched = model_switched
                resp.model_used = current_model
                resp.errors = errors
                resp.duration_s = time.time() - t0
                if model_switched:
                    logger.info(
                        "Succeeded with fallback model %s (originally %s)",
                        current_model,
                        original_model,
                    )
                self._clear_effective_endpoint()
                return resp

            except Exception as exc:
                self._clear_effective_endpoint()
                err_msg = f"{exc.__class__.__name__}: {str(exc)[:200]}"
                errors.append(err_msg)
                logger.warning("LLM attempt %d/%d failed: %s", attempt, retries, err_msg)

                if attempt == retries:
                    logger.error("LLM all %d retries exhausted. Errors: %s", retries, errors)
                    raise

                is_429 = self._is_rate_limit_error(exc)

                if is_429:
                    # ── Rate limit: rotate to next available model immediately ──
                    rate_limited_models.add(current_model)
                    available = [m for m in model_pool if m not in rate_limited_models]
                    retry_after = _extract_retry_after(exc)

                    if available:
                        current_model = available[0]
                        model_switched = True
                        # Respect server Retry-After if present, else short backoff
                        if retry_after and retry_after > 0:
                            wait = min(retry_after, 60.0)
                        else:
                            wait = random.uniform(0.5, 2.0)
                        logger.info(
                            "Rate-limited on %s, switching to %s",
                            rate_limited_models,
                            current_model,
                        )
                        print(f"  [LLM fallback] rate-limited, switching to model: {current_model}")
                    else:
                        # All models in pool are rate-limited — reset and wait longer
                        rate_limited_models.clear()
                        if retry_after and retry_after > 0:
                            wait = min(retry_after, 60.0)
                        else:
                            # Longer base when pool is single-model (no rotation benefit)
                            base = 3.0 if len(model_pool) <= 1 else 2.0
                            wait = _backoff_wait(attempt, base=base, max_wait=45.0)
                        logger.info(
                            "All models rate-limited, resetting pool and waiting %.1fs", wait
                        )
                        print(
                            f"  [LLM fallback] all models rate-limited, waiting {wait:.1f}s before retrying"
                        )
                    # Feed Retry-After into the pacer so subsequent requests
                    # are spaced at least that far apart.
                    if retry_after and retry_after > 0:
                        self._pacer.backoff(retry_after)
                else:
                    # ── Non-rate-limit error: existing strategy ──
                    wait = _backoff_wait(attempt, base=1.0, max_wait=60.0)
                    print(f"  [LLM retry {attempt}/{retries}] {err_msg}")

                    # Strategy: first try stripping thinking, then escalate tier
                    if current_thinking and attempt == 1:
                        current_thinking = None
                        thinking_stripped = True
                        logger.info("Stripping thinking param for next attempt")
                        print("  [LLM fallback] stripping thinking param for next attempt")
                    elif current_tier != ModelTier.POWERFUL and attempt >= 2:
                        prev_tier = current_tier
                        current_tier = current_tier.escalate()
                        escalated_model = self.resolve_model(current_tier)
                        # Only switch if the escalated model isn't already rate-limited
                        if escalated_model not in rate_limited_models:
                            current_model = escalated_model
                        tier_escalated = True
                        # Re-resolve endpoint for the escalated tier (may route
                        # to a different provider, e.g. FAST→OpenAI, POWERFUL→Anthropic)
                        tier_url, tier_key, tier_fmt_hint = self.resolve_endpoint(current_tier)
                        new_fmt = self._detect_format(
                            url_override=tier_url, fmt_override=tier_fmt_hint
                        )
                        if new_fmt != fmt:
                            fmt = new_fmt
                        logger.info(
                            "Escalating tier %s→%s (model: %s)",
                            prev_tier.value,
                            current_tier.value,
                            current_model,
                        )
                        print(
                            f"  [LLM fallback] escalating tier {prev_tier.value}→{current_tier.value} "
                            f"(model: {current_model})"
                        )

                logger.debug("Backing off %.1fs before retry", wait)
                time.sleep(wait)

        raise RuntimeError("unreachable")

    # ── Format detection ──────────────────────────────────────────────────────

    def _detect_format(
        self,
        url_override: str | None = None,
        fmt_override: str | None = None,
    ) -> str:
        """Auto-detect API format by probing the endpoint. Cached after first call.

        When *url_override* or *fmt_override* are provided (per-tier routing),
        the detection runs against that URL and the result is cached in
        ``self._format_cache[url]`` rather than the global ``self._format``.

        Results are persisted to disk so subsequent runs don't make network
        probe calls again.
        """
        probe_url = url_override or self.base_url
        probe_key = self._effective_key

        # If a per-tier format was explicitly set, return it directly
        if fmt_override:
            mapped = {
                "anthropic": self.ANTHROPIC_NATIVE,
                "anthropic_proxy": self.ANTHROPIC_PROXY,
                "openai": self.OPENAI_COMPAT,
            }.get(fmt_override, fmt_override)
            self._format_cache[probe_url] = mapped
            return mapped

        # Check in-memory cache for this URL
        if probe_url in self._format_cache:
            return self._format_cache[probe_url]

        # For the default URL, check the original cached format
        if probe_url == self.base_url and self._format:
            return self._format

        # If it's the official Anthropic endpoint, use the SDK
        if "api.anthropic.com" in probe_url:
            detected = self.ANTHROPIC_NATIVE
            self._format_cache[probe_url] = detected
            if probe_url == self.base_url:
                self._format = detected
            return detected

        # Try loading cached format from disk
        cache_file = Path.home() / ".ept" / "format_cache.json"
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
                if cache.get("base_url") == probe_url:
                    detected = cache["format"]
                    self._format_cache[probe_url] = detected
                    if probe_url == self.base_url:
                        self._format = detected
                    logger.debug("Loaded cached format: %s", detected)
                    return detected
            except Exception:
                pass

        headers = {
            "Authorization": f"Bearer {probe_key}",
            "x-api-key": probe_key,
        }

        # Probe 1: OpenAI /v1/models
        try:
            resp = httpx.get(f"{probe_url}/v1/models", headers=headers, timeout=5)
            if resp.status_code == 200:
                detected = self.OPENAI_COMPAT
                print("  [llm_client] detected OpenAI-compatible endpoint")
                self._format_cache[probe_url] = detected
                if probe_url == self.base_url:
                    self._format = detected
                self._save_format_cache(probe_url, detected)
                return detected
        except Exception:
            pass

        # Probe 2: Anthropic behind /anthropic prefix (Hyperspace / LiteLLM)
        try:
            resp = httpx.post(
                f"{probe_url}/anthropic/v1/messages",
                headers={
                    **headers,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.default_model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                timeout=5,
            )
            if resp.status_code in (200, 400, 401, 429):
                detected = self.ANTHROPIC_PROXY
                print("  [llm_client] detected Anthropic proxy endpoint (/anthropic/v1/messages)")
                self._format_cache[probe_url] = detected
                if probe_url == self.base_url:
                    self._format = detected
                self._save_format_cache(probe_url, detected)
                return detected
        except Exception:
            pass

        # Probe 3: Plain Anthropic /v1/messages
        try:
            resp = httpx.post(
                f"{probe_url}/v1/messages",
                headers={
                    **headers,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.default_model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                timeout=5,
            )
            if resp.status_code in (200, 400, 401, 429):
                detected = self.ANTHROPIC_NATIVE
                print("  [llm_client] detected Anthropic native endpoint (/v1/messages)")
                self._format_cache[probe_url] = detected
                if probe_url == self.base_url:
                    self._format = detected
                self._save_format_cache(probe_url, detected)
                return detected
        except Exception:
            pass

        # Default fallback
        detected = self.ANTHROPIC_PROXY
        print("  [llm_client] defaulting to Anthropic proxy format")
        self._format_cache[probe_url] = detected
        if probe_url == self.base_url:
            self._format = detected
        self._save_format_cache(probe_url, detected)
        return detected

    def _save_format_cache(self, url: str | None = None, fmt: str | None = None) -> None:
        """Persist detected format to disk so subsequent runs skip probes."""
        try:
            cache_file = Path.home() / ".ept" / "format_cache.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Write atomically via temp-file + rename
            content = json.dumps(
                {
                    "base_url": url or self.base_url,
                    "format": fmt or self._format,
                }
            )
            tmp = str(cache_file) + ".tmp"
            Path(tmp).write_text(content)
            os.replace(tmp, cache_file)
        except Exception:
            pass  # non-critical, just an optimization

    # ── Anthropic SDK (native — streaming + thinking + prompt caching) ────────

    def _get_anthropic_client(self):
        """Return a cached Anthropic SDK client for the current effective URL."""
        # Backward compat: tests may set _anthropic_client directly
        legacy = getattr(self, "_anthropic_client", None)
        if legacy is not None:
            return legacy

        url = self._effective_url
        key = self._effective_key
        cache_key = f"{url}|{key[:8]}"  # cache per endpoint+key prefix
        if not hasattr(self, "_sdk_clients"):
            self._sdk_clients: dict[str, Any] = {}
        if cache_key not in self._sdk_clients:
            import anthropic

            self._sdk_clients[cache_key] = anthropic.Anthropic(
                base_url=url if "api.anthropic.com" not in url else None,
                api_key=key or None,
            )
        return self._sdk_clients[cache_key]

    def _chat_anthropic_sdk(
        self,
        system,
        messages,
        model,
        temperature,
        max_tokens,
        thinking,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        client = self._get_anthropic_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        # Extended thinking only works with supported models
        if thinking:
            kwargs["thinking"] = thinking
        else:
            kwargs["temperature"] = temperature

        with client.messages.stream(**kwargs) as stream:
            if on_token:
                # Stream tokens to callback as they arrive
                collected_text: list[str] = []
                for event in stream:
                    if hasattr(event, "type") and event.type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "type", "") == "text_delta":
                            token = getattr(delta, "text", "")
                            if token:
                                on_token(token)
                                collected_text.append(token)
                response = stream.get_final_message()
                text = "".join(collected_text) or next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
            else:
                response = stream.get_final_message()
                text = next((b.text for b in response.content if b.type == "text"), "")
        usage = response.usage
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            stop_reason=getattr(response, "stop_reason", ""),
            raw=response,
        )

    # ── Anthropic HTTP (proxy — no SDK, plain httpx) ─────────────────────────

    def _chat_anthropic_http(
        self,
        system,
        messages,
        model,
        temperature,
        max_tokens,
        thinking,
        path_prefix: str = "",
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        url = self._effective_url
        key = self._effective_key
        headers = {
            "Authorization": f"Bearer {key}",
            "x-api-key": key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if thinking:
            payload["thinking"] = thinking
        else:
            payload["temperature"] = temperature

        # Enable SSE streaming when on_token callback is provided
        if on_token:
            payload["stream"] = True
            return self._stream_anthropic_http(headers, payload, model, path_prefix, on_token, url)

        resp = httpx.post(
            f"{url}{path_prefix}/v1/messages",
            headers=headers,
            json=payload,
            timeout=self.http_timeout,
        )

        # If 400 with thinking enabled, retry without it (proxy may not support it)
        if resp.status_code == 400 and thinking and "thinking" in payload:
            print("  [llm_client] thinking not supported by proxy, retrying without it")
            del payload["thinking"]
            payload["temperature"] = 0
            resp = httpx.post(
                f"{url}{path_prefix}/v1/messages",
                headers=headers,
                json=payload,
                timeout=self.http_timeout,
            )

        if resp.status_code != 200:
            # Include body in error for diagnostics
            body_preview = resp.text[:300] if resp.text else "(empty)"
            raise httpx.HTTPStatusError(
                f"{resp.status_code} for {resp.url} — {body_preview}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                break

        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
            stop_reason=data.get("stop_reason", ""),
            raw=data,
        )

    def _stream_anthropic_http(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
        path_prefix: str,
        on_token: Callable[[str], None],
        base_url: str | None = None,
    ) -> LLMResponse:
        """Stream Anthropic HTTP SSE and invoke on_token for each text delta."""
        url = base_url or self._effective_url
        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cache_read = 0
        cache_write = 0
        stop_reason = ""

        with httpx.stream(
            "POST",
            f"{url}{path_prefix}/v1/messages",
            headers=headers,
            json=payload,
            timeout=self.http_timeout,
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                body_preview = resp.text[:300] if resp.text else "(empty)"
                raise httpx.HTTPStatusError(
                    f"{resp.status_code} for {resp.url} — {body_preview}",
                    request=resp.request,
                    response=resp,
                )
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")
                if etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        token = delta.get("text", "")
                        if token:
                            on_token(token)
                            collected.append(token)
                elif etype == "message_start":
                    msg = event.get("message", {})
                    usage = msg.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                elif etype == "message_delta":
                    usage = event.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)
                    stop_reason = event.get("delta", {}).get("stop_reason", "")

        return LLMResponse(
            text="".join(collected),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            stop_reason=stop_reason,
            raw=None,
        )

    # ── OpenAI-compatible HTTP ────────────────────────────────────────────────

    def _chat_openai(
        self,
        system,
        messages,
        model,
        temperature,
        max_tokens,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        url = self._effective_url
        key = self._effective_key
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        # Prepend system message in OpenAI format
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for m in messages:
            # Flatten Anthropic-style content blocks to plain strings if needed
            content = m.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            api_messages.append({"role": m["role"], "content": content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Enable SSE streaming when on_token callback is provided
        if on_token:
            payload["stream"] = True
            return self._stream_openai(headers, payload, model, on_token, url)

        resp = httpx.post(
            f"{url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.http_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})

        return LLMResponse(
            text=text,
            model=data.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            stop_reason=choice.get("finish_reason", ""),
            raw=data,
        )

    def _stream_openai(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
        on_token: Callable[[str], None],
        base_url: str | None = None,
    ) -> LLMResponse:
        """Stream OpenAI-compatible SSE and invoke on_token for each delta."""
        url = base_url or self._effective_url
        collected: list[str] = []
        stop_reason = ""

        with httpx.stream(
            "POST",
            f"{url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.http_timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = event.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        on_token(token)
                        collected.append(token)
                    finish = choices[0].get("finish_reason")
                    if finish:
                        stop_reason = finish

        # OpenAI streaming doesn't always include usage — estimate from text
        text = "".join(collected)
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=0,  # not available in streaming responses
            output_tokens=0,
            stop_reason=stop_reason,
            raw=None,
        )

    # ── Cache control helper ──────────────────────────────────────────────────

    @staticmethod
    def _inject_cache_control(messages: list[dict]) -> list[dict]:
        """Inject Anthropic prompt-caching marker on the last prefix message."""
        msgs = copy.deepcopy(messages)
        if not msgs:
            return msgs
        last = msgs[-1]
        content = last.get("content", "")
        if isinstance(content, str):
            last["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list) and content:
            if isinstance(content[-1], dict):
                content[-1]["cache_control"] = {"type": "ephemeral"}
        return msgs

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def format_name(self) -> str:
        return self._format or "not detected yet"

    def __repr__(self) -> str:
        tier_info = ""
        if self._tier_config:
            tiers = ", ".join(self._tier_config.keys())
            tier_info = f", tier_overrides=[{tiers}]"
        return (
            f"LLMClient(base_url={self.base_url!r}, model={self.default_model!r}, "
            f"big={self.model_big!r}, small={self.model_small!r}, "
            f"format={self.format_name!r}{tier_info})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton — configured from env vars, ready to import
# ─────────────────────────────────────────────────────────────────────────────

llm = LLMClient()
