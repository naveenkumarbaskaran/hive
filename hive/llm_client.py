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
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
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
#  Model tiers — agents request capability, not a specific model
# ─────────────────────────────────────────────────────────────────────────────

class ModelTier(str, Enum):
    """Capability tiers that agents request. Resolved to concrete models at runtime."""
    FAST      = "fast"       # classification, short JSON, quick checks
    BALANCED  = "balanced"   # PRD writing, moderate reasoning, summaries
    POWERFUL  = "powerful"   # architecture, code generation, deep reasoning

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
    tier_requested: str = ""           # tier the caller asked for
    tier_used: str = ""                # tier actually used (may be escalated)
    model_requested: str = ""          # model resolved from requested tier
    retries: int = 0                   # how many retries before success
    tier_escalated: bool = False       # was tier bumped?
    thinking_stripped: bool = False     # was thinking removed on fallback?
    model_switched: bool = False       # was model switched due to rate limit?
    model_used: str = ""               # actual model used (may differ from requested)
    errors: list[str] = field(default_factory=list)  # error msgs from failed attempts
    duration_s: float = 0.0            # wall-clock seconds


# ─────────────────────────────────────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Pluggable LLM connector. Users can swap backends by changing env vars —
    no code changes needed.
    """

    ANTHROPIC_NATIVE = "anthropic"       # direct Anthropic SDK (streaming, thinking, cache)
    ANTHROPIC_PROXY  = "anthropic_proxy"  # /anthropic/v1/messages (Hyperspace / LiteLLM)
    OPENAI_COMPAT    = "openai"           # /v1/chat/completions

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        model_big: str | None = None,
        model_small: str | None = None,
        api_format: str | None = None,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.anthropic.com")).rstrip("/")
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
        return bool(re.search(r'\b429\b', str(exc)[:80]))

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve_model(self, tier: ModelTier) -> str:
        """Resolve a capability tier to a concrete model name."""
        return {
            ModelTier.FAST:     self.model_small,
            ModelTier.BALANCED: self.default_model,
            ModelTier.POWERFUL: self.model_big,
        }[tier]

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
        fmt = self._detect_format()

        requested_tier = tier or ModelTier.BALANCED
        current_tier = requested_tier
        original_model = model or self.resolve_model(requested_tier)
        current_model = original_model
        current_thinking = thinking

        if cache_control_msgs and fmt in (self.ANTHROPIC_NATIVE, self.ANTHROPIC_PROXY):
            messages = self._inject_cache_control(messages)

        errors: list[str] = []
        tier_escalated = False
        thinking_stripped = False
        model_switched = False

        # Build a rotation pool for 429 rate-limit scenarios
        model_pool = self._build_model_pool(original_model)
        rate_limited_models: set[str] = set()
        if len(model_pool) <= 1 and not getattr(self, '_single_model_warned', False):
            self._single_model_warned = True
            logger.warning(
                "Model pool has only 1 model (%s). "
                "Set LLM_FALLBACK_MODELS for 429 rotation.",
                model_pool[0] if model_pool else "none",
            )
            print("  [LLM] ⚠️ Single-model pool — 429 rotation disabled. "
                  "Set LLM_FALLBACK_MODELS for resilience.")

        for attempt in range(1, retries + 1):
            try:
                if fmt == self.ANTHROPIC_NATIVE:
                    resp = self._chat_anthropic_sdk(system, messages, current_model,
                                                   temperature, max_tokens, current_thinking,
                                                   on_token=on_token)
                elif fmt == self.ANTHROPIC_PROXY:
                    resp = self._chat_anthropic_http(system, messages, current_model,
                                                    temperature, max_tokens, current_thinking,
                                                    path_prefix="/anthropic",
                                                    on_token=on_token)
                else:
                    resp = self._chat_openai(system, messages, current_model,
                                            temperature, max_tokens,
                                            on_token=on_token)

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
                    logger.info("Succeeded with fallback model %s (originally %s)",
                                current_model, original_model)
                return resp

            except Exception as exc:
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

                    if available:
                        current_model = available[0]
                        model_switched = True
                        wait = random.uniform(0.5, 2.0)  # short backoff — switching model
                        logger.info("Rate-limited on %s, switching to %s",
                                    rate_limited_models, current_model)
                        print(f"  [LLM fallback] rate-limited, switching to model: {current_model}")
                    else:
                        # All models in pool are rate-limited — reset and wait longer
                        rate_limited_models.clear()
                        # Longer base when pool is single-model (no rotation benefit)
                        base = 3.0 if len(model_pool) <= 1 else 2.0
                        wait = _backoff_wait(attempt, base=base, max_wait=45.0)
                        logger.info("All models rate-limited, resetting pool and waiting %.1fs", wait)
                        print(f"  [LLM fallback] all models rate-limited, waiting {wait:.1f}s before retrying")
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
                        logger.info("Escalating tier %s→%s (model: %s)",
                                   prev_tier.value, current_tier.value, current_model)
                        print(f"  [LLM fallback] escalating tier {prev_tier.value}→{current_tier.value} "
                              f"(model: {current_model})")

                logger.debug("Backing off %.1fs before retry", wait)
                time.sleep(wait)

        raise RuntimeError("unreachable")

    # ── Format detection ──────────────────────────────────────────────────────

    def _detect_format(self) -> str:
        """Auto-detect API format by probing the endpoint. Cached after first call.

        Results are persisted to disk so subsequent runs don't make network
        probe calls again.
        """
        if self._format:
            return self._format

        # If it's the official Anthropic endpoint, use the SDK
        if "api.anthropic.com" in self.base_url:
            self._format = self.ANTHROPIC_NATIVE
            return self._format

        # Try loading cached format from disk
        cache_file = Path.home() / ".ept" / "format_cache.json"
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
                if cache.get("base_url") == self.base_url:
                    self._format = cache["format"]
                    logger.debug("Loaded cached format: %s", self._format)
                    return self._format
            except Exception:
                pass

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
        }

        # Probe 1: OpenAI /v1/models
        try:
            resp = httpx.get(f"{self.base_url}/v1/models", headers=headers, timeout=5)
            if resp.status_code == 200:
                self._format = self.OPENAI_COMPAT
                print("  [llm_client] detected OpenAI-compatible endpoint")
                self._save_format_cache()
                return self._format
        except Exception:
            pass

        # Probe 2: Anthropic behind /anthropic prefix (Hyperspace / LiteLLM)
        try:
            resp = httpx.post(
                f"{self.base_url}/anthropic/v1/messages",
                headers={**headers, "Content-Type": "application/json", "anthropic-version": "2023-06-01"},
                json={"model": self.default_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=5,
            )
            if resp.status_code in (200, 400, 401, 429):
                self._format = self.ANTHROPIC_PROXY
                print("  [llm_client] detected Anthropic proxy endpoint (/anthropic/v1/messages)")
                self._save_format_cache()
                return self._format
        except Exception:
            pass

        # Probe 3: Plain Anthropic /v1/messages
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/messages",
                headers={**headers, "Content-Type": "application/json", "anthropic-version": "2023-06-01"},
                json={"model": self.default_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=5,
            )
            if resp.status_code in (200, 400, 401, 429):
                self._format = self.ANTHROPIC_NATIVE
                print("  [llm_client] detected Anthropic native endpoint (/v1/messages)")
                self._save_format_cache()
                return self._format
        except Exception:
            pass

        # Default fallback
        self._format = self.ANTHROPIC_PROXY
        print("  [llm_client] defaulting to Anthropic proxy format")

        # Persist detected format to disk cache
        self._save_format_cache()
        return self._format

    def _save_format_cache(self) -> None:
        """Persist detected format to disk so subsequent runs skip probes."""
        try:
            cache_file = Path.home() / ".ept" / "format_cache.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Write atomically via temp-file + rename
            content = json.dumps({"base_url": self.base_url, "format": self._format})
            tmp = str(cache_file) + ".tmp"
            Path(tmp).write_text(content)
            os.replace(tmp, cache_file)
        except Exception:
            pass  # non-critical, just an optimization

    # ── Anthropic SDK (native — streaming + thinking + prompt caching) ────────

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                base_url=self.base_url if "api.anthropic.com" not in self.base_url else None,
                api_key=self.api_key or None,
            )
        return self._anthropic_client

    def _chat_anthropic_sdk(
        self, system, messages, model, temperature, max_tokens, thinking,
        *, on_token: Callable[[str], None] | None = None,
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
        self, system, messages, model, temperature, max_tokens, thinking,
        path_prefix: str = "",
        *, on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
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
            return self._stream_anthropic_http(
                headers, payload, model, path_prefix, on_token
            )

        resp = httpx.post(
            f"{self.base_url}{path_prefix}/v1/messages",
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
                f"{self.base_url}{path_prefix}/v1/messages",
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
    ) -> LLMResponse:
        """Stream Anthropic HTTP SSE and invoke on_token for each text delta."""
        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cache_read = 0
        cache_write = 0
        stop_reason = ""

        with httpx.stream(
            "POST",
            f"{self.base_url}{path_prefix}/v1/messages",
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
        self, system, messages, model, temperature, max_tokens,
        *, on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
            return self._stream_openai(headers, payload, model, on_token)

        resp = httpx.post(
            f"{self.base_url}/v1/chat/completions",
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
    ) -> LLMResponse:
        """Stream OpenAI-compatible SSE and invoke on_token for each delta."""
        collected: list[str] = []
        stop_reason = ""

        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
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
            last["content"] = [{"type": "text", "text": content,
                                "cache_control": {"type": "ephemeral"}}]
        elif isinstance(content, list) and content:
            if isinstance(content[-1], dict):
                content[-1]["cache_control"] = {"type": "ephemeral"}
        return msgs

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def format_name(self) -> str:
        return self._format or "not detected yet"

    def __repr__(self) -> str:
        return (f"LLMClient(base_url={self.base_url!r}, model={self.default_model!r}, "
                f"big={self.model_big!r}, small={self.model_small!r}, "
                f"format={self.format_name!r})")


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton — configured from env vars, ready to import
# ─────────────────────────────────────────────────────────────────────────────

llm = LLMClient()
