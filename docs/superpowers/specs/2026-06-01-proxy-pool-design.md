# ProxyPool — Multi-Proxy Health Routing for hive

**Date:** 2026-06-01
**Status:** Approved

---

## Problem

`LLMClient` currently supports one proxy URL per tier (`LLM_BASE_URL_FAST`, etc.). At volume, a single endpoint becomes a single point of failure: a degraded or rate-limited proxy stalls all requests with no visibility into why. There is no latency tracking, no error-rate awareness, and no way to distribute load across multiple proxy instances.

---

## Goals

- Support multiple proxy URLs per tier for load distribution and failover
- Track per-endpoint health: latency, error rate, circuit state
- Route intelligently: prefer fast, healthy endpoints; skip broken ones
- Remain backward-compatible: existing single-URL config unchanged
- Allow programmatic configuration for code users

---

## Architecture

### New file: `hive/proxy_pool.py`

Two classes: `ProxyEndpoint` (one URL) and `ProxyPool` (collection with routing logic).

```
ProxyEndpoint
  url: str
  api_key: str
  format_hint: str | None
  # health (mutable, updated after every call)
  latency_ewma_ms: float          # exponential weighted moving average
  error_count: int                # consecutive errors (resets on success)
  total_calls: int
  total_errors: int
  circuit_state: "closed" | "open" | "half-open"
  circuit_opened_at: float | None # timestamp when circuit opened

ProxyPool
  _endpoints: dict[str, list[ProxyEndpoint]]   # keyed by tier name
  add(tier, endpoint)
  next(tier) -> ProxyEndpoint                  # pick best healthy endpoint
  record_success(endpoint, latency_ms)         # update EWMA, close circuit
  record_failure(endpoint)                     # increment errors, maybe open circuit
  stats() -> dict                              # observable health snapshot
```

### Circuit Breaker (per endpoint)

States: `closed` (healthy) → `open` (skip) → `half-open` (probe) → `closed`

- `closed → open`: after `HIVE_CB_ERROR_THRESHOLD` (default 3) consecutive errors
- `open → half-open`: after `HIVE_CB_RECOVERY_SECS` (default 30) seconds
- `half-open → closed`: on next successful call
- `half-open → open`: on next failed call

### Routing Algorithm

1. Partition endpoints into `closed` and `half-open` (available) vs `open` (skipped)
2. Weight each available endpoint: `weight = 1 / max(latency_ewma_ms, 1)`
3. Weighted-random pick among available endpoints
4. If all endpoints are `open` (all circuits tripped): pick least-recently-failed as last resort and log a warning

### Latency EWMA

```
latency_ewma = alpha * new_sample + (1 - alpha) * latency_ewma
```

Alpha configurable via `HIVE_CB_LATENCY_ALPHA` (default `0.2`). Initial value: `500ms` (conservative start, converges quickly).

---

## Configuration

### Env vars — backward-compatible

Existing single-URL (unchanged, still works):
```
LLM_BASE_URL_FAST=http://proxy1:4000
LLM_API_KEY_FAST=key1
LLM_FORMAT_FAST=openai
```

New multi-URL (comma-separated, positional — index N of URLs matches index N of keys/formats):
```
LLM_BASE_URLS_FAST=http://p1:4000,http://p2:4000,http://p3:4000
LLM_API_KEYS_FAST=key1,key2,key3      # optional; falls back to LLM_API_KEY per slot
LLM_FORMATS_FAST=openai,openai,anthropic_proxy  # optional; auto-detected if omitted
```

Same pattern for `_BALANCED` and `_POWERFUL` suffixes.

Circuit breaker tuning:
```
HIVE_CB_ERROR_THRESHOLD=3     # consecutive errors before opening circuit
HIVE_CB_RECOVERY_SECS=30      # seconds before attempting half-open probe
HIVE_CB_LATENCY_ALPHA=0.2     # EWMA smoothing factor (0 < alpha <= 1)
```

### Programmatic (code users)

```python
from hive.proxy_pool import ProxyPool, ProxyEndpoint
from hive.llm_client import LLMClient, ModelTier

pool = ProxyPool()
pool.add(ModelTier.FAST, ProxyEndpoint(url="http://p1:4000", api_key="k1"))
pool.add(ModelTier.FAST, ProxyEndpoint(url="http://p2:4000", api_key="k2"))
pool.add(ModelTier.POWERFUL, ProxyEndpoint(url="https://api.anthropic.com", api_key="sk-ant-..."))

llm = LLMClient(proxy_pool=pool)
```

---

## `LLMClient` Changes (minimal)

- `__init__` accepts optional `proxy_pool: ProxyPool | None = None`
- On init, if no `proxy_pool` passed, `LLMClient` auto-builds one from env vars (parses both old `LLM_BASE_URL_*` and new `LLM_BASE_URLS_*`)
- In `chat()`, replace `resolve_endpoint(tier)` call with `pool.next(tier)` 
- After successful call: `pool.record_success(endpoint, latency_ms)`
- After failed call: `pool.record_failure(endpoint)`
- Existing `_tier_config` / `resolve_endpoint` path kept for cases where pool has no endpoints for a tier (graceful fallback to default URL)

---

## Data Flow

```
chat(tier=FAST)
  → pool.next(FAST)
      → filter out open circuits
      → weighted-random by 1/latency
      → return ProxyEndpoint
  → _set_effective_endpoint(endpoint.url, endpoint.api_key)
  → _detect_format(endpoint.url, endpoint.format_hint)
  → _chat_anthropic_sdk / _chat_anthropic_http / _chat_openai
  → success: pool.record_success(endpoint, elapsed_ms)
  → failure: pool.record_failure(endpoint)
             → if consecutive errors >= threshold: circuit open
             → chat() retries with pool.next(FAST) → different endpoint
```

---

## Error Handling

- `pool.next(tier)` never raises — always returns an endpoint (worst case: least-bad open circuit)
- Thread-safe: all health state mutations use a `threading.Lock` per endpoint
- If a tier has no endpoints configured: `pool.next(tier)` returns `None`, `LLMClient` falls back to existing `resolve_endpoint(tier)` behavior
- Circuit open log: `logger.warning("Circuit open for %s, skipping", endpoint.url)`
- All-circuits-tripped log: `logger.error("All proxies for tier %s are open — using least-bad", tier)`

---

## Testing

- Unit tests for `ProxyPool`: routing weights, circuit state transitions, all-open fallback
- Unit tests for `ProxyEndpoint`: EWMA convergence, error counting
- Integration test: `LLMClient` with a `ProxyPool` of two endpoints, first fails N times → circuit opens → requests route to second
- Existing `LLMClient` tests unchanged (single-endpoint path preserved)

---

## Files Changed

| File | Change |
|------|--------|
| `hive/proxy_pool.py` | **New** — `ProxyEndpoint`, `ProxyPool` |
| `hive/llm_client.py` | Accept `proxy_pool` param, wire `pool.next()` and `record_*()` into `chat()`, parse multi-URL env vars |
| `tests/test_proxy_pool.py` | **New** — unit + integration tests |
