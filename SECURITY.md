# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Hive, please report it responsibly:

1. **Do NOT open a public issue.**
2. Email **naveenkumarbaskaran@example.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive an acknowledgment within **48 hours**.
4. A fix will be developed and released as a patch within **7 days** for critical issues.

## Security Considerations

### API Keys

Hive requires LLM API keys to function. These are handled as follows:

- Keys are read exclusively from **environment variables** (`LLM_API_KEY`)
- Keys are **never logged**, written to disk, or included in checkpoint files
- Keys are **never embedded** in generated project source code

### Generated Code

Hive generates source code via LLM agents. Important caveats:

- Generated code is **not audited for security** by default
- The **Quinn** (QA) and **Judge** agents review for correctness, not security
- Always review generated code before deploying to production
- Never run generated code with elevated privileges without review

### File System Access

- Hive writes output **only** to the `projects/` directory (configurable via `PROJECTS_DIR`)
- Checkpoint files contain project state as JSON — no executable content
- Path sanitization prevents directory traversal in filenames
- Atomic writes prevent corruption from interrupted saves

### Network Access

- Hive makes outbound HTTPS calls **only** to the configured LLM endpoint
- The `--repo` flag clones git repositories using `git clone` (read-only)
- No telemetry, analytics, or phone-home behavior

### Dependencies

Hive has minimal dependencies by design:

| Dependency | Purpose | Required? |
|------------|---------|-----------|
| `httpx`    | HTTP client for OpenAI-compatible APIs | Yes |
| `anthropic`| Native Anthropic SDK | Optional |

## Hardening Features

Hive includes production-grade hardening:

- **Atomic file writes** — crash-safe persistence via temp file + rename
- **File locking** — `fcntl.flock` prevents corruption under concurrent access
- **Disk space pre-checks** — fails fast if filesystem is nearly full
- **Input sanitization** — filenames are sanitized against traversal attacks
- **Token budget management** — prevents runaway context window costs
- **Exponential backoff** — graceful retry on LLM API failures
- **Checkpoint schema validation** — rejects corrupt or tampered state files
