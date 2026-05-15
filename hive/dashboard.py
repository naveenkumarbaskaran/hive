"""
Lightweight live dashboard — SSE-based, stdlib only (no new dependencies).

Starts a background HTTP server that serves:
  GET /           → HTML dashboard page
  GET /events     → SSE stream of Blackboard events + telemetry
  GET /status     → JSON snapshot of current board state

Usage:
  from hive.dashboard import DashboardServer
  ds = DashboardServer(board, cost_tracker, port=8765)
  ds.start()   # non-blocking, runs in background thread
  ...
  ds.stop()

Or via CLI:
  hive --dashboard "Build a REST API"

The dashboard auto-refreshes via SSE and requires zero external dependencies.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hive.state import Blackboard
    from hive.telemetry import CostTracker

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  HTML template (inlined to avoid filesystem dependencies)
# ─────────────────────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hive — Live Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'SF Mono','Fira Code',monospace;background:#0d1117;color:#c9d1d9;
     padding:1rem;max-width:1200px;margin:0 auto}
h1{color:#58a6ff;font-size:1.4rem;margin-bottom:.8rem}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;margin-bottom:1rem}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.8rem}
.card h3{font-size:.75rem;color:#8b949e;text-transform:uppercase;margin-bottom:.4rem}
.card .val{font-size:1.6rem;font-weight:700;color:#58a6ff}
.card .val.cost{color:#f0883e}
.card .val.files{color:#3fb950}
.phase{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.8rem;
       margin-bottom:1rem}
.phase h2{font-size:1rem;color:#d2a8ff}
.progress{height:6px;background:#21262d;border-radius:3px;margin-top:.4rem;overflow:hidden}
.progress .bar{height:100%;background:linear-gradient(90deg,#58a6ff,#3fb950);
               transition:width .5s ease}
.files-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
            gap:.5rem;margin-bottom:1rem}
.file{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:.5rem .7rem;
      font-size:.8rem}
.file .name{color:#c9d1d9;font-weight:600}
.file .status{font-size:.7rem;margin-top:.2rem}
.file .status.approved{color:#3fb950}.file .status.building{color:#58a6ff}
.file .status.reviewing{color:#d2a8ff}.file .status.failed{color:#f85149}
#events{background:#161b22;border:1px solid #30363d;border-radius:8px;
        padding:.8rem;max-height:400px;overflow-y:auto;font-size:.78rem;line-height:1.5}
.event{padding:.15rem 0;border-bottom:1px solid #21262d}
.event .agent{color:#d2a8ff;font-weight:600;margin-right:.4rem}
.event .type{color:#8b949e;font-size:.7rem;margin-right:.4rem}
.event .time{color:#484f58;font-size:.65rem;float:right}
</style>
</head>
<body>
<h1>🐝 Hive — Live Dashboard</h1>

<div class="grid">
 <div class="card"><h3>Phase</h3><div class="val" id="phase">—</div></div>
 <div class="card"><h3>Cost</h3><div class="val cost" id="cost">$0.00</div></div>
 <div class="card"><h3>Files</h3><div class="val files" id="files">0/0</div></div>
</div>

<div class="phase">
 <h2 id="phase-label">Waiting…</h2>
 <div class="progress"><div class="bar" id="progress-bar" style="width:0%"></div></div>
</div>

<div class="files-grid" id="files-grid"></div>

<h3 style="color:#8b949e;font-size:.8rem;margin-bottom:.5rem">Event Log</h3>
<div id="events"></div>

<script>
const PHASES = ['welcome','knowledge','research','interview','prd','feasibility',
  'architecture','ratification','crew','build','integration','test_docs','release'];

function updateDashboard(d) {
  const pi = PHASES.indexOf(d.current_phase);
  const pct = pi >= 0 ? Math.round(((pi+1)/PHASES.length)*100) : 0;
  document.getElementById('phase').textContent = d.current_phase || '—';
  document.getElementById('phase-label').textContent =
    'Phase ' + (pi+1) + '/' + PHASES.length + ': ' + (d.current_phase || '—');
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('cost').textContent = '$' + (d.total_cost||0).toFixed(4);
  const approved = d.files_approved || 0;
  const total = d.files_total || 0;
  document.getElementById('files').textContent = approved + '/' + total;

  // Files grid
  const grid = document.getElementById('files-grid');
  grid.innerHTML = '';
  for (const [name, info] of Object.entries(d.files || {})) {
    const div = document.createElement('div');
    div.className = 'file';
    const st = info.approved ? 'approved' : (info.status || 'pending');
    div.innerHTML = '<div class="name">' + esc(name) + '</div>' +
      '<div class="status ' + st + '">● ' + st +
      (info.dev ? ' — ' + esc(info.dev) : '') + '</div>';
    grid.appendChild(div);
  }
}

function addEvent(ev) {
  const el = document.getElementById('events');
  const div = document.createElement('div');
  div.className = 'event';
  const t = new Date(ev.timestamp * 1000).toLocaleTimeString();
  div.innerHTML = '<span class="time">' + t + '</span>' +
    '<span class="agent">' + esc(ev.agent) + '</span>' +
    '<span class="type">[' + esc(ev.type) + ']</span> ' +
    esc(ev.content.substring(0, 200));
  el.prepend(div);
  while (el.children.length > 200) el.removeChild(el.lastChild);
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// SSE connection
const evtSrc = new EventSource('/events');
evtSrc.addEventListener('status', e => updateDashboard(JSON.parse(e.data)));
evtSrc.addEventListener('event', e => addEvent(JSON.parse(e.data)));
evtSrc.onerror = () => setTimeout(() => location.reload(), 3000);

// Initial fetch
fetch('/status').then(r=>r.json()).then(updateDashboard).catch(()=>{});
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  SSE request handler
# ─────────────────────────────────────────────────────────────────────────────


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves the dashboard HTML and SSE event stream."""

    # Silenced: don't spam terminal with HTTP log lines
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._serve_html()
        elif self.path == "/events":
            self._serve_sse()
        elif self.path == "/status":
            self._serve_status()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        body = _DASHBOARD_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_status(self) -> None:
        server: DashboardServer = self.server  # type: ignore[assignment]
        data = server.snapshot()
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self) -> None:
        """Long-lived SSE connection — pushes events + periodic status."""
        server: DashboardServer = self.server  # type: ignore[assignment]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Send initial status
        self._send_sse("status", json.dumps(server.snapshot()))

        last_event_idx = len(server.board.events)
        last_status_time = time.time()

        try:
            while not server._stop_event.is_set():
                # Push new events
                events = server.board.events
                if len(events) > last_event_idx:
                    for ev in events[last_event_idx:]:
                        payload = {
                            "type": ev.type.value if hasattr(ev.type, "value") else str(ev.type),
                            "agent": ev.agent,
                            "content": ev.content[:500],
                            "target": ev.target,
                            "timestamp": ev.timestamp,
                        }
                        self._send_sse("event", json.dumps(payload))
                    last_event_idx = len(events)

                # Periodic status update (every 2s)
                now = time.time()
                if now - last_status_time >= 2.0:
                    self._send_sse("status", json.dumps(server.snapshot()))
                    last_status_time = now

                time.sleep(0.3)  # poll interval
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client disconnected

    def _send_sse(self, event_type: str, data: str) -> None:
        msg = f"event: {event_type}\ndata: {data}\n\n"
        self.wfile.write(msg.encode())
        self.wfile.flush()


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard server
# ─────────────────────────────────────────────────────────────────────────────


class DashboardServer(HTTPServer):
    """Background HTTP server for the live dashboard.

    Usage::

        ds = DashboardServer(board, cost_tracker, port=8765)
        ds.start()    # non-blocking
        ds.stop()     # graceful shutdown
    """

    def __init__(
        self,
        board: Blackboard,
        cost_tracker: CostTracker | None = None,
        port: int = 8765,
    ):
        self.board = board
        self.cost_tracker = cost_tracker
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        super().__init__(("0.0.0.0", port), _DashboardHandler)

    def snapshot(self) -> dict[str, Any]:
        """Build a JSON-serializable summary of current board state."""
        board = self.board
        files: dict[str, dict[str, Any]] = {}
        approved_count = 0
        for name, entry in board.registry.items():
            files[name] = {
                "approved": entry.approved,
                "status": "approved" if entry.approved else "building",
                "dev": entry.assigned_dev or "",
                "revision": entry.revision,
            }
            if entry.approved:
                approved_count += 1

        total_cost = 0.0
        if self.cost_tracker:
            total_cost = self.cost_tracker.total_cost

        return {
            "feature": board.feature,
            "current_phase": board.current_phase,
            "completed_phases": list(board.completed_phases),
            "files": files,
            "files_total": len(board.registry),
            "files_approved": approved_count,
            "total_cost": total_cost,
            "event_count": len(board.events),
        }

    def start(self) -> None:
        """Start the dashboard server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self.serve_forever,
            daemon=True,
            name="hive-dashboard",
        )
        self._thread.start()
        logger.info("Dashboard started at http://localhost:%d", self.server_address[1])
        print(f"  🌐 Live dashboard: http://localhost:{self.server_address[1]}")

    def stop(self) -> None:
        """Gracefully shut down the dashboard server."""
        self._stop_event.set()
        self.shutdown()
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Dashboard stopped")
