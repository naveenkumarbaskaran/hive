"""
Lightweight live dashboard v2 — SSE-based, stdlib only (no new dependencies).

Starts a background HTTP server that serves:
  GET /           → Rich HTML dashboard with crew, files, signoffs, events
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

PHASES = [
    "welcome", "knowledge", "research", "interview", "prd", "feasibility",
    "architecture", "ratification", "crew", "build", "integration",
    "test_docs", "release",
]

# ─────────────────────────────────────────────────────────────────────────────
#  HTML template v2 — rich dashboard with crew, signoffs, events
# ─────────────────────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hive — Live Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0d1117;--card:#161b22;--border:#30363d;--border-light:#21262d;
  --text:#c9d1d9;--text-dim:#8b949e;--text-faint:#484f58;
  --blue:#58a6ff;--green:#3fb950;--orange:#f0883e;--purple:#d2a8ff;
  --red:#f85149;--cyan:#39d353;
}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
     background:var(--bg);color:var(--text);padding:0;min-height:100vh}
.container{max-width:1320px;margin:0 auto;padding:1rem 1.2rem}

/* ── Header ── */
header{display:flex;align-items:center;justify-content:space-between;
       margin-bottom:1rem;padding-bottom:.8rem;border-bottom:1px solid var(--border)}
.logo{display:flex;align-items:center;gap:.6rem}
.logo h1{font-size:1.3rem;color:var(--blue);font-weight:800;letter-spacing:-.5px}
.logo .bee{font-size:1.5rem}
.feature-name{color:var(--text);font-size:.85rem;font-weight:400;
              max-width:500px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.live-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
          display:inline-block;animation:pulse 2s infinite;margin-right:6px}
.header-right{display:flex;align-items:center;gap:.5rem;font-size:.75rem;color:var(--text-dim)}

/* ── Stats Row ── */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-bottom:1rem}
.stat{background:var(--card);border:1px solid var(--border);border-radius:10px;
      padding:.7rem .9rem;position:relative;overflow:hidden}
.stat::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.stat:nth-child(1)::before{background:var(--blue)}
.stat:nth-child(2)::before{background:var(--orange)}
.stat:nth-child(3)::before{background:var(--green)}
.stat:nth-child(4)::before{background:var(--purple)}
.stat-label{font-size:.65rem;color:var(--text-dim);text-transform:uppercase;
            letter-spacing:.5px;margin-bottom:.25rem}
.stat-val{font-size:1.5rem;font-weight:700;font-family:'SF Mono','Fira Code',monospace}
.stat:nth-child(1) .stat-val{color:var(--blue)}
.stat:nth-child(2) .stat-val{color:var(--orange)}
.stat:nth-child(3) .stat-val{color:var(--green)}
.stat:nth-child(4) .stat-val{color:var(--purple)}
.stat-sub{font-size:.7rem;color:var(--text-dim);margin-top:.15rem}

/* ── Phase Progress ── */
.progress-section{background:var(--card);border:1px solid var(--border);border-radius:10px;
                   padding:.8rem 1rem;margin-bottom:1rem}
.progress-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem}
.progress-label{font-size:.85rem;color:var(--text);font-weight:600}
.progress-pct{font-size:.85rem;color:var(--blue);font-weight:700;
              font-family:'SF Mono',monospace}
.progress-track{height:5px;background:var(--border-light);border-radius:3px;
                overflow:hidden;margin-bottom:.7rem}
.progress-bar{height:100%;background:linear-gradient(90deg,var(--blue),var(--green));
              border-radius:3px;transition:width .6s ease}
.phase-pills{display:flex;flex-wrap:wrap;gap:4px}
.phase-pill{display:inline-flex;align-items:center;gap:3px;padding:2px 8px;
            border-radius:10px;font-size:.6rem;background:var(--border-light);
            color:var(--text-faint);transition:all .3s}
.phase-pill .pip{width:5px;height:5px;border-radius:50%;background:var(--text-faint)}
.phase-pill.done{background:rgba(63,185,80,.12);color:var(--green)}
.phase-pill.done .pip{background:var(--green)}
.phase-pill.current{background:rgba(88,166,255,.15);color:var(--blue);
                    font-weight:700;box-shadow:0 0 0 1px rgba(88,166,255,.25)}
.phase-pill.current .pip{background:var(--blue);box-shadow:0 0 4px var(--blue)}

/* ── Section Headers ── */
section{margin-bottom:1.2rem}
.section-hdr{display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem}
.section-hdr h2{font-size:.8rem;color:var(--text-dim);text-transform:uppercase;
                letter-spacing:.8px;font-weight:600}
.section-hdr .line{flex:1;height:1px;background:var(--border)}
.section-hdr .count{font-size:.7rem;color:var(--text-faint);
                    font-family:'SF Mono',monospace}

/* ── Crew Grid ── */
.crew-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:.6rem}
.agent-card{background:var(--card);border:1px solid var(--border);border-radius:10px;
            padding:0;overflow:hidden;transition:border-color .3s,transform .15s}
.agent-card:hover{transform:translateY(-1px)}
.agent-card.working{border-color:rgba(63,185,80,.4)}
.agent-card.idle{border-color:var(--border)}
.agent-titlebar{display:flex;align-items:center;gap:6px;padding:8px 12px;
                background:rgba(255,255,255,.02);border-bottom:1px solid var(--border-light)}
.agent-titlebar .dots{display:flex;gap:4px}
.agent-titlebar .dots span{width:7px;height:7px;border-radius:50%;background:var(--text-faint)}
.agent-card.working .agent-titlebar .dots span:first-child{background:var(--green)}
.agent-card.idle .agent-titlebar .dots span:first-child{background:var(--text-faint)}
.agent-titlebar .agent-label{font-size:.75rem;color:var(--text-dim);flex:1;
                             white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.agent-body{padding:10px 12px}
.agent-header{display:flex;align-items:center;gap:8px;margin-bottom:2px}
.agent-emoji{font-size:1.3rem;line-height:1}
.agent-name{font-weight:700;font-size:.9rem;color:#e6edf3}
.agent-role{font-size:.7rem;color:var(--text-dim);margin-bottom:8px}
.agent-status{display:flex;align-items:center;gap:6px;margin-bottom:6px}
.status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.status-dot.working{background:var(--green);animation:pulse 1.5s infinite}
.status-dot.idle{background:var(--text-faint)}
.status-text{font-size:.75rem;font-weight:600}
.status-text.working{color:var(--green)}
.status-text.idle{color:var(--text-faint)}
.agent-feed{font-size:.68rem;color:var(--text-dim);line-height:1.5;
            font-family:'SF Mono','Fira Code',monospace}
.feed-line{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:1px 0}
.feed-line::before{content:'▸ ';color:var(--text-faint)}
.crew-empty{color:var(--text-faint);font-size:.8rem;padding:.5rem;font-style:italic}

/* ── Signoffs ── */
.signoffs-list{display:flex;flex-wrap:wrap;gap:.5rem}
.signoff-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;
               background:var(--card);border:1px solid var(--border);border-radius:8px;
               font-size:.78rem;transition:all .3s}
.signoff-badge.approved{border-color:rgba(63,185,80,.3);background:rgba(63,185,80,.06)}
.signoff-badge .check{color:var(--green);font-weight:700}
.signoff-badge .artifact{font-weight:700;color:var(--text);text-transform:capitalize}
.signoff-badge .by{color:var(--text-dim);font-size:.7rem}
.signoffs-empty{color:var(--text-faint);font-size:.8rem;font-style:italic}

/* ── Files Grid ── */
.files-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.5rem}
.file-card{background:var(--card);border:1px solid var(--border);border-radius:8px;
           padding:.55rem .75rem;transition:all .3s}
.file-card.approved{border-left:3px solid var(--green)}
.file-card.building{border-left:3px solid var(--blue)}
.file-card.reviewing{border-left:3px solid var(--purple)}
.file-card.pending{border-left:3px solid var(--text-faint)}
.file-card.failed{border-left:3px solid var(--red)}
.file-name{font-weight:600;font-size:.8rem;color:var(--text);
           font-family:'SF Mono',monospace;white-space:nowrap;overflow:hidden;
           text-overflow:ellipsis}
.file-meta{display:flex;justify-content:space-between;align-items:center;margin-top:.25rem}
.file-status{font-size:.68rem;font-weight:600}
.file-status.approved{color:var(--green)}
.file-status.building{color:var(--blue)}
.file-status.reviewing{color:var(--purple)}
.file-status.pending{color:var(--text-faint)}
.file-status.failed{color:var(--red)}
.file-dev{font-size:.65rem;color:var(--text-dim)}

/* ── Event Log ── */
.events-section{background:var(--card);border:1px solid var(--border);border-radius:10px;
                padding:.8rem 1rem}
.event-filters{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:.6rem}
.filter-btn{padding:3px 10px;border-radius:12px;font-size:.65rem;
            background:var(--border-light);color:var(--text-faint);
            border:none;cursor:pointer;transition:all .2s;font-family:inherit}
.filter-btn:hover{color:var(--text);background:var(--border)}
.filter-btn.active{background:rgba(88,166,255,.15);color:var(--blue);font-weight:600}
.events-log{max-height:350px;overflow-y:auto;font-size:.73rem;line-height:1.5;
            scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.events-log::-webkit-scrollbar{width:4px}
.events-log::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.event-row{display:flex;gap:.5rem;padding:3px 0;border-bottom:1px solid var(--border-light);
           align-items:baseline;transition:opacity .2s}
.event-row.hidden{display:none}
.event-time{color:var(--text-faint);font-size:.65rem;font-family:'SF Mono',monospace;
            flex-shrink:0;min-width:65px}
.event-agent{font-weight:700;color:var(--purple);flex-shrink:0;min-width:70px;
             white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.event-type{color:var(--text-faint);font-size:.62rem;flex-shrink:0;min-width:60px;
            text-transform:uppercase;letter-spacing:.3px}
.event-content{color:var(--text-dim);flex:1;white-space:nowrap;overflow:hidden;
               text-overflow:ellipsis}
.events-empty{color:var(--text-faint);font-style:italic;padding:.5rem 0}

/* ── Animations ── */
@keyframes pulse{
  0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(63,185,80,.4)}
  50%{opacity:.7;box-shadow:0 0 0 5px rgba(63,185,80,0)}
}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.event-row{animation:fadeIn .3s ease}

/* ── Responsive ── */
@media(max-width:768px){
  .stats{grid-template-columns:repeat(2,1fr)}
  .crew-grid{grid-template-columns:1fr 1fr}
  .feature-name{display:none}
}
@media(max-width:480px){
  .crew-grid{grid-template-columns:1fr}
  .files-grid{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>
<div class="container">

<header>
  <div class="logo">
    <span class="bee">🐝</span>
    <h1>HIVE</h1>
    <span style="color:var(--text-faint);margin:0 .4rem">·</span>
    <span class="feature-name" id="feature-name">Loading…</span>
  </div>
  <div class="header-right">
    <span class="live-dot"></span>
    <span id="updated">connected</span>
  </div>
</header>

<div class="stats">
  <div class="stat">
    <div class="stat-label">Phase</div>
    <div class="stat-val" id="stat-phase">—</div>
    <div class="stat-sub" id="stat-phase-name">waiting</div>
  </div>
  <div class="stat">
    <div class="stat-label">Cost</div>
    <div class="stat-val" id="stat-cost">$0.00</div>
    <div class="stat-sub">total spend</div>
  </div>
  <div class="stat">
    <div class="stat-label">Files</div>
    <div class="stat-val" id="stat-files">0/0</div>
    <div class="stat-sub">approved</div>
  </div>
  <div class="stat">
    <div class="stat-label">Events</div>
    <div class="stat-val" id="stat-events">0</div>
    <div class="stat-sub">logged</div>
  </div>
</div>

<div class="progress-section">
  <div class="progress-header">
    <span class="progress-label" id="progress-label">Initializing…</span>
    <span class="progress-pct" id="progress-pct">0%</span>
  </div>
  <div class="progress-track"><div class="progress-bar" id="progress-bar" style="width:0%"></div></div>
  <div class="phase-pills" id="phase-pills"></div>
</div>

<section id="crew-section">
  <div class="section-hdr">
    <h2>🤖 Crew</h2>
    <div class="line"></div>
    <span class="count" id="crew-count">0 agents</span>
  </div>
  <div class="crew-grid" id="crew-grid">
    <div class="crew-empty">Crew assembles in Phase 9…</div>
  </div>
</section>

<section id="signoffs-section">
  <div class="section-hdr">
    <h2>✅ Sign-offs</h2>
    <div class="line"></div>
    <span class="count" id="signoff-count">0</span>
  </div>
  <div class="signoffs-list" id="signoffs-list">
    <div class="signoffs-empty">Artifacts reviewed as the project progresses</div>
  </div>
</section>

<section id="files-section">
  <div class="section-hdr">
    <h2>📁 Files</h2>
    <div class="line"></div>
    <span class="count" id="file-count">0</span>
  </div>
  <div class="files-grid" id="files-grid"></div>
</section>

<section>
  <div class="section-hdr">
    <h2>📜 Event Log</h2>
    <div class="line"></div>
    <span class="count" id="event-log-count">0</span>
  </div>
  <div class="events-section">
    <div class="event-filters" id="event-filters">
      <button class="filter-btn active" data-filter="all" onclick="setFilter('all')">All</button>
    </div>
    <div class="events-log" id="events-log">
      <div class="events-empty">Waiting for events…</div>
    </div>
  </div>
</section>

</div>

<script>
const PHASES=['welcome','knowledge','research','interview','prd','feasibility',
  'architecture','ratification','crew','build','integration','test_docs','release'];
const AGENT_COLORS={
  scout:'#58a6ff',penny:'#d2a8ff',archie:'#f0883e',quinn:'#3fb950',
  judge:'#f85149',pixel:'#ff7b72',flow:'#79c0ff',alex:'#ffa657',
  dm:'#d2a8ff',morgan:'#d2a8ff',system:'#8b949e',user:'#c9d1d9'
};

let activeFilter='all';
let eventCount=0;
let initialized=false;
let knownAgents=new Set();

/* ── Escape HTML ── */
function esc(s){const d=document.createElement('div');d.textContent=s||'';return d.innerHTML}

/* ── Agent color by ID ── */
function agentColor(id){
  if(!id)return'var(--text-dim)';
  for(const[k,v]of Object.entries(AGENT_COLORS)){if(id.startsWith(k))return v}
  if(id.startsWith('dev_')||id.startsWith('reviewer_'))return'#f0883e';
  return'var(--purple)';
}

/* ── Phase stepper pills ── */
function renderPhases(completed,current){
  const el=document.getElementById('phase-pills');
  el.innerHTML=PHASES.map(p=>{
    const done=completed.includes(p);
    const cur=p===current;
    const cls=done?'done':cur?'current':'';
    return '<span class="phase-pill '+cls+'"><span class="pip"></span>'+p+'</span>';
  }).join('');
}

/* ── Crew Cards ── */
function renderCrew(crew,activity){
  const grid=document.getElementById('crew-grid');
  if(!crew||crew.length===0){
    grid.innerHTML='<div class="crew-empty">Crew assembles in Phase 9…</div>';
    document.getElementById('crew-count').textContent='0 agents';
    return;
  }
  const active=crew.filter(a=>a.active);
  document.getElementById('crew-count').textContent=active.length+' agents';
  grid.innerHTML=active.map(a=>{
    const act=activity[a.id]||{};
    const st=act.status||'idle';
    const recent=act.recent||[];
    const target=act.target?(' · '+esc(act.target)):'';
    knownAgents.add(a.id);
    return '<div class="agent-card '+st+'">'+
      '<div class="agent-titlebar">'+
        '<div class="dots"><span></span><span></span><span></span></div>'+
        '<span class="agent-label">'+esc(a.name)+' · '+esc(a.role)+'</span>'+
      '</div>'+
      '<div class="agent-body">'+
        '<div class="agent-header">'+
          '<span class="agent-emoji">'+esc(a.emoji)+'</span>'+
          '<span class="agent-name">'+esc(a.name)+'</span>'+
        '</div>'+
        '<div class="agent-role">'+esc(a.role)+' · '+esc(a.tier||'')+'</div>'+
        '<div class="agent-status">'+
          '<span class="status-dot '+st+'"></span>'+
          '<span class="status-text '+st+'">'+st.toUpperCase()+target+'</span>'+
        '</div>'+
        '<div class="agent-feed">'+
          (recent.length?recent.map(r=>'<div class="feed-line">'+esc(r)+'</div>').join(''):
           '<div class="feed-line" style="color:var(--text-faint)">awaiting task…</div>')+
        '</div>'+
      '</div>'+
    '</div>';
  }).join('');
}

/* ── Signoffs ── */
function renderSignoffs(signoffs){
  const el=document.getElementById('signoffs-list');
  document.getElementById('signoff-count').textContent=signoffs.length;
  if(!signoffs||signoffs.length===0){
    el.innerHTML='<div class="signoffs-empty">Artifacts reviewed as the project progresses</div>';
    return;
  }
  el.innerHTML=signoffs.map(s=>{
    const cls=s.approved?'approved':'';
    const icon=s.approved?'✅':'⏳';
    const by=s.produced_by?' by '+esc(s.produced_by):'';
    const rev=s.reviewed_by&&s.reviewed_by.length?' · reviewed: '+s.reviewed_by.map(esc).join(', '):'';
    return '<div class="signoff-badge '+cls+'">'+
      '<span class="check">'+icon+'</span>'+
      '<span class="artifact">'+esc(s.artifact)+' v'+s.version+'</span>'+
      '<span class="by">'+by+rev+'</span>'+
    '</div>';
  }).join('');
}

/* ── Files Grid ── */
function renderFiles(files){
  const grid=document.getElementById('files-grid');
  const entries=Object.entries(files||{});
  document.getElementById('file-count').textContent=entries.length+' files';
  if(entries.length===0){grid.innerHTML='';return}
  grid.innerHTML=entries.map(([name,info])=>{
    const st=info.approved?'approved':(info.status||'pending');
    const icon={approved:'✅',building:'🔨',reviewing:'🔍',pending:'⏳',failed:'❌'}[st]||'⏳';
    const dev=info.dev?esc(info.dev):'';
    const rev=info.revision>1?'rev '+info.revision:'';
    return '<div class="file-card '+st+'">'+
      '<div class="file-name">'+icon+' '+esc(name)+'</div>'+
      '<div class="file-meta">'+
        '<span class="file-status '+st+'">'+st+'</span>'+
        '<span class="file-dev">'+dev+(dev&&rev?' · ':'')+rev+'</span>'+
      '</div>'+
    '</div>';
  }).join('');
}

/* ── Event Filters ── */
function buildFilters(){
  const el=document.getElementById('event-filters');
  const agents=['all',...knownAgents];
  el.innerHTML=agents.map(a=>{
    const cls=a===activeFilter?'active':'';
    const label=a==='all'?'All':esc(a);
    return '<button class="filter-btn '+cls+'" data-filter="'+esc(a)+'">'+label+'</button>';
  }).join('');
  el.querySelectorAll('.filter-btn').forEach(btn=>{
    btn.addEventListener('click',()=>setFilter(btn.dataset.filter));
  });
}

function setFilter(f){
  activeFilter=f;
  buildFilters();
  document.querySelectorAll('#events-log .event-row').forEach(el=>{
    if(f==='all'||el.dataset.agent===f)el.classList.remove('hidden');
    else el.classList.add('hidden');
  });
}

/* ── Add Event to Log ── */
function addEvent(ev,prepend){
  if(prepend===undefined)prepend=true;
  const el=document.getElementById('events-log');
  // Clear empty placeholder
  const empty=el.querySelector('.events-empty');
  if(empty)empty.remove();

  const div=document.createElement('div');
  div.className='event-row';
  div.dataset.agent=ev.agent||'';
  if(activeFilter!=='all'&&div.dataset.agent!==activeFilter)div.classList.add('hidden');

  const t=new Date((ev.timestamp||0)*1000).toLocaleTimeString();
  const color=agentColor(ev.agent);
  div.innerHTML=
    '<span class="event-time">'+t+'</span>'+
    '<span class="event-agent" style="color:'+color+'">'+esc(ev.agent)+'</span>'+
    '<span class="event-type">'+esc(ev.type)+'</span>'+
    '<span class="event-content">'+esc((ev.content||'').substring(0,250))+'</span>';

  if(prepend)el.prepend(div);else el.appendChild(div);
  while(el.children.length>300)el.removeChild(el.lastChild);

  // Track agent for filters
  if(ev.agent&&ev.agent!=='system'&&ev.agent!=='user'){
    if(!knownAgents.has(ev.agent)){knownAgents.add(ev.agent);buildFilters()}
  }
  eventCount++;
  document.getElementById('event-log-count').textContent=eventCount;
}

/* ── Main update from status snapshot ── */
function updateDashboard(d){
  // Feature name
  document.getElementById('feature-name').textContent=d.feature||'Loading…';

  // Stats
  const pi=PHASES.indexOf(d.current_phase);
  const phaseNum=pi>=0?pi+1:0;
  document.getElementById('stat-phase').textContent=phaseNum+'/'+PHASES.length;
  document.getElementById('stat-phase-name').textContent=d.current_phase||'waiting';
  document.getElementById('stat-cost').textContent='$'+(d.total_cost||0).toFixed(4);
  const ap=d.files_approved||0;const tot=d.files_total||0;
  document.getElementById('stat-files').textContent=ap+'/'+tot;
  document.getElementById('stat-events').textContent=d.event_count||0;

  // Progress bar
  const pct=phaseNum>0?Math.round((phaseNum/PHASES.length)*100):0;
  document.getElementById('progress-label').textContent=
    'Phase '+phaseNum+'/'+PHASES.length+': '+(d.current_phase||'—');
  document.getElementById('progress-pct').textContent=pct+'%';
  document.getElementById('progress-bar').style.width=pct+'%';

  // Phase pills
  renderPhases(d.completed_phases||[],d.current_phase);

  // Crew
  renderCrew(d.crew||[],d.agent_activity||{});

  // Signoffs
  renderSignoffs(d.signoffs||[]);

  // Files
  renderFiles(d.files||{});

  // Update time
  document.getElementById('updated').textContent=
    'live · '+new Date().toLocaleTimeString();
}

/* ── SSE Connection with auto-reconnect ── */
let sseActive=false;
let evtSrc=null;
let lastFeature='';

function connectSSE(){
  if(evtSrc){evtSrc.close();evtSrc=null;}
  try{
    evtSrc=new EventSource('/events');
    evtSrc.addEventListener('status',e=>{
      sseActive=true;
      const d=JSON.parse(e.data);
      updateDashboard(d);
    });
    evtSrc.addEventListener('event',e=>{
      addEvent(JSON.parse(e.data),true);
    });
    evtSrc.onerror=()=>{
      sseActive=false;
      document.getElementById('updated').textContent='reconnecting…';
      evtSrc.close();evtSrc=null;
      setTimeout(connectSSE,2000);
    };
  }catch(e){console.warn('SSE not available, using polling');}
}
connectSSE();

/* ── Periodic polling fallback — also detects new runs ── */
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(d=>{
    // New run detected: feature changed — clear stale event log and reconnect SSE
    if(lastFeature && d.feature && d.feature!==lastFeature){
      eventCount=0;
      knownAgents=new Set();
      document.getElementById('events-log').innerHTML='<div class="events-empty">Waiting for events…</div>';
      document.getElementById('event-log-count').textContent='0';
      buildFilters();
      if(!evtSrc)connectSSE();
    }
    lastFeature=d.feature||lastFeature;
    updateDashboard(d);
  }).catch(()=>{
    document.getElementById('updated').textContent='waiting for run…';
  });
},3000);

// Initial fetch
fetch('/status').then(r=>r.json()).then(d=>{
  lastFeature=d.feature||'';
  updateDashboard(d);
}).catch(()=>{});
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  SSE request handler
# ─────────────────────────────────────────────────────────────────────────────


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves the dashboard HTML and SSE event stream."""

    # Force HTTP/1.1 so SSE keep-alive and chunked responses work in browsers
    protocol_version = "HTTP/1.1"

    # Suppress default HTTP logging to terminal
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
        self.send_header("Content-Length", str(len(body)))
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

        # ── Initial payload: full snapshot ──
        try:
            self._send_sse("status", json.dumps(server.snapshot()))
        except Exception:
            return

        # ── Send existing events (last 100) to populate the event log ──
        current_board = server.board
        events = current_board.events
        start = max(0, len(events) - 100)
        for ev in events[start:]:
            self._send_sse("event", json.dumps(server._serialize_event(ev)))

        last_event_idx = len(events)
        last_status_time = time.time()

        try:
            while not server._stop_event.is_set():
                # Detect board swap (standalone reload from checkpoint)
                if server.board is not current_board:
                    current_board = server.board
                    events = current_board.events
                    # Resend last 100 events from the new board
                    start = max(0, len(events) - 100)
                    for ev in events[start:]:
                        self._send_sse(
                            "event", json.dumps(server._serialize_event(ev))
                        )
                    last_event_idx = len(events)
                    self._send_sse("status", json.dumps(server.snapshot()))
                    last_status_time = time.time()
                    continue

                # Push new events from same board
                events = current_board.events
                if len(events) > last_event_idx:
                    for ev in events[last_event_idx:]:
                        self._send_sse(
                            "event", json.dumps(server._serialize_event(ev))
                        )
                    last_event_idx = len(events)

                # Periodic status update (every 3s)
                now = time.time()
                if now - last_status_time >= 3.0:
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
        self._agents: dict[str, dict[str, Any]] = {}  # populated via set_agents()
        super().__init__(("0.0.0.0", port), _DashboardHandler)

    def set_agents(self, agents: dict) -> None:
        """Store crew agent data for the dashboard.

        Called after Phase 9 (Crew Assembly) with the Agent dict from EPTCrew.
        """
        self._agents = {}
        for aid, a in agents.items():
            self._agents[aid] = {
                "id": a.id,
                "name": a.name,
                "role": a.role,
                "emoji": a.emoji,
                "tagline": a.tagline,
                "active": a.active,
                "tier": a.tier.value if hasattr(a.tier, "value") else str(a.tier),
            }

    @staticmethod
    def _serialize_event(ev: Any) -> dict[str, Any]:
        """Convert a board Event to a JSON-safe dict."""
        return {
            "type": ev.type.value if hasattr(ev.type, "value") else str(ev.type),
            "agent": ev.agent,
            "content": ev.content[:500],
            "target": ev.target,
            "timestamp": ev.timestamp,
        }

    def snapshot(self) -> dict[str, Any]:
        """Build a JSON-serializable summary of current board state."""
        board = self.board

        # ── Files ──
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

        # ── Cost ──
        total_cost = 0.0
        if self.cost_tracker:
            total_cost = self.cost_tracker.total_cost

        # ── Signoffs ──
        signoffs: list[dict[str, Any]] = []
        for s in board.signoffs:
            signoffs.append({
                "artifact": s.artifact,
                "version": s.version,
                "approved": s.approved,
                "feedback": s.feedback or "",
                "produced_by": s.produced_by or "",
                "reviewed_by": list(s.reviewed_by) if s.reviewed_by else [],
                "timestamp": s.timestamp,
            })

        # ── Agent activity (derived from recent events) ──
        agent_activity: dict[str, dict[str, Any]] = {}
        now = time.time()
        working_types = {"thinking", "writing", "reviewing"}
        for ev in reversed(board.events):
            aid = ev.agent
            if aid in ("system", "user"):
                continue
            ev_type = ev.type.value if hasattr(ev.type, "value") else str(ev.type)
            if aid not in agent_activity:
                age = now - ev.timestamp
                status = (
                    "working" if age < 45 and ev_type in working_types else "idle"
                )
                agent_activity[aid] = {
                    "status": status,
                    "last_type": ev_type,
                    "last_content": ev.content[:120],
                    "target": ev.target,
                    "age": round(age),
                    "recent": [ev.content[:80]],
                }
            elif len(agent_activity[aid]["recent"]) < 3:
                agent_activity[aid]["recent"].append(ev.content[:80])

        return {
            "feature": board.feature,
            "current_phase": board.current_phase,
            "completed_phases": list(board.completed_phases),
            "files": files,
            "files_total": len(board.registry),
            "files_approved": approved_count,
            "total_cost": total_cost,
            "event_count": len(board.events),
            "crew": list(self._agents.values()),
            "signoffs": signoffs,
            "agent_activity": agent_activity,
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
