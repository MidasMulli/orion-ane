#!/usr/bin/env python3
"""
Heartbeat — Live monitoring dashboard for the Phantom/Midas agent framework.
==============================================================================

Single-file dashboard: aiohttp backend + embedded HTML/CSS/JS frontend.
Polls all subsystems every 2 seconds and renders a cyberpunk monitoring grid.

Usage:
    python heartbeat.py

Opens at http://localhost:8423

Subsystems monitored:
    - Four-path speculative decode server (port 8899)
    - Deterministic router (routing_log/)
    - Memory daemon (port 8422)
    - Enricher service (launchd, heartbeat file)
    - Hardware / process stats
"""

import asyncio
import glob
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from aiohttp import web

# ── Config ──────────────────────────────────────────────────────────────────

PORT = 8423
INFERENCE_SERVER = "http://127.0.0.1:8899"
MEMORY_SERVER = "http://127.0.0.1:8422"

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTING_LOG_DIR = os.path.join(AGENT_DIR, "routing_log")
DECISIONS_JSONL = os.path.join(ROUTING_LOG_DIR, "decisions.jsonl")
STATS_JSON = os.path.join(ROUTING_LOG_DIR, "stats.json")

VAULT_PATH = "/Users/midas/Desktop/cowork/vault"
ENRICHER_HEARTBEAT = os.path.join(VAULT_PATH, "midas", ".enricher_heartbeat")
ENRICHER_PID_FILE = "/tmp/phantom-enricher.pid"
INSIGHTS_DIR = os.path.join(VAULT_PATH, "memory", "insights")
RELATIONSHIPS_FILE = os.path.join(VAULT_PATH, "memory", "relationships.md")

# Key processes to monitor
WATCHED_PROCESSES = ["mlx", "ane_server", "enricher", "telegram_bot", "heartbeat", "four_path"]

# ── Ring Buffer ─────────────────────────────────────────────────────────────

inference_ring: deque = deque(maxlen=50)

# ── Data Gatherers ──────────────────────────────────────────────────────────

async def _http_get_json(url: str, timeout: float = 2.0) -> dict | None:
    """Fetch JSON from a URL. Returns None on any failure."""
    import aiohttp as ah
    try:
        async with ah.ClientSession() as s:
            async with s.get(url, timeout=ah.ClientTimeout(total=timeout)) as r:
                if r.status == 200:
                    return await r.json()
    except Exception:
        pass
    return None


async def gather_inference() -> dict:
    """Inference panel: four-path server health + ring buffer stats."""
    health = await _http_get_json(f"{INFERENCE_SERVER}/health")

    entries = list(inference_ring)
    recent_20 = entries[-20:] if len(entries) >= 20 else entries

    current_tps = entries[-1]["tok_per_sec"] if entries else None
    avg_tps = (
        round(sum(e["tok_per_sec"] for e in entries) / len(entries), 1)
        if entries else None
    )
    sparkline = [round(e["tok_per_sec"], 1) for e in recent_20]

    # Source totals from last request
    last_sources = entries[-1]["sources"] if entries else {}
    last_draft = entries[-1].get("draft_ratio", 0) if entries else 0
    last_total = entries[-1].get("total_tokens", 0) if entries else 0

    return {
        "health": health,
        "current_tps": round(current_tps, 1) if current_tps else None,
        "avg_tps": avg_tps,
        "sparkline": sparkline,
        "last_sources": last_sources,
        "last_draft_ratio": round(last_draft, 3) if last_draft else 0,
        "last_total_tokens": last_total,
        "buffer_size": len(entries),
    }


async def gather_router() -> dict:
    """Router panel: decision stats from feedback_loop JSONL files."""
    def _read():
        result = {
            "total_decisions": 0,
            "l1_count": 0,
            "l2_count": 0,
            "conv_count": 0,
            "accuracy_pct": None,
            "total_corrections": 0,
            "total_confirmations": 0,
            "recent": [],
            "most_corrected": [],
        }

        # Read stats.json
        try:
            with open(STATS_JSON) as f:
                stats = json.load(f)
            result["total_corrections"] = stats.get("total_corrections", 0)
            result["total_confirmations"] = stats.get("total_confirmations", 0)
            corrections_map = stats.get("corrections", {})
            if corrections_map:
                result["most_corrected"] = sorted(
                    corrections_map.items(), key=lambda x: x[1], reverse=True
                )[:5]
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Read decisions JSONL
        try:
            with open(DECISIONS_JSONL) as f:
                lines = f.readlines()
            result["total_decisions"] = len(lines)

            for line in lines:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("l1"):
                    result["l1_count"] += 1
                elif d.get("l2"):
                    result["l2_count"] += 1
                elif d.get("final") == "conversation":
                    result["conv_count"] += 1

            # Last 5 decisions
            tail = lines[-5:]
            for line in reversed(tail):
                try:
                    d = json.loads(line)
                    result["recent"].append({
                        "tool": d.get("final", "?"),
                        "msg": d.get("msg", "")[:80],
                        "l1": d.get("l1"),
                        "l2": d.get("l2"),
                        "ts": d.get("ts", ""),
                    })
                except json.JSONDecodeError:
                    continue

            total = result["total_decisions"]
            corr = result["total_corrections"]
            if total > 0:
                result["accuracy_pct"] = round((total - corr) / total * 100, 1)

        except FileNotFoundError:
            pass

        return result

    return await asyncio.to_thread(_read)


async def gather_memory() -> dict:
    """Memory panel: stats from memory daemon API."""
    data = await _http_get_json(f"{MEMORY_SERVER}/api/stats")
    if data:
        return {
            "online": True,
            "total": data.get("total_memories", 0),
            "ingested": data.get("ingested", 0),
            "extracted": data.get("extracted", 0),
            "stored": data.get("stored", 0),
            "deduped": data.get("deduped", 0),
            "superseded": data.get("superseded", 0),
            "entities": data.get("entity_count", 0),
            "session_id": data.get("session_id", ""),
        }
    return {"online": False}


async def gather_enricher() -> dict:
    """Enricher panel: heartbeat, PID, latest patterns."""
    def _read():
        result = {
            "status": "offline",
            "heartbeat_age": None,
            "pid": None,
            "pid_alive": False,
            "pattern_count": 0,
            "relationship_count": 0,
            "latest_sweep": None,
        }

        # Check PID
        try:
            with open(ENRICHER_PID_FILE) as f:
                pid = int(f.read().strip())
            result["pid"] = pid
            try:
                os.kill(pid, 0)
                result["pid_alive"] = True
            except (OSError, ProcessLookupError):
                pass
        except (FileNotFoundError, ValueError):
            pass

        # Heartbeat age
        try:
            mtime = os.path.getmtime(ENRICHER_HEARTBEAT)
            age = time.time() - mtime
            result["heartbeat_age"] = round(age, 1)
            if result["pid_alive"] and age < 600:
                result["status"] = "running"
            elif result["pid_alive"]:
                result["status"] = "stale"
        except FileNotFoundError:
            pass

        # Latest patterns file
        try:
            pattern_files = sorted(glob.glob(os.path.join(INSIGHTS_DIR, "patterns-*.md")))
            if pattern_files:
                latest = pattern_files[-1]
                result["latest_sweep"] = os.path.basename(latest)
                with open(latest) as f:
                    content = f.read()
                # Count pattern entries (lines starting with "- " or "## ")
                result["pattern_count"] = content.count("\n## ") + (1 if content.startswith("## ") else 0)
                if result["pattern_count"] == 0:
                    result["pattern_count"] = content.count("\n- ")
        except (FileNotFoundError, OSError):
            pass

        # Relationships
        try:
            with open(RELATIONSHIPS_FILE) as f:
                content = f.read()
            result["relationship_count"] = content.count("\n## ") + (1 if content.startswith("## ") else 0)
        except FileNotFoundError:
            pass

        return result

    return await asyncio.to_thread(_read)


async def gather_hardware() -> dict:
    """Hardware panel: memory, process list, optional powermetrics."""
    def _read():
        result = {
            "mem_total": None,
            "mem_used": None,
            "mem_pressure": None,
            "cpu_brand": None,
            "processes": [],
            "gpu_power": None,
            "ane_energy": None,
        }

        # sysctl info
        try:
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], timeout=2
            ).strip())
            result["mem_total"] = round(mem_bytes / (1024**3), 1)
        except Exception:
            pass

        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=2
            ).decode().strip()
            result["cpu_brand"] = brand
        except Exception:
            pass

        # vm_stat for memory pressure
        try:
            vm = subprocess.check_output(["vm_stat"], timeout=2).decode()
            page_size = 16384  # Apple Silicon default
            free = 0
            active = 0
            inactive = 0
            speculative = 0
            wired = 0
            compressed = 0
            for line in vm.splitlines():
                if "page size" in line.lower():
                    try:
                        page_size = int(line.split()[-2])
                    except (ValueError, IndexError):
                        pass
                if "Pages free:" in line:
                    free = int(line.split()[-1].rstrip("."))
                elif "Pages active:" in line:
                    active = int(line.split()[-1].rstrip("."))
                elif "Pages inactive:" in line:
                    inactive = int(line.split()[-1].rstrip("."))
                elif "Pages speculative:" in line:
                    speculative = int(line.split()[-1].rstrip("."))
                elif "Pages wired" in line:
                    wired = int(line.split()[-1].rstrip("."))
                elif "Pages occupied by compressor:" in line:
                    compressed = int(line.split()[-1].rstrip("."))

            used_pages = active + wired + compressed
            total_pages = free + active + inactive + speculative + wired + compressed
            if total_pages > 0:
                result["mem_used"] = round(used_pages * page_size / (1024**3), 1)
                result["mem_pressure"] = round(used_pages / total_pages * 100, 1)
        except Exception:
            pass

        # Process list
        try:
            ps_out = subprocess.check_output(
                ["ps", "aux"], timeout=3
            ).decode()
            for line in ps_out.splitlines()[1:]:
                cols = line.split(None, 10)
                if len(cols) < 11:
                    continue
                cmd = cols[10].lower()
                for name in WATCHED_PROCESSES:
                    if name in cmd and "grep" not in cmd:
                        rss_kb = int(cols[5])
                        result["processes"].append({
                            "name": name,
                            "pid": int(cols[1]),
                            "rss_mb": round(rss_kb / 1024, 1),
                            "cpu": cols[2],
                            "cmd": cols[10][:60],
                        })
                        break
        except Exception:
            pass

        # powermetrics (best effort, may need sudo)
        try:
            pm = subprocess.check_output(
                ["sudo", "-n", "powermetrics",
                 "--samplers", "gpu_power,ane_energy",
                 "-n", "1", "-i", "100"],
                timeout=3, stderr=subprocess.DEVNULL
            ).decode()
            for line in pm.splitlines():
                if "GPU Power:" in line:
                    try:
                        result["gpu_power"] = float(line.split(":")[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                if "ANE Energy:" in line or "ANE Power:" in line:
                    try:
                        result["ane_energy"] = float(line.split(":")[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass

        return result

    return await asyncio.to_thread(_read)


# ── API Handlers ────────────────────────────────────────────────────────────

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def handle_api_all(request: web.Request) -> web.Response:
    inference, router, memory, enricher, hardware = await asyncio.gather(
        gather_inference(),
        gather_router(),
        gather_memory(),
        gather_enricher(),
        gather_hardware(),
    )
    return web.json_response({
        "ts": datetime.now().isoformat(),
        "inference": inference,
        "router": router,
        "memory": memory,
        "enricher": enricher,
        "hardware": hardware,
    })


async def handle_inference_report(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    entry = {
        "tok_per_sec": data.get("tok_per_sec", 0),
        "sources": data.get("sources", {}),
        "draft_ratio": data.get("draft_ratio", 0),
        "total_tokens": data.get("total_tokens", 0),
        "elapsed": data.get("elapsed", 0),
        "ts": time.time(),
    }
    inference_ring.append(entry)
    return web.json_response({"ok": True, "buffer_size": len(inference_ring)})


# ── Dashboard HTML ──────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HEARTBEAT - Phantom/Midas</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

*{margin:0;padding:0;box-sizing:border-box}

:root{
  --bg:#0a0e14;
  --surface:#111820;
  --surface-2:#1a2230;
  --border:#1e2a3a;
  --text:#c5d0dc;
  --text-dim:#5c6a7a;
  --cyan:#00e5ff;
  --green:#00ff9d;
  --amber:#ffb300;
  --red:#ff3d71;
  --purple:#b388ff;
  --white:#e8edf3;
}

body{
  background:var(--bg);
  color:var(--text);
  font-family:'Inter',system-ui,sans-serif;
  font-size:13px;
  min-height:100vh;
}

/* ── Title Bar ── */
.title-bar{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 20px;
  border-bottom:1px solid var(--border);
  background:var(--surface);
}
.title-bar h1{
  font-family:'JetBrains Mono',monospace;
  font-size:16px;font-weight:700;
  letter-spacing:4px;color:var(--cyan);
  display:flex;align-items:center;gap:10px;
}
.pulse-dot{
  width:10px;height:10px;border-radius:50%;
  background:var(--green);
  animation:pulse 2s ease-in-out infinite;
}
@keyframes pulse{
  0%,100%{opacity:1;box-shadow:0 0 4px var(--green)}
  50%{opacity:.4;box-shadow:0 0 12px var(--green)}
}
.title-meta{
  font-family:'JetBrains Mono',monospace;
  font-size:11px;color:var(--text-dim);
}

/* ── Grid ── */
.grid{
  display:grid;
  grid-template-columns:1fr 1fr;
  grid-template-rows:auto auto auto;
  gap:1px;
  padding:1px;
  background:var(--border);
}
.grid>.panel:nth-child(1){grid-column:1/2}
.grid>.panel:nth-child(2){grid-column:2/3}
.grid>.panel:nth-child(3){grid-column:1/2}
.grid>.panel:nth-child(4){grid-column:2/3}
.grid>.panel:nth-child(5){grid-column:1/3}

@media(max-width:700px){
  .grid{grid-template-columns:1fr}
  .grid>.panel:nth-child(1),
  .grid>.panel:nth-child(2),
  .grid>.panel:nth-child(3),
  .grid>.panel:nth-child(4),
  .grid>.panel:nth-child(5){grid-column:1/2}
}

/* ── Panel ── */
.panel{
  background:var(--surface);
  padding:16px 18px;
  min-height:180px;
}
.panel-header{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:14px;
  padding-bottom:10px;
  border-bottom:1px solid var(--border);
}
.panel-title{
  font-family:'JetBrains Mono',monospace;
  font-size:11px;font-weight:700;
  letter-spacing:3px;color:var(--text-dim);
  text-transform:uppercase;
}
.status-dot{
  width:8px;height:8px;border-radius:50%;
  display:inline-block;
  flex-shrink:0;
}
.status-dot.green{background:var(--green);box-shadow:0 0 6px var(--green)}
.status-dot.amber{background:var(--amber);box-shadow:0 0 6px var(--amber)}
.status-dot.red{background:var(--red);box-shadow:0 0 6px var(--red)}
.status-dot.off{background:var(--text-dim)}

/* ── Big Number ── */
.big-num{
  font-family:'JetBrains Mono',monospace;
  font-size:36px;font-weight:700;
  color:var(--cyan);
  line-height:1;
  transition:color .3s,text-shadow .3s;
}
.big-num.flash{
  color:var(--green);
  text-shadow:0 0 20px var(--green);
}
.big-unit{
  font-size:14px;font-weight:400;
  color:var(--text-dim);
  margin-left:4px;
}
.stat-label{
  font-size:11px;color:var(--text-dim);
  margin-top:2px;
  font-family:'Inter',sans-serif;
}

/* ── Stat Row ── */
.stat-row{
  display:flex;align-items:baseline;gap:16px;
  margin-top:10px;flex-wrap:wrap;
}
.stat-item{display:flex;flex-direction:column;gap:2px;min-width:60px}
.stat-val{
  font-family:'JetBrains Mono',monospace;
  font-size:15px;font-weight:500;color:var(--text);
  transition:color .3s;
}
.stat-val.flash{color:var(--cyan)}

/* ── Source Bar ── */
.source-bar{
  display:flex;height:14px;border-radius:3px;overflow:hidden;
  background:var(--surface-2);margin-top:8px;
  width:100%;
}
.source-bar>div{
  height:100%;
  transition:width .5s ease;
  min-width:0;
}
.source-bar .s-ngram{background:var(--cyan)}
.source-bar .s-pld{background:#448aff}
.source-bar .s-mtp{background:var(--purple)}
.source-bar .s-ane{background:var(--amber)}
.source-bar .s-gpu{background:var(--white);opacity:.25}

.source-legend{
  display:flex;gap:12px;margin-top:6px;flex-wrap:wrap;
}
.source-legend span{
  display:flex;align-items:center;gap:4px;
  font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-dim);
}
.source-legend .dot{
  width:8px;height:8px;border-radius:2px;display:inline-block;
}

/* ── Sparkline ── */
.sparkline-wrap{
  margin-top:10px;
  height:44px;
  position:relative;
}
.sparkline-wrap canvas{
  width:100%;height:100%;
  display:block;
}

/* ── Path Dots ── */
.path-dots{
  display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;
}
.path-dot{
  display:flex;align-items:center;gap:5px;
  font-family:'JetBrains Mono',monospace;
  font-size:11px;color:var(--text-dim);
}

/* ── Decision List ── */
.decision-list{margin-top:10px}
.decision-item{
  display:flex;align-items:center;gap:8px;
  padding:5px 0;
  border-bottom:1px solid var(--border);
  font-size:12px;
}
.decision-item:last-child{border-bottom:none}
.decision-tool{
  font-family:'JetBrains Mono',monospace;
  font-size:11px;font-weight:600;
  color:var(--cyan);
  min-width:90px;
  flex-shrink:0;
}
.decision-msg{
  color:var(--text-dim);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  flex:1;
}
.decision-layer{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;
  color:var(--surface-2);
  background:var(--text-dim);
  border-radius:3px;
  padding:1px 5px;
  flex-shrink:0;
}

/* ── Horiz Bar (router split) ── */
.horiz-bar{
  display:flex;height:10px;border-radius:2px;overflow:hidden;
  background:var(--surface-2);margin-top:8px;width:100%;
}
.horiz-bar>div{height:100%;transition:width .5s ease}
.horiz-bar .bar-l1{background:var(--cyan)}
.horiz-bar .bar-l2{background:var(--purple)}
.horiz-bar .bar-conv{background:var(--text-dim)}

/* ── Corrected List ── */
.corrected-list{
  margin-top:10px;
}
.corrected-item{
  font-family:'JetBrains Mono',monospace;
  font-size:11px;color:var(--amber);
  padding:2px 0;
}
.corrected-count{color:var(--text-dim);margin-left:4px}

/* ── Process Table ── */
.proc-table{
  width:100%;border-collapse:collapse;margin-top:10px;
}
.proc-table th{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;font-weight:600;
  color:var(--text-dim);
  text-align:left;padding:4px 10px 4px 0;
  border-bottom:1px solid var(--border);
  letter-spacing:1px;
}
.proc-table td{
  font-family:'JetBrains Mono',monospace;
  font-size:11px;padding:4px 10px 4px 0;
  border-bottom:1px solid var(--border);
  color:var(--text);
}
.proc-name{color:var(--cyan)}

/* ── HW Stats Row ── */
.hw-stats{
  display:flex;gap:24px;flex-wrap:wrap;
  margin-bottom:10px;
}
.hw-stat{
  display:flex;flex-direction:column;gap:2px;
}
.hw-val{
  font-family:'JetBrains Mono',monospace;
  font-size:18px;font-weight:600;color:var(--text);
}
.hw-label{font-size:10px;color:var(--text-dim)}

/* ── Offline ── */
.offline-msg{
  font-family:'JetBrains Mono',monospace;
  font-size:12px;color:var(--text-dim);
  padding:20px 0;
  text-align:center;
}
</style>
</head>
<body>

<div class="title-bar">
  <h1><span class="pulse-dot"></span> HEARTBEAT</h1>
  <div class="title-meta" id="meta">connecting...</div>
</div>

<div class="grid">

  <!-- Panel 1: INFERENCE -->
  <div class="panel" id="p-inference">
    <div class="panel-header">
      <span class="panel-title">INFERENCE</span>
      <span class="status-dot off" id="inf-dot"></span>
    </div>
    <div id="inf-body"><div class="offline-msg">waiting for data...</div></div>
  </div>

  <!-- Panel 2: ROUTER -->
  <div class="panel" id="p-router">
    <div class="panel-header">
      <span class="panel-title">ROUTER</span>
      <span class="status-dot off" id="rtr-dot"></span>
    </div>
    <div id="rtr-body"><div class="offline-msg">no decisions yet</div></div>
  </div>

  <!-- Panel 3: MEMORY -->
  <div class="panel" id="p-memory">
    <div class="panel-header">
      <span class="panel-title">MEMORY</span>
      <span class="status-dot off" id="mem-dot"></span>
    </div>
    <div id="mem-body"><div class="offline-msg">daemon offline</div></div>
  </div>

  <!-- Panel 4: ENRICHER -->
  <div class="panel" id="p-enricher">
    <div class="panel-header">
      <span class="panel-title">ENRICHER</span>
      <span class="status-dot off" id="enr-dot"></span>
    </div>
    <div id="enr-body"><div class="offline-msg">offline</div></div>
  </div>

  <!-- Panel 5: HARDWARE -->
  <div class="panel" id="p-hardware">
    <div class="panel-header">
      <span class="panel-title">HARDWARE</span>
      <span class="status-dot off" id="hw-dot"></span>
    </div>
    <div id="hw-body"><div class="offline-msg">loading...</div></div>
  </div>

</div>

<script>
const API = '/api/all';
let prevData = null;
let sparkCanvas = null;

function setDot(id, color) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'status-dot ' + color;
}

function flash(el) {
  if (!el) return;
  el.classList.add('flash');
  setTimeout(() => el.classList.remove('flash'), 600);
}

function pct(a, b) {
  if (!b || b === 0) return '0';
  return Math.round(a / b * 100);
}

function fmtAge(seconds) {
  if (seconds == null) return '--';
  if (seconds < 60) return Math.round(seconds) + 's';
  if (seconds < 3600) return Math.round(seconds / 60) + 'm';
  return Math.round(seconds / 3600) + 'h';
}

function drawSparkline(canvasId, values, maxVal) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !values.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const max = maxVal || Math.max(...values, 1);
  const step = values.length > 1 ? w / (values.length - 1) : w;

  // Fill
  ctx.beginPath();
  ctx.moveTo(0, h);
  values.forEach((v, i) => {
    const x = i * step;
    const y = h - (v / max) * (h - 4);
    if (i === 0) ctx.lineTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo((values.length - 1) * step, h);
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,229,255,0.08)';
  ctx.fill();

  // Line
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = i * step;
    const y = h - (v / max) * (h - 4);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Last point dot
  if (values.length > 0) {
    const lastX = (values.length - 1) * step;
    const lastY = h - (values[values.length - 1] / max) * (h - 4);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#00e5ff';
    ctx.fill();
  }
}

function renderInference(d) {
  const inf = d.inference;
  const health = inf.health;
  const online = health && health.status === 'ok';

  setDot('inf-dot', online ? 'green' : 'red');

  if (!online && !inf.buffer_size) {
    document.getElementById('inf-body').innerHTML = '<div class="offline-msg">four-path server offline</div>';
    return;
  }

  const tps = inf.current_tps != null ? inf.current_tps : '--';
  const avg = inf.avg_tps != null ? inf.avg_tps : '--';
  const dr = inf.last_draft_ratio ? Math.round(inf.last_draft_ratio * 100) : 0;

  const src = inf.last_sources || {};
  const ngram = src.ngram || 0;
  const pld = src.prompt_lookup || 0;
  const mtp = src.mtp || 0;
  const ane = src.ane || 0;
  const gpu = src.gpu || 0;
  const total = ngram + pld + mtp + ane + gpu || 1;

  const model = health ? (health.model || '').split('/').pop() : '--';
  const paths = health || {};

  let html = `
    <div style="display:flex;align-items:baseline;gap:16px">
      <div>
        <span class="big-num" id="tps-val">${tps}</span>
        <span class="big-unit">tok/s</span>
        <div class="stat-label">current</div>
      </div>
      <div>
        <span class="stat-val" id="avg-val">${avg}</span>
        <span class="big-unit" style="font-size:11px">avg</span>
      </div>
      <div>
        <span class="stat-val">${dr}%</span>
        <span class="big-unit" style="font-size:11px">drafted</span>
      </div>
      <div>
        <span class="stat-val">${inf.buffer_size}</span>
        <span class="big-unit" style="font-size:11px">reqs</span>
      </div>
    </div>

    <div class="sparkline-wrap">
      <canvas id="spark-tps"></canvas>
    </div>

    <div class="source-bar">
      <div class="s-ngram" style="width:${pct(ngram, total)}%"></div>
      <div class="s-pld" style="width:${pct(pld, total)}%"></div>
      <div class="s-mtp" style="width:${pct(mtp, total)}%"></div>
      <div class="s-ane" style="width:${pct(ane, total)}%"></div>
      <div class="s-gpu" style="width:${pct(gpu, total)}%"></div>
    </div>
    <div class="source-legend">
      <span><span class="dot" style="background:var(--cyan)"></span>N-gram ${ngram}</span>
      <span><span class="dot" style="background:#448aff"></span>PLD ${pld}</span>
      <span><span class="dot" style="background:var(--purple)"></span>MTP ${mtp}</span>
      <span><span class="dot" style="background:var(--amber)"></span>ANE ${ane}</span>
      <span><span class="dot" style="background:var(--white);opacity:.4"></span>GPU ${gpu}</span>
    </div>

    <div class="path-dots">
      <div class="path-dot"><span class="status-dot ${paths.ngram_n ? 'green' : 'off'}"></span>N-gram (n=${paths.ngram_n || '--'})</div>
      <div class="path-dot"><span class="status-dot ${paths.mtp ? 'green' : 'off'}"></span>MTP</div>
      <div class="path-dot"><span class="status-dot ${paths.ane ? 'green' : 'off'}"></span>ANE</div>
    </div>
    <div style="margin-top:6px;font-size:10px;color:var(--text-dim);font-family:'JetBrains Mono',monospace">${model}</div>
  `;
  document.getElementById('inf-body').innerHTML = html;

  // Draw sparkline
  if (inf.sparkline && inf.sparkline.length > 1) {
    drawSparkline('spark-tps', inf.sparkline, null);
  }

  // Flash on change
  if (prevData && prevData.inference.current_tps !== inf.current_tps) {
    flash(document.getElementById('tps-val'));
  }
}

function renderRouter(d) {
  const r = d.router;
  const has = r.total_decisions > 0;
  setDot('rtr-dot', has ? 'green' : 'off');

  if (!has) {
    document.getElementById('rtr-body').innerHTML = '<div class="offline-msg">no decisions yet</div>';
    return;
  }

  const total = r.total_decisions;
  const l1 = r.l1_count || 0;
  const l2 = r.l2_count || 0;
  const conv = r.conv_count || 0;
  const acc = r.accuracy_pct != null ? r.accuracy_pct + '%' : '--';
  const corr = r.total_corrections || 0;
  const conf = r.total_confirmations || 0;

  let html = `
    <div style="display:flex;align-items:baseline;gap:16px">
      <div>
        <span class="big-num" style="font-size:28px" id="rtr-total">${total}</span>
        <div class="stat-label">decisions</div>
      </div>
      <div>
        <span class="stat-val" style="color:var(--green)">${acc}</span>
        <div class="stat-label">accuracy</div>
      </div>
      <div>
        <span class="stat-val" style="color:var(--amber)">${corr}</span>
        <div class="stat-label">corrections</div>
      </div>
      <div>
        <span class="stat-val" style="color:var(--green)">${conf}</span>
        <div class="stat-label">confirmed</div>
      </div>
    </div>

    <div class="horiz-bar">
      <div class="bar-l1" style="width:${pct(l1, total)}%" title="L1: ${l1}"></div>
      <div class="bar-l2" style="width:${pct(l2, total)}%" title="L2: ${l2}"></div>
      <div class="bar-conv" style="width:${pct(conv, total)}%" title="Conv: ${conv}"></div>
    </div>
    <div class="source-legend" style="margin-top:4px">
      <span><span class="dot" style="background:var(--cyan)"></span>L1 ${l1}</span>
      <span><span class="dot" style="background:var(--purple)"></span>L2 ${l2}</span>
      <span><span class="dot" style="background:var(--text-dim)"></span>Conv ${conv}</span>
    </div>
  `;

  // Most corrected
  if (r.most_corrected && r.most_corrected.length) {
    html += '<div class="corrected-list">';
    r.most_corrected.forEach(([tool, count]) => {
      html += `<div class="corrected-item">${tool}<span class="corrected-count">x${count}</span></div>`;
    });
    html += '</div>';
  }

  // Recent decisions
  if (r.recent && r.recent.length) {
    html += '<div class="decision-list">';
    r.recent.forEach(d => {
      const layer = d.l1 ? 'L1' : (d.l2 ? 'L2' : '--');
      html += `<div class="decision-item">
        <span class="decision-layer">${layer}</span>
        <span class="decision-tool">${d.tool}</span>
        <span class="decision-msg">${escHtml(d.msg)}</span>
      </div>`;
    });
    html += '</div>';
  }

  document.getElementById('rtr-body').innerHTML = html;
}

function renderMemory(d) {
  const m = d.memory;
  setDot('mem-dot', m.online ? 'green' : 'red');

  if (!m.online) {
    document.getElementById('mem-body').innerHTML = '<div class="offline-msg">daemon offline</div>';
    return;
  }

  const html = `
    <div>
      <span class="big-num" style="font-size:28px" id="mem-total">${m.total}</span>
      <div class="stat-label">total memories</div>
    </div>
    <div class="stat-row">
      <div class="stat-item">
        <span class="stat-val">${m.ingested}</span>
        <span class="stat-label">ingested</span>
      </div>
      <div class="stat-item">
        <span class="stat-val">${m.extracted}</span>
        <span class="stat-label">extracted</span>
      </div>
      <div class="stat-item">
        <span class="stat-val">${m.stored}</span>
        <span class="stat-label">stored</span>
      </div>
      <div class="stat-item">
        <span class="stat-val">${m.deduped}</span>
        <span class="stat-label">deduped</span>
      </div>
      <div class="stat-item">
        <span class="stat-val">${m.superseded}</span>
        <span class="stat-label">superseded</span>
      </div>
    </div>
    <div class="stat-row" style="margin-top:12px">
      <div class="stat-item">
        <span class="stat-val" style="color:var(--purple)">${m.entities}</span>
        <span class="stat-label">entities</span>
      </div>
      <div class="stat-item">
        <span class="stat-val" style="color:var(--text-dim);font-size:11px">${m.session_id || '--'}</span>
        <span class="stat-label">session</span>
      </div>
    </div>
  `;
  document.getElementById('mem-body').innerHTML = html;
}

function renderEnricher(d) {
  const e = d.enricher;
  const st = e.status;
  const dotColor = st === 'running' ? 'green' : st === 'stale' ? 'amber' : 'red';
  setDot('enr-dot', dotColor);

  const html = `
    <div style="display:flex;align-items:baseline;gap:12px">
      <div>
        <span class="stat-val" style="color:var(${st==='running'?'--green':st==='stale'?'--amber':'--red'});font-size:16px;font-weight:600">${st.toUpperCase()}</span>
        <div class="stat-label">status</div>
      </div>
      <div>
        <span class="stat-val">${e.pid || '--'}</span>
        <div class="stat-label">PID</div>
      </div>
    </div>
    <div class="stat-row">
      <div class="stat-item">
        <span class="stat-val">${fmtAge(e.heartbeat_age)}</span>
        <span class="stat-label">heartbeat age</span>
      </div>
      <div class="stat-item">
        <span class="stat-val" style="color:var(--cyan)">${e.pattern_count}</span>
        <span class="stat-label">patterns</span>
      </div>
      <div class="stat-item">
        <span class="stat-val" style="color:var(--purple)">${e.relationship_count}</span>
        <span class="stat-label">relationships</span>
      </div>
    </div>
    ${e.latest_sweep ? `<div style="margin-top:10px;font-size:10px;color:var(--text-dim);font-family:'JetBrains Mono',monospace">latest: ${escHtml(e.latest_sweep)}</div>` : ''}
  `;
  document.getElementById('enr-body').innerHTML = html;
}

function renderHardware(d) {
  const hw = d.hardware;
  setDot('hw-dot', 'green');

  let html = '<div class="hw-stats">';

  if (hw.cpu_brand) {
    html += `<div class="hw-stat"><span class="hw-val" style="font-size:13px;color:var(--cyan)">${escHtml(hw.cpu_brand)}</span><span class="hw-label">processor</span></div>`;
  }
  if (hw.mem_total != null) {
    html += `<div class="hw-stat"><span class="hw-val">${hw.mem_total} GB</span><span class="hw-label">total memory</span></div>`;
  }
  if (hw.mem_used != null) {
    html += `<div class="hw-stat"><span class="hw-val">${hw.mem_used} GB</span><span class="hw-label">used</span></div>`;
  }
  if (hw.mem_pressure != null) {
    const pColor = hw.mem_pressure > 85 ? 'var(--red)' : hw.mem_pressure > 70 ? 'var(--amber)' : 'var(--green)';
    html += `<div class="hw-stat"><span class="hw-val" style="color:${pColor}">${hw.mem_pressure}%</span><span class="hw-label">pressure</span></div>`;
  }
  if (hw.gpu_power != null) {
    html += `<div class="hw-stat"><span class="hw-val">${hw.gpu_power} mW</span><span class="hw-label">GPU power</span></div>`;
  }
  if (hw.ane_energy != null) {
    html += `<div class="hw-stat"><span class="hw-val">${hw.ane_energy} mJ</span><span class="hw-label">ANE energy</span></div>`;
  }
  html += '</div>';

  // Process table
  if (hw.processes && hw.processes.length) {
    html += `<table class="proc-table">
      <tr><th>PROCESS</th><th>PID</th><th>RSS</th><th>CPU</th><th>CMD</th></tr>`;
    hw.processes.forEach(p => {
      html += `<tr>
        <td class="proc-name">${escHtml(p.name)}</td>
        <td>${p.pid}</td>
        <td>${p.rss_mb} MB</td>
        <td>${p.cpu}%</td>
        <td style="color:var(--text-dim);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(p.cmd)}</td>
      </tr>`;
    });
    html += '</table>';
  } else {
    html += '<div style="font-size:11px;color:var(--text-dim);margin-top:4px">no watched processes found</div>';
  }

  document.getElementById('hw-body').innerHTML = html;
}

function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function poll() {
  try {
    const r = await fetch(API);
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();

    document.getElementById('meta').textContent =
      new Date(d.ts).toLocaleTimeString() + ' | poll ok';

    renderInference(d);
    renderRouter(d);
    renderMemory(d);
    renderEnricher(d);
    renderHardware(d);

    prevData = d;
  } catch (e) {
    document.getElementById('meta').textContent = 'poll failed: ' + e.message;
  }
}

// Initial + interval
poll();
setInterval(poll, 2000);
</script>

</body>
</html>"""


# ── Application ─────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/all", handle_api_all)
    app.router.add_post("/api/inference/report", handle_inference_report)
    return app


def main():
    print(f"\033[36m[heartbeat]\033[0m starting on http://localhost:{PORT}")
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)


if __name__ == "__main__":
    main()
