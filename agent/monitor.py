#!/usr/bin/env python3
"""
Phantom Monitor — System & Inference Dashboard
================================================

Real-time monitoring of GPU, CPU, ANE, and model inference metrics.
Runs alongside the Midas agent on port 8425.

Usage:
    python monitor.py
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import platform
from datetime import datetime

from aiohttp import web

# ── Config ──
PORT = 8425
METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")
ANE_SERVER = "http://localhost:8423"
MEMORY_DASHBOARD = "http://localhost:8422"

# ── System Metrics Collection ──

def get_system_metrics():
    """Collect system metrics without sudo."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {},
        "memory": {},
        "gpu": {},
        "ane": {},
        "processes": {},
    }

    # CPU usage via top
    try:
        out = subprocess.run(
            ["top", "-l", "1", "-n", "0", "-stats", "cpu"],
            capture_output=True, text=True, timeout=5
        )
        for line in out.stdout.split("\n"):
            if "CPU usage" in line:
                parts = line.split(",")
                for p in parts:
                    p = p.strip()
                    if "user" in p:
                        metrics["cpu"]["user"] = float(p.split("%")[0].split()[-1])
                    elif "sys" in p:
                        metrics["cpu"]["system"] = float(p.split("%")[0].split()[-1])
                    elif "idle" in p:
                        metrics["cpu"]["idle"] = float(p.split("%")[0].split()[-1])
    except:
        pass

    # Memory via vm_stat + sysctl
    try:
        # Total RAM
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
        total_bytes = int(out.stdout.strip())
        metrics["memory"]["total_gb"] = round(total_bytes / (1024**3), 1)

        # Memory pressure
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3)
        pages = {}
        for line in out.stdout.split("\n"):
            if ":" in line:
                key, val = line.split(":")
                key = key.strip().lower()
                val = val.strip().rstrip(".")
                try:
                    pages[key] = int(val)
                except:
                    pass

        page_size = 16384  # Apple Silicon page size
        free = pages.get("pages free", 0) * page_size
        active = pages.get("pages active", 0) * page_size
        inactive = pages.get("pages inactive", 0) * page_size
        wired = pages.get("pages wired down", 0) * page_size
        compressed = pages.get("pages occupied by compressor", 0) * page_size

        used_bytes = active + wired + compressed
        metrics["memory"]["used_gb"] = round(used_bytes / (1024**3), 1)
        metrics["memory"]["free_gb"] = round(free / (1024**3), 1)
        metrics["memory"]["pressure_pct"] = round(used_bytes / total_bytes * 100, 1)

        # Swap
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=3)
        swap_line = out.stdout.strip()
        for part in swap_line.split(","):
            part = part.strip()
            if part.startswith("used"):
                val = part.split("=")[1].strip().rstrip("M").strip()
                metrics["memory"]["swap_mb"] = float(val)
    except:
        pass

    # GPU via ioreg (Apple Silicon integrated GPU)
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            capture_output=True, text=True, timeout=3
        )
        for line in out.stdout.split("\n"):
            if "gpu-core-count" in line.lower() or "GPUCoreCoun" in line:
                val = line.split("=")[-1].strip().strip('"')
                try:
                    metrics["gpu"]["cores"] = int(val)
                except:
                    pass
            if "PerformanceStatistics" in line:
                # Try to extract utilization
                pass

        # GPU power from ioreg
        out2 = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AppleARMIODevice"],
            capture_output=True, text=True, timeout=3
        )
        # Get GPU-related frequency if available
        metrics["gpu"]["chip"] = platform.processor() or "Apple Silicon"
    except:
        pass

    # Process-level metrics for MLX and daemon
    try:
        out = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=3
        )
        for line in out.stdout.split("\n"):
            if "mlx_lm" in line and "server" in line and "grep" not in line:
                parts = line.split()
                if len(parts) > 5:
                    try:
                        rss_kb = int(parts[5])
                        rss_mb = round(rss_kb / 1024, 1)
                    except:
                        rss_mb = 0
                    metrics["processes"]["mlx_server"] = {
                        "pid": parts[1],
                        "cpu_pct": float(parts[2]),
                        "mem_pct": float(parts[3]),
                        "rss_mb": rss_mb,
                    }
            elif "mcp_server.py" in line:
                parts = line.split()
                if len(parts) > 5:
                    metrics["processes"]["memory_daemon"] = {
                        "pid": parts[1],
                        "cpu_pct": float(parts[2]),
                        "mem_pct": float(parts[3]),
                    }
            elif "dashboard.py" in line:
                parts = line.split()
                if len(parts) > 5:
                    metrics["processes"]["dashboard"] = {
                        "pid": parts[1],
                        "cpu_pct": float(parts[2]),
                        "mem_pct": float(parts[3]),
                    }
    except:
        pass

    # ANE server health
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"{ANE_SERVER}/health", timeout=1)
        data = json.loads(resp.read())
        metrics["ane"]["status"] = "online"
        metrics["ane"]["backend"] = data.get("backend", "unknown")
        metrics["ane"]["uptime_s"] = data.get("uptime_s", 0)
    except:
        metrics["ane"]["status"] = "offline"

    return metrics


# ── Dashboard HTML ──

MONITOR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phantom Monitor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #0a0e14;
    --surface: #111820;
    --surface-2: #1a2230;
    --border: #1e2a3a;
    --text: #c5d0dc;
    --text-dim: #5c6a7a;
    --cyan: #00e5ff;
    --green: #00ff9d;
    --amber: #ffb300;
    --red: #ff3d71;
    --purple: #b388ff;
    --blue: #448aff;
    --cyan-dim: rgba(0, 229, 255, 0.15);
    --green-dim: rgba(0, 255, 157, 0.15);
    --amber-dim: rgba(255, 179, 0, 0.15);
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    height: 100vh;
    overflow: hidden;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 700;
    color: var(--cyan);
    text-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
  }
  .status-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    border: 1px solid;
  }
  .status-online {
    background: var(--green-dim);
    color: var(--green);
    border-color: rgba(0, 255, 157, 0.3);
  }
  .status-offline {
    background: rgba(255, 61, 113, 0.1);
    color: var(--red);
    border-color: rgba(255, 61, 113, 0.3);
  }
  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }
  .status-dot.online { background: var(--green); }
  .status-dot.offline { background: var(--red); }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto 1fr;
    height: calc(100vh - 49px);
    gap: 1px;
    background: var(--border);
  }

  .panel {
    background: var(--surface);
    padding: 16px;
    overflow-y: auto;
  }
  .panel-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* ── Metric Cards ── */
  .metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .metric-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
  }
  .metric-card.wide { grid-column: span 2; }
  .metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
  }
  .metric-value.cyan { color: var(--cyan); }
  .metric-value.green { color: var(--green); }
  .metric-value.amber { color: var(--amber); }
  .metric-value.purple { color: var(--purple); }
  .metric-value.red { color: var(--red); }
  .metric-unit {
    font-size: 11px;
    font-weight: 400;
    color: var(--text-dim);
    margin-left: 4px;
  }
  .metric-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 4px;
  }

  /* ── Progress Bars ── */
  .bar-container {
    margin-top: 8px;
  }
  .bar-label {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-dim);
    margin-bottom: 4px;
  }
  .bar-track {
    height: 6px;
    background: var(--bg);
    border-radius: 3px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
  }
  .bar-fill.cyan { background: var(--cyan); }
  .bar-fill.green { background: var(--green); }
  .bar-fill.amber { background: var(--amber); }
  .bar-fill.red { background: var(--red); }
  .bar-fill.purple { background: var(--purple); }

  /* ── Sparkline Chart ── */
  .chart-container {
    margin-top: 12px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }
  .chart-canvas {
    width: 100%;
    height: 80px;
  }

  /* ── Tier Status ── */
  .tier {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 8px;
  }
  .tier-icon {
    font-size: 20px;
    width: 32px;
    text-align: center;
  }
  .tier-info { flex: 1; }
  .tier-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
  }
  .tier-detail {
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 2px;
  }
  .tier-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
  }

  /* ── Token Flow Log ── */
  .flow-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    max-height: 200px;
    overflow-y: auto;
  }
  .flow-entry {
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .flow-entry:last-child { border-bottom: none; }
  .flow-tokens { color: var(--cyan); }
  .flow-speed { color: var(--green); }
  .flow-time { color: var(--text-dim); font-size: 10px; }

  /* ── Model Info ── */
  .model-info {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
  }
  .model-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--cyan);
    margin-bottom: 6px;
  }
  .model-detail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-dim);
    line-height: 1.6;
  }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <span class="logo">◆ PHANTOM MONITOR</span>
    <span id="gpu-badge" class="status-badge status-offline">
      <span class="status-dot offline" id="gpu-dot"></span>
      <span id="gpu-status-text">CHECKING</span>
    </span>
  </div>
  <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-dim);">
    <span id="header-time"></span>
  </div>
</div>

<div class="main">
  <!-- TOP LEFT: Hardware Tiers -->
  <div class="panel">
    <div class="panel-header">COMPUTE TIERS</div>

    <div class="tier">
      <div class="tier-icon">🔥</div>
      <div class="tier-info">
        <div class="tier-name" style="color: var(--cyan);">GPU — Metal / MLX</div>
        <div class="tier-detail" id="gpu-detail">Qwen 3.5 9B 4-bit</div>
      </div>
      <div class="tier-status" id="gpu-tier-status" style="color: var(--green);">—</div>
    </div>

    <div class="tier">
      <div class="tier-icon">🧠</div>
      <div class="tier-info">
        <div class="tier-name" style="color: var(--green);">CPU — Memory Daemon</div>
        <div class="tier-detail">sentence-transformers + ChromaDB</div>
      </div>
      <div class="tier-status" id="cpu-tier-status" style="color: var(--green);">—</div>
    </div>

    <div class="tier">
      <div class="tier-icon">⚡</div>
      <div class="tier-info">
        <div class="tier-name" style="color: var(--purple);">ANE — Neural Engine</div>
        <div class="tier-detail" id="ane-detail">Classifier</div>
      </div>
      <div class="tier-status" id="ane-tier-status" style="color: var(--text-dim);">—</div>
    </div>

    <!-- Memory Bars -->
    <div style="margin-top: 16px;">
      <div class="bar-container">
        <div class="bar-label">
          <span>UNIFIED MEMORY</span>
          <span id="mem-label">— / — GB</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill cyan" id="mem-bar" style="width: 0%"></div>
        </div>
      </div>
      <div class="bar-container">
        <div class="bar-label">
          <span>CPU</span>
          <span id="cpu-label">—%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill green" id="cpu-bar" style="width: 0%"></div>
        </div>
      </div>
      <div class="bar-container">
        <div class="bar-label">
          <span>MLX SERVER RSS</span>
          <span id="mlx-mem-label">— MB</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill amber" id="mlx-mem-bar" style="width: 0%"></div>
        </div>
      </div>
      <div class="bar-container">
        <div class="bar-label">
          <span>SWAP</span>
          <span id="swap-label">— MB</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill red" id="swap-bar" style="width: 0%"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- TOP RIGHT: Inference Metrics -->
  <div class="panel">
    <div class="panel-header">INFERENCE METRICS</div>

    <div id="model-info" class="model-info">
      <div class="model-name" id="model-name">—</div>
      <div class="model-detail" id="model-details">Waiting for first inference...</div>
    </div>

    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">GEN SPEED</div>
        <div class="metric-value cyan" id="gen-tps">—<span class="metric-unit">tok/s</span></div>
        <div class="metric-sub" id="gen-tps-avg">avg: —</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">PROMPT SPEED</div>
        <div class="metric-value green" id="prompt-tps">—<span class="metric-unit">tok/s</span></div>
        <div class="metric-sub" id="prompt-tps-avg">avg: —</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">LATEST LATENCY</div>
        <div class="metric-value amber" id="latency">—<span class="metric-unit">s</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-label">PEAK GEN</div>
        <div class="metric-value purple" id="peak-tps">—<span class="metric-unit">tok/s</span></div>
      </div>
    </div>

    <div class="chart-container">
      <div class="panel-header" style="margin-bottom: 8px;">GENERATION tok/s</div>
      <canvas class="chart-canvas" id="tps-chart"></canvas>
    </div>
  </div>

  <!-- BOTTOM LEFT: Token Budget -->
  <div class="panel">
    <div class="panel-header">SESSION TOKEN BUDGET</div>

    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">TOTAL TOKENS</div>
        <div class="metric-value cyan" id="total-tokens">0</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">INFERENCES</div>
        <div class="metric-value green" id="total-inferences">0</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">PROMPT TOKENS</div>
        <div class="metric-value" style="color: var(--text);" id="total-prompt">0</div>
        <div class="metric-sub" id="prompt-pct">—% of total</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">COMPLETION TOKENS</div>
        <div class="metric-value" style="color: var(--text);" id="total-completion">0</div>
        <div class="metric-sub" id="completion-pct">—% of total</div>
      </div>
    </div>

    <div class="chart-container">
      <div class="panel-header" style="margin-bottom: 8px;">TOKEN FLOW (prompt vs completion)</div>
      <canvas class="chart-canvas" id="token-chart"></canvas>
    </div>
  </div>

  <!-- BOTTOM RIGHT: Inference Log -->
  <div class="panel">
    <div class="panel-header">
      INFERENCE LOG
      <span style="color: var(--cyan);" id="log-count">0</span>
    </div>
    <div class="flow-log" id="flow-log">
      <div style="color: var(--text-dim); text-align: center; padding: 40px 0;">
        Waiting for inference...
      </div>
    </div>
  </div>
</div>

<script>
  // ── State ──
  let lastMetricsTimestamp = '';
  let tpsHistory = [];
  let promptHistory = [];
  let completionHistory = [];

  // ── Polling ──
  async function poll() {
    try {
      const [sysRes, metricsRes] = await Promise.all([
        fetch('/api/system').catch(() => null),
        fetch('/api/metrics').catch(() => null),
      ]);

      if (sysRes && sysRes.ok) updateSystem(await sysRes.json());
      if (metricsRes && metricsRes.ok) updateMetrics(await metricsRes.json());
    } catch(e) {}

    document.getElementById('header-time').textContent =
      new Date().toLocaleTimeString('en-US', { hour12: true });
  }

  function updateSystem(sys) {
    // Memory
    const used = sys.memory?.used_gb || 0;
    const total = sys.memory?.total_gb || 16;
    const pct = sys.memory?.pressure_pct || 0;
    document.getElementById('mem-label').textContent = `${used} / ${total} GB`;
    document.getElementById('mem-bar').style.width = pct + '%';
    document.getElementById('mem-bar').className = 'bar-fill ' + (pct > 90 ? 'red' : pct > 75 ? 'amber' : 'cyan');

    // CPU
    const cpuUsed = (sys.cpu?.user || 0) + (sys.cpu?.system || 0);
    document.getElementById('cpu-label').textContent = cpuUsed.toFixed(1) + '%';
    document.getElementById('cpu-bar').style.width = Math.min(cpuUsed, 100) + '%';
    document.getElementById('cpu-bar').className = 'bar-fill ' + (cpuUsed > 80 ? 'red' : cpuUsed > 50 ? 'amber' : 'green');

    // MLX server memory
    const mlxRss = sys.processes?.mlx_server?.rss_mb || 0;
    document.getElementById('mlx-mem-label').textContent = mlxRss.toFixed(0) + ' MB';
    document.getElementById('mlx-mem-bar').style.width = Math.min(mlxRss / (total * 1024) * 100, 100) + '%';

    // Swap
    const swap = sys.memory?.swap_mb || 0;
    document.getElementById('swap-label').textContent = swap.toFixed(0) + ' MB';
    document.getElementById('swap-bar').style.width = Math.min(swap / 1024 * 100, 100) + '%';
    document.getElementById('swap-bar').className = 'bar-fill ' + (swap > 500 ? 'red' : swap > 100 ? 'amber' : 'green');

    // GPU tier status
    const mlxProc = sys.processes?.mlx_server;
    if (mlxProc) {
      document.getElementById('gpu-tier-status').textContent = `PID ${mlxProc.pid}`;
      document.getElementById('gpu-tier-status').style.color = 'var(--green)';
      document.getElementById('gpu-badge').className = 'status-badge status-online';
      document.getElementById('gpu-dot').className = 'status-dot online';
      document.getElementById('gpu-status-text').textContent = 'GPU ONLINE';
    } else {
      document.getElementById('gpu-tier-status').textContent = 'OFFLINE';
      document.getElementById('gpu-tier-status').style.color = 'var(--red)';
      document.getElementById('gpu-badge').className = 'status-badge status-offline';
      document.getElementById('gpu-dot').className = 'status-dot offline';
      document.getElementById('gpu-status-text').textContent = 'GPU OFFLINE';
    }

    // CPU tier
    const daemonProc = sys.processes?.memory_daemon;
    document.getElementById('cpu-tier-status').textContent = daemonProc ? `PID ${daemonProc.pid}` : 'IN-PROC';
    document.getElementById('cpu-tier-status').style.color = 'var(--green)';

    // ANE tier
    const aneStatus = sys.ane?.status || 'offline';
    document.getElementById('ane-tier-status').textContent = aneStatus.toUpperCase();
    document.getElementById('ane-tier-status').style.color = aneStatus === 'online' ? 'var(--purple)' : 'var(--text-dim)';
    if (sys.ane?.backend) {
      document.getElementById('ane-detail').textContent = `${sys.ane.backend} classifier`;
    }
  }

  function updateMetrics(m) {
    if (!m || !m.latest) return;

    // Model info
    document.getElementById('model-name').textContent = m.model || '—';
    document.getElementById('model-details').textContent = `endpoint: ${m.mlx_endpoint || '—'}`;
    document.getElementById('gpu-detail').textContent = m.model || 'Qwen 3.5 9B 4-bit';

    const latest = m.latest;
    const totals = m.session_totals || {};

    // Speed metrics
    document.getElementById('gen-tps').innerHTML = `${latest.gen_tok_s}<span class="metric-unit">tok/s</span>`;
    document.getElementById('gen-tps-avg').textContent = `avg: ${totals.avg_gen_tok_s || '—'}`;
    document.getElementById('prompt-tps').innerHTML = `${latest.prompt_tok_s}<span class="metric-unit">tok/s</span>`;
    document.getElementById('prompt-tps-avg').textContent = `avg: ${totals.avg_prompt_tok_s || '—'}`;
    document.getElementById('latency').innerHTML = `${latest.elapsed_s}<span class="metric-unit">s</span>`;
    document.getElementById('peak-tps').innerHTML = `${totals.peak_gen_tok_s || '—'}<span class="metric-unit">tok/s</span>`;

    // Session totals
    document.getElementById('total-tokens').textContent = (totals.total_tokens || 0).toLocaleString();
    document.getElementById('total-inferences').textContent = totals.total_inferences || 0;
    document.getElementById('total-prompt').textContent = (totals.total_prompt_tokens || 0).toLocaleString();
    document.getElementById('total-completion').textContent = (totals.total_completion_tokens || 0).toLocaleString();

    const tt = totals.total_tokens || 1;
    document.getElementById('prompt-pct').textContent =
      `${((totals.total_prompt_tokens || 0) / tt * 100).toFixed(1)}% of total`;
    document.getElementById('completion-pct').textContent =
      `${((totals.total_completion_tokens || 0) / tt * 100).toFixed(1)}% of total`;

    // Update charts from history
    if (m.history && m.history.length > 0) {
      tpsHistory = m.history.map(h => h.gen_tok_s);
      promptHistory = m.history.map(h => h.prompt_tokens);
      completionHistory = m.history.map(h => h.completion_tokens);
      drawChart('tps-chart', tpsHistory, 'cyan');
      drawTokenChart('token-chart', promptHistory, completionHistory);

      // Log count
      document.getElementById('log-count').textContent = m.history.length;

      // Update log
      const logEl = document.getElementById('flow-log');
      logEl.innerHTML = m.history.slice().reverse().map(h => {
        const t = new Date(h.timestamp).toLocaleTimeString('en-US', { hour12: false });
        return `<div class="flow-entry">
          <span class="flow-time">${t}</span>
          <span class="flow-tokens">${h.prompt_tokens}→${h.completion_tokens} tok</span>
          <span class="flow-speed">${h.gen_tok_s} t/s</span>
          <span class="flow-time">${h.elapsed_s}s</span>
        </div>`;
      }).join('');
    }
  }

  // ── Charts ──
  function drawChart(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = 2;

    ctx.clearRect(0, 0, w, h);

    if (data.length < 2) return;

    const max = Math.max(...data) * 1.1 || 1;
    const step = (w - pad * 2) / (data.length - 1);

    const colors = { cyan: '#00e5ff', green: '#00ff9d', amber: '#ffb300', purple: '#b388ff' };
    const c = colors[color] || colors.cyan;

    // Fill
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    data.forEach((v, i) => {
      ctx.lineTo(pad + i * step, h - pad - (v / max) * (h - pad * 2));
    });
    ctx.lineTo(pad + (data.length - 1) * step, h - pad);
    ctx.closePath();
    ctx.fillStyle = c.replace(')', ', 0.1)').replace('rgb', 'rgba').replace('#', '');
    // Simpler: just use a low-opacity version
    ctx.globalAlpha = 0.15;
    ctx.fillStyle = c;
    ctx.fill();
    ctx.globalAlpha = 1;

    // Line
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad + i * step;
      const y = h - pad - (v / max) * (h - pad * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = c;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Dots
    data.forEach((v, i) => {
      const x = pad + i * step;
      const y = h - pad - (v / max) * (h - pad * 2);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = c;
      ctx.fill();
    });
  }

  function drawTokenChart(canvasId, prompts, completions) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = 2;
    const n = prompts.length;

    if (n < 1) return;

    ctx.clearRect(0, 0, w, h);

    const maxVal = Math.max(...prompts, ...completions) * 1.1 || 1;
    const barW = Math.max(4, (w - pad * 2) / n - 2);

    for (let i = 0; i < n; i++) {
      const x = pad + i * ((w - pad * 2) / n);

      // Prompt bar (cyan)
      const ph = (prompts[i] / maxVal) * (h - pad * 2);
      ctx.fillStyle = 'rgba(0, 229, 255, 0.4)';
      ctx.fillRect(x, h - pad - ph, barW / 2, ph);

      // Completion bar (green)
      const ch = (completions[i] / maxVal) * (h - pad * 2);
      ctx.fillStyle = 'rgba(0, 255, 157, 0.4)';
      ctx.fillRect(x + barW / 2, h - pad - ch, barW / 2, ch);
    }
  }

  // ── Start polling ──
  poll();
  setInterval(poll, 2000);
</script>

</body>
</html>
"""


# ── API Handlers ──

async def handle_index(request):
    return web.Response(text=MONITOR_HTML, content_type="text/html")


async def handle_system(request):
    """System metrics — CPU, memory, GPU, processes."""
    loop = asyncio.get_event_loop()
    metrics = await loop.run_in_executor(None, get_system_metrics)
    return web.json_response(metrics)


async def handle_metrics(request):
    """Agent inference metrics — read from shared file."""
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r") as f:
                data = json.load(f)
            return web.json_response(data)
        else:
            return web.json_response({"error": "no metrics yet"})
    except Exception as e:
        return web.json_response({"error": str(e)})


# ── Server ──
app = web.Application()
app.router.add_get('/', handle_index)
app.router.add_get('/api/system', handle_system)
app.router.add_get('/api/metrics', handle_metrics)

def snapshot():
    """Print current system + routing stats and exit. For Midas self-observation."""
    sys_metrics = get_system_metrics()

    lines = []
    lines.append("=== PHANTOM MONITOR SNAPSHOT ===")
    lines.append("")

    # System
    mem = sys_metrics.get("memory", {})
    cpu = sys_metrics.get("cpu", {})
    lines.append(f"Memory: {mem.get('used_gb', '?')}/{mem.get('total_gb', '?')} GB ({mem.get('pressure_pct', '?')}%) | Swap: {mem.get('swap_mb', 0):.0f} MB")
    cpu_used = (cpu.get("user", 0) + cpu.get("system", 0))
    lines.append(f"CPU: {cpu_used:.1f}% used | Idle: {cpu.get('idle', 0):.1f}%")

    # Processes
    procs = sys_metrics.get("processes", {})
    mlx = procs.get("mlx_server")
    if mlx:
        lines.append(f"MLX Server: PID {mlx['pid']}, {mlx['rss_mb']:.0f} MB RSS, {mlx['cpu_pct']}% CPU")
    else:
        lines.append("MLX Server: NOT RUNNING")

    # ANE
    ane = sys_metrics.get("ane", {})
    lines.append(f"ANE: {ane.get('status', 'unknown')}" + (f" ({ane.get('backend', '')})" if ane.get("backend") else ""))

    # Routing stats (from feedback_loop)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from feedback_loop import get_routing_stats
        rs = get_routing_stats()
        lines.append("")
        lines.append("=== ROUTING ===")
        total_d = rs.get("total_decisions", 0)
        lines.append(f"Decisions: {total_d} | Corrections: {rs.get('total_corrections', 0)} | Confirmations: {rs.get('total_confirmations', 0)}")
        if rs.get("accuracy_pct") is not None:
            lines.append(f"Accuracy: {rs['accuracy_pct']}%")
        most = rs.get("most_corrected", [])
        if most:
            lines.append(f"Most corrected: {', '.join(f'{t}({c})' for t, c in most)}")
    except Exception:
        pass

    # Inference metrics
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE) as f:
                m = json.load(f)
            lines.append("")
            lines.append("=== INFERENCE ===")
            if m.get("latest"):
                lat = m["latest"]
                lines.append(f"Last gen: {lat.get('gen_tok_s', '?')} tok/s | Prompt: {lat.get('prompt_tok_s', '?')} tok/s | Latency: {lat.get('elapsed_s', '?')}s")
            totals = m.get("session_totals", {})
            if totals:
                lines.append(f"Session: {totals.get('total_inferences', 0)} inferences, {totals.get('total_tokens', 0)} tokens")
                lines.append(f"Avg gen: {totals.get('avg_gen_tok_s', '?')} tok/s | Peak: {totals.get('peak_gen_tok_s', '?')} tok/s")
    except Exception:
        pass

    print("\n".join(lines))


if __name__ == '__main__':
    if "--snapshot" in sys.argv:
        snapshot()
        sys.exit(0)

    print(f"\033[36m  ╔══════════════════════════════╗\033[0m")
    print(f"\033[36m  ║   \033[1m◆ PHANTOM MONITOR ◆\033[0m\033[36m        ║\033[0m")
    print(f"\033[36m  ╚══════════════════════════════╝\033[0m")
    print(f"  \033[2mhttp://localhost:{PORT}\033[0m")
    print()
    web.run_app(app, port=PORT, print=None)
