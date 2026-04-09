"""Subconscious Daemon — Main 34 Phase 3 (S2A-E).

Single-process coordinator that:
  1. Schedules maintenance.run_all() (replaces launchd while alive)
  2. Monitors tools/realtime_enricher.py subprocess (restarts on crash)
  3. Polls upstream health (72B :8899, 8B ANE :8891, Midas UI :8450)
  4. Tails the disk queue lifecycle log (queue_persistence.QUEUE_LOG)
     and emits queue_* events
  5. Hosts an in-process event bus broadcast over WebSocket
  6. Persists all events to vault/subconscious/event_log_YYYY-MM-DD.jsonl
  7. Exposes HTTP /api/subconscious/{status,health} and WS /api/subconscious/events

Design notes:
  * The daemon is an OBSERVER + COORDINATOR. It does not re-implement the
    research loop (that depends on llm_fn inside midas_ui). midas_ui owns
    the queue worker; the daemon owns the disk state log + event surface.
  * Maintenance scheduling is the one component the daemon takes over
    fully. The launchd entry remains as a failsafe — see directive S2A.
  * Threading model: aiohttp event loop in the main thread; all polling
    and disk-tailing work runs in worker threads that push events into
    a thread-safe `queue.Queue`. A bridge coroutine drains the queue
    and broadcasts to all WS subscribers + the disk log writer.
  * stdlib + aiohttp only.

Run:
    /Users/midas/.mlx-env/bin/python3 orion-ane/agent/subconscious_daemon.py
"""
from __future__ import annotations
import asyncio
import json
import os
import queue as _queue
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Make sibling modules importable
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/midas/Desktop/cowork/vault/subconscious")

from aiohttp import web, WSMsgType  # noqa: E402
import queue_persistence as qp  # noqa: E402

# ── Configuration ───────────────────────────────────────────────────────────
PORT = 8452
VAULT_ROOT = Path("/Users/midas/Desktop/cowork/vault")
EVENT_LOG_DIR = VAULT_ROOT / "subconscious"
REALTIME_ENRICHER = Path("/Users/midas/Desktop/cowork/tools/realtime_enricher.py")
PYTHON = "/Users/midas/.mlx-env/bin/python3"

MAINTENANCE_INTERVAL_S = 3600          # hourly
# Lowered from 30s → 2s on 2026-04-09 so the visualization can catch
# in-flight state on the 8B ANE server, whose embed calls finish in ~1
# second. The /health endpoint is a cheap GET on each upstream and the
# catch rate at 2s is good enough for both fast (8B embed) and slow (72B
# generation) workloads.
HEALTH_INTERVAL_S = 2
ENRICHER_RESTART_BACKOFF_S = 10
QUEUE_TAIL_INTERVAL_S = 1.0
EVENT_LOG_RETENTION_DAYS = 30
EVENT_QUEUE_MAX = 4096

# ── Event bus (sync side) ───────────────────────────────────────────────────
_bus: _queue.Queue = _queue.Queue(maxsize=EVENT_QUEUE_MAX)
_subscribers_lock = threading.Lock()
_subscribers: list[asyncio.Queue] = []  # async queues, populated by aiohttp loop

_state: dict[str, Any] = {
    "started_at": time.time(),
    "components": {
        "maintenance": {"status": "init", "last_run_ts": None,
                        "last_run_stats": None, "next_run_ts": None},
        "enricher": {"status": "init", "pid": None, "restarts": 0,
                     "last_event_ts": None},
        "health_monitor": {"status": "init", "last_check_ts": None,
                           "upstream": {}},
        "queue_observer": {"status": "init", "log_offset": 0,
                           "last_event_ts": None, "events_seen": 0},
        "event_bus": {"status": "init", "events_emitted": 0,
                      "queue_depth": 0, "subscribers": 0},
    },
}
_state_lock = threading.Lock()


def emit(ev_type: str, component: str, **details) -> None:
    """Push an event onto the bus. Safe to call from any thread."""
    rec = {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "type": ev_type,
        "component": component,
        "details": details,
    }
    try:
        _bus.put_nowait(rec)
    except _queue.Full:
        # Drop oldest by getting one then putting; cheap protection.
        try:
            _bus.get_nowait()
            _bus.put_nowait(rec)
        except Exception:
            pass
    with _state_lock:
        _state["components"]["event_bus"]["events_emitted"] += 1
        _state["components"]["event_bus"]["queue_depth"] = _bus.qsize()


# ── Worker: maintenance scheduler ───────────────────────────────────────────
def _maintenance_loop():
    with _state_lock:
        _state["components"]["maintenance"]["status"] = "running"
    # On boot: don't run immediately (avoid double-firing if launchd just ran).
    next_run = time.time() + 60
    while True:
        with _state_lock:
            _state["components"]["maintenance"]["next_run_ts"] = next_run
        sleep_s = max(1, next_run - time.time())
        time.sleep(sleep_s)
        t0 = time.time()
        try:
            from maintenance import run_all
            stats = run_all()
            duration_ms = int((time.time() - t0) * 1000)
            with _state_lock:
                comp = _state["components"]["maintenance"]
                comp["last_run_ts"] = t0
                comp["last_run_stats"] = stats
            emit("loop_fired", "maintenance",
                 loop_name="run_all",
                 duration_ms=duration_ms,
                 memories_scanned=stats.get("total_memories", 0),
                 memories_modified=(
                     stats.get("decayed", 0)
                     + stats.get("consolidated", 0)
                     + stats.get("vault_synced", 0)
                     + stats.get("semantic_superseded", 0)),
                 supersessions=stats.get("semantic_superseded", 0),
                 contradictions_found=stats.get("contradictions_resolved", 0),
                 errors=[])
        except Exception as e:
            emit("loop_fired", "maintenance",
                 loop_name="run_all",
                 duration_ms=int((time.time() - t0) * 1000),
                 errors=[f"{type(e).__name__}: {e}"])
            traceback.print_exc()
        next_run = time.time() + MAINTENANCE_INTERVAL_S


# ── Worker: realtime enricher subprocess monitor ────────────────────────────
def _enricher_monitor():
    if not REALTIME_ENRICHER.exists():
        with _state_lock:
            _state["components"]["enricher"]["status"] = "missing_script"
        return
    while True:
        try:
            proc = subprocess.Popen(
                [PYTHON, str(REALTIME_ENRICHER)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True)
        except Exception as e:
            emit("enricher_error", "enricher", error=str(e))
            time.sleep(ENRICHER_RESTART_BACKOFF_S)
            continue
        with _state_lock:
            comp = _state["components"]["enricher"]
            comp["status"] = "running"
            comp["pid"] = proc.pid
            comp["restarts"] += 1
        emit("enricher_started", "enricher", pid=proc.pid,
             restart_count=_state["components"]["enricher"]["restarts"])
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                with _state_lock:
                    _state["components"]["enricher"]["last_event_ts"] = time.time()
                low = line.lower()
                if "ingest" in low or "extract" in low or "memory" in low:
                    emit("enrichment_complete", "enricher", line=line[:240])
        except Exception as e:
            emit("enricher_error", "enricher", error=str(e))
        rc = proc.wait()
        with _state_lock:
            _state["components"]["enricher"]["status"] = f"exited_{rc}"
        emit("enricher_exited", "enricher", returncode=rc)
        time.sleep(ENRICHER_RESTART_BACKOFF_S)


# ── Worker: health poller ───────────────────────────────────────────────────
_HEALTH_TARGETS = {
    "qwen72b": "http://127.0.0.1:8899/health",
    "ane8b": "http://127.0.0.1:8891/health",
    "midas_ui": "http://127.0.0.1:8450/api/stats",
}


def _health_loop():
    with _state_lock:
        _state["components"]["health_monitor"]["status"] = "running"
    while True:
        snap: dict[str, Any] = {}
        for name, url in _HEALTH_TARGETS.items():
            t0 = time.time()
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    body = resp.read(8192)
                    entry = {"ok": True, "code": resp.status,
                             "latency_ms": int((time.time() - t0) * 1000),
                             "body_bytes": len(body)}
                    # Main 35 +5 Tier 1: parse upstream /health body to
                    # surface "thinking now" state. The 72B + 8B both
                    # report current_request_started_at + status which
                    # tells us if a request is in flight and for how long.
                    try:
                        parsed = json.loads(body.decode("utf-8", "replace"))
                        if isinstance(parsed, dict):
                            for k in ("status", "current_request_started_at",
                                       "current_request_n_msgs",
                                       "current_request_max_tokens",
                                       "request_count", "completed_count",
                                       "tasks_completed", "uptime"):
                                if k in parsed:
                                    entry[k] = parsed[k]
                            # Compute "in_flight_for" if currently generating
                            started = parsed.get("current_request_started_at")
                            if started and parsed.get("status") in ("generating", "queued"):
                                entry["in_flight_s"] = round(time.time() - float(started), 1)
                                if "_thinking_emitted" not in _state.setdefault("_thinking_track", {}).setdefault(name, {}):
                                    emit("upstream_thinking_started", "health_monitor",
                                         upstream=name,
                                         started_at=started,
                                         status=parsed.get("status"))
                                    _state["_thinking_track"][name]["_thinking_emitted"] = started
                                elif _state["_thinking_track"][name].get("_thinking_emitted") != started:
                                    # New request started — emit a fresh event
                                    emit("upstream_thinking_started", "health_monitor",
                                         upstream=name, started_at=started)
                                    _state["_thinking_track"][name]["_thinking_emitted"] = started
                            else:
                                # Idle — emit thinking_done if we previously tracked one
                                tt = _state.setdefault("_thinking_track", {}).setdefault(name, {})
                                if tt.get("_thinking_emitted"):
                                    emit("upstream_thinking_done", "health_monitor",
                                         upstream=name)
                                    tt["_thinking_emitted"] = None
                    except Exception:
                        pass
                    snap[name] = entry
            except Exception as e:
                snap[name] = {"ok": False, "error": str(e)[:120],
                              "latency_ms": int((time.time() - t0) * 1000)}
        with _state_lock:
            _state["components"]["health_monitor"]["last_check_ts"] = time.time()
            _state["components"]["health_monitor"]["upstream"] = snap
        emit("health_check", "health_monitor", upstream=snap)
        time.sleep(HEALTH_INTERVAL_S)


# ── Worker: queue log tailer ────────────────────────────────────────────────
def _queue_tail_loop():
    with _state_lock:
        _state["components"]["queue_observer"]["status"] = "running"
    offset = 0
    while True:
        try:
            for new_offset, rec in qp.tail_events(offset):
                offset = new_offset
                op = rec.get("op", "?")
                emit(f"queue_{op}", "queue_observer",
                     queue_id=rec.get("queue_id"),
                     task_idx=rec.get("task_idx"),
                     task_count=rec.get("task_count"))
                with _state_lock:
                    qo = _state["components"]["queue_observer"]
                    qo["log_offset"] = offset
                    qo["events_seen"] += 1
                    qo["last_event_ts"] = time.time()
        except Exception as e:
            emit("queue_observer_error", "queue_observer", error=str(e))
        time.sleep(QUEUE_TAIL_INTERVAL_S)


# ── Worker: event log writer with daily rotation ────────────────────────────
def _current_log_path() -> Path:
    EVENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return EVENT_LOG_DIR / f"event_log_{datetime.utcnow().date().isoformat()}.jsonl"


def _prune_old_logs():
    cutoff = datetime.utcnow().date() - timedelta(days=EVENT_LOG_RETENTION_DAYS)
    for p in EVENT_LOG_DIR.glob("event_log_*.jsonl"):
        try:
            datestr = p.stem.replace("event_log_", "")
            d = datetime.strptime(datestr, "%Y-%m-%d").date()
            if d < cutoff:
                p.unlink()
        except Exception:
            pass


# ── Async bridge: drain sync bus → broadcast + disk log ─────────────────────
async def _bus_bridge(loop: asyncio.AbstractEventLoop):
    _prune_old_logs()
    last_prune_day = datetime.utcnow().date()
    log_fh = open(_current_log_path(), "a")
    try:
        while True:
            try:
                rec = await loop.run_in_executor(None, _bus.get)
            except Exception:
                await asyncio.sleep(0.1)
                continue
            today = datetime.utcnow().date()
            if today != last_prune_day:
                try:
                    log_fh.close()
                except Exception:
                    pass
                log_fh = open(_current_log_path(), "a")
                last_prune_day = today
                _prune_old_logs()
            line = json.dumps(rec, separators=(",", ":"))
            try:
                log_fh.write(line + "\n")
                log_fh.flush()
            except Exception:
                pass
            with _subscribers_lock:
                subs = list(_subscribers)
                _state["components"]["event_bus"]["subscribers"] = len(subs)
                _state["components"]["event_bus"]["queue_depth"] = _bus.qsize()
            for q in subs:
                try:
                    q.put_nowait(rec)
                except asyncio.QueueFull:
                    pass
    finally:
        try:
            log_fh.close()
        except Exception:
            pass


# ── HTTP / WebSocket handlers ──────────────────────────────────────────────
async def handle_status(_request):
    with _state_lock:
        snap = json.loads(json.dumps(_state, default=str))
    snap["uptime_s"] = round(time.time() - snap["started_at"], 1)
    return web.json_response(snap)


async def handle_health(_request):
    out: dict[str, Any] = {"ts": time.time()}
    try:
        from semantic_supersede import count_unresolved_contradictions
        out["contradictions_unresolved"] = count_unresolved_contradictions()
    except Exception as e:
        out["contradictions_unresolved"] = -1
        out["contradictions_error"] = str(e)
    with _state_lock:
        out["upstream"] = _state["components"]["health_monitor"].get("upstream", {})
        out["components"] = {
            name: {"status": comp.get("status")}
            for name, comp in _state["components"].items()
        }
    return web.json_response(out)


async def handle_events(request):
    # Main 35 +5 fix: disabled heartbeat. The 30s ping / 15s pong-timeout
    # was killing healthy Safari WebSocket connections every ~50s — Safari
    # doesn't reliably pong inside the 15s window, server closed the
    # connection, viz reconnected, repeat. TCP keepalive handles truly
    # dead connections; the viz auto-reconnects on real failures.
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    sub: asyncio.Queue = asyncio.Queue(maxsize=512)
    with _subscribers_lock:
        _subscribers.append(sub)
    emit("ws_subscriber_joined", "event_bus", remote=str(request.remote))
    try:
        while not ws.closed:
            try:
                rec = await asyncio.wait_for(sub.get(), timeout=15.0)
                await ws.send_json(rec)
            except asyncio.TimeoutError:
                # heartbeat ping handled by aiohttp; loop continues
                continue
    except Exception:
        pass
    finally:
        with _subscribers_lock:
            try:
                _subscribers.remove(sub)
            except ValueError:
                pass
        emit("ws_subscriber_left", "event_bus", remote=str(request.remote))
    return ws


async def handle_emit(request):
    """Main 34 S3 KT4: external components push events into the bus.

    POST body: {"type": str, "component": str, "details": {...}}
    Used by midas_ui to publish memory_recalled events without holding
    a long-lived WS connection.
    """
    try:
        data = await request.json()
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    ev_type = data.get("type") or "external"
    component = data.get("component") or "external"
    details = data.get("details") or {}
    emit(ev_type, component, **details)
    return web.json_response({"ok": True})


async def handle_root(_request):
    return web.json_response({
        "name": "subconscious_daemon",
        "endpoints": [
            "GET  /api/subconscious/status",
            "GET  /api/subconscious/health",
            "WS   /api/subconscious/events",
        ],
    })


def _start_workers():
    for target, name in (
        (_maintenance_loop, "maintenance"),
        (_enricher_monitor, "enricher"),
        (_health_loop, "health"),
        (_queue_tail_loop, "queue_tail"),
    ):
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()


async def _on_startup(app):
    loop = asyncio.get_event_loop()
    app["bridge_task"] = loop.create_task(_bus_bridge(loop))
    _start_workers()
    with _state_lock:
        _state["components"]["event_bus"]["status"] = "running"
    emit("daemon_started", "daemon", pid=os.getpid())


async def _on_cleanup(app):
    task = app.get("bridge_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@web.middleware
async def cors_middleware(request, handler):
    """Main 35 +2: permissive CORS so the localhost:3000 viz can call
    the daemon's HTTP endpoints. WebSocket protocol bypasses CORS.
    """
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        try:
            resp = await handler(request)
        except web.HTTPException as e:
            resp = e
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


async def handle_graph(request):
    """Main 35 +2 Task 3 — knowledge graph for the visualization.

    Returns the same shape as tools/subconscious-viz/vault_graph.json
    (built by a static script). Live version reads the vault wikilink
    index + classifies topics on the fly. Cached for 60 s to keep
    polling cheap.
    """
    import re as _re
    cache = _state.get("_graph_cache")
    if cache and time.time() - cache["ts"] < 60:
        return web.json_response(cache["data"])

    from context_tracker import DEFAULT_TOPIC_KEYWORDS
    patterns = {
        t: _re.compile(r"\b(" + "|".join(_re.escape(k) for k in kws) + r")\b", _re.IGNORECASE)
        for t, kws in DEFAULT_TOPIC_KEYWORDS.items()
    }
    def classify(text):
        scores = {}
        for t, p in patterns.items():
            h = len(p.findall(text))
            if h:
                scores[t] = h
        return max(scores, key=scores.get) if scores else "general"

    nodes_by_id = {}
    edges_raw = []
    SKIP_DIRS = {".git", "__pycache__", ".obsidian", "archive"}
    # Main 35 +3 T2B: filter auto-generated noise from the graph.
    # 85% of the prior "general" cluster was stub files. The graph
    # is for meaningful research content, not extraction artifacts.
    SKIP_PATH_PREFIXES = (
        "memory/entities/",       # 324 entity stubs (Dentist, IDE, RSS, ...)
        "memory/sessions/",       # 30 daily session auto-files
        "agent_reports/queue/",   # queue task results / summaries
        "memory/.entities/",      # alternate entity dir
        "memory/facts/",          # auto-extracted fact stubs
        "memory/relationships/",  # auto-extracted relationship stubs
    )
    vault_root = VAULT_ROOT
    for root, dirs, files in os.walk(vault_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if not f.endswith(".md"):
                continue
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, vault_root).replace("\\", "/")
            if any(rel.startswith(p) for p in SKIP_PATH_PREFIXES):
                continue
            try:
                content = open(fpath, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            # Skip files too small to be meaningful (< 50 words)
            word_count = len(content.split())
            if word_count < 50:
                continue
            mc = max(1, word_count // 200)
            topic = classify(content[:5000])
            nodes_by_id[rel] = {"id": rel, "label": f.replace(".md", ""),
                                "topic": topic, "memory_count": mc}
            for link in _re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", content):
                edges_raw.append((rel, link.strip()))

    basename_to_id = {}
    for nid, n in nodes_by_id.items():
        basename_to_id.setdefault(n["label"], nid)
    edges = []
    for src, label in edges_raw:
        tgt = basename_to_id.get(label)
        if tgt and tgt != src:
            edges.append({"source": src, "target": tgt})
    data = {"nodes": list(nodes_by_id.values()), "edges": edges}
    _state["_graph_cache"] = {"ts": time.time(), "data": data}
    return web.json_response(data)


def make_app() -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/", handle_root)
    app.router.add_get("/api/subconscious/status", handle_status)
    app.router.add_get("/api/subconscious/health", handle_health)
    app.router.add_get("/api/subconscious/events", handle_events)
    app.router.add_post("/api/subconscious/emit", handle_emit)
    app.router.add_get("/api/subconscious/graph", handle_graph)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def main():
    app = make_app()
    print(f"subconscious_daemon listening on http://127.0.0.1:{PORT}")
    web.run_app(app, host="127.0.0.1", port=PORT, print=None,
                handle_signals=True)


if __name__ == "__main__":
    main()
