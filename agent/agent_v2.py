#!/usr/bin/env python3
"""
Midas Agent v2 — Deterministic Router Architecture
====================================================

Architecture:
  Layer 1: Keyword routing (router.py) — instant, zero LLM calls
  Layer 2: LLM single-word classification (router.py) — only when L1 misses
  Layer 3: Tool execution (tool_executor.py) — code-constructed calls
  Layer 4: LLM text generation (synthesizer.py) — text only, no tool decisions

The LLM never decides which tool to call or constructs arguments.
"""

import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import urllib.error
import warnings
from datetime import datetime

from openai import OpenAI

# ── Local modules ────────────────────────────────────────────────────────────

from router import route, layer1_route, layer2_classify
from tool_executor import execute, set_memory, set_browser
from synthesizer import synthesize, SYSTEM_PROMPT
from feedback_loop import log_decision, detect_feedback, log_correction, get_routing_stats

# ── Config ───────────────────────────────────────────────────────────────────

MLX_BASE_URL = os.environ.get("MLX_BASE_URL", "http://127.0.0.1:8899/v1")
MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")
VAULT_PATH = "/Users/midas/Desktop/cowork/vault"
DASHBOARD_PORT = 8422
METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")
MAX_HISTORY = 24

# ANSI
CYAN = "\033[36m"
GREEN = "\033[32m"
DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

TOOL_ICONS = {
    "memory_ingest": "💾", "memory_recall": "🔍", "memory_stats": "📊",
    "memory_insights": "🧠", "playbook_update": "📖", "browse_navigate": "🌐",
    "browse_read": "📄", "browse_click": "👆", "browse_type": "⌨️",
    "browse_js": "⚙️", "browse_search": "🔎", "browse_x_feed": "🐦",
    "browse_tabs": "📑", "vault_read": "📖", "vault_insight": "🔮",
    "scan_digest": "📡", "message_claude": "📨", "shell": "⚡",
    "self_test": "🧪", "brain_snapshot": "🧠", "self_improve": "🔄",
}


# ── Memory & Browser (singletons) ───────────────────────────────────────────

class MemoryBridge:
    """Direct Python bridge to the memory daemon."""

    def __init__(self):
        self.daemon = None
        self._started = False

    def start(self):
        try:
            from phantom_memory.daemon import MemoryDaemon
        except ImportError:
            daemon_dir = os.path.join(os.path.dirname(__file__), "..", "memory")
            sys.path.insert(0, os.path.abspath(daemon_dir))
            from daemon import MemoryDaemon

        db_path = os.path.join(os.path.dirname(__file__), "..", "memory", "chromadb_live")
        self.daemon = MemoryDaemon(
            vault_path=VAULT_PATH, db_path=db_path,
            enable_enricher=True, enricher_interval=300,
        )
        self.daemon.start()
        self._started = True

    def ingest(self, role, text):
        if not self._started:
            return {"error": "daemon not started"}
        self.daemon.ingest(role, text)
        time.sleep(0.3)
        s = self.daemon.stats
        return {"status": "stored", "extracted": s["extracted"],
                "stored": s["stored"], "total_memories": s["total_memories"]}

    def recall(self, query, n_results=5, type_filter=""):
        if not self._started:
            return {"error": "daemon not started"}
        fv = type_filter if type_filter in ("decision", "task", "preference", "quantitative", "general") else None
        memories = self.daemon.store.recall(query, n_results=n_results, type_filter=fv)
        results = []
        for m in memories:
            meta = m["metadata"]
            results.append({
                "text": m["text"], "type": meta.get("type", "unknown"),
                "score": round(m["score"], 3),
                "entities": json.loads(meta.get("entities", "[]")),
                "timestamp": meta.get("timestamp", ""),
            })
        return {"query": query, "results": results, "total_memories": self.daemon.store.count()}

    def stats(self):
        if not self._started:
            return {"error": "daemon not started"}
        s = self.daemon.stats
        return {"session": self.daemon.session_id, "ingested": s["ingested"],
                "extracted": s["extracted"], "stored": s["stored"],
                "deduped": s["deduped"], "superseded": s.get("superseded", 0),
                "total_memories": s["total_memories"]}

    def get_insights(self):
        if not self._started:
            return {"error": "daemon not started"}
        vault_path = self.daemon.vault.vault_path
        lines = []
        heartbeat_path = os.path.join(vault_path, "midas", ".enricher_heartbeat")
        if os.path.exists(heartbeat_path):
            try:
                with open(heartbeat_path) as f:
                    ts = f.read().strip()
                hb_time = datetime.fromisoformat(ts)
                age_min = (datetime.now() - hb_time).total_seconds() / 60
                lines.append(f"Enricher: {'running' if age_min < 10 else 'stale'} (heartbeat {age_min:.0f}m ago)")
            except Exception:
                lines.append("Enricher: unknown")
        rel_path = os.path.join(vault_path, "memory", "relationships.md")
        if os.path.exists(rel_path):
            with open(rel_path) as f:
                lines.append(f"Relationships: {f.read().count('## ')} entities")
        insights_dir = os.path.join(vault_path, "memory", "insights")
        if os.path.exists(insights_dir):
            files = sorted([f for f in os.listdir(insights_dir) if f.startswith("patterns-")], reverse=True)
            if files:
                with open(os.path.join(insights_dir, files[0])) as f:
                    headers = [l.replace("## ", "").strip() for l in f if l.startswith("## ")]
                if headers:
                    lines.append(f"Insights ({files[0]}): {', '.join(headers[:5])}")
        return {"summary": "\n".join(lines)}

    def stop(self):
        if self._started and self.daemon:
            self.daemon.stop()


memory = MemoryBridge()

try:
    from browser import BrowserBridge
    browser = BrowserBridge()
except ImportError:
    browser = None


# ── LLM wrapper ──────────────────────────────────────────────────────────────

_client = None

def llm_fn(messages, max_tokens=500, temperature=0.7):
    """OpenAI-compatible chat completion. Returns text string."""
    resp = _client.chat.completions.create(
        model=MLX_MODEL, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def llm_classify(prompt, max_tokens=8, temperature=0.0):
    """Single-shot classification call for Layer 2."""
    resp = _client.chat.completions.create(
        model=MLX_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content or ""


# ── TUI helpers ──────────────────────────────────────────────────────────────

def print_banner(mem_count=0, browser_ok=False, ane_ok=False):
    bar_width = 32
    for i in range(1, 11):
        filled = int(bar_width * i / 10)
        bar = "█" * filled + "▓" * min(4, bar_width - filled) + "░" * max(0, bar_width - filled - 4)
        print(f"\r{CYAN}  {bar[:bar_width]}{RESET}", end="", flush=True)
        time.sleep(0.04)
    print(f"\r{CYAN}  ╔══════════════════════════════╗{RESET}")
    print(f"{CYAN}  ║      {BOLD}◆  M I D A S  ◆{RESET}{CYAN}         ║{RESET}")
    print(f"{CYAN}  ╚══════════════════════════════╝{RESET}")
    b_dot = f"{GREEN}●{RESET}" if browser_ok else f"{RED}○{RESET}"
    a_dot = f"{GREEN}●{RESET}" if ane_ok else f"{RED}○{RESET}"
    print(f"  {DIM}{mem_count} memories │ browser {RESET}{b_dot}{DIM} │ ane {RESET}{a_dot}{DIM} │ v2 router{RESET}")
    print()


def print_tool_call(name, args):
    icon = TOOL_ICONS.get(name, "🔧")
    label = ""
    if name == "memory_ingest":
        label = f"storing: {args.get('text', '')[:60]}..."
    elif name == "memory_recall":
        label = f"searching: {args.get('query', '')}"
    elif name == "memory_stats":
        label = "checking stats"
    elif name == "memory_insights":
        label = "reading enricher insights"
    elif name == "browse_search":
        label = f"searching: {args.get('query', '')}"
    elif name == "browse_x_feed":
        label = f"scanning X feed (top {args.get('count', 5)})..."
    elif name == "vault_read":
        p, q = args.get("path", ""), args.get("query", "")
        label = f"vault search: {q}" if q else (f"vault: {p}" if p else "listing vault")
    elif name == "vault_insight":
        label = f"cross-referencing: {args.get('topic', '')[:50]}"
    elif name == "scan_digest":
        label = f"scans: {args.get('mode', 'latest')}"
    elif name == "shell":
        label = f"$ {args.get('command', '')[:60]}"
    elif name == "playbook_update":
        label = f"playbook: {args.get('action', 'read')} {args.get('section', '')}"
    elif name == "message_claude":
        label = f"→ Claude: {args.get('message', '')[:50]}"
    elif name == "self_test":
        label = f"running {args.get('mode', 'light')} test..."
    elif name == "brain_snapshot":
        label = f"snapshot: {args.get('scope', 'session')}"
    elif name == "self_improve":
        label = "analyzing for improvements..."
    else:
        label = str(args)[:60]
    print(f"  {DIM}{icon} {label}{RESET}")


def _clean_response(text):
    """Strip think tags, special tokens, repetition."""
    if not text:
        return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    for special in ['<|endoftext|>', '<|im_end|>', '<|im_start|>']:
        text = text.replace(special, '')
    text = re.sub(r'\n(user|assistant|system)\s*$', '', text)
    return text.strip()


# ── History management ───────────────────────────────────────────────────────

def _trim_history(history, max_count):
    if len(history) <= max_count:
        return history
    # Keep last max_count entries, summarize dropped
    cut = len(history) - max_count
    dropped = history[:cut]
    parts = []
    for msg in dropped:
        c = msg.get("content", "")
        if not c:
            continue
        if msg["role"] == "user":
            parts.append(f"User: {c[:150]}")
        elif msg["role"] == "assistant":
            parts.append(f"Midas: {c[:150]}")
    summary = "Earlier in this conversation:\n" + "\n".join(parts[-6:]) if parts else ""
    kept = history[cut:]
    if summary:
        kept.insert(0, {"role": "system", "content": summary})
    return kept


# ── Briefing ─────────────────────────────────────────────────────────────────

def generate_briefing(stats, playbook_content=""):
    lines = []
    total = stats.get("total_memories", 0)
    marker_path = os.path.join(VAULT_PATH, "midas", ".last_session_memories")
    last_count = 0
    try:
        with open(marker_path) as f:
            last_count = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        pass
    delta = total - last_count
    if delta > 0:
        lines.append(f"- {delta} new memories since last session ({total} total)")
    else:
        lines.append(f"- {total} memories in store")
    try:
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        with open(marker_path, "w") as f:
            f.write(str(total))
    except Exception:
        pass

    # Enricher
    today = datetime.now().strftime("%Y-%m-%d")
    insights_dir = os.path.join(VAULT_PATH, "memory", "insights")
    if os.path.isdir(insights_dir):
        today_file = os.path.join(insights_dir, f"patterns-{today}.md")
        if os.path.exists(today_file):
            try:
                with open(today_file) as f:
                    sections = re.findall(r'^## .+', f.read(), re.MULTILINE)
                if sections:
                    lines.append(f"- Enricher: {len(sections)} insight(s) today")
            except Exception:
                pass

    # Services
    for label, pid_path in [("Enricher", "/tmp/phantom-enricher.pid"),
                            ("Scanner", "/tmp/phantom-scanner.pid")]:
        try:
            with open(pid_path) as f:
                os.kill(int(f.read().strip()), 0)
            lines.append(f"- {label} service: running")
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            lines.append(f"- {label} service: not running")

    # Playbook queue
    if playbook_content:
        queue = re.findall(r'- \[ \] (.+)', playbook_content)
        if queue:
            lines.append(f"- Improvement queue: {len(queue)} items")

    return "\n".join(lines) if lines else ""


# ── Main loop ────────────────────────────────────────────────────────────────

def run_agent():
    global _client

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client",
                   "phantom.enricher", "ane_server",
                   "sentence_transformers", "chromadb", "huggingface_hub"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False

    _client = OpenAI(base_url=MLX_BASE_URL, api_key="not-needed", timeout=120.0)

    # Boot silently
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        memory.start()
        stats = memory.stats()
    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        print(f"\n  {RED}Boot failed:{RESET} {e}")
        traceback.print_exc()
        return

    sys.stdout, sys.stderr = old_stdout, old_stderr

    # Wire singletons into tool_executor
    set_memory(memory)
    if browser:
        set_browser(browser)

    # Check ANE
    ane_online = False
    try:
        ane_online = urllib.request.urlopen("http://localhost:8423/health", timeout=2).status == 200
    except (urllib.error.URLError, OSError):
        pass

    # Test LLM
    try:
        _client.chat.completions.create(
            model=MLX_MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=5)
    except Exception:
        print(f"\n  {RED}LLM offline — is MLX server running?{RESET}")
        return

    # Browser
    browser_online = False
    if browser:
        browser_online = browser.is_available()
        if not browser_online:
            try:
                chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                if os.path.exists(chrome):
                    subprocess.Popen(
                        [chrome, "--remote-debugging-port=9222",
                         "--user-data-dir=" + os.path.expanduser("~/.chrome-debug"),
                         "--no-first-run", "--window-size=800,600", "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True)
                    for _ in range(5):
                        time.sleep(1)
                        if browser.is_available():
                            browser_online = True
                            break
            except Exception:
                pass
        if browser_online:
            try:
                browser.connect()
            except Exception:
                pass

    # Dashboard
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "memory", "dashboard.py")
    dashboard_path = os.path.abspath(dashboard_path)
    try:
        urllib.request.urlopen(f"http://localhost:{DASHBOARD_PORT}/api/stats", timeout=1)
    except (urllib.error.URLError, OSError):
        try:
            subprocess.Popen([sys.executable, dashboard_path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    # Playbook
    playbook_content = ""
    try:
        with open(os.path.join(VAULT_PATH, "midas/playbook.md")) as f:
            playbook_content = f.read()
    except FileNotFoundError:
        pass

    # Banner + briefing
    print_banner(mem_count=stats['total_memories'], browser_ok=browser_online, ane_ok=ane_online)
    briefing = generate_briefing(stats, playbook_content)
    if briefing:
        print(f"  {CYAN}┌─ Session Briefing ─────────────────────────{RESET}")
        for line in briefing.split("\n"):
            print(f"  {CYAN}│{RESET} {DIM}{line}{RESET}")
        print(f"  {CYAN}└─────────────────────────────────────────────{RESET}")
        print()

    # Conversation state
    history = []  # list of {role, content} — no system prompt stored here
    _last_tool = None  # for feedback detection
    _last_user_msg = None

    while True:
        try:
            user_input = input(f"{GREEN}▸ {RESET}")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/stats":
            s = memory.stats()
            rs = get_routing_stats()
            print(f"  {CYAN}Memory: {s['total_memories']} facts | Session: {s['extracted']} extracted, {s['stored']} stored{RESET}")
            if rs.get("total_decisions", 0) > 0:
                print(f"  {CYAN}Routing: {rs['total_decisions']} decisions, {rs.get('accuracy_pct', '?')}% accuracy, {rs.get('total_corrections', 0)} corrections{RESET}")
            continue
        if cmd == "/clear":
            history = []
            print(f"  {DIM}Conversation cleared (memory preserved){RESET}")
            continue
        if cmd == "/help":
            print(f"  {CYAN}Commands:{RESET}\n  {DIM}/stats /clear /quit /help{RESET}")
            continue

        # ── Feedback detection (before routing) ──
        feedback = detect_feedback(user_input, _last_tool, _last_user_msg)
        if feedback:
            log_correction(feedback)
            if feedback["type"] == "correction":
                print(f"  {DIM}📝 Correction logged: {_last_tool} was wrong{RESET}")

        # Auto-ingest user message
        memory.ingest("user", user_input)

        # ── LAYER 1+2: Route ──
        l1_result = layer1_route(user_input)
        l2_category = None
        tool_name, tool_args = route(user_input, llm_fn=llm_classify)
        if not l1_result:
            l2_category = tool_name  # Layer 2 was used

        # Log every routing decision
        log_decision(user_input, l1_result, l2_category, tool_name, tool_args)

        if tool_name == "conversation":
            # ── LAYER 4: Direct LLM response (no tool) ──
            response = synthesize(llm_fn, history, user_input, temperature=0.7)
            response = _clean_response(response)
            if response:
                print(f"\n{CYAN}{response}{RESET}\n")
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
        else:
            # ── LAYER 3: Execute tool ──
            print_tool_call(tool_name, tool_args)
            result = execute(tool_name, tool_args)
            print(f"  {DIM}  └─ done{RESET}")

            # ── LAYER 4: Synthesize tool result ──
            response = synthesize(
                llm_fn, history, user_input,
                tool_name=tool_name, tool_args=tool_args,
                tool_result=result, temperature=0.3,
            )
            response = _clean_response(response)
            if response:
                print(f"\n{CYAN}{response}{RESET}\n")
            else:
                # Fallback: just show the raw result
                print(f"\n{CYAN}{result[:2000]}{RESET}\n")

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response or result[:500]})

            # Store assistant response
            if response:
                memory.ingest("assistant", response)

        # Track for feedback detection on next turn
        _last_tool = tool_name
        _last_user_msg = user_input

        # Trim history
        history = _trim_history(history, MAX_HISTORY)

    # Shutdown
    threading.excepthook = lambda args: None
    print(f"  {DIM}Shutting down...{RESET}")
    memory.stop()
    if browser:
        browser.disconnect()
    print(f"  {GREEN}Session saved. Goodbye.{RESET}")
    sys.stdout.flush()
    try:
        os.close(2)
    except OSError:
        pass
    os._exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: None)
    run_agent()
