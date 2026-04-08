#!/usr/bin/env python3
"""
Midas Agent v2 — Hybrid Router Architecture
==============================================

Architecture:
  Layer 1: Keyword routing (router.py) — instant, zero LLM calls
  Layer 2: LLM self-routing (router.py) — 70B picks tool + constructs args
  Layer 3: Tool execution (tool_executor.py) — dispatches tool calls
  Layer 4: LLM text generation (synthesizer.py) — conversation and tool result synthesis

Layer 1 handles 80%+ of messages deterministically. Layer 2 gives the 70B
the full tool list and lets it decide which tool to call and construct the args.
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

# OpenAI client no longer needed — all LLM calls use raw urllib

# ── Local modules ────────────────────────────────────────────────────────────

from router import route, layer1_route, layer2_llm_route
from tool_executor import execute, set_memory, set_browser
from synthesizer import synthesize, SYSTEM_PROMPT
from feedback_loop import log_decision, detect_feedback, log_correction, get_routing_stats
from memory_bridge import MemoryBridge

# ── Config ───────────────────────────────────────────────────────────────────

MLX_BASE_URL = os.environ.get("MLX_BASE_URL", "http://127.0.0.1:8899/v1")
MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Llama-3.3-70B-Instruct-3bit")
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
    "heartbeat": "💓",
    "scgp_convert": "📄", "scgp_registry": "🏛️", "scgp_pipeline": "📋",
}


# ── Memory & Browser (singletons) ───────────────────────────────────────────


# MemoryBridge imported from memory_bridge.py (shared with telegram_bot)


memory = MemoryBridge()

try:
    from browser import BrowserBridge
    browser = BrowserBridge()
except ImportError:
    browser = None


# ── LLM wrapper ──────────────────────────────────────────────────────────────

_last_stats = {}  # tok/s, accept_rate, etc from last call


def _llm_call(messages, max_tokens, temperature):
    """Raw HTTP call to MLX server with retry on connection errors."""
    payload = json.dumps({
        "model": MLX_MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "repetition_penalty": 1.35,
    }).encode()
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                MLX_BASE_URL.rstrip("/") + "/chat/completions",
                data=payload, headers={"Content-Type": "application/json",
                                       "Connection": "close"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except Exception as e:
            last_err = e
            err_name = type(e).__name__
            retryable = any(k in err_name for k in
                          ['URL', 'Connection', 'Remote', 'Broken', 'Reset', 'EOF', 'Timeout'])
            if not retryable:
                raise
            time.sleep(1 + attempt)
    raise last_err


def llm_fn(messages, max_tokens=300, temperature=0.7, **_kw):
    """OpenAI-compatible chat completion. Q4 70B + CPU 1B spec decode server."""
    global _last_stats
    data = _llm_call(messages, max_tokens, temperature)
    _last_stats = data.get("x_spec_decode", {})

    # Signal bus: write 70B generation stats
    try:
        from signal_bus import update_batch
        spec = data.get("x_spec_decode", {})
        usage = data.get("usage", {})
        update_batch({
            "70b_tok_s": spec.get("tps", 0),
            "70b_tokens_generated": usage.get("completion_tokens", 0),
            "70b_finish_reason": data["choices"][0].get("finish_reason", ""),
            "generation_active": False,
        })
    except Exception:
        pass

    return data["choices"][0]["message"]["content"] or ""


def llm_route_fn(messages, max_tokens=120, temperature=0.0, **_kw):
    """LLM call for Layer 2 self-routing. Temperature=0.0 for deterministic routing."""
    data = _llm_call(messages, max_tokens, temperature)
    return data["choices"][0]["message"]["content"] or ""


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
    print(f"  {DIM}{mem_count} memories │ browser {RESET}{b_dot}{DIM} │ ane {RESET}{a_dot}{DIM} │ 70B │ v2 router{RESET}")
    print()


def print_stats():
    """Print tok/s and spec decode stats from last LLM call."""
    s = _last_stats
    if not s:
        return
    tps = s.get("tps", 0)
    tokens = s.get("tokens", 0)
    elapsed = s.get("elapsed", 0)
    acc = s.get("accept_rate", 0)
    ng_d = s.get("ngram_drafted", 0)
    ng_a = s.get("ngram_accepted", 0)
    cpu_d = s.get("cpu_drafted", 0)
    cpu_a = s.get("cpu_accepted", 0)

    parts = [f"{tps} tok/s"]
    if ng_d > 0:
        parts.append(f"ngram {ng_a}/{ng_d}")
    if cpu_d > 0:
        parts.append(f"cpu {cpu_a}/{cpu_d}")
    if acc > 0:
        parts.append(f"{acc}% acc")
    parts.append(f"{tokens} tok")
    if elapsed > 0:
        parts.append(f"{elapsed}s")

    print(f"  {DIM}{'  '.join(parts)}{RESET}")


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
    elif name == "heartbeat":
        label = "launching heartbeat dashboard..."
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
    for special in ['<|endoftext|>', '<|im_end|>', '<|im_start|>',
                     '<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>',
                     '<|begin_of_text|>', 'assistant']:
        text = text.replace(special, '')
    text = re.sub(r'\n(user|assistant|system)\s*$', '', text)

    # Paragraph-level repetition: if a line/paragraph appears 2+ times, keep only first
    lines = text.split('\n')
    seen = set()
    deduped = []
    for line in lines:
        key = line.strip().lower()
        if len(key) > 30 and key in seen:
            continue  # skip repeated paragraph
        if len(key) > 30:
            seen.add(key)
        deduped.append(line)
    text = '\n'.join(deduped)

    # Phrase-level repetition: if any 5+ word phrase repeats 2+ times, cut
    words = text.split()
    for phrase_len in range(12, 4, -1):
        for i in range(len(words) - phrase_len):
            phrase = ' '.join(words[i:i + phrase_len])
            rest = ' '.join(words[i + phrase_len:])
            if rest.count(phrase) >= 1:
                first_end = text.find(phrase) + len(phrase)
                second_start = text.find(phrase, first_end)
                if second_start > 0:
                    text = text[:second_start].rstrip(' \n,-')
                    break
        else:
            continue
        break

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

def _sigint_handler(signum, frame):
    """Force KeyboardInterrupt even when httpx/openai swallows SIGINT."""
    raise KeyboardInterrupt


def run_agent():
    # Ensure Ctrl+C always raises KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_handler)

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client",
                   "phantom.enricher", "ane_server",
                   "sentence_transformers", "chromadb", "huggingface_hub"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False

    # Boot silently
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
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

    # Start system monitor (Phase 5A)
    try:
        from system_monitor import get_monitor
        _sys_monitor = get_monitor()
        _sys_monitor.start()
    except Exception:
        _sys_monitor = None

    # Check ANE
    ane_online = False
    try:
        ane_online = urllib.request.urlopen("http://localhost:8423/health", timeout=2).status == 200
    except (urllib.error.URLError, OSError):
        pass

    # Test LLM
    try:
        llm_fn([{"role": "user", "content": "hi"}], max_tokens=5, temperature=0.0)
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
            print(f"\n{DIM}  bye{RESET}")
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

        # ── Ctrl+C cancels the current turn, returns to prompt ──
        try:
            # ── Feedback detection (before routing) ──
            feedback = detect_feedback(user_input, _last_tool, _last_user_msg)
            if feedback:
                log_correction(feedback)
                if feedback["type"] == "correction":
                    print(f"  {DIM}📝 Correction logged: {_last_tool} was wrong{RESET}")

            # Signal bus: update turn + idle
            try:
                from signal_bus import update_batch
                update_batch({
                    "conversation_turn": len(history) // 2 + 1,
                    "user_idle_seconds": 0,
                    "generation_active": True,
                })
                # Touch system monitor to reset idle timer
                if _sys_monitor:
                    _sys_monitor.touch()
            except Exception:
                pass

            # Auto-ingest user message
            memory.ingest("user", user_input)

            # ── LAYER 1+2: Route ──
            l1_result = layer1_route(user_input)
            l2_result = None
            tool_name, tool_args = route(user_input, llm_fn=llm_route_fn)
            if not l1_result:
                l2_result = tool_name  # Layer 2 was used

            # Log every routing decision
            log_decision(user_input, l1_result, l2_result, tool_name, tool_args)

            # Show routing decision
            print(f"  {DIM}⚡ {tool_name}{RESET}")

            # ── Subconscious: adaptive recall + briefing assembly ──
            mem_ctx = None
            _recall_top10 = []
            _briefing = None
            try:
                # Phase 4B: Adaptive retrieval from signal bus
                try:
                    from signal_bus import read as sig_read, update as sig_update
                    from ane_classifier import classify as classify_domain
                    turn = sig_read("conversation_turn", 1)
                    last_rel = sig_read("last_retrieval_relevance", 0.5)
                    prev_domain = sig_read("domain_detected", "unknown")

                    # Detect current domain
                    cur_domain, _ = classify_domain(user_input)
                    topic_switch = (prev_domain != "unknown"
                                    and cur_domain != prev_domain
                                    and turn > 1)
                    sig_update("domain_detected", cur_domain)

                    # Adaptive k and threshold
                    if turn <= 1 or topic_switch:
                        retrieval_k = 20  # First turn or topic switch: go broad
                        threshold = 0.25
                    elif last_rel < 0.3:
                        retrieval_k = 20
                        threshold = 0.25
                    elif last_rel > 0.6:
                        retrieval_k = 8
                        threshold = 0.40
                    else:
                        retrieval_k = 12
                        threshold = 0.35

                    sig_update("retrieval_breadth", retrieval_k)
                    sig_update("retrieval_threshold", threshold)

                    if topic_switch:
                        print(f"  {DIM}  📡 topic switch: {prev_domain} → {cur_domain}{RESET}")
                except Exception:
                    retrieval_k = 15
                    threshold = 0.35

                import time as _t
                _recall_t0 = _t.perf_counter()
                recall_result = memory.recall(user_input, n_results=30)
                _recall_ms = (_t.perf_counter() - _recall_t0) * 1000

                if recall_result and recall_result.get("results"):
                    _recall_top10 = recall_result["results"]
                    filtered = []
                    max_relevance = 0
                    for r in _recall_top10:
                        score = r.get("score", 0)
                        if score < threshold:
                            continue
                        text = r.get("text", "")
                        if text.startswith("[") and ".md]" in text[:50]:
                            continue
                        if text.strip().endswith("?"):
                            continue
                        filtered.append(r)
                        max_relevance = max(max_relevance, score)
                        if len(filtered) >= retrieval_k:
                            break
                    mem_ctx = [r["text"] for r in filtered]

                    # Signal bus: write retrieval stats
                    try:
                        from signal_bus import update_batch
                        update_batch({
                            "last_retrieval_relevance": max_relevance,
                            "retrieval_ms": _recall_ms,
                            "memory_count": recall_result.get("total_memories", 0),
                        })
                    except Exception:
                        pass

                    if filtered:
                        print(f"  {DIM}  🧠 {len(filtered)} memories (k={retrieval_k}, "
                              f"t={threshold:.2f}, {_recall_ms:.1f}ms){RESET}")
                        # Assemble briefing from memories
                        try:
                            from briefing_assembler import assemble_briefing
                            _briefing = assemble_briefing(filtered)
                        except Exception:
                            pass
            except Exception:
                pass

            if tool_name == "conversation":
                # ── LAYER 4: Direct LLM response (no tool) ──
                response = synthesize(llm_fn, history, user_input,
                                      temperature=0.3, memory_context=mem_ctx,
                                      briefing=_briefing)
                response = _clean_response(response)
                if response:
                    print(f"\n{CYAN}{response}{RESET}")
                    print_stats()
                    print()
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "assistant", "content": response})
            else:
                # ── LAYER 3: Execute tool ──
                print_tool_call(tool_name, tool_args)
                result = execute(tool_name, tool_args)
                print(f"  {DIM}  └─ done{RESET}")

                # Short-circuit tools that return direct messages (skip 9B synthesis)
                if tool_name == "heartbeat" and isinstance(result, dict):
                    status = result.get("status", "")
                    url = result.get("url", "http://localhost:8423")
                    msg = f"Heartbeat {'already running' if 'already' in status else 'launched'} at {url}"
                    print(f"\n{CYAN}{msg}{RESET}\n")
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "assistant", "content": msg})
                    _last_tool = tool_name
                    _last_user_msg = user_input
                    continue

                # ── LAYER 4: Synthesize tool result ──
                response = synthesize(
                    llm_fn, history, user_input,
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=result, temperature=0.3,
                    max_tokens=800, memory_context=mem_ctx,
                    briefing=_briefing,
                )
                response = _clean_response(response)
                if response:
                    print(f"\n{CYAN}{response}{RESET}")
                    print_stats()
                    print()
                else:
                    # Fallback: just show the raw result
                    print(f"\n{CYAN}{str(result)[:2000]}{RESET}\n")

                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response or result[:500]})

                # Store assistant response
                if response:
                    memory.ingest("assistant", response)

            # ── Retrieval quality logging + memory usage tracking ──
            if response and _recall_top10:
                try:
                    from retrieval_logger import log_retrieval
                    log_retrieval(user_input, _recall_top10, response, tool_name)
                except Exception:
                    pass
                # Phase 7A: Track which memories the 70B actually used
                try:
                    from memory_usage_tracker import track_memory_usage
                    from signal_bus import update as sig_update
                    usage = track_memory_usage(response, mem_ctx, sig_update)
                    if usage["used"] > 0:
                        print(f"  {DIM}  📊 {usage['used']} memories used, "
                              f"{usage['ignored']} ignored{RESET}")
                except Exception:
                    pass

            # Track for feedback detection on next turn
            _last_tool = tool_name
            _last_user_msg = user_input

            # Trim history
            history = _trim_history(history, MAX_HISTORY)

        except KeyboardInterrupt:
            print(f"\n  {YELLOW}⚡ interrupted{RESET}\n")
            continue

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
