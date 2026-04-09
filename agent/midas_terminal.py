#!/usr/bin/env python3
"""Midas terminal — unified production-grade chat REPL.

Single terminal frontend for the Midas system. Hits the same backend as
the web UI and Telegram bot (/api/chat/stream on midas_ui :8450), so:

  - Memory recall goes through MemoryBridge.recall (multi-path 5-signal
    fusion + canonical boost) — same as production
  - Daemon events fire on every chat (memory_recalled,
    memory_recalled_item × N) — the visualization lights up live
  - Streaming response unrolls token-by-token instead of one big block
  - Session history is shared with the web UI (chatting via terminal
    appears in the web UI session log and vice versa)

Replaces both `midas` (legacy agent_v2.py) and `midas-chat` (the bash
one-shot wrapper). Same behavior whether you launch from the phone or
the Mac terminal.

Usage:
  midas                                    interactive REPL
  midas "your question"                    one-shot, prints reply, exits
  midas --json "msg"                       one-shot, raw JSON dump
  midas --no-stream "msg"                  fall back to /api/chat (non-streaming)

Slash commands inside the REPL:
  /memories       toggle showing recalled memory list after each reply
  /no-memories    hide memory list
  /clear          clear screen
  /reset          reset session history (server-side via /api/chat/reset)
  /stats          toggle showing tps/elapsed stats after each reply
  /help           show commands
  /exit /quit :q  exit
"""
import argparse
import json
import os
import readline
import signal
import sys
import time
import urllib.error
import urllib.request

URL_BASE = "http://127.0.0.1:8450"

# Chat-template stop tokens that the streaming endpoint doesn't strip
# server-side. The non-streaming endpoint runs _clean_response which
# handles these; the streaming path passes raw chunks. Strip client-side.
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "</s>")
def _strip_stop_tokens(s):
    for t in STOP_TOKENS:
        s = s.replace(t, "")
    return s

URL_STREAM = URL_BASE + "/api/chat/stream"
URL_NONSTREAM = URL_BASE + "/api/chat"
URL_CONTEXT = URL_BASE + "/api/session/context"

HISTORY_FILE = os.path.expanduser("~/.midas_terminal_history")

# ANSI colors. Set MIDAS_NO_COLOR=1 to disable.
COLOR = os.environ.get("MIDAS_NO_COLOR", "") == ""
def _c(code): return code if COLOR else ""
RESET   = _c("\033[0m")
DIM     = _c("\033[2m")
BOLD    = _c("\033[1m")
CYAN    = _c("\033[36m")
GREEN   = _c("\033[32m")
YELLOW  = _c("\033[33m")
MAGENTA = _c("\033[35m")
GRAY    = _c("\033[90m")
RED     = _c("\033[31m")


def check_server():
    """Liveness check on midas_ui."""
    try:
        urllib.request.urlopen(URL_CONTEXT, timeout=3).read()
        return True
    except Exception:
        return False


def stream_chat(message, show_stats=True, show_memories=False):
    """Stream a chat reply via /api/chat/stream. Yields nothing — prints
    directly. Returns the (response_text, stats_dict, n_memories) tuple."""
    body = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(
        URL_STREAM,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    response_text_parts = []
    stats = {}
    n_memories = 0
    t_first_token = None
    t_start = time.perf_counter()

    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as e:
        print(f"{RED}HTTP {e.code}: {e.reason}{RESET}")
        return "", {}, 0
    except Exception as e:
        print(f"{RED}error reaching midas_ui: {e}{RESET}")
        return "", {}, 0

    print(f"{CYAN}midas{RESET}{DIM} ›{RESET} ", end="", flush=True)

    buf = b""
    for chunk in iter(lambda: resp.read(64), b""):
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                try:
                    data = json.loads(line[6:].decode("utf-8"))
                except Exception:
                    continue
                t = data.get("type")
                if t == "token":
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    content = _strip_stop_tokens(data.get("content", ""))
                    if not content:
                        continue
                    response_text_parts.append(content)
                    print(content, end="", flush=True)
                elif t == "done":
                    stats = data.get("stats", {})
                    n_memories = data.get("memories_recalled", 0)

    print()  # final newline
    elapsed = time.perf_counter() - t_start
    ttft = (t_first_token - t_start) if t_first_token else 0

    if show_memories and n_memories > 0:
        print(f"{DIM}── {n_memories} memories injected ──{RESET}")

    if show_stats:
        tps = stats.get("tps", 0)
        accept = stats.get("accept_rate", 0)
        bits = []
        if ttft > 0:
            bits.append(f"ttft {ttft*1000:.0f}ms")
        bits.append(f"total {elapsed:.1f}s")
        if tps:
            bits.append(f"{tps:.1f} tok/s")
        if accept:
            bits.append(f"{accept:.0f}% accept")
        print(f"{GRAY}  {' · '.join(bits)}{RESET}")

    return "".join(response_text_parts), stats, n_memories


def nonstream_chat(message, show_memories=False, show_stats=True):
    """Fallback path via /api/chat (non-streaming). Slower visual but
    same backend, same daemon events, same recall."""
    body = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(
        URL_NONSTREAM,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        d = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"{RED}error: {e}{RESET}")
        return
    elapsed = time.perf_counter() - t0
    reply = _strip_stop_tokens(d.get("response") or d.get("reply") or d.get("text") or "")
    print(f"{CYAN}midas{RESET}{DIM} ›{RESET} {reply}")
    if show_memories:
        mems = d.get("memories_recalled") or d.get("memory_recalled") or []
        if mems:
            print(f"{DIM}── {len(mems)} memories ──{RESET}")
            for i, m in enumerate(mems[:5], 1):
                txt = (m.get("text") or "")[:100].replace("\n", " ")
                print(f"  {GRAY}{i}. [{m.get('score','?')}] {txt}{RESET}")
    if show_stats:
        stats = d.get("stats", {})
        tps = stats.get("tps", 0)
        bits = [f"total {elapsed:.1f}s"]
        if tps:
            bits.append(f"{tps:.1f} tok/s")
        print(f"{GRAY}  {' · '.join(bits)}{RESET}")


def banner():
    print(f"{BOLD}{CYAN}midas{RESET} {DIM}— production terminal · {URL_STREAM}{RESET}")
    print(f"{DIM}/help for commands · /exit to quit · ↑ for history{RESET}")
    print()


def repl(streaming=True):
    if not check_server():
        print(f"{RED}✗ midas_ui not reachable on :8450{RESET}")
        print(f"{DIM}  start it: cd ~/Desktop/cowork/orion-ane/agent && \\")
        print(f"            ~/.mlx-env/bin/python3 midas_ui.py &{RESET}")
        return 1

    # Set up readline history
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(2000)

    banner()
    show_memories = False
    show_stats = True

    # Ctrl-C inside generation cancels the current request,
    # not the whole process.
    def sigint_handler(sig, frame):
        # readline catches it normally; this is a no-op handler so
        # interrupted generation doesn't kill us.
        print(f"\n{YELLOW}^C — interrupted{RESET}")
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        while True:
            try:
                line = input(f"{BOLD}{GREEN}you{RESET}{DIM} ›{RESET} ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue

            # Slash commands
            if line.startswith("/") or line in ("exit", "quit", ":q"):
                cmd = line.lstrip("/").split()[0] if line.startswith("/") else line
                if cmd in ("exit", "quit", "q"):
                    break
                elif cmd == "memories":
                    show_memories = True
                    print(f"{DIM}memory display: ON{RESET}")
                elif cmd in ("no-memories", "nomemories"):
                    show_memories = False
                    print(f"{DIM}memory display: OFF{RESET}")
                elif cmd == "stats":
                    show_stats = not show_stats
                    print(f"{DIM}stats: {'ON' if show_stats else 'OFF'}{RESET}")
                elif cmd == "clear":
                    os.system("clear")
                elif cmd == "reset":
                    try:
                        urllib.request.urlopen(
                            urllib.request.Request(URL_BASE + "/api/chat/reset", method="POST"),
                            timeout=5,
                        )
                        print(f"{DIM}session history reset{RESET}")
                    except Exception as e:
                        print(f"{RED}reset failed: {e}{RESET}")
                elif cmd == "help":
                    print(f"{DIM}  /memories /no-memories /stats /clear /reset /exit{RESET}")
                else:
                    print(f"{DIM}unknown command: {line}{RESET}")
                continue

            try:
                if streaming:
                    stream_chat(line, show_stats=show_stats, show_memories=show_memories)
                else:
                    nonstream_chat(line, show_memories=show_memories, show_stats=show_stats)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}cancelled{RESET}")
            print()
    finally:
        try:
            readline.write_history_file(HISTORY_FILE)
        except Exception:
            pass

    return 0


def main():
    ap = argparse.ArgumentParser(
        description="midas terminal — production-grade chat REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("message", nargs="*", help="one-shot message (omit for REPL)")
    ap.add_argument("--json", action="store_true", help="raw JSON output (one-shot only)")
    ap.add_argument("--no-stream", action="store_true", help="use non-streaming /api/chat")
    ap.add_argument("--memories", action="store_true", help="show recalled memories")
    args = ap.parse_args()

    msg = " ".join(args.message).strip()

    if not msg:
        return repl(streaming=not args.no_stream)

    if not check_server():
        print(f"{RED}✗ midas_ui not reachable on :8450{RESET}", file=sys.stderr)
        return 1

    if args.json:
        body = json.dumps({"message": msg}).encode()
        req = urllib.request.Request(URL_NONSTREAM, data=body,
                                     headers={"Content-Type":"application/json"}, method="POST")
        d = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
        print(json.dumps(d, indent=2))
        return 0

    if args.no_stream:
        nonstream_chat(msg, show_memories=args.memories)
    else:
        stream_chat(msg, show_memories=args.memories)
    return 0


if __name__ == "__main__":
    sys.exit(main())
