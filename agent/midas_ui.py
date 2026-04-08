#!/usr/bin/env python3
"""
Midas Agent — Web UI
Single-file Flask app with embedded HTML/CSS/JS.
Operations console aesthetic. Dark, monospace, terminal-native.
"""

import io
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import urllib.request
import urllib.error
import warnings
from datetime import datetime
from flask import Flask, request, jsonify, Response

# ── Suppress noisy loggers ──────────────────────────────────────────────────

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
for noisy in ("httpx", "httpcore", "openai", "sentence_transformers",
              "chromadb", "huggingface_hub", "werkzeug"):
    lg = logging.getLogger(noisy)
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False

# ── Boot memory silently ────────────────────────────────────────────────────

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(__file__))

from memory_bridge import MemoryBridge
from router import route, layer1_route
from tool_executor import execute, set_memory, set_browser
from synthesizer import synthesize
from idle_queue import IdleQueue

# ── Config ──────────────────────────────────────────────────────────────────

MLX_BASE_URL = os.environ.get("MLX_BASE_URL", "http://127.0.0.1:8899/v1")
MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-72B-Instruct-4bit")
MAX_HISTORY = 24
PORT = 8450

# ── Singletons ──────────────────────────────────────────────────────────────

memory = MemoryBridge()
_last_stats = {}
_session = {
    "messages_sent": 0,
    "memories_recalled": 0,
    "facts_extracted": 0,
    "tools_used": 0,
    "start_time": datetime.now().isoformat(),
}
_feed = []  # last 20 memory events
_last_subconscious = []  # memories injected on last turn
_history = []  # conversation history

# Main 25 Build 0: stable per-session briefing for prompt cache hit rate.
# Built once at session start (or every N=5 turns), reused across turns so
# the system message stays byte-stable and the verifier's prefix KV cache
# hits on every subsequent turn. Per-query memories ride in the user message
# (tail) so they don't invalidate the prefix.
_session_briefing = None
_session_briefing_built_at_turn = -1
_SESSION_BRIEFING_REFRESH_EVERY = 5
_lock = threading.Lock()
_idle_queue = None  # Initialized after memory.start()

# ── LLM ─────────────────────────────────────────────────────────────────────

def _llm_call(messages, max_tokens, temperature):
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
                data=payload,
                headers={"Content-Type": "application/json", "Connection": "close"})
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


def llm_fn(messages, max_tokens=1500, temperature=0.7, **_kw):
    global _last_stats
    data = _llm_call(messages, max_tokens, temperature)
    _last_stats = data.get("x_spec_decode", {})
    return data["choices"][0]["message"]["content"] or ""


def llm_stream(messages, max_tokens=300, temperature=0.7):
    """Streaming LLM call — yields text chunks as they arrive."""
    global _last_stats
    payload = json.dumps({
        "model": MLX_MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "repetition_penalty": 1.35, "stream": True,
    }).encode()
    req = urllib.request.Request(
        MLX_BASE_URL.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)

    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            stats = chunk.get("x_spec_decode")
            if stats:
                _last_stats = stats
            if content:
                yield content
        except json.JSONDecodeError:
            continue


def llm_route_fn(messages, max_tokens=120, temperature=0.0, **_kw):
    data = _llm_call(messages, max_tokens, temperature)
    return data["choices"][0]["message"]["content"] or ""


def _clean_response(text):
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
    # Main 22 follow-up: strip training-template token leakage. The 72B
    # sometimes generates "Human: ... Assistant: ..." patterns at the tail
    # of long completions, leaking its training format. Cut everything from
    # the first such marker onward.
    text = re.split(r'\n*(?:Human|User|### ?Human|### ?User|<\|eos\|>|</s>)\s*:?\s*', text, maxsplit=1)[0]
    text = text.rstrip()
    # Paragraph-level dedup
    lines = text.split('\n')
    seen = set()
    deduped = []
    for line in lines:
        key = line.strip().lower()
        if len(key) > 30 and key in seen:
            continue
        if len(key) > 30:
            seen.add(key)
        deduped.append(line)
    text = '\n'.join(deduped)
    # Phrase-level repetition
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
    # Strip self-referential / meta-commentary sentences
    meta_patterns = [
        r"I rewrote.*", r"I rephrased.*", r"I made a mistake.*",
        r"I should have.*", r"Let me (?:give|rephrase|correct).*",
        r"I'll rephrase.*", r"\d+ priorities are set.*",
    ]
    for pat in meta_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    # Sentence-level dedup: if a sentence repeats (even with minor variations), keep first
    import difflib
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) > 2:
        kept = [sentences[0]]
        for s in sentences[1:]:
            is_dup = False
            for k in kept:
                ratio = difflib.SequenceMatcher(None, s.lower(), k.lower()).ratio()
                if ratio > 0.6:  # Tighter threshold
                    is_dup = True
                    break
            if not is_dup and len(s) > 5:
                kept.append(s)
        text = ' '.join(kept)

    # Final cleanup: remove double spaces and trailing fragments
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _regenerate_if_garbage(llm_fn, messages, response, max_retries=1):
    """Regenerate if response is garbage (too short, just a number, etc)."""
    if len(response.strip()) < 10 and not any(c.isalpha() for c in response[:5]):
        for _ in range(max_retries):
            retry = llm_fn(messages, max_tokens=300, temperature=0.5)
            if len(retry.strip()) >= 10:
                return retry
    return response


def _add_feed(event_type, text):
    _feed.insert(0, {
        "type": event_type,
        "text": text[:120],
        "time": datetime.now().strftime("%H:%M:%S"),
    })
    if len(_feed) > 20:
        _feed.pop()


def _trim_history(history, max_count):
    if len(history) <= max_count:
        return history
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


# ── Flask app ───────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    global _history, _last_subconscious, _session_briefing, _session_briefing_built_at_turn
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    with _lock:
        try:
            # Cancel any idle work — user message takes priority
            if _idle_queue:
                _idle_queue.cancel()

            _session["messages_sent"] += 1

            # 1. Ingest user message
            ingest_result = memory.ingest("user", message)
            if ingest_result.get("extracted", 0) > 0:
                _session["facts_extracted"] += ingest_result.get("extracted", 0)
                _add_feed("extraction", f"[user] {message[:80]}")

            # 2. Route
            l1_result = layer1_route(message)
            tool_name, tool_args = route(message, llm_fn=llm_route_fn)
            route_layer = "L1" if l1_result else "L2"

            # 3. Recall memories (subconscious) — skip for casual greetings
            mem_ctx = None
            presentation_briefing = None
            _last_subconscious = []
            _casual_greetings = {"hey","yo","hi","hello","sup","morning","evening",
                                 "k","ok","thanks","bye","later","night","yo!","hey!"}
            skip_memory = message.lower().rstrip("!?., ") in _casual_greetings
            try:
                if not skip_memory:
                    recall_result = memory.recall(message, n_results=15)
                else:
                    recall_result = None
                if recall_result and recall_result.get("results"):
                    filtered = []
                    for r in recall_result["results"]:
                        if r.get("score", 0) < 0.40:
                            continue
                        text = r.get("text", "")
                        if text.startswith("[") and ".md]" in text[:50]:
                            continue
                        if text.strip().endswith("?"):
                            continue
                        filtered.append(r)
                        if len(filtered) >= 12:
                            break
                    # Dedup: cluster similar memories, keep most specific.
                    # Main 24 Build 1: compare *body* not prefix. Meta memories
                    # share a fixed "Session work on YYYY-MM-DD ... — shipped,
                    # changed, decided: " prefix that made every meta entry
                    # dedup against the first one. Strip a leading prefix
                    # (everything before the first ": ") before comparing.
                    import difflib as _dl
                    def _dedup_body(t: str) -> str:
                        s = t.lower().strip()
                        if ": " in s[:140]:
                            s = s.split(": ", 1)[1]
                        return s[:160]
                    deduped = []
                    for r in filtered:
                        body = _dedup_body(r["text"])
                        is_dup = False
                        for kept in deduped:
                            ratio = _dl.SequenceMatcher(None, body,
                                _dedup_body(kept["text"])).ratio()
                            if ratio > 0.78:
                                is_dup = True
                                break
                        if not is_dup:
                            deduped.append(r)
                    filtered = deduped[:8]
                    mem_ctx = [r["text"] for r in filtered]
                    _last_subconscious = [
                        {"text": r["text"][:200], "score": r["score"]}
                        for r in filtered
                    ]
                    if mem_ctx:
                        _session["memories_recalled"] += len(mem_ctx)
                        for m in mem_ctx[:3]:
                            _add_feed("recall", m[:80])
                # Main 24 Build 0: presentation layer.
                # Build a query-context-aware briefing from the filtered
                # memories (preserving the dict shape so source_role canonical
                # markers come through). If anything fails, leave the briefing
                # empty and the synthesizer falls back to the legacy
                # `RELEVANT MEMORIES:` raw mem_ctx dump.
                presentation_briefing = None
                # Main 25 Build 0: stable session briefing for prompt cache.
                # If we have one and it's fresh, use it. Otherwise rebuild from
                # a generic seed-recall (project state, not the user's specific
                # query) so the system message is byte-stable across turns.
                turns_since_build = (_session["messages_sent"]
                                     - _session_briefing_built_at_turn)
                if (_session_briefing is None
                    or turns_since_build >= _SESSION_BRIEFING_REFRESH_EVERY):
                    try:
                        import sys as _sys
                        _sys.path.insert(0, "/Users/midas/Desktop/cowork/vault/subconscious")
                        from multi_path_retrieve import present as _present
                        SEED_QUERY = ("Current state and active work on Apple "
                                      "Silicon ML, ANE, spec decode, "
                                      "Subconscious, paper status.")
                        seed_recall = memory.recall(SEED_QUERY, n_results=12)
                        seed_filtered = []
                        for r in (seed_recall.get("results") or []):
                            if r.get("score", 0) < 0.30:
                                continue
                            t = r.get("text", "")
                            if t.startswith("[") and ".md]" in t[:50]:
                                continue
                            seed_filtered.append(r)
                            if len(seed_filtered) >= 8:
                                break
                        _session_briefing = _present(
                            seed_filtered, SEED_QUERY, max_chars=1500)
                        _session_briefing_built_at_turn = _session["messages_sent"]
                    except Exception:
                        _session_briefing = None
                # System slot uses the stable session briefing
                presentation_briefing = _session_briefing
                # Per-query memories ride in the user message (tail) so they
                # don't invalidate the verifier's prefix KV cache.
                per_query_block = None
                if not skip_memory and recall_result and recall_result.get("results"):
                    try:
                        import sys as _sys
                        _sys.path.insert(0, "/Users/midas/Desktop/cowork/vault/subconscious")
                        from multi_path_retrieve import present as _present
                        per_query_block = _present(filtered, message, max_chars=1500)
                    except Exception:
                        per_query_block = None
            except Exception:
                pass

            # 4. Execute tool or direct conversation
            tool_result = None
            if tool_name != "conversation":
                _session["tools_used"] += 1
                tool_result = execute(tool_name, tool_args)

            # 5. Synthesize response
            # Storage operations skip the LLM entirely. The tool already did the
            # work; the LLM has nothing to add and would bleed prior context
            # into the acknowledgement. Hard-code the ack with the extracted count.
            if tool_name == "memory_ingest":
                extracted = 0
                if isinstance(tool_result, dict):
                    extracted = tool_result.get("extracted", 0)
                if extracted > 0:
                    response = (f"Got it. Stored {extracted} fact"
                                f"{'s' if extracted != 1 else ''}.")
                else:
                    response = "Got it. Noted."
            elif tool_name == "conversation":
                # Main 22 follow-up: casual greetings bypass the LLM entirely.
                # With no memories injected (skip_memory=True) and a very
                # short user message, the 72B free-associates into unrelated
                # content (Chinese COVID text, Hungarian translations, etc.)
                # — even with a tight max_tokens clamp. Same structural fix
                # as the memory_ingest LLM bypass: hardcode the response.
                # The agent has nothing useful to add to a "hey" anyway.
                if skip_memory:
                    import random as _rnd
                    _greetings = [
                        "Hey Nick. What's up?",
                        "Hey. What can I help with?",
                        "Hey Nick. Need an update or have a specific question?",
                        "Hey. Ready when you are.",
                        "Hey Nick. What do you want to dig into?",
                    ]
                    response = _rnd.choice(_greetings)
                else:
                    # Main 25 Build 0: per-query memories go into the user
                    # message (tail) so the system message stays byte-stable
                    # for the verifier's prefix cache. Memory_context kwarg
                    # is dropped to avoid the synthesizer's fallback path
                    # also injecting them into system.
                    augmented = message
                    if per_query_block:
                        augmented = (f"{per_query_block}\n\n---\n\n{message}")
                    response = synthesize(llm_fn, _history, augmented,
                                          temperature=0.3,
                                          briefing=presentation_briefing)
            else:
                augmented = message
                if per_query_block:
                    augmented = (f"{per_query_block}\n\n---\n\n{message}")
                response = synthesize(
                    llm_fn, _history, augmented,
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=tool_result, temperature=0.3,
                    max_tokens=1500,
                    briefing=presentation_briefing,
                )

            response = _clean_response(response)
            # Regenerate if garbage (too short, just a number) — but skip
            # for memory_ingest, which intentionally returns a short ack.
            if len(response.strip()) < 10 and tool_name != "memory_ingest":
                response = synthesize(
                    llm_fn, _history, augmented,
                    tool_name=tool_name if tool_name != "conversation" else None,
                    tool_args=tool_args if tool_name != "conversation" else None,
                    tool_result=tool_result if tool_name != "conversation" else None,
                    temperature=0.5,
                    briefing=presentation_briefing)
                response = _clean_response(response)
            if not response and tool_result:
                response = str(tool_result)[:2000]
            if not response or len(response.strip()) < 2:
                response = "I don't have information about that right now."

            # 6. Update history
            _history.append({"role": "user", "content": message})
            _history.append({"role": "assistant", "content": response})
            _history = _trim_history(_history, MAX_HISTORY)

            # 7. Ingest assistant response
            if response:
                ai_result = memory.ingest("assistant", response)
                if ai_result.get("extracted", 0) > 0:
                    _session["facts_extracted"] += ai_result.get("extracted", 0)
                    _add_feed("extraction", f"[midas] {response[:80]}")

            # 8. Stats from last LLM call
            stats = {}
            s = _last_stats
            if s:
                stats = {
                    "tps": s.get("tps", 0),
                    "tokens": s.get("tokens", 0),
                    "accept_rate": s.get("accept_rate", 0),
                    "ngram_drafted": s.get("ngram_drafted", 0),
                    "ngram_accepted": s.get("ngram_accepted", 0),
                    "cpu_drafted": s.get("cpu_drafted", 0),
                    "cpu_accepted": s.get("cpu_accepted", 0),
                    "elapsed": s.get("elapsed", 0),
                }

            # Schedule idle tasks (contradiction scan + retrieval scoring)
            if _idle_queue and response:
                _idle_queue.schedule(
                    injected_memories=_last_subconscious,
                    response_text=response)

            return jsonify({
                "response": response,
                "tool": tool_name,
                "tool_args": tool_args,
                "route_layer": route_layer,
                "memories_recalled": _last_subconscious,
                "stats": stats,
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """Streaming chat — SSE response with tokens as they generate."""
    global _history, _last_subconscious
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    def generate_sse():
        global _history, _last_subconscious
        with _lock:
            if _idle_queue:
                _idle_queue.cancel()
            _session["messages_sent"] += 1

            # Ingest
            memory.ingest("user", message)

            # Route
            l1_result = layer1_route(message)
            tool_name, tool_args = route(message, llm_fn=llm_route_fn)

            # Recall memories
            mem_ctx = None
            presentation_briefing = None
            _last_subconscious = []
            try:
                recall_result = memory.recall(message, n_results=15)
                if recall_result and recall_result.get("results"):
                    filtered = []
                    for r in recall_result["results"]:
                        if r.get("score", 0) < 0.40:
                            continue
                        text = r.get("text", "")
                        if text.startswith("[") and ".md]" in text[:50]:
                            continue
                        if text.strip().endswith("?"):
                            continue
                        filtered.append(r)
                        if len(filtered) >= 8:
                            break
                    mem_ctx = [r["text"] for r in filtered]
                    _last_subconscious = [{"text": r["text"][:200], "score": r["score"]} for r in filtered]
            except Exception:
                pass

            # Build messages for LLM
            from synthesizer import build_messages
            if tool_name == "conversation":
                msgs = build_messages(_history, message, memory_context=mem_ctx)
            else:
                tool_result = execute(tool_name, tool_args)
                msgs = build_messages(_history, message, tool_name=tool_name,
                                     tool_args=tool_args, tool_result=tool_result,
                                     memory_context=mem_ctx)

            # Stream response
            full_response = []
            for chunk in llm_stream(msgs, max_tokens=300, temperature=0.3):
                full_response.append(chunk)
                sse_data = json.dumps({"type": "token", "content": chunk})
                yield f"data: {sse_data}\n\n"

            response = _clean_response("".join(full_response))

            # Update history
            _history.append({"role": "user", "content": message})
            _history.append({"role": "assistant", "content": response})
            _history = _trim_history(_history, MAX_HISTORY)

            # Ingest response
            if response:
                memory.ingest("assistant", response)

            # Schedule idle
            if _idle_queue and response:
                _idle_queue.schedule(injected_memories=_last_subconscious, response_text=response)

            # Send final stats
            stats = _last_stats or {}
            sse_data = json.dumps({"type": "done", "stats": {
                "tps": stats.get("tps", 0), "tokens": stats.get("tokens", 0),
                "accept_rate": stats.get("accept_rate", 0),
                "elapsed": stats.get("elapsed", 0),
            }, "memories_recalled": len(_last_subconscious)})
            yield f"data: {sse_data}\n\n"

    return app.response_class(generate_sse(), mimetype="text/event-stream")


@app.route("/api/queue_depth")
def api_queue_depth():
    try:
        s = memory.stats()
        return jsonify({"ane_queue_depth": s.get("ane_queue_depth", 0)})
    except Exception:
        return jsonify({"ane_queue_depth": -1})


@app.route("/api/stats")
def api_stats():
    mem_stats = {}
    try:
        mem_stats = memory.stats()
    except Exception:
        pass

    # Check hardware
    gpu_online = False
    try:
        r = urllib.request.urlopen(MLX_BASE_URL.rstrip("/") + "/models", timeout=2)
        gpu_online = r.status == 200
    except Exception:
        pass

    ane_online = False
    try:
        ane_online = urllib.request.urlopen("http://localhost:8423/health", timeout=2).status == 200
    except Exception:
        pass

    return jsonify({
        "hardware": {
            "gpu": {"status": "active" if gpu_online else "down", "label": "70B Q4 | 9.5 tok/s"},
            "ane": {"status": "active" if ane_online else "idle", "label": "1B Fused | 50.2 tok/s | 25d+C"},
            "cpu": {"status": "active", "label": "AMX + N-gram"},
            "memory": {"status": "active", "label": f"{mem_stats.get('total_memories', 0):,} facts | LocalMemoryStore"},
        },
        "session": _session,
        "subconscious": _last_subconscious,
    })


@app.route("/api/feed")
def api_feed():
    return jsonify({"events": _feed})


# ── HTML/CSS/JS ─────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MIDAS</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg: #0a0e14;
    --bg-elevated: #0d1117;
    --bg-panel: #0c1018;
    --border: #1a1f2e;
    --text: #c5cdd8;
    --text-dim: #4a5568;
    --text-muted: #2d3748;
    --cyan: #00d4ff;
    --green: #00ff88;
    --red: #ff4444;
    --cyan-glow: rgba(0, 212, 255, 0.15);
    --green-glow: rgba(0, 255, 136, 0.15);
}

html, body {
    height: 100%;
    overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 13px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1a1f2e; }
::-webkit-scrollbar-thumb:hover { background: #2d3748; }

/* Layout */
.container {
    display: flex;
    height: 100vh;
    width: 100vw;
}

/* Left panel: Chat */
.chat-panel {
    flex: 0 0 65%;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    min-width: 0;
}

.chat-header {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
}

.chat-header .logo {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 4px;
    color: var(--cyan);
}

.chat-header .status {
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 1px;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.msg {
    max-width: 85%;
    padding: 8px 12px;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}

.msg-user {
    align-self: flex-end;
    border: 1px solid var(--border);
    color: var(--text);
}

.msg-midas {
    align-self: flex-start;
    border-left: 2px solid var(--cyan);
    padding-left: 14px;
    color: var(--text);
}

.msg-meta {
    font-size: 11px;
    color: var(--text-dim);
    padding: 2px 14px 8px 14px;
    align-self: flex-start;
}

.msg-meta span { margin-right: 12px; }
.msg-meta .route { color: var(--cyan); opacity: 0.6; }
.msg-meta .tps { color: var(--green); opacity: 0.6; }
.msg-meta .tool-tag { color: var(--text-dim); }

.typing-indicator {
    align-self: flex-start;
    padding: 8px 14px;
    color: var(--cyan);
    font-size: 12px;
    opacity: 0.7;
    display: none;
}

.typing-indicator.visible { display: block; }

.chat-input-bar {
    padding: 12px 20px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
}

.chat-input-bar input {
    width: 100%;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text);
    font-family: inherit;
    font-size: 13px;
    padding: 10px 12px;
    outline: none;
    transition: border-color 0.2s;
}

.chat-input-bar input:focus {
    border-color: var(--cyan);
}

.chat-input-bar input::placeholder {
    color: var(--text-muted);
}

/* Right panel: Dashboard */
.dash-panel {
    flex: 0 0 35%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
}

.dash-section {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}

.dash-section.feed-section {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
}

.dash-section.sub-section {
    flex: 0 0 auto;
    max-height: 180px;
    overflow-y: auto;
}

.section-title {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 3px;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Hardware rows */
.hw-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 11px;
}

.hw-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}

.hw-dot.active {
    background: var(--green);
    box-shadow: 0 0 6px var(--green-glow);
}

.hw-dot.idle {
    background: var(--text-muted);
}

.hw-dot.down {
    background: var(--red);
}

.hw-dot.pulse {
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.hw-label {
    width: 52px;
    color: var(--text-dim);
    flex-shrink: 0;
    font-size: 11px;
}

.hw-metric {
    color: var(--text);
    font-size: 11px;
}

/* Session counters */
.counter-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 16px;
    font-size: 11px;
}

.counter-item {
    display: flex;
    justify-content: space-between;
}

.counter-label { color: var(--text-dim); }
.counter-value { color: var(--text); font-weight: 500; }

/* Feed */
.feed-item {
    padding: 4px 0;
    font-size: 11px;
    line-height: 1.4;
    display: flex;
    gap: 8px;
}

.feed-time {
    color: var(--text-muted);
    flex-shrink: 0;
    font-size: 10px;
}

.feed-text {
    color: var(--text-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.feed-item.extraction .feed-text { color: var(--green); opacity: 0.7; }
.feed-item.recall .feed-text { color: var(--cyan); opacity: 0.7; }

/* Subconscious */
.sub-item {
    padding: 3px 0;
    font-size: 11px;
    line-height: 1.4;
    display: flex;
    gap: 8px;
}

.sub-score {
    color: var(--cyan);
    opacity: 0.6;
    flex-shrink: 0;
    font-size: 10px;
    min-width: 32px;
    text-align: right;
}

.sub-text {
    color: var(--text-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.empty-state {
    font-size: 11px;
    color: var(--text-muted);
    font-style: italic;
}

/* Code blocks in responses */
.msg code {
    background: var(--bg-elevated);
    padding: 1px 4px;
    font-size: 12px;
}

.msg pre {
    background: var(--bg-elevated);
    padding: 8px 10px;
    margin: 4px 0;
    overflow-x: auto;
    font-size: 12px;
}
</style>
</head>
<body>
<div class="container">

    <!-- Left: Chat -->
    <div class="chat-panel">
        <div class="chat-header">
            <span class="logo">MIDAS</span>
            <span class="status">LOCAL / 70B Q4 / BARE METAL</span>
        </div>
        <div class="chat-messages" id="messages"></div>
        <div class="typing-indicator" id="typing">processing...</div>
        <div class="chat-input-bar">
            <input type="text" id="input" placeholder="..." autocomplete="off" spellcheck="false">
        </div>
    </div>

    <!-- Right: Dashboard -->
    <div class="dash-panel">
        <div class="dash-section">
            <div class="section-title">HARDWARE</div>
            <div id="hw-gpu" class="hw-row">
                <div class="hw-dot idle"></div>
                <span class="hw-label">GPU</span>
                <span class="hw-metric">--</span>
            </div>
            <div id="hw-ane" class="hw-row">
                <div class="hw-dot idle"></div>
                <span class="hw-label">ANE</span>
                <span class="hw-metric">--</span>
            </div>
            <div id="hw-cpu" class="hw-row">
                <div class="hw-dot idle"></div>
                <span class="hw-label">CPU</span>
                <span class="hw-metric">--</span>
            </div>
            <div id="hw-mem" class="hw-row">
                <div class="hw-dot idle"></div>
                <span class="hw-label">MEM</span>
                <span class="hw-metric">--</span>
            </div>
        </div>

        <div class="dash-section">
            <div class="section-title">THIS SESSION</div>
            <div class="counter-grid">
                <div class="counter-item">
                    <span class="counter-label">messages</span>
                    <span class="counter-value" id="ct-msg">0</span>
                </div>
                <div class="counter-item">
                    <span class="counter-label">recalled</span>
                    <span class="counter-value" id="ct-recall">0</span>
                </div>
                <div class="counter-item">
                    <span class="counter-label">extracted</span>
                    <span class="counter-value" id="ct-extract">0</span>
                </div>
                <div class="counter-item">
                    <span class="counter-label">tools</span>
                    <span class="counter-value" id="ct-tools">0</span>
                </div>
            </div>
        </div>

        <div class="dash-section feed-section">
            <div class="section-title">MEMORY FEED</div>
            <div id="feed"><div class="empty-state">no activity yet</div></div>
        </div>

        <div class="dash-section sub-section">
            <div class="section-title">SUBCONSCIOUS</div>
            <div id="subconscious"><div class="empty-state">no memories injected</div></div>
        </div>
    </div>

</div>

<script>
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const typingEl = document.getElementById('typing');
let sending = false;

function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

function addMessage(role, text, meta) {
    const div = document.createElement('div');
    div.className = 'msg msg-' + role;
    div.textContent = text;
    messagesEl.appendChild(div);

    if (meta && role === 'midas') {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'msg-meta';
        let parts = [];
        if (meta.route_layer) parts.push('<span class="route">' + meta.route_layer + '</span>');
        if (meta.tool && meta.tool !== 'conversation') parts.push('<span class="tool-tag">' + escapeHtml(meta.tool) + '</span>');
        if (meta.stats && meta.stats.tps) parts.push('<span class="tps">' + meta.stats.tps + ' tok/s</span>');
        if (meta.stats && meta.stats.accept_rate) parts.push('<span class="route">' + meta.stats.accept_rate + '% acc</span>');
        if (meta.stats && meta.stats.tokens) parts.push('<span class="route">' + meta.stats.tokens + ' tok</span>');
        if (parts.length) {
            metaDiv.innerHTML = parts.join('');
            messagesEl.appendChild(metaDiv);
        }
    }

    messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || sending) return;
    sending = true;
    inputEl.value = '';
    addMessage('user', text);
    typingEl.classList.add('visible');

    // Pulse GPU dot while generating
    const gpuDot = document.querySelector('#hw-gpu .hw-dot');
    gpuDot && gpuDot.classList.add('pulse');

    try {
        const resp = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text}),
        });

        // Create message div for streaming
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg msg-midas';
        msgDiv.textContent = '';
        messagesEl.appendChild(msgDiv);

        const metaDiv = document.createElement('div');
        metaDiv.className = 'msg-meta';
        messagesEl.appendChild(metaDiv);

        let fullText = '';
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.slice(6).trim();
                if (!dataStr || dataStr === '[DONE]') continue;

                try {
                    const chunk = JSON.parse(dataStr);
                    if (chunk.type === 'token') {
                        fullText += chunk.content;
                        msgDiv.textContent = fullText;
                        messagesEl.scrollTop = messagesEl.scrollHeight;
                    } else if (chunk.type === 'done' && chunk.stats) {
                        const s = chunk.stats;
                        let parts = [];
                        if (s.tps) parts.push('<span class="tps">' + s.tps.toFixed(1) + ' tok/s</span>');
                        if (s.accept_rate) parts.push('<span class="route">' + s.accept_rate.toFixed(0) + '% acc</span>');
                        if (s.tokens) parts.push('<span class="route">' + s.tokens + ' tok</span>');
                        metaDiv.innerHTML = parts.join(' ');
                    }
                } catch (e) {}
            }
        }
    } catch (e) {
        addMessage('midas', 'CONNECTION ERROR: ' + e.message);
    }

    gpuDot && gpuDot.classList.remove('pulse');
    typingEl.classList.remove('visible');
    sending = false;
    refreshFeed();
    refreshStats();
}

inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Dashboard polling
function updateHw(id, data) {
    const row = document.getElementById(id);
    if (!row || !data) return;
    const dot = row.querySelector('.hw-dot');
    const metric = row.querySelector('.hw-metric');
    dot.className = 'hw-dot ' + data.status;
    if (data.status === 'active') {
        dot.style.background = 'var(--green)';
        dot.style.boxShadow = '0 0 6px var(--green-glow)';
    } else if (data.status === 'down') {
        dot.style.background = 'var(--red)';
        dot.style.boxShadow = 'none';
    } else {
        dot.style.background = 'var(--text-muted)';
        dot.style.boxShadow = 'none';
    }
    metric.textContent = data.label;
}

function updateSubconscious(items) {
    const el = document.getElementById('subconscious');
    if (!items || !items.length) {
        el.innerHTML = '<div class="empty-state">no memories injected</div>';
        return;
    }
    el.innerHTML = items.map(m =>
        '<div class="sub-item">' +
        '<span class="sub-score">' + (m.score || 0).toFixed(2) + '</span>' +
        '<span class="sub-text">' + escapeHtml(m.text) + '</span>' +
        '</div>'
    ).join('');
}

async function refreshStats() {
    try {
        const resp = await fetch('/api/stats');
        const data = await resp.json();
        if (data.hardware) {
            updateHw('hw-gpu', data.hardware.gpu);
            updateHw('hw-ane', data.hardware.ane);
            updateHw('hw-cpu', data.hardware.cpu);
            updateHw('hw-mem', data.hardware.memory);
        }
        if (data.session) {
            document.getElementById('ct-msg').textContent = data.session.messages_sent || 0;
            document.getElementById('ct-recall').textContent = data.session.memories_recalled || 0;
            document.getElementById('ct-extract').textContent = data.session.facts_extracted || 0;
            document.getElementById('ct-tools').textContent = data.session.tools_used || 0;
        }
        if (data.subconscious) {
            updateSubconscious(data.subconscious);
        }
    } catch (e) {}
}

async function refreshFeed() {
    try {
        const resp = await fetch('/api/feed');
        const data = await resp.json();
        const el = document.getElementById('feed');
        if (!data.events || !data.events.length) {
            el.innerHTML = '<div class="empty-state">no activity yet</div>';
            return;
        }
        el.innerHTML = data.events.map(ev =>
            '<div class="feed-item ' + ev.type + '">' +
            '<span class="feed-time">' + ev.time + '</span>' +
            '<span class="feed-text">' + escapeHtml(ev.text) + '</span>' +
            '</div>'
        ).join('');
    } catch (e) {}
}

// Poll every 2s
setInterval(refreshStats, 2000);
setInterval(refreshFeed, 5000);

// Initial load
refreshStats();
refreshFeed();
inputEl.focus();
</script>
</body>
</html>
"""


# ── Boot & Run ──────────────────────────────────────────────────────────────

def boot():
    """Initialize memory and wire singletons."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        memory.start()
    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        print(f"Boot failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    sys.stdout, sys.stderr = old_stdout, old_stderr

    set_memory(memory)

    global _idle_queue
    _idle_queue = IdleQueue(memory)

    # Browser (optional)
    try:
        from browser import BrowserBridge
        browser = BrowserBridge()
        if browser.is_available():
            browser.connect()
            set_browser(browser)
    except Exception:
        pass

    # Test LLM
    try:
        llm_fn([{"role": "user", "content": "hi"}], max_tokens=5, temperature=0.0)
    except Exception:
        print("LLM offline -- is MLX server running on :8899?")
        sys.exit(1)

    stats = memory.stats()
    print(f"  MIDAS UI booting...")
    print(f"  {stats.get('total_memories', 0)} memories loaded")
    print(f"  http://127.0.0.1:{PORT}")


if __name__ == "__main__":
    boot()
    # Suppress Flask request logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
