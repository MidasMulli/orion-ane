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
from pathlib import Path
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
import research_tools

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

# Main 35 +5 Tier 1: live ContextTracker singleton.
# The tracker built in S+1 (multi-topic + brief-query guard, S+4) was
# only used by smoke tests until now. Wire it into the live chat path
# so its state reflects real conversations and can be exposed via
# /api/session/context for the viz to render the active-topic HUD.
_context_tracker = None
def _get_context_tracker():
    global _context_tracker
    if _context_tracker is None:
        try:
            import sys as _sys
            _sys.path.insert(0, "/Users/midas/Desktop/cowork/orion-ane/agent")
            from context_tracker import ContextTracker
            _context_tracker = ContextTracker()
            _context_tracker.on_session_start()
        except Exception as e:
            print(f"context tracker init failed: {e}")
            _context_tracker = None
    return _context_tracker

# ── LLM ─────────────────────────────────────────────────────────────────────

def _llm_call(messages, max_tokens, temperature, stop=None):
    body = {
        "model": MLX_MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "repetition_penalty": 1.35,
    }
    if stop:
        body["stop"] = stop
    payload = json.dumps(body).encode()
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                MLX_BASE_URL.rstrip("/") + "/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json", "Connection": "close"})
            # Main 35 +2: bumped 120→900 per real-transcript prefill diagnosis
            # (m35-72b-real-transcript-wedge.md). Researcher tool calls
            # under 72B contention need long timeouts.
            with urllib.request.urlopen(req, timeout=900) as resp:
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


def llm_fn(messages, max_tokens=1500, temperature=0.7, stop=None, **_kw):
    global _last_stats
    data = _llm_call(messages, max_tokens, temperature, stop=stop)
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
    # Main 35 +2: streaming reads — also bumped to 900s per the same diagnosis.
    resp = urllib.request.urlopen(req, timeout=900)

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
    inject_context = (data or {}).get("inject_context", True)
    if not message:
        return jsonify({"error": "empty message"}), 400

    with _lock:
        try:
            # Cancel any idle work — user message takes priority
            if _idle_queue:
                _idle_queue.cancel()

            _session["messages_sent"] += 1

            # Main 35 +5 Tier 1: feed message to live context tracker.
            # The tracker maintains a multi-topic weight vector which the
            # viz reads via /api/session/context to render the active
            # topic HUD.
            try:
                tracker = _get_context_tracker()
                if tracker is not None:
                    tracker.on_message(message, role="human")
                    # Emit topic state as an event for the viz event log
                    _emit_subconscious_event(
                        "context_topic_update",
                        active_topics=[
                            {"topic": t, "weight": round(w, 3)}
                            for t, w in tracker.active_topics
                        ],
                        message_count=tracker.message_count,
                    )
            except Exception:
                pass

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
            skip_memory = (not inject_context) or (message.lower().rstrip("!?., ") in _casual_greetings)
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
                    _emit_subconscious_event(
                        "memory_recalled",
                        query=message[:100],
                        top_k=len(filtered),
                        top_score=round(filtered[0].get("score", 0), 3) if filtered else 0,
                        path="api_chat")
                    # Main 35 +5 Tier 1: emit one per-memory event so the
                    # viz can animate the retrieval flow as N sequential
                    # node activations rather than a single top-1 flash.
                    #
                    # Rich payload (post Tier 1 fallback fix): include topic,
                    # source_role, type, text so the viz can do 3-tier
                    # fallback matching when `file` is empty (legacy memories
                    # have no source file). Viz tries file → text fuzzy →
                    # topic cluster pulse in that order.
                    for _i, _r in enumerate(filtered):
                        # MemoryBridge.recall flattens the result — fields
                        # live at the top level, not under 'metadata'.
                        _meta = _r.get("metadata", {}) or {}  # legacy fallback
                        _src = (_r.get("source") or _r.get("file")
                                or _meta.get("source") or _meta.get("file") or "")
                        _basename = _src.rsplit("/", 1)[-1] if _src else ""
                        # Derive a "topic" from entities/type since the
                        # bridge doesn't pass through topic directly. Use
                        # query_category as a hint if present.
                        _entities = _r.get("entities") or []
                        _topic_hint = _r.get("query_category", "") or ""
                        _emit_subconscious_event(
                            "memory_recalled_item",
                            index=_i,
                            of=len(filtered),
                            score=round(_r.get("score", 0), 3),
                            file=_basename,
                            text=_r.get("text", "")[:200],
                            topic=_topic_hint,
                            source_role=(_r.get("source_role")
                                          or _meta.get("source_role", "")) or "",
                            mem_type=(_r.get("type")
                                       or _meta.get("type", "")) or "",
                            entities=_entities[:5] if isinstance(_entities, list) else [],
                            query=message[:80],
                        )
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

                # Main 35 +4 T3: prepend a session-open status line on the
                # FIRST message of this process. The /api/session/context
                # endpoint already aggregates daemon health, recent activity,
                # active topic, and overnight queue summaries — surface a
                # compact version of it so the 72B's response naturally
                # incorporates "where did we leave off" awareness.
                if _session.get("messages_sent") == 1:
                    try:
                        import urllib.request as _ur
                        with _ur.urlopen(
                            "http://127.0.0.1:8450/api/session/context",
                            timeout=2.5,
                        ) as r:
                            ctx = json.loads(r.read())
                        lines = ["SESSION OPEN — system context for this turn:"]
                        d = ctx.get("daemon", {}) or {}
                        if d.get("uptime_h") is not None:
                            lines.append(
                                f"  daemon uptime {d['uptime_h']}h, "
                                f"{d.get('events_emitted','?')} events emitted, "
                                f"{d.get('components_running','?')}/5 components running"
                            )
                        sl = ctx.get("since_last_event", {}) or {}
                        if sl.get("hours_ago") is not None:
                            lines.append(
                                f"  last activity {sl['hours_ago']}h ago — "
                                f"{sl.get('n_events_24h',0)} events / "
                                f"{sl.get('loops_fired_24h',0)} maint loops / "
                                f"{sl.get('memories_recalled_24h',0)} recalls / "
                                f"{sl.get('enrichments_24h',0)} enrichments in 24h"
                            )
                        ac = ctx.get("active_context", {}) or {}
                        if ac.get("primary_topic_guess"):
                            lines.append(
                                f"  current focus appears to be {ac['primary_topic_guess']} "
                                f"(topic counts: {ac.get('topic_counts_24h',{})})"
                            )
                        rq = ctx.get("recent_queues", []) or []
                        if rq:
                            queues_str = ", ".join(
                                f"{q['file']} ({q['modified_h_ago']}h ago)"
                                for q in rq[:3]
                            )
                            lines.append(f"  recent overnight queues: {queues_str}")
                        mem = ctx.get("memory", {}) or {}
                        if mem.get("total"):
                            lines.append(
                                f"  memory store: {mem['total']} memories")
                        session_context_block = "\n".join(lines)
                        # Prepend to existing briefing (if any) or use alone
                        if presentation_briefing:
                            presentation_briefing = (
                                session_context_block
                                + "\n\n"
                                + presentation_briefing
                            )
                        else:
                            presentation_briefing = session_context_block
                    except Exception as _e:
                        # Non-fatal: session-open context is bonus, not required
                        pass
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


# ── /api/research — Midas Researcher v1 (Main 27 close) ────────────────────
#
# Agentic read-only research loop. The 70B verifier on :8899 plans tool calls,
# we dispatch them, append results to the rolling context, re-prompt, and stop
# when FINAL_REPORT: is emitted or the iteration cap is hit. Tools are read-
# only (grep, read_file, list_dir, recall_memory) and sandboxed to ~/cowork.
#
# Designed to be delegated to from external agents (e.g. CC subagent loop):
# POST a research goal, get back {report, trace, iterations, elapsed_s}. The
# trace lets the caller audit what Midas actually verified vs hallucinated.
#
# Calibration history lives in vault/CLAUDE_session_log.md and the auto-memory
# `finding_midas_calibration.md`. The first calibration question is the 8B
# fusion target enumeration that motivated this build.

RESEARCHER_SYSTEM_PROMPT = """You are Midas Researcher, an agentic research assistant with read-only access to a software-engineering codebase under /Users/midas/Desktop/cowork. Your job is to investigate a research question by iteratively calling tools, gathering evidence, and producing a final report grounded in real file:line citations.

""" + research_tools.TOOL_CATALOG + """

Rules:
1. Call ONE tool per turn. Wait for the result before calling the next.
2. Plan first. Before your first tool call, write a 1-3 sentence plan.
3. Cite file:line for every code claim in the final report. No fabricated citations.
4. Memory recall is for priors; current code state MUST be verified with grep or read_file.
5. If a sub-claim cannot be verified after 3 tool calls, mark it UNVERIFIED in the report and move on. Do not loop.
6. **STOP CRITERION: as soon as you have ONE concrete grep or read_file result that supports or refutes a claim, that claim is RESOLVED. Do not seek confirmation from a second source. Emit FINAL_REPORT: as soon as every claim is resolved or marked UNVERIFIED.** Padding is failure. Looping past resolution is failure.
7. The iteration cap is 18 turns and the wall clock cap is 480 seconds. Budget accordingly.
8. The FINAL_REPORT marker MUST be at the start of a line. Do not put the literal string "FINAL_REPORT:" anywhere else in your output until you are actually finalizing.
"""

RESEARCHER_MAX_ITERS = 18
RESEARCHER_MAX_WALL_S = 480
RESEARCHER_TOOL_RESULT_CHARS = 6000  # truncate big results to keep context manageable


def _strip_chat_template_tokens(text):
    """Remove Qwen/Llama special tokens and any post-EOS hallucinated turns.

    Calibration test 1 showed the 70B emits `<|im_end|>` after a tool call
    JSON and then hallucinates a fake `Human: ... Assistant: ...` turn from
    its training distribution. We need to truncate at the first EOS marker
    before parsing TOOL_CALL / FINAL_REPORT, otherwise the parser sees noise
    and the FINAL_REPORT regex matches echoed nudge-prompt text.
    """
    if not text:
        return ""
    for marker in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
    # Also cut at any post-completion fake human turn
    text = re.split(r'\n*(?:Human|User|### ?Human|### ?User)\s*:\s*', text, maxsplit=1)[0]
    return text.strip()


def _parse_tool_call(text):
    """Extract a TOOL_CALL: {...} JSON object or detect FINAL_REPORT:.

    Returns ('tool', tool_name, args) | ('final', report_text) | ('none', raw_text).

    Calibration test 1 fixes:
    - Strip chat-template special tokens BEFORE parsing (model emits
      `<|im_end|>` immediately after the JSON brace, no newline).
    - FINAL_REPORT match requires the marker at the START of a line (^), so
      echoed nudge-prompt text containing the substring doesn't false-match.
    - TOOL_CALL terminator accepts EOL OR EOS — JSON object boundary is
      detected by balanced-brace counting, not by the trailing newline.
    """
    if not text:
        return ("none", "")
    text = _strip_chat_template_tokens(text)
    if not text:
        return ("none", "")

    # Final report? Marker MUST be at start of a line (avoids echoed prompt match).
    final_match = re.search(r'(?:^|\n)\s*FINAL_REPORT:\s*(.*)', text, re.DOTALL)
    if final_match:
        report = final_match.group(1).strip()
        if report:
            return ("final", report)

    # Tool call: find "TOOL_CALL:" then parse the next balanced JSON object.
    tc_idx = text.find("TOOL_CALL:")
    if tc_idx >= 0:
        # Skip past "TOOL_CALL:" and any whitespace
        cursor = tc_idx + len("TOOL_CALL:")
        while cursor < len(text) and text[cursor] in " \t\r\n":
            cursor += 1
        if cursor < len(text) and text[cursor] == "{":
            # Walk balanced braces to find the JSON object end
            depth = 0
            in_string = False
            escape = False
            end = cursor
            for i in range(cursor, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_blob = text[cursor:end]
            try:
                obj = json.loads(json_blob)
                tool = obj.get("tool")
                args = obj.get("args", {})
                if tool:
                    return ("tool", tool, args)
            except json.JSONDecodeError:
                pass
    return ("none", text)


@app.route("/api/research", methods=["POST"])
def api_research():
    """Run a directed research loop with read-only tools.

    Request: {"goal": "...", "max_iters": 15 (optional), "max_wall_s": 240 (opt)}
    Response: {"report": "...", "trace": [...], "iterations": N, "elapsed_s": F,
               "stop_reason": "final|cap|wall|error", "tool_calls": N}
    """
    data = request.get_json() or {}
    goal = (data.get("goal") or "").strip()
    if not goal:
        return jsonify({"error": "empty goal"}), 400

    max_iters = min(int(data.get("max_iters", RESEARCHER_MAX_ITERS)), 30)
    max_wall = min(float(data.get("max_wall_s", RESEARCHER_MAX_WALL_S)), 600)

    started = time.time()
    trace = []
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Research goal:\n\n{goal}\n\nBegin with your plan, then your first tool call."},
    ]

    stop_reason = "cap"
    final_report = None
    tool_call_count = 0

    for iteration in range(1, max_iters + 1):
        if time.time() - started > max_wall:
            stop_reason = "wall"
            break

        try:
            # Calibration test 3 finding: qwen_spec_decode_server.py ignores
            # the OpenAI-compat `stop` parameter, so the 70B keeps generating
            # fabricated training-data Q&A after every <|im_end|> token (5-10x
            # token waste per turn). max_tokens=300 caps the bleeding — the
            # actual TOOL_CALL JSON or FINAL_REPORT line fits in <250 tokens.
            # The trailing garbage gets stripped client-side by _parse_tool_call.
            # On the FINAL turn we want the full report so detect "evidence
            # gathered" by allowing more tokens once we've made >=2 tool calls.
            this_turn_max = 300 if tool_call_count < 2 else 800
            assistant_text = llm_fn(messages, max_tokens=this_turn_max, temperature=0.2,
                                    stop=["<|im_end|>", "<|endoftext|>", "\nHuman:", "\nUser:"])
        except Exception as e:
            trace.append({"iter": iteration, "kind": "error", "detail": f"LLM call failed: {type(e).__name__}: {e}"})
            stop_reason = "error"
            break

        parsed = _parse_tool_call(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        trace.append({"iter": iteration, "kind": "assistant", "text": assistant_text})

        if parsed[0] == "final":
            final_report = parsed[1]
            stop_reason = "final"
            break
        elif parsed[0] == "tool":
            _, tool_name, args = parsed
            tool_call_count += 1
            try:
                result = research_tools.dispatch(tool_name, args, memory_bridge=memory)
            except Exception as e:
                result = f"ERROR: tool dispatch raised {type(e).__name__}: {e}"
            if len(result) > RESEARCHER_TOOL_RESULT_CHARS:
                result = result[:RESEARCHER_TOOL_RESULT_CHARS] + f"\n... (truncated; total was {len(result)} chars)"
            trace.append({"iter": iteration, "kind": "tool", "tool": tool_name, "args": args, "result": result})
            messages.append({
                "role": "user",
                "content": f"TOOL_RESULT ({tool_name}):\n{result}\n\nContinue. Call the next tool, or emit FINAL_REPORT: when done."
            })
        else:
            # Model emitted neither TOOL_CALL nor FINAL_REPORT — nudge it.
            # Calibration test 1: avoid putting the literal terminator strings
            # in the nudge prompt because the model may echo them back and
            # cause false-match exits. Refer to them obliquely instead.
            trace.append({"iter": iteration, "kind": "nudge"})
            messages.append({
                "role": "user",
                "content": "I could not parse a tool invocation or terminator from your last message. Re-emit either a tool call line (the TOOL underscore CALL prefix followed by a single JSON object) or a final-report line (the FINAL underscore REPORT prefix at the start of a line, followed by your report body). Use those exact prefixes with the underscores removed when you actually emit them. Continue."
            })

    elapsed = time.time() - started
    return jsonify({
        "report": final_report or "(no final report — loop ended via " + stop_reason + ")",
        "trace": trace,
        "iterations": iteration,
        "tool_calls": tool_call_count,
        "elapsed_s": round(elapsed, 2),
        "stop_reason": stop_reason,
    })


# ── /api/research/queue — Main 30 B2 ───────────────────────────────────────
#
# Queue multiple research tasks for sequential overnight processing. The 70B
# server can only serve one /api/research at a time (single-threaded MLX
# generation), so the queue runs tasks one after another in a background
# thread and writes each result to vault/agent_reports/queue/<queue_id>_*.md.
#
# Result files use wiki-link convention so they're picked up by the Main 30
# B1 realtime enricher and become recallable from Subconscious immediately
# after each task completes.
#
# v1: in-memory queue, lost on server restart. Future: persist to disk.

import threading
import uuid as _uuid
# Main 34 S2A: disk-backed queue lifecycle log so pending tasks survive restart.
import queue_persistence as _qp


def _emit_subconscious_event(ev_type: str, **details) -> None:
    """Main 34 S3 KT4: fire-and-forget POST to subconscious_daemon event bus.
    Daemon down = no event, never blocks the chat path.
    """
    try:
        import urllib.request as _ur
        body = json.dumps({
            "type": ev_type, "component": "midas_ui", "details": details
        }).encode()
        req = _ur.Request(
            "http://127.0.0.1:8452/api/subconscious/emit",
            data=body, headers={"Content-Type": "application/json"})
        _ur.urlopen(req, timeout=0.5).read()
    except Exception:
        pass

_queue_state: dict = {}  # queue_id → {tasks, completed, status, result_paths}
_queue_lock = threading.Lock()
_queue_thread: threading.Thread | None = None
QUEUE_DIR = "/Users/midas/Desktop/cowork/vault/agent_reports/queue"
PER_TASK_TIMEOUT_S = 3600  # Main 35 +4: 600→3600. The Researcher needs
                            # ~30-50 min on real-content paginated reads of
                            # 100KB+ JSON files. Same class as the _llm_call
                            # 120→900 fix from S+2 — original 600s was a
                            # synthetic-input estimate that doesn't survive
                            # real prefill latency.


def _process_queue(queue_id: str):
    """Worker thread: process all tasks in the queue sequentially.

    Wraps the body in try/except so any crash updates state to 'failed'
    rather than leaving it stuck in 'running'.

    Main 34 S4 P1: honors `idx_offset` from state (set by recovery so the
    resumed worker writes lifecycle records and result filenames using
    the GLOBAL task index from the original enqueue, not its local index
    over the remaining-tasks slice. Without this, a second crash-resume
    cycle would scramble the lifecycle log.
    """
    try:
        _process_queue_inner(queue_id)
    except Exception as e:
        with _queue_lock:
            state = _queue_state.get(queue_id)
            if state is not None:
                state["status"] = "failed"
                state["error"] = f"{type(e).__name__}: {e}"
                state["finished_at"] = datetime.now().isoformat()
        try: _qp.log_fail(queue_id, f"{type(e).__name__}: {e}")
        except Exception: pass
        traceback.print_exc()


def _process_queue_inner(queue_id: str):
    """Worker body — sequential task processing with per-task timeout."""
    os.makedirs(QUEUE_DIR, exist_ok=True)
    state = _queue_state[queue_id]
    # P1: when resumed, log/filename indices use the global numbering.
    idx_offset = state.get("idx_offset", 0)
    summary_lines = [
        f"# Research queue {queue_id}",
        f"",
        f"> Started: {state['started_at']}",
        f"> Tasks: {len(state['tasks'])}",
        f"",
    ]

    for i, task in enumerate(state["tasks"]):
        query = task.get("query", "")
        if not query:
            continue
        global_idx = i + idx_offset
        with _queue_lock:
            state["current"] = i
            state["status"] = "running"
        try: _qp.log_start(queue_id, global_idx)
        except Exception: pass

        task_started = time.time()
        # Build a synthetic /api/research call inline. Reuse the same loop driver.
        messages = [
            {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Research goal:\n\n{query}\n\nBegin with your plan, then your first tool call."},
        ]
        trace = []
        final_report = None
        stop_reason = "cap"
        tool_call_count = 0
        max_iters = min(int(task.get("max_iters", 18)), 30)
        for iteration in range(1, max_iters + 1):
            if time.time() - task_started > PER_TASK_TIMEOUT_S:
                stop_reason = "timeout"
                break
            try:
                this_turn_max = 300 if tool_call_count < 2 else 800
                assistant_text = llm_fn(messages, max_tokens=this_turn_max, temperature=0.2,
                                        stop=["<|im_end|>", "<|endoftext|>", "\nHuman:", "\nUser:"])
            except Exception as e:
                trace.append({"iter": iteration, "kind": "error", "detail": f"{type(e).__name__}: {e}"})
                stop_reason = "error"
                break
            parsed = _parse_tool_call(assistant_text)
            messages.append({"role": "assistant", "content": assistant_text})
            trace.append({"iter": iteration, "kind": "assistant", "text": assistant_text})
            if parsed[0] == "final":
                final_report = parsed[1]
                stop_reason = "final"
                break
            elif parsed[0] == "tool":
                _, tool_name, args = parsed
                tool_call_count += 1
                try:
                    result = research_tools.dispatch(tool_name, args, memory_bridge=memory)
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
                if len(result) > RESEARCHER_TOOL_RESULT_CHARS:
                    result = result[:RESEARCHER_TOOL_RESULT_CHARS] + "\n... (truncated)"
                trace.append({"iter": iteration, "kind": "tool", "tool": tool_name, "args": args, "result": result})
                messages.append({"role": "user", "content": f"TOOL_RESULT ({tool_name}):\n{result}\n\nContinue. Call the next tool, or emit FINAL_REPORT: when done."})
            else:
                trace.append({"iter": iteration, "kind": "nudge"})
                messages.append({"role": "user", "content": "I could not parse a tool invocation or terminator. Re-emit either a tool call line or a final-report line."})

        elapsed = time.time() - task_started
        # Write result file (P1: global index for resume safety)
        result_path = f"{QUEUE_DIR}/{queue_id}_task_{global_idx+1:02d}.md"
        report_text = final_report or f"(no final report — stop_reason={stop_reason})"
        body = [
            f"# Queue task {i+1}/{len(state['tasks'])}",
            f"",
            f"> queue: [[{queue_id}_summary|{queue_id}]]",
            f"> stop_reason: {stop_reason} | iterations: {iteration} | tool_calls: {tool_call_count} | elapsed: {elapsed:.1f}s",
            f"",
            f"## Query",
            f"",
            f"```",
            query,
            f"```",
            f"",
            f"## Final Report",
            f"",
            report_text,
            f"",
            f"## Tool Trace",
            f"",
        ]
        for t in trace:
            if t.get("kind") == "tool":
                body.append(f"- **i{t['iter']} {t['tool']}** `{json.dumps(t.get('args', {}))[:140]}`")
                body.append(f"  - {(t.get('result') or '')[:240].replace(chr(10), ' | ')}")
            elif t.get("kind") == "nudge":
                body.append(f"- i{t['iter']} NUDGE")
            elif t.get("kind") == "error":
                body.append(f"- i{t['iter']} ERROR: {t.get('detail', '')[:200]}")
        try:
            with open(result_path, "w") as fh:
                fh.write("\n".join(body) + "\n")
        except OSError as e:
            print(f"queue: failed to write {result_path}: {e}")
        with _queue_lock:
            state["completed"] = i + 1
            state["result_paths"].append(result_path)
        try: _qp.log_task_complete(queue_id, global_idx, result_path)
        except Exception: pass
        summary_lines.append(f"- Task {i+1}: stop={stop_reason} elapsed={elapsed:.1f}s [[{Path(result_path).stem}]]")

    # Write summary file
    summary_lines.append("")
    summary_lines.append(f"> Finished: {datetime.now().isoformat()}")
    summary_path = f"{QUEUE_DIR}/{queue_id}_summary.md"
    try:
        with open(summary_path, "w") as fh:
            fh.write("\n".join(summary_lines) + "\n")
    except OSError as e:
        print(f"queue: failed to write summary {summary_path}: {e}")
    with _queue_lock:
        state["status"] = "complete"
        state["summary_path"] = summary_path
        state["finished_at"] = datetime.now().isoformat()
    try: _qp.log_complete(queue_id)
    except Exception: pass


@app.route("/api/research/queue", methods=["POST"])
def api_research_queue():
    """Queue multiple research tasks for sequential processing.

    Body: {"tasks": [{"query": "..."}, {"query": "..."}, ...]}
    Response: {"queue_id": "...", "task_count": N, "status": "queued"}
    """
    global _queue_thread
    data = request.get_json() or {}
    tasks = data.get("tasks") or []
    if not tasks or not isinstance(tasks, list):
        return jsonify({"error": "tasks must be a non-empty list"}), 400

    queue_id = "q" + datetime.now().strftime("%Y%m%d%H%M%S")
    state = {
        "queue_id": queue_id,
        "tasks": tasks,
        "completed": 0,
        "current": 0,
        "status": "queued",
        "started_at": datetime.now().isoformat(),
        "result_paths": [],
    }
    with _queue_lock:
        _queue_state[queue_id] = state
    try: _qp.log_enqueue(queue_id, tasks)
    except Exception: pass

    t = threading.Thread(target=_process_queue, args=(queue_id,), daemon=True)
    t.start()

    return jsonify({"queue_id": queue_id, "task_count": len(tasks), "status": "queued"})


@app.route("/api/research/queue/<queue_id>", methods=["GET"])
def api_research_queue_status(queue_id):
    with _queue_lock:
        state = _queue_state.get(queue_id)
    if not state:
        return jsonify({"error": f"unknown queue {queue_id}"}), 404
    return jsonify({
        "queue_id": state["queue_id"],
        "completed": state["completed"],
        "total": len(state["tasks"]),
        "status": state["status"],
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "summary_path": state.get("summary_path"),
        "result_paths": state.get("result_paths", []),
    })


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """Streaming chat — SSE response with tokens as they generate."""
    global _history, _last_subconscious
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    inject_context = (data or {}).get("inject_context", True)
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
                recall_result = memory.recall(message, n_results=15) if inject_context else None
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


@app.route("/api/subconscious/health")
def api_subconscious_health():
    """Main 34 S1C: aggregated subconscious health.

    Foundation for the Phase 3 unified daemon. Surfaces:
      - memory store size
      - ane queue depth
      - unresolved contradiction count (from semantic_supersede records)
      - upstream server health (72B :8899, 8B ANE :8891)
    """
    out = {"ts": time.time()}
    try:
        out["memory_stats"] = memory.stats()
    except Exception as e:
        out["memory_stats"] = {"error": str(e)}
    try:
        import sys as _sys
        _sys.path.insert(0, "/Users/midas/Desktop/cowork/vault/subconscious")
        from semantic_supersede import count_unresolved_contradictions
        out["contradictions_unresolved"] = count_unresolved_contradictions()
    except Exception as e:
        out["contradictions_unresolved"] = -1
        out["contradictions_error"] = str(e)
    upstream = {}
    for name, url in (("qwen72b", "http://127.0.0.1:8899/health"),
                      ("ane8b", "http://127.0.0.1:8891/health")):
        try:
            import urllib.request as _u
            with _u.urlopen(url, timeout=1.0) as r:
                upstream[name] = {"ok": True, "code": r.status}
        except Exception as e:
            upstream[name] = {"ok": False, "error": str(e)[:120]}
    out["upstream"] = upstream
    out["session"] = {
        "messages_sent": _session.get("messages_sent", 0),
        "memories_recalled": _session.get("memories_recalled", 0),
        "facts_extracted": _session.get("facts_extracted", 0),
    }
    return jsonify(out)


@app.route("/api/session/context")
def api_session_context():
    """Main 35 +3 T3 — what does the user see when they sit down at Midas?

    Aggregates: time-since-last-session, recent vault activity, daemon
    event log summary, and a heuristic active-topic guess. Returns a
    compact JSON the chat surface can render as a brief status line.
    """
    import os as _os, glob as _glob, json as _json
    out = {"ts": time.time()}

    # ── 1. recent event log activity (since 24h) ──
    log_dir = "/Users/midas/Desktop/cowork/vault/subconscious"
    cutoff = time.time() - 24 * 3600
    event_counts = {}
    n_events = 0
    n_loops = 0
    n_recalls = 0
    n_enrichments = 0
    last_event_ts = None
    for fp in sorted(_glob.glob(f"{log_dir}/event_log_*.jsonl")):
        try:
            with open(fp) as fh:
                for line in fh:
                    try:
                        rec = _json.loads(line)
                    except Exception:
                        continue
                    ts_str = rec.get("ts") or ""
                    # ts is ISO8601 with Z; quick parse
                    try:
                        from datetime import datetime as _dt
                        ts_epoch = _dt.fromisoformat(ts_str.rstrip("Z")).timestamp()
                    except Exception:
                        continue
                    if ts_epoch < cutoff:
                        continue
                    n_events += 1
                    last_event_ts = max(last_event_ts or 0, ts_epoch)
                    t = rec.get("type", "?")
                    event_counts[t] = event_counts.get(t, 0) + 1
                    if t == "loop_fired":
                        n_loops += 1
                    elif t == "memory_recalled":
                        n_recalls += 1
                    elif t == "enrichment_complete":
                        n_enrichments += 1
        except OSError:
            continue
    hours_since_last = (time.time() - last_event_ts) / 3600.0 if last_event_ts else None
    out["since_last_event"] = {
        "hours_ago": round(hours_since_last, 2) if hours_since_last else None,
        "n_events_24h": n_events,
        "loops_fired_24h": n_loops,
        "memories_recalled_24h": n_recalls,
        "enrichments_24h": n_enrichments,
        "by_type": event_counts,
    }

    # ── 2. daemon health ──
    try:
        import urllib.request as _u
        with _u.urlopen("http://127.0.0.1:8452/api/subconscious/status",
                        timeout=2.0) as r:
            ds = _json.loads(r.read())
        out["daemon"] = {
            "uptime_h": round(ds.get("uptime_s", 0) / 3600, 1),
            "events_emitted": ds["components"]["event_bus"]["events_emitted"],
            "components_running": sum(
                1 for c in ds["components"].values()
                if c.get("status") == "running"),
            "upstream": ds["components"]["health_monitor"].get("upstream", {}),
        }
    except Exception as e:
        out["daemon"] = {"error": str(e)[:120]}

    # ── 3. memory store snapshot ──
    try:
        ms = memory.stats()
        out["memory"] = {
            "total": ms.get("total_memories", 0),
            "session": ms.get("session"),
        }
    except Exception as e:
        out["memory"] = {"error": str(e)[:120]}

    # ── 4. recent queue activity ──
    try:
        from pathlib import Path
        queue_dir = Path("/Users/midas/Desktop/cowork/vault/agent_reports/queue")
        recent = sorted(queue_dir.glob("*_summary.md"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:3]
        out["recent_queues"] = [
            {"file": p.name, "modified_h_ago":
             round((time.time() - p.stat().st_mtime) / 3600, 1)}
            for p in recent
        ]
    except Exception as e:
        out["recent_queues"] = []

    # ── 5. heuristic active topic (from recent event log topic tags) ──
    # If the daemon has a context tracker hooked up later, swap to that.
    # For now: count topic mentions in last-24h memory_recalled events.
    topic_counts = {}
    try:
        for fp in sorted(_glob.glob(f"{log_dir}/event_log_*.jsonl"))[-2:]:
            with open(fp) as fh:
                for line in fh:
                    try:
                        rec = _json.loads(line)
                    except Exception:
                        continue
                    if rec.get("type") not in ("memory_recalled", "memory_ingested"):
                        continue
                    details = rec.get("details") or {}
                    q = (details.get("query") or "").lower()
                    for kw, tag in (
                        ("slc", "hardware"), ("ane", "hardware"),
                        ("amcc", "hardware"), ("dcs", "hardware"),
                        ("midas", "midas"), ("/api", "midas"),
                        ("subconscious", "subconscious"),
                        ("memory", "subconscious"),
                        ("paper", "paper"), ("locomo", "paper"),
                        ("cen", "cen"), ("isda", "cen"),
                    ):
                        if kw in q:
                            topic_counts[tag] = topic_counts.get(tag, 0) + 1
                            break
    except Exception:
        pass
    primary = max(topic_counts, key=topic_counts.get) if topic_counts else None
    out["active_context"] = {
        "primary_topic_guess": primary,
        "topic_counts_24h": topic_counts,
    }

    # Main 35 +5 Tier 1: live context tracker state. This is the SOURCE
    # OF TRUTH for the viz's active-topic HUD — it reflects the actual
    # multi-topic weight vector after on_message has been called for every
    # chat message in this midas process. Falls back to the heuristic
    # primary_topic_guess above if the tracker hasn't seen any messages.
    try:
        tracker = _get_context_tracker()
        if tracker is not None:
            out["context_tracker"] = tracker.state()
    except Exception as e:
        out["context_tracker"] = {"error": str(e)[:120]}

    return jsonify(out)


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


def _resume_incomplete_queues():
    """Main 34 S2A: on boot, scan disk-backed queue log and re-fire any
    queues that started but never reached complete. Each recovered queue
    gets a fresh worker thread for its remaining tasks under the SAME
    queue_id so its lifecycle log is continuous.
    """
    try:
        pending = _qp.recover_incomplete_queues()
    except Exception as e:
        print(f"queue recovery failed: {e}")
        return
    if not pending:
        return
    print(f"queue recovery: re-firing {len(pending)} incomplete queue(s)")
    for p in pending:
        qid = p["queue_id"]
        remaining = p["remaining_tasks"]
        completed_so_far = p["completed"]
        state = {
            "queue_id": qid,
            "tasks": remaining,
            "completed": 0,
            "current": 0,
            "status": "queued",
            "started_at": datetime.now().isoformat(),
            "result_paths": [],
            "recovered_from_disk": True,
            "completed_before_recovery": completed_so_far,
            "idx_offset": completed_so_far,  # P1: write global indices in lifecycle log
        }
        with _queue_lock:
            _queue_state[qid] = state
        threading.Thread(target=_process_queue, args=(qid,), daemon=True).start()
        print(f"  resumed {qid}: {len(remaining)} tasks remaining")


if __name__ == "__main__":
    boot()
    _resume_incomplete_queues()
    # Suppress Flask request logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
