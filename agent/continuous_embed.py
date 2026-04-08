"""Main 24 Build 2 — Continuous embedding PoC.

Streams a 72B response, embeds the partial output every N tokens via the
CoreML MiniLM (0.84 ms/embed on ANE), retrieves top-K from LocalMemoryStore
on each embed, and logs "discoveries" — memories whose similarity to the
in-progress thought crosses a threshold.

The architectural question this answers: can the agent notice relevant
prior memories *while it's still talking*, instead of only at the moment
the user submitted the query? At 0.84 ms per embed, the answer is yes —
embedding every 4 tokens of generation costs ~13% of an 8B-ANE step
(7 ms/tok), or ~0.5% of a 72B step (~150 ms/tok at warm spec-decode rate).

PoC scope (deliberately small):
  • runs against the live :8450 UI's underlying llm_stream
  • single query in, log out
  • not wired into the live agent's response path — that's a follow-up
  • the goal is to prove the wiring works and measure overhead

Usage:
    python3 continuous_embed.py "your query here"
    python3 continuous_embed.py "your query here" --window 4 --threshold 0.55
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Generator

# Wire in the existing llm_stream from midas_ui without booting the whole UI
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.expanduser("~/Desktop/cowork/orion-ane/memory"))

from local_store import LocalMemoryStore  # noqa: E402

DB_PATH = os.path.expanduser("~/Desktop/cowork/orion-ane/memory/chromadb_live")


# ─────────────────────────────────────────────────────────────────────────────
# llm_stream — copied from midas_ui to avoid Flask import side-effects.
# Hits the same MLX server endpoint with SSE streaming.
# ─────────────────────────────────────────────────────────────────────────────
import json
import urllib.request

MLX_BASE_URL = "http://127.0.0.1:8899/v1"
MLX_MODEL = "mlx-community/Qwen2.5-72B-Instruct-4bit"


def llm_stream(messages, max_tokens=300, temperature=0.7) -> Generator[str, None, None]:
    payload = json.dumps({
        "model": MLX_MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "repetition_penalty": 1.35, "stream": True,
    }).encode()
    req = urllib.request.Request(
        MLX_BASE_URL.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=300)
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
            if content:
                yield content
        except json.JSONDecodeError:
            continue


# ─────────────────────────────────────────────────────────────────────────────
# Continuous embedder
# ─────────────────────────────────────────────────────────────────────────────
def run(query: str, window_tokens: int = 4, threshold: float = 0.55,
        max_tokens: int = 400, temperature: float = 0.3,
        recall_k: int = 3, max_log: int = 30) -> dict:
    """Stream a 72B response with continuous embedding overlay.

    Args:
        query: the user message
        window_tokens: embed every N tokens of generation
        threshold: similarity threshold for "discovery" logging
        max_tokens: max generation length
        recall_k: number of top memories to retrieve per embed
        max_log: cap on logged events for brevity

    Returns:
        dict with stats + discovery log
    """
    print(f"[continuous-embed] booting LocalMemoryStore (CoreML embedder)...")
    store = LocalMemoryStore(DB_PATH)
    print(f"[continuous-embed]   {store.count()} memories loaded")
    print(f"[continuous-embed]   embedder: {type(store.emb_model).__name__}")

    print(f"\n[continuous-embed] query: {query}")
    print(f"[continuous-embed] window={window_tokens} tok, threshold={threshold}")
    print(f"[continuous-embed] streaming generation...\n")

    messages = [
        {"role": "system", "content": (
            "You are a senior research engineer working on Apple Silicon ML "
            "infrastructure. Answer the user's question briefly and "
            "concretely.")},
        {"role": "user", "content": query},
    ]

    # Streaming buffer state
    full_text = ""
    pending_tokens: list[str] = []
    embed_events: list[dict] = []
    discovery_events: list[dict] = []
    embed_latencies_ms: list[float] = []
    recall_latencies_ms: list[float] = []
    seen_memory_ids: set[str] = set()

    t_start = time.perf_counter()
    t_first_token = None
    n_tokens = 0
    n_embeds = 0

    for chunk in llm_stream(messages, max_tokens=max_tokens, temperature=temperature):
        if t_first_token is None:
            t_first_token = time.perf_counter()
        full_text += chunk
        # Treat each yielded chunk as ~1 token (the SSE delta from MLX is
        # one decoded token's worth in incremental-decode mode). For tighter
        # accounting we'd tokenize, but the PoC doesn't need perfect counts.
        pending_tokens.append(chunk)
        n_tokens += 1
        sys.stdout.write(chunk)
        sys.stdout.flush()

        if len(pending_tokens) >= window_tokens:
            # Embed the rolling tail (last 32 tokens of full_text), retrieve
            # top-K, log discoveries.
            tail_window = full_text[-256:]  # ~32-64 tokens of context
            t0 = time.perf_counter()
            results = store.recall(tail_window, n_results=recall_k)
            recall_ms = (time.perf_counter() - t0) * 1000
            recall_latencies_ms.append(recall_ms)
            # The recall path embeds internally — measure separately for clarity
            t1 = time.perf_counter()
            _ = store.emb_model.encode([tail_window], normalize_embeddings=True)
            embed_ms = (time.perf_counter() - t1) * 1000
            embed_latencies_ms.append(embed_ms)
            n_embeds += 1

            top = results[0] if results else None
            event = {
                "tok_idx": n_tokens,
                "tail_chars": len(tail_window),
                "embed_ms": round(embed_ms, 3),
                "recall_ms": round(recall_ms, 3),
                "top_score": round(top["score"], 3) if top else 0.0,
                "top_text": (top["text"][:100] if top else ""),
                "top_source_role": (top["metadata"].get("source_role", "")
                                    if top else ""),
            }
            embed_events.append(event)

            # Discovery: a memory we haven't logged yet, scoring above threshold
            for r in results:
                fid_proxy = r["text"][:80]  # cheap id proxy
                if r["score"] >= threshold and fid_proxy not in seen_memory_ids:
                    seen_memory_ids.add(fid_proxy)
                    discovery_events.append({
                        "tok_idx": n_tokens,
                        "score": round(r["score"], 3),
                        "source_role": r["metadata"].get("source_role", ""),
                        "text": r["text"][:140],
                    })

            pending_tokens = []

    elapsed = time.perf_counter() - t_start
    ttft = (t_first_token - t_start) if t_first_token else 0.0

    print(f"\n\n[continuous-embed] generation done")
    print(f"  total tokens (chunks): {n_tokens}")
    print(f"  total elapsed:         {elapsed:.2f}s  (ttft {ttft:.2f}s)")
    print(f"  effective tps:         {n_tokens/max(elapsed-ttft, 0.001):.1f}")
    print(f"  embeds run:            {n_embeds}")
    if embed_latencies_ms:
        avg_emb = sum(embed_latencies_ms) / len(embed_latencies_ms)
        avg_rec = sum(recall_latencies_ms) / len(recall_latencies_ms)
        print(f"  avg embed latency:     {avg_emb:.2f} ms")
        print(f"  avg recall latency:    {avg_rec:.2f} ms (includes embed)")
        print(f"  total overhead:        {sum(recall_latencies_ms):.0f} ms "
              f"({sum(recall_latencies_ms)/elapsed/10:.1f}% of wall)")
    print(f"  discoveries:           {len(discovery_events)}")

    if discovery_events:
        print(f"\n[continuous-embed] discoveries (sim ≥ {threshold}):")
        for i, d in enumerate(discovery_events[:max_log], 1):
            print(f"  {i:2d}. [tok {d['tok_idx']:3d}] "
                  f"score={d['score']:.3f} [{d['source_role']:9s}] "
                  f"{d['text'][:120]}")

    return {
        "query": query,
        "n_tokens": n_tokens,
        "n_embeds": n_embeds,
        "elapsed_s": round(elapsed, 2),
        "ttft_s": round(ttft, 2),
        "tps": round(n_tokens / max(elapsed - ttft, 0.001), 1),
        "avg_embed_ms": round(sum(embed_latencies_ms) / max(len(embed_latencies_ms), 1), 2),
        "avg_recall_ms": round(sum(recall_latencies_ms) / max(len(recall_latencies_ms), 1), 2),
        "total_overhead_ms": round(sum(recall_latencies_ms), 0),
        "discoveries": discovery_events,
        "embed_events": embed_events,
        "full_response": full_text,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?",
                   default="How does the new LocalMemoryStore architecture "
                           "compare to the chromadb backend it replaced?")
    p.add_argument("--window", type=int, default=4,
                   help="embed every N token chunks")
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--out", default="/tmp/main24_build2_continuous_embed.json")
    args = p.parse_args()

    result = run(args.query, window_tokens=args.window,
                 threshold=args.threshold, max_tokens=args.max_tokens)

    import json as _json
    with open(args.out, "w") as f:
        _json.dump(result, f, indent=2)
    print(f"\n[continuous-embed] full log: {args.out}")
