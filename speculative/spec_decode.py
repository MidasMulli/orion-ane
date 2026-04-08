"""
Speculative Decoding: ANE Draft + GPU Verify
============================================

Real Qwen3-0.6B on Neural Engine drafts K candidate tokens.
Qwen 3.5 9B on GPU verifies them — accepted tokens are free.
Same tokenizer family = real acceptance rates.

Dashboard at http://localhost:8470
"""

import asyncio
import json
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from aiohttp import web
import aiohttp
import requests

from real_draft import RealDraftModel
from mlx_verifier import MLXVerifier

# ── Config ───────────────────────────────────────────────────────────
K_DRAFT = 5              # Draft tokens per speculation round
MAX_ROUNDS = 6           # Max speculation rounds per generation
MAX_TOKENS = 30          # Max total tokens to generate
PORT = 8470

# ── State ────────────────────────────────────────────────────────────
ws_clients = set()
history = []
stats = {
    "total_tokens": 0,
    "accepted_tokens": 0,
    "rejected_tokens": 0,
    "total_rounds": 0,
    "acceptance_rate": 0.0,
    "ane_time_ms": 0,
    "gpu_time_ms": 0,
    "ane_tok_per_sec": 0,
    "gpu_tok_per_sec": 0,
    "tokens_per_round": 0,
    "draft_model": "Qwen3-0.6B (ANE)",
    "verify_model": "Qwen3.5-9B (GPU)",
}


async def broadcast(msg):
    global ws_clients
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    ws_clients -= dead


# ── Speculative Decoding Engine ──────────────────────────────────────

class SpeculativeDecoder:
    def __init__(self):
        self.draft = None
        self.verifier = None
        self.initialized = False

    async def initialize(self):
        """Initialize ANE draft model and GPU verifier."""
        await broadcast({"type": "status", "msg": "Loading Qwen3-0.6B → Neural Engine..."})

        self.draft = RealDraftModel()

        # Compile ANE kernels with fusion (runs synchronously — heavy)
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, self.draft.load_and_compile,
            lambda msg: print(f"  [ANE] {msg}"),
            True)  # fused=True

        if not success:
            await broadcast({"type": "error", "msg": "ANE model compilation failed"})
            return False

        await broadcast({
            "type": "status",
            "msg": f"ANE ready: {self.draft.n_kernels} kernels, "
                   f"{self.draft.n_layers}L Qwen3-0.6B ✓"
        })

        # Init GPU verifier
        await broadcast({"type": "status", "msg": "Connecting to GPU verifier..."})
        self.verifier = MLXVerifier()
        if not self.verifier.health_check():
            await broadcast({"type": "error", "msg": "MLX server not running on port 8899"})
            return False

        await broadcast({
            "type": "status",
            "msg": f"GPU verifier ready: {self.verifier.model} ✓"
        })

        self.initialized = True
        return True

    async def speculative_generate(self, prompt, max_tokens=None):
        """
        True speculative decoding with token-level acceptance.

        Loop:
          1. ANE drafts K tokens autoregressively
          2. GPU generates from same context
          3. Compare token-by-token — accept matching prefix
          4. On first mismatch: use GPU's token, continue from there
        """
        if not self.initialized:
            return {"error": "Not initialized"}

        if max_tokens is None:
            max_tokens = MAX_TOKENS

        await broadcast({"type": "generation_start", "prompt": prompt})

        # Tokenize prompt
        prompt_ids = self.draft.encode(prompt)
        generated_ids = []
        all_rounds = []
        total_ane_time = 0
        total_gpu_time = 0
        total_accepted = 0
        total_rejected = 0
        n_rounds = 0

        # Build context for GPU (chat format)
        context_text = prompt

        while len(generated_ids) < max_tokens and n_rounds < MAX_ROUNDS:
            n_rounds += 1
            k = min(K_DRAFT, max_tokens - len(generated_ids))

            # ── Phase 1: ANE Draft ──
            await broadcast({
                "type": "phase", "phase": "draft",
                "msg": f"Round {n_rounds}: ANE drafting {k} tokens...",
                "round": n_rounds
            })

            t0 = time.time()
            # Draft from current context
            current_prompt_ids = prompt_ids + generated_ids
            drafts = await asyncio.get_event_loop().run_in_executor(
                None, self.draft.generate_draft, current_prompt_ids, k)
            ane_time = (time.time() - t0) * 1000
            total_ane_time += ane_time

            draft_ids = [d[0] for d in drafts]
            draft_text = self.draft.decode(draft_ids)
            ane_tok_sec = len(draft_ids) / (ane_time / 1000) if ane_time > 0 else 0

            await broadcast({
                "type": "draft_complete",
                "round": n_rounds,
                "tokens": draft_ids,
                "text": draft_text,
                "time_ms": round(ane_time, 1),
                "tok_per_sec": round(ane_tok_sec, 1),
            })

            # ── Phase 2: GPU Verify ──
            await broadcast({
                "type": "phase", "phase": "verify",
                "msg": f"Round {n_rounds}: GPU verifying {k} tokens...",
                "round": n_rounds
            })

            t0 = time.time()
            # Raw text completion (no chat template) — matches base model behavior
            gpu_response, gpu_time = await asyncio.get_event_loop().run_in_executor(
                None, self.verifier.complete_raw,
                context_text, k + 2, 0.0)
            total_gpu_time += gpu_time

            # Tokenize GPU output as continuation (preserves leading-space context)
            # MLX completions API strips leading space, so re-add it for correct tokenization
            separator = " " if gpu_response and not gpu_response[0].isspace() else ""
            full_gpu_ids = self.draft.encode(context_text + separator + gpu_response)
            context_ids = self.draft.encode(context_text)
            gpu_ids = full_gpu_ids[len(context_ids):]  # Extract only generated tokens
            print(f"  [DEBUG R{n_rounds}] gpu_response={gpu_response[:30]!r} sep={separator!r} "
                  f"draft[0]={draft_ids[0]} gpu[0]={gpu_ids[0] if gpu_ids else 'EMPTY'} "
                  f"match={draft_ids[0]==gpu_ids[0] if gpu_ids else False}")
            gpu_tok_sec = len(gpu_ids) / (gpu_time / 1000) if gpu_time > 0 else 0

            # ── Phase 3: Token-level comparison ──
            accepted = 0
            token_results = []

            for i in range(min(len(draft_ids), len(gpu_ids))):
                if draft_ids[i] == gpu_ids[i]:
                    accepted += 1
                    token_results.append({
                        "pos": i,
                        "draft": draft_ids[i],
                        "gpu": gpu_ids[i],
                        "draft_text": self.draft.decode([draft_ids[i]]),
                        "status": "accepted"
                    })
                else:
                    token_results.append({
                        "pos": i,
                        "draft": draft_ids[i],
                        "gpu": gpu_ids[i],
                        "draft_text": self.draft.decode([draft_ids[i]]),
                        "gpu_text": self.draft.decode([gpu_ids[i]]),
                        "status": "rejected"
                    })
                    break  # Stop at first mismatch

            # Accept matched prefix + use GPU token at divergence
            if accepted > 0:
                generated_ids.extend(draft_ids[:accepted])
                context_text += self.draft.decode(draft_ids[:accepted])

            # If we rejected, use GPU's token at the mismatch point
            if accepted < len(draft_ids) and accepted < len(gpu_ids):
                gpu_tok = gpu_ids[accepted]
                generated_ids.append(gpu_tok)
                context_text += self.draft.decode([gpu_tok])
                total_rejected += 1

            total_accepted += accepted
            round_rate = accepted / k if k > 0 else 0

            round_info = {
                "round": n_rounds,
                "k_draft": k,
                "accepted": accepted,
                "draft_ids": draft_ids,
                "gpu_ids": gpu_ids[:k+1],
                "draft_text": draft_text,
                "gpu_text": gpu_response[:100],
                "tokens": token_results,
                "acceptance_rate": round(round_rate, 2),
                "ane_time_ms": round(ane_time, 1),
                "gpu_time_ms": round(gpu_time, 1),
            }
            all_rounds.append(round_info)

            await broadcast({
                "type": "verify_complete",
                "round": n_rounds,
                "accepted": accepted,
                "rejected": k - accepted,
                "acceptance_rate": round(round_rate, 2),
                "tokens": token_results,
                "gpu_text": gpu_response[:100],
                "time_ms": round(gpu_time, 1),
                "tok_per_sec": round(gpu_tok_sec, 1),
            })

            # If nothing was accepted and GPU gave empty response, stop
            if accepted == 0 and len(gpu_ids) == 0:
                break

        # ── Final result ──
        generated_text = self.draft.decode(generated_ids)
        total_time = total_ane_time + total_gpu_time
        overall_acceptance = total_accepted / (total_accepted + total_rejected) \
            if (total_accepted + total_rejected) > 0 else 0

        # Measure baseline (GPU only, same prompt, raw completion)
        baseline_text, baseline_time = self.verifier.complete_raw(
            prompt, max_tokens=max(len(generated_ids) + 2, 10), temperature=0.0)

        # Effective throughput
        tokens_generated = len(generated_ids)
        spec_tok_sec = tokens_generated / (total_time / 1000) if total_time > 0 else 0
        baseline_tok_sec = len(self.draft.encode(baseline_text)) / (baseline_time / 1000) \
            if baseline_time > 0 else 0

        # Update stats
        stats["total_tokens"] += tokens_generated
        stats["accepted_tokens"] += total_accepted
        stats["rejected_tokens"] += total_rejected
        stats["total_rounds"] += n_rounds
        stats["acceptance_rate"] = round(overall_acceptance, 3)
        stats["ane_time_ms"] += total_ane_time
        stats["gpu_time_ms"] += total_gpu_time
        stats["ane_tok_per_sec"] = round(
            tokens_generated / (total_ane_time / 1000) if total_ane_time > 0 else 0, 1)
        stats["gpu_tok_per_sec"] = round(baseline_tok_sec, 1)
        stats["tokens_per_round"] = round(
            stats["total_tokens"] / stats["total_rounds"] if stats["total_rounds"] > 0 else 0, 1)

        result = {
            "prompt": prompt,
            "response": generated_text,
            "baseline_response": baseline_text[:200],
            "tokens_generated": tokens_generated,
            "n_rounds": n_rounds,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "acceptance_rate": round(overall_acceptance, 3),
            "ane_time_ms": round(total_ane_time, 1),
            "gpu_time_ms": round(total_gpu_time, 1),
            "total_time_ms": round(total_time, 1),
            "baseline_time_ms": round(baseline_time, 1),
            "spec_tok_per_sec": round(spec_tok_sec, 1),
            "baseline_tok_per_sec": round(baseline_tok_sec, 1),
            "rounds": all_rounds,
            "generated_ids": generated_ids,
            "timestamp": time.time(),
        }

        history.append(result)
        await broadcast({"type": "result", "data": result})
        return result


# ── Global decoder ───────────────────────────────────────────────────
decoder = SpeculativeDecoder()


# ── HTTP Routes ──────────────────────────────────────────────────────

async def handle_index(request):
    path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    return web.FileResponse(path)

async def handle_init(request):
    success = await decoder.initialize()
    return web.json_response({"success": success})

async def handle_generate(request):
    data = await request.json()
    prompt = data.get("prompt", "What is 2+2?")
    max_tokens = data.get("max_tokens", MAX_TOKENS)
    result = await decoder.speculative_generate(prompt, max_tokens)
    return web.json_response(result)

async def handle_stats(request):
    return web.json_response(stats)

async def handle_history(request):
    return web.json_response(history[-20:])

async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws_clients.add(ws)
    try:
        async for msg in ws:
            pass
    finally:
        ws_clients.discard(ws)
    return ws


# ── Main ─────────────────────────────────────────────────────────────
app = web.Application()
app.router.add_get("/", handle_index)
app.router.add_post("/api/init", handle_init)
app.router.add_post("/api/generate", handle_generate)
app.router.add_get("/api/stats", handle_stats)
app.router.add_get("/api/history", handle_history)
app.router.add_get("/ws", handle_ws)

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SPECULATIVE DECODING — Fused C Forward Pass            ║")
    print("║  ANE: Qwen3-0.6B (122 fused) ↔ GPU: Qwen3.5-9B        ║")
    print(f"║  Dashboard: http://localhost:{PORT}                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    web.run_app(app, host="0.0.0.0", port=PORT, print=lambda *a: None)
