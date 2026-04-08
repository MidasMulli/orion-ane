"""
Head-to-head: GPU-only vs Speculative Decoding (ANE draft + GPU verify)
======================================================================

Same prompt, same output length target. Measures:
  1. GPU-only: Qwen3.5-9B via MLX server, standard generation
  2. Speculative: ANE Qwen3-0.6B drafts K tokens → GPU verifies batch

Uses incremental drafting — KV cache persists across rounds,
only new tokens are forwarded (no re-prefill).
"""

import sys
import os
import time
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speculative"))
from real_draft import RealDraftModel, MAX_SEQ

# ── Config ───────────────────────────────────────────────────────────

MLX_URL = "http://localhost:8899/v1"
MLX_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
K_DRAFT = 5           # Tokens to draft per round
MAX_TOKENS = 30       # Target generation length
PROMPT = "The capital of France is"


# ── GPU-only baseline ────────────────────────────────────────────────

def gpu_only_generate(prompt, max_tokens):
    """Standard GPU generation via MLX completions API."""
    t0 = time.time()
    r = requests.post(f"{MLX_URL}/completions", json={
        "model": MLX_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }, timeout=60)
    t_total = (time.time() - t0) * 1000
    result = r.json()
    text = result["choices"][0]["text"]
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", max_tokens)
    return text, t_total, completion_tokens


# ── Speculative decoding (incremental) ───────────────────────────────

def speculative_generate(draft_model, prompt, max_tokens, k_draft):
    """
    Incremental speculative decoding:
      - Prefill once, then draft_continue() each round (no re-prefill)
      - On rejection: feed correction token, continue from there
    """
    prompt_ids = draft_model.encode(prompt)
    generated_ids = []
    context_text = prompt

    # Phase 0: Prefill prompt once
    t0 = time.time()
    draft_model.prefill(prompt_ids)
    prefill_ms = (time.time() - t0) * 1000

    total_ane_ms = prefill_ms
    total_gpu_ms = 0
    total_accepted = 0
    total_rejected = 0
    n_rounds = 0
    round_details = []

    # Track confirmed position in KV cache
    confirmed_pos = len(prompt_ids)

    while len(generated_ids) < max_tokens:
        n_rounds += 1
        k = min(k_draft, max_tokens - len(generated_ids))

        # Phase 1: ANE Draft (incremental — only K new tokens)
        t0 = time.time()
        drafts = draft_model.draft_continue(k)
        ane_ms = (time.time() - t0) * 1000
        total_ane_ms += ane_ms

        draft_ids = [d[0] for d in drafts]
        draft_text = draft_model.decode(draft_ids)

        # Phase 2: GPU Verify (generate from current context)
        t0 = time.time()
        gpu_resp = requests.post(f"{MLX_URL}/completions", json={
            "model": MLX_MODEL,
            "prompt": context_text,
            "max_tokens": k + 2,
            "temperature": 0.0,
            "stream": False,
        }, timeout=60)
        gpu_ms = (time.time() - t0) * 1000
        total_gpu_ms += gpu_ms

        gpu_text = gpu_resp.json()["choices"][0]["text"]

        # Tokenize GPU output for comparison
        full_gpu_ids = draft_model.encode(context_text + gpu_text)
        context_ids = draft_model.encode(context_text)
        gpu_ids = full_gpu_ids[len(context_ids):]

        # Phase 3: Token-level comparison
        accepted = 0
        for i in range(min(len(draft_ids), len(gpu_ids))):
            if draft_ids[i] == gpu_ids[i]:
                accepted += 1
            else:
                break

        # Accept matched prefix
        if accepted > 0:
            generated_ids.extend(draft_ids[:accepted])
            context_text += draft_model.decode(draft_ids[:accepted])

        # Handle divergence
        if accepted < len(draft_ids) and accepted < len(gpu_ids):
            # Use GPU's token at the mismatch point
            gpu_tok = gpu_ids[accepted]
            generated_ids.append(gpu_tok)
            context_text += draft_model.decode([gpu_tok])
            total_rejected += 1

            # Roll back draft KV cache to the divergence point and feed
            # the GPU's correction token so the cache is consistent
            rollback_pos = confirmed_pos + accepted
            draft_model.feed_tokens([gpu_tok], rollback_pos)
            confirmed_pos = rollback_pos + 1
        elif accepted == len(draft_ids):
            # All drafted tokens accepted — cache is already correct
            confirmed_pos += accepted
        else:
            # GPU returned empty — stuck
            confirmed_pos += accepted
            if accepted == 0 and len(gpu_ids) == 0:
                break

        total_accepted += accepted

        tokens_this_round = accepted + (1 if accepted < len(draft_ids) else 0)
        round_details.append({
            "round": n_rounds,
            "drafted": k,
            "accepted": accepted,
            "ane_ms": round(ane_ms, 1),
            "gpu_ms": round(gpu_ms, 1),
            "tokens_produced": tokens_this_round,
            "draft_text": draft_text,
        })

    generated_text = draft_model.decode(generated_ids)
    total_ms = total_ane_ms + total_gpu_ms
    acceptance_rate = total_accepted / (total_accepted + total_rejected) \
        if (total_accepted + total_rejected) > 0 else 0

    return {
        "text": generated_text,
        "tokens": len(generated_ids),
        "total_ms": total_ms,
        "ane_ms": total_ane_ms,
        "gpu_ms": total_gpu_ms,
        "prefill_ms": prefill_ms,
        "n_rounds": n_rounds,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "acceptance_rate": acceptance_rate,
        "rounds": round_details,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SPECULATIVE DECODING BENCHMARK (incremental draft)         ║")
    print("║  GPU-only vs ANE Draft + GPU Verify                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Warm up GPU ──
    print("\n▸ Warming up GPU (Qwen3.5-9B)...")
    requests.post(f"{MLX_URL}/completions", json={
        "model": MLX_MODEL, "prompt": "Hello", "max_tokens": 5,
        "temperature": 0.0}, timeout=30)
    print("  ✓ GPU warm")

    # ── Test 1: GPU-only ──
    print(f"\n▸ Test 1: GPU-only generation")
    print(f'  Prompt: "{PROMPT}"')
    print(f"  Target: {MAX_TOKENS} tokens")

    gpu_text, gpu_ms, gpu_tokens = gpu_only_generate(PROMPT, MAX_TOKENS)
    gpu_tok_s = gpu_tokens / (gpu_ms / 1000)

    print(f'  Output: "{PROMPT}{gpu_text}"')
    print(f"  {gpu_tokens} tokens in {gpu_ms:.0f}ms = {gpu_tok_s:.1f} tok/s")

    # ── Load ANE draft model ──
    print(f"\n▸ Loading ANE draft model (fused kernels)...")
    draft = RealDraftModel()
    draft.load_and_compile(lambda msg: print(f"  {msg}"), fused=True)

    if not draft.compiled:
        print("  ✗ ANE compilation failed!")
        return

    c_active = draft._c_model is not None
    print(f"  C forward pass: {'✓ active' if c_active else '✗ Python fallback'}")

    # ── Test 2: Speculative ──
    print(f"\n▸ Test 2: Speculative decoding (K={K_DRAFT}, incremental)")
    print(f'  Prompt: "{PROMPT}"')
    print(f"  Draft: Qwen3-0.6B (ANE, {'C fused' if c_active else 'Python'})")
    print(f"  Verify: Qwen3.5-9B (GPU)")

    spec = speculative_generate(draft, PROMPT, MAX_TOKENS, K_DRAFT)

    print(f'\n  Output: "{PROMPT}{spec["text"]}"')
    print(f"  {spec['tokens']} tokens in {spec['total_ms']:.0f}ms")
    spec_tok_s = spec["tokens"] / (spec["total_ms"] / 1000) if spec["total_ms"] > 0 else 0
    print(f"  Effective: {spec_tok_s:.1f} tok/s")
    print(f"  Acceptance rate: {spec['acceptance_rate']:.1%}")
    print(f"  Rounds: {spec['n_rounds']} (accepted {spec['total_accepted']}, "
          f"rejected {spec['total_rejected']})")
    print(f"  Prefill: {spec['prefill_ms']:.0f}ms")

    # ── Round-by-round breakdown ──
    print(f"\n  Round-by-round:")
    print(f"  {'Rnd':>3} {'Draft':>5} {'Accept':>6} {'Rate':>5} "
          f"{'ANE ms':>7} {'GPU ms':>7} {'Tokens':>6} {'Draft Text'}")
    print(f"  {'───':>3} {'─────':>5} {'──────':>6} {'─────':>5} "
          f"{'──────':>7} {'──────':>7} {'──────':>6} {'──────────'}")
    for rd in spec["rounds"]:
        rate = rd["accepted"] / rd["drafted"] if rd["drafted"] > 0 else 0
        print(f"  {rd['round']:3d} {rd['drafted']:5d} {rd['accepted']:6d} "
              f"{rate:5.0%} {rd['ane_ms']:7.0f} {rd['gpu_ms']:7.0f} "
              f"{rd['tokens_produced']:6d} {rd['draft_text']!r}")

    # ── Head-to-head summary ──
    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║  RESULTS                                                     ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  GPU-only:    {gpu_tokens:3d} tokens │ {gpu_ms:7.0f}ms │ {gpu_tok_s:5.1f} tok/s       ║")
    print(f"║  Speculative: {spec['tokens']:3d} tokens │ {spec['total_ms']:7.0f}ms │ {spec_tok_s:5.1f} tok/s       ║")

    if spec_tok_s > gpu_tok_s:
        speedup = spec_tok_s / gpu_tok_s
        print(f"║                                                              ║")
        print(f"║  → Speculative is {speedup:.2f}x FASTER                            ║")
    elif gpu_tok_s > spec_tok_s:
        ratio = gpu_tok_s / spec_tok_s
        print(f"║                                                              ║")
        print(f"║  → GPU-only is {ratio:.2f}x faster                                ║")

    print(f"║                                                              ║")
    print(f"║  Time breakdown (speculative):                               ║")
    print(f"║    Prefill:    {spec['prefill_ms']:7.0f}ms (one-time)                         ║")
    pct_ane = spec['ane_ms']/spec['total_ms']*100 if spec['total_ms'] > 0 else 0
    pct_gpu = spec['gpu_ms']/spec['total_ms']*100 if spec['total_ms'] > 0 else 0
    print(f"║    ANE draft:  {spec['ane_ms']:7.0f}ms ({pct_ane:4.1f}%)                        ║")
    print(f"║    GPU verify: {spec['gpu_ms']:7.0f}ms ({pct_gpu:4.1f}%)                        ║")
    matched = spec['total_accepted']
    total = spec['total_accepted'] + spec['total_rejected']
    print(f"║    Acceptance: {spec['acceptance_rate']:5.1%}  ({matched}/{total} tokens)                  ║")
    tpr = spec['tokens']/spec['n_rounds'] if spec['n_rounds'] > 0 else 0
    print(f"║    Tokens/round: {tpr:.1f} avg                                ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
