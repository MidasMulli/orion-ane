#!/usr/bin/env python3
"""
Head-to-head: GPU-only vs Speculative Decoding (same-process)
=============================================================

Both models loaded in the same process:
  - Draft: Qwen3-0.6B on ANE (C forward pass, fused kernels)
  - Verify: Qwen3.5-9B on GPU (MLX, persistent KV cache)

No HTTP. No re-prefill. Batch verification with KVCache.trim().

Run with: ~/.mlx-env/bin/python3 bench_speculative_local.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speculative"))

from real_draft import RealDraftModel, MAX_SEQ
from mlx_local_verifier import MLXLocalVerifier

import mlx.core as mx

# ── Config ───────────────────────────────────────────────────────────

K_DRAFT = 5           # Tokens to draft per round
MAX_TOKENS = 30       # Target generation length
PROMPT = "The capital of France is"
VERIFIER_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"  # Same Qwen3 tokenizer as draft


# ── GPU-only baseline (in-process MLX) ───────────────────────────────

def gpu_only_generate(verifier, prompt, max_tokens):
    """Standard autoregressive generation using MLX model."""
    token_ids = verifier.tokenizer.encode(prompt)
    verifier.reset_cache()

    t0 = time.time()
    # Prefill
    x = mx.array([token_ids])
    logits = verifier.model(x, cache=verifier.cache)
    mx.eval(logits)

    # Decode one token at a time
    generated = []
    for _ in range(max_tokens):
        tok = mx.argmax(logits[0, -1, :]).item()
        generated.append(tok)
        x = mx.array([[tok]])
        logits = verifier.model(x, cache=verifier.cache)
        mx.eval(logits)

    t_total = (time.time() - t0) * 1000
    text = verifier.tokenizer.decode(generated)
    return text, t_total, len(generated)


# ── Speculative decoding (same-process) ──────────────────────────────

def speculative_generate(draft_model, verifier, prompt, max_tokens, k_draft):
    """
    Same-process speculative decoding with proper KV cache management.

    Each round:
      1. ANE drafts K tokens (incremental from cache)
      2. GPU batch-verifies K tokens in one forward pass
      3. Compare token-by-token:
         - Accept matching prefix
         - On first mismatch: use GPU's token, trim caches, continue
         - If all K match: accept all + GPU's bonus token
    """
    prompt_ids = draft_model.encode(prompt)
    # Same tokenizer family — use draft model's tokenizer for both
    verifier_prompt_ids = prompt_ids

    generated_ids = []
    total_ane_ms = 0
    total_gpu_ms = 0
    total_accepted = 0
    total_rejected = 0
    n_rounds = 0
    round_details = []

    # ── Prefill both models ──
    t0 = time.time()
    draft_model.prefill(prompt_ids)
    prefill_ane_ms = (time.time() - t0) * 1000
    total_ane_ms += prefill_ane_ms

    t0 = time.time()
    verifier.reset_cache()
    x = mx.array([verifier_prompt_ids])
    v_logits = verifier.model(x, cache=verifier.cache)
    mx.eval(v_logits)
    # v_logits[0, -1, :] predicts the first generated token
    prefill_gpu_ms = (time.time() - t0) * 1000
    total_gpu_ms += prefill_gpu_ms

    while len(generated_ids) < max_tokens:
        n_rounds += 1
        k = min(k_draft, max_tokens - len(generated_ids))

        # ── Phase 1: ANE drafts K tokens ──
        t0 = time.time()
        drafts = draft_model.draft_continue(k)
        ane_ms = (time.time() - t0) * 1000
        total_ane_ms += ane_ms

        draft_ids = [d[0] for d in drafts]
        draft_text = draft_model.decode(draft_ids)

        # ── Phase 2: GPU batch-verifies ──
        # v_logits[0, -1, :] from previous round/prefill predicts position 0
        t0 = time.time()

        # What does the verifier predict before seeing any draft tokens?
        v_tok_0 = mx.argmax(v_logits[0, -1, :]).item()

        # Feed all K draft tokens to verifier in one forward pass
        x = mx.array([draft_ids])
        v_logits_batch = verifier.model(x, cache=verifier.cache)
        mx.eval(v_logits_batch)
        # v_logits_batch[0, i, :] = prediction after seeing draft[0..i]
        v_toks = mx.argmax(v_logits_batch[0], axis=-1).tolist()

        gpu_ms = (time.time() - t0) * 1000

        # ── Phase 3: Compare ──
        # Position 0: compare v_tok_0 vs draft_ids[0]
        # Position i (i>=1): compare v_toks[i-1] vs draft_ids[i]
        accepted = 0
        if v_tok_0 == draft_ids[0]:
            accepted = 1
            for i in range(1, k):
                if v_toks[i - 1] == draft_ids[i]:
                    accepted += 1
                else:
                    break

        # ── Phase 4: Update state ──
        if accepted == k:
            # All K tokens accepted!
            # Accept all draft tokens + bonus token from verifier
            generated_ids.extend(draft_ids)
            bonus_tok = v_toks[k - 1]  # verifier's prediction after seeing all K
            generated_ids.append(bonus_tok)
            total_accepted += k

            # Feed bonus token to both models
            # Draft: feed at current position
            bonus_pos = len(prompt_ids) + len(generated_ids) - 1
            draft_model.feed_tokens([bonus_tok], bonus_pos)

            # Verifier: feed bonus token to update cache + get logits for next round
            x = mx.array([[bonus_tok]])
            v_logits = verifier.model(x, cache=verifier.cache)
            mx.eval(v_logits)

            tokens_this_round = k + 1

        else:
            # Rejected at position `accepted`
            # Accept draft[0..accepted-1], use verifier's token at `accepted`
            if accepted > 0:
                generated_ids.extend(draft_ids[:accepted])
            total_accepted += accepted
            total_rejected += 1

            # Correction token
            if accepted == 0:
                correction_tok = v_tok_0
            else:
                correction_tok = v_toks[accepted - 1]
            generated_ids.append(correction_tok)

            # Re-prefill verifier with full context (prompt + generated so far)
            # ArraysCache doesn't support trim, so we rebuild from scratch.
            # Same tokenizer = same token IDs, so we can use generated_ids directly.
            t_reprefill = time.time()
            all_verifier_ids = list(prompt_ids) + list(generated_ids)
            verifier.reset_cache()
            x = mx.array([all_verifier_ids])
            v_logits = verifier.model(x, cache=verifier.cache)
            mx.eval(v_logits)
            gpu_ms += (time.time() - t_reprefill) * 1000  # count re-prefill in GPU time

            # Roll back draft model and feed correction
            rollback_pos = len(prompt_ids) + len(generated_ids) - 1
            draft_model.feed_tokens([correction_tok], rollback_pos)

            tokens_this_round = accepted + 1

        total_gpu_ms += gpu_ms

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
        "prefill_ane_ms": prefill_ane_ms,
        "prefill_gpu_ms": prefill_gpu_ms,
        "n_rounds": n_rounds,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "acceptance_rate": acceptance_rate,
        "rounds": round_details,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SPECULATIVE DECODING — Same Process                        ║")
    print("║  ANE 0.6B draft (C fused) + GPU 4B verify (MLX in-process)   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Load verifier ──
    print(f"\n▸ Loading GPU verifier ({VERIFIER_MODEL})...")
    verifier = MLXLocalVerifier(model_path=VERIFIER_MODEL)
    verifier.load(lambda msg: print(f"  {msg}"))

    # ── GPU-only baseline ──
    print(f'\n▸ Test 1: GPU-only generation (in-process MLX)')
    print(f'  Prompt: "{PROMPT}"')

    # Warm up
    gpu_only_generate(verifier, "Hello", 5)

    gpu_text, gpu_ms, gpu_tokens = gpu_only_generate(verifier, PROMPT, MAX_TOKENS)
    gpu_tok_s = gpu_tokens / (gpu_ms / 1000)
    print(f'  Output: "{PROMPT}{gpu_text}"')
    print(f"  {gpu_tokens} tokens in {gpu_ms:.0f}ms = {gpu_tok_s:.1f} tok/s")

    # ── Load ANE draft ──
    print(f"\n▸ Loading ANE draft model (fused kernels)...")
    draft = RealDraftModel()
    draft.load_and_compile(lambda msg: print(f"  {msg}"), fused=True)

    if not draft.compiled:
        print("  ✗ ANE compilation failed!")
        return

    c_active = draft._c_model is not None
    print(f"  C forward pass: {'✓ active' if c_active else '✗ Python fallback'}")

    # ── Speculative decode ──
    print(f"\n▸ Test 2: Speculative decoding (K={K_DRAFT}, same-process)")
    print(f"  Draft: Qwen3-0.6B (ANE, {'C fused' if c_active else 'Python'})")
    print(f"  Verify: Qwen3-4B (MLX in-process, cached)")

    spec = speculative_generate(draft, verifier, PROMPT, MAX_TOKENS, K_DRAFT)

    print(f'\n  Output: "{PROMPT}{spec["text"]}"')
    print(f"  {spec['tokens']} tokens in {spec['total_ms']:.0f}ms")
    spec_tok_s = spec["tokens"] / (spec["total_ms"] / 1000) if spec["total_ms"] > 0 else 0
    print(f"  Effective: {spec_tok_s:.1f} tok/s")
    print(f"  Acceptance rate: {spec['acceptance_rate']:.1%}")
    print(f"  Rounds: {spec['n_rounds']} (accepted {spec['total_accepted']}, "
          f"rejected {spec['total_rejected']})")

    # Round details
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

    # ── Summary ──
    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║  RESULTS                                                     ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  GPU-only:    {gpu_tokens:3d} tokens │ {gpu_ms:7.0f}ms │ {gpu_tok_s:5.1f} tok/s       ║")
    print(f"║  Speculative: {spec['tokens']:3d} tokens │ {spec['total_ms']:7.0f}ms │ {spec_tok_s:5.1f} tok/s       ║")

    if spec_tok_s > gpu_tok_s:
        speedup = spec_tok_s / gpu_tok_s
        print(f"║  → Speculative is {speedup:.2f}x FASTER                            ║")
    elif gpu_tok_s > spec_tok_s:
        ratio = gpu_tok_s / spec_tok_s
        print(f"║  → GPU-only is {ratio:.2f}x faster                                ║")

    pct_ane = spec['ane_ms']/spec['total_ms']*100 if spec['total_ms'] > 0 else 0
    pct_gpu = spec['gpu_ms']/spec['total_ms']*100 if spec['total_ms'] > 0 else 0
    print(f"║                                                              ║")
    print(f"║  Prefill: ANE {spec['prefill_ane_ms']:.0f}ms + GPU {spec['prefill_gpu_ms']:.0f}ms                       ║")
    print(f"║  ANE draft:  {spec['ane_ms']:7.0f}ms ({pct_ane:4.1f}%)                        ║")
    print(f"║  GPU verify: {spec['gpu_ms']:7.0f}ms ({pct_gpu:4.1f}%)                        ║")
    matched = spec['total_accepted']
    total = matched + spec['total_rejected']
    print(f"║  Acceptance: {spec['acceptance_rate']:5.1%}  ({matched}/{total})                           ║")
    tpr = spec['tokens']/spec['n_rounds'] if spec['n_rounds'] > 0 else 0
    print(f"║  Tokens/round: {tpr:.1f} avg                                  ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
