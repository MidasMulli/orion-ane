"""
Acceptance Rate Benchmark: Qwen3 Draft (ANE) vs Target (GPU)
============================================================

Measures how often the draft model agrees with the target model,
token-by-token. This is the key metric for speculative decoding —
higher acceptance = more free tokens.

Both models generate N tokens greedily from the same prompts.
We count position-by-position matches.

Usage:
  python test_acceptance.py                          # 1.7B ANE vs 8B GPU (default)
  python test_acceptance.py --draft Qwen/Qwen3-0.6B  # 0.6B ANE vs 8B GPU
  python test_acceptance.py --target mlx-community/Qwen3-14B-4bit  # vs 14B
  python test_acceptance.py --distilled path/to/weights  # distilled weights
"""

import time
import sys
import os
import gc
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(__file__))

# ── Test Prompts ──────────────────────────────────────────────────
PROMPTS = [
    "The capital of France is",
    "In a collateral agreement, the minimum transfer amount",
    "Once upon a time, there was a",
    "The derivative contract specifies that the counterparty must",
    "To calculate the net present value, you need to",
    "The quick brown fox jumps over the",
    "Under the ISDA Master Agreement, events of default include",
    "Machine learning models are trained by",
    "The interest rate swap has a notional amount of",
    "In Python, you can define a function using",
]

K_TOKENS = 20  # Generate this many tokens per prompt


def main():
    parser = argparse.ArgumentParser(description="Speculative Decode Acceptance Benchmark")
    parser.add_argument("--draft", type=str, default="Qwen/Qwen3-1.7B",
                        help="Draft model (ANE). Default: Qwen/Qwen3-1.7B")
    parser.add_argument("--target", type=str, default="mlx-community/Qwen3-8B-4bit",
                        help="Target model (GPU via MLX). Default: mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--distilled", type=str, default=None,
                        help="Path to distilled weights (safetensors dir)")
    args = parser.parse_args()

    draft_model_name = args.draft
    target_model_name = args.target
    distilled_weights = args.distilled

    # Derive labels
    draft_label = draft_model_name.split("/")[-1]
    if distilled_weights:
        draft_label += " (DISTILLED)"
    target_label = target_model_name.split("/")[-1]

    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm.models.cache import make_prompt_cache

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ACCEPTANCE RATE BENCHMARK                              ║")
    print(f"║  Draft:  {draft_label:47s} ║")
    print(f"║  Target: {target_label:47s} ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── Load Draft Model FIRST (ANE) ─────────────────────────────
    # Load draft first to minimize peak memory — draft weights get
    # extracted to numpy then torch model is freed before target loads
    print(f"Loading draft model ({draft_label}) on ANE...")
    from real_draft import RealDraftModel
    draft = RealDraftModel(model_name=draft_model_name)
    t0 = time.time()
    ok = draft.load_and_compile(
        status_fn=lambda msg: print(f"  {msg}"),
        fused=True,
        weights_path=distilled_weights
    )
    if not ok:
        print("FATAL: ANE model failed to compile")
        sys.exit(1)
    print(f"  Draft loaded in {time.time()-t0:.1f}s")
    print(f"  Architecture: dim={draft.dim}, layers={draft.n_layers}, "
          f"heads={draft.n_heads}/{draft.n_kv_heads}")
    gc.collect()
    print()

    # ── Load Target Model (GPU via MLX) ───────────────────────────
    print(f"Loading target model ({target_model_name}) on GPU...")
    t0 = time.time()
    target_model, target_tokenizer = mlx_load(target_model_name)
    print(f"  Target loaded in {time.time()-t0:.1f}s")
    print()

    # ── Tokenizer Check ───────────────────────────────────────────
    test_text = "Hello world"
    draft_ids = draft.encode(test_text)
    target_ids = target_tokenizer.encode(test_text)
    print(f"Tokenizer check: '{test_text}'")
    print(f"  Draft:  {draft_ids}")
    print(f"  Target: {target_ids}")
    same_tokenizer = (draft_ids == target_ids)
    print(f"  Match: {'✓' if same_tokenizer else '✗ (will affect acceptance rate)'}")
    print()

    # ── Generate from Target (GPU) ────────────────────────────────
    def target_generate(prompt_ids, n_tokens):
        """Generate n_tokens greedily from target model."""
        cache = make_prompt_cache(target_model)
        x = mx.array([prompt_ids])
        logits = target_model(x, cache=cache)
        mx.eval(logits)

        tokens = []
        for _ in range(n_tokens):
            next_tok = mx.argmax(logits[0, -1, :]).item()
            tokens.append(next_tok)
            x = mx.array([[next_tok]])
            logits = target_model(x, cache=cache)
            mx.eval(logits)
        return tokens

    # ── Generate from Draft (ANE) ─────────────────────────────────
    def draft_generate(prompt_ids, n_tokens):
        """Generate n_tokens greedily from draft model."""
        draft.reset_cache()
        logits = None
        for i, tid in enumerate(prompt_ids):
            logits = draft.forward_token(tid, i)

        tokens = []
        pos = len(prompt_ids)
        for _ in range(n_tokens):
            tok = int(np.argmax(logits))
            tokens.append(tok)
            logits = draft.forward_token(tok, pos)
            pos += 1
        return tokens

    # ── Run Benchmark ─────────────────────────────────────────────
    print(f"Running {len(PROMPTS)} prompts × {K_TOKENS} tokens each...")
    print("─" * 70)

    total_tokens = 0
    total_matches = 0
    total_consecutive = 0
    total_draft_ms = 0
    total_target_ms = 0
    results = []

    for i, prompt in enumerate(PROMPTS):
        # Use draft tokenizer (Qwen3 family) for both
        prompt_ids = draft.encode(prompt)

        # Generate from target (GPU)
        t0 = time.time()
        target_tokens = target_generate(prompt_ids, K_TOKENS)
        target_ms = (time.time() - t0) * 1000
        total_target_ms += target_ms

        # Generate from draft (ANE)
        t0 = time.time()
        draft_tokens = draft_generate(prompt_ids, K_TOKENS)
        draft_ms = (time.time() - t0) * 1000
        total_draft_ms += draft_ms

        # Compare
        matches = 0
        consecutive = 0
        first_mismatch = K_TOKENS
        for j in range(K_TOKENS):
            if draft_tokens[j] == target_tokens[j]:
                matches += 1
                if j == consecutive:
                    consecutive += 1
            elif first_mismatch == K_TOKENS:
                first_mismatch = j

        total_tokens += K_TOKENS
        total_matches += matches
        total_consecutive += consecutive

        # Decode for display
        draft_text = draft.decode(draft_tokens)
        target_text = draft.decode(target_tokens)

        accept_rate = matches / K_TOKENS * 100
        results.append({
            "prompt": prompt,
            "accept_rate": accept_rate,
            "matches": matches,
            "consecutive": consecutive,
            "draft_ms": draft_ms,
            "target_ms": target_ms,
        })

        # Visual output
        status = "🟢" if accept_rate >= 60 else "🟡" if accept_rate >= 30 else "🔴"
        print(f"\n{status} Prompt {i+1}: \"{prompt[:50]}...\"")
        print(f"   Accept: {matches}/{K_TOKENS} ({accept_rate:.0f}%)  "
              f"Consecutive: {consecutive}  "
              f"First mismatch: pos {first_mismatch}")
        print(f"   Draft  ({draft_ms:6.0f}ms): {draft_text[:80]}")
        print(f"   Target ({target_ms:6.0f}ms): {target_text[:80]}")

        # Token-by-token comparison
        comp = ""
        for j in range(min(K_TOKENS, 15)):
            if draft_tokens[j] == target_tokens[j]:
                comp += "✓"
            else:
                comp += "✗"
        if K_TOKENS > 15:
            comp += "..."
        print(f"   Tokens: {comp}")

    # ── Summary ───────────────────────────────────────────────────
    overall_rate = total_matches / total_tokens * 100
    avg_consecutive = total_consecutive / len(PROMPTS)
    draft_tok_sec = total_tokens / (total_draft_ms / 1000)
    target_tok_sec = total_tokens / (total_target_ms / 1000)

    print()
    print("═" * 70)
    print("RESULTS")
    print("═" * 70)
    print(f"  Overall acceptance rate:  {total_matches}/{total_tokens} "
          f"({overall_rate:.1f}%)")
    print(f"  Avg consecutive matches:  {avg_consecutive:.1f} / {K_TOKENS}")
    print(f"  Draft speed (ANE):        {draft_tok_sec:.1f} tok/s "
          f"({total_draft_ms/len(PROMPTS):.0f}ms per {K_TOKENS}-token draft)")
    print(f"  Target speed (GPU):       {target_tok_sec:.1f} tok/s "
          f"({total_target_ms/len(PROMPTS):.0f}ms per {K_TOKENS}-token gen)")
    print(f"  Draft speedup:            {draft_tok_sec/target_tok_sec:.1f}x faster")
    print()

    # ── Speculative Decode Estimate ───────────────────────────────
    alpha = overall_rate / 100
    for k in [3, 5, 8]:
        expected_per_round = sum(alpha**i for i in range(k)) + 1
        draft_per_tok = (total_draft_ms / total_tokens)
        target_per_tok = (total_target_ms / total_tokens)
        round_time = k * draft_per_tok + (k+1) * target_per_tok
        effective_tok_sec = expected_per_round / (round_time / 1000)
        speedup = effective_tok_sec / target_tok_sec
        print(f"  K={k}: ~{expected_per_round:.1f} tokens/round, "
              f"~{effective_tok_sec:.1f} tok/s "
              f"({speedup:.2f}x vs target-only)")

    print()
    if overall_rate < 20:
        print("⚠️  Very low acceptance — consider distillation or larger draft model")
    elif overall_rate < 40:
        print("📊 Moderate acceptance — distillation could improve by 20-30%")
    elif overall_rate < 60:
        print("✅ Good acceptance — speculative decode is profitable")
    else:
        print("🚀 Excellent acceptance — significant speedup expected")


if __name__ == "__main__":
    main()
