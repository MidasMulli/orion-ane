"""
Async Speculative Decode: ANE Draft + GPU Verify (Zero Contention)
==================================================================

The key insight: ANE and GPU are independent compute paths. While the
GPU verifies batch N, the ANE can speculatively draft batch N+1.

Sequential pipeline (what mlx-lm does):
  [ANE draft K] → [GPU verify K] → [ANE draft K] → [GPU verify K] → ...
  Total per round = draft_time + verify_time

Async pipeline (this):
  [ANE draft K] → [GPU verify K]─────────────┐
                  [ANE draft K (speculative)]──┤  ← overlapped!
                                               └→ accept/reject → continue
  Total per round ≈ max(draft_time, verify_time)

If draft=43ms and verify=125ms: sequential=168ms, async=125ms → 1.34x additional speedup.
On top of the 1.29x from spec decode itself → ~1.7x total vs baseline.
"""

import os
import sys
import time
import gc
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock

sys.path.insert(0, os.path.dirname(__file__))

# ── Prompts ──────────────────────────────────────────────────────────
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

K_DRAFT = 3       # Draft tokens per round (K=3 is sweet spot)
MAX_TOKENS = 50   # Total tokens to generate per prompt


def main():
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm.models.cache import make_prompt_cache

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ASYNC SPECULATIVE DECODE: ANE Draft + GPU Verify          ║")
    print("║  Draft: Qwen3-0.6B (ANE)   Target: Qwen3-8B-4bit (GPU)    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Load Draft Model (ANE) ───────────────────────────────────────
    print("Loading draft model (Qwen3-0.6B) on ANE...")
    from real_draft import RealDraftModel
    draft = RealDraftModel(model_name="Qwen/Qwen3-0.6B")
    t0 = time.time()
    ok = draft.load_and_compile(
        status_fn=lambda msg: print(f"  {msg}"),
        fused=True
    )
    if not ok:
        print("FATAL: ANE model failed to compile")
        sys.exit(1)
    print(f"  Draft loaded in {time.time()-t0:.1f}s")
    gc.collect()
    print()

    # ── Load Target Model (GPU) ──────────────────────────────────────
    print("Loading target model (Qwen3-8B-4bit) on GPU...")
    t0 = time.time()
    target_model, target_tok = mlx_load("mlx-community/Qwen3-8B-4bit")
    print(f"  Target loaded in {time.time()-t0:.1f}s")
    print()

    # ── Tokenizer check ──────────────────────────────────────────────
    test = "Hello world"
    d_ids = draft.encode(test)
    t_ids = target_tok.encode(test)
    print(f"Tokenizer check: draft={d_ids} target={t_ids} match={'✓' if d_ids == t_ids else '✗'}")
    print()

    # ── GPU helpers ──────────────────────────────────────────────────
    gpu_lock = Lock()  # MLX isn't thread-safe

    def gpu_prefill(prompt_ids):
        """Prefill GPU KV cache, return (cache, first_logits)."""
        cache = make_prompt_cache(target_model)
        x = mx.array([prompt_ids])
        logits = target_model(x, cache=cache)
        mx.eval(logits)
        return cache, logits

    def gpu_verify(cache, draft_token_ids):
        """
        Verify K draft tokens in one forward pass.
        Returns (verifier_tokens, logits_after_last, time_ms).
        verifier_tokens[i] = what GPU predicts at position i.
        """
        t0 = time.time()
        with gpu_lock:
            x = mx.array([draft_token_ids])
            logits = target_model(x, cache=cache)
            mx.eval(logits)
            # logits[0, i, :] = prediction after seeing draft[0..i]
            verifier_tokens = mx.argmax(logits[0], axis=-1).tolist()
        return verifier_tokens, logits, (time.time() - t0) * 1000

    def gpu_verify_single(cache, token_id):
        """Feed a single token to GPU, return next predicted token."""
        with gpu_lock:
            x = mx.array([[token_id]])
            logits = target_model(x, cache=cache)
            mx.eval(logits)
            return mx.argmax(logits[0, -1, :]).item(), logits

    # ── ANE helpers ──────────────────────────────────────────────────
    def ane_draft_k(start_logits, start_pos, k):
        """Draft K tokens from ANE starting from given logits/position."""
        t0 = time.time()
        tokens = []
        logits = start_logits
        pos = start_pos
        for _ in range(k):
            tok = int(np.argmax(logits))
            tokens.append(tok)
            logits = draft.forward_token(tok, pos)
            pos += 1
        return tokens, logits, pos, (time.time() - t0) * 1000

    # ── Sequential Spec Decode ───────────────────────────────────────
    def sequential_spec_decode(prompt_ids, max_tokens):
        """Standard sequential: draft K → verify → accept/reject → repeat."""
        # Prefill both models
        draft.reset_cache()
        draft_logits = None
        for i, tid in enumerate(prompt_ids):
            draft_logits = draft.forward_token(tid, i)
        draft_pos = len(prompt_ids)

        gpu_cache, gpu_logits = gpu_prefill(prompt_ids)
        # GPU's prediction for next token (from prefill)
        gpu_next = mx.argmax(gpu_logits[0, -1, :]).item()

        generated = []
        total_draft_ms = 0
        total_verify_ms = 0
        total_accepted = 0
        total_drafted = 0

        while len(generated) < max_tokens:
            # Draft K tokens from ANE
            draft_tokens, draft_logits_after, draft_pos_after, d_ms = ane_draft_k(
                draft_logits, draft_pos, min(K_DRAFT, max_tokens - len(generated)))
            total_draft_ms += d_ms
            total_drafted += len(draft_tokens)

            # Verify with GPU
            # First check: does draft[0] match GPU's prediction from last round?
            n_accepted = 0
            if draft_tokens[0] == gpu_next:
                n_accepted = 1
                # Verify remaining via batch forward
                if len(draft_tokens) > 1:
                    v_tokens, v_logits, v_ms = gpu_verify(gpu_cache, draft_tokens)
                    total_verify_ms += v_ms
                    # v_tokens[i] = GPU prediction after seeing draft[0..i]
                    # draft_tokens[i+1] should match v_tokens[i]
                    for j in range(len(draft_tokens) - 1):
                        if draft_tokens[j + 1] == v_tokens[j]:
                            n_accepted += 1
                        else:
                            break
                    # Get GPU's next prediction after accepted tokens
                    if n_accepted == len(draft_tokens):
                        gpu_next = v_tokens[-1]
                    else:
                        # Correction: use GPU's prediction at rejection point
                        gpu_next = v_tokens[n_accepted - 1]
                        # Trim GPU cache — roll back unaccepted tokens
                        for lc in gpu_cache:
                            if hasattr(lc, 'trim'):
                                lc.trim(len(draft_tokens) - n_accepted)
                else:
                    # Only 1 token drafted and it was accepted
                    _, v_logits = gpu_verify_single(gpu_cache, draft_tokens[0])
                    gpu_next = mx.argmax(v_logits[0, -1, :]).item()
                    total_verify_ms += 0  # negligible
            else:
                # First token rejected — use GPU's prediction instead
                t0v = time.time()
                # Feed GPU's token to get next prediction
                correction_tok = gpu_next
                gpu_next_new, _ = gpu_verify_single(gpu_cache, correction_tok)
                total_verify_ms += (time.time() - t0v) * 1000

                # Add the correction token
                generated.append(correction_tok)
                # Reset ANE to correction token
                draft.forward_token(correction_tok, draft_pos)
                draft_pos = draft_pos + 1
                draft_logits = draft.forward_token(correction_tok, draft_pos - 1)
                gpu_next = gpu_next_new
                continue

            # Add accepted tokens
            accepted = draft_tokens[:n_accepted]
            generated.extend(accepted)
            total_accepted += n_accepted

            # Sync ANE position
            if n_accepted < len(draft_tokens):
                # Roll back ANE — re-forward from accepted position
                # Add correction token from GPU
                correction_tok = gpu_next
                generated.append(correction_tok)

                # Reset ANE cache position
                draft_pos = len(prompt_ids) + len(generated)
                # Re-feed correction token to ANE
                draft_logits = draft.forward_token(correction_tok, draft_pos - 1)

                # Feed correction to GPU
                gpu_next_new, _ = gpu_verify_single(gpu_cache, correction_tok)
                gpu_next = gpu_next_new
            else:
                # All accepted — ANE is already at the right position
                draft_pos = draft_pos_after
                draft_logits = draft_logits_after

        return generated[:max_tokens], total_draft_ms, total_verify_ms, total_accepted, total_drafted

    # ── Async Spec Decode ────────────────────────────────────────────
    def async_spec_decode(prompt_ids, max_tokens):
        """
        Async pipeline: ANE drafts batch N+1 while GPU verifies batch N.
        """
        executor = ThreadPoolExecutor(max_workers=1)

        # Prefill both models
        draft.reset_cache()
        draft_logits = None
        for i, tid in enumerate(prompt_ids):
            draft_logits = draft.forward_token(tid, i)
        draft_pos = len(prompt_ids)

        gpu_cache, gpu_logits = gpu_prefill(prompt_ids)
        gpu_next = mx.argmax(gpu_logits[0, -1, :]).item()

        generated = []
        total_draft_ms = 0
        total_verify_ms = 0
        total_accepted = 0
        total_drafted = 0
        total_overlap_ms = 0

        # First round: draft K tokens (no overlap possible yet)
        draft_tokens, draft_logits_after, draft_pos_after, d_ms = ane_draft_k(
            draft_logits, draft_pos, min(K_DRAFT, max_tokens))
        total_draft_ms += d_ms
        total_drafted += len(draft_tokens)

        while len(generated) < max_tokens:
            round_start = time.time()

            # ── Submit GPU verification (runs in thread) ─────────
            def do_verify(tokens, cache, expected_first):
                """GPU verification in background thread."""
                t0 = time.time()
                n_acc = 0
                next_pred = expected_first

                if tokens[0] == expected_first:
                    n_acc = 1
                    if len(tokens) > 1:
                        with gpu_lock:
                            x = mx.array([tokens])
                            logits = target_model(x, cache=cache)
                            mx.eval(logits)
                            v_toks = mx.argmax(logits[0], axis=-1).tolist()

                        for j in range(len(tokens) - 1):
                            if tokens[j + 1] == v_toks[j]:
                                n_acc += 1
                            else:
                                break

                        if n_acc == len(tokens):
                            next_pred = v_toks[-1]
                        else:
                            next_pred = v_toks[n_acc - 1]
                            # Trim cache
                            for lc in cache:
                                if hasattr(lc, 'trim'):
                                    lc.trim(len(tokens) - n_acc)
                    else:
                        with gpu_lock:
                            x = mx.array([[tokens[0]]])
                            logits = target_model(x, cache=cache)
                            mx.eval(logits)
                            next_pred = mx.argmax(logits[0, -1, :]).item()

                v_ms = (time.time() - t0) * 1000
                return n_acc, next_pred, v_ms

            verify_future = executor.submit(
                do_verify, draft_tokens, gpu_cache, gpu_next)

            # ── ANE drafts NEXT batch speculatively (overlapped!) ────
            # Assume all K tokens accepted → draft from draft_pos_after
            spec_draft_start = time.time()
            remaining = max_tokens - len(generated) - len(draft_tokens)
            if remaining > 0:
                spec_tokens, spec_logits, spec_pos, spec_d_ms = ane_draft_k(
                    draft_logits_after, draft_pos_after,
                    min(K_DRAFT, remaining))
            else:
                spec_tokens, spec_logits, spec_pos, spec_d_ms = [], None, 0, 0
            spec_draft_time = (time.time() - spec_draft_start) * 1000

            # ── Wait for GPU verification ────────────────────────
            n_accepted, gpu_next, v_ms = verify_future.result()
            total_verify_ms += v_ms

            round_time = (time.time() - round_start) * 1000
            overlap = max(0, spec_draft_time + v_ms - round_time)
            total_overlap_ms += overlap if spec_tokens else 0

            # ── Process results ──────────────────────────────────
            accepted = draft_tokens[:n_accepted]
            generated.extend(accepted)
            total_accepted += n_accepted

            if n_accepted == len(draft_tokens) and spec_tokens:
                # All accepted! Speculative draft is valid.
                draft_tokens = spec_tokens
                draft_logits_after = spec_logits
                draft_pos_after = spec_pos
                draft_pos = draft_pos_after - len(spec_tokens)
                total_draft_ms += spec_d_ms
                total_drafted += len(spec_tokens)
            else:
                # Rejection or no more spec tokens — re-draft from correct position
                correction_tok = gpu_next
                generated.append(correction_tok)

                # Reset ANE position and re-draft
                new_pos = len(prompt_ids) + len(generated)
                draft_logits = draft.forward_token(correction_tok, new_pos - 1)
                draft_pos = new_pos

                # Feed correction to GPU
                gpu_next_new, _ = gpu_verify_single(gpu_cache, correction_tok)
                gpu_next = gpu_next_new

                # Draft new batch
                remaining = max_tokens - len(generated)
                if remaining > 0:
                    draft_tokens, draft_logits_after, draft_pos_after, d_ms = ane_draft_k(
                        draft_logits, draft_pos, min(K_DRAFT, remaining))
                    total_draft_ms += d_ms
                    total_drafted += len(draft_tokens)
                else:
                    break

            if len(generated) >= max_tokens:
                break

        executor.shutdown(wait=False)
        return (generated[:max_tokens], total_draft_ms, total_verify_ms,
                total_accepted, total_drafted, total_overlap_ms)

    # ── Run Benchmarks ───────────────────────────────────────────────
    print(f"Running {len(PROMPTS)} prompts × {MAX_TOKENS} tokens, K={K_DRAFT}")
    print()

    # === BASELINE: 8B alone ===
    print("═" * 70)
    print("BASELINE: 8B GPU alone (no spec decode)")
    print("═" * 70)
    baseline_total_tokens = 0
    baseline_total_ms = 0

    for i, prompt in enumerate(PROMPTS):
        prompt_ids = draft.encode(prompt)
        cache = make_prompt_cache(target_model)
        x = mx.array([prompt_ids])
        logits = target_model(x, cache=cache)
        mx.eval(logits)

        t0 = time.time()
        tokens = []
        for _ in range(MAX_TOKENS):
            tok = mx.argmax(logits[0, -1, :]).item()
            tokens.append(tok)
            x = mx.array([[tok]])
            logits = target_model(x, cache=cache)
            mx.eval(logits)
        elapsed_ms = (time.time() - t0) * 1000

        baseline_total_tokens += MAX_TOKENS
        baseline_total_ms += elapsed_ms
        tok_s = MAX_TOKENS / (elapsed_ms / 1000)
        text = draft.decode(tokens)[:60]
        print(f"  [{i+1:2d}] {tok_s:5.1f} tok/s  {text}")

    baseline_tok_s = baseline_total_tokens / (baseline_total_ms / 1000)
    print(f"\n  BASELINE: {baseline_tok_s:.1f} tok/s\n")

    # === SEQUENTIAL SPEC DECODE ===
    print("═" * 70)
    print(f"SEQUENTIAL SPEC DECODE: K={K_DRAFT} (ANE draft → GPU verify)")
    print("═" * 70)
    seq_total_tokens = 0
    seq_total_ms = 0
    seq_total_accepted = 0
    seq_total_drafted = 0

    for i, prompt in enumerate(PROMPTS):
        prompt_ids = draft.encode(prompt)
        t0 = time.time()
        tokens, d_ms, v_ms, n_acc, n_draft = sequential_spec_decode(prompt_ids, MAX_TOKENS)
        elapsed_ms = (time.time() - t0) * 1000

        seq_total_tokens += len(tokens)
        seq_total_ms += elapsed_ms
        seq_total_accepted += n_acc
        seq_total_drafted += n_draft
        tok_s = len(tokens) / (elapsed_ms / 1000)
        accept_pct = (n_acc / n_draft * 100) if n_draft > 0 else 0
        text = draft.decode(tokens)[:60]
        print(f"  [{i+1:2d}] {tok_s:5.1f} tok/s  acc={accept_pct:4.0f}%  {text}")

    seq_tok_s = seq_total_tokens / (seq_total_ms / 1000)
    seq_accept = seq_total_accepted / seq_total_drafted * 100 if seq_total_drafted > 0 else 0
    seq_speedup = seq_tok_s / baseline_tok_s
    print(f"\n  SEQUENTIAL: {seq_tok_s:.1f} tok/s ({seq_speedup:.2f}x)  "
          f"acceptance: {seq_accept:.0f}%\n")

    # === ASYNC SPEC DECODE ===
    print("═" * 70)
    print(f"ASYNC SPEC DECODE: K={K_DRAFT} (ANE draft || GPU verify)")
    print("═" * 70)
    async_total_tokens = 0
    async_total_ms = 0
    async_total_accepted = 0
    async_total_drafted = 0
    async_total_overlap = 0

    for i, prompt in enumerate(PROMPTS):
        prompt_ids = draft.encode(prompt)
        t0 = time.time()
        tokens, d_ms, v_ms, n_acc, n_draft, overlap_ms = async_spec_decode(
            prompt_ids, MAX_TOKENS)
        elapsed_ms = (time.time() - t0) * 1000

        async_total_tokens += len(tokens)
        async_total_ms += elapsed_ms
        async_total_accepted += n_acc
        async_total_drafted += n_draft
        async_total_overlap += overlap_ms
        tok_s = len(tokens) / (elapsed_ms / 1000)
        accept_pct = (n_acc / n_draft * 100) if n_draft > 0 else 0
        text = draft.decode(tokens)[:60]
        print(f"  [{i+1:2d}] {tok_s:5.1f} tok/s  acc={accept_pct:4.0f}%  {text}")

    async_tok_s = async_total_tokens / (async_total_ms / 1000)
    async_accept = async_total_accepted / async_total_drafted * 100 if async_total_drafted > 0 else 0
    async_speedup = async_tok_s / baseline_tok_s
    async_vs_seq = async_tok_s / seq_tok_s if seq_tok_s > 0 else 0
    print(f"\n  ASYNC: {async_tok_s:.1f} tok/s ({async_speedup:.2f}x vs baseline, "
          f"{async_vs_seq:.2f}x vs sequential)  "
          f"acceptance: {async_accept:.0f}%")
    print(f"  Total overlap saved: {async_total_overlap:.0f}ms")

    # === SUMMARY ===
    print()
    print("═" * 70)
    print("SUMMARY")
    print("═" * 70)
    print(f"  Baseline (8B GPU alone):   {baseline_tok_s:6.1f} tok/s  (1.00x)")
    print(f"  Sequential (ANE→GPU):      {seq_tok_s:6.1f} tok/s  ({seq_speedup:.2f}x)")
    print(f"  Async (ANE || GPU):        {async_tok_s:6.1f} tok/s  ({async_speedup:.2f}x)")
    print(f"  Async vs Sequential:       {async_vs_seq:.2f}x additional speedup")
    print()
    print(f"  Acceptance rate:           {async_accept:.0f}%")
    print(f"  K (draft tokens/round):    {K_DRAFT}")
    print(f"  Memory:                    ~5 GB (0.6B fp32 + 8B 4-bit)")
    print()
    if async_speedup > 1.3:
        print("🚀 Async pipeline delivers meaningful speedup!")
    elif async_speedup > 1.1:
        print("✅ Modest improvement — acceptance rate is the bottleneck")
    else:
        print("⚠️  No improvement — draft model too slow or acceptance too low")


if __name__ == "__main__":
    main()
