"""
Test the C forward pass vs Python forward pass.
Loads Qwen3-0.6B, compiles ANE kernels, then runs forward_token in C.
Compares output and timing against the Python implementation.
"""

import numpy as np
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speculative"))
from real_draft import RealDraftModel, ROPE_THETA, ANE_SPATIAL, MAX_SEQ, CLS_CHUNK

# ── Load the bridge with forward pass ────────────────────────────────

BRIDGE_PATH = os.path.join(os.path.dirname(__file__), "bridge", "libane_bridge.dylib")
lib = ctypes.CDLL(BRIDGE_PATH)

# Forward pass C API
class FPModelConfig(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_int),
        ("n_heads", ctypes.c_int),
        ("n_kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("hidden_dim", ctypes.c_int),
        ("vocab_size", ctypes.c_int),
        ("n_layers", ctypes.c_int),
        ("max_seq", ctypes.c_int),
        ("ane_spatial", ctypes.c_int),
        ("rope_theta", ctypes.c_float),
    ]

class ForwardTiming(ctypes.Structure):
    _fields_ = [
        ("total_ms", ctypes.c_double),
        ("ane_ms", ctypes.c_double),
        ("ane_pack_ms", ctypes.c_double),
        ("ane_eval_ms", ctypes.c_double),
        ("ane_read_ms", ctypes.c_double),
        ("rmsnorm_ms", ctypes.c_double),
        ("rope_ms", ctypes.c_double),
        ("qknorm_ms", ctypes.c_double),
        ("attention_ms", ctypes.c_double),
        ("silu_ms", ctypes.c_double),
        ("embed_ms", ctypes.c_double),
        ("classify_ms", ctypes.c_double),
    ]

# Set up function signatures
lib.forward_model_create.argtypes = [ctypes.POINTER(FPModelConfig)]
lib.forward_model_create.restype = ctypes.c_void_p

lib.forward_model_free.argtypes = [ctypes.c_void_p]

lib.forward_model_set_embed.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.forward_model_set_final_norm.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

lib.forward_model_set_layer_weights.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
]

lib.forward_model_set_layer_kernels.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

lib.forward_model_set_layer_kernels_fused.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

lib.forward_model_add_cls_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

lib.forward_model_reset_cache.argtypes = [ctypes.c_void_p]

lib.forward_model_forward_token.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.forward_model_forward_token.restype = ctypes.POINTER(ctypes.c_float)

lib.forward_model_get_timing.argtypes = [ctypes.c_void_p, ctypes.POINTER(ForwardTiming)]


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  C FORWARD PASS TEST (FUSED KERNELS)                        ║")
    print("║  Python setup → C hot loop → compare with Python reference  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Step 1: Load model via Python (reuse existing infrastructure) ──
    print("\n▸ Loading model via Python...")
    py_model = RealDraftModel()
    py_model.load_and_compile(lambda msg: print(f"  {msg}"), fused=True)

    if not py_model.compiled:
        print("✗ Model compilation failed!")
        return

    # ── Step 2: Create C forward model ──
    print("\n▸ Creating C forward model...")
    config = FPModelConfig(
        dim=py_model.dim,
        n_heads=py_model.n_heads,
        n_kv_heads=py_model.n_kv_heads,
        head_dim=py_model.head_dim,
        hidden_dim=py_model.hidden_dim,
        vocab_size=py_model.vocab_size,
        n_layers=py_model.n_layers,
        max_seq=MAX_SEQ,
        ane_spatial=ANE_SPATIAL,
        rope_theta=ROPE_THETA,
    )

    c_model = lib.forward_model_create(ctypes.byref(config))
    if not c_model:
        print("✗ forward_model_create failed!")
        return

    print(f"  Config: dim={config.dim}, heads={config.n_heads}, "
          f"kv_heads={config.n_kv_heads}, head_dim={config.head_dim}, "
          f"layers={config.n_layers}")

    # ── Step 3: Transfer weights to C ──
    print("\n▸ Transferring weights to C...")

    # Embedding
    embed_flat = py_model.embed_w.astype(np.float32).ravel()
    lib.forward_model_set_embed(c_model,
        embed_flat.ctypes.data_as(ctypes.c_void_p))

    # Final norm
    fn = py_model.final_norm_w.astype(np.float32)
    lib.forward_model_set_final_norm(c_model,
        fn.ctypes.data_as(ctypes.c_void_p))

    # Per-layer weights and kernels
    for l in range(py_model.n_layers):
        lw = py_model.layer_weights[l]
        lk = py_model.kernels[l]

        an = lw['attn_norm'].astype(np.float32)
        ffn = lw['ffn_norm'].astype(np.float32)
        qn = lw['q_norm'].astype(np.float32)
        kn = lw['k_norm'].astype(np.float32)

        lib.forward_model_set_layer_weights(c_model, l,
            an.ctypes.data_as(ctypes.c_void_p),
            ffn.ctypes.data_as(ctypes.c_void_p),
            qn.ctypes.data_as(ctypes.c_void_p),
            kn.ctypes.data_as(ctypes.c_void_p))

        if py_model.fused:
            lib.forward_model_set_layer_kernels_fused(c_model, l,
                lk['qkv'], lk['o'], lk['gate_up'], lk['down'])
        else:
            lib.forward_model_set_layer_kernels(c_model, l,
                lk['q'], lk['k'], lk['v'], lk['o'],
                lk['gate'], lk['up'], lk['down'])

    # Classifier chunks
    for kernel, out_ch in py_model.cls_kernels:
        lib.forward_model_add_cls_kernel(c_model, kernel, out_ch)

    kernels_per_layer = 4 if py_model.fused else 7
    total_dispatches = py_model.n_layers * kernels_per_layer + len(py_model.cls_kernels)
    print(f"  ✓ Transferred {py_model.n_layers} layers ({kernels_per_layer} kernels/layer, {'fused' if py_model.fused else 'unfused'}) + {len(py_model.cls_kernels)} classifier chunks")
    print(f"  ✓ Total ANE dispatches per token: {total_dispatches}")

    # ── Step 4: Test prompt ──
    prompt = "The capital of France is"
    prompt_ids = py_model.encode(prompt)
    print(f"\n▸ Prompt: \"{prompt}\"")
    print(f"  Token IDs: {prompt_ids}")

    # ── Step 5: Run Python forward pass ──
    print("\n▸ Running Python forward pass...")
    py_model.reset_cache()
    t0 = time.time()
    for i, tid in enumerate(prompt_ids[:-1]):
        py_model.forward_token(tid, i)
    py_logits = py_model.forward_token(prompt_ids[-1], len(prompt_ids) - 1)
    py_time = (time.time() - t0) * 1000
    py_top = int(np.argmax(py_logits))
    print(f"  Python: {py_time:.1f}ms total, top token: {py_top} = \"{py_model.decode([py_top])}\"")

    # ── Step 6: Run C forward pass ──
    print("\n▸ Running C forward pass...")
    lib.forward_model_reset_cache(c_model)
    t0 = time.time()
    for i, tid in enumerate(prompt_ids[:-1]):
        lib.forward_model_forward_token(c_model, int(tid), i)
    c_logits_ptr = lib.forward_model_forward_token(c_model, int(prompt_ids[-1]), len(prompt_ids) - 1)
    c_time = (time.time() - t0) * 1000
    c_logits = np.ctypeslib.as_array(c_logits_ptr, shape=(py_model.vocab_size,)).copy()
    c_top = int(np.argmax(c_logits))
    print(f"  C:      {c_time:.1f}ms total, top token: {c_top} = \"{py_model.decode([c_top])}\"")

    # ── Step 7: Get C timing breakdown ──
    timing = ForwardTiming()
    lib.forward_model_get_timing(c_model, ctypes.byref(timing))
    print(f"\n  C Forward Timing (last token):")
    print(f"    Total:       {timing.total_ms:7.2f}ms")
    print(f"    ANE total:   {timing.ane_ms:7.2f}ms  ({timing.ane_ms/timing.total_ms*100:.0f}%)")
    print(f"      ├ pack:    {timing.ane_pack_ms:7.2f}ms  (memset + strided write)")
    print(f"      ├ eval:    {timing.ane_eval_ms:7.2f}ms  (ANE dispatch)")
    print(f"      └ read:    {timing.ane_read_ms:7.2f}ms  (strided extract)")
    print(f"    RMSNorm:     {timing.rmsnorm_ms:7.2f}ms")
    print(f"    RoPE:        {timing.rope_ms:7.2f}ms")
    print(f"    QK-Norm:     {timing.qknorm_ms:7.2f}ms")
    print(f"    Attention:   {timing.attention_ms:7.2f}ms")
    print(f"    SiLU:        {timing.silu_ms:7.2f}ms")
    print(f"    Embed:       {timing.embed_ms:7.2f}ms")
    print(f"    Classify:    {timing.classify_ms:7.2f}ms")

    # ── Step 8: Compare outputs ──
    print(f"\n▸ Comparison:")
    err = np.max(np.abs(c_logits - py_logits))
    rel_err = err / (np.max(np.abs(py_logits)) + 1e-10)
    match = c_top == py_top
    print(f"  Max absolute error: {err:.6f}")
    print(f"  Max relative error: {rel_err:.6f}")
    print(f"  Top token match:    {'✓ YES' if match else '✗ NO'}")
    print(f"  Speedup:            {py_time/c_time:.2f}x")

    # ── Step 9: Benchmark C forward pass ──
    print(f"\n▸ Benchmarking C forward pass (single token, pos=5)...")
    lib.forward_model_reset_cache(c_model)
    # Quick prefill
    for i, tid in enumerate(prompt_ids):
        lib.forward_model_forward_token(c_model, int(tid), i)

    n_bench = 20
    times = []
    for i in range(n_bench):
        t0 = time.time()
        lib.forward_model_forward_token(c_model, int(prompt_ids[-1]), len(prompt_ids) + i)
        times.append((time.time() - t0) * 1000)

    avg = np.mean(times)
    std = np.std(times)
    best = np.min(times)
    print(f"  {n_bench} iterations: avg={avg:.1f}ms, std={std:.1f}ms, best={best:.1f}ms")
    print(f"  → {1000/avg:.1f} tok/s (avg), {1000/best:.1f} tok/s (peak)")

    timing = ForwardTiming()
    lib.forward_model_get_timing(c_model, ctypes.byref(timing))
    print(f"\n  Last token breakdown:")
    print(f"    ANE total:       {timing.ane_ms:.2f}ms")
    print(f"      ├ pack I/O:    {timing.ane_pack_ms:.2f}ms  ({timing.ane_pack_ms/timing.total_ms*100:.1f}%)")
    print(f"      ├ eval:        {timing.ane_eval_ms:.2f}ms  ({timing.ane_eval_ms/timing.total_ms*100:.1f}%)")
    print(f"      └ read I/O:    {timing.ane_read_ms:.2f}ms  ({timing.ane_read_ms/timing.total_ms*100:.1f}%)")
    print(f"    CPU (all):       {timing.rmsnorm_ms + timing.rope_ms + timing.qknorm_ms + timing.attention_ms + timing.silu_ms:.2f}ms")
    print(f"    Classifier:      {timing.classify_ms:.2f}ms")

    # ── Step 10: Generation test ──
    print(f"\n▸ Generation test (C forward pass)...")
    lib.forward_model_reset_cache(c_model)

    # Prefill
    for i, tid in enumerate(prompt_ids):
        c_logits_ptr = lib.forward_model_forward_token(c_model, int(tid), i)

    # Generate
    pos = len(prompt_ids)
    generated = []
    t_gen_start = time.time()
    for _ in range(20):
        c_logits = np.ctypeslib.as_array(c_logits_ptr, shape=(py_model.vocab_size,)).copy()
        tok = int(np.argmax(c_logits))
        generated.append(tok)
        c_logits_ptr = lib.forward_model_forward_token(c_model, tok, pos)
        pos += 1
    t_gen = (time.time() - t_gen_start) * 1000

    text = py_model.decode(generated)
    print(f"  Generated: \"{prompt}{text}\"")
    print(f"  {len(generated)} tokens in {t_gen:.0f}ms = {len(generated)/t_gen*1000:.1f} tok/s")

    # Cleanup
    lib.forward_model_free(c_model)
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
