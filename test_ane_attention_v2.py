"""
ANE Attention v2 — Split approach based on proven ops.

Findings from v1:
- mul + reduce_sum WORKS (dot product)
- softmax WORKS
- 3 inputs MIGHT fail → test with 2-kernel split
- reduce_sum over spatial → output < SPATIAL might fail → keep output padded

Strategy:
  Kernel A: scores(Q, K) = softmax(sum(Q * K, axis=channels) * scale)
  Kernel B: weighted_sum(scores, V) = scores * V  (no reduce — read out, sum on CPU)
  OR
  Kernel C: Full fused with output kept at spatial=SPATIAL (pad output)
"""

import numpy as np
import ctypes
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speculative"))
from ane_draft import init_ane, lib, compile_ane_kernel

SPATIAL = 16


def compile_mil_program(prog, model_name, in_sizes, out_sizes):
    import coremltools as ct
    model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15)
    pkg_path = f"/tmp/{model_name}.mlpackage"
    model.save(pkg_path)
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg_path, "/tmp/"],
                   capture_output=True)
    compiled = f"/tmp/{model_name}.mlmodelc"
    with open(f"{compiled}/model.mil", 'rb') as f:
        mil_bytes = f.read()
    weight_path = f"{compiled}/weights/weight.bin"
    weight_blob = None
    if os.path.exists(weight_path):
        with open(weight_path, 'rb') as f:
            weight_blob = f.read()
    return compile_ane_kernel(mil_bytes, weight_blob, in_sizes, out_sizes)


def test_scores_kernel():
    """Kernel A: Q,K → attention scores with softmax.
    Input: Q[1, head_dim, 1, SPATIAL], K[1, head_dim, 1, SPATIAL]
    Output: scores[1, 1, 1, SPATIAL] (softmax over spatial = seq positions)
    """
    print("\n═══ KERNEL A: Scores — Q×K → softmax ═══")
    from coremltools.converters.mil.mil import Builder as mb

    head_dim = 128
    scale = np.float32(1.0 / np.sqrt(head_dim))

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),
    ])
    def prog(q, k):
        qk = mb.mul(x=q, y=k, name="qk_mul")
        raw = mb.reduce_sum(x=qk, axes=[1], keep_dims=True, name="raw_scores")
        sc = mb.const(val=scale, name="scale")
        scaled = mb.mul(x=raw, y=sc, name="scaled")
        return mb.softmax(x=scaled, axis=3, name="attn_weights")

    in_size = head_dim * SPATIAL * 4
    out_size = 1 * SPATIAL * 4  # [1, 1, 1, SPATIAL]

    kernel = compile_mil_program(prog, "scores_kernel", [in_size, in_size], [out_size])
    if not kernel:
        print("  ✗ Compile failed")
        return None, False

    print("  ✓ Compiled! Testing...")
    q_vec = np.random.randn(head_dim).astype(np.float32)
    k_cache = np.random.randn(head_dim, SPATIAL).astype(np.float32)

    q_buf = np.zeros((head_dim, SPATIAL), dtype=np.float32)
    for i in range(SPATIAL):
        q_buf[:, i] = q_vec

    q_flat = q_buf.ravel()
    k_flat = k_cache.ravel()

    lib.ane_bridge_write_input(kernel, 0,
        q_flat.ctypes.data_as(ctypes.c_void_p), q_flat.nbytes)
    lib.ane_bridge_write_input(kernel, 1,
        k_flat.ctypes.data_as(ctypes.c_void_p), k_flat.nbytes)

    success = lib.ane_bridge_eval(kernel)
    if not success:
        print("  ✗ eval() returned False")
        return kernel, False

    out = np.zeros(SPATIAL, dtype=np.float32)
    lib.ane_bridge_read_output(kernel, 0,
        out.ctypes.data_as(ctypes.c_void_p), out.nbytes)

    # CPU reference
    scores = (q_vec @ k_cache) * scale
    expected = np.exp(scores - scores.max())
    expected = expected / expected.sum()

    err = np.max(np.abs(out - expected))
    print(f"  ✓ EXECUTED! Max error: {err:.6f}")
    print(f"  Sum (should be ~1.0): {out.sum():.6f}")
    print(f"  Expected[:5]: {expected[:5]}")
    print(f"  Got[:5]:      {out[:5]}")
    return kernel, True


def test_weighted_v_kernel():
    """Kernel B: scores × V → weighted values (no reduce, sum on CPU).
    Input: scores[1, 1, 1, SPATIAL], V[1, head_dim, 1, SPATIAL]
    Output: weighted[1, head_dim, 1, SPATIAL] (broadcast multiply)
    """
    print("\n═══ KERNEL B: Weighted V — scores × V ═══")
    from coremltools.converters.mil.mil import Builder as mb

    head_dim = 128

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, 1, 1, SPATIAL)),           # scores (softmax output)
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),    # V_cache
    ])
    def prog(scores, v):
        # Broadcast: [1,1,1,S] * [1,hd,1,S] = [1,hd,1,S]
        return mb.mul(x=scores, y=v, name="weighted_v")

    scores_size = 1 * SPATIAL * 4
    v_size = head_dim * SPATIAL * 4
    out_size = head_dim * SPATIAL * 4

    kernel = compile_mil_program(prog, "weightedv_kernel",
                                 [scores_size, v_size], [out_size])
    if not kernel:
        print("  ✗ Compile failed")
        return None, False

    print("  ✓ Compiled! Testing...")
    scores = np.random.rand(SPATIAL).astype(np.float32)
    scores = scores / scores.sum()  # Fake softmax output
    v_cache = np.random.randn(head_dim, SPATIAL).astype(np.float32)

    s_buf = scores.reshape(1, 1, 1, SPATIAL).ravel()
    v_flat = v_cache.ravel()

    lib.ane_bridge_write_input(kernel, 0,
        s_buf.ctypes.data_as(ctypes.c_void_p), s_buf.nbytes)
    lib.ane_bridge_write_input(kernel, 1,
        v_flat.ctypes.data_as(ctypes.c_void_p), v_flat.nbytes)

    success = lib.ane_bridge_eval(kernel)
    if not success:
        print("  ✗ eval() returned False")
        return kernel, False

    out = np.zeros(head_dim * SPATIAL, dtype=np.float32)
    lib.ane_bridge_read_output(kernel, 0,
        out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
    out = out.reshape(head_dim, SPATIAL)

    # CPU reference
    expected = scores[np.newaxis, :] * v_cache  # [head_dim, SPATIAL]

    err = np.max(np.abs(out - expected))
    print(f"  ✓ EXECUTED! Max error: {err:.6f}")
    print(f"  Expected[0,:5]: {expected[0,:5]}")
    print(f"  Got[0,:5]:      {out[0,:5]}")

    # Final step: sum over spatial on CPU
    attn_out = out.sum(axis=1)  # [head_dim]
    expected_out = expected.sum(axis=1)
    final_err = np.max(np.abs(attn_out - expected_out))
    print(f"  CPU sum step error: {final_err:.6f}")
    return kernel, True


def test_3input_kernel():
    """Test if 3 inputs work at all (maybe the fused failure was output shape)."""
    print("\n═══ TEST: 3-input kernel (simple) ═══")
    from coremltools.converters.mil.mil import Builder as mb

    ch = 64

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, ch, 1, SPATIAL)),
        mb.TensorSpec(shape=(1, ch, 1, SPATIAL)),
        mb.TensorSpec(shape=(1, ch, 1, SPATIAL)),
    ])
    def prog(a, b, c):
        ab = mb.mul(x=a, y=b, name="ab")
        return mb.add(x=ab, y=c, name="out")

    sz = ch * SPATIAL * 4
    kernel = compile_mil_program(prog, "test_3in", [sz, sz, sz], [sz])
    if not kernel:
        print("  ✗ Compile failed")
        return False

    a = np.random.randn(ch, SPATIAL).astype(np.float32)
    b = np.random.randn(ch, SPATIAL).astype(np.float32)
    c = np.random.randn(ch, SPATIAL).astype(np.float32)

    for i, arr in enumerate([a, b, c]):
        buf = arr.ravel()
        lib.ane_bridge_write_input(kernel, i,
            buf.ctypes.data_as(ctypes.c_void_p), buf.nbytes)

    success = lib.ane_bridge_eval(kernel)
    if not success:
        print("  ✗ eval() returned False")
        return False

    out = np.zeros(ch * SPATIAL, dtype=np.float32)
    lib.ane_bridge_read_output(kernel, 0,
        out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
    out = out.reshape(ch, SPATIAL)

    expected = a * b + c
    err = np.max(np.abs(out - expected))
    print(f"  ✓ EXECUTED! 3 inputs work. Max error: {err:.6f}")
    return True


def test_fused_attn_fixed_output():
    """Fused attention with output kept at SPATIAL (no reduce over spatial)."""
    print("\n═══ FUSED ATTENTION v2 — output at SPATIAL ═══")
    from coremltools.converters.mil.mil import Builder as mb

    head_dim = 128
    scale = np.float32(1.0 / np.sqrt(head_dim))

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),   # Q (broadcast)
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),   # K_cache
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),   # V_cache
    ])
    def prog(q, k, v):
        # Scores
        qk = mb.mul(x=q, y=k, name="qk_mul")
        raw = mb.reduce_sum(x=qk, axes=[1], keep_dims=True, name="raw_scores")
        sc = mb.const(val=scale, name="scale")
        scaled = mb.mul(x=raw, y=sc, name="scaled")
        attn_w = mb.softmax(x=scaled, axis=3, name="attn_weights")

        # Weighted V — keep at [1, head_dim, 1, SPATIAL], sum on CPU later
        weighted = mb.mul(x=attn_w, y=v, name="weighted_v")
        return weighted

    sz = head_dim * SPATIAL * 4

    kernel = compile_mil_program(prog, "fused_attn_v2", [sz, sz, sz], [sz])
    if not kernel:
        print("  ✗ Compile failed")
        return False

    q_vec = np.random.randn(head_dim).astype(np.float32)
    k_cache = np.random.randn(head_dim, SPATIAL).astype(np.float32)
    v_cache = np.random.randn(head_dim, SPATIAL).astype(np.float32)

    q_buf = np.zeros((head_dim, SPATIAL), dtype=np.float32)
    for i in range(SPATIAL):
        q_buf[:, i] = q_vec

    for i, arr in enumerate([q_buf, k_cache, v_cache]):
        buf = arr.ravel()
        lib.ane_bridge_write_input(kernel, i,
            buf.ctypes.data_as(ctypes.c_void_p), buf.nbytes)

    success = lib.ane_bridge_eval(kernel)
    if not success:
        print("  ✗ eval() returned False")
        return False

    out = np.zeros(head_dim * SPATIAL, dtype=np.float32)
    lib.ane_bridge_read_output(kernel, 0,
        out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
    out = out.reshape(head_dim, SPATIAL)

    # Sum over spatial on CPU → final attention output
    attn_out = out.sum(axis=1)  # [head_dim]

    # CPU reference
    scores = (q_vec @ k_cache) * scale
    scores_sm = np.exp(scores - scores.max())
    scores_sm = scores_sm / scores_sm.sum()
    expected = v_cache @ scores_sm  # [head_dim]

    err = np.max(np.abs(attn_out - expected))
    rel_err = err / (np.max(np.abs(expected)) + 1e-8)
    print(f"  ✓ EXECUTED! Max abs error: {err:.6f}, relative: {rel_err:.6f}")
    print(f"  Expected[:5]: {expected[:5]}")
    print(f"  Got[:5]:      {attn_out[:5]}")

    if rel_err < 0.05:
        print("  ★★★ FUSED ATTENTION ON ANE WORKS! ★★★")
        return True
    else:
        print(f"  ⚠ High error — may need debugging")
        return False


def benchmark_attention(n_iters=100):
    """Benchmark: ANE fused attention vs CPU numpy attention."""
    print("\n═══ BENCHMARK: ANE vs CPU attention ═══")
    from coremltools.converters.mil.mil import Builder as mb

    head_dim = 128
    scale = np.float32(1.0 / np.sqrt(head_dim))

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),
    ])
    def prog(q, k, v):
        qk = mb.mul(x=q, y=k, name="qk_mul")
        raw = mb.reduce_sum(x=qk, axes=[1], keep_dims=True, name="raw_scores")
        sc = mb.const(val=scale, name="scale")
        scaled = mb.mul(x=raw, y=sc, name="scaled")
        attn_w = mb.softmax(x=scaled, axis=3, name="attn_weights")
        weighted = mb.mul(x=attn_w, y=v, name="weighted_v")
        return weighted

    sz = head_dim * SPATIAL * 4
    kernel = compile_mil_program(prog, "bench_attn", [sz, sz, sz], [sz])
    if not kernel:
        print("  Compile failed, skipping benchmark")
        return

    q_vec = np.random.randn(head_dim).astype(np.float32)
    k_cache = np.random.randn(head_dim, SPATIAL).astype(np.float32)
    v_cache = np.random.randn(head_dim, SPATIAL).astype(np.float32)
    q_buf = np.zeros((head_dim, SPATIAL), dtype=np.float32)
    for i in range(SPATIAL):
        q_buf[:, i] = q_vec

    out = np.zeros(head_dim * SPATIAL, dtype=np.float32)

    # Warmup
    for _ in range(10):
        for i, arr in enumerate([q_buf, k_cache, v_cache]):
            buf = arr.ravel()
            lib.ane_bridge_write_input(kernel, i,
                buf.ctypes.data_as(ctypes.c_void_p), buf.nbytes)
        lib.ane_bridge_eval(kernel)
        lib.ane_bridge_read_output(kernel, 0,
            out.ctypes.data_as(ctypes.c_void_p), out.nbytes)

    # Benchmark ANE
    t0 = time.time()
    for _ in range(n_iters):
        for i, arr in enumerate([q_buf, k_cache, v_cache]):
            buf = arr.ravel()
            lib.ane_bridge_write_input(kernel, i,
                buf.ctypes.data_as(ctypes.c_void_p), buf.nbytes)
        lib.ane_bridge_eval(kernel)
        lib.ane_bridge_read_output(kernel, 0,
            out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
        # CPU sum
        result = out.reshape(head_dim, SPATIAL).sum(axis=1)
    ane_time = (time.time() - t0) * 1000

    # Benchmark CPU (numpy)
    t0 = time.time()
    for _ in range(n_iters):
        scores = (q_vec @ k_cache) * scale
        scores_sm = np.exp(scores - scores.max())
        scores_sm = scores_sm / scores_sm.sum()
        result_cpu = v_cache @ scores_sm
    cpu_time = (time.time() - t0) * 1000

    print(f"  ANE attention: {ane_time:.1f}ms for {n_iters} iters = {ane_time/n_iters:.3f}ms/iter")
    print(f"  CPU attention: {cpu_time:.1f}ms for {n_iters} iters = {cpu_time/n_iters:.3f}ms/iter")
    speedup = cpu_time / ane_time if ane_time > 0 else 0
    print(f"  Speedup: {speedup:.2f}x {'(ANE faster)' if speedup > 1 else '(CPU faster)'}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ANE ATTENTION v2 — Split Kernel Approach            ║")
    print("╚══════════════════════════════════════════════════════╝")

    if not init_ane():
        print("ANE init failed!")
        sys.exit(1)

    r = {}
    r["3_inputs"] = test_3input_kernel()
    _, r["scores_kernel"] = test_scores_kernel()
    _, r["weighted_v"] = test_weighted_v_kernel()
    r["fused_attn_v2"] = test_fused_attn_fixed_output()

    print("\n" + "═" * 56)
    print("RESULTS:")
    for name, passed in r.items():
        print(f"  {'✓' if passed else '✗'}  {name}")

    if r.get("fused_attn_v2"):
        print("\n★ FUSED ATTENTION WORKS! Running benchmark...")
        benchmark_attention()
