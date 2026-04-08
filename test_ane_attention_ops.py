"""
Test which MIL ops ANE can execute for attention.
Goal: Find a path to ANE-native attention.

Tests:
1. matmul between two dynamic inputs (Q × K^T)
2. reduce_max, reduce_sum (for softmax)
3. softmax op directly
4. Multi-input kernel (2 data tensors)
5. Full fused attention kernel
"""

import numpy as np
import ctypes
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speculative"))
from ane_draft import init_ane, lib, compile_ane_kernel

SPATIAL = 16  # ANE minimum


def compile_mil_program(prog, model_name, in_sizes, out_sizes):
    """Compile a coremltools MIL program and load onto ANE."""
    import coremltools as ct

    model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15)
    pkg_path = f"/tmp/{model_name}.mlpackage"
    model.save(pkg_path)
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg_path, "/tmp/"],
                   capture_output=True)
    compiled = f"/tmp/{model_name}.mlmodelc"
    with open(f"{compiled}/model.mil", 'rb') as f:
        mil_bytes = f.read()

    # Collect all weight files
    weight_dir = f"{compiled}/weights"
    weight_blob = None
    if os.path.exists(f"{weight_dir}/weight.bin"):
        with open(f"{weight_dir}/weight.bin", 'rb') as f:
            weight_blob = f.read()

    kernel = compile_ane_kernel(mil_bytes, weight_blob, in_sizes, out_sizes)
    return kernel


def test_matmul_two_inputs():
    """Test: Can ANE do matmul between two dynamic input tensors?"""
    print("\n═══ TEST 1: matmul(A, B) — two dynamic inputs ═══")
    from coremltools.converters.mil.mil import Builder as mb

    M, K, N = 1, 128, 64  # Q[1,128] × K^T[128,64] → scores[1,64]

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, M, 1, K)),   # Q: [1, 1, 1, 128]
        mb.TensorSpec(shape=(1, K, 1, N)),   # K: [1, 128, 1, 64]
    ])
    def prog(q, k):
        # Reshape to 2D for matmul
        q_2d = mb.reshape(x=q, shape=[M, K], name="q_flat")
        k_2d = mb.reshape(x=k, shape=[K, N], name="k_flat")
        out = mb.matmul(x=q_2d, y=k_2d, name="qk")
        return mb.reshape(x=out, shape=[1, M, 1, N], name="scores")

    in_sizes = [M * K * SPATIAL * 4, K * N * SPATIAL * 4]
    # Actually let's use the raw shapes without spatial padding for matmul
    # ANE needs NCHW with spatial=16, but matmul might work differently
    # Let's try with the actual tensor sizes first
    in_sizes_raw = [1 * M * 1 * K * 4, 1 * K * 1 * N * 4]
    out_sizes_raw = [1 * M * 1 * N * 4]

    try:
        kernel = compile_mil_program(prog, "test_matmul_2in", in_sizes_raw, out_sizes_raw)
        if kernel:
            print("  ✓ Compiled! Testing execution...")
            q = np.random.randn(M, K).astype(np.float32)
            k = np.random.randn(K, N).astype(np.float32)
            expected = q @ k

            q_buf = q.ravel().astype(np.float32)
            k_buf = k.ravel().astype(np.float32)
            lib.ane_bridge_write_input(kernel, 0,
                q_buf.ctypes.data_as(ctypes.c_void_p), q_buf.nbytes)
            lib.ane_bridge_write_input(kernel, 1,
                k_buf.ctypes.data_as(ctypes.c_void_p), k_buf.nbytes)

            success = lib.ane_bridge_eval(kernel)
            if success:
                out = np.zeros(M * N, dtype=np.float32)
                lib.ane_bridge_read_output(kernel, 0,
                    out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
                out = out.reshape(M, N)
                err = np.max(np.abs(out - expected))
                print(f"  ✓ EXECUTED! Max error: {err:.6f}")
                print(f"  Expected[0,:5]: {expected[0,:5]}")
                print(f"  Got[0,:5]:      {out[0,:5]}")
                return True
            else:
                print("  ✗ eval() returned False")
        else:
            print("  ✗ Compile failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    return False


def test_matmul_nchw():
    """Test: matmul with NCHW layout (spatial=16)."""
    print("\n═══ TEST 2: matmul in NCHW layout (spatial=16) ═══")
    from coremltools.converters.mil.mil import Builder as mb

    # For single-head attention: Q[1,head_dim] × K_cache[seq_len, head_dim]^T
    # In NCHW: Q as [1, head_dim, 1, 16], K as [1, head_dim, 1, 16]
    # But we need different shapes...
    #
    # Alternative: encode as conv1x1 with K as "weight"?
    # No — K is dynamic. Let's try matmul with proper shapes.

    head_dim = 128
    seq_len = 16  # Use spatial dim as seq_len!

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, head_dim, 1, 1)),       # Q: single query
        mb.TensorSpec(shape=(1, head_dim, 1, seq_len)),  # K: cached keys
    ])
    def prog(q, k):
        # Q: [1, 128, 1, 1] → transpose to [1, 1, 1, 128]
        q_t = mb.transpose(x=q, perm=[0, 2, 3, 1], name="q_perm")
        # K: [1, 128, 1, 16] stays as is (128 channels, 16 spatial = 16 positions)
        # We want Q[1,128] dot K[128,16] = scores[1,16]
        # In NCHW: q_t[1,1,1,128] × k[1,128,1,16]
        # This is a 1x1 conv where q is the "filter" applied to k... but q is dynamic
        # Let's try explicit matmul
        q_2d = mb.reshape(x=q_t, shape=[1, head_dim], name="q_flat")
        k_2d = mb.reshape(x=k, shape=[head_dim, seq_len], name="k_flat")
        scores = mb.matmul(x=q_2d, y=k_2d, name="qk_scores")
        return mb.reshape(x=scores, shape=[1, seq_len, 1, 1], name="out")

    q_size = head_dim * 1 * 4  # Tiny input
    k_size = head_dim * seq_len * 4
    out_size = seq_len * 1 * 4

    # With NCHW padding
    q_nchw_size = 1 * head_dim * 1 * SPATIAL * 4  # pad spatial to 16
    k_nchw_size = 1 * head_dim * 1 * seq_len * 4

    try:
        # Try with raw sizes first
        kernel = compile_mil_program(prog, "test_matmul_nchw",
                                     [1 * head_dim * 1 * 1 * 4, 1 * head_dim * 1 * seq_len * 4],
                                     [1 * seq_len * 1 * 1 * 4])
        if kernel:
            print("  ✓ Compiled!")

            q = np.random.randn(head_dim).astype(np.float32)
            k_cache = np.random.randn(head_dim, seq_len).astype(np.float32)
            expected = q @ k_cache  # [seq_len]

            # Pack in NCHW: Q[1, head_dim, 1, 1], K[1, head_dim, 1, seq_len]
            q_buf = q.reshape(1, head_dim, 1, 1).astype(np.float32).ravel()
            k_buf = k_cache.reshape(1, head_dim, 1, seq_len).astype(np.float32).ravel()

            lib.ane_bridge_write_input(kernel, 0,
                q_buf.ctypes.data_as(ctypes.c_void_p), q_buf.nbytes)
            lib.ane_bridge_write_input(kernel, 1,
                k_buf.ctypes.data_as(ctypes.c_void_p), k_buf.nbytes)

            success = lib.ane_bridge_eval(kernel)
            if success:
                out = np.zeros(seq_len, dtype=np.float32)
                lib.ane_bridge_read_output(kernel, 0,
                    out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
                err = np.max(np.abs(out - expected))
                print(f"  ✓ EXECUTED! Max error: {err:.6f}")
                print(f"  Expected[:5]: {expected[:5]}")
                print(f"  Got[:5]:      {out[:5]}")
                return True
            else:
                print("  ✗ eval() returned False")
        else:
            print("  ✗ Compile failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    return False


def test_softmax():
    """Test: Can ANE run softmax?"""
    print("\n═══ TEST 3: softmax op ═══")
    from coremltools.converters.mil.mil import Builder as mb

    seq_len = 16

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, seq_len, 1, SPATIAL)),
    ])
    def prog(x):
        # Softmax along channel dim (axis=1 = the seq_len channels)
        return mb.softmax(x=x, axis=1, name="sm_out")

    in_size = seq_len * SPATIAL * 4
    out_size = seq_len * SPATIAL * 4

    try:
        kernel = compile_mil_program(prog, "test_softmax", [in_size], [out_size])
        if kernel:
            print("  ✓ Compiled! Testing...")
            x = np.random.randn(seq_len, SPATIAL).astype(np.float32)
            x_nchw = x.reshape(1, seq_len, 1, SPATIAL).ravel()

            lib.ane_bridge_write_input(kernel, 0,
                x_nchw.ctypes.data_as(ctypes.c_void_p), x_nchw.nbytes)

            success = lib.ane_bridge_eval(kernel)
            if success:
                out = np.zeros(seq_len * SPATIAL, dtype=np.float32)
                lib.ane_bridge_read_output(kernel, 0,
                    out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
                out = out.reshape(seq_len, SPATIAL)

                # Expected: softmax along axis 0 (channels)
                expected = np.exp(x - x.max(axis=0, keepdims=True))
                expected = expected / expected.sum(axis=0, keepdims=True)

                err = np.max(np.abs(out - expected))
                print(f"  ✓ EXECUTED! Max error: {err:.6f}")
                print(f"  Sum per spatial (should be 1.0): {out.sum(axis=0)[:4]}")
                return True
            else:
                print("  ✗ eval() returned False")
        else:
            print("  ✗ Compile failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    return False


def test_reduce_ops():
    """Test: reduce_max and reduce_sum (needed for manual softmax)."""
    print("\n═══ TEST 4: reduce_max + reduce_sum ═══")
    from coremltools.converters.mil.mil import Builder as mb

    ch = 64

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, ch, 1, SPATIAL)),
    ])
    def prog(x):
        mx = mb.reduce_max(x=x, axes=[1], keep_dims=True, name="rmax")
        return mb.reduce_sum(x=x, axes=[1], keep_dims=True, name="rsum")

    in_size = ch * SPATIAL * 4
    out_size = 1 * SPATIAL * 4  # reduced to 1 channel

    try:
        kernel = compile_mil_program(prog, "test_reduce", [in_size], [out_size])
        if kernel:
            print("  ✓ Compiled! Testing...")
            x = np.random.randn(ch, SPATIAL).astype(np.float32)
            x_buf = x.reshape(1, ch, 1, SPATIAL).ravel()

            lib.ane_bridge_write_input(kernel, 0,
                x_buf.ctypes.data_as(ctypes.c_void_p), x_buf.nbytes)

            success = lib.ane_bridge_eval(kernel)
            if success:
                out = np.zeros(1 * SPATIAL, dtype=np.float32)
                lib.ane_bridge_read_output(kernel, 0,
                    out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
                expected_sum = x.sum(axis=0)
                err = np.max(np.abs(out - expected_sum))
                print(f"  ✓ EXECUTED! Max error vs sum: {err:.6f}")
                print(f"  Expected[:4]: {expected_sum[:4]}")
                print(f"  Got[:4]:      {out[:4]}")
                return True
            else:
                print("  ✗ eval() returned False")
        else:
            print("  ✗ Compile failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    return False


def test_einsum_attention():
    """Test: Fused Q×K^T as conv1x1 where K_cache is the 'image' and Q is reshaped."""
    print("\n═══ TEST 5: Attention via conv — Q as filter on K_cache ═══")
    print("  Idea: K_cache as [1, head_dim, 1, seq_len] input")
    print("  Q reshaped as [1, 1, head_dim, 1] conv weight (but Q is dynamic...)")
    print("  → Can't use conv for this. Need matmul or einsum.")
    from coremltools.converters.mil.mil import Builder as mb

    head_dim = 128
    seq_len = 16

    # Try: two inputs, use mul + reduce_sum to compute dot products
    # scores[s] = sum(Q * K[s]) for each position s
    # If K is [1, head_dim, 1, seq_len], and Q is [1, head_dim, 1, 1]:
    # Q * K broadcasts to [1, head_dim, 1, seq_len]
    # reduce_sum over axis=1 (channels) → [1, 1, 1, seq_len] = scores!

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),   # Q (value in channel 0 of each head_dim row)
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),   # K_cache (seq_len positions in spatial dim)
    ])
    def prog(q, k):
        # Element-wise multiply: broadcasts Q across spatial positions
        qk = mb.mul(x=q, y=k, name="qk_mul")
        # Sum over channels (head_dim) → attention scores
        scores = mb.reduce_sum(x=qk, axes=[1], keep_dims=True, name="scores")
        return scores

    in_size = head_dim * SPATIAL * 4
    out_size = 1 * SPATIAL * 4

    try:
        kernel = compile_mil_program(prog, "test_dot_attn",
                                     [in_size, in_size], [out_size])
        if kernel:
            print("  ✓ Compiled! Testing dot-product attention...")

            # Q: single query vector [head_dim], broadcast to all spatial positions
            q_vec = np.random.randn(head_dim).astype(np.float32)
            q_buf = np.zeros((head_dim, SPATIAL), dtype=np.float32)
            for i in range(SPATIAL):
                q_buf[:, i] = q_vec  # Same Q at every spatial position

            # K_cache: seq_len key vectors packed into spatial dim
            k_cache = np.random.randn(head_dim, SPATIAL).astype(np.float32)

            q_flat = q_buf.ravel()
            k_flat = k_cache.ravel()

            lib.ane_bridge_write_input(kernel, 0,
                q_flat.ctypes.data_as(ctypes.c_void_p), q_flat.nbytes)
            lib.ane_bridge_write_input(kernel, 1,
                k_flat.ctypes.data_as(ctypes.c_void_p), k_flat.nbytes)

            success = lib.ane_bridge_eval(kernel)
            if success:
                out = np.zeros(1 * SPATIAL, dtype=np.float32)
                lib.ane_bridge_read_output(kernel, 0,
                    out.ctypes.data_as(ctypes.c_void_p), out.nbytes)

                # Expected: dot product of Q with each K position
                expected = q_vec @ k_cache  # [seq_len]
                err = np.max(np.abs(out - expected))
                print(f"  ✓ EXECUTED! Max error: {err:.6f}")
                print(f"  Expected[:5]: {expected[:5]}")
                print(f"  Got[:5]:      {out[:5]}")
                return True
            else:
                print("  ✗ eval() returned False")
        else:
            print("  ✗ Compile failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    return False


def test_fused_attention_head():
    """Test: Full single-head attention on ANE.
    Q[head_dim] × K_cache[head_dim, seq_len] → scores → softmax → × V_cache → output[head_dim]
    """
    print("\n═══ TEST 6: Fused single-head attention ═══")
    from coremltools.converters.mil.mil import Builder as mb

    head_dim = 128
    seq_len = SPATIAL  # Pack seq positions into spatial dim

    scale = 1.0 / np.sqrt(head_dim)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),   # Q (broadcast)
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),   # K_cache
        mb.TensorSpec(shape=(1, head_dim, 1, SPATIAL)),   # V_cache
    ])
    def prog(q, k, v):
        # Step 1: Dot product — Q · K per position
        qk = mb.mul(x=q, y=k, name="qk_mul")
        scores = mb.reduce_sum(x=qk, axes=[1], keep_dims=True, name="raw_scores")

        # Step 2: Scale
        scale_const = mb.const(val=np.float32(scale), name="scale")
        scores_scaled = mb.mul(x=scores, y=scale_const, name="scaled_scores")

        # Step 3: Softmax over spatial dim (seq positions)
        attn_weights = mb.softmax(x=scores_scaled, axis=3, name="attn_weights")

        # Step 4: Weighted sum of V
        # attn_weights: [1, 1, 1, seq_len], V: [1, head_dim, 1, seq_len]
        # Broadcast multiply: [1, head_dim, 1, seq_len]
        weighted_v = mb.mul(x=attn_weights, y=v, name="weighted_v")

        # Step 5: Sum over spatial (seq positions) → [1, head_dim, 1, 1]
        output = mb.reduce_sum(x=weighted_v, axes=[3], keep_dims=True, name="attn_out")
        return output

    in_size = head_dim * SPATIAL * 4
    out_size = head_dim * 1 * 4  # Output: [1, head_dim, 1, 1]

    try:
        kernel = compile_mil_program(prog, "test_fused_attn",
                                     [in_size, in_size, in_size],
                                     [out_size])
        if kernel:
            print("  ✓ Compiled! Testing full attention head...")

            q_vec = np.random.randn(head_dim).astype(np.float32)
            k_cache = np.random.randn(head_dim, SPATIAL).astype(np.float32)
            v_cache = np.random.randn(head_dim, SPATIAL).astype(np.float32)

            # Pack Q — broadcast to all spatial positions
            q_buf = np.zeros((head_dim, SPATIAL), dtype=np.float32)
            for i in range(SPATIAL):
                q_buf[:, i] = q_vec

            q_flat = q_buf.ravel()
            k_flat = k_cache.ravel()
            v_flat = v_cache.ravel()

            lib.ane_bridge_write_input(kernel, 0,
                q_flat.ctypes.data_as(ctypes.c_void_p), q_flat.nbytes)
            lib.ane_bridge_write_input(kernel, 1,
                k_flat.ctypes.data_as(ctypes.c_void_p), k_flat.nbytes)
            lib.ane_bridge_write_input(kernel, 2,
                v_flat.ctypes.data_as(ctypes.c_void_p), v_flat.nbytes)

            success = lib.ane_bridge_eval(kernel)
            if success:
                out = np.zeros(head_dim, dtype=np.float32)
                lib.ane_bridge_read_output(kernel, 0,
                    out.ctypes.data_as(ctypes.c_void_p), out.nbytes)

                # CPU reference: standard attention
                scores = (q_vec @ k_cache) * scale  # [seq_len]
                scores_sm = np.exp(scores - scores.max())
                scores_sm = scores_sm / scores_sm.sum()
                expected = v_cache @ scores_sm  # [head_dim]

                err = np.max(np.abs(out - expected))
                rel_err = err / (np.max(np.abs(expected)) + 1e-8)
                print(f"  ✓ EXECUTED! Max error: {err:.6f}, relative: {rel_err:.6f}")
                print(f"  Expected[:5]: {expected[:5]}")
                print(f"  Got[:5]:      {out[:5]}")
                if rel_err < 0.01:
                    print("  ★★★ FUSED ATTENTION WORKS ON ANE! ★★★")
                return True
            else:
                print("  ✗ eval() returned False")
        else:
            print("  ✗ Compile failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    return False


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ANE ATTENTION OP EXPLORATION                       ║")
    print("║  Testing which MIL ops execute on Neural Engine      ║")
    print("╚══════════════════════════════════════════════════════╝")

    if not init_ane():
        print("ANE init failed!")
        sys.exit(1)

    results = {}

    results["matmul_2input"] = test_matmul_two_inputs()
    results["matmul_nchw"] = test_matmul_nchw()
    results["softmax"] = test_softmax()
    results["reduce_ops"] = test_reduce_ops()
    results["dot_attn"] = test_einsum_attention()
    results["fused_attn"] = test_fused_attention_head()

    print("\n" + "═" * 56)
    print("RESULTS SUMMARY:")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")

    n_pass = sum(1 for v in results.values() if v)
    print(f"\n{n_pass}/{len(results)} tests passed")

    if results.get("fused_attn"):
        print("\n★ FUSED ATTENTION ON ANE IS VIABLE!")
        print("  Next: integrate into real_draft.py forward pass")
    elif results.get("dot_attn"):
        print("\n◆ Dot-product attention works — can build softmax + weighted sum separately")
    elif results.get("softmax") or results.get("reduce_ops"):
        print("\n◆ Element-wise ops work — can compose attention from primitives")
