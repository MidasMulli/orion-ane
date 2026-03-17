"""
Test fusing Q, K, V projections into a single ANE kernel.
Currently: 3 dispatches per layer for QKV = 84 dispatches across 28 layers.
Goal: 1 dispatch for all 3 → save ~0.12ms × 28 layers × 2 = 6.7ms
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


def test_fused_qkv():
    """Fuse Q+K+V into single conv with concatenated output channels."""
    print("═══ FUSED QKV KERNEL ═══")
    from coremltools.converters.mil.mil import Builder as mb
    import coremltools as ct

    dim = 1024
    q_dim = 2048   # 16 heads × 128
    kv_dim = 1024  # 8 heads × 128
    qkv_dim = q_dim + kv_dim + kv_dim  # 4096 total output channels

    # Create random weights simulating Q, K, V concatenated
    w_q = np.random.randn(q_dim, dim).astype(np.float32) * 0.02
    w_k = np.random.randn(kv_dim, dim).astype(np.float32) * 0.02
    w_v = np.random.randn(kv_dim, dim).astype(np.float32) * 0.02
    w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)  # [4096, 1024]

    w_conv = w_qkv.reshape(qkv_dim, dim, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, SPATIAL))])
    def prog(x):
        W = mb.const(val=w_conv, name="W")
        return mb.conv(x=x, weight=W, name="qkv_out")

    model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15)
    pkg_path = "/tmp/fused_qkv.mlpackage"
    model.save(pkg_path)
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg_path, "/tmp/"],
                   capture_output=True)
    compiled = "/tmp/fused_qkv.mlmodelc"
    with open(f"{compiled}/model.mil", 'rb') as f:
        mil_bytes = f.read()
    weight_path = f"{compiled}/weights/weight.bin"
    weight_blob = None
    if os.path.exists(weight_path):
        with open(weight_path, 'rb') as f:
            weight_blob = f.read()

    in_size = dim * SPATIAL * 4
    out_size = qkv_dim * SPATIAL * 4

    kernel = compile_ane_kernel(mil_bytes, weight_blob, [in_size], [out_size])
    if not kernel:
        print("  ✗ Compile failed — QKV too large for single kernel")
        return False

    print("  ✓ Compiled!")

    # Test
    x = np.random.randn(dim).astype(np.float32)
    x_padded = np.zeros((dim, SPATIAL), dtype=np.float32)
    x_padded[:, 0] = x
    x_flat = x_padded.ravel()

    lib.ane_bridge_write_input(kernel, 0,
        x_flat.ctypes.data_as(ctypes.c_void_p), x_flat.nbytes)
    success = lib.ane_bridge_eval(kernel)

    if not success:
        print("  ✗ eval() failed")
        return False

    out = np.zeros(qkv_dim * SPATIAL, dtype=np.float32)
    lib.ane_bridge_read_output(kernel, 0,
        out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
    out = out.reshape(qkv_dim, SPATIAL)[:, 0]

    # CPU reference
    expected = w_qkv @ x
    err = np.max(np.abs(out - expected))
    print(f"  ✓ EXECUTED! Max error: {err:.6f}")

    # Split output
    q_out = out[:q_dim]
    k_out = out[q_dim:q_dim + kv_dim]
    v_out = out[q_dim + kv_dim:]
    print(f"  Q output shape: {q_out.shape}, K: {k_out.shape}, V: {v_out.shape}")

    # Benchmark: fused vs 3 separate
    n_iters = 500

    # Fused
    t0 = time.time()
    for _ in range(n_iters):
        lib.ane_bridge_write_input(kernel, 0,
            x_flat.ctypes.data_as(ctypes.c_void_p), x_flat.nbytes)
        lib.ane_bridge_eval(kernel)
        lib.ane_bridge_read_output(kernel, 0,
            out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
    fused_time = (time.time() - t0) * 1000

    print(f"\n  Fused QKV: {fused_time:.1f}ms for {n_iters} iters = {fused_time/n_iters:.3f}ms/iter")
    print(f"  Current 3-separate: ~{0.06*3:.3f}ms/iter (estimate)")
    print(f"  Savings per layer: ~{0.06*3 - fused_time/n_iters:.3f}ms")
    print(f"  Savings for 28 layers: ~{(0.06*3 - fused_time/n_iters)*28:.1f}ms")

    return True


def test_fused_gate_up():
    """Fuse gate + up projections into single kernel."""
    print("\n═══ FUSED GATE+UP KERNEL ═══")
    from coremltools.converters.mil.mil import Builder as mb
    import coremltools as ct

    dim = 1024
    hidden_dim = 2816
    fused_dim = hidden_dim * 2  # gate + up concatenated

    w_gate = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
    w_up = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
    w_fused = np.concatenate([w_gate, w_up], axis=0)
    w_conv = w_fused.reshape(fused_dim, dim, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, SPATIAL))])
    def prog(x):
        W = mb.const(val=w_conv, name="W")
        return mb.conv(x=x, weight=W, name="gate_up_out")

    model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15)
    pkg_path = "/tmp/fused_gate_up.mlpackage"
    model.save(pkg_path)
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg_path, "/tmp/"],
                   capture_output=True)
    compiled = "/tmp/fused_gate_up.mlmodelc"
    with open(f"{compiled}/model.mil", 'rb') as f:
        mil_bytes = f.read()
    weight_path = f"{compiled}/weights/weight.bin"
    weight_blob = None
    if os.path.exists(weight_path):
        with open(weight_path, 'rb') as f:
            weight_blob = f.read()

    in_size = dim * SPATIAL * 4
    out_size = fused_dim * SPATIAL * 4

    kernel = compile_ane_kernel(mil_bytes, weight_blob, [in_size], [out_size])
    if not kernel:
        print("  ✗ Compile failed")
        return False

    print(f"  ✓ Compiled! ({fused_dim} output channels)")

    x = np.random.randn(dim).astype(np.float32)
    x_padded = np.zeros((dim, SPATIAL), dtype=np.float32)
    x_padded[:, 0] = x
    x_flat = x_padded.ravel()

    lib.ane_bridge_write_input(kernel, 0,
        x_flat.ctypes.data_as(ctypes.c_void_p), x_flat.nbytes)
    success = lib.ane_bridge_eval(kernel)

    if not success:
        print("  ✗ eval() failed")
        return False

    out = np.zeros(fused_dim * SPATIAL, dtype=np.float32)
    lib.ane_bridge_read_output(kernel, 0,
        out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
    out = out.reshape(fused_dim, SPATIAL)[:, 0]

    expected = w_fused @ x
    err = np.max(np.abs(out - expected))
    print(f"  ✓ EXECUTED! Max error: {err:.6f}")
    return True


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  KERNEL FUSION TESTS                                 ║")
    print("╚══════════════════════════════════════════════════════╝")

    if not init_ane():
        print("ANE init failed!")
        sys.exit(1)

    r1 = test_fused_qkv()
    r2 = test_fused_gate_up()

    print("\n" + "=" * 56)
    if r1 and r2:
        print("★ KERNEL FUSION WORKS!")
        print("  Current: 7 kernels/layer × 28 layers = 196 dispatches")
        print("  Fused:   4 kernels/layer × 28 layers = 112 dispatches")
        print("  That's 84 fewer round trips = significant speedup")
    elif r1:
        print("QKV fusion works, gate+up failed")
    elif r2:
        print("gate+up fusion works, QKV failed")
    else:
        print("Neither fusion worked")
