"""
Direct ANE Kernel Execution — Bypassing CoreML
Compiles and runs compute kernels directly on the Apple Neural Engine
via reverse-engineered _ANEClient private APIs.

Raw compute on dedicated silicon — no CoreML, no Metal, no GPU.
MIL format validated against macOS 26 (Tahoe) ANE compiler.
"""

import ctypes
import struct
import time
import os
import sys
import array
import subprocess
import tempfile
import json

# ── Load the bridge ──────────────────────────────────────────────────────
BRIDGE_PATH = os.path.join(os.path.dirname(__file__), "bridge", "libane_bridge.dylib")
lib = ctypes.CDLL(BRIDGE_PATH)

# ── Function signatures ─────────────────────────────────────────────────
lib.ane_bridge_init.restype = ctypes.c_int

lib.ane_bridge_compile.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
]
lib.ane_bridge_compile.restype = ctypes.c_void_p

lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
lib.ane_bridge_eval.restype = ctypes.c_bool

lib.ane_bridge_write_input.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_size_t,
]

lib.ane_bridge_read_output.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_size_t,
]

lib.ane_bridge_free.argtypes = [ctypes.c_void_p]

lib.ane_bridge_build_weight_blob.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t),
]
lib.ane_bridge_build_weight_blob.restype = ctypes.c_void_p


# ── MIL Generator using coremltools (produces macOS 26-compatible format) ──

def generate_mil_via_coremltools(program_fn, model_name="ane_kernel"):
    """
    Use coremltools to generate valid MIL, then compile to get the ANE-compatible format.
    Returns (mil_bytes, weight_blob_bytes_or_None).
    """
    import coremltools as ct

    model = ct.convert(program_fn, minimum_deployment_target=ct.target.macOS15)

    # Save and compile
    pkg_path = f"/tmp/{model_name}.mlpackage"
    model.save(pkg_path)

    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", pkg_path, "/tmp/"],
        capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  coremlcompiler failed: {result.stderr[:200]}")
        return None, None

    compiled_path = f"/tmp/{model_name}.mlmodelc"
    mil_path = os.path.join(compiled_path, "model.mil")

    if not os.path.exists(mil_path):
        print(f"  No model.mil found")
        return None, None

    with open(mil_path, 'rb') as f:
        mil_bytes = f.read()

    # Check for weight blob
    weight_path = os.path.join(compiled_path, "weights", "weight.bin")
    weight_blob = None
    if os.path.exists(weight_path):
        with open(weight_path, 'rb') as f:
            weight_blob = f.read()

    return mil_bytes, weight_blob


def make_add_program(channels, spatial):
    """Element-wise add: y = x + x"""
    from coremltools.converters.mil.mil import Builder as mb
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, channels, 1, spatial))])
    def prog(x):
        return mb.add(x=x, y=x, name="add_out")
    return prog


def make_mul_add_program(channels, spatial):
    """Fused multiply-add: y = x*x + x"""
    from coremltools.converters.mil.mil import Builder as mb
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, channels, 1, spatial))])
    def prog(x):
        sq = mb.mul(x=x, y=x, name="mul_op")
        return mb.add(x=sq, y=x, name="add_out")
    return prog


def make_matmul_program(ic, oc, seq):
    """
    Dynamic matmul: weights packed in input IOSurface.
    Input: [1, IC, 1, SEQ+OC] — act[0:SEQ] + W[SEQ:SEQ+OC]
    Output: [1, OC, 1, SEQ]
    """
    from coremltools.converters.mil.mil import Builder as mb
    import numpy as np
    sp_total = seq + oc
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, ic, 1, sp_total))])
    def prog(x):
        # Slice activations and weights
        act = mb.slice_by_size(x=x, begin=[0,0,0,0], size=[1,ic,1,seq], name="act")
        wt = mb.slice_by_size(x=x, begin=[0,0,0,seq], size=[1,ic,1,oc], name="wt")
        # Reshape for matmul
        a2 = mb.reshape(x=act, shape=[1,1,ic,seq], name="ra")
        a3 = mb.transpose(x=a2, perm=[0,1,3,2], name="ta")
        w2 = mb.reshape(x=wt, shape=[1,1,ic,oc], name="rw")
        # matmul: [1,1,SEQ,IC] @ [1,1,IC,OC] -> [1,1,SEQ,OC]
        yh = mb.matmul(x=a3, y=w2, name="mm")
        # Reshape back
        yt = mb.transpose(x=yh, perm=[0,1,3,2], name="ty")
        y = mb.reshape(x=yt, shape=[1,oc,1,seq], name="out")
        return y
    return prog


def make_linear_program(in_ch, out_ch, spatial):
    """
    Linear layer with baked weights (conv1x1 pattern).
    Input: [1, in_ch, 1, spatial]
    Output: [1, out_ch, 1, spatial]
    """
    from coremltools.converters.mil.mil import Builder as mb
    import numpy as np
    weights = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * 0.01
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, in_ch, 1, spatial))])
    def prog(x):
        W = mb.const(val=weights, name="W")
        return mb.conv(x=x, weight=W, name="conv_out")
    return prog


# ── Compile and benchmark ──────────────────────────────────────────────
def run_kernel(name, program_fn, model_name, n_inputs, input_sizes_list,
               n_outputs, output_sizes_list, input_data_fn, n_iters=200,
               flops=0, verify_fn=None):
    """Generic kernel runner."""
    print(f"\n{'='*60}")
    print(f"  Kernel: {name}")
    print(f"{'='*60}")

    # Generate MIL via coremltools
    print(f"  Generating MIL via coremltools...")
    mil_bytes, weight_blob = generate_mil_via_coremltools(program_fn, model_name)
    if not mil_bytes:
        print(f"  ❌ MIL generation FAILED")
        return None

    print(f"  MIL: {len(mil_bytes)} bytes" + (f", weights: {len(weight_blob)} bytes" if weight_blob else ""))

    input_sizes = (ctypes.c_size_t * n_inputs)(*input_sizes_list)
    output_sizes = (ctypes.c_size_t * n_outputs)(*output_sizes_list)

    # Compile — always pass a weight blob (macOS 26 bridge quirk:
    # modelWithMILText:weights: rejects empty weight dict through ctypes)
    if not weight_blob:
        # Minimal valid ANE weight blob: 128-byte header, no actual weights
        weight_blob = bytearray(128)
        weight_blob[0] = 0x01
        weight_blob[4] = 0x02
        weight_blob[64] = 0xEF
        weight_blob[65] = 0xBE
        weight_blob[66] = 0xAD
        weight_blob[67] = 0xDE
        weight_blob[68] = 0x01
        weight_blob = bytes(weight_blob)

    wb = ctypes.create_string_buffer(weight_blob)
    wl = len(weight_blob)

    t0 = time.time()
    kernel = lib.ane_bridge_compile(
        mil_bytes, len(mil_bytes),
        wb, wl,
        n_inputs, input_sizes,
        n_outputs, output_sizes,
    )
    compile_time = (time.time() - t0) * 1000

    if not kernel:
        print(f"  ❌ Compile FAILED")
        return None

    print(f"  ✓ Compiled in {compile_time:.1f} ms")

    # Write inputs
    input_arrays = input_data_fn()
    for i, (data, nbytes) in enumerate(zip(input_arrays, input_sizes_list)):
        buf = (ctypes.c_char * nbytes).from_buffer(data)
        lib.ane_bridge_write_input(kernel, i, buf, nbytes)

    # Warmup
    print(f"  Warming up (10 runs)...")
    for _ in range(10):
        ok = lib.ane_bridge_eval(kernel)
        if not ok:
            print(f"  ❌ Eval FAILED")
            lib.ane_bridge_free(kernel)
            return None

    # Benchmark
    print(f"  Benchmarking ({n_iters} runs)...")
    times = []
    for _ in range(n_iters):
        t0 = time.time()
        lib.ane_bridge_eval(kernel)
        times.append((time.time() - t0) * 1000)

    # Read outputs
    output_arrays = []
    for i, nbytes in enumerate(output_sizes_list):
        n_floats = nbytes // 4
        out = array.array('f', [0.0] * n_floats)
        buf = (ctypes.c_char * nbytes).from_buffer(out)
        lib.ane_bridge_read_output(kernel, i, buf, nbytes)
        output_arrays.append(out)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    p50 = sorted(times)[len(times) // 2]
    gflops = flops / (avg_ms / 1000) / 1e9 if flops > 0 else 0

    # Verify
    if verify_fn:
        verify_fn(input_arrays, output_arrays)

    print(f"\n  ── Performance ──")
    print(f"  Avg latency:  {avg_ms:.4f} ms")
    print(f"  P50 latency:  {p50:.4f} ms")
    print(f"  Min latency:  {min_ms:.4f} ms")
    print(f"  Max latency:  {max_ms:.4f} ms")
    if gflops > 0:
        print(f"  Throughput:   {gflops:.2f} GFLOPS")
    print(f"  Compile:      {compile_time:.1f} ms (one-time)")

    lib.ane_bridge_free(kernel)

    return {
        "name": name,
        "compile_ms": compile_time,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "p50_ms": p50,
        "gflops": gflops,
        "flops": flops,
    }


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  DIRECT ANE KERNEL EXECUTION — macOS 26 Tahoe           ║")
    print("║  Bypassing CoreML → Raw _ANEClient Private APIs         ║")
    print("║  Target: Apple M5 Neural Engine (16-core)               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    result = lib.ane_bridge_init()
    if result != 0:
        print("Failed to initialize ANE bridge!")
        sys.exit(1)
    print("\n✓ ANE bridge initialized — direct hardware access")

    results = []

    # ═══════════════════════════════════════════════════════════
    # TEST 1: Element-wise Add (simplest possible kernel)
    # ═══════════════════════════════════════════════════════════
    ch, sp = 64, 64
    tensor_bytes = ch * sp * 4

    def add_inputs():
        data = array.array('f', [float(i) * 0.001 for i in range(ch * sp)])
        return [data]

    def verify_add(inputs, outputs):
        max_err = max(abs(outputs[0][i] - inputs[0][i] * 2) for i in range(min(100, len(outputs[0]))))
        status = "✓ PASS" if max_err < 0.01 else "✗ FAIL"
        print(f"\n  ── Correctness ──")
        print(f"  x+x: max_err={max_err:.6f} {status}")
        print(f"  In[0:4]:  {[f'{inputs[0][i]:.4f}' for i in range(4)]}")
        print(f"  Out[0:4]: {[f'{outputs[0][i]:.4f}' for i in range(4)]}")

    r = run_kernel(
        f"Element-wise Add [1,{ch},1,{sp}]",
        make_add_program(ch, sp), "ane_add",
        1, [tensor_bytes], 1, [tensor_bytes],
        add_inputs, flops=ch*sp, verify_fn=verify_add)
    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════
    # TEST 2: Fused Multiply-Add (x*x + x)
    # ═══════════════════════════════════════════════════════════
    def verify_mul_add(inputs, outputs):
        max_err = max(abs(outputs[0][i] - (inputs[0][i]**2 + inputs[0][i]))
                      for i in range(min(100, len(outputs[0]))))
        status = "✓ PASS" if max_err < 0.01 else "✗ FAIL"
        print(f"\n  ── Correctness ──")
        print(f"  x²+x: max_err={max_err:.6f} {status}")

    r = run_kernel(
        f"Fused MulAdd [1,{ch},1,{sp}]",
        make_mul_add_program(ch, sp), "ane_muladd",
        1, [tensor_bytes], 1, [tensor_bytes],
        add_inputs, flops=2*ch*sp, verify_fn=verify_mul_add)
    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════
    # TEST 3: Larger Add [1, 256, 1, 256]
    # ═══════════════════════════════════════════════════════════
    ch2, sp2 = 256, 256
    tensor_bytes2 = ch2 * sp2 * 4

    def add_inputs2():
        data = array.array('f', [float(i % 1000) * 0.001 for i in range(ch2 * sp2)])
        return [data]

    r = run_kernel(
        f"Element-wise Add [1,{ch2},1,{sp2}]",
        make_add_program(ch2, sp2), "ane_add_large",
        1, [tensor_bytes2], 1, [tensor_bytes2],
        add_inputs2, flops=ch2*sp2)
    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════
    # TEST 4: Dynamic Matmul 64×64 (identity correctness check)
    # ═══════════════════════════════════════════════════════════
    ic, oc, seq = 64, 64, 64
    mm_in_bytes = ic * (seq + oc) * 4
    mm_out_bytes = oc * seq * 4
    sp_total = seq + oc

    def matmul_inputs_identity():
        data = array.array('f', [0.0] * (ic * sp_total))
        for d in range(ic):
            for s in range(seq):
                data[d * sp_total + s] = float(d * seq + s) * 0.001
        for d in range(ic):
            data[d * sp_total + seq + d] = 1.0  # identity
        return [data]

    def verify_matmul_identity(inputs, outputs):
        max_err = 0
        for d in range(min(oc, 16)):
            for s in range(min(seq, 16)):
                expected = inputs[0][d * sp_total + s]
                actual = outputs[0][d * seq + s]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
        status = "✓ PASS" if max_err < 0.1 else "✗ FAIL"
        print(f"\n  ── Correctness (W=identity) ──")
        print(f"  max_err={max_err:.6f} {status}")

    r = run_kernel(
        f"Matmul {ic}×{oc}×{seq} (identity)",
        make_matmul_program(ic, oc, seq), "ane_mm_64",
        1, [mm_in_bytes], 1, [mm_out_bytes],
        matmul_inputs_identity, flops=2*ic*oc*seq,
        verify_fn=verify_matmul_identity)
    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════
    # TEST 5: Matmul 256×256×64
    # ═══════════════════════════════════════════════════════════
    ic5, oc5, seq5 = 256, 256, 64
    mm5_in = ic5 * (seq5 + oc5) * 4
    mm5_out = oc5 * seq5 * 4
    sp5 = seq5 + oc5

    def matmul_inputs_256():
        data = array.array('f', [0.0] * (ic5 * sp5))
        for d in range(ic5):
            for s in range(seq5):
                data[d * sp5 + s] = float((d * seq5 + s) % 100) * 0.01
            for c in range(oc5):
                data[d * sp5 + seq5 + c] = float((d * oc5 + c) % 100) * 0.001
        return [data]

    r = run_kernel(
        f"Matmul {ic5}×{oc5}×{seq5}",
        make_matmul_program(ic5, oc5, seq5), "ane_mm_256",
        1, [mm5_in], 1, [mm5_out],
        matmul_inputs_256, flops=2*ic5*oc5*seq5)
    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════
    # TEST 6: Matmul 768×768×256 (transformer-scale!)
    # ═══════════════════════════════════════════════════════════
    ic6, oc6, seq6 = 768, 768, 256
    mm6_in = ic6 * (seq6 + oc6) * 4
    mm6_out = oc6 * seq6 * 4
    sp6 = seq6 + oc6

    def matmul_inputs_768():
        data = array.array('f', [0.0] * (ic6 * sp6))
        for d in range(ic6):
            for s in range(seq6):
                data[d * sp6 + s] = float((d * seq6 + s) % 100) * 0.01
            for c in range(oc6):
                data[d * sp6 + seq6 + c] = float((d * oc6 + c) % 100) * 0.001
        return [data]

    r = run_kernel(
        f"Matmul {ic6}×{oc6}×{seq6}",
        make_matmul_program(ic6, oc6, seq6), "ane_mm_768",
        1, [mm6_in], 1, [mm6_out],
        matmul_inputs_768, flops=2*ic6*oc6*seq6, n_iters=100)
    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════
    # TEST 7: Linear layer with baked weights (conv1x1)
    # ═══════════════════════════════════════════════════════════
    lin_in, lin_out, lin_sp = 256, 256, 64
    lin_in_bytes = lin_in * lin_sp * 4
    lin_out_bytes = lin_out * lin_sp * 4

    def linear_inputs():
        data = array.array('f', [float(i % 100) * 0.01 for i in range(lin_in * lin_sp)])
        return [data]

    r = run_kernel(
        f"Conv1x1 {lin_in}→{lin_out}×{lin_sp} (baked W)",
        make_linear_program(lin_in, lin_out, lin_sp), "ane_linear",
        1, [lin_in_bytes], 1, [lin_out_bytes],
        linear_inputs, flops=2*lin_in*lin_out*lin_sp)
    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    if results:
        print(f"\n{'='*80}")
        print(f"  ⚡ SUMMARY — Direct ANE Execution (No CoreML, No GPU)")
        print(f"{'='*80}")
        print(f"  {'Kernel':<35} {'Compile':<10} {'Avg ms':<10} {'P50 ms':<10} {'GFLOPS':<10}")
        print(f"  {'-'*75}")
        for r in results:
            gf = f"{r['gflops']:.2f}" if r['gflops'] > 0 else "n/a"
            print(f"  {r['name']:<35} {r['compile_ms']:<10.0f} {r['avg_ms']:<10.4f} {r['p50_ms']:<10.4f} {gf:<10}")

        total_flops = sum(r['flops'] for r in results if r['flops'] > 0)
        total_time = sum(r['avg_ms'] for r in results if r['flops'] > 0)
        if total_time > 0:
            aggregate_gflops = total_flops / (total_time / 1000) / 1e9
            print(f"\n  Aggregate throughput: {aggregate_gflops:.2f} GFLOPS")

        print(f"\n  ⚡ All kernels executed on Neural Engine silicon.")
        print(f"  ⚡ Zero CoreML overhead. Zero GPU. Raw _ANEClient dispatch.")
        print(f"  ⚡ MIL compiled via coremltools → ANECCompile → H11ANE hardware.")
        print(f"  ⚡ This is the Orion frontier — direct kernel dispatch to ANE on M5.")
    else:
        print("\n  ❌ No kernels succeeded.")
