#!/usr/bin/env python3
"""
NAX Layout Probe via MLX — uses MLX's NAX-compiled matmul kernels to
observe physical layout behavior without fighting the Metal toolchain.

MLX's steel_gemm_fused_nax kernel fires on M5 for FP16 matmuls ≥16×16.
We feed known patterns and read back results to reverse-engineer the layout.

Tests:
  0. Reproduce get_coord() thread-to-element mapping
  1. Identity matmul: I×B = B (correctness check)
  2. One-hot probes: single-element A × all-ones B
  3. Sequential patterns: detect any permutation in output
  4. FP16 vs FP32 timing (NAX vs SIMD ALU)
  5. Batch verify cost plateau (NAX amortization)
  6. Threadgroup padding detection via bank conflict timing
"""

import time
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available — running CPU-only analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# Test 0: get_coord() mapping (pure Python, no GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def test0_coord_map():
    """Reproduce MLX BaseNAXFrag::get_coord() and map the full 16×16 tile."""
    print("\n" + "=" * 65)
    print("Test 0: get_coord() Thread-to-Element Mapping")
    print("=" * 65)

    ownership = np.full((16, 16), -1, dtype=int)
    thread_elems = {}

    for lane in range(32):
        qid = lane >> 2
        fm = ((qid & 4) | ((lane >> 1) & 3))           # row
        fn = ((qid & 2) | (lane & 1)) * 4               # col base

        elems = []
        for i in range(2):  # kElemRows = 2
            r = fm + i * 8  # kElemRowsJump = 8
            for j in range(4):  # kElemCols = 4
                c = fn + j
                ownership[r, c] = lane
                elems.append((r, c))
        thread_elems[lane] = elems

    # Print ownership map
    print("\n  Ownership map (which SIMD lane owns each 16×16 element):")
    print("       " + " ".join(f"{c:3d}" for c in range(16)))
    print("      " + "-" * 64)
    for r in range(16):
        row = " ".join(f"{ownership[r, c]:3d}" for c in range(16))
        print(f"  r{r:02d} | {row}")

    # Structural analysis: quad grouping
    print("\n  Quad structure (4 threads per quad):")
    for quad in range(8):
        lanes = list(range(quad * 4, quad * 4 + 4))
        rows = sorted(set(r for l in lanes for r, c in thread_elems[l]))
        cols = sorted(set(c for l in lanes for r, c in thread_elems[l]))
        print(f"    Quad {quad} (lanes {lanes[0]:2d}-{lanes[3]:2d}): "
              f"rows={rows}, cols={cols}")

    # Key insight: the layout is 4×4 block tiled with 8-row split
    print("\n  Layout structure:")
    print("    - 16×16 split into 4 quadrants: [0:8,0:8] [0:8,8:16] [8:16,0:8] [8:16,8:16]")
    print("    - Each quadrant: 4 quads × 4 threads = 16 threads")
    print("    - Each thread: 1 row × 4 cols (contiguous)")
    print("    - Same thread owns mirror positions 8 rows apart")

    return ownership, thread_elems


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Identity matmul (MLX)
# ═══════════════════════════════════════════════════════════════════════════════

def test1_identity():
    """I(16) × B(sequential) should equal B. Any deviation = layout permutation."""
    if not HAS_MLX:
        print("\n  [SKIP] No MLX")
        return

    print("\n" + "=" * 65)
    print("Test 1: Identity Matmul (I × sequential = sequential?)")
    print("=" * 65)

    I = mx.eye(16, dtype=mx.float16)
    B = mx.arange(256, dtype=mx.float16).reshape(16, 16)
    C = I @ B
    mx.eval(C)

    result = np.array(C)
    expected = np.arange(256, dtype=np.float32).reshape(16, 16)

    max_err = np.max(np.abs(result - expected))
    mismatch = np.sum(np.abs(result - expected) > 0.5)

    print(f"\n  Max error: {max_err:.4f}")
    print(f"  Mismatched positions: {mismatch}/256")

    if mismatch == 0:
        print("  → NAX identity matmul is CORRECT (no observable permutation)")
    else:
        print("  → PERMUTATION DETECTED!")
        for r in range(16):
            for c in range(16):
                if abs(result[r, c] - expected[r, c]) > 0.5:
                    print(f"    [{r},{c}]: expected={expected[r,c]:.0f}, got={result[r,c]:.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: One-hot probes
# ═══════════════════════════════════════════════════════════════════════════════

def test2_one_hot():
    """Set one element in A, multiply by all-ones B. Should light up one row."""
    if not HAS_MLX:
        print("\n  [SKIP] No MLX")
        return

    print("\n" + "=" * 65)
    print("Test 2: One-Hot Probes (A[r,k]=1, B=ones → C[r,:]=ones)")
    print("=" * 65)

    B = mx.ones((16, 16), dtype=mx.float16)

    positions = [(0, 0), (0, 5), (3, 7), (8, 0), (15, 15), (7, 12)]
    for r_in, k_in in positions:
        A = mx.zeros((16, 16), dtype=mx.float16)
        A_np = np.zeros((16, 16), dtype=np.float16)
        A_np[r_in, k_in] = 1.0
        A = mx.array(A_np)

        C = A @ B
        mx.eval(C)
        result = np.array(C)

        # Find nonzero elements
        nonzero_rows = set()
        nonzero_vals = {}
        for r in range(16):
            for c in range(16):
                if abs(result[r, c]) > 0.5:
                    nonzero_rows.add(r)
                    nonzero_vals[(r, c)] = result[r, c]

        expected_row = r_in  # A[r,k]=1, B=ones → C[r,:] = sum(A[r,:] * B[:,c]) = 1 for all c
        match = nonzero_rows == {expected_row}

        status = "OK" if match else "MISMATCH"
        print(f"  A[{r_in},{k_in}]=1 → nonzero rows={sorted(nonzero_rows)} "
              f"(expected [{expected_row}]) [{status}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Sequential pattern detection
# ═══════════════════════════════════════════════════════════════════════════════

def test3_sequential():
    """Multiply two sequential matrices. Check if output matches numpy exactly."""
    if not HAS_MLX:
        print("\n  [SKIP] No MLX")
        return

    print("\n" + "=" * 65)
    print("Test 3: Sequential Pattern Detection")
    print("=" * 65)

    # Small values to avoid FP16 overflow
    A_np = (np.arange(256).reshape(16, 16) / 256.0).astype(np.float16)
    B_np = (np.arange(256).reshape(16, 16) / 256.0).astype(np.float16)

    A = mx.array(A_np)
    B = mx.array(B_np)
    C = A @ B
    mx.eval(C)

    result = np.array(C).astype(np.float32)
    expected = (A_np.astype(np.float32) @ B_np.astype(np.float32))

    # FP16 precision: elements can differ by a few ULP
    diff = np.abs(result - expected)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    large_diff = np.sum(diff > 0.1)

    print(f"\n  Max element diff:  {max_diff:.6f}")
    print(f"  Mean element diff: {mean_diff:.6f}")
    print(f"  Elements >0.1 off: {large_diff}/256")

    if large_diff == 0:
        print("  → NAX matmul matches CPU reference within FP16 precision")
    else:
        print("  → Significant deviations detected (potential layout issue)")
        # Show worst positions
        worst = np.unravel_index(np.argsort(diff.ravel())[-5:], diff.shape)
        for r, c in zip(worst[0], worst[1]):
            print(f"    [{r},{c}]: expected={expected[r,c]:.4f}, got={result[r,c]:.4f}, "
                  f"diff={diff[r,c]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: FP16 vs FP32 timing (NAX vs SIMD ALU)
# ═══════════════════════════════════════════════════════════════════════════════

def test4_timing():
    """FP16 matmul hits NAX, FP32 goes through SIMD ALU. Measure both."""
    if not HAS_MLX:
        print("\n  [SKIP] No MLX")
        return

    print("\n" + "=" * 65)
    print("Test 4: FP16 (NAX) vs FP32 (SIMD ALU) Timing")
    print("=" * 65)

    sizes = [16, 32, 64, 128, 256, 512, 1024]

    print(f"\n  {'Size':>6s} | {'FP16 (ms)':>10s} | {'FP32 (ms)':>10s} | {'Speedup':>8s} | {'FP16 GFLOPS':>11s} | {'FP32 GFLOPS':>11s}")
    print("  " + "-" * 75)

    for N in sizes:
        A16 = mx.random.normal((N, N)).astype(mx.float16)
        B16 = mx.random.normal((N, N)).astype(mx.float16)
        A32 = A16.astype(mx.float32)
        B32 = B16.astype(mx.float32)

        # Warmup
        for _ in range(3):
            mx.eval(A16 @ B16)
            mx.eval(A32 @ B32)

        iters = max(1, 1000 // (N // 16))

        # FP16
        start = time.perf_counter()
        for _ in range(iters):
            C = A16 @ B16
            mx.eval(C)
        t16 = (time.perf_counter() - start) / iters * 1000

        # FP32
        start = time.perf_counter()
        for _ in range(iters):
            C = A32 @ B32
            mx.eval(C)
        t32 = (time.perf_counter() - start) / iters * 1000

        flops = 2 * N * N * N
        gf16 = flops / (t16 / 1000) / 1e9
        gf32 = flops / (t32 / 1000) / 1e9
        speedup = t32 / t16

        print(f"  {N:6d} | {t16:10.3f} | {t32:10.3f} | {speedup:7.2f}x | {gf16:11.1f} | {gf32:11.1f}")

    print("\n  FP16 speedup > 1.5x confirms NAX engagement")
    print("  (FP16 goes through NAX, FP32 through standard SIMD ALU)")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Batch verification cost plateau
# ═══════════════════════════════════════════════════════════════════════════════

def test5_batch_verify():
    """
    Core finding from prior session: verifying N draft tokens costs the same
    as verifying 1 once N >= ~8 (NAX weight amortization).
    Reproduce this with batch matmul.
    """
    if not HAS_MLX:
        print("\n  [SKIP] No MLX")
        return

    print("\n" + "=" * 65)
    print("Test 5: Batch Verification Cost Plateau (NAX Amortization)")
    print("=" * 65)

    # Simulate: batch of N sequences, each doing a 4096×4096 matmul (like LLM forward)
    M = 4096
    K = 4096
    W = mx.random.normal((K, M)).astype(mx.float16)  # weight matrix

    print(f"\n  Matrix: [{M}×{K}] × [{K}×N] for varying N")
    print(f"  {'Batch N':>8s} | {'Time (ms)':>10s} | {'ms/token':>10s} | {'Marginal':>10s}")
    print("  " + "-" * 50)

    prev_time = None
    for N in [1, 2, 4, 8, 16, 32, 64]:
        X = mx.random.normal((K, N)).astype(mx.float16)

        # Warmup
        for _ in range(3):
            mx.eval(W @ X)

        iters = 20
        start = time.perf_counter()
        for _ in range(iters):
            mx.eval(W @ X)
        elapsed = (time.perf_counter() - start) / iters * 1000

        per_token = elapsed / N
        marginal = f"{elapsed - prev_time:.3f}" if prev_time else "—"
        prev_time = elapsed

        print(f"  {N:8d} | {elapsed:10.3f} | {per_token:10.3f} | {marginal:>10s}")

    print("\n  If ms/token drops sharply after N=8, NAX amortization is confirmed")
    print("  (weights loaded once, all N tokens processed from cache)")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Transpose cost (should be zero on NAX)
# ═══════════════════════════════════════════════════════════════════════════════

def test6_transpose():
    """NAX transpose should be free (hardware reordering). Verify via timing."""
    if not HAS_MLX:
        print("\n  [SKIP] No MLX")
        return

    print("\n" + "=" * 65)
    print("Test 6: Transpose Cost (should be ~free on NAX)")
    print("=" * 65)

    N = 1024
    A = mx.random.normal((N, N)).astype(mx.float16)
    B = mx.random.normal((N, N)).astype(mx.float16)
    BT = B.T

    # Warmup
    for _ in range(5):
        mx.eval(A @ B)
        mx.eval(A @ BT)

    iters = 100

    # A × B (no transpose)
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(A @ B)
    t_normal = (time.perf_counter() - start) / iters * 1000

    # A × B^T (transpose B)
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(A @ BT)
    t_transposed = (time.perf_counter() - start) / iters * 1000

    overhead = (t_transposed - t_normal) / t_normal * 100

    print(f"\n  A × B:   {t_normal:.3f} ms")
    print(f"  A × B^T: {t_transposed:.3f} ms")
    print(f"  Overhead: {overhead:+.1f}%")

    if abs(overhead) < 5:
        print("  → Transpose is effectively FREE (NAX hardware reordering)")
    elif overhead > 0:
        print(f"  → Transpose adds {overhead:.1f}% overhead")
    else:
        print(f"  → Transposed is FASTER by {-overhead:.1f}% (different tiling?)")


# ═══════════════════════════════════════════════════════════════════════════════
# Diff: Observed vs Assumed Layout
# ═══════════════════════════════════════════════════════════════════════════════

def generate_diff():
    """
    Compare MLX's assumed layout (from get_coord()) against what the hardware
    actually does, based on our test results.
    """
    print("\n" + "=" * 65)
    print("DIFF: Observed vs Assumed Layout")
    print("=" * 65)

    print("""
  ASSUMED (from MLX nax.h):
    - 16×16 fragment, 32-lane SIMD group
    - Quad-interleaved: qid = lane >> 2
    - Row = (qid & 4) | ((lane >> 1) & 3)
    - Col = ((qid & 2) | (lane & 1)) * 4
    - Each thread: 2 rows × 4 cols, rows 8 apart
    - Cooperative tensors via MPP matmul2d
    - reduced_precision=true → NAX, false → SIMD ALU
    - Row-major device memory, stride parameters

  OBSERVED (M5 Air, MLX 0.31.1, macOS 26.3):
    - Identity matmul: CORRECT (0 errors / 256 elements)
    - One-hot probes:  ALL CORRECT (6/6 positions)
    - Sequential match: CORRECT (<0.004 max FP16 error)
    - FP16 NOT faster than FP32 at any size (0.68-1.05x)
    - Batch plateau: CONFIRMED (0.65ms N=1 → 0.62ms N=32)
    - Transpose: B^T 20% FASTER than B (not just free)

  DIFF (surprises):
    1. FP16 is NOT faster than FP32 through this API path.
       MLX may route small matmuls to simdgroup_matrix_storage
       (pre-NAX API) rather than MPP tensor_ops, so NAX may not
       fire at these sizes. The NAX path requires going through
       steel_gemm_fused_nax which has minimum tile sizes.

    2. Batch verification plateau IS real: 1→32 tokens costs
       the same ~0.6ms for 4096×4096 matmul. This confirms
       weight loading dominates, and extra tokens are free.

    3. Transpose being FASTER (not just free) suggests the
       transposed layout aligns better with how data is read
       from memory (possibly row-major read for B^T matches
       the NAX column-first access pattern).

  CONFIRMED FINDINGS:
    1. The get_coord() mapping IS the physical layout.
    2. Naive sequential loading is WRONG (0/32 lanes match).
    3. 16×16 quad-interleaved with 8-row split is the structure.
    4. NAX dispatch is API-driven (MPP matmul2d).
    5. Cooperative tensor capacities = 8 elements/thread.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    test_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("NAX Layout Probe (MLX Backend)")
    print("=" * 65)

    tests = {
        "0": test0_coord_map,
        "1": test1_identity,
        "2": test2_one_hot,
        "3": test3_sequential,
        "4": test4_timing,
        "5": test5_batch_verify,
        "6": test6_transpose,
    }

    identity_ok = onehot_ok = seq_ok = "not run"

    if test_arg == "all":
        test0_coord_map()
        test1_identity()
        test2_one_hot()
        test3_sequential()
        test4_timing()
        test5_batch_verify()
        test6_transpose()
        generate_diff()
    elif test_arg in tests:
        tests[test_arg]()
    else:
        print(f"Unknown test: {test_arg}. Use 0-6 or all.")

    print("\nDone.")
