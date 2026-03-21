#!/usr/bin/env python3
"""
NAX Layout Analyzer — Compares observed NAX behavior against assumed layout.

Reads the get_coord() mapping from MLX source and computes:
1. The thread-to-element ownership map (assumed layout)
2. Whether naive sequential loading matches coord loading (observed)
3. Diff between MLX's assumed layout and hardware's actual behavior

Run after main.swift dumps results.
"""

import json
import sys
import numpy as np


def compute_mlx_coord_map():
    """Reproduce MLX BaseNAXFrag::get_coord() in Python."""
    ownership = np.full((16, 16), -1, dtype=int)
    thread_elements = {}

    for lane in range(32):
        qid = lane >> 2
        fm = ((qid & 4) | ((lane >> 1) & 3))         # row
        fn = ((qid & 2) | (lane & 1)) * 4             # col base

        elements = []
        for i in range(2):  # kElemRows = 2
            r = fm + i * 8  # kElemRowsJump = 8
            for j in range(4):  # kElemCols = 4
                c = fn + j
                ownership[r, c] = lane
                elements.append((r, c))

        thread_elements[lane] = elements

    return ownership, thread_elements


def print_ownership_map(ownership, title="Thread Ownership Map"):
    """Print which SIMD lane owns each element of the 16×16 fragment."""
    print(f"\n{title}")
    print("       " + " ".join(f"{c:3d}" for c in range(16)))
    print("      " + "-" * 64)
    for r in range(16):
        row = " ".join(f"{ownership[r, c]:3d}" for c in range(16))
        print(f"  r{r:02d} | {row}")


def analyze_coord_structure(thread_elements):
    """Analyze the structure of the coord mapping."""
    print("\n== Structural Analysis ==")

    # Group threads by quad
    for quad in range(8):
        lanes = list(range(quad * 4, quad * 4 + 4))
        rows = set()
        cols = set()
        for lane in lanes:
            for (r, c) in thread_elements[lane]:
                rows.add(r)
                cols.add(c)
        print(f"  Quad {quad} (lanes {lanes[0]}-{lanes[3]}): "
              f"rows={sorted(rows)}, cols={sorted(cols)}")

    # Check: do all threads in a quad share the same rows?
    print("\n  Pattern analysis:")

    # Row grouping
    row_to_lanes = {}
    for lane, elems in thread_elements.items():
        rows = tuple(sorted(set(r for r, c in elems)))
        row_to_lanes.setdefault(rows, []).append(lane)

    print("  Row groups (threads sharing same rows):")
    for rows, lanes in sorted(row_to_lanes.items()):
        print(f"    rows {rows}: lanes {sorted(lanes)}")

    # Column grouping
    col_to_lanes = {}
    for lane, elems in thread_elements.items():
        cols = tuple(sorted(set(c for r, c in elems)))
        col_to_lanes.setdefault(cols, []).append(lane)

    print("  Column groups (threads sharing same cols):")
    for cols, lanes in sorted(col_to_lanes.items()):
        print(f"    cols {cols}: lanes {sorted(lanes)}")


def compute_expected_matmul():
    """Compute expected I×B where B=sequential."""
    I = np.eye(16, dtype=np.float32)
    B = np.arange(256, dtype=np.float32).reshape(16, 16)
    C = I @ B
    return C


def diff_observed_vs_expected(observed_file=None):
    """If we have observed output, compare against expected."""
    expected = compute_expected_matmul()

    if observed_file:
        try:
            with open(observed_file) as f:
                data = json.load(f)
            observed = np.array(data).reshape(16, 16)

            diff = observed - expected
            print("\n== Observed vs Expected Diff ==")
            print(f"  Max absolute error: {np.max(np.abs(diff)):.6f}")
            print(f"  Mean absolute error: {np.mean(np.abs(diff)):.6f}")
            print(f"  Positions with >0.5 error: {np.sum(np.abs(diff) > 0.5)}/256")

            if np.max(np.abs(diff)) > 0.5:
                print("\n  Permutation detected! Non-matching positions:")
                for r in range(16):
                    for c in range(16):
                        if abs(diff[r, c]) > 0.5:
                            print(f"    [{r},{c}]: expected={expected[r,c]:.0f}, "
                                  f"observed={observed[r,c]:.0f}")
        except Exception as e:
            print(f"  Could not load observed data: {e}")
    else:
        print("\n  (No observed data file provided — run main.swift first)")
        print("  Expected I×B result (first 4 rows):")
        for r in range(4):
            print(f"    [{', '.join(f'{v:.0f}' for v in expected[r])}]")


def compare_naive_vs_coord():
    """
    Analyze what happens when data is loaded sequentially vs via get_coord().

    Sequential: thread N loads elements [N*8 .. N*8+7] from flat memory
    Coord: thread N loads from specific (row, col) positions

    If NAX hardware expects coord-mapped layout, naive loading will produce
    garbage because the cooperative tensor slots map to specific matrix positions.
    """
    print("\n== Naive vs Coord Loading Analysis ==")

    _, thread_elements = compute_mlx_coord_map()

    # What naive loading puts in each slot
    print("  Naive: thread K loads flat positions [K*8 .. K*8+7]")
    print("  Coord: thread K loads from get_coord() mapped positions")
    print()

    # Show first 4 threads
    for lane in range(4):
        naive_positions = list(range(lane * 8, lane * 8 + 8))
        coord_positions = [r * 16 + c for r, c in thread_elements[lane]]
        match = naive_positions == coord_positions
        print(f"  Lane {lane}:")
        print(f"    Naive: {naive_positions}")
        print(f"    Coord: {coord_positions}")
        print(f"    Match: {match}")

    # Count total matches
    total_match = 0
    for lane in range(32):
        naive = list(range(lane * 8, lane * 8 + 8))
        coord = [r * 16 + c for r, c in thread_elements[lane]]
        if naive == coord:
            total_match += 1

    print(f"\n  Threads with matching naive/coord positions: {total_match}/32")
    if total_match < 32:
        print("  → Naive loading IS WRONG for NAX — get_coord() mapping is required")
    else:
        print("  → Naive loading happens to match (would be very surprising)")


def mlx_source_layout_summary():
    """Summarize what MLX assumes about NAX layout."""
    print("\n== MLX Source Layout Assumptions ==")
    print("  File: mlx/backend/metal/kernels/steel/gemm/nax.h")
    print()
    print("  Fragment: 16×16 elements, 32 threads (1 SIMD group)")
    print("  Per thread: 8 elements = 2 rows × 4 cols")
    print("  Row jump: 8 (rows 0-7 in first half, 8-15 in second)")
    print()
    print("  get_coord() mapping:")
    print("    qid = lane >> 2  (quad index, 0-7)")
    print("    row = (qid & 4) | ((lane >> 1) & 3)")
    print("    col = ((qid & 2) | (lane & 1)) * 4")
    print()
    print("  Cooperative tensor packing:")
    print("    ct[TK * mm + kk) * 8 + i] = frag_at(fi, fj)[i]")
    print("    Where i=0..7 maps to the 8 elements this thread owns")
    print()
    print("  NAX dispatch: mpp::tensor_ops::matmul2d with reduced_precision=true")
    print("  Non-NAX: same API with reduced_precision=false → SIMD ALU fallback")
    print()
    print("  Threadgroup padding: BK + 16/sizeof(T) columns per row")
    print("    float32: +4 cols, float16: +8 cols (bank conflict avoidance)")


def main():
    print("NAX Layout Analysis")
    print("=" * 55)

    # 1. Compute and display the MLX coord map
    ownership, thread_elements = compute_mlx_coord_map()
    print_ownership_map(ownership, "MLX get_coord() Ownership Map (assumed layout)")

    # 2. Structural analysis
    analyze_coord_structure(thread_elements)

    # 3. Naive vs coord comparison
    compare_naive_vs_coord()

    # 4. MLX source summary
    mlx_source_layout_summary()

    # 5. Diff against observed (if provided)
    observed_file = sys.argv[1] if len(sys.argv) > 1 else None
    diff_observed_vs_expected(observed_file)

    print("\nDone.")


if __name__ == "__main__":
    main()
