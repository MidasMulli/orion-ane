#!/bin/bash
# NAX Layout Probe — build and run
# Requires macOS 26+ with MLX installed in ~/.mlx-env

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== NAX Layout Probe ==="
echo

# Primary: MLX-based probe (uses pre-compiled NAX kernels)
echo "1. Running MLX-based NAX probe..."
echo "   (Uses MLX's steel_gemm_fused_nax via Python)"
echo
~/.mlx-env/bin/python3 probe_mlx.py all 2>&1

echo
echo "2. Running Python structural analysis..."
python3 analyze.py

echo
echo "=== Done ==="
echo
echo "Note: probe.metal + main.swift require Metal Toolchain with MPP support."
echo "The Metal 4 beta toolchain has a bug in extents.h (index_sequence missing)."
echo "Once fixed, run: swift -framework Metal -framework Accelerate main.swift"
