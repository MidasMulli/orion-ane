# NAX Layout Probe — Findings

## M5 Air (10 GPU cores, 153.6 GB/s LPDDR5X)

### 1. NAX is Confirmed Active

**Peak throughput:**
- FP16: **12.19 TFLOPS** at 3072×3072 (via `steel_gemm_fused_nax`)
- FP32: **7.79 TFLOPS** at 2048×2048 (via `steel_gemm_fused_nax_float32`)
- FP16/FP32 ratio: **1.6x**

**Theoretical SIMD-only ceiling: ~3.6 TFLOPS FP16**
- FP16 exceeds this by 3.4× — NAX tensor units are definitively engaged
- FP32 exceeds its theoretical ceiling too — FP32 NAX kernels exist and fire

### 2. Layout Structure (get_coord() Mapping)

16×16 fragment, 32-lane SIMD group, 8 elements per thread:

```
qid = lane >> 2           (quad index 0-7)
row = (qid & 4) | ((lane >> 1) & 3)
col = ((qid & 2) | (lane & 1)) * 4
Each thread: [row, col..col+3] AND [row+8, col..col+3]
```

**Quad structure:**
- 8 quads of 4 threads each
- Each quad covers 4 rows × 8 columns
- Upper/lower 8-row halves share the same thread
- Naive sequential loading is WRONG (0/32 lanes match)

### 3. MLX Kernel Inventory (metallib)

| Category | Count | Notes |
|----------|-------|-------|
| NAX fused GEMM | 72 | FP16, BF16, FP32 |
| NAX splitk GEMM | 48+ | For non-square matrices |
| NAX quantized (affine) | 324 | 2/3/4/5/6/8-bit |
| NAX FP4/FP8 quantized | 48 | nvfp4, mxfp8 |
| NAX total | **720** | All pre-compiled |
| Old (non-NAX) GEMM | 96 | Still in metallib |

**NAX block configs:**
- Dense: BM=64/128, BN=64/128, BK=64/256/512
- Quantized: BM=64, BN=64, BK=64 (all configs)
- Subtile: UM=16, UN=32, UK=16

### 4. Batch Verification Plateau

**Raw matmul (3584×3584, FP16 — Qwen3.5-9B hidden dim):**

| Tokens | Total (ms) | ms/token | vs N=1 |
|--------|-----------|----------|--------|
| 1 | 0.45 | 0.45 | 1.00x |
| 4 | 0.58 | 0.15 | 1.30x |
| 8 | 0.62 | 0.08 | 1.39x |
| 16 | 0.71 | 0.04 | 1.57x |
| 32 | 0.68 | 0.02 | 1.51x |

**Plateau confirmed:** 32 tokens costs only 1.5x what 1 token costs.

**Full model (Qwen3.5-9B, 4-bit quantized):**

| Tokens | Total (ms) | ms/token | vs N=1 |
|--------|-----------|----------|--------|
| 1 | 79 | 79 | 1.00x |
| 2 | 107 | 54 | 1.35x |
| 8 | 281 | 35 | 3.54x |
| 32 | 256 | 8 | 3.23x |

**Weaker plateau** due to GatedDeltaNet SSM layers (31% of params, sequential).

### 5. GatedDeltaNet Impact

Qwen3.5-9B architecture per layer:
- GatedDeltaNet: 8.45M params (31%) — **sequential**, each token depends on prior state
- FFN: 18.87M params (69%) — **parallelizable**, benefits from NAX batch amortization

The SSM layers break the clean batch verification plateau. For spec decode:
- Raw matmul amortization: near-perfect (1.5x at N=32)
- Full model amortization: degraded (3.2x at N=32)
- **Pure attention models (Llama 70B) would show much flatter plateau**

### 6. Transpose Behavior

Variable by size — not consistently faster in either direction:

| Size | NN (ms) | NT (ms) | TN (ms) | TT (ms) |
|------|---------|---------|---------|---------|
| 2048 | 1.87 | 1.52 | 1.53 | 1.78 |
| 4096 | 12.6 | 12.7 | 13.1 | 13.9 |

At 2048: NT 19% faster (tiling alignment). At 4096: differences <5%.

### 7. Metal Toolchain Status

`probe.metal` (direct NAX shader) blocked by beta toolchain bug:
- `__HAVE_TENSOR__` guard in MPP headers requires tensor module
- `metal_tensor` module has `index_sequence` typedef missing in `extents.h`
- Workaround: use MLX's pre-compiled metallib (all 720 NAX kernels work)
- Fix expected in next Xcode beta

### 8. Quantized 4-bit NAX Batch Scaling (LLM Inference Path)

These use `affine_qmm_nax` kernels — the actual kernels fired during Qwen3.5-9B inference.

**QKV projection (3584 → 10752, 4-bit):**

| Tokens | Total (ms) | vs N=1 |
|--------|-----------|--------|
| 1 | 2.10 | 1.00x |
| 8 | 2.68 | 1.28x |
| 32 | 2.39 | **1.14x** |

**FFN up (3584 → 18944, 4-bit):**

| Tokens | Total (ms) | vs N=1 |
|--------|-----------|--------|
| 1 | 2.97 | 1.00x |
| 8 | 3.80 | 1.28x |
| 32 | 3.44 | **1.16x** |

**Near-perfect amortization:** 32 tokens costs only 14-16% more than 1 token
for QKV and FFN up projections. This is the hardware evidence for spec decode:
verifying 32 draft tokens through quantized weight matrices is essentially free
once the weights are loaded.

FFN down (18944 → 3584) shows weaker amortization (2.05x at N=32) — likely
because the input dimension is much larger than output, changing the memory
access pattern.

### Key Takeaways for Spec Decode Paper

1. **NAX is real and measurable:** 12 TFLOPS FP16, 3.4× SIMD-only ceiling
2. **Batch verification plateau is real for matmuls** — 32 tokens ≈ 1.5x cost of 1
3. **SSM layers break the plateau** — Qwen3.5's GDN adds sequential cost per token
4. **Pure attention models (Pro target) will show the clean plateau**
5. **The layout is quad-interleaved 16×16** — matched to NAX tensor unit hardware
6. **720 pre-compiled NAX kernels** cover all quantization formats MLX supports
