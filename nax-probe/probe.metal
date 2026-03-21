// NAX Layout Probe — Metal Compute Shaders
// Writes known patterns through NAX-dispatched ops and reads back physical layout.
// Requires macOS 26+ with Metal 4 and MetalPerformancePrimitives.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// ============================================================================
// Test 0: get_coord() mapping — reproduce MLX's thread-to-element mapping
// Each thread writes its lane_id and computed (row, col) to output buffer.
// Output: 32 entries × 4 values each: [lane_id, row, col_base, elem_index_0..7]
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(32)]]
void probe_coord_map(
    device float* out [[buffer(0)]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    // Reproduce MLX BaseNAXFrag::get_coord()
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));           // row
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;             // col base

    // Each thread owns 8 elements: 2 rows × 4 cols, rows separated by 8
    // Row 0: fm, cols fn..fn+3
    // Row 1: fm+8, cols fn..fn+3
    uint base = simd_lane_id * 12;
    out[base + 0] = float(simd_lane_id);
    out[base + 1] = float(fm);       // row 0
    out[base + 2] = float(fn);       // col base
    out[base + 3] = float(fm + 8);   // row 1

    // Write the 8 (row,col) pairs this thread owns
    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            short c = fn + j;
            // Element index in the 16×16 matrix
            out[base + 4 + i * 4 + j] = float(r * 16 + c);
        }
    }
}

// ============================================================================
// Test 1: Identity matmul — A=I(16), B=sequential → C should equal B
// Verifies NAX matmul produces correct results and reveals any permutation.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(32)]]
void probe_identity_matmul(
    const device half* A [[buffer(0)]],   // 16×16 identity
    const device half* B [[buffer(1)]],   // 16×16 sequential (0..255)
    device half* C       [[buffer(2)]],   // 16×16 output
    device float* debug  [[buffer(3)]],   // per-thread debug: what went in/out of cooperative tensors
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    // NAX 16×16×16 matmul
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 16, 16,
        false, false,
        true,   // reduced_precision → NAX path
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a = op.get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = op.get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    // Load using MLX's get_coord() mapping
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;

    // Load A fragment (row-major, stride=16)
    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            ct_a[i * 4 + j] = A[r * 16 + fn + j];
        }
    }

    // Load B fragment
    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            ct_b[i * 4 + j] = B[r * 16 + fn + j];
        }
    }

    // Zero accumulator
    for (int i = 0; i < ct_c.get_capacity(); i++) {
        ct_c[i] = 0.0f;
    }

    // NAX matmul
    op.run(ct_a, ct_b, ct_c);

    // Store C back using same coord mapping
    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            C[r * 16 + fn + j] = half(ct_c[i * 4 + j]);
        }
    }

    // Debug: dump what each thread's cooperative tensor slots held
    uint dbase = simd_lane_id * 32;
    for (int i = 0; i < 8; i++) {
        debug[dbase + i]      = float(ct_a[i]);       // A input
        debug[dbase + 8 + i]  = float(ct_b[i]);       // B input
        debug[dbase + 16 + i] = float(ct_c[i]);       // C output
    }
    debug[dbase + 24] = float(fm);
    debug[dbase + 25] = float(fn);
    debug[dbase + 26] = float(simd_lane_id);
    debug[dbase + 27] = float(ct_a.get_capacity());
    debug[dbase + 28] = float(ct_b.get_capacity());
    debug[dbase + 29] = float(ct_c.get_capacity());
}

// ============================================================================
// Test 2: Naive load vs coord load — load sequentially into cooperative tensor
// slots, run matmul, compare output to coord-loaded version.
// Reveals how the hardware interprets sequential register slots.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(32)]]
void probe_naive_load(
    const device half* A [[buffer(0)]],   // 16×16 identity
    const device half* B [[buffer(1)]],   // 16×16 sequential
    device half* C       [[buffer(2)]],   // output with naive load
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 16, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a = op.get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = op.get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    // NAIVE load: sequential 8 elements per thread, no coord mapping
    for (int i = 0; i < 8; i++) {
        ct_a[i] = A[simd_lane_id * 8 + i];
        ct_b[i] = B[simd_lane_id * 8 + i];
    }

    for (int i = 0; i < ct_c.get_capacity(); i++) ct_c[i] = 0.0f;

    op.run(ct_a, ct_b, ct_c);

    // Store back naively (same sequential mapping)
    for (int i = 0; i < 8; i++) {
        C[simd_lane_id * 8 + i] = half(ct_c[i]);
    }
}

// ============================================================================
// Test 3: One-hot probe — set exactly one element to 1.0 in A, all 1s in B
// Reveals which output elements light up for each input position.
// Run once per input position (controlled by buffer(3) = position index).
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(32)]]
void probe_one_hot(
    const device half* A       [[buffer(0)]],   // 16×16, one element = 1.0
    const device half* B       [[buffer(1)]],   // 16×16, all 1.0
    device half* C             [[buffer(2)]],   // 16×16 output
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 16, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a = op.get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = op.get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    // Load using coord mapping
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;

    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            ct_a[i * 4 + j] = A[r * 16 + fn + j];
            ct_b[i * 4 + j] = B[r * 16 + fn + j];
        }
    }

    for (int i = 0; i < ct_c.get_capacity(); i++) ct_c[i] = 0.0f;

    op.run(ct_a, ct_b, ct_c);

    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            C[r * 16 + fn + j] = half(ct_c[i * 4 + j]);
        }
    }
}

// ============================================================================
// Test 4: FP32 control — same matmul but reduced_precision=false → SIMD ALUs
// Compare timing and output vs FP16 NAX path.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(32)]]
void probe_fp32_control(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 16, 16, false, false,
        false,   // NOT reduced_precision → standard SIMD ALU path
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a = op.get_left_input_cooperative_tensor<float, float, float>();
    auto ct_b = op.get_right_input_cooperative_tensor<float, float, float>();
    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;

    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            ct_a[i * 4 + j] = A[r * 16 + fn + j];
            ct_b[i * 4 + j] = B[r * 16 + fn + j];
        }
    }

    for (int i = 0; i < ct_c.get_capacity(); i++) ct_c[i] = 0.0f;

    op.run(ct_a, ct_b, ct_c);

    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            C[r * 16 + fn + j] = ct_c[i * 4 + j];
        }
    }
}

// ============================================================================
// Test 5: Throughput — run N iterations of 16×16 NAX matmul in a tight loop.
// Host measures wall time to derive ops/sec.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(32)]]
void probe_throughput(
    const device half* A   [[buffer(0)]],
    const device half* B   [[buffer(1)]],
    device half* C         [[buffer(2)]],
    const device uint& N   [[buffer(3)]],   // iteration count
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 16, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a = op.get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = op.get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;

    // Load once
    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            ct_a[i * 4 + j] = A[r * 16 + fn + j];
            ct_b[i * 4 + j] = B[r * 16 + fn + j];
        }
    }
    for (int i = 0; i < ct_c.get_capacity(); i++) ct_c[i] = 0.0f;

    // Tight compute loop
    for (uint iter = 0; iter < N; iter++) {
        op.run(ct_a, ct_b, ct_c);
    }

    // Store to prevent dead code elimination
    for (short i = 0; i < 2; i++) {
        short r = fm + i * 8;
        for (short j = 0; j < 4; j++) {
            C[r * 16 + fn + j] = half(ct_c[i * 4 + j]);
        }
    }
}
