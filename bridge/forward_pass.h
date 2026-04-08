// forward_pass.h — C forward pass for ANE transformer inference
// Eliminates Python from the hot loop. All CPU ops via Accelerate SIMD.
// ANE dispatch via ane_bridge.h.

#ifndef FORWARD_PASS_H
#define FORWARD_PASS_H

#include "ane_bridge.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Model Configuration ──────────────────────────────────────────────

typedef struct {
    int dim;           // Hidden dimension (e.g. 1024)
    int n_heads;       // Number of attention heads (e.g. 16)
    int n_kv_heads;    // Number of KV heads for GQA (e.g. 8)
    int head_dim;      // Per-head dimension (e.g. 128)
    int hidden_dim;    // FFN intermediate size (e.g. 2816)
    int vocab_size;    // Vocabulary size (e.g. 151936)
    int n_layers;      // Number of transformer layers (e.g. 28)
    int max_seq;       // Maximum sequence length for KV cache
    int ane_spatial;   // ANE spatial padding (16)
    float rope_theta;  // RoPE base frequency
} FPModelConfig;

// ── Opaque model handle ──────────────────────────────────────────────

typedef struct ForwardModel ForwardModel;

// ── Lifecycle ────────────────────────────────────────────────────────

// Create model with given config. Allocates all buffers.
ForwardModel *forward_model_create(const FPModelConfig *config);

// Free model and all resources (does NOT free ANE kernels — Python owns those)
void forward_model_free(ForwardModel *m);

// ── Weight Setup (call once after create) ────────────────────────────

// Set embedding table. Data is COPIED. embed_w: [vocab_size * dim] row-major float32.
void forward_model_set_embed(ForwardModel *m, const float *embed_w);

// Set final RMSNorm weights. w: [dim] float32.
void forward_model_set_final_norm(ForwardModel *m, const float *w);

// Set per-layer CPU weights: attn_norm[dim], ffn_norm[dim], q_norm[head_dim], k_norm[head_dim]
void forward_model_set_layer_weights(ForwardModel *m, int layer,
    const float *attn_norm, const float *ffn_norm,
    const float *q_norm, const float *k_norm);

// Set per-layer ANE kernel handles (pointers, not copied — Python retains ownership)
// Unfused mode: 7 separate kernels per layer
void forward_model_set_layer_kernels(ForwardModel *m, int layer,
    ANEKernelHandle *q, ANEKernelHandle *k, ANEKernelHandle *v,
    ANEKernelHandle *o, ANEKernelHandle *gate, ANEKernelHandle *up,
    ANEKernelHandle *down);

// Fused mode: 4 kernels per layer (qkv, o, gate_up, down)
// qkv output = [q_dim + kv_dim + kv_dim], gate_up output = [hidden_dim * 2]
void forward_model_set_layer_kernels_fused(ForwardModel *m, int layer,
    ANEKernelHandle *qkv, ANEKernelHandle *o,
    ANEKernelHandle *gate_up, ANEKernelHandle *down);

// Add a classifier chunk kernel. Call in order (chunk 0, 1, 2, ...).
void forward_model_add_cls_kernel(ForwardModel *m, ANEKernelHandle *kernel, int out_channels);

// ── Inference ────────────────────────────────────────────────────────

// Reset KV caches to zero (call before new generation)
void forward_model_reset_cache(ForwardModel *m);

// Run one token through the full transformer. Returns pointer to logits[vocab_size].
// The returned pointer is valid until the next call to forward_token.
const float *forward_model_forward_token(ForwardModel *m, int token_id, int pos);

// Get timing breakdown from last forward_token call (milliseconds)
typedef struct {
    double total_ms;
    double ane_ms;       // All ANE dispatches (pack + eval + extract)
    double ane_pack_ms;  // memset + strided pack into IOSurface
    double ane_eval_ms;  // ane_bridge_eval() call only
    double ane_read_ms;  // strided extract from IOSurface
    double rmsnorm_ms;   // All RMSNorm ops
    double rope_ms;      // RoPE
    double qknorm_ms;    // QK-norm
    double attention_ms; // Attention (scores + softmax + weighted sum)
    double silu_ms;      // SiLU activation
    double embed_ms;     // Embedding lookup
    double classify_ms;  // Classifier (ANE)
} ForwardTiming;

void forward_model_get_timing(const ForwardModel *m, ForwardTiming *out);

#ifdef __cplusplus
}
#endif

#endif // FORWARD_PASS_H
