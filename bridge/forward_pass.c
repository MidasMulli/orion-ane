// forward_pass.c — C forward pass for ANE transformer inference
// Zero Python in the hot loop. CPU ops via Accelerate vDSP/cblas.
// ANE dispatch via ane_bridge.h.

#include "forward_pass.h"
#include "ane_bridge.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>

// ── Internal Structures ──────────────────────────────────────────────

typedef struct {
    float *attn_norm;   // [dim]
    float *ffn_norm;    // [dim]
    float *q_norm;      // [head_dim]
    float *k_norm;      // [head_dim]
} LayerWeights;

typedef struct {
    // Unfused mode (7 kernels)
    ANEKernelHandle *q;
    ANEKernelHandle *k;
    ANEKernelHandle *v;
    ANEKernelHandle *o;
    ANEKernelHandle *gate;
    ANEKernelHandle *up;
    ANEKernelHandle *down;
    // Fused mode (4 kernels) — qkv replaces q/k/v, gate_up replaces gate/up
    ANEKernelHandle *qkv;      // output: [q_dim + kv_dim + kv_dim]
    ANEKernelHandle *gate_up;  // output: [hidden_dim * 2]
    bool fused;
} LayerKernels;

typedef struct {
    ANEKernelHandle *kernel;
    int out_channels;
} ClassifierChunk;

#define MAX_CLS_CHUNKS 32

struct ForwardModel {
    FPModelConfig cfg;

    // Weights
    float *embed_w;         // [vocab_size * dim]
    float *final_norm_w;    // [dim]
    LayerWeights *lw;       // [n_layers]
    LayerKernels *lk;       // [n_layers]

    // Classifier
    ClassifierChunk cls_chunks[MAX_CLS_CHUNKS];
    int n_cls_chunks;

    // KV cache: k_caches[layer][pos * kv_dim + h * head_dim + d]
    float **k_caches;       // [n_layers] -> [max_seq * kv_dim]
    float **v_caches;       // [n_layers] -> [max_seq * kv_dim]

    // Precomputed
    float *rope_freqs;      // [head_dim / 2]

    // Scratch buffers (pre-allocated, zero malloc in hot path)
    float *x;               // [dim]
    float *xn;              // [dim]
    float *q_buf;           // [n_heads * head_dim]
    float *k_buf;           // [n_kv_heads * head_dim]
    float *v_buf;           // [n_kv_heads * head_dim]
    float *attn_out;        // [n_heads * head_dim]
    float *gate_buf;        // [hidden_dim]
    float *up_buf;          // [hidden_dim]
    float *down_buf;        // [dim]
    float *silu_buf;        // [hidden_dim]
    float *logits;          // [vocab_size]
    float *ane_in;          // [max_ane_dim * spatial]
    float *ane_out;         // [max_ane_dim * spatial]
    float *scores;          // [max_seq] for attention scores per head

    // Timing
    ForwardTiming timing;

    // Mach timing
    mach_timebase_info_data_t timebase;
};

// ── Timing Helpers ───────────────────────────────────────────────────

static inline double now_ms(const ForwardModel *m) {
    uint64_t t = mach_absolute_time();
    return (double)t * m->timebase.numer / m->timebase.denom / 1e6;
}

// ── Core Ops ─────────────────────────────────────────────────────────

static void rmsnorm(float *out, const float *x, const float *w, int dim) {
    // ss = mean(x * x)
    float ss;
    vDSP_dotpr(x, 1, x, 1, &ss, dim);
    ss = ss / (float)dim + 1e-6f;
    float inv = 1.0f / sqrtf(ss);
    // out = x * inv * w
    vDSP_vsmul(x, 1, &inv, out, 1, dim);
    vDSP_vmul(out, 1, w, 1, out, 1, dim);
}

static void rope(float *x, int pos, int n_heads, int head_dim, const float *freqs) {
    int half = head_dim / 2;
    // Precompute cos/sin for this position (same across all heads)
    // Stack allocate for small sizes (head_dim/2 <= 64 typically)
    float cos_cache[256], sin_cache[256];
    for (int i = 0; i < half; i++) {
        float angle = (float)pos * freqs[i];
        cos_cache[i] = cosf(angle);
        sin_cache[i] = sinf(angle);
    }

    for (int h = 0; h < n_heads; h++) {
        float *xh = x + h * head_dim;
        for (int i = 0; i < half; i++) {
            float x0 = xh[2 * i];
            float x1 = xh[2 * i + 1];
            xh[2 * i]     = x0 * cos_cache[i] - x1 * sin_cache[i];
            xh[2 * i + 1] = x0 * sin_cache[i] + x1 * cos_cache[i];
        }
    }
}

static void qk_norm(float *q, float *k,
                     const float *q_norm_w, const float *k_norm_w,
                     int n_heads, int n_kv_heads, int head_dim) {
    // Vectorized QK-norm: per-head RMSNorm
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        float ss;
        vDSP_dotpr(qh, 1, qh, 1, &ss, head_dim);
        ss = ss / (float)head_dim + 1e-6f;
        float inv = 1.0f / sqrtf(ss);
        vDSP_vsmul(qh, 1, &inv, qh, 1, head_dim);
        vDSP_vmul(qh, 1, q_norm_w, 1, qh, 1, head_dim);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        float ss;
        vDSP_dotpr(kh, 1, kh, 1, &ss, head_dim);
        ss = ss / (float)head_dim + 1e-6f;
        float inv = 1.0f / sqrtf(ss);
        vDSP_vsmul(kh, 1, &inv, kh, 1, head_dim);
        vDSP_vmul(kh, 1, k_norm_w, 1, kh, 1, head_dim);
    }
}

static void attention(ForwardModel *m, const float *q,
                      const float *k_cache, const float *v_cache,
                      float *out, int pos) {
    int hd = m->cfg.head_dim;
    int n_heads = m->cfg.n_heads;
    int n_kv_heads = m->cfg.n_kv_heads;
    int heads_per_kv = n_heads / n_kv_heads;
    int kv_dim = n_kv_heads * hd;
    int seq = pos + 1;
    float scale = 1.0f / sqrtf((float)hd);
    float *scores = m->scores;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        const float *q_h = q + h * hd;
        float *out_h = out + h * hd;

        // Compute attention scores: dot(q_h, k_cache[s, kv_h]) * scale
        float max_score = -1e30f;
        for (int s = 0; s < seq; s++) {
            const float *k_s = k_cache + s * kv_dim + kv_h * hd;
            float dot;
            vDSP_dotpr(q_h, 1, k_s, 1, &dot, hd);
            scores[s] = dot * scale;
            if (scores[s] > max_score) max_score = scores[s];
        }

        // Softmax
        float sum = 0.0f;
        for (int s = 0; s < seq; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum += scores[s];
        }
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(scores, 1, &inv_sum, scores, 1, seq);

        // Weighted sum of V
        memset(out_h, 0, hd * sizeof(float));
        for (int s = 0; s < seq; s++) {
            const float *v_s = v_cache + s * kv_dim + kv_h * hd;
            cblas_saxpy(hd, scores[s], v_s, 1, out_h, 1);
        }
    }
}

static void silu_gate_mul(float *out, const float *gate, const float *up, int dim) {
    // out = silu(gate) * up = (gate / (1 + exp(-gate))) * up
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        // Clip for numerical stability
        if (g < -20.0f) g = -20.0f;
        if (g > 20.0f) g = 20.0f;
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

// ── ANE Linear (zero-copy IOSurface) ─────────────────────────────────

// Run a single ANE kernel — zero-copy, lockless IOSurface access.
// Uses raw base addresses + ARM64 memory barriers instead of IOSurface lock/unlock.
static void ane_linear(ForwardModel *m, ANEKernelHandle *kernel,
                       const float *x, float *out, int in_d, int out_d) {
    int spatial = m->cfg.ane_spatial;
    double t;

    // Pack input directly into IOSurface memory
    t = now_ms(m);
    float *io_in = (float *)ane_bridge_get_input_base(kernel, 0);
    memset(io_in, 0, in_d * spatial * sizeof(float));
    for (int i = 0; i < in_d; i++) {
        io_in[i * spatial] = x[i];
    }
    __asm__ volatile("dsb sy" ::: "memory");
    m->timing.ane_pack_ms += now_ms(m) - t;

    // Dispatch to ANE
    t = now_ms(m);
    ane_bridge_eval(kernel);
    __asm__ volatile("dsb sy" ::: "memory");
    m->timing.ane_eval_ms += now_ms(m) - t;

    // Extract output directly from IOSurface memory
    t = now_ms(m);
    const float *io_out = (const float *)ane_bridge_get_output_base(kernel, 0);
    for (int i = 0; i < out_d; i++) {
        out[i] = io_out[i * spatial];
    }
    m->timing.ane_read_ms += now_ms(m) - t;
}

// Run a fused ANE kernel and split the output into multiple buffers.
// out_bufs: array of output buffer pointers
// out_dims: array of output dimensions per buffer
// n_outs: number of output splits
static void ane_linear_fused(ForwardModel *m, ANEKernelHandle *kernel,
                              const float *x, int in_d, int total_out_d,
                              float **out_bufs, const int *out_dims, int n_outs) {
    int spatial = m->cfg.ane_spatial;
    double t;

    // Pack input
    t = now_ms(m);
    float *io_in = (float *)ane_bridge_get_input_base(kernel, 0);
    memset(io_in, 0, in_d * spatial * sizeof(float));
    for (int i = 0; i < in_d; i++) {
        io_in[i * spatial] = x[i];
    }
    __asm__ volatile("dsb sy" ::: "memory");
    m->timing.ane_pack_ms += now_ms(m) - t;

    // Dispatch
    t = now_ms(m);
    ane_bridge_eval(kernel);
    __asm__ volatile("dsb sy" ::: "memory");
    m->timing.ane_eval_ms += now_ms(m) - t;

    // Extract and split output
    t = now_ms(m);
    const float *io_out = (const float *)ane_bridge_get_output_base(kernel, 0);
    int offset = 0;
    for (int b = 0; b < n_outs; b++) {
        for (int i = 0; i < out_dims[b]; i++) {
            out_bufs[b][i] = io_out[(offset + i) * spatial];
        }
        offset += out_dims[b];
    }
    m->timing.ane_read_ms += now_ms(m) - t;
}

// ── Public API ───────────────────────────────────────────────────────

ForwardModel *forward_model_create(const FPModelConfig *config) {
    if (!config) return NULL;
    ForwardModel *m = (ForwardModel *)calloc(1, sizeof(ForwardModel));
    if (!m) return NULL;

    m->cfg = *config;
    mach_timebase_info(&m->timebase);

    // Use cfg (already copied) for all references
    FPModelConfig *c = &m->cfg;
    int dim = c->dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hd = c->hidden_dim;
    int spatial = c->ane_spatial;
    int max_ane_dim = hd > q_dim ? hd : q_dim;

    // Allocate weight storage
    m->embed_w = NULL;
    m->final_norm_w = (float *)calloc(dim, sizeof(float));
    m->lw = (LayerWeights *)calloc(c->n_layers, sizeof(LayerWeights));
    m->lk = (LayerKernels *)calloc(c->n_layers, sizeof(LayerKernels));

    for (int l = 0; l < c->n_layers; l++) {
        m->lw[l].attn_norm = (float *)calloc(dim, sizeof(float));
        m->lw[l].ffn_norm  = (float *)calloc(dim, sizeof(float));
        m->lw[l].q_norm    = (float *)calloc(c->head_dim, sizeof(float));
        m->lw[l].k_norm    = (float *)calloc(c->head_dim, sizeof(float));
    }

    // KV caches
    m->k_caches = (float **)calloc(c->n_layers, sizeof(float *));
    m->v_caches = (float **)calloc(c->n_layers, sizeof(float *));
    for (int l = 0; l < c->n_layers; l++) {
        m->k_caches[l] = (float *)calloc(c->max_seq * kv_dim, sizeof(float));
        m->v_caches[l] = (float *)calloc(c->max_seq * kv_dim, sizeof(float));
    }

    // Precompute RoPE frequencies
    m->rope_freqs = (float *)malloc((c->head_dim / 2) * sizeof(float));
    for (int i = 0; i < c->head_dim / 2; i++) {
        float exponent = (float)(2 * i) / (float)c->head_dim;
        m->rope_freqs[i] = 1.0f / powf(c->rope_theta, exponent);
    }

    // Scratch buffers
    m->x        = (float *)calloc(dim, sizeof(float));
    m->xn       = (float *)calloc(dim, sizeof(float));
    m->q_buf    = (float *)calloc(q_dim, sizeof(float));
    m->k_buf    = (float *)calloc(kv_dim, sizeof(float));
    m->v_buf    = (float *)calloc(kv_dim, sizeof(float));
    m->attn_out = (float *)calloc(q_dim, sizeof(float));
    m->gate_buf = (float *)calloc(hd, sizeof(float));
    m->up_buf   = (float *)calloc(hd, sizeof(float));
    m->down_buf = (float *)calloc(dim, sizeof(float));
    m->silu_buf = (float *)calloc(hd, sizeof(float));
    m->logits   = (float *)calloc(c->vocab_size, sizeof(float));
    m->ane_in   = (float *)calloc(max_ane_dim * spatial, sizeof(float));
    m->ane_out  = (float *)calloc(max_ane_dim * spatial, sizeof(float));
    m->scores   = (float *)calloc(c->max_seq, sizeof(float));

    m->n_cls_chunks = 0;

    return m;
}

void forward_model_free(ForwardModel *m) {
    if (!m) return;

    free(m->embed_w);
    free(m->final_norm_w);

    for (int l = 0; l < m->cfg.n_layers; l++) {
        free(m->lw[l].attn_norm);
        free(m->lw[l].ffn_norm);
        free(m->lw[l].q_norm);
        free(m->lw[l].k_norm);
        free(m->k_caches[l]);
        free(m->v_caches[l]);
    }
    free(m->lw);
    free(m->lk);
    free(m->k_caches);
    free(m->v_caches);
    free(m->rope_freqs);

    free(m->x);
    free(m->xn);
    free(m->q_buf);
    free(m->k_buf);
    free(m->v_buf);
    free(m->attn_out);
    free(m->gate_buf);
    free(m->up_buf);
    free(m->down_buf);
    free(m->silu_buf);
    free(m->logits);
    free(m->ane_in);
    free(m->ane_out);
    free(m->scores);

    free(m);
}

void forward_model_set_embed(ForwardModel *m, const float *embed_w) {
    size_t bytes = (size_t)m->cfg.vocab_size * m->cfg.dim * sizeof(float);
    if (m->embed_w) free(m->embed_w);
    m->embed_w = (float *)malloc(bytes);
    memcpy(m->embed_w, embed_w, bytes);
}

void forward_model_set_final_norm(ForwardModel *m, const float *w) {
    memcpy(m->final_norm_w, w, m->cfg.dim * sizeof(float));
}

void forward_model_set_layer_weights(ForwardModel *m, int layer,
    const float *attn_norm, const float *ffn_norm,
    const float *q_norm, const float *k_norm)
{
    if (layer < 0 || layer >= m->cfg.n_layers) return;
    memcpy(m->lw[layer].attn_norm, attn_norm, m->cfg.dim * sizeof(float));
    memcpy(m->lw[layer].ffn_norm, ffn_norm, m->cfg.dim * sizeof(float));
    memcpy(m->lw[layer].q_norm, q_norm, m->cfg.head_dim * sizeof(float));
    memcpy(m->lw[layer].k_norm, k_norm, m->cfg.head_dim * sizeof(float));
}

void forward_model_set_layer_kernels(ForwardModel *m, int layer,
    ANEKernelHandle *q, ANEKernelHandle *k, ANEKernelHandle *v,
    ANEKernelHandle *o, ANEKernelHandle *gate, ANEKernelHandle *up,
    ANEKernelHandle *down)
{
    if (layer < 0 || layer >= m->cfg.n_layers) return;
    m->lk[layer].q    = q;
    m->lk[layer].k    = k;
    m->lk[layer].v    = v;
    m->lk[layer].o    = o;
    m->lk[layer].gate = gate;
    m->lk[layer].up   = up;
    m->lk[layer].down = down;
    m->lk[layer].fused = false;
}

void forward_model_set_layer_kernels_fused(ForwardModel *m, int layer,
    ANEKernelHandle *qkv, ANEKernelHandle *o,
    ANEKernelHandle *gate_up, ANEKernelHandle *down)
{
    if (layer < 0 || layer >= m->cfg.n_layers) return;
    m->lk[layer].qkv     = qkv;
    m->lk[layer].o       = o;
    m->lk[layer].gate_up = gate_up;
    m->lk[layer].down    = down;
    m->lk[layer].fused   = true;
}

void forward_model_add_cls_kernel(ForwardModel *m, ANEKernelHandle *kernel, int out_channels) {
    if (m->n_cls_chunks >= MAX_CLS_CHUNKS) {
        fprintf(stderr, "forward_pass: too many classifier chunks (max %d)\n", MAX_CLS_CHUNKS);
        return;
    }
    m->cls_chunks[m->n_cls_chunks].kernel = kernel;
    m->cls_chunks[m->n_cls_chunks].out_channels = out_channels;
    m->n_cls_chunks++;
}

void forward_model_reset_cache(ForwardModel *m) {
    int kv_dim = m->cfg.n_kv_heads * m->cfg.head_dim;
    for (int l = 0; l < m->cfg.n_layers; l++) {
        memset(m->k_caches[l], 0, m->cfg.max_seq * kv_dim * sizeof(float));
        memset(m->v_caches[l], 0, m->cfg.max_seq * kv_dim * sizeof(float));
    }
}

const float *forward_model_forward_token(ForwardModel *m, int token_id, int pos) {
    double t_start = now_ms(m);
    double t;

    // Reset timing
    memset(&m->timing, 0, sizeof(ForwardTiming));

    int dim = m->cfg.dim;
    int q_dim = m->cfg.n_heads * m->cfg.head_dim;
    int kv_dim = m->cfg.n_kv_heads * m->cfg.head_dim;
    int hd = m->cfg.hidden_dim;

    // ── Embedding lookup ──
    t = now_ms(m);
    memcpy(m->x, m->embed_w + token_id * dim, dim * sizeof(float));
    m->timing.embed_ms += now_ms(m) - t;

    // ── Transformer layers ──
    for (int l = 0; l < m->cfg.n_layers; l++) {
        LayerWeights *lw = &m->lw[l];
        LayerKernels *lk = &m->lk[l];

        // -- Attention block --

        // RMSNorm
        t = now_ms(m);
        rmsnorm(m->xn, m->x, lw->attn_norm, dim);
        m->timing.rmsnorm_ms += now_ms(m) - t;

        // QKV projections (ANE)
        t = now_ms(m);
        if (lk->fused) {
            // Fused QKV: single dispatch, split output into q/k/v
            float *qkv_outs[3] = { m->q_buf, m->k_buf, m->v_buf };
            int qkv_dims[3] = { q_dim, kv_dim, kv_dim };
            ane_linear_fused(m, lk->qkv, m->xn, dim, q_dim + kv_dim + kv_dim,
                             qkv_outs, qkv_dims, 3);
        } else {
            ane_linear(m, lk->q, m->xn, m->q_buf, dim, q_dim);
            ane_linear(m, lk->k, m->xn, m->k_buf, dim, kv_dim);
            ane_linear(m, lk->v, m->xn, m->v_buf, dim, kv_dim);
        }
        m->timing.ane_ms += now_ms(m) - t;

        // QK-norm
        t = now_ms(m);
        qk_norm(m->q_buf, m->k_buf, lw->q_norm, lw->k_norm,
                m->cfg.n_heads, m->cfg.n_kv_heads, m->cfg.head_dim);
        m->timing.qknorm_ms += now_ms(m) - t;

        // RoPE
        t = now_ms(m);
        rope(m->q_buf, pos, m->cfg.n_heads, m->cfg.head_dim, m->rope_freqs);
        rope(m->k_buf, pos, m->cfg.n_kv_heads, m->cfg.head_dim, m->rope_freqs);
        m->timing.rope_ms += now_ms(m) - t;

        // KV cache write
        memcpy(m->k_caches[l] + pos * kv_dim, m->k_buf, kv_dim * sizeof(float));
        memcpy(m->v_caches[l] + pos * kv_dim, m->v_buf, kv_dim * sizeof(float));

        // Attention
        t = now_ms(m);
        attention(m, m->q_buf, m->k_caches[l], m->v_caches[l], m->attn_out, pos);
        m->timing.attention_ms += now_ms(m) - t;

        // O projection (ANE)
        t = now_ms(m);
        ane_linear(m, lk->o, m->attn_out, m->down_buf, q_dim, dim);
        m->timing.ane_ms += now_ms(m) - t;

        // Residual
        vDSP_vadd(m->x, 1, m->down_buf, 1, m->x, 1, dim);

        // -- FFN block --

        // RMSNorm
        t = now_ms(m);
        rmsnorm(m->xn, m->x, lw->ffn_norm, dim);
        m->timing.rmsnorm_ms += now_ms(m) - t;

        // Gate + Up projections (ANE)
        t = now_ms(m);
        if (lk->fused) {
            // Fused Gate+Up: single dispatch, split into gate/up
            float *gu_outs[2] = { m->gate_buf, m->up_buf };
            int gu_dims[2] = { hd, hd };
            ane_linear_fused(m, lk->gate_up, m->xn, dim, hd * 2,
                             gu_outs, gu_dims, 2);
        } else {
            ane_linear(m, lk->gate, m->xn, m->gate_buf, dim, hd);
            ane_linear(m, lk->up,   m->xn, m->up_buf,   dim, hd);
        }
        m->timing.ane_ms += now_ms(m) - t;

        // SiLU gate * up
        t = now_ms(m);
        silu_gate_mul(m->silu_buf, m->gate_buf, m->up_buf, hd);
        m->timing.silu_ms += now_ms(m) - t;

        // Down projection (ANE)
        t = now_ms(m);
        ane_linear(m, lk->down, m->silu_buf, m->down_buf, hd, dim);
        m->timing.ane_ms += now_ms(m) - t;

        // Residual
        vDSP_vadd(m->x, 1, m->down_buf, 1, m->x, 1, dim);
    }

    // ── Final norm ──
    t = now_ms(m);
    rmsnorm(m->xn, m->x, m->final_norm_w, dim);
    m->timing.rmsnorm_ms += now_ms(m) - t;

    // ── Classifier (tiled ANE) ──
    t = now_ms(m);
    int offset = 0;
    for (int ci = 0; ci < m->n_cls_chunks; ci++) {
        int out_ch = m->cls_chunks[ci].out_channels;
        ane_linear(m, m->cls_chunks[ci].kernel, m->xn, m->logits + offset, dim, out_ch);
        offset += out_ch;
    }
    m->timing.classify_ms = now_ms(m) - t;

    m->timing.total_ms = now_ms(m) - t_start;
    return m->logits;
}

void forward_model_get_timing(const ForwardModel *m, ForwardTiming *out) {
    if (m && out) *out = m->timing;
}
