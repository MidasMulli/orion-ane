// cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, embedding
#pragma once
#include "config.h"

static float *g_rms_tmp = NULL;

static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
    free(ss);
}

static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float *rrms = (float*)malloc(S*4);
    int n = S; vvrsqrtf(rrms, ss, &n);
    float *dot = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsma(g_rms_tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, dot, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsub(g_rms_tmp, 1, dy+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(g_rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(g_rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
    free(ss); free(rrms); free(dot);
}

static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps, float wd) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i=0; i<s->n; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*g[i];
        s->v[i] = b2*s->v[i] + (1-b2)*g[i]*g[i];
        float mh = s->m[i]/bc1, vh = s->v[i]/bc2;
        w[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * w[i]);
    }
}

// Cross-entropy loss: operates on logits[V, S] column-major (each column = one token)
// Avoids transposing by using a per-token temp buffer
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    float *col = (float*)malloc(V * 4);  // single column buffer
    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        // Gather column t: logits[v, t] = logits[v*S + t], stride=S
        cblas_scopy(V, logits + t, S, col, 1);
        // Softmax
        float maxv; vDSP_maxv(col, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(col, 1, &neg_max, col, 1, (vDSP_Length)V);
        int n = V; vvexpf(col, col, &n);
        float sum; vDSP_sve(col, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(col, 1, &inv_sum, col, 1, (vDSP_Length)V);
        // Loss + gradient
        int tgt = targets[t];
        total_loss -= logf(col[tgt] + 1e-10f);
        col[tgt] -= 1.0f;
        vDSP_vsmul(col, 1, &invS, col, 1, (vDSP_Length)V);
        // Scatter back: dlogits[v*S + t] = col[v]
        cblas_scopy(V, col, 1, dlogits + t, S);
    }
    free(col);
    return total_loss / S;
}

// KL-divergence loss for knowledge distillation
// teacher_top_ids[top_k * S]: teacher's top-K token IDs per position (int32, column-major: ids for pos 0, then pos 1, ...)
// teacher_top_logits[top_k * S]: teacher's top-K logits per position (float32, same layout)
// full_to_compact[VOCAB]: maps full vocab ID → compact ID (-1 if unused)
// temperature: softmax temperature for both student and teacher distributions
static float kl_divergence_loss(float *dlogits, const float *logits,
                                const int *teacher_top_ids, const float *teacher_top_logits,
                                const int *full_to_compact, int top_k,
                                int V, int S, float temperature) {
    float *s_col = (float*)malloc(V * 4);   // student softmax column
    float *t_dist = (float*)calloc(V, 4);   // teacher distribution over compact vocab
    float total_loss = 0;
    float invS = 1.0f / S;
    float invT = 1.0f / temperature;
    float T2 = temperature * temperature;  // gradient scaling for temperature

    for (int t = 0; t < S; t++) {
        // --- Student softmax (temperature-scaled) ---
        cblas_scopy(V, logits + t, S, s_col, 1);
        // Scale by 1/T
        vDSP_vsmul(s_col, 1, &invT, s_col, 1, (vDSP_Length)V);
        // Softmax
        float maxv; vDSP_maxv(s_col, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(s_col, 1, &neg_max, s_col, 1, (vDSP_Length)V);
        int n = V; vvexpf(s_col, s_col, &n);
        float sum; vDSP_sve(s_col, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(s_col, 1, &inv_sum, s_col, 1, (vDSP_Length)V);

        // --- Teacher distribution from top-K ---
        memset(t_dist, 0, V * 4);
        // Apply temperature to teacher logits and softmax over top-K
        float t_max = -1e30f;
        const int *t_ids = teacher_top_ids + t * top_k;
        const float *t_logits = teacher_top_logits + t * top_k;
        for (int k = 0; k < top_k; k++) {
            float scaled = t_logits[k] * invT;
            if (scaled > t_max) t_max = scaled;
        }
        float t_sum = 0;
        float t_probs[top_k];
        for (int k = 0; k < top_k; k++) {
            t_probs[k] = expf(t_logits[k] * invT - t_max);
            t_sum += t_probs[k];
        }
        float t_inv = 1.0f / t_sum;
        for (int k = 0; k < top_k; k++) {
            int cid = full_to_compact[t_ids[k]];
            if (cid >= 0) {
                t_dist[cid] = t_probs[k] * t_inv;
            }
        }

        // --- KL divergence: sum(teacher * log(teacher / student)) ---
        for (int v = 0; v < V; v++) {
            if (t_dist[v] > 1e-10f) {
                total_loss += t_dist[v] * logf(t_dist[v] / (s_col[v] + 1e-10f));
            }
        }

        // --- Gradient: T^2 * (student - teacher) / S ---
        // Same form as cross-entropy but with soft targets, scaled by T^2
        for (int v = 0; v < V; v++) {
            s_col[v] = (s_col[v] - t_dist[v]) * invS * T2;
        }
        cblas_scopy(V, s_col, 1, dlogits + t, S);
    }

    free(s_col);
    free(t_dist);
    return total_loss / S;
}

// Vocab compaction: build mapping from full 32K vocab to compact vocab
typedef struct {
    int compact_vocab;          // number of active tokens
    int *full_to_compact;       // [VOCAB] → compact id (-1 if unused)
    int *compact_to_full;       // [compact_vocab] → full vocab id
} VocabMap;

static VocabMap vocab_map_build_u32(const uint32_t *data, size_t n_tokens, int full_vocab) {
    VocabMap vm;
    vm.full_to_compact = (int*)malloc(full_vocab * sizeof(int));
    memset(vm.full_to_compact, -1, full_vocab * sizeof(int));
    for (size_t i = 0; i < n_tokens; i++) {
        if ((int)data[i] < full_vocab)
            vm.full_to_compact[data[i]] = 0;
    }
    int cid = 0;
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] == 0)
            vm.full_to_compact[v] = cid++;
        else
            vm.full_to_compact[v] = -1;
    }
    vm.compact_vocab = cid;
    vm.compact_to_full = (int*)malloc(cid * sizeof(int));
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] >= 0)
            vm.compact_to_full[vm.full_to_compact[v]] = v;
    }
    return vm;
}

static VocabMap vocab_map_build(const uint16_t *data, size_t n_tokens, int full_vocab) {
    VocabMap vm;
    vm.full_to_compact = (int*)malloc(full_vocab * sizeof(int));
    memset(vm.full_to_compact, -1, full_vocab * sizeof(int));
    // Scan for used tokens
    for (size_t i = 0; i < n_tokens; i++) {
        vm.full_to_compact[data[i]] = 0;  // mark as used
    }
    // Assign compact IDs
    int cid = 0;
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] == 0)
            vm.full_to_compact[v] = cid++;
        else
            vm.full_to_compact[v] = -1;
    }
    vm.compact_vocab = cid;
    vm.compact_to_full = (int*)malloc(cid * sizeof(int));
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] >= 0)
            vm.compact_to_full[vm.full_to_compact[v]] = v;
    }
    return vm;
}

// Create compact embedding from full embedding
static float *vocab_compact_embed(const float *full_embed, const VocabMap *vm, int dim) {
    float *ce = (float*)malloc((size_t)vm->compact_vocab * dim * 4);
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(ce + c*dim, full_embed + vm->compact_to_full[c]*dim, dim*4);
    return ce;
}

// Scatter compact embed gradients back to full embed
static void vocab_scatter_grads(float *full_gembed, const float *compact_gembed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++) {
        int fv = vm->compact_to_full[c];
        for (int d = 0; d < dim; d++)
            full_gembed[fv*dim + d] += compact_gembed[c*dim + d];
    }
}

// Update full embed from compact embed (after adam)
static void vocab_update_full(float *full_embed, const float *compact_embed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(full_embed + vm->compact_to_full[c]*dim, compact_embed + c*dim, dim*4);
}

static void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            x[d*seq + t] = embed[tok*dim + d];
    }
}

static void embed_lookup_u32(float *x, const float *embed, const uint32_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = (int)tokens[t];
        for (int d = 0; d < dim; d++)
            x[d*seq + t] = embed[tok*dim + d];
    }
}

static void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            d_embed[tok*dim + d] += dx[d*seq + t];
    }
}

static void embed_backward_u32(float *d_embed, const float *dx, const uint32_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = (int)tokens[t];
        for (int d = 0; d < dim; d++)
            d_embed[tok*dim + d] += dx[d*seq + t];
    }
}

// RoPE backward (in-place): inverse rotation on dQ/dK gradients
// Data layout: [DIM, SEQ] channel-first, DIM = nheads * hd
static void rope_backward_inplace(float *dx, int seq, int dim, int hd) {
    int nheads = dim / hd;
    for (int h = 0; h < nheads; h++) {
        for (int i = 0; i < hd/2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)hd);
            for (int p = 0; p < seq; p++) {
                float theta = p * freq;
                float cos_t = cosf(theta), sin_t = sinf(theta);
                int idx0 = (h * hd + 2 * i) * seq + p;
                int idx1 = (h * hd + 2 * i + 1) * seq + p;
                float v0 = dx[idx0], v1 = dx[idx1];
                dx[idx0] = v0 * cos_t + v1 * sin_t;
                dx[idx1] = -v0 * sin_t + v1 * cos_t;
            }
        }
    }
}
