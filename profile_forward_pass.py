"""
Profile where time goes in the real_draft.py forward pass.
Identifies the actual bottlenecks to target.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speculative"))
from real_draft import RealDraftModel, ROPE_THETA, ANE_SPATIAL

# Load the model
print("Loading model...")
model = RealDraftModel()
model.load_and_compile(lambda msg: print(f"  {msg}"))

# Test prompt
prompt = "The capital of France is"
prompt_ids = model.encode(prompt)
print(f"\nPrompt: {prompt}")
print(f"Token IDs: {prompt_ids}")

# Profile one full forward pass (after prefill)
model.reset_cache()
# Prefill
for i, tid in enumerate(prompt_ids[:-1]):
    model.forward_token(tid, i)

# Now profile the last token in detail
token_id = prompt_ids[-1]
pos = len(prompt_ids) - 1

# Detailed timing per component
n_iters = 3
times = {
    "embedding": [],
    "rmsnorm_attn": [],
    "qkv_projection": [],
    "qk_norm": [],
    "rope": [],
    "kv_cache_write": [],
    "attention": [],
    "o_projection": [],
    "rmsnorm_ffn": [],
    "gate_up_projection": [],
    "silu_gate": [],
    "down_projection": [],
    "final_norm": [],
    "classifier": [],
}

for iteration in range(n_iters):
    model.reset_cache()
    # Quick prefill
    for i, tid in enumerate(prompt_ids[:-1]):
        model.forward_token(tid, i)

    # Profile final token
    t_start = time.time()

    t = time.time()
    x = model.embed_w[token_id].copy()
    times["embedding"].append(time.time() - t)

    q_dim = model.n_heads * model.head_dim
    kv_dim = model.n_kv_heads * model.head_dim

    for l in range(model.n_layers):
        lw = model.layer_weights[l]
        lk = model.kernels[l]

        # RMSNorm attn
        t = time.time()
        xn = model._rmsnorm(x, lw['attn_norm'])
        times["rmsnorm_attn"].append(time.time() - t)

        # QKV projections (ANE)
        t = time.time()
        q = model._ane_linear(lk['q'], xn, model.dim, q_dim)
        k = model._ane_linear(lk['k'], xn, model.dim, kv_dim)
        v = model._ane_linear(lk['v'], xn, model.dim, kv_dim)
        times["qkv_projection"].append(time.time() - t)

        # QK-norm
        t = time.time()
        for h in range(model.n_heads):
            s, e = h * model.head_dim, (h+1) * model.head_dim
            q[s:e] = model._rmsnorm(q[s:e], lw['q_norm'])
        for h in range(model.n_kv_heads):
            s, e = h * model.head_dim, (h+1) * model.head_dim
            k[s:e] = model._rmsnorm(k[s:e], lw['k_norm'])
        times["qk_norm"].append(time.time() - t)

        # RoPE
        t = time.time()
        q = model._rope(q, pos, model.n_heads, model.head_dim)
        k = model._rope(k, pos, model.n_kv_heads, model.head_dim)
        times["rope"].append(time.time() - t)

        # KV cache write
        t = time.time()
        model.k_caches[l][pos] = k
        model.v_caches[l][pos] = v
        times["kv_cache_write"].append(time.time() - t)

        # Attention
        t = time.time()
        attn = model._attention(q, model.k_caches[l], model.v_caches[l], pos)
        times["attention"].append(time.time() - t)

        # O projection (ANE)
        t = time.time()
        o = model._ane_linear(lk['o'], attn, q_dim, model.dim)
        times["o_projection"].append(time.time() - t)

        x = x + o

        # RMSNorm FFN
        t = time.time()
        xn = model._rmsnorm(x, lw['ffn_norm'])
        times["rmsnorm_ffn"].append(time.time() - t)

        # Gate + Up projections (ANE)
        t = time.time()
        gate = model._ane_linear(lk['gate'], xn, model.dim, model.hidden_dim)
        up = model._ane_linear(lk['up'], xn, model.dim, model.hidden_dim)
        times["gate_up_projection"].append(time.time() - t)

        # SiLU gate
        t = time.time()
        silu_gate = gate / (1.0 + np.exp(-np.clip(gate, -20, 20)))
        times["silu_gate"].append(time.time() - t)

        # Down projection (ANE)
        t = time.time()
        down = model._ane_linear(lk['down'], silu_gate * up, model.hidden_dim, model.dim)
        times["down_projection"].append(time.time() - t)

        x = x + down

    # Final norm
    t = time.time()
    x = model._rmsnorm(x, model.final_norm_w)
    times["final_norm"].append(time.time() - t)

    # Classifier
    t = time.time()
    logits = model._classify(x)
    times["classifier"].append(time.time() - t)

    total = time.time() - t_start
    print(f"  Iteration {iteration}: {total*1000:.1f}ms total")

# Summary
print("\n" + "=" * 60)
print(f"FORWARD PASS PROFILE ({model.n_layers} layers, pos={pos})")
print("=" * 60)

total_avg = 0
for name, t_list in times.items():
    avg_ms = np.mean(t_list) * 1000
    total_avg += avg_ms
    # Per-layer items need to be shown as total across all layers
    if name in ["embedding", "final_norm", "classifier"]:
        pct = avg_ms / (np.mean(t_list) * 1000 + 1e-9) * 100
    bar = "█" * int(avg_ms / 2)
    print(f"  {name:25s} {avg_ms:8.2f}ms  {bar}")

print(f"  {'TOTAL':25s} {total_avg:8.2f}ms")
print(f"  → {1000/total_avg:.1f} tokens/sec")

# Categorize
ane_time = (np.mean(times["qkv_projection"]) + np.mean(times["o_projection"]) +
            np.mean(times["gate_up_projection"]) + np.mean(times["down_projection"]) +
            np.mean(times["classifier"])) * 1000
cpu_loop_time = (np.mean(times["attention"]) + np.mean(times["rope"]) +
                 np.mean(times["qk_norm"])) * 1000
cpu_vectorized = (np.mean(times["rmsnorm_attn"]) + np.mean(times["rmsnorm_ffn"]) +
                  np.mean(times["silu_gate"]) + np.mean(times["embedding"]) +
                  np.mean(times["final_norm"]) + np.mean(times["kv_cache_write"])) * 1000

print(f"\nBREAKDOWN:")
print(f"  ANE (linear projections):  {ane_time:.1f}ms ({ane_time/total_avg*100:.0f}%)")
print(f"  CPU loops (attn+rope+qkn): {cpu_loop_time:.1f}ms ({cpu_loop_time/total_avg*100:.0f}%)")
print(f"  CPU vectorized (norm etc): {cpu_vectorized:.1f}ms ({cpu_vectorized/total_avg*100:.0f}%)")
