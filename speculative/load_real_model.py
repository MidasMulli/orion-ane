"""
Load Real Model Weights for ANE Draft
Downloads Qwen3-0.6B, extracts first N layers, compiles to ANE kernels.
Same tokenizer as Qwen 3.5 9B verifier → real acceptance rates.
"""

import os
import sys
import time
import json
import numpy as np
import ctypes
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
from ane_draft import init_ane, lib, compile_ane_kernel, generate_conv_mil_with_weights

# ── Config ───────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-0.6B"
N_DRAFT_LAYERS = 28  # All 28 layers — full Qwen3-0.6B on ANE
MAX_SEQ = 32        # Short sequences for draft

print("╔══════════════════════════════════════════════════════════╗")
print("║  LOADING REAL MODEL WEIGHTS → ANE KERNELS               ║")
print(f"║  Model: {MODEL_NAME:<44} ║")
print(f"║  Draft layers: {N_DRAFT_LAYERS} of 28                              ║")
print("╚══════════════════════════════════════════════════════════╝")

# ── Step 1: Download model ───────────────────────────────────────────────
print("\n[1/5] Downloading model...")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32,
                                               trust_remote_code=True)
model.eval()

download_time = time.time() - t0
print(f"  ✓ Downloaded in {download_time:.1f}s")

# ── Step 2: Extract architecture info ────────────────────────────────────
print("\n[2/5] Extracting architecture...")
config = model.config
dim = config.hidden_size
n_heads = config.num_attention_heads
n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
head_dim = getattr(config, 'head_dim', dim // n_heads)
hidden_dim = config.intermediate_size
vocab_size = config.vocab_size
n_layers_total = config.num_hidden_layers

print(f"  dim={dim}, heads={n_heads}, kv_heads={n_kv_heads}, head_dim={head_dim}")
print(f"  hidden_dim={hidden_dim}, vocab={vocab_size}, layers={n_layers_total}")
print(f"  Using first {N_DRAFT_LAYERS} layers for draft model")

# Memory estimate
embed_mb = vocab_size * dim * 2 / 1e6  # fp16
layer_mb = (4 * dim * dim * 2 + 3 * dim * hidden_dim * 2) / 1e6  # fp16
cls_mb = vocab_size * dim * 2 / 1e6
total_mb = embed_mb + N_DRAFT_LAYERS * layer_mb + cls_mb
print(f"  Memory: embed={embed_mb:.0f}MB, {N_DRAFT_LAYERS}×layer={N_DRAFT_LAYERS*layer_mb:.0f}MB, cls={cls_mb:.0f}MB")
print(f"  Total: {total_mb:.0f}MB (fp16)")

# ── Step 3: Extract weights as numpy ─────────────────────────────────────
print("\n[3/5] Extracting weights...")

state = model.state_dict()

# Embedding
embed_w = state['model.embed_tokens.weight'].numpy()  # [vocab, dim]
print(f"  embed: {embed_w.shape}")

# Per-layer weights
layer_weights = []
for l in range(N_DRAFT_LAYERS):
    prefix = f'model.layers.{l}'
    lw = {
        'q': state[f'{prefix}.self_attn.q_proj.weight'].numpy(),     # [q_dim, dim]
        'k': state[f'{prefix}.self_attn.k_proj.weight'].numpy(),     # [kv_dim, dim]
        'v': state[f'{prefix}.self_attn.v_proj.weight'].numpy(),     # [kv_dim, dim]
        'o': state[f'{prefix}.self_attn.o_proj.weight'].numpy(),     # [dim, q_dim]
        'gate': state[f'{prefix}.mlp.gate_proj.weight'].numpy(),     # [hidden, dim]
        'up': state[f'{prefix}.mlp.up_proj.weight'].numpy(),         # [hidden, dim]
        'down': state[f'{prefix}.mlp.down_proj.weight'].numpy(),     # [dim, hidden]
        'attn_norm': state[f'{prefix}.input_layernorm.weight'].numpy(),
        'ffn_norm': state[f'{prefix}.post_attention_layernorm.weight'].numpy(),
        'q_norm': state[f'{prefix}.self_attn.q_norm.weight'].numpy(),   # [head_dim]
        'k_norm': state[f'{prefix}.self_attn.k_norm.weight'].numpy(),   # [head_dim]
    }
    layer_weights.append(lw)
    print(f"  layer {l}: q={lw['q'].shape} gate={lw['gate'].shape}")

# Final norm + classifier
final_norm_w = state['model.norm.weight'].numpy()
lm_head_w = state['lm_head.weight'].numpy()  # [vocab, dim]
print(f"  lm_head: {lm_head_w.shape}")
print(f"  final_norm: {final_norm_w.shape}")

# Free the PyTorch model
del model, state
import gc; gc.collect()
print("  ✓ Weights extracted, PyTorch model freed")

# ── Step 4: Compile ANE kernels ──────────────────────────────────────────
print("\n[4/5] Compiling ANE kernels...")

if not init_ane():
    print("  ❌ ANE init failed")
    sys.exit(1)

spatial = 16  # ANE minimum spatial dimension (pad single token into slot 0)
kernels = {}
compile_times = []

kv_dim = n_kv_heads * head_dim

for l in range(N_DRAFT_LAYERS):
    lw = layer_weights[l]
    layer_kernels = {}

    q_dim = n_heads * head_dim       # 16 * 128 = 2048
    kv_dim_proj = n_kv_heads * head_dim  # 8 * 128 = 1024

    for name, w, in_d, out_d in [
        ('q', lw['q'], dim, q_dim),
        ('k', lw['k'], dim, kv_dim_proj),
        ('v', lw['v'], dim, kv_dim_proj),
        ('o', lw['o'], q_dim, dim),
        ('gate', lw['gate'], dim, hidden_dim),
        ('up', lw['up'], dim, hidden_dim),
        ('down', lw['down'], hidden_dim, dim),
    ]:
        kern_name = f"l{l}_{name}"
        t0 = time.time()
        mil, wb = generate_conv_mil_with_weights(in_d, out_d, spatial, w, f"real_{kern_name}")
        k = compile_ane_kernel(mil, wb, [in_d * spatial * 4], [out_d * spatial * 4])
        ct = (time.time() - t0) * 1000
        compile_times.append(ct)
        if not k:
            print(f"    ❌ {kern_name} FAILED ({in_d}→{out_d})")
            continue
        layer_kernels[name] = k
        print(f"    ✓ {kern_name} ({in_d}→{out_d}) {ct:.0f}ms")

    kernels[l] = layer_kernels

# Classifier — tile into chunks
CLS_CHUNK = 8000
n_cls_chunks = (vocab_size + CLS_CHUNK - 1) // CLS_CHUNK
cls_kernels = []
print(f"  Classifier: {dim}→{vocab_size} ({n_cls_chunks} chunks of ≤{CLS_CHUNK})...")

for ci in range(n_cls_chunks):
    start = ci * CLS_CHUNK
    end = min(start + CLS_CHUNK, vocab_size)
    out_ch = end - start
    w_chunk = lm_head_w[start:end]  # [out_ch, dim]
    kern_name = f"cls_{ci}"
    t0 = time.time()
    mil, wb = generate_conv_mil_with_weights(dim, out_ch, spatial, w_chunk, f"real_{kern_name}")
    k = compile_ane_kernel(mil, wb, [dim * spatial * 4], [out_ch * spatial * 4])
    ct = (time.time() - t0) * 1000
    compile_times.append(ct)
    if not k:
        print(f"    ❌ cls chunk {ci} FAILED ({dim}→{out_ch})")
        continue
    cls_kernels.append((k, out_ch))
    if ci % 5 == 0 or ci == n_cls_chunks - 1:
        print(f"    ✓ cls chunk {ci}/{n_cls_chunks} ({dim}→{out_ch}) {ct:.0f}ms")

total_compile = sum(compile_times)
n_total_kernels = sum(len(v) for v in kernels.values()) + len(cls_kernels)
print(f"\n  ✓ {n_total_kernels} ANE kernels compiled in {total_compile:.0f}ms")

# ── Step 5: Test inference ───────────────────────────────────────────────
print("\n[5/5] Running inference test...")


def rmsnorm(x, w, eps=1e-6):
    ss = np.mean(x * x) + eps
    return x / np.sqrt(ss) * w


ROPE_THETA = 1000000.0  # Qwen3 uses 1M, not 10K

def rope(x, pos, n_h, hd):
    """Apply RoPE to x of shape [n_h * hd]."""
    out = x.copy()
    for h in range(n_h):
        for i in range(0, hd, 2):
            freq = 1.0 / (ROPE_THETA ** (float(i) / hd))
            val = pos * freq
            cos_v, sin_v = np.cos(val), np.sin(val)
            off = h * hd + i
            x0, x1 = out[off], out[off + 1]
            out[off] = x0 * cos_v - x1 * sin_v
            out[off + 1] = x0 * sin_v + x1 * cos_v
    return out


def ane_linear(kernel, x, in_d, out_d):
    """Run ANE conv1x1 kernel. Pads to spatial=16, extracts position 0."""
    # ANE layout: [ch0_sp0..ch0_sp15, ch1_sp0..ch1_sp15, ...] (NCHW contiguous)
    # Pack single token: x[i] goes to channel i, spatial position 0
    x_padded = np.zeros((in_d, spatial), dtype=np.float32)
    x_padded[:, 0] = x.astype(np.float32)
    x_padded = x_padded.ravel()
    out_padded = np.zeros(out_d * spatial, dtype=np.float32)
    lib.ane_bridge_write_input(kernel, 0,
                                x_padded.ctypes.data_as(ctypes.c_void_p), in_d * spatial * 4)
    lib.ane_bridge_eval(kernel)
    lib.ane_bridge_read_output(kernel, 0,
                                out_padded.ctypes.data_as(ctypes.c_void_p), out_d * spatial * 4)
    # Extract spatial position 0 from each channel
    out = out_padded.reshape(out_d, spatial)[:, 0]
    return out


def attention_single(q, k_cache, v_cache, pos, n_h, n_kv, hd):
    """Single-query attention with GQA and KV cache."""
    scale = 1.0 / np.sqrt(hd)
    out = np.zeros(n_h * hd, dtype=np.float32)
    heads_per_kv = n_h // n_kv

    for h in range(n_h):
        kv_h = h // heads_per_kv
        q_h = q[h * hd:(h + 1) * hd]
        scores = np.zeros(pos + 1, dtype=np.float32)
        for s in range(pos + 1):
            k_h = k_cache[s, kv_h * hd:(kv_h + 1) * hd]
            scores[s] = np.dot(q_h, k_h) * scale
        # Softmax
        scores -= scores.max()
        scores = np.exp(scores)
        scores /= scores.sum()
        # Weighted sum
        for s in range(pos + 1):
            v_h = v_cache[s, kv_h * hd:(kv_h + 1) * hd]
            out[h * hd:(h + 1) * hd] += scores[s] * v_h
    return out


def classify_ane(x):
    """Run tiled classifier on ANE."""
    logits = np.zeros(vocab_size, dtype=np.float32)
    offset = 0
    for kernel, out_ch in cls_kernels:
        chunk = ane_linear(kernel, x, dim, out_ch)
        logits[offset:offset + out_ch] = chunk
        offset += out_ch
    return logits


def forward_token(token_id, pos, k_caches, v_caches):
    """Full forward pass for one token position."""
    x = embed_w[token_id].copy()

    for l in range(N_DRAFT_LAYERS):
        lw = layer_weights[l]
        lk = kernels[l]

        # Attention
        q_dim = n_heads * head_dim        # 2048
        kv_dim_proj = n_kv_heads * head_dim  # 1024
        xn = rmsnorm(x, lw['attn_norm'])
        q = ane_linear(lk['q'], xn, dim, q_dim)
        k = ane_linear(lk['k'], xn, dim, kv_dim_proj)
        v = ane_linear(lk['v'], xn, dim, kv_dim_proj)

        # QK-norm: per-head RMSNorm on Q and K (Qwen3 specific)
        q_norm_w = lw['q_norm']  # [head_dim]
        k_norm_w = lw['k_norm']  # [head_dim]
        for h in range(n_heads):
            q[h*head_dim:(h+1)*head_dim] = rmsnorm(q[h*head_dim:(h+1)*head_dim], q_norm_w)
        for h in range(n_kv_heads):
            k[h*head_dim:(h+1)*head_dim] = rmsnorm(k[h*head_dim:(h+1)*head_dim], k_norm_w)

        q = rope(q, pos, n_heads, head_dim)
        k = rope(k, pos, n_kv_heads, head_dim)

        k_caches[l][pos] = k
        v_caches[l][pos] = v

        attn = attention_single(q, k_caches[l], v_caches[l], pos, n_heads, n_kv_heads, head_dim)
        o = ane_linear(lk['o'], attn, q_dim, dim)
        x = x + o

        # FFN (SwiGLU)
        xn = rmsnorm(x, lw['ffn_norm'])
        gate = ane_linear(lk['gate'], xn, dim, hidden_dim)
        up = ane_linear(lk['up'], xn, dim, hidden_dim)
        # SiLU(gate) * up
        silu_gate = gate / (1.0 + np.exp(-np.clip(gate, -20, 20)))
        h = silu_gate * up
        down = ane_linear(lk['down'], h, hidden_dim, dim)
        x = x + down

    x = rmsnorm(x, final_norm_w)
    logits = classify_ane(x)
    return logits


# Test with a real prompt
test_prompt = "The capital of France is"
token_ids = tokenizer.encode(test_prompt)
print(f"  Prompt: '{test_prompt}'")
print(f"  Token IDs: {token_ids}")

# Init KV caches
kv_dim_total = n_kv_heads * head_dim  # 8 * 128 = 1024
k_caches = [np.zeros((MAX_SEQ, kv_dim_total), dtype=np.float32) for _ in range(N_DRAFT_LAYERS)]
v_caches = [np.zeros((MAX_SEQ, kv_dim_total), dtype=np.float32) for _ in range(N_DRAFT_LAYERS)]

# Process prompt
t0 = time.time()
for i, tid in enumerate(token_ids):
    logits = forward_token(tid, i, k_caches, v_caches)
prompt_time = (time.time() - t0) * 1000

# Show top-5 after prompt
top5_idx = np.argsort(logits)[-5:][::-1]
print(f"  Top-5 after prompt: {[(int(i), tokenizer.decode([i]), float(logits[i])) for i in top5_idx]}")
print(f"  Logit stats: min={logits.min():.2f} max={logits.max():.2f} mean={logits.mean():.4f} std={logits.std():.4f}")

# Generate tokens
n_gen = 10
generated = []
pos = len(token_ids)
t0 = time.time()
for _ in range(n_gen):
    next_tok = int(np.argmax(logits))
    generated.append(next_tok)
    logits = forward_token(next_tok, pos, k_caches, v_caches)
    pos += 1
gen_time = (time.time() - t0) * 1000

gen_text = tokenizer.decode(generated)
tok_per_sec = n_gen / (gen_time / 1000)

print(f"\n  ═══════════════════════════════════════════")
print(f"  REAL ANE DRAFT MODEL — {N_DRAFT_LAYERS} layers of Qwen3-0.6B")
print(f"  ═══════════════════════════════════════════")
print(f"  Prompt:     '{test_prompt}'")
print(f"  Generated:  '{gen_text}'")
print(f"  Token IDs:  {generated}")
print(f"  Prompt time: {prompt_time:.1f} ms ({len(token_ids)} tokens)")
print(f"  Gen time:    {gen_time:.1f} ms ({n_gen} tokens)")
print(f"  Speed:       {tok_per_sec:.1f} tok/s on ANE")
print(f"  Kernels:     {n_total_kernels} compiled")

# Compare with full model via MLX
print(f"\n  Comparing with GPU (Qwen 3.5 9B)...")
import requests
try:
    r = requests.post("http://localhost:8899/v1/chat/completions", json={
        "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": 20, "temperature": 0.0,
    }, timeout=30)
    gpu_text = r.json()["choices"][0]["message"]["content"]
    print(f"  GPU says:    '{gpu_text[:100]}'")
    print(f"  ANE says:    '{gen_text}'")
except Exception as e:
    print(f"  GPU comparison failed: {e}")

# Save model info for the speculative decoder
model_info = {
    "model_name": MODEL_NAME,
    "dim": dim,
    "n_heads": n_heads,
    "n_kv_heads": n_kv_heads,
    "head_dim": head_dim,
    "hidden_dim": hidden_dim,
    "vocab_size": vocab_size,
    "n_draft_layers": N_DRAFT_LAYERS,
    "n_kernels": n_total_kernels,
    "tok_per_sec": tok_per_sec,
}
with open(os.path.join(os.path.dirname(__file__), "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)
print(f"\n  Model info saved to model_info.json")

# Save numpy weights for fast reload
np.savez_compressed(
    os.path.join(os.path.dirname(__file__), "draft_weights.npz"),
    embed=embed_w,
    final_norm=final_norm_w,
    lm_head=lm_head_w,
    **{f"l{l}_{k}": v for l in range(N_DRAFT_LAYERS) for k, v in layer_weights[l].items()},
)
print(f"  Weights saved to draft_weights.npz")

print(f"\n  ⚡ Real ANE draft model ready for speculative decoding!")
