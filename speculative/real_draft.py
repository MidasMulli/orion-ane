"""
Real ANE Draft Model — Qwen3-0.6B running on Apple Neural Engine
================================================================

Loads real model weights, compiles 215 ANE kernels, and provides
token-level generation for speculative decoding.

All linear projections execute on the Neural Engine via direct
_ANEClient dispatch. CPU handles element-wise ops (RMSNorm, RoPE,
attention, SwiGLU activation).
"""

import os
import sys
import time
import json
import numpy as np
import ctypes

sys.path.insert(0, os.path.dirname(__file__))
from ane_draft import init_ane, lib, compile_ane_kernel, generate_conv_mil_with_weights

# ── Constants ─────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-0.6B"
ROPE_THETA = 1000000.0
ANE_SPATIAL = 16   # ANE minimum spatial dimension
CLS_CHUNK = 8000   # Classifier tiling chunk size
MAX_SEQ = 64       # Maximum sequence length for KV cache


class RealDraftModel:
    """Full Qwen3-0.6B running on Apple Neural Engine."""

    def __init__(self):
        self.dim = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.head_dim = 0
        self.hidden_dim = 0
        self.vocab_size = 0
        self.n_layers = 0

        self.embed_w = None
        self.final_norm_w = None
        self.layer_weights = []
        self.kernels = {}
        self.cls_kernels = []
        self.tokenizer = None

        self.k_caches = []
        self.v_caches = []

        self.compiled = False
        self.compile_time_ms = 0
        self.n_kernels = 0

    def load_and_compile(self, status_fn=None):
        """Download model, extract weights, compile all ANE kernels."""
        def status(msg):
            if status_fn:
                status_fn(msg)
            print(f"  {msg}")

        # ── Step 1: Download ──
        status("Downloading Qwen3-0.6B...")
        t0 = time.time()

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
        model.eval()

        status(f"Model downloaded in {time.time()-t0:.1f}s")

        # ── Step 2: Architecture ──
        config = model.config
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = getattr(config, 'num_key_value_heads', self.n_heads)
        self.head_dim = getattr(config, 'head_dim', self.dim // self.n_heads)
        self.hidden_dim = config.intermediate_size
        self.vocab_size = config.vocab_size
        self.n_layers = config.num_hidden_layers

        status(f"Architecture: dim={self.dim}, heads={self.n_heads}, "
               f"kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
               f"layers={self.n_layers}")

        # ── Step 3: Extract weights ──
        status("Extracting weights...")
        state = model.state_dict()

        self.embed_w = state['model.embed_tokens.weight'].numpy()
        self.final_norm_w = state['model.norm.weight'].numpy()
        lm_head_w = state['lm_head.weight'].numpy()

        self.layer_weights = []
        for l in range(self.n_layers):
            prefix = f'model.layers.{l}'
            lw = {
                'q': state[f'{prefix}.self_attn.q_proj.weight'].numpy(),
                'k': state[f'{prefix}.self_attn.k_proj.weight'].numpy(),
                'v': state[f'{prefix}.self_attn.v_proj.weight'].numpy(),
                'o': state[f'{prefix}.self_attn.o_proj.weight'].numpy(),
                'gate': state[f'{prefix}.mlp.gate_proj.weight'].numpy(),
                'up': state[f'{prefix}.mlp.up_proj.weight'].numpy(),
                'down': state[f'{prefix}.mlp.down_proj.weight'].numpy(),
                'attn_norm': state[f'{prefix}.input_layernorm.weight'].numpy(),
                'ffn_norm': state[f'{prefix}.post_attention_layernorm.weight'].numpy(),
                'q_norm': state[f'{prefix}.self_attn.q_norm.weight'].numpy(),
                'k_norm': state[f'{prefix}.self_attn.k_norm.weight'].numpy(),
            }
            self.layer_weights.append(lw)

        del model, state
        import gc; gc.collect()
        status("Weights extracted, PyTorch model freed")

        # ── Step 4: Init ANE ──
        if not init_ane():
            status("ANE bridge init FAILED")
            return False

        # ── Step 5: Compile kernels ──
        status(f"Compiling {self.n_layers * 7 + (self.vocab_size // CLS_CHUNK + 1)} ANE kernels...")
        t_compile = time.time()

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        for l in range(self.n_layers):
            lw = self.layer_weights[l]
            layer_kernels = {}

            for name, w, in_d, out_d in [
                ('q', lw['q'], self.dim, q_dim),
                ('k', lw['k'], self.dim, kv_dim),
                ('v', lw['v'], self.dim, kv_dim),
                ('o', lw['o'], q_dim, self.dim),
                ('gate', lw['gate'], self.dim, self.hidden_dim),
                ('up', lw['up'], self.dim, self.hidden_dim),
                ('down', lw['down'], self.hidden_dim, self.dim),
            ]:
                mil, wb = generate_conv_mil_with_weights(
                    in_d, out_d, ANE_SPATIAL, w, f"real_l{l}_{name}")
                k = compile_ane_kernel(
                    mil, wb, [in_d * ANE_SPATIAL * 4], [out_d * ANE_SPATIAL * 4])
                if not k:
                    status(f"Kernel l{l}_{name} FAILED")
                    return False
                layer_kernels[name] = k

            self.kernels[l] = layer_kernels
            if l % 7 == 0 or l == self.n_layers - 1:
                status(f"Layer {l}/{self.n_layers} compiled")

        # Classifier tiles
        n_chunks = (self.vocab_size + CLS_CHUNK - 1) // CLS_CHUNK
        for ci in range(n_chunks):
            start = ci * CLS_CHUNK
            end = min(start + CLS_CHUNK, self.vocab_size)
            out_ch = end - start
            w_chunk = lm_head_w[start:end]
            mil, wb = generate_conv_mil_with_weights(
                self.dim, out_ch, ANE_SPATIAL, w_chunk, f"real_cls_{ci}")
            k = compile_ane_kernel(
                mil, wb, [self.dim * ANE_SPATIAL * 4], [out_ch * ANE_SPATIAL * 4])
            if not k:
                status(f"Classifier chunk {ci} FAILED")
                return False
            self.cls_kernels.append((k, out_ch))

        self.compile_time_ms = (time.time() - t_compile) * 1000
        self.n_kernels = sum(len(v) for v in self.kernels.values()) + len(self.cls_kernels)
        self.compiled = True
        status(f"{self.n_kernels} kernels compiled in {self.compile_time_ms:.0f}ms")
        return True

    # ── Forward pass primitives ─────────────────────────────────────────

    @staticmethod
    def _rmsnorm(x, w, eps=1e-6):
        ss = np.mean(x * x) + eps
        return x / np.sqrt(ss) * w

    @staticmethod
    def _rope(x, pos, n_h, hd):
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

    @staticmethod
    def _ane_linear(kernel, x, in_d, out_d):
        x_padded = np.zeros((in_d, ANE_SPATIAL), dtype=np.float32)
        x_padded[:, 0] = x.astype(np.float32)
        x_padded = x_padded.ravel()
        out_padded = np.zeros(out_d * ANE_SPATIAL, dtype=np.float32)
        lib.ane_bridge_write_input(kernel, 0,
            x_padded.ctypes.data_as(ctypes.c_void_p), in_d * ANE_SPATIAL * 4)
        lib.ane_bridge_eval(kernel)
        lib.ane_bridge_read_output(kernel, 0,
            out_padded.ctypes.data_as(ctypes.c_void_p), out_d * ANE_SPATIAL * 4)
        return out_padded.reshape(out_d, ANE_SPATIAL)[:, 0]

    def _attention(self, q, k_cache, v_cache, pos):
        scale = 1.0 / np.sqrt(self.head_dim)
        out = np.zeros(self.n_heads * self.head_dim, dtype=np.float32)
        heads_per_kv = self.n_heads // self.n_kv_heads

        for h in range(self.n_heads):
            kv_h = h // heads_per_kv
            q_h = q[h * self.head_dim:(h + 1) * self.head_dim]
            scores = np.zeros(pos + 1, dtype=np.float32)
            for s in range(pos + 1):
                k_h = k_cache[s, kv_h * self.head_dim:(kv_h + 1) * self.head_dim]
                scores[s] = np.dot(q_h, k_h) * scale
            scores -= scores.max()
            scores = np.exp(scores)
            scores /= scores.sum()
            for s in range(pos + 1):
                v_h = v_cache[s, kv_h * self.head_dim:(kv_h + 1) * self.head_dim]
                out[h * self.head_dim:(h + 1) * self.head_dim] += scores[s] * v_h
        return out

    def _classify(self, x):
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        offset = 0
        for kernel, out_ch in self.cls_kernels:
            chunk = self._ane_linear(kernel, x, self.dim, out_ch)
            logits[offset:offset + out_ch] = chunk
            offset += out_ch
        return logits

    # ── Token-level forward ─────────────────────────────────────────────

    def forward_token(self, token_id, pos):
        """Single token forward pass → logits."""
        x = self.embed_w[token_id].copy()
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        for l in range(self.n_layers):
            lw = self.layer_weights[l]
            lk = self.kernels[l]

            # Attention
            xn = self._rmsnorm(x, lw['attn_norm'])
            q = self._ane_linear(lk['q'], xn, self.dim, q_dim)
            k = self._ane_linear(lk['k'], xn, self.dim, kv_dim)
            v = self._ane_linear(lk['v'], xn, self.dim, kv_dim)

            # QK-norm (Qwen3 specific)
            for h in range(self.n_heads):
                s, e = h * self.head_dim, (h+1) * self.head_dim
                q[s:e] = self._rmsnorm(q[s:e], lw['q_norm'])
            for h in range(self.n_kv_heads):
                s, e = h * self.head_dim, (h+1) * self.head_dim
                k[s:e] = self._rmsnorm(k[s:e], lw['k_norm'])

            q = self._rope(q, pos, self.n_heads, self.head_dim)
            k = self._rope(k, pos, self.n_kv_heads, self.head_dim)

            self.k_caches[l][pos] = k
            self.v_caches[l][pos] = v

            attn = self._attention(q, self.k_caches[l], self.v_caches[l], pos)
            o = self._ane_linear(lk['o'], attn, q_dim, self.dim)
            x = x + o

            # FFN (SwiGLU)
            xn = self._rmsnorm(x, lw['ffn_norm'])
            gate = self._ane_linear(lk['gate'], xn, self.dim, self.hidden_dim)
            up = self._ane_linear(lk['up'], xn, self.dim, self.hidden_dim)
            silu_gate = gate / (1.0 + np.exp(-np.clip(gate, -20, 20)))
            down = self._ane_linear(lk['down'], silu_gate * up, self.hidden_dim, self.dim)
            x = x + down

        x = self._rmsnorm(x, self.final_norm_w)
        return self._classify(x)

    # ── Generation API ──────────────────────────────────────────────────

    def reset_cache(self):
        """Reset KV caches for new generation."""
        kv_dim = self.n_kv_heads * self.head_dim
        self.k_caches = [np.zeros((MAX_SEQ, kv_dim), dtype=np.float32)
                         for _ in range(self.n_layers)]
        self.v_caches = [np.zeros((MAX_SEQ, kv_dim), dtype=np.float32)
                         for _ in range(self.n_layers)]

    def encode(self, text):
        """Tokenize text to token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def generate_draft(self, prompt_ids, k_draft):
        """
        Generate K draft tokens from prompt.
        Returns list of (token_id, logits_top5) tuples.
        """
        self.reset_cache()

        # Process prompt
        for i, tid in enumerate(prompt_ids):
            logits = self.forward_token(tid, i)

        # Generate K tokens
        pos = len(prompt_ids)
        drafts = []
        for _ in range(k_draft):
            tok = int(np.argmax(logits))
            # Get top-5 for diagnostics
            top5_idx = np.argsort(logits)[-5:][::-1]
            top5 = [(int(idx), float(logits[idx])) for idx in top5_idx]
            drafts.append((tok, top5))
            logits = self.forward_token(tok, pos)
            pos += 1

        return drafts
