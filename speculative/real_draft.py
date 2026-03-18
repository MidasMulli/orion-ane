"""
Real ANE Draft Model — Qwen3 models running on Apple Neural Engine
==================================================================

Loads real model weights, compiles ANE kernels, and provides
token-level generation for speculative decoding.

Supports any Qwen3 dense model (0.6B, 1.7B, 4B, 8B) — architecture
is auto-detected from HuggingFace config.

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
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
ROPE_THETA = 1000000.0
ANE_SPATIAL = 16   # ANE minimum spatial dimension
CLS_CHUNK = 16000  # Classifier tiling chunk size (was 8000 → fewer ANE dispatches)
MAX_SEQ = 64       # Maximum sequence length for KV cache


class RealDraftModel:
    """Qwen3 model running on Apple Neural Engine. Supports 0.6B, 1.7B, 4B."""

    def __init__(self, model_name=None):
        self.model_name = model_name or DEFAULT_MODEL
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
        self._rope_freqs = None  # Precomputed RoPE frequencies

        # C forward pass (optional acceleration)
        self._c_model = None
        self._c_lib = None

    def load_and_compile(self, status_fn=None, fused=False, weights_path=None):
        """Download model, extract weights, compile all ANE kernels.
        If fused=True, compile fused QKV and Gate+Up kernels (fewer dispatches).
        If weights_path is set, load weights from local safetensors instead of HuggingFace."""
        self.fused = fused
        def status(msg):
            if status_fn:
                status_fn(msg)
            print(f"  {msg}")

        if weights_path:
            # ── Load from local safetensors ──
            status(f"Loading weights from {weights_path}...")
            t0 = time.time()

            from safetensors.numpy import load_file
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            sf_path = weights_path
            if os.path.isdir(sf_path):
                sf_path = os.path.join(sf_path, "model.safetensors")
            state = load_file(sf_path)

            # Auto-detect architecture from HuggingFace config
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self.dim = config.hidden_size
            self.n_heads = config.num_attention_heads
            self.n_kv_heads = getattr(config, 'num_key_value_heads', self.n_heads)
            self.head_dim = getattr(config, 'head_dim', self.dim // self.n_heads)
            self.hidden_dim = config.intermediate_size
            self.vocab_size = config.vocab_size
            self.n_layers = config.num_hidden_layers

            status(f"Loaded {len(state)} tensors from safetensors in {time.time()-t0:.1f}s")
            status(f"Architecture: dim={self.dim}, heads={self.n_heads}, "
                   f"kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
                   f"layers={self.n_layers}")

            self.embed_w = state['model.embed_tokens.weight']
            self.final_norm_w = state['model.norm.weight']
            # Handle tied embeddings (0.6B, 1.7B, 4B use tie_word_embeddings=true)
            if 'lm_head.weight' in state:
                lm_head_w = state['lm_head.weight']
            else:
                lm_head_w = self.embed_w  # Tied to embed_tokens
        else:
            # ── Step 1: Download safetensors directly (memory-efficient) ──
            status(f"Downloading {self.model_name}...")
            t0 = time.time()

            from transformers import AutoTokenizer, AutoConfig
            from huggingface_hub import snapshot_download
            from safetensors.numpy import load_file
            import glob as globmod

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            # Download model files (safetensors + config)
            model_dir = snapshot_download(self.model_name)
            status(f"Model downloaded in {time.time()-t0:.1f}s")

            # ── Step 2: Architecture from config ──
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
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

            # ── Step 3: Load weights from safetensors ──
            # Use torch loader for bfloat16 support, then convert to numpy fp32
            status("Loading safetensors weights...")
            try:
                from safetensors.numpy import load_file
                sf_files = sorted(globmod.glob(os.path.join(model_dir, "*.safetensors")))
                if not sf_files:
                    status("FATAL: No safetensors files found")
                    return False
                state = {}
                for sf in sf_files:
                    state.update(load_file(sf))
            except TypeError:
                # bfloat16 not supported by numpy — fall back to torch loader
                import torch
                from safetensors.torch import load_file as torch_load_file
                sf_files = sorted(globmod.glob(os.path.join(model_dir, "*.safetensors")))
                if not sf_files:
                    status("FATAL: No safetensors files found")
                    return False
                state = {}
                for sf in sf_files:
                    shard = torch_load_file(sf)
                    # Convert to numpy fp32 immediately to free torch tensors
                    for k, v in shard.items():
                        state[k] = v.float().numpy()
                    del shard
            status(f"Loaded {len(state)} tensors from {len(sf_files)} safetensors file(s)")

            self.embed_w = state['model.embed_tokens.weight']
            self.final_norm_w = state['model.norm.weight']
            # Handle tied embeddings (0.6B, 1.7B, 4B use tie_word_embeddings=true)
            if 'lm_head.weight' in state:
                lm_head_w = state['lm_head.weight']
            else:
                lm_head_w = self.embed_w  # Tied to embed_tokens

        # Ensure embed/norm weights are fp32 (safetensors may store as fp16)
        self.embed_w = np.array(self.embed_w, dtype=np.float32)
        self.final_norm_w = np.array(self.final_norm_w, dtype=np.float32)
        lm_head_w = np.array(lm_head_w, dtype=np.float32)

        # Helper: convert to fp32 numpy whether torch tensor, numpy fp16, or fp32
        def to_np(x):
            if hasattr(x, 'numpy'):
                x = x.numpy()
            return np.array(x, dtype=np.float32)

        self.layer_weights = []
        for l in range(self.n_layers):
            prefix = f'model.layers.{l}'
            lw = {
                'q': to_np(state[f'{prefix}.self_attn.q_proj.weight']),
                'k': to_np(state[f'{prefix}.self_attn.k_proj.weight']),
                'v': to_np(state[f'{prefix}.self_attn.v_proj.weight']),
                'o': to_np(state[f'{prefix}.self_attn.o_proj.weight']),
                'gate': to_np(state[f'{prefix}.mlp.gate_proj.weight']),
                'up': to_np(state[f'{prefix}.mlp.up_proj.weight']),
                'down': to_np(state[f'{prefix}.mlp.down_proj.weight']),
                'attn_norm': to_np(state[f'{prefix}.input_layernorm.weight']),
                'ffn_norm': to_np(state[f'{prefix}.post_attention_layernorm.weight']),
                'q_norm': to_np(state[f'{prefix}.self_attn.q_norm.weight']),
                'k_norm': to_np(state[f'{prefix}.self_attn.k_norm.weight']),
            }
            self.layer_weights.append(lw)

        del state
        import gc; gc.collect()
        status("Weights extracted")

        # ── Step 4: Init ANE ──
        if not init_ane():
            status("ANE bridge init FAILED")
            return False

        # ── Step 5: Compile kernels ──
        kpl = 4 if fused else 7
        n_cls = (self.vocab_size + CLS_CHUNK - 1) // CLS_CHUNK
        status(f"Compiling {self.n_layers * kpl + n_cls} ANE kernels ({'fused' if fused else 'unfused'})...")
        t_compile = time.time()

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        for l in range(self.n_layers):
            lw = self.layer_weights[l]
            layer_kernels = {}

            if fused:
                # Fused QKV: concatenate Q, K, V weights → single kernel
                qkv_dim = q_dim + kv_dim + kv_dim
                w_qkv = np.concatenate([lw['q'], lw['k'], lw['v']], axis=0)
                mil, wb = generate_conv_mil_with_weights(
                    self.dim, qkv_dim, ANE_SPATIAL, w_qkv, f"real_l{l}_qkv")
                k = compile_ane_kernel(
                    mil, wb, [self.dim * ANE_SPATIAL * 4], [qkv_dim * ANE_SPATIAL * 4])
                if not k:
                    status(f"Kernel l{l}_qkv FAILED")
                    return False
                layer_kernels['qkv'] = k

                # Fused Gate+Up: concatenate gate, up weights → single kernel
                gate_up_dim = self.hidden_dim * 2
                w_gate_up = np.concatenate([lw['gate'], lw['up']], axis=0)
                mil, wb = generate_conv_mil_with_weights(
                    self.dim, gate_up_dim, ANE_SPATIAL, w_gate_up, f"real_l{l}_gate_up")
                k = compile_ane_kernel(
                    mil, wb, [self.dim * ANE_SPATIAL * 4], [gate_up_dim * ANE_SPATIAL * 4])
                if not k:
                    status(f"Kernel l{l}_gate_up FAILED")
                    return False
                layer_kernels['gate_up'] = k

                # O and Down are not fused (different input dims)
                for name, w, in_d, out_d in [
                    ('o', lw['o'], q_dim, self.dim),
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
            else:
                # Separate kernels (original mode)
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

        # Precompute RoPE frequency table
        i_vals = np.arange(0, self.head_dim, 2, dtype=np.float32)
        self._rope_freqs = (1.0 / (ROPE_THETA ** (i_vals / self.head_dim))).astype(np.float32)

        self.compiled = True
        status(f"{self.n_kernels} kernels compiled in {self.compile_time_ms:.0f}ms")

        # ── Step 6: Initialize C forward pass ──
        self._init_c_forward(status)

        return True

    # ── C Forward Pass Acceleration ────────────────────────────────────

    def _init_c_forward(self, status):
        """Try to load libane_bridge.dylib and init C forward model."""
        bridge_path = os.path.join(os.path.dirname(__file__), "..", "bridge", "libane_bridge.dylib")
        if not os.path.exists(bridge_path):
            status("C bridge not found — using Python forward pass")
            return

        try:
            c_lib = ctypes.CDLL(bridge_path)

            # Define the config struct for C
            class FPModelConfig(ctypes.Structure):
                _fields_ = [
                    ("dim", ctypes.c_int), ("n_heads", ctypes.c_int),
                    ("n_kv_heads", ctypes.c_int), ("head_dim", ctypes.c_int),
                    ("hidden_dim", ctypes.c_int), ("vocab_size", ctypes.c_int),
                    ("n_layers", ctypes.c_int), ("max_seq", ctypes.c_int),
                    ("ane_spatial", ctypes.c_int), ("rope_theta", ctypes.c_float),
                ]

            # Set up function signatures
            c_lib.forward_model_create.argtypes = [ctypes.POINTER(FPModelConfig)]
            c_lib.forward_model_create.restype = ctypes.c_void_p
            c_lib.forward_model_free.argtypes = [ctypes.c_void_p]
            c_lib.forward_model_set_embed.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            c_lib.forward_model_set_final_norm.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            c_lib.forward_model_set_layer_weights.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            c_lib.forward_model_set_layer_kernels.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            c_lib.forward_model_set_layer_kernels_fused.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            c_lib.forward_model_add_cls_kernel.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            c_lib.forward_model_reset_cache.argtypes = [ctypes.c_void_p]
            c_lib.forward_model_forward_token.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            c_lib.forward_model_forward_token.restype = ctypes.POINTER(ctypes.c_float)

            # Create C model
            config = FPModelConfig(
                dim=self.dim, n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim, hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size, n_layers=self.n_layers,
                max_seq=MAX_SEQ, ane_spatial=ANE_SPATIAL, rope_theta=ROPE_THETA)

            c_model = c_lib.forward_model_create(ctypes.byref(config))
            if not c_model:
                status("C forward_model_create failed — using Python forward pass")
                return

            # Transfer weights
            embed_flat = self.embed_w.astype(np.float32).ravel()
            c_lib.forward_model_set_embed(c_model,
                embed_flat.ctypes.data_as(ctypes.c_void_p))

            fn = self.final_norm_w.astype(np.float32)
            c_lib.forward_model_set_final_norm(c_model,
                fn.ctypes.data_as(ctypes.c_void_p))

            for l in range(self.n_layers):
                lw = self.layer_weights[l]
                lk = self.kernels[l]

                an = lw['attn_norm'].astype(np.float32)
                ffn = lw['ffn_norm'].astype(np.float32)
                qn = lw['q_norm'].astype(np.float32)
                kn = lw['k_norm'].astype(np.float32)

                c_lib.forward_model_set_layer_weights(c_model, l,
                    an.ctypes.data_as(ctypes.c_void_p),
                    ffn.ctypes.data_as(ctypes.c_void_p),
                    qn.ctypes.data_as(ctypes.c_void_p),
                    kn.ctypes.data_as(ctypes.c_void_p))

                if self.fused:
                    c_lib.forward_model_set_layer_kernels_fused(c_model, l,
                        lk['qkv'], lk['o'], lk['gate_up'], lk['down'])
                else:
                    c_lib.forward_model_set_layer_kernels(c_model, l,
                        lk['q'], lk['k'], lk['v'], lk['o'],
                        lk['gate'], lk['up'], lk['down'])

            for kernel, out_ch in self.cls_kernels:
                c_lib.forward_model_add_cls_kernel(c_model, kernel, out_ch)

            self._c_model = c_model
            self._c_lib = c_lib
            status("C forward pass initialized ✓ (zero Python in hot loop)")

        except Exception as e:
            status(f"C forward init failed: {e} — using Python forward pass")
            self._c_model = None
            self._c_lib = None

    # ── Forward pass primitives ─────────────────────────────────────────

    @staticmethod
    def _rmsnorm(x, w, eps=1e-6):
        ss = np.mean(x * x) + eps
        return x / np.sqrt(ss) * w

    def _rope(self, x, pos, n_h, hd):
        angles = pos * self._rope_freqs        # (hd//2,)
        cos_v = np.cos(angles)                  # (hd//2,)
        sin_v = np.sin(angles)                  # (hd//2,)
        x_r = x.reshape(n_h, hd)               # (n_h, hd)
        x_even = x_r[:, 0::2]                  # (n_h, hd//2)
        x_odd = x_r[:, 1::2]                   # (n_h, hd//2)
        out = np.empty_like(x_r)
        out[:, 0::2] = x_even * cos_v - x_odd * sin_v
        out[:, 1::2] = x_even * sin_v + x_odd * cos_v
        return out.reshape(-1)

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
        hd = self.head_dim
        heads_per_kv = self.n_heads // self.n_kv_heads
        seq = pos + 1

        # Reshape Q: [n_heads, head_dim]
        q_heads = q.reshape(self.n_heads, hd)

        # K/V cache: [seq, kv_dim] → [seq, n_kv_heads, head_dim]
        k_seq = k_cache[:seq].reshape(seq, self.n_kv_heads, hd)
        v_seq = v_cache[:seq].reshape(seq, self.n_kv_heads, hd)

        # Expand KV heads to match query heads via GQA repeat
        # k_exp: [seq, n_heads, head_dim]
        k_exp = np.repeat(k_seq, heads_per_kv, axis=1)
        v_exp = np.repeat(v_seq, heads_per_kv, axis=1)

        # Scores: [n_heads, seq] = Q[n_heads, hd] @ K[seq, n_heads, hd]^T
        # einsum: 'nh,snh->ns'
        scores = np.einsum('nh,snh->ns', q_heads, k_exp) * scale

        # Softmax per head
        scores -= scores.max(axis=1, keepdims=True)
        scores = np.exp(scores)
        scores /= scores.sum(axis=1, keepdims=True)

        # Weighted V: [n_heads, hd] = scores[n_heads, seq] @ V[seq, n_heads, hd]
        out = np.einsum('ns,snh->nh', scores, v_exp)
        return out.reshape(-1)

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
        """Single token forward pass → logits. Routes to C if available."""
        if self._c_model is not None:
            ptr = self._c_lib.forward_model_forward_token(
                self._c_model, int(token_id), int(pos))
            return np.ctypeslib.as_array(ptr, shape=(self.vocab_size,)).copy()

        # Python fallback
        x = self.embed_w[token_id].copy()
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        for l in range(self.n_layers):
            lw = self.layer_weights[l]
            lk = self.kernels[l]

            # Attention
            xn = self._rmsnorm(x, lw['attn_norm'])
            if self.fused:
                qkv = self._ane_linear(lk['qkv'], xn, self.dim, q_dim + kv_dim + kv_dim)
                q = qkv[:q_dim].copy()
                k = qkv[q_dim:q_dim + kv_dim].copy()
                v = qkv[q_dim + kv_dim:].copy()
            else:
                q = self._ane_linear(lk['q'], xn, self.dim, q_dim)
                k = self._ane_linear(lk['k'], xn, self.dim, kv_dim)
                v = self._ane_linear(lk['v'], xn, self.dim, kv_dim)

            # QK-norm (Qwen3 specific) — vectorized across all heads
            q_r = q.reshape(self.n_heads, self.head_dim)
            q_ss = np.mean(q_r * q_r, axis=1, keepdims=True) + 1e-6
            q[:] = (q_r / np.sqrt(q_ss) * lw['q_norm']).reshape(-1)
            k_r = k.reshape(self.n_kv_heads, self.head_dim)
            k_ss = np.mean(k_r * k_r, axis=1, keepdims=True) + 1e-6
            k[:] = (k_r / np.sqrt(k_ss) * lw['k_norm']).reshape(-1)

            q = self._rope(q, pos, self.n_heads, self.head_dim)
            k = self._rope(k, pos, self.n_kv_heads, self.head_dim)

            self.k_caches[l][pos] = k
            self.v_caches[l][pos] = v

            attn = self._attention(q, self.k_caches[l], self.v_caches[l], pos)
            o = self._ane_linear(lk['o'], attn, q_dim, self.dim)
            x = x + o

            # FFN (SwiGLU)
            xn = self._rmsnorm(x, lw['ffn_norm'])
            if self.fused:
                gate_up = self._ane_linear(lk['gate_up'], xn, self.dim, self.hidden_dim * 2)
                gate = gate_up[:self.hidden_dim].copy()
                up = gate_up[self.hidden_dim:].copy()
            else:
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
        if self._c_model is not None:
            self._c_lib.forward_model_reset_cache(self._c_model)
            return
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
        Generate K draft tokens from prompt (full reset + prefill).
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

    # ── Incremental draft API (for speculative decoding) ─────────────

    def prefill(self, token_ids):
        """Reset cache and prefill prompt. Returns logits from last token."""
        self.reset_cache()
        self._draft_pos = 0
        logits = None
        for tid in token_ids:
            logits = self.forward_token(int(tid), self._draft_pos)
            self._draft_pos += 1
        self._last_logits = logits
        return logits

    def draft_continue(self, k_draft):
        """
        Draft K tokens from current cache position (no reset, no re-prefill).
        Returns list of (token_id, logits_top5) tuples.
        """
        logits = self._last_logits
        drafts = []
        for _ in range(k_draft):
            tok = int(np.argmax(logits))
            top5_idx = np.argsort(logits)[-5:][::-1]
            top5 = [(int(idx), float(logits[idx])) for idx in top5_idx]
            drafts.append((tok, top5))
            logits = self.forward_token(tok, self._draft_pos)
            self._draft_pos += 1
        self._last_logits = logits
        return drafts

    def rollback_to(self, pos):
        """
        Roll back draft position after rejection.
        The KV cache entries beyond `pos` are stale but harmless —
        they'll be overwritten when we forward_token at those positions.
        We just need to reset the position counter.
        """
        self._draft_pos = pos

    def feed_tokens(self, token_ids, start_pos):
        """
        Feed correction tokens into the cache at specific positions.
        Used after GPU verification rejects some draft tokens.
        """
        for i, tid in enumerate(token_ids):
            logits = self.forward_token(int(tid), start_pos + i)
        self._draft_pos = start_pos + len(token_ids)
        self._last_logits = logits
        return logits
