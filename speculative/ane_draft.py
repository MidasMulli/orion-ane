"""
ANE Draft Model for Speculative Decoding
Runs a tiny transformer on the Neural Engine via _ANEClient private APIs.

Architecture: Embedding → N transformer layers (ANE matmuls) → Classifier
All linear projections run on ANE, element-wise ops on CPU.
"""

import ctypes
import numpy as np
import os
import sys
import time
import math
import subprocess

# ── Load the bridge ─────────────────────────────────────────────────────
BRIDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "bridge")
BRIDGE_PATH = os.path.join(BRIDGE_DIR, "libane_bridge.dylib")
lib = ctypes.CDLL(BRIDGE_PATH)

lib.ane_bridge_init.restype = ctypes.c_int
lib.ane_bridge_compile.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
]
lib.ane_bridge_compile.restype = ctypes.c_void_p
lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
lib.ane_bridge_eval.restype = ctypes.c_bool
lib.ane_bridge_write_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.ane_bridge_read_output.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.ane_bridge_free.argtypes = [ctypes.c_void_p]


def _ensure_dummy_weight(weight_blob):
    """Always provide a weight blob (macOS 26 bridge quirk)."""
    if weight_blob is None:
        weight_blob = bytearray(128)
        weight_blob[0] = 0x01; weight_blob[4] = 0x02
        weight_blob[64] = 0xEF; weight_blob[65] = 0xBE
        weight_blob[66] = 0xAD; weight_blob[67] = 0xDE
        weight_blob[68] = 0x01
        weight_blob = bytes(weight_blob)
    return weight_blob


def compile_ane_kernel(mil_bytes, weight_blob, in_sizes, out_sizes):
    """Compile a MIL program to an ANE kernel."""
    weight_blob = _ensure_dummy_weight(weight_blob)
    wb = ctypes.create_string_buffer(weight_blob)
    n_in = len(in_sizes)
    n_out = len(out_sizes)
    in_arr = (ctypes.c_size_t * n_in)(*in_sizes)
    out_arr = (ctypes.c_size_t * n_out)(*out_sizes)
    kernel = lib.ane_bridge_compile(mil_bytes, len(mil_bytes), wb, len(weight_blob),
                                     n_in, in_arr, n_out, out_arr)
    return kernel


def generate_conv_mil(in_ch, out_ch, spatial, model_name):
    """Generate MIL for conv1x1 with baked weights via coremltools."""
    import coremltools as ct
    from coremltools.converters.mil.mil import Builder as mb

    weights_np = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * 0.01

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, in_ch, 1, spatial))])
    def prog(x):
        W = mb.const(val=weights_np, name="W")
        return mb.conv(x=x, weight=W, name="conv_out")

    model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15)
    pkg_path = f"/tmp/{model_name}.mlpackage"
    model.save(pkg_path)
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg_path, "/tmp/"],
                   capture_output=True)
    compiled = f"/tmp/{model_name}.mlmodelc"
    with open(f"{compiled}/model.mil", 'rb') as f:
        mil_bytes = f.read()
    weight_path = f"{compiled}/weights/weight.bin"
    weight_blob = None
    if os.path.exists(weight_path):
        with open(weight_path, 'rb') as f:
            weight_blob = f.read()
    return mil_bytes, weight_blob, weights_np


def generate_conv_mil_with_weights(in_ch, out_ch, spatial, weights_np, model_name):
    """Generate MIL for conv1x1 with specific weights."""
    import coremltools as ct
    from coremltools.converters.mil.mil import Builder as mb

    w = weights_np.reshape(out_ch, in_ch, 1, 1).astype(np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, in_ch, 1, spatial))])
    def prog(x):
        W = mb.const(val=w, name="W")
        return mb.conv(x=x, weight=W, name="conv_out")

    model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15)
    pkg_path = f"/tmp/{model_name}.mlpackage"
    model.save(pkg_path)
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg_path, "/tmp/"],
                   capture_output=True)
    compiled = f"/tmp/{model_name}.mlmodelc"
    with open(f"{compiled}/model.mil", 'rb') as f:
        mil_bytes = f.read()
    weight_path = f"{compiled}/weights/weight.bin"
    weight_blob = None
    if os.path.exists(weight_path):
        with open(weight_path, 'rb') as f:
            weight_blob = f.read()
    return mil_bytes, weight_blob


class ANEDraftModel:
    """
    Tiny transformer draft model running on ANE.

    Architecture per layer:
    - RMSNorm (CPU)
    - QKV projection (ANE conv1x1)
    - RoPE (CPU)
    - Attention (CPU — small seq len)
    - Output projection (ANE conv1x1)
    - RMSNorm (CPU)
    - FFN up w1, w3 (ANE conv1x1)
    - SiLU gate (CPU)
    - FFN down w2 (ANE conv1x1)

    For speculative decoding: generates K draft tokens autoregressively.
    """

    def __init__(self, dim=256, hidden_dim=512, n_heads=4, n_layers=2,
                 vocab_size=32000, max_seq=64):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.kernels = {}
        self.initialized = False

        # Weight storage
        self.embed_weights = None  # [vocab_size, dim]
        self.rms_att_w = []       # [n_layers][dim]
        self.rms_ffn_w = []       # [n_layers][dim]
        self.rms_final_w = None   # [dim]
        # Linear weights stored in ANE kernels (baked)

        # KV cache for autoregressive generation
        self.k_cache = []  # [n_layers][max_seq, dim]
        self.v_cache = []  # [n_layers][max_seq, dim]
        self.cache_pos = 0

    def init_random_weights(self):
        """Initialize with random weights (for pipeline testing)."""
        print(f"  Initializing random draft model: dim={self.dim}, layers={self.n_layers}, "
              f"vocab={self.vocab_size}")
        self.embed_weights = np.random.randn(self.vocab_size, self.dim).astype(np.float32) * 0.02
        self.rms_att_w = [np.ones(self.dim, dtype=np.float32) for _ in range(self.n_layers)]
        self.rms_ffn_w = [np.ones(self.dim, dtype=np.float32) for _ in range(self.n_layers)]
        self.rms_final_w = np.ones(self.dim, dtype=np.float32)

    def compile_kernels(self, spatial=1):
        """Compile all ANE kernels for the model."""
        print(f"  Compiling ANE kernels (spatial={spatial})...")
        t0 = time.time()
        d, hd = self.dim, self.hidden_dim

        for layer in range(self.n_layers):
            # QKV projections
            for name in ['q', 'k', 'v', 'o']:
                kern_name = f"l{layer}_{name}"
                mil, wb = generate_conv_mil_with_weights(
                    d, d, spatial,
                    np.random.randn(d, d).astype(np.float32) * 0.02,
                    f"draft_{kern_name}")
                in_bytes = d * spatial * 4
                out_bytes = d * spatial * 4
                k = compile_ane_kernel(mil, wb, [in_bytes], [out_bytes])
                if not k:
                    print(f"    ❌ {kern_name} compile failed")
                    return False
                self.kernels[kern_name] = k

            # FFN w1 (d → hidden_dim), w3 (d → hidden_dim), w2 (hidden_dim → d)
            for name, in_d, out_d in [('w1', d, hd), ('w3', d, hd), ('w2', hd, d)]:
                kern_name = f"l{layer}_{name}"
                mil, wb = generate_conv_mil_with_weights(
                    in_d, out_d, spatial,
                    np.random.randn(out_d, in_d).astype(np.float32) * 0.02,
                    f"draft_{kern_name}")
                in_bytes = in_d * spatial * 4
                out_bytes = out_d * spatial * 4
                k = compile_ane_kernel(mil, wb, [in_bytes], [out_bytes])
                if not k:
                    print(f"    ❌ {kern_name} compile failed")
                    return False
                self.kernels[kern_name] = k

        # Classifier (d → vocab)
        # For large vocab, tile into chunks
        chunk_size = 8000  # max output channels per ANE kernel
        n_chunks = (self.vocab_size + chunk_size - 1) // chunk_size
        self.cls_chunks = []
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, self.vocab_size)
            out_ch = end - start
            kern_name = f"cls_{ci}"
            mil, wb = generate_conv_mil_with_weights(
                d, out_ch, spatial,
                np.random.randn(out_ch, d).astype(np.float32) * 0.02,
                f"draft_{kern_name}")
            in_bytes = d * spatial * 4
            out_bytes = out_ch * spatial * 4
            k = compile_ane_kernel(mil, wb, [in_bytes], [out_bytes])
            if not k:
                print(f"    ❌ {kern_name} compile failed")
                return False
            self.cls_chunks.append((k, out_ch))

        compile_time = (time.time() - t0) * 1000
        n_kernels = len(self.kernels) + len(self.cls_chunks)
        print(f"  ✓ Compiled {n_kernels} ANE kernels in {compile_time:.0f} ms")

        # Init KV cache
        self.k_cache = [np.zeros((self.max_seq, self.dim), dtype=np.float32)
                        for _ in range(self.n_layers)]
        self.v_cache = [np.zeros((self.max_seq, self.dim), dtype=np.float32)
                        for _ in range(self.n_layers)]
        self.cache_pos = 0
        self.initialized = True
        return True

    def _ane_linear(self, kern_name, x):
        """Run a single ANE conv1x1 kernel. x shape: [dim] → [out_dim]."""
        kernel = self.kernels[kern_name]
        in_dim = x.shape[0]
        # ANE expects channels-first: [1, in_dim, 1, 1]
        x_ane = x.astype(np.float32).copy()
        in_bytes = in_dim * 4
        in_buf = x_ane.ctypes.data_as(ctypes.c_void_p)
        lib.ane_bridge_write_input(kernel, 0, in_buf, in_bytes)
        lib.ane_bridge_eval(kernel)
        # Read output
        out_dim = in_dim  # same for q/k/v/o
        if 'w1' in kern_name or 'w3' in kern_name:
            out_dim = self.hidden_dim
        elif 'w2' in kern_name:
            out_dim = self.dim
        out = np.zeros(out_dim, dtype=np.float32)
        out_bytes = out_dim * 4
        out_buf = out.ctypes.data_as(ctypes.c_void_p)
        lib.ane_bridge_read_output(kernel, 0, out_buf, out_bytes)
        return out

    def _rmsnorm(self, x, w):
        """RMSNorm on CPU."""
        ss = np.mean(x * x) + 1e-5
        return x / math.sqrt(ss) * w

    def _rope(self, q, k, pos):
        """Apply RoPE at position pos. q, k shape: [n_heads * head_dim]."""
        for h in range(self.n_heads):
            for i in range(0, self.head_dim, 2):
                freq = 1.0 / (10000.0 ** (float(i) / self.head_dim))
                val = pos * freq
                cos_v, sin_v = math.cos(val), math.sin(val)
                off = h * self.head_dim + i
                q0, q1 = q[off], q[off + 1]
                q[off] = q0 * cos_v - q1 * sin_v
                q[off + 1] = q0 * sin_v + q1 * cos_v
                k0, k1 = k[off], k[off + 1]
                k[off] = k0 * cos_v - k1 * sin_v
                k[off + 1] = k0 * sin_v + k1 * cos_v
        return q, k

    def _attention(self, q, layer, pos):
        """Single-token attention with KV cache."""
        scale = 1.0 / math.sqrt(self.head_dim)
        out = np.zeros(self.dim, dtype=np.float32)
        for h in range(self.n_heads):
            hd = self.head_dim
            q_h = q[h * hd:(h + 1) * hd]
            # Attend to all cached positions
            scores = np.zeros(pos + 1, dtype=np.float32)
            for s in range(pos + 1):
                k_h = self.k_cache[layer][s, h * hd:(h + 1) * hd]
                scores[s] = np.dot(q_h, k_h) * scale
            # Softmax
            scores -= scores.max()
            scores = np.exp(scores)
            scores /= scores.sum()
            # Weighted sum of values
            for s in range(pos + 1):
                v_h = self.v_cache[layer][s, h * hd:(h + 1) * hd]
                out[h * hd:(h + 1) * hd] += scores[s] * v_h
        return out

    def _classifier(self, x):
        """Run classifier head on ANE (tiled for large vocab)."""
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        in_bytes = self.dim * 4
        x_ane = x.astype(np.float32).copy()
        in_buf = x_ane.ctypes.data_as(ctypes.c_void_p)
        offset = 0
        for kernel, out_ch in self.cls_chunks:
            lib.ane_bridge_write_input(kernel, 0, in_buf, in_bytes)
            lib.ane_bridge_eval(kernel)
            chunk_out = np.zeros(out_ch, dtype=np.float32)
            lib.ane_bridge_read_output(kernel, 0,
                                        chunk_out.ctypes.data_as(ctypes.c_void_p),
                                        out_ch * 4)
            logits[offset:offset + out_ch] = chunk_out
            offset += out_ch
        return logits

    def forward_token(self, token_id, pos):
        """Forward pass for a single token at position pos. Returns logits."""
        # Embedding lookup (CPU)
        x = self.embed_weights[token_id].copy()

        for layer in range(self.n_layers):
            # Attention block
            xn = self._rmsnorm(x, self.rms_att_w[layer])
            q = self._ane_linear(f"l{layer}_q", xn)
            k = self._ane_linear(f"l{layer}_k", xn)
            v = self._ane_linear(f"l{layer}_v", xn)
            q, k = self._rope(q, k, pos)

            # Cache K, V
            self.k_cache[layer][pos] = k
            self.v_cache[layer][pos] = v

            # Attention
            attn_out = self._attention(q, layer, pos)
            o = self._ane_linear(f"l{layer}_o", attn_out)
            x = x + o

            # FFN block
            xn = self._rmsnorm(x, self.rms_ffn_w[layer])
            h1 = self._ane_linear(f"l{layer}_w1", xn)
            h3 = self._ane_linear(f"l{layer}_w3", xn)

            # SiLU gate (CPU)
            silu = h1 / (1.0 + np.exp(-h1)) * h3
            ffn_out = self._ane_linear(f"l{layer}_w2", silu)
            x = x + ffn_out

        # Final norm + classifier
        x = self._rmsnorm(x, self.rms_final_w)
        logits = self._classifier(x)
        return logits

    def generate_draft(self, prompt_ids, k_tokens, temperature=0.0):
        """
        Generate K draft tokens autoregressively on ANE.
        Returns list of (token_id, logits) tuples.
        """
        self.cache_pos = 0
        drafts = []

        # Process prompt (fill KV cache)
        for i, tid in enumerate(prompt_ids):
            logits = self.forward_token(tid, i)
            self.cache_pos = i + 1

        # Generate K draft tokens
        if len(prompt_ids) > 0:
            next_token = int(np.argmax(logits)) if temperature == 0 else self._sample(logits, temperature)
        else:
            next_token = 0

        for k in range(k_tokens):
            logits = self.forward_token(next_token, self.cache_pos)
            self.cache_pos += 1

            if temperature == 0:
                sampled = int(np.argmax(logits))
            else:
                sampled = self._sample(logits, temperature)

            drafts.append((next_token, logits))
            next_token = sampled

        # Add the last predicted token
        drafts.append((next_token, None))
        return drafts

    def _sample(self, logits, temperature):
        """Sample from logits with temperature."""
        logits = logits / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def reset_cache(self):
        """Reset KV cache."""
        for layer in range(self.n_layers):
            self.k_cache[layer][:] = 0
            self.v_cache[layer][:] = 0
        self.cache_pos = 0

    def cleanup(self):
        """Free ANE kernels."""
        for k in self.kernels.values():
            lib.ane_bridge_free(k)
        for k, _ in self.cls_chunks:
            lib.ane_bridge_free(k)


# ── Init ─────────────────────────────────────────────────────────────────
def init_ane():
    """Initialize ANE bridge."""
    result = lib.ane_bridge_init()
    if result != 0:
        print("Failed to initialize ANE bridge!")
        return False
    return True
