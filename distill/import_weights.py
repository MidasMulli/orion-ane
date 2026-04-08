#!/usr/bin/env python3
"""Import pretrained HuggingFace Qwen3-0.6B weights into ANE training checkpoint.

Downloads the model from HuggingFace, extracts float32 weights, and writes
a valid checkpoint file that train.m can load with --resume. Adam optimizer
states are zeroed (fresh optimizer for fine-tuning).

Usage:
  python import_weights.py --model Qwen/Qwen3-0.6B \
                           --output ../training/training_dynamic/ane_qwen3_06b_dyn_ckpt.bin

  # Then resume training with distillation:
  cd ../training/training_dynamic
  ./train --resume --token32 --data ../../distill/qwen3_data.bin \
          --distill ../../distill/teacher_logits.bin
"""

import argparse
import struct
import sys
import os
import time
import numpy as np

sys.path.insert(0, "/Users/midas/.mlx-env/lib/python3.11/site-packages")

# Qwen3-0.6B config (must match qwen3_06b.h exactly)
DIM = 1024
HIDDEN = 3072
HEADS = 16
KV_HEADS = 8
HD = 128
Q_DIM = HEADS * HD       # 2048
KV_DIM = KV_HEADS * HD   # 1024
SEQ = 256
NLAYERS = 28
VOCAB = 151936

# Derived weight sizes
WQ_SZ = Q_DIM * DIM      # 2097152
WK_SZ = KV_DIM * DIM     # 1048576
WV_SZ = KV_DIM * DIM     # 1048576
WO_SZ = DIM * Q_DIM      # 2097152
W1_SZ = HIDDEN * DIM     # 3145728
W2_SZ = DIM * HIDDEN     # 3145728
W3_SZ = HIDDEN * DIM     # 3145728

# Checkpoint magic and version
CKPT_MAGIC = 0x424C5A54
CKPT_VERSION = 4


def write_zeros(f, count):
    """Write count zero float32s."""
    zeros = np.zeros(count, dtype=np.float32)
    f.write(zeros.tobytes())


def write_f32(f, arr):
    """Write float32 array, ensuring contiguous row-major."""
    data = np.ascontiguousarray(arr, dtype=np.float32)
    f.write(data.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Import HF Qwen3-0.6B to ANE checkpoint")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model ID")
    parser.add_argument("--output", default="../training/training_dynamic/ane_qwen3_06b_dyn_ckpt.bin",
                        help="Output checkpoint path")
    parser.add_argument("--total_steps", type=int, default=10000,
                        help="Total training steps to set in checkpoint header")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate to set in checkpoint header")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  IMPORT PRETRAINED WEIGHTS → ANE CHECKPOINT")
    print(f"  Model:  {args.model}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    # Load model
    print("[1/3] Loading HuggingFace model...")
    t0 = time.time()

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    sd = model.state_dict()
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s ({len(sd)} tensors)")

    # Validate dimensions
    embed_w = sd["model.embed_tokens.weight"].numpy()  # [VOCAB, DIM]
    assert embed_w.shape == (VOCAB, DIM), f"Embed shape {embed_w.shape} != ({VOCAB}, {DIM})"
    print(f"  Embed: {embed_w.shape}, layers: {NLAYERS}")
    print(f"  Q_DIM={Q_DIM}, KV_DIM={KV_DIM}, HIDDEN={HIDDEN}")

    # Verify a layer's shapes
    q0 = sd["model.layers.0.self_attn.q_proj.weight"].numpy()
    assert q0.shape == (Q_DIM, DIM), f"Wq shape {q0.shape} != ({Q_DIM}, {DIM})"

    # Write checkpoint
    print(f"\n[2/3] Writing checkpoint...")
    t0 = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, 'wb') as f:
        # === Header (CkptHdr, 96 bytes) ===
        # Layout: 10 ints, 2 floats, 3 doubles, 6 ints = 96 bytes
        f.write(struct.pack('<i', CKPT_MAGIC))       # 0: magic
        f.write(struct.pack('<i', CKPT_VERSION))      # 4: version
        f.write(struct.pack('<i', 0))                  # 8: step
        f.write(struct.pack('<i', args.total_steps))   # 12: total_steps
        f.write(struct.pack('<i', NLAYERS))            # 16: n_layers
        f.write(struct.pack('<i', VOCAB))              # 20: vocab_size
        f.write(struct.pack('<i', DIM))                # 24: dim
        f.write(struct.pack('<i', HIDDEN))             # 28: hidden_dim
        f.write(struct.pack('<i', HEADS))              # 32: n_heads
        f.write(struct.pack('<i', SEQ))                # 36: seq_len
        f.write(struct.pack('<f', args.lr))            # 40: lr
        f.write(struct.pack('<f', 0.0))                # 44: loss
        f.write(struct.pack('<d', 0.0))                # 48: cum_compile
        f.write(struct.pack('<d', 0.0))                # 56: cum_train
        f.write(struct.pack('<d', 0.0))                # 64: cum_wall
        f.write(struct.pack('<i', 0))                  # 72: cum_steps
        f.write(struct.pack('<i', 0))                  # 76: cum_batches
        f.write(struct.pack('<i', 0))                  # 80: adam_t
        f.write(struct.pack('<i', KV_HEADS))           # 84: kv_heads
        f.write(struct.pack('<i', HD))                 # 88: head_dim
        f.write(struct.pack('<i', Q_DIM))              # 92: q_dim

        assert f.tell() == 96, f"Header size mismatch: {f.tell()} != 96"

        # === Per-layer weights + zeroed Adam states ===
        for L in range(NLAYERS):
            prefix = f"model.layers.{L}"

            # Weights (row-major float32, shapes match exactly)
            Wq = sd[f"{prefix}.self_attn.q_proj.weight"].numpy()    # [Q_DIM, DIM]
            Wk = sd[f"{prefix}.self_attn.k_proj.weight"].numpy()    # [KV_DIM, DIM]
            Wv = sd[f"{prefix}.self_attn.v_proj.weight"].numpy()    # [KV_DIM, DIM]
            Wo = sd[f"{prefix}.self_attn.o_proj.weight"].numpy()    # [DIM, Q_DIM]
            W1 = sd[f"{prefix}.mlp.gate_proj.weight"].numpy()       # [HIDDEN, DIM]
            W2 = sd[f"{prefix}.mlp.down_proj.weight"].numpy()       # [DIM, HIDDEN]
            W3 = sd[f"{prefix}.mlp.up_proj.weight"].numpy()         # [HIDDEN, DIM]
            rms_att = sd[f"{prefix}.input_layernorm.weight"].numpy()             # [DIM]
            rms_ffn = sd[f"{prefix}.post_attention_layernorm.weight"].numpy()    # [DIM]

            # Verify shapes
            assert Wq.shape == (Q_DIM, DIM), f"L{L} Wq {Wq.shape}"
            assert Wk.shape == (KV_DIM, DIM), f"L{L} Wk {Wk.shape}"
            assert Wv.shape == (KV_DIM, DIM), f"L{L} Wv {Wv.shape}"
            assert Wo.shape == (DIM, Q_DIM), f"L{L} Wo {Wo.shape}"
            assert W1.shape == (HIDDEN, DIM), f"L{L} W1 {W1.shape}"
            assert W2.shape == (DIM, HIDDEN), f"L{L} W2 {W2.shape}"
            assert W3.shape == (HIDDEN, DIM), f"L{L} W3 {W3.shape}"

            # Write weights (exact order from save_checkpoint)
            write_f32(f, Wq)       # WQ_SZ
            write_f32(f, Wk)       # WK_SZ
            write_f32(f, Wv)       # WV_SZ
            write_f32(f, Wo)       # WO_SZ
            write_f32(f, W1)       # W1_SZ
            write_f32(f, W2)       # W2_SZ
            write_f32(f, W3)       # W3_SZ
            write_f32(f, rms_att)  # DIM
            write_f32(f, rms_ffn)  # DIM

            # Adam states: all zeros (fresh optimizer)
            write_zeros(f, WQ_SZ)   # Wq.m
            write_zeros(f, WQ_SZ)   # Wq.v
            write_zeros(f, WK_SZ)   # Wk.m
            write_zeros(f, WK_SZ)   # Wk.v
            write_zeros(f, WV_SZ)   # Wv.m
            write_zeros(f, WV_SZ)   # Wv.v
            write_zeros(f, WO_SZ)   # Wo.m
            write_zeros(f, WO_SZ)   # Wo.v
            write_zeros(f, W1_SZ)   # W1.m
            write_zeros(f, W1_SZ)   # W1.v
            write_zeros(f, W2_SZ)   # W2.m
            write_zeros(f, W2_SZ)   # W2.v
            write_zeros(f, W3_SZ)   # W3.m
            write_zeros(f, W3_SZ)   # W3.v
            write_zeros(f, DIM)     # rms_att.m
            write_zeros(f, DIM)     # rms_att.v
            write_zeros(f, DIM)     # rms_ffn.m
            write_zeros(f, DIM)     # rms_ffn.v

            if L % 7 == 0 or L == NLAYERS - 1:
                print(f"  Layer {L}/{NLAYERS-1}: Wq{list(Wq.shape)} rms_att[{rms_att.mean():.4f}]")

        # === Global weights ===
        rms_final = sd["model.norm.weight"].numpy()  # [DIM]
        embed = sd["model.embed_tokens.weight"].numpy()  # [VOCAB, DIM]

        # Flatten embed to 1D (train.m reads as flat array, VOCAB*DIM floats)
        embed_flat = embed.reshape(-1)  # [VOCAB*DIM]

        # rms_final + Adam state
        write_f32(f, rms_final)         # DIM
        write_zeros(f, DIM)             # arms_final.m
        write_zeros(f, DIM)             # arms_final.v

        # embed + Adam state
        write_f32(f, embed_flat)        # VOCAB*DIM
        write_zeros(f, VOCAB * DIM)     # aembed.m
        write_zeros(f, VOCAB * DIM)     # aembed.v

        total_bytes = f.tell()

    elapsed = time.time() - t0
    print(f"\n  Written {total_bytes:,} bytes ({total_bytes/1024/1024:.1f} MB) in {elapsed:.1f}s")

    # === Verify ===
    print(f"\n[3/3] Verifying checkpoint...")

    with open(args.output, 'rb') as f:
        # Read header
        hdr = f.read(96)
        magic = struct.unpack_from('<i', hdr, 0)[0]
        version = struct.unpack_from('<i', hdr, 4)[0]
        step = struct.unpack_from('<i', hdr, 8)[0]
        n_layers = struct.unpack_from('<i', hdr, 16)[0]
        vocab = struct.unpack_from('<i', hdr, 20)[0]
        dim = struct.unpack_from('<i', hdr, 24)[0]
        hidden = struct.unpack_from('<i', hdr, 28)[0]
        kv_heads = struct.unpack_from('<i', hdr, 84)[0]
        hd = struct.unpack_from('<i', hdr, 88)[0]
        q_dim = struct.unpack_from('<i', hdr, 92)[0]

        assert magic == CKPT_MAGIC, f"Bad magic: {hex(magic)}"
        assert version == CKPT_VERSION, f"Bad version: {version}"
        print(f"  Header: magic={hex(magic)}, v{version}, step={step}")
        print(f"  Config: layers={n_layers}, vocab={vocab}, dim={dim}, hidden={hidden}")
        print(f"  GQA: kv_heads={kv_heads}, head_dim={hd}, q_dim={q_dim}")

        # Read first layer Wq and verify against HF
        Wq_read = np.frombuffer(f.read(WQ_SZ * 4), dtype=np.float32)
        Wq_expected = sd["model.layers.0.self_attn.q_proj.weight"].numpy().flatten()
        max_diff = np.abs(Wq_read - Wq_expected).max()
        print(f"  Layer 0 Wq max diff: {max_diff:.2e} {'PASS' if max_diff < 1e-6 else 'FAIL'}")

        # Skip to embed and verify
        # Seek to: 96 + NLAYERS * (layer_weights + layer_adam) + DIM*3 (rms_final + adam)
        layer_w_bytes = (WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM) * 4
        layer_a_bytes = (WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM) * 2 * 4
        embed_offset = 96 + NLAYERS * (layer_w_bytes + layer_a_bytes) + DIM * 3 * 4
        f.seek(embed_offset)
        embed_read = np.frombuffer(f.read(min(1024*4, VOCAB*DIM*4)), dtype=np.float32)
        embed_expected = embed_flat[:1024]
        max_diff_e = np.abs(embed_read - embed_expected).max()
        print(f"  Embed first 1024 max diff: {max_diff_e:.2e} {'PASS' if max_diff_e < 1e-6 else 'FAIL'}")

    print(f"\n  Checkpoint ready: {args.output}")
    print(f"  Use with: ./train --resume --token32 --data <data> --distill <teacher>")
    print()


if __name__ == "__main__":
    main()
