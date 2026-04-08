#!/usr/bin/env python3
"""Export ANE-trained checkpoint weights to HuggingFace safetensors format.

Reads the binary checkpoint saved by train.m and writes safetensors files
compatible with the inference pipeline (real_draft.py).

The checkpoint format (from train.m save_checkpoint):
  - Header: step, total_steps, lr, loss, cum_train, cum_wall, cum_steps, adam_t
  - Per layer (28 for Qwen3-0.6B):
    Wq[Q_DIM * DIM], Wk[KV_DIM * DIM], Wv[KV_DIM * DIM], Wo[DIM * Q_DIM],
    W1[HIDDEN * DIM], W2[DIM * HIDDEN], W3[HIDDEN * DIM],
    rms_att[DIM], rms_ffn[DIM],
    + Adam states (m,v for each)
  - rms_final[DIM]
  - embed[VOCAB * DIM]

Usage:
  python export_weights.py --checkpoint ane_qwen3_06b_dyn_ckpt.bin \
                           --output distilled_qwen3_06b/
"""

import argparse
import struct
import sys
import os
import numpy as np

# Model config (must match qwen3_06b.h)
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

# Weight sizes
WQ_SZ = Q_DIM * DIM
WK_SZ = KV_DIM * DIM
WV_SZ = KV_DIM * DIM
WO_SZ = DIM * Q_DIM
W1_SZ = HIDDEN * DIM
W2_SZ = DIM * HIDDEN
W3_SZ = HIDDEN * DIM


def read_checkpoint(path):
    """Read ANE training checkpoint, return weights dict."""
    print(f"Reading checkpoint: {path}")

    with open(path, 'rb') as f:
        # CkptHdr: 96 bytes
        hdr = f.read(96)
        magic = struct.unpack_from('<i', hdr, 0)[0]
        version = struct.unpack_from('<i', hdr, 4)[0]
        step = struct.unpack_from('<i', hdr, 8)[0]
        total_steps = struct.unpack_from('<i', hdr, 12)[0]
        lr = struct.unpack_from('<f', hdr, 40)[0]
        loss = struct.unpack_from('<f', hdr, 44)[0]

        if magic != 0x424C5A54 or version != 4:
            print(f"  ERROR: Bad checkpoint (magic={hex(magic)}, version={version})")
            sys.exit(1)

        print(f"  Step {step}/{total_steps}, loss={loss:.4f}, lr={lr:.2e}")

        weights = {}

        for L in range(NLAYERS):
            prefix = f"model.layers.{L}"

            # Read weights (row-major, same as train.m layout)
            Wq = np.frombuffer(f.read(WQ_SZ * 4), dtype=np.float32).reshape(Q_DIM, DIM).copy()
            Wk = np.frombuffer(f.read(WK_SZ * 4), dtype=np.float32).reshape(KV_DIM, DIM).copy()
            Wv = np.frombuffer(f.read(WV_SZ * 4), dtype=np.float32).reshape(KV_DIM, DIM).copy()
            Wo = np.frombuffer(f.read(WO_SZ * 4), dtype=np.float32).reshape(DIM, Q_DIM).copy()
            W1 = np.frombuffer(f.read(W1_SZ * 4), dtype=np.float32).reshape(HIDDEN, DIM).copy()
            W2 = np.frombuffer(f.read(W2_SZ * 4), dtype=np.float32).reshape(DIM, HIDDEN).copy()
            W3 = np.frombuffer(f.read(W3_SZ * 4), dtype=np.float32).reshape(HIDDEN, DIM).copy()
            rms_att = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
            rms_ffn = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()

            # Skip Adam states (m, v for each weight)
            adam_bytes = (WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + DIM + DIM) * 2 * 4
            f.read(adam_bytes)

            # Map to HuggingFace names
            weights[f"{prefix}.self_attn.q_proj.weight"] = Wq
            weights[f"{prefix}.self_attn.k_proj.weight"] = Wk
            weights[f"{prefix}.self_attn.v_proj.weight"] = Wv
            weights[f"{prefix}.self_attn.o_proj.weight"] = Wo
            weights[f"{prefix}.mlp.gate_proj.weight"] = W1
            weights[f"{prefix}.mlp.down_proj.weight"] = W2
            weights[f"{prefix}.mlp.up_proj.weight"] = W3
            weights[f"{prefix}.input_layernorm.weight"] = rms_att
            weights[f"{prefix}.post_attention_layernorm.weight"] = rms_ffn

            # Qwen3-0.6B has q_norm and k_norm but training doesn't update them
            # They stay at 1.0 (identity) — we'll add them as ones
            weights[f"{prefix}.self_attn.q_norm.weight"] = np.ones(HD, dtype=np.float32)
            weights[f"{prefix}.self_attn.k_norm.weight"] = np.ones(HD, dtype=np.float32)

        # Final RMSNorm
        rms_final = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
        f.read(DIM * 2 * 4)  # skip Adam state

        weights["model.norm.weight"] = rms_final

        # Embedding
        embed = np.frombuffer(f.read(VOCAB * DIM * 4), dtype=np.float32).reshape(VOCAB, DIM).copy()
        weights["model.embed_tokens.weight"] = embed
        weights["lm_head.weight"] = embed.copy()  # tied weights

        print(f"  Loaded {len(weights)} weight tensors")
        return weights, step, loss


def save_safetensors(weights, output_dir):
    """Save weights as safetensors (numpy-based, no torch dependency)."""
    try:
        from safetensors.numpy import save_file
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "model.safetensors")
        save_file(weights, output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Saved to {output_path} ({size_mb:.1f} MB)")
    except ImportError:
        # Fallback: save as numpy .npz
        output_path = os.path.join(output_dir, "weights.npz")
        os.makedirs(output_dir, exist_ok=True)
        np.savez(output_path, **weights)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Saved to {output_path} ({size_mb:.1f} MB)")
        print("  Note: Install safetensors for HF-compatible format: pip install safetensors")


def main():
    parser = argparse.ArgumentParser(description="Export ANE checkpoint to HF format")
    parser.add_argument("--checkpoint", required=True, help="Path to ANE training checkpoint")
    parser.add_argument("--output", default="distilled_qwen3_06b", help="Output directory")
    args = parser.parse_args()

    weights, step, loss = read_checkpoint(args.checkpoint)
    save_safetensors(weights, args.output)

    print(f"\n  Export complete. To use with inference pipeline:")
    print(f"  python speculative/real_draft.py --weights {args.output}/")


if __name__ == "__main__":
    main()
