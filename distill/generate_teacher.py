#!/usr/bin/env python3
"""Generate teacher logits for knowledge distillation.

Runs Qwen3-4B (or any MLX model) on the same tokenized training data used by
the ANE student, saving top-K logits per position for each sequence.

Output format (binary):
  Header: [magic(4B), version(4B), top_k(4B), seq_len(4B), n_sequences(4B), vocab_size(4B)]
  Per sequence (n_sequences total):
    Per position (seq_len total, shifted by 1 for next-token prediction):
      token_ids: int32[top_k]   — full vocab IDs of top-K tokens
      logits:    float32[top_k] — raw logits (pre-softmax) for those tokens

Usage:
  python generate_teacher.py --data ../training/tinystories_data00.bin \
                             --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
                             --output teacher_logits.bin \
                             --n_sequences 1000 --top_k 32

The student training pipeline loads this file and reconstructs soft target
distributions from the top-K logits at training time.
"""

import argparse
import struct
import sys
import time
import numpy as np

# MLX imports (from ~/.mlx-env/)
sys.path.insert(0, "/Users/midas/.mlx-env/lib/python3.11/site-packages")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load


MAGIC = 0x544C4F47  # "GOLT" (GOLden Teacher)
VERSION = 1


def load_training_data(path, vocab_size=151936):
    """Load pretokenized training data (uint16 binary, same as train.m)."""
    data = np.fromfile(path, dtype=np.uint16)
    print(f"  Loaded {len(data):,} tokens from {path}")
    # Validate token range
    max_tok = int(data.max())
    if max_tok >= vocab_size:
        print(f"  WARNING: max token {max_tok} >= vocab_size {vocab_size}")
    return data


def extract_sequences(data, seq_len, n_sequences, stride=None):
    """Extract training sequences from token stream.

    Each sequence is seq_len+1 tokens: input[0:seq_len] → target[1:seq_len+1].
    Matches the windowing used by train.m.
    """
    if stride is None:
        stride = seq_len  # non-overlapping by default

    sequences = []
    pos = 0
    while len(sequences) < n_sequences and pos + seq_len + 1 <= len(data):
        seq = data[pos:pos + seq_len + 1].astype(np.int32)
        sequences.append(seq)
        pos += stride

    if len(sequences) < n_sequences:
        print(f"  WARNING: Only got {len(sequences)} sequences (requested {n_sequences})")

    return sequences


def generate_teacher_logits(model, tokenizer, sequences, seq_len, top_k, batch_size=1):
    """Run teacher model on sequences, extract top-K logits per position.

    Returns list of (token_ids[seq_len, top_k], logits[seq_len, top_k]) per sequence.
    """
    results = []
    total = len(sequences)
    t_start = time.time()

    for i, seq in enumerate(sequences):
        input_ids = seq[:seq_len]  # [seq_len]

        # Forward pass through teacher
        x = mx.array(input_ids.reshape(1, -1))  # [1, seq_len]
        logits = model(x)  # [1, seq_len, vocab_size]
        mx.eval(logits)

        logits_np = np.array(logits[0])  # [seq_len, vocab_size]

        # Extract top-K per position (shifted: position t predicts token t+1)
        seq_top_ids = np.zeros((seq_len, top_k), dtype=np.int32)
        seq_top_logits = np.zeros((seq_len, top_k), dtype=np.float32)

        for t in range(seq_len):
            top_idx = np.argpartition(logits_np[t], -top_k)[-top_k:]
            # Sort by logit value (descending)
            sorted_order = np.argsort(logits_np[t, top_idx])[::-1]
            top_idx = top_idx[sorted_order]

            seq_top_ids[t] = top_idx
            seq_top_logits[t] = logits_np[t, top_idx]

        results.append((seq_top_ids, seq_top_logits))

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (total - i - 1) / rate if rate > 0 else 0

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {rate:.1f} seq/s, ETA {eta:.0f}s")

    return results


def save_teacher_data(path, results, top_k, seq_len, vocab_size):
    """Save teacher logits in binary format."""
    n_sequences = len(results)

    with open(path, 'wb') as f:
        # Header
        f.write(struct.pack('<IIIIII', MAGIC, VERSION, top_k, seq_len, n_sequences, vocab_size))

        # Per sequence, per position: top_k int32 IDs + top_k float32 logits
        for seq_ids, seq_logits in results:
            for t in range(seq_len):
                f.write(seq_ids[t].tobytes())      # int32[top_k]
                f.write(seq_logits[t].tobytes())    # float32[top_k]

    file_size = 24 + n_sequences * seq_len * top_k * 8  # 4 bytes ID + 4 bytes logit
    print(f"  Saved {n_sequences} sequences to {path} ({file_size / 1024 / 1024:.1f} MB)")


def verify_teacher_data(path):
    """Read back and verify the saved file."""
    with open(path, 'rb') as f:
        magic, version, top_k, seq_len, n_seq, vocab = struct.unpack('<IIIIII', f.read(24))
        assert magic == MAGIC, f"Bad magic: {hex(magic)}"
        assert version == VERSION

        # Read first sequence, first position
        ids = np.frombuffer(f.read(top_k * 4), dtype=np.int32)
        logits = np.frombuffer(f.read(top_k * 4), dtype=np.float32)

        # Compute softmax of top-K
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()

        print(f"\n  Verification — seq 0, pos 0:")
        print(f"  Top-5 tokens: {ids[:5]}")
        print(f"  Top-5 logits: {logits[:5]}")
        print(f"  Top-5 probs:  {probs[:5]}")
        print(f"  Top-1 prob:   {probs[0]:.4f}")
        print(f"  Top-K mass:   {probs.sum():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate teacher logits for ANE distillation")
    parser.add_argument("--data", required=True, help="Path to pretokenized training data (uint16 binary)")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                        help="MLX model to use as teacher")
    parser.add_argument("--output", default="teacher_logits.bin", help="Output file path")
    parser.add_argument("--n_sequences", type=int, default=1000, help="Number of sequences to process")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length (must match student)")
    parser.add_argument("--top_k", type=int, default=32, help="Number of top logits to save per position")
    parser.add_argument("--vocab_size", type=int, default=151936, help="Full vocabulary size")
    parser.add_argument("--stride", type=int, default=None, help="Stride between sequences (default: seq_len)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  TEACHER DATA GENERATION")
    print(f"  Model:      {args.model}")
    print(f"  Data:       {args.data}")
    print(f"  Sequences:  {args.n_sequences}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Top-K:      {args.top_k}")
    print(f"{'='*60}\n")

    # Load training data
    print("[1/4] Loading training data...")
    data = load_training_data(args.data, args.vocab_size)

    # Extract sequences
    print("[2/4] Extracting sequences...")
    sequences = extract_sequences(data, args.seq_len, args.n_sequences, args.stride)
    print(f"  Got {len(sequences)} sequences of length {args.seq_len + 1}")

    # Load teacher model
    print(f"[3/4] Loading teacher model: {args.model}")
    t0 = time.time()
    model, tokenizer = mlx_load(args.model)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Generate logits
    print(f"[4/4] Generating teacher logits...")
    t0 = time.time()
    results = generate_teacher_logits(model, tokenizer, sequences, args.seq_len, args.top_k)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(results) * args.seq_len / elapsed:.0f} tokens/s)")

    # Save
    print("\nSaving...")
    save_teacher_data(args.output, results, args.top_k, args.seq_len, args.vocab_size)

    # Verify
    verify_teacher_data(args.output)

    # Storage estimate
    storage_per_seq = args.seq_len * args.top_k * 8  # bytes
    print(f"\n  Storage: {storage_per_seq} bytes/seq, "
          f"{storage_per_seq * len(results) / 1024 / 1024:.1f} MB total")
    print(f"  Estimated top-K probability mass: ~95-99% (top-{args.top_k})")
    print()


if __name__ == "__main__":
    main()
