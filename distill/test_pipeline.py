#!/usr/bin/env python3
"""Quick end-to-end test of the distillation data pipeline.

Creates a small synthetic dataset, generates teacher logits, and verifies
the binary format is correct for train.m consumption.

No model download needed — uses synthetic data.
"""

import struct
import sys
import os
import numpy as np

sys.path.insert(0, "/Users/midas/.mlx-env/lib/python3.11/site-packages")

MAGIC = 0x544C4F47
VERSION = 1


def test_data_format():
    """Create and verify uint32 token data."""
    print("=== Test 1: Token Data Format ===")

    # Simulate Qwen3 token IDs (some > 65535)
    np.random.seed(42)
    n_tokens = 10000
    tokens = np.random.randint(0, 151936, size=n_tokens, dtype=np.uint32)

    # Ensure some tokens > 65535 to test uint32 handling
    assert (tokens > 65535).any(), "Need tokens > 65535 for uint32 test"
    print(f"  Created {n_tokens} synthetic tokens, max ID = {tokens.max()}")

    # Save as uint32
    path = "/tmp/test_qwen3_tokens.bin"
    tokens.tofile(path)
    size = os.path.getsize(path)
    assert size == n_tokens * 4, f"Expected {n_tokens*4} bytes, got {size}"
    print(f"  Saved to {path} ({size} bytes)")

    # Read back
    verify = np.fromfile(path, dtype=np.uint32)
    assert np.array_equal(verify, tokens)
    print(f"  Verified: read back matches")

    os.unlink(path)
    print("  PASS\n")


def test_teacher_format():
    """Create and verify teacher logits file format."""
    print("=== Test 2: Teacher Logits Format ===")

    top_k = 32
    seq_len = 256
    n_sequences = 5
    vocab_size = 151936

    path = "/tmp/test_teacher_logits.bin"

    # Write header + data
    with open(path, 'wb') as f:
        f.write(struct.pack('<IIIIII', MAGIC, VERSION, top_k, seq_len, n_sequences, vocab_size))

        for s in range(n_sequences):
            for t in range(seq_len):
                # Random top-K token IDs
                ids = np.random.randint(0, vocab_size, size=top_k, dtype=np.int32)
                # Decreasing logits (most likely first)
                logits = np.linspace(5.0, -2.0, top_k).astype(np.float32)
                f.write(ids.tobytes())
                f.write(logits.tobytes())

    size = os.path.getsize(path)
    expected_data = n_sequences * seq_len * top_k * 8  # 4 bytes ID + 4 bytes logit
    expected_total = 24 + expected_data
    assert size == expected_total, f"Expected {expected_total} bytes, got {size}"
    print(f"  Created {path} ({size} bytes)")

    # Read back and verify header
    with open(path, 'rb') as f:
        hdr = struct.unpack('<IIIIII', f.read(24))
        assert hdr[0] == MAGIC, f"Bad magic: {hex(hdr[0])}"
        assert hdr[1] == VERSION
        assert hdr[2] == top_k
        assert hdr[3] == seq_len
        assert hdr[4] == n_sequences
        assert hdr[5] == vocab_size

        # Read first position
        ids = np.frombuffer(f.read(top_k * 4), dtype=np.int32)
        logits = np.frombuffer(f.read(top_k * 4), dtype=np.float32)
        print(f"  Header: magic={hex(hdr[0])}, top_k={hdr[2]}, seq={hdr[3]}, n_seq={hdr[4]}")
        print(f"  First pos: ids[0:5]={ids[:5]}, logits[0:5]={logits[:5]}")

        # Verify softmax of top-K
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()
        print(f"  Top-1 prob: {probs[0]:.4f}, top-K mass: {probs.sum():.4f}")

    os.unlink(path)
    print("  PASS\n")


def test_combined_loss_gradient():
    """Verify gradient math: α*CE + (1-α)*KL gradient shape."""
    print("=== Test 3: Combined Loss Gradient Shape ===")

    V = 100  # compact vocab
    S = 4    # sequence length
    alpha = 0.5

    # Random student logits
    student_logits = np.random.randn(V, S).astype(np.float32)

    # Softmax
    exp_l = np.exp(student_logits - student_logits.max(axis=0, keepdims=True))
    student_probs = exp_l / exp_l.sum(axis=0, keepdims=True)

    # CE gradient (with random hard targets)
    targets = np.random.randint(0, V, size=S)
    ce_grad = student_probs.copy()
    for t in range(S):
        ce_grad[targets[t], t] -= 1.0
    ce_grad /= S

    # KL gradient (with random teacher soft targets)
    teacher_probs = np.random.dirichlet(np.ones(V), size=S).T.astype(np.float32)
    kl_grad = (student_probs - teacher_probs) / S

    # Combined
    combined_grad = alpha * ce_grad + (1 - alpha) * kl_grad

    # Check: gradients sum to ~0 per position
    for t in range(S):
        s = combined_grad[:, t].sum()
        assert abs(s) < 1e-5, f"Gradient sum at pos {t}: {s}"

    print(f"  CE grad shape: {ce_grad.shape}")
    print(f"  KL grad shape: {kl_grad.shape}")
    print(f"  Combined grad shape: {combined_grad.shape}")
    print(f"  Gradient sums: {[f'{combined_grad[:, t].sum():.8f}' for t in range(S)]}")
    print(f"  All sums ~0: PASS")
    print("  PASS\n")


if __name__ == "__main__":
    test_data_format()
    test_teacher_format()
    test_combined_loss_gradient()
    print("=== All pipeline tests passed ===")
