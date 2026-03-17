#!/usr/bin/env python3
"""Prepare Qwen3-tokenized training data for ANE distillation.

Takes text input (file or HuggingFace dataset) and tokenizes with the Qwen3
tokenizer, producing the flat uint16 binary format expected by train.m.

Usage:
  # From text file:
  python prepare_data.py --input stories.txt --output qwen3_tinystories.bin

  # From HuggingFace (TinyStories):
  python prepare_data.py --hf roneneldan/TinyStories --output qwen3_tinystories.bin \
                         --max_tokens 5000000

  # Verify tokenized output:
  python prepare_data.py --verify qwen3_tinystories.bin
"""

import argparse
import struct
import sys
import os
import time
import numpy as np

sys.path.insert(0, "/Users/midas/.mlx-env/lib/python3.11/site-packages")


def tokenize_text_file(input_path, tokenizer, max_tokens=None):
    """Tokenize a text file, return flat token array."""
    print(f"  Reading {input_path}...")
    with open(input_path, 'r') as f:
        text = f.read()
    print(f"  {len(text):,} characters")

    print(f"  Tokenizing...")
    t0 = time.time()
    tokens = tokenizer.encode(text)
    elapsed = time.time() - t0
    print(f"  {len(tokens):,} tokens in {elapsed:.1f}s ({len(tokens)/elapsed:.0f} tok/s)")

    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"  Truncated to {max_tokens:,} tokens")

    return np.array(tokens, dtype=np.uint16)


def tokenize_hf_dataset(dataset_name, tokenizer, split="train", max_tokens=5_000_000,
                        text_field="text"):
    """Tokenize a HuggingFace dataset, return flat token array."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: pip install datasets (required for HF datasets)")
        sys.exit(1)

    print(f"  Loading {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    all_tokens = []
    n_docs = 0
    t0 = time.time()

    for example in ds:
        text = example[text_field]
        if not text.strip():
            continue
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        n_docs += 1

        if len(all_tokens) >= max_tokens:
            all_tokens = all_tokens[:max_tokens]
            break

        if n_docs % 1000 == 0:
            elapsed = time.time() - t0
            rate = len(all_tokens) / elapsed
            print(f"    {n_docs} docs, {len(all_tokens):,} tokens ({rate:.0f} tok/s)")

    elapsed = time.time() - t0
    print(f"  Done: {n_docs} docs, {len(all_tokens):,} tokens in {elapsed:.1f}s")

    return np.array(all_tokens, dtype=np.uint16)


def save_binary(tokens, output_path):
    """Save tokens as flat uint16 binary (same format as train.m expects)."""
    tokens.tofile(output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved {len(tokens):,} tokens to {output_path} ({size_mb:.1f} MB)")

    # Verify
    verify = np.fromfile(output_path, dtype=np.uint16)
    assert len(verify) == len(tokens)
    assert np.array_equal(verify, tokens)
    print(f"  Verified: first 10 tokens = {verify[:10].tolist()}")


def verify_data(path, tokenizer=None):
    """Verify and inspect tokenized data file."""
    tokens = np.fromfile(path, dtype=np.uint16)
    size_mb = os.path.getsize(path) / 1024 / 1024
    unique = len(np.unique(tokens))
    print(f"  File: {path}")
    print(f"  Tokens: {len(tokens):,} ({size_mb:.1f} MB)")
    print(f"  Unique tokens: {unique:,}")
    print(f"  Max token ID: {tokens.max()}")
    print(f"  Min token ID: {tokens.min()}")
    print(f"  First 20 IDs: {tokens[:20].tolist()}")

    if tokenizer:
        text = tokenizer.decode(tokens[:50].tolist())
        print(f"  First 50 tokens decoded: {text[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Qwen3-tokenized training data")
    parser.add_argument("--input", help="Path to text file to tokenize")
    parser.add_argument("--hf", help="HuggingFace dataset name (e.g., roneneldan/TinyStories)")
    parser.add_argument("--output", default="qwen3_data.bin", help="Output binary file path")
    parser.add_argument("--max_tokens", type=int, default=5_000_000, help="Max tokens to generate")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model for tokenizer")
    parser.add_argument("--verify", help="Verify a tokenized data file")
    parser.add_argument("--text_field", default="text", help="Text field name in HF dataset")
    args = parser.parse_args()

    if args.verify:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        verify_data(args.verify, tokenizer)
        return

    if not args.input and not args.hf:
        parser.error("Provide --input (text file) or --hf (HuggingFace dataset)")

    # Load tokenizer
    print(f"\n[1/3] Loading tokenizer: {args.model}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Tokenize
    print(f"\n[2/3] Tokenizing...")
    if args.input:
        tokens = tokenize_text_file(args.input, tokenizer, args.max_tokens)
    else:
        tokens = tokenize_hf_dataset(args.hf, tokenizer, max_tokens=args.max_tokens,
                                     text_field=args.text_field)

    # Qwen3 vocab (151K) exceeds uint16, save as uint32 with header
    max_tok = int(tokens.max())
    if max_tok > 65535:
        print(f"  Token IDs exceed uint16 (max={max_tok}), saving as uint32 format")
        tokens = tokens.astype(np.uint32)  # upgrade to uint32

    # Save
    print(f"\n[3/3] Saving...")
    save_binary(tokens, args.output)
    verify_data(args.output, tokenizer)

    print(f"\n  Next steps:")
    print(f"  1. Generate teacher logits:")
    print(f"     python generate_teacher.py --data {args.output} --output teacher_logits.bin")
    print(f"  2. Train with distillation:")
    print(f"     ./train --data {args.output} --distill teacher_logits.bin --scratch")


if __name__ == "__main__":
    main()
