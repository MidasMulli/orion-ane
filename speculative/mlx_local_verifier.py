"""
In-process MLX Verifier for Speculative Decoding
=================================================

Loads Qwen3.5-9B directly via mlx_lm — no HTTP, no re-prefill.
Maintains a KV cache across rounds. Verifies K draft tokens in a
single batch forward pass producing K+1 logits.

This is the correct architecture for speculative decoding:
  - Shared process = no serialization overhead
  - Persistent KV cache = no re-prefill
  - Batch verify = one forward pass for K tokens
"""

import time
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


MLX_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


class MLXLocalVerifier:
    """In-process MLX verifier with persistent KV cache."""

    def __init__(self, model_path=MLX_MODEL):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.cache_pos = 0  # Tokens consumed by cache

    def load(self, status_fn=None):
        """Load the MLX model."""
        def status(msg):
            if status_fn:
                status_fn(msg)
            print(f"  [MLX] {msg}")

        status(f"Loading {self.model_path}...")
        t0 = time.time()
        self.model, self.tokenizer = load(self.model_path)
        status(f"Model loaded in {time.time() - t0:.1f}s")
        return True

    def reset_cache(self):
        """Reset KV cache for new generation."""
        self.cache = make_prompt_cache(self.model)
        self.cache_pos = 0

    def prefill(self, token_ids):
        """Prefill KV cache with prompt tokens. Returns logits for last token."""
        self.reset_cache()
        x = mx.array([token_ids])
        logits = self.model(x, cache=self.cache)
        mx.eval(logits)
        self.cache_pos = len(token_ids)
        # Return logits for the last position → next token prediction
        return logits[0, -1, :].tolist()

    def verify_draft(self, draft_token_ids):
        """
        Verify K draft tokens in a single forward pass.

        Feeds all K draft tokens through the model at once.
        Returns the model's greedy token at each position:
          - Position 0: what the model predicts after the context (before seeing draft[0])
          - Position i: what the model predicts after seeing draft[0..i-1]

        The caller already has logits from prefill for position 0.
        This call produces logits for positions 1..K.

        Returns: (verifier_tokens, time_ms)
          verifier_tokens[i] = argmax of logits after feeding draft[0..i]
        """
        t0 = time.time()

        # Feed all K draft tokens at once
        x = mx.array([draft_token_ids])
        logits = self.model(x, cache=self.cache)
        mx.eval(logits)

        # logits shape: [1, K, vocab_size]
        # logits[0, i, :] = prediction after seeing draft[0..i]
        # So verifier_tokens[i] = argmax(logits[0, i, :])
        verifier_tokens = mx.argmax(logits[0], axis=-1).tolist()

        self.cache_pos += len(draft_token_ids)
        t_ms = (time.time() - t0) * 1000

        return verifier_tokens, t_ms

    def rollback_cache(self, n_tokens):
        """
        Trim the last n_tokens from the KV cache.
        Used after draft rejection to remove speculative entries.
        """
        if self.cache is not None:
            for layer_cache in self.cache:
                # KVCache objects in mlx_lm have a trim method
                if hasattr(layer_cache, 'trim'):
                    layer_cache.trim(n_tokens)
            self.cache_pos -= n_tokens

    def feed_token(self, token_id):
        """Feed a single correction token (after rejection)."""
        x = mx.array([[token_id]])
        logits = self.model(x, cache=self.cache)
        mx.eval(logits)
        self.cache_pos += 1
        return mx.argmax(logits[0, -1, :]).item()
