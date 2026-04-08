"""CoreML MiniLM-L6-v2 embedder — drop-in replacement for SentenceTransformer.

Loads the precompiled CoreML package at ~/models/minilm-coreml/minilm.mlpackage
(built by ngram-engine/convert_minilm.py, validated cosine 0.999985 vs CPU
SentenceTransformer). Routes through ANE via CPU_AND_NE compute unit.

Measured on M5 Pro:
  • single-query ANE: 0.84 ms/embed (1,197 embeds/s)
  • single-query CPU baseline: 2.68 ms/embed (~3.2× slower)
  • batch CPU is competitive (0.33 ms/embed at batch 128) but ANE wins on
    the live single-query path that the agent hits per turn.

API: `encode(texts, normalize_embeddings=True, batch_size=..., show_progress_bar=...)`
matches the subset of SentenceTransformer's interface that the daemon, the
canonical_inject loop, the meta_memory loop, and the migration script use.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

SEQ_LEN = 128
HIDDEN = 384

DEFAULT_PACKAGE = Path.home() / "models" / "minilm-coreml" / "minilm.mlpackage"


class CoreMLMiniLMEmbedder:
    """Drop-in for sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2').

    Only implements the subset used by this codebase: `encode(texts, ...)`
    returning a normalized float32 ndarray of shape (N, 384). Tokenization
    uses the same HF tokenizer as SentenceTransformer so embeddings are
    drop-in compatible with the existing in-store vectors.
    """

    def __init__(self, package_path: str | os.PathLike = DEFAULT_PACKAGE):
        import coremltools as ct
        from transformers import AutoTokenizer

        self._ct = ct
        self.package_path = Path(package_path)
        if not self.package_path.exists():
            raise FileNotFoundError(
                f"CoreML MiniLM package not found at {self.package_path}. "
                "Run ngram-engine/convert_minilm.py to build it."
            )
        self.model = ct.models.MLModel(
            str(self.package_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    # ── public API (SentenceTransformer-compatible subset) ──
    def encode(self, texts, normalize_embeddings: bool = True,
               batch_size: int = 32, show_progress_bar: bool = False,
               convert_to_numpy: bool = True):
        """Embed a list (or single string) of texts. Returns (N, 384) float32."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, HIDDEN), dtype=np.float32)

        out = np.empty((len(texts), HIDDEN), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self._encode_one(text, normalize=normalize_embeddings)
        return out

    # ── internals ──
    def _encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        enc = self.tokenizer(
            text,
            padding="max_length",
            max_length=SEQ_LEN,
            truncation=True,
            return_tensors="np",
        )
        ids = enc["input_ids"].astype(np.int32)
        mask = enc["attention_mask"].astype(np.int32)

        result = self.model.predict({
            "input_ids": ids,
            "attention_mask": mask,
        })
        hidden = result["hidden_states"]  # (1, 128, 384) float16

        # Mean-pool over masked tokens (same as SentenceTransformer's mean
        # pooling layer for all-MiniLM-L6-v2).
        pmask = mask[:, :, None].astype(np.float32)
        pooled = (hidden.astype(np.float32) * pmask).sum(axis=1) / pmask.sum(axis=1)

        if normalize:
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.maximum(norm, 1e-12)
        return pooled[0].astype(np.float32)


def maybe_load_coreml_embedder():
    """Load the CoreML MiniLM embedder if available, else return None.

    Used as the feature-flag entry point. The daemon falls back to
    sentence_transformers when this returns None, so unwiring is one
    env var: `MIDAS_DISABLE_COREML_EMBED=1`.
    """
    if os.environ.get("MIDAS_DISABLE_COREML_EMBED"):
        return None
    try:
        return CoreMLMiniLMEmbedder()
    except Exception as e:
        import sys
        print(f"[coreml_embedder] failed to load: {e}; "
              f"falling back to CPU SentenceTransformer", file=sys.stderr)
        return None
