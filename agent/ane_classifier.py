#!/usr/bin/env python3
"""
ANE-backed memory type classifier using compiled Neuron model.

Neuron (160M FFN-only Pythia, trained on ChromaDB data) classifies
memory types via CoreML ct.predict on ANE in ~900µs.

Domain classification stays with rule_classifier (regex, <1µs).
Neuron handles: decision, task, preference, quantitative, general.

Interface matches rule_classifier.py: classify(text) -> (domain, memory_type)

Copyright 2026 Nick Lo. MIT License.
"""

import logging
import os
import time
import numpy as np

log = logging.getLogger("ane_classifier")

NEURON_PATH = "/tmp/neuron_classifier/neuron_domain.mlpackage"
TYPE_NAMES = ["decision", "task", "preference", "quantitative", "general"]

_neuron_model = None
_tokenizer = None
_neuron_failed = False
MAX_SEQ_LEN = 128


def _load_neuron():
    """Load compiled Neuron classifier. One-time cost."""
    global _neuron_model, _tokenizer, _neuron_failed
    if _neuron_failed:
        return False
    if _neuron_model is not None:
        return True
    try:
        import coremltools as ct
        from transformers import AutoTokenizer
        _neuron_model = ct.models.MLModel(
            NEURON_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE)
        _tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m")
        log.info("Neuron classifier loaded from %s", NEURON_PATH)
        return True
    except Exception as e:
        log.warning("Neuron load failed: %s — using rule_classifier", e)
        _neuron_failed = True
        return False


def classify_type_neuron(text: str) -> str:
    """Classify memory type via Neuron on ANE. Returns type string."""
    if not _load_neuron():
        return None

    enc = _tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN,
                     padding="max_length", return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int32)

    result = _neuron_model.predict({"input_ids": input_ids})
    logits = result["domain_logits"].flatten()
    pred = int(logits.argmax())
    return TYPE_NAMES[pred]


def classify(text: str) -> tuple:
    """Classify text. Rule-based domain + Neuron type (ANE).

    Returns (domain, memory_type) matching rule_classifier interface.
    """
    from rule_classifier import classify_domain

    # Domain: rule-based (regex, <1µs, good for topic-switch detection)
    domain = classify_domain(text)

    # Memory type: Neuron on ANE (~900µs), fallback to rule_classifier
    mtype = classify_type_neuron(text)
    if mtype is None:
        from rule_classifier import classify_type
        mtype = classify_type(text)

    return domain, mtype


if __name__ == "__main__":
    tests = [
        "GPT-2 achieves 135.9 tok/s at 37 dispatches on ANE",
        "The user never sees Subconscious",
        "ANE has a 93us dispatch floor on M5 Pro",
        "ane-compiler shipped on GitHub MIT",
        "I prefer dark mode dashboards",
        "The next step is to wire retrieval into the agent",
    ]
    for t in tests:
        t0 = time.perf_counter()
        domain, mtype = classify(t)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        print(f"  {elapsed_us:7.0f}µs [{domain:>10}] [{mtype:>12}] {t}")
