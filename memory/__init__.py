"""
Phantom Memory — Zero-cost persistent memory for local LLMs.

Usage:
    from phantom_memory import MemoryDaemon
    daemon = MemoryDaemon(vault_path="~/vault", enable_enricher=True)
    daemon.start()
"""

from phantom_memory.daemon import MemoryDaemon, FactExtractor, MemoryStore, VaultWriter
from phantom_memory.enricher import PhantomEnricher, SweepEngine

__version__ = "0.1.0"
__all__ = [
    "MemoryDaemon",
    "FactExtractor",
    "MemoryStore",
    "VaultWriter",
    "PhantomEnricher",
    "SweepEngine",
]
