#!/usr/bin/env python3
"""
Phantom Enricher Service — runs continuously as a background daemon.

Launched by launchd at login. Runs the five enrichment sweeps on a timer,
writing results to the Obsidian vault. No LLM needed — CPU only.

Usage:
    python3 enricher_service.py                    # Default: 300s interval
    python3 enricher_service.py --interval 60      # Custom interval
    python3 enricher_service.py --once             # One-shot, then exit
"""

import os
import sys
import signal
import argparse
import logging
import warnings
from datetime import datetime

# Suppress noisy library output
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for name in ("httpx", "httpcore", "sentence_transformers", "chromadb",
             "huggingface_hub", "ane_server"):
    logging.getLogger(name).setLevel(logging.CRITICAL)

# Setup logging for the service itself
log = logging.getLogger("phantom.service")
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [enricher] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(handler)

# Paths
VAULT_PATH = "/Users/midas/Desktop/cowork/vault"
DB_PATH = "/Users/midas/Desktop/cowork/orion-ane/memory/chromadb_live"
PID_FILE = "/tmp/phantom-enricher.pid"


def write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def cleanup_pid(*_):
    try:
        os.remove(PID_FILE)
    except FileNotFoundError:
        pass
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Phantom Enricher Service")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between sweeps (default: 300)")
    parser.add_argument("--once", action="store_true", help="Run one sweep cycle and exit")
    args = parser.parse_args()

    # PID file for status checks
    write_pid()
    signal.signal(signal.SIGTERM, cleanup_pid)
    signal.signal(signal.SIGINT, cleanup_pid)

    log.info("Starting Phantom Enricher Service (pid=%d, interval=%ds)", os.getpid(), args.interval)

    # Import and boot the daemon with enricher
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from phantom_memory.daemon import MemoryDaemon

    daemon = MemoryDaemon(
        vault_path=VAULT_PATH,
        db_path=DB_PATH,
        enable_enricher=not args.once,  # Background loop unless --once
        enricher_interval=args.interval,
    )

    if args.once:
        # One-shot: run all sweeps once and exit
        log.info("One-shot mode — running all sweeps")
        daemon.start()
        try:
            daemon.enrich_once()
        except AttributeError:
            # If enrich_once doesn't exist, run via enricher directly
            from phantom_memory.enricher import PhantomEnricher
            enricher = PhantomEnricher(store=daemon.store, vault=daemon.vault, interval=args.interval)
            enricher.run_once()
        log.info("Sweeps complete. Exiting.")
        cleanup_pid()
    else:
        # Continuous mode — start daemon with enricher thread, then block
        daemon.start()
        log.info("Enricher running. Sweeps every %ds. Vault: %s", args.interval, VAULT_PATH)

        # Write a heartbeat file so the briefing can check freshness
        heartbeat_path = os.path.join(VAULT_PATH, "midas", ".enricher_heartbeat")
        os.makedirs(os.path.dirname(heartbeat_path), exist_ok=True)

        try:
            import time
            while True:
                # Update heartbeat
                with open(heartbeat_path, "w") as f:
                    f.write(datetime.now().isoformat())
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            log.info("Shutting down enricher service")
            daemon.stop()
            cleanup_pid()


if __name__ == "__main__":
    main()
