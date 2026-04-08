"""
Phantom Memory CLI — command-line interface.

Usage:
    phantom start                          # Memory daemon only
    phantom start --enricher               # Memory + continuous enrichment
    phantom start --enricher --interval 60 # Custom interval
    phantom enrich                         # One-shot enrichment, then exit
    phantom dashboard                      # Launch web dashboard
    phantom eval                           # Run three-tier eval suite
    phantom stats                          # Print memory stats
    phantom version                        # Print version
"""

import argparse
import sys
import os
import signal


def cmd_start(args):
    """Start the memory daemon."""
    from phantom_memory.daemon import MemoryDaemon

    vault = os.path.expanduser(args.vault)
    db = os.path.expanduser(args.db) if args.db else None

    daemon = MemoryDaemon(
        vault_path=vault,
        db_path=db,
        enable_enricher=args.enricher,
        enricher_interval=args.interval,
    )
    daemon.start()

    status = "memory + enricher" if args.enricher else "memory only"
    print(f"[phantom] Started ({status}), vault: {vault}")
    if args.enricher:
        print(f"[phantom] Enricher interval: {args.interval}s")

    # Block until Ctrl+C
    try:
        signal.pause()
    except AttributeError:
        # Windows fallback
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[phantom] Shutting down...")
        daemon.stop()


def cmd_enrich(args):
    """Run one enrichment cycle and exit."""
    from phantom_memory.daemon import MemoryDaemon

    vault = os.path.expanduser(args.vault)
    db = os.path.expanduser(args.db) if args.db else None

    daemon = MemoryDaemon(vault_path=vault, db_path=db)
    daemon.start()

    print("[phantom] Running enrichment sweeps...")
    try:
        daemon.enrich_once()
    except AttributeError:
        from phantom_memory.enricher import PhantomEnricher
        enricher = PhantomEnricher(
            store=daemon.store, vault=daemon.vault, interval=300
        )
        enricher.run_once()
    print("[phantom] Sweeps complete.")
    daemon.stop()


def cmd_dashboard(args):
    """Launch the web dashboard."""
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dashboard.py"
    )
    port = args.port
    print(f"[phantom] Dashboard at http://localhost:{port}")
    os.execvp(sys.executable, [sys.executable, dashboard_path])


def cmd_eval(args):
    """Run the three-tier eval suite."""
    eval_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval_tiers.py"
    )
    os.execvp(sys.executable, [sys.executable, eval_path])


def cmd_stats(args):
    """Print memory stats."""
    from phantom_memory.daemon import MemoryStore

    vault = os.path.expanduser(args.vault)
    # Derive db path from vault
    db = os.path.expanduser(args.db) if args.db else os.path.join(
        os.path.dirname(vault), "memory", "chromadb_live"
    )

    if not os.path.exists(db):
        print(f"[phantom] No database found at {db}")
        return

    store = MemoryStore(db)
    total = store.collection.count()
    print(f"[phantom] Memories: {total}")
    print(f"[phantom] Database: {db}")
    print(f"[phantom] Vault: {vault}")


def cmd_version(args):
    """Print version."""
    from phantom_memory import __version__
    print(f"phantom-memory {__version__}")


def main():
    parser = argparse.ArgumentParser(
        prog="phantom",
        description="Phantom Memory — persistent memory for local LLMs",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Common args
    vault_default = os.environ.get("PHANTOM_VAULT", "~/vault")
    db_default = os.environ.get("PHANTOM_DB", None)

    # start
    p_start = subparsers.add_parser("start", help="Start memory daemon")
    p_start.add_argument("--enricher", action="store_true", help="Enable continuous enrichment")
    p_start.add_argument("--interval", type=int, default=300, help="Enricher interval in seconds (default: 300)")
    p_start.add_argument("--vault", default=vault_default, help=f"Vault path (default: {vault_default})")
    p_start.add_argument("--db", default=db_default, help="ChromaDB path (default: auto)")
    p_start.set_defaults(func=cmd_start)

    # enrich
    p_enrich = subparsers.add_parser("enrich", help="Run one enrichment cycle")
    p_enrich.add_argument("--vault", default=vault_default, help=f"Vault path (default: {vault_default})")
    p_enrich.add_argument("--db", default=db_default, help="ChromaDB path (default: auto)")
    p_enrich.set_defaults(func=cmd_enrich)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch web dashboard")
    p_dash.add_argument("--port", type=int, default=8422, help="Dashboard port (default: 8422)")
    p_dash.set_defaults(func=cmd_dashboard)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run three-tier eval suite")
    p_eval.set_defaults(func=cmd_eval)

    # stats
    p_stats = subparsers.add_parser("stats", help="Print memory stats")
    p_stats.add_argument("--vault", default=vault_default, help=f"Vault path (default: {vault_default})")
    p_stats.add_argument("--db", default=db_default, help="ChromaDB path (default: auto)")
    p_stats.set_defaults(func=cmd_stats)

    # version
    p_ver = subparsers.add_parser("version", help="Print version")
    p_ver.set_defaults(func=cmd_version)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
