#!/usr/bin/env python3
"""
Phantom Scanner Service — multi-source feed scanner running as background daemon.

Launched by launchd at login. Scans HN, RSS, Reddit (API tier) on every cycle.
Scans X only when Chrome CDP is available (browser tier — opportunistic).

Usage:
    python3 scanner_service.py                    # Default: 3600s (1 hour)
    python3 scanner_service.py --interval 1800    # Every 30 minutes
    python3 scanner_service.py --once             # One-shot, then exit
"""

import os
import sys
import signal
import argparse
import logging
import warnings
import time
from datetime import datetime

# Suppress noisy library output
warnings.filterwarnings("ignore")

# Setup logging
log = logging.getLogger("phantom.scanner")
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [scanner] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(handler)

# Paths
VAULT_PATH = "/Users/midas/Desktop/cowork/vault"
PID_FILE = "/tmp/phantom-scanner.pid"
HEARTBEAT_PATH = os.path.join(VAULT_PATH, "midas", ".scanner_heartbeat")


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
    parser = argparse.ArgumentParser(description="Phantom Scanner Service")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between scans (default: 3600)")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle and exit")
    args = parser.parse_args()

    write_pid()
    signal.signal(signal.SIGTERM, cleanup_pid)
    signal.signal(signal.SIGINT, cleanup_pid)

    log.info("Starting Phantom Scanner Service (pid=%d, interval=%ds)", os.getpid(), args.interval)

    # Import scanner
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scanner import Scanner

    # Try to set up browser bridge for Tier 2 sources
    browser = None
    try:
        from browser import BrowserBridge
        b = BrowserBridge()
        if b.is_available():
            b.connect()
            browser = b
            log.info("Chrome CDP available — Tier 2 sources (X) enabled")
        else:
            log.info("Chrome CDP unavailable — Tier 2 sources skipped")
    except Exception as e:
        log.info("Browser bridge unavailable: %s — Tier 2 sources skipped", e)

    scanner = Scanner(browser=browser)

    if args.once:
        log.info("One-shot mode — scanning all sources")
        result = scanner.run_cycle()
        log.info("Scan complete: %d candidates from %d sources (%s failed)",
                 result["total"], 4 - len(result["failed"]), result["failed"] or "none")
        cleanup_pid()
    else:
        log.info("Continuous mode — scanning every %ds", args.interval)
        os.makedirs(os.path.dirname(HEARTBEAT_PATH), exist_ok=True)

        while True:
            try:
                # Reconnect browser if it became available
                if not browser:
                    try:
                        from browser import BrowserBridge
                        b = BrowserBridge()
                        if b.is_available():
                            b.connect()
                            browser = b
                            scanner.browser = browser
                            log.info("Chrome CDP reconnected — Tier 2 sources enabled")
                    except Exception:
                        pass

                result = scanner.run_cycle()
                log.info("Scan: %d candidates, failed: %s", result["total"], result["failed"] or "none")

                # Write heartbeat
                with open(HEARTBEAT_PATH, "w") as f:
                    f.write(datetime.now().isoformat())

            except Exception as e:
                log.error("Scan cycle failed: %s", e)

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
