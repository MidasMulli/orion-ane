"""
Phase 5A: System Monitor — background thread for environmental awareness.

Monitors system state and writes to signal_bus:
  - user_idle_seconds: time since last user input
  - thermal_pressure: from sysctl
  - memory_pressure_mb: from psutil
  - system_mode: active / idle / deep_idle

Modes:
  ACTIVE (input within 60s): Normal operation
  IDLE (60-300s): Light maintenance eligible
  DEEP_IDLE (>300s): Heavy maintenance eligible (conceptual extraction, reranking)
"""

import time
import threading
import logging
import subprocess

log = logging.getLogger("system_monitor")

ACTIVE_THRESHOLD = 60     # seconds
IDLE_THRESHOLD = 300      # seconds


class SystemMonitor:
    """Background thread that monitors system state."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._last_input_time = time.time()

    def start(self):
        """Start the monitor background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("System monitor started")

    def stop(self):
        self._running = False

    def touch(self):
        """Called on user input to reset idle timer."""
        self._last_input_time = time.time()

    def _run(self):
        while self._running:
            try:
                self._update()
            except Exception as e:
                log.debug(f"Monitor update error: {e}")
            time.sleep(10)  # Update every 10 seconds

    def _update(self):
        from signal_bus import update_batch

        idle_sec = time.time() - self._last_input_time

        # Determine system mode
        if idle_sec < ACTIVE_THRESHOLD:
            mode = "active"
        elif idle_sec < IDLE_THRESHOLD:
            mode = "idle"
        else:
            mode = "deep_idle"

        # Memory pressure (lightweight check)
        try:
            import psutil
            mem = psutil.virtual_memory()
            pressure_mb = int((mem.total - mem.available) / 1e6)
        except ImportError:
            pressure_mb = 0

        # Thermal (lightweight)
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
                capture_output=True, text=True, timeout=2)
            level = int(result.stdout.strip()) if result.stdout.strip() else 0
            thermal = "nominal" if level < 50 else "warm" if level < 80 else "hot"
        except Exception:
            thermal = "nominal"

        update_batch({
            "user_idle_seconds": int(idle_sec),
            "system_mode": mode,
            "thermal_pressure": thermal,
            "memory_pressure_mb": pressure_mb,
        })


# Singleton
_monitor = None


def get_monitor():
    """Get or create the system monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    return _monitor
