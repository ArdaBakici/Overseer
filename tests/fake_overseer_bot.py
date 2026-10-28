"""Stands in for `overseer run` in the background-process tests: takes the bot's real lock (core/daemon.py), then
behaves as told: run until stopped (exit 0 on SIGTERM, like the real bot), crash once first, or fail to start."""
import signal
import sys
import time
from pathlib import Path

from overseer.core import daemon

state, mode = Path(sys.argv[1]), sys.argv[2]
if mode == "config-error":
    print("config error: telegram.allowed_chat_ids is empty", file=sys.stderr)
    sys.exit(daemon.EXIT_CONFIG)
try:
    daemon.acquire_lock(state)
except daemon.AlreadyRunning as e:
    print(e, file=sys.stderr)
    sys.exit(daemon.EXIT_RUNNING)
marker = state / "crashed-once"
if mode == "crash-once" and not marker.exists():
    marker.write_text("x")
    time.sleep(0.3)
    sys.exit(1)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
while True:
    time.sleep(0.1)
