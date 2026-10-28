"""Running a bot in the background: `overseer start | stop | restart | status` (and the TUI's keys).

    overseer start      the bot runs on, detached from the terminal (closing it, or logging out, doesn't stop it).
                        With local.restart_on_crash (the default) a small watcher process runs it and starts it
                        again after a crash; never after a stop, nor when it can't start (a config error).
    overseer stop       a clean stop (like Ctrl-C): runs in progress are offered again at the next start.

Only one copy of a bot runs at a time, however it was started (here, in tmux, by systemd): `overseer run` takes a lock
in the bot's state dir, and a second copy refuses to start (two would fight over Telegram's updates).

What's where, in the bot's state dir:
    bot.lock            held by the running bot; holds its pid
    supervisor.json     the watcher: its pid, the bot's, its restarts (gone when it ends)
    logs/overseer.log   the bot's log (rotated); logs/console.log: what it printed (errors before logging started)
"""

from __future__ import annotations

import fcntl
import json
import logging
import logging.handlers
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

LOCK_FILE = "bot.lock"
SUPERVISOR_FILE = "supervisor.json"
LOG_FILE = "overseer.log"
CONSOLE_FILE = "console.log"
EXIT_CONFIG = 2  # `overseer run`'s exit codes the watcher doesn't restart after: a config error,
EXIT_RUNNING = 3  # another copy already running
STOP_WAIT = 35  # seconds a stop waits before killing (the bot stops its agent runs first)
FIRST_DELAY = 30  # seconds before the first restart after a crash; doubled each time, up to MAX_DELAY
MAX_DELAY = 600
FAST_CRASH = 60  # a crash this soon after starting counts as "can't start"
MAX_FAST_CRASHES = 5  # ... and after this many in a row, the watcher gives up


class AlreadyRunning(RuntimeError):
    pass


# ============================================================================ in the bot: lock, log file
def acquire_lock(state_dir: Path) -> int:
    """The bot's single-instance lock (kept until the process ends): its fd. AlreadyRunning if another copy holds it."""
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(state_dir / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        pid = read_pid(state_dir / LOCK_FILE)
        raise AlreadyRunning(f"this bot is already running (pid {pid or '?'}): `overseer stop` stops it") from None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    os.set_inheritable(fd, False)
    return fd


def log_to_file(state_dir: Path) -> Path:
    """The bot's log also goes to state/logs/overseer.log (the TUI's Log tab), rotated at 2 MB, 3 kept."""
    d = state_dir / "logs"
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = d / LOG_FILE
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    os.chmod(path, 0o600)
    return path


# ============================================================================ looking (read-only)
def read_pid(path: Path) -> int:
    try:
        return int(path.read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def alive(pid: int) -> bool:
    """A live process of Overseer's (not a recycled pid, nor a zombie)."""
    if pid <= 0:
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return b"overseer" in cmd and stat.rsplit(")", 1)[-1].split()[0] != "Z"


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@dataclass
class State:
    """A bot's process, as seen from outside."""
    pid: int = 0  # the bot (0: not running)
    supervisor: dict = field(default_factory=dict)  # the watcher's file, if it runs

    @property
    def running(self) -> bool:
        return self.pid > 0

    @property
    def watched(self) -> bool:
        return bool(self.supervisor)

    def restarts_since(self, since: float) -> int:
        return sum(1 for t in self.supervisor.get("restarts", []) if t >= since)


def state(state_dir: Path) -> State:
    pid = read_pid(state_dir / LOCK_FILE)
    sup = read_json(state_dir / SUPERVISOR_FILE)
    return State(pid=pid if alive(pid) else 0, supervisor=sup if alive(int(sup.get("pid") or 0)) else {})


# ============================================================================ start, stop
def _argv(config: Path, command: str) -> list[str]:
    return [sys.executable, "-m", "overseer", command, "-c", str(config)]


def spawn_detached(argv: list[str], cwd: Path, out: Path) -> None:
    """Run argv in the background, in a session of its own, adopted by init: it outlives this process and its
    terminal (and is never a zombie of ours). Its output goes to `out`."""
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    script = f'exec "$@" >>{shlex.quote(str(out))} 2>&1 </dev/null &'
    subprocess.run(["/bin/sh", "-c", script, "overseer", *argv], cwd=str(cwd), start_new_session=True, check=True,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start(cfg: Config, watched: bool | None = None, wait: float = 20) -> tuple[bool, str]:
    """Start the bot in the background: (started, what happened)."""
    state_dir = cfg.local.state()
    st = state(state_dir)
    if st.running:
        return False, f"already running (pid {st.pid})"
    config = Path(getattr(cfg, "source", "") or Path(cfg.local.base_dir) / "config.yaml")
    watched = cfg.local.restart_on_crash if watched is None else watched
    console = state_dir / "logs" / CONSOLE_FILE
    size = console.stat().st_size if console.exists() else 0
    spawn_detached(_argv(config, "supervise" if watched else "run"), Path(cfg.local.base_dir), console)
    began = time.time()
    while time.time() - began < wait:
        st = state(state_dir)
        if st.running:
            how = " (restarted if it crashes)" if watched else ""
            return True, f"started in the background, pid {st.pid}{how}"
        if time.time() - began > 2 and not st.watched and _console_since(console, size).strip():
            break  # it ended, saying why
        time.sleep(0.2)
    if state(state_dir).watched:
        return False, ("it hasn't started yet: the watcher restarts it after a pause - see the log"
                       + (":\n" + "\n".join(_console_since(console, size).strip().splitlines()[-8:])))
    tail = _console_since(console, size).strip().splitlines()[-8:]
    return False, "it didn't start" + (":\n" + "\n".join(tail) if tail else f" (see {console})")


def _console_since(path: Path, offset: int) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def stop(cfg: Config, wait: float = STOP_WAIT) -> tuple[bool, str]:
    """A clean stop (SIGTERM, like Ctrl-C) of the bot and its watcher; killed if it takes longer than `wait`."""
    state_dir = cfg.local.state()
    st = state(state_dir)
    if not st.running and not st.watched:
        return False, "not running"
    sup_pid = int(st.supervisor.get("pid") or 0)
    for pid in (sup_pid, st.pid):  # (the watcher passes it on to the bot; the bot too, in case it isn't watched)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    end = time.time() + wait
    while time.time() < end:
        st = state(state_dir)
        if not st.running and not st.watched:
            return True, "stopped"
        time.sleep(0.2)
    for pid in (st.pid, int(st.supervisor.get("pid") or 0)):
        if pid and alive(pid):
            os.kill(pid, signal.SIGKILL)
    return True, f"stopped (killed: it didn't stop within {int(wait)} s)"


# ============================================================================ the watcher (`overseer supervise`)
def supervise(cfg: Config) -> int:
    """Run the bot, and run it again after a crash (with a growing pause). Ends with the bot when it stops cleanly,
    can't start (a config error, another copy running), or keeps crashing right away; a SIGTERM stops both."""
    state_dir = cfg.local.state()
    sup_file = state_dir / SUPERVISOR_FILE
    config = Path(getattr(cfg, "source", "") or Path(cfg.local.base_dir) / "config.yaml")
    if alive(int(read_json(sup_file).get("pid") or 0)):
        print("a watcher already runs this bot", file=sys.stderr)
        return EXIT_RUNNING
    info: dict = {"pid": os.getpid(), "started": time.time(), "restarts": [], "state": "running", "bot": 0,
                  "last_exit": None}
    stopping = {"now": False}
    child: list[subprocess.Popen] = []

    def on_term(signum, frame):
        stopping["now"] = True
        if child and child[0].poll() is None:
            child[0].send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    from .heartbeat import write_json
    delay, fast = FIRST_DELAY, 0
    try:
        while not stopping["now"]:
            began = time.time()
            proc = subprocess.Popen(_argv(config, "run"), cwd=cfg.local.base_dir, stdin=subprocess.DEVNULL)
            child[:] = [proc]
            info.update(bot=proc.pid, state="running", bot_started=began)
            write_json(sup_file, info)
            try:
                rc = proc.wait()
            except KeyboardInterrupt:
                rc = proc.wait()
            if stopping["now"]:
                deadline = time.time() + STOP_WAIT
                while proc.poll() is None and time.time() < deadline:
                    time.sleep(0.2)
                if proc.poll() is None:
                    proc.kill()
                break
            info["last_exit"] = rc
            if rc in (0, EXIT_CONFIG, EXIT_RUNNING) or rc in (-signal.SIGTERM, -signal.SIGINT):
                break  # stopped on purpose, or it can't start: not a crash
            fast = fast + 1 if time.time() - began < FAST_CRASH else 0
            if fast >= MAX_FAST_CRASHES:
                info["state"] = "gave up"
                write_json(sup_file, info)
                print(f"overseer supervise: the bot crashed {fast} times right after starting; not restarting it",
                      file=sys.stderr)
                break
            if time.time() - began > MAX_DELAY:
                delay = FIRST_DELAY  # (it ran a good while: start the pauses over)
            info["restarts"].append(time.time())
            info.update(state=f"restarting in {delay} s", bot=0, restart_at=time.time() + delay)
            write_json(sup_file, info)
            print(f"overseer supervise: the bot exited with {rc}; restarting it in {delay} s", file=sys.stderr)
            end = time.time() + delay
            while time.time() < end and not stopping["now"]:
                time.sleep(0.5)
            delay = min(delay * 2, MAX_DELAY)
    finally:
        sup_file.unlink(missing_ok=True)
    return 0


# ============================================================================ logging out
def linger_note() -> str:
    """Why bots started in the background might stop at logout, or "": systemd-logind kills a user's processes at
    logout when KillUserProcesses=yes, unless the user has lingering on."""
    kill = False
    for f in [Path("/etc/systemd/logind.conf"), *sorted(Path("/etc/systemd/logind.conf.d").glob("*.conf"))]:
        try:
            for line in f.read_text().splitlines():
                k, _, v = line.strip().partition("=")
                if k.strip() == "KillUserProcesses":
                    kill = v.strip().lower() in ("yes", "true", "1", "on")
        except OSError:
            continue
    if not kill:
        return ""
    user = os.environ.get("USER", "")
    if user and Path(f"/var/lib/systemd/linger/{user}").exists():
        return ""
    return ("This system ends a user's processes at logout (KillUserProcesses=yes): the bots would stop when you "
            "log out. `loginctl enable-linger` keeps them running.")
