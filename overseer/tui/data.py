"""What the TUI shows, read from the bots' files - and only read: the database is opened read-only, nothing is written,
nothing is asked of the bots, Telegram or the HPC.

A bot is a config file in Overseer's folder (config.yaml, assistant.yaml, or one a bot noted in state/bots/). From
its state dir: status.json (the running bot's heartbeat), overseer.db, transcripts/, logs/, and the process files
of core/daemon.py.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from ..core import daemon
from ..core.config import Config, ConfigError, _build, read_yaml
from ..core.heartbeat import STALE, STATUS_FILE

DOMAIN_LABELS = {"hpc": "HPC bot", "assistant": "Assistant"}


def peek_config(path: Path) -> Config:
    """A bot's config, as `overseer run` would read it - without loading its .env or changing this process's time
    zone (the TUI shows several bots)."""
    cfg = _build(Config, read_yaml(path), "")
    cfg.local.base_dir = str(path.resolve().parent)
    cfg.source = str(path.resolve())
    cfg.validate()
    return cfg


@dataclass
class Bot:
    config: Path
    cfg: Config | None = None
    error: str = ""  # the config doesn't load

    @property
    def domain(self) -> str:
        return self.cfg.domain if self.cfg else "?"

    @property
    def label(self) -> str:
        name = DOMAIN_LABELS.get(self.domain, self.domain)
        return name if self.config.name in ("config.yaml", "assistant.yaml") else f"{name} ({self.config.name})"

    @property
    def state_dir(self) -> Path | None:
        return self.cfg.local.state() if self.cfg else None


def discover(base: Path) -> list[Bot]:
    """The bots of this Overseer folder: config.yaml, assistant.yaml, and any other config a bot noted when it ran."""
    paths = [p for p in (base / "config.yaml", base / "assistant.yaml") if p.is_file()]
    for f in sorted((base / "state" / "bots").glob("*.json")) if (base / "state" / "bots").is_dir() else []:
        info = daemon.read_json(f)
        p = Path(info.get("config") or "")
        if p.is_file() and p.resolve() not in [x.resolve() for x in paths]:
            paths.append(p)
    bots = []
    for p in paths:
        try:
            bots.append(Bot(p, peek_config(p)))
        except (OSError, ConfigError) as e:
            bots.append(Bot(p, error=str(e)))
    return bots


# ============================================================================ the process and its heartbeat
@dataclass
class Health:
    running: bool = False
    pid: int = 0
    watched: bool = False  # the watcher (restart after a crash) runs it
    supervisor: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)  # its heartbeat (the last one, if it stopped)
    stale: bool = False  # running, but the heartbeat stopped coming (stuck, or still starting)

    @property
    def label(self) -> str:
        if self.running:
            return "● running" + (" (not responding)" if self.stale else "")
        if self.watched:
            return f"↻ {self.supervisor.get('state', 'restarting')}"
        return "○ stopped"


def health(bot: Bot) -> Health:
    if not bot.state_dir:
        return Health()
    st = daemon.state(bot.state_dir)
    status = daemon.read_json(bot.state_dir / STATUS_FILE)
    stale = st.running and time.time() - float(status.get("updated") or 0) > STALE
    return Health(running=st.running, pid=st.pid, watched=st.watched, supervisor=st.supervisor, status=status,
                  stale=stale)


# ============================================================================ the database, read-only
def connect_ro(db: Path) -> sqlite3.Connection | None:
    """The bot's database, read-only (SQLite refuses any write through it); None if there's none yet."""
    if not db.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{quote(str(db))}?mode=ro", uri=True, timeout=2, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con
    except sqlite3.Error:
        return None


def _rows(con: sqlite3.Connection | None, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    if con is None:
        return []
    try:
        return con.execute(sql, args).fetchall()
    except sqlite3.Error:  # (a database from an older Overseer, or locked for a moment: shown next refresh)
        return []


def conversations(con, limit: int = 200) -> list[dict]:
    rows = _rows(con, "SELECT * FROM sessions ORDER BY updated DESC LIMIT ?", (limit,))
    return [{"id": r["id"], "project": _get(r, "project") or "", "backend": _get(r, "agent_backend") or r["backend"] or "",
             "role": _get(r, "role") or "", "name": _get(r, "name") or "", "job": r["job_id"] or "",
             "updated": r["updated"] or 0} for r in rows]


def wakeups(con) -> list[dict]:
    rows = _rows(con, "SELECT * FROM watches WHERE state='active' ORDER BY due")
    return [{"id": r["id"], "session": r["session_id"], "kind": r["kind"], "jobs": r["jobs"] or "",
             "every": r["every"] or 0, "due": r["due"] or 0, "note": r["note"] or "", "until": _get(r, "until") or 0}
            for r in rows]


def usage(con, since: float) -> list[dict]:
    return [dict(r) for r in _rows(con, "SELECT * FROM usage WHERE ts >= ?", (since,))]


def _get(row: sqlite3.Row, key: str):
    return row[key] if key in row.keys() else None


# ============================================================================ transcripts and the log
def transcript_events(state_dir: Path, sid: str, last: int = 400) -> list[dict]:
    path = state_dir / "transcripts" / f"{sid}.jsonl"
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 400_000))
            lines = fh.read().split(b"\n")
    except OSError:
        return []
    events = []
    for raw in lines[1:] if len(lines) > 1 and size > 400_000 else lines:
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events[-last:]


def log_tail(state_dir: Path, lines: int = 400) -> list[str]:
    out: list[str] = []
    for name in (daemon.LOG_FILE, daemon.CONSOLE_FILE):
        path = state_dir / "logs" / name
        try:
            with open(path, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 200_000))
                out = fh.read().decode(errors="replace").splitlines()[-lines:]
        except OSError:
            continue
        if out:
            return out
    return out


def ago(ts: float) -> str:
    d = max(0, int(time.time() - (ts or 0)))
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if d >= n:
            return f"{d // n}{unit}"
    return f"{d}s"


def until(ts: float) -> str:
    d = int((ts or 0) - time.time())
    if d <= 0:
        return "now"
    return "in " + ago(time.time() - d)


def human(n: float) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else str(int(n))
