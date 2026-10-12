"""Local SQLite state (lives on the homelab, in local.state_dir)."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    name TEXT, state TEXT, partition TEXT,
    workdir TEXT, stdout TEXT, stderr TEXT,
    batch_script TEXT,
    by_agent INTEGER DEFAULT 0,
    session_id TEXT,
    first_seen REAL, last_status_at REAL, ended_at REAL,
    missing_polls INTEGER DEFAULT 0,
    report TEXT,
    info TEXT,
    agent_reported INTEGER DEFAULT 0,
    checkins INTEGER DEFAULT 0,
    last_checkin REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    backend TEXT,
    cli_session_id TEXT,
    created REAL, updated REAL
);
CREATE TABLE IF NOT EXISTS messages (
    chat_id INTEGER, message_id INTEGER,
    job_id TEXT, session_id TEXT,
    PRIMARY KEY (chat_id, message_id)
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS threads (
    chat_id INTEGER, thread_id INTEGER,
    session_id TEXT,
    created REAL,
    PRIMARY KEY (chat_id, thread_id)
);
CREATE TABLE IF NOT EXISTS projects (
    name TEXT PRIMARY KEY,
    dir TEXT,
    context TEXT,
    threads TEXT DEFAULT '{}',
    archived INTEGER DEFAULT 0,
    source TEXT,
    created REAL
);
CREATE TABLE IF NOT EXISTS autopilot (
    session_id TEXT PRIMARY KEY,
    active INTEGER DEFAULT 0,
    goal TEXT,
    max_rounds INTEGER DEFAULT 0,
    rounds INTEGER DEFAULT 0,
    checkin_every REAL DEFAULT 0,
    status TEXT,
    created REAL, updated REAL
);
CREATE TABLE IF NOT EXISTS watches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    kind TEXT,
    jobs TEXT DEFAULT '',
    every REAL DEFAULT 0,
    due REAL DEFAULT 0,
    note TEXT DEFAULT '',
    oneoff INTEGER DEFAULT 0,
    state TEXT DEFAULT 'active',
    created REAL, fired REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    ts REAL, day TEXT,
    session_id TEXT, project TEXT,
    backend TEXT, model TEXT,
    origin TEXT,
    input INTEGER DEFAULT 0, cached INTEGER DEFAULT 0, output INTEGER DEFAULT 0,
    cost REAL
);
"""
# Columns added after the first release: (table, column, definition).
ADDED_COLUMNS = [
    ("jobs", "agent_reported", "INTEGER DEFAULT 0"),
    ("jobs", "checkins", "INTEGER DEFAULT 0"),
    ("jobs", "last_checkin", "REAL DEFAULT 0"),
    ("jobs", "project", "TEXT"),
    ("sessions", "project", "TEXT"),
    ("jobs", "ignored", "INTEGER DEFAULT 0"),
    ("autopilot", "hours", "REAL DEFAULT 0"),
    ("autopilot", "expires", "REAL DEFAULT 0"),
    ("watches", "until", "REAL DEFAULT 0"),
]


@dataclass
class JobRecord:
    job_id: str
    name: str = ""
    state: str = ""
    partition: str = ""
    workdir: str = ""
    stdout: str = ""
    stderr: str = ""
    batch_script: str = ""
    by_agent: int = 0  # submitted by the agent (detected from its sbatch output)
    session_id: str | None = None  # agent conversation that submitted it
    first_seen: float = 0.0
    last_status_at: float = 0.0
    ended_at: float | None = None
    missing_polls: int = 0
    report: str = ""
    info: str = "{}"
    agent_reported: int = 0  # autopilot: the agent has been woken with this job's final report
    checkins: int = 0  # autopilot: check-ins on this job while it ran
    last_checkin: float = 0.0
    project: str | None = None  # None = the main chat
    ignored: int = 0  # "deleted": Overseer stops tracking and reporting it (the job itself isn't touched)

    @property
    def info_dict(self) -> dict[str, Any]:
        try:
            return json.loads(self.info or "{}")
        except json.JSONDecodeError:
            return {}


@dataclass
class SessionRecord:
    id: str
    job_id: str | None = None
    backend: str = ""
    cli_session_id: str | None = None  # Claude Code session id / Codex thread id
    created: float = 0.0
    updated: float = 0.0
    project: str | None = None  # None = the main chat


@dataclass
class ProjectRecord:
    name: str
    dir: str = ""  # project root on the HPC
    context: str = ""  # standing instructions for the agent
    threads: str = "{}"  # {chat_id: Telegram topic (message_thread_id)}
    archived: int = 0
    source: str = ""  # config | telegram
    created: float = 0.0

    @property
    def thread_map(self) -> dict[str, int]:
        try:
            return {str(k): int(v) for k, v in json.loads(self.threads or "{}").items()}
        except (ValueError, TypeError):
            return {}

    def set_thread(self, chat_id: int, thread_id: int | None) -> None:
        m = self.thread_map
        if thread_id is None:
            m.pop(str(chat_id), None)
        else:
            m[str(chat_id)] = int(thread_id)
        self.threads = json.dumps(m)


@dataclass
class AutopilotRecord:
    session_id: str
    active: int = 0
    goal: str = ""
    max_rounds: int = 0
    rounds: int = 0  # wake-ups after jobs finished (check-ins don't count)
    checkin_every: float = 0.0  # seconds between check-ins on running jobs; 0 = none
    status: str = ""  # requested | declined | waiting | working | idle | need_help | done | limit | expired | paused | off
    created: float = 0.0
    updated: float = 0.0
    hours: float = 0.0  # how long the approval lasts
    expires: float = 0.0  # when it ends (set when it starts)


@dataclass
class WatchRecord:
    """Something that wakes the agent: jobs ending (kind "jobs") or a timer (kind "timer")."""
    session_id: str
    kind: str  # jobs | timer
    jobs: str = ""  # job ids, space-separated: the agent is woken when all have ended, or one failed
    every: float = 0.0  # jobs: seconds between check-ins while they run; timer: repeat interval (0 = once)
    due: float = 0.0  # timer: when it fires next
    note: str = ""  # what the agent wanted to do when woken
    oneoff: int = 0  # (no longer used: from when wake-ups needed approval)
    state: str = "active"  # active | fired | cancelled
    created: float = 0.0
    fired: float = 0.0
    until: float = 0.0  # repeating timer: no more wake-ups after this
    id: int | None = None

    @property
    def job_ids(self) -> list[str]:
        return self.jobs.split()


def _row(cls, row):
    names = {f.name for f in fields(cls)}
    return cls(**{k: row[k] for k in row.keys() if k in names})


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        os.chmod(path, 0o600)
        self.db.row_factory = sqlite3.Row
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(jobs)")}
        if cols and "by_agent" not in cols:  # database from an older Overseer version: keep it aside
            self.db.execute(f"ALTER TABLE jobs RENAME TO jobs_old_{int(time.time())}")
        self.db.executescript(SCHEMA)
        for table, col, definition in ADDED_COLUMNS:
            if col not in {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
        self.db.commit()

    def _upsert(self, table: str, key: str, obj: Any) -> None:
        d = obj.__dict__
        cols = ",".join(d)
        updates = ",".join(f"{k}=excluded.{k}" for k in d if k != key)
        self.db.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({','.join('?' * len(d))}) "
            f"ON CONFLICT({key}) DO UPDATE SET {updates}",
            tuple(d.values()),
        )
        self.db.commit()

    # ---------------------------------------------------------------- jobs
    def get_job(self, job_id: str) -> JobRecord | None:
        row = self.db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _row(JobRecord, row) if row else None

    def upsert_job(self, rec: JobRecord) -> None:
        self._upsert("jobs", "job_id", rec)

    def active_jobs(self) -> list[JobRecord]:
        rows = self.db.execute("SELECT * FROM jobs WHERE ended_at IS NULL AND NOT ignored ORDER BY first_seen").fetchall()
        return [_row(JobRecord, r) for r in rows]

    def recent_jobs(self, limit: int = 15) -> list[JobRecord]:
        rows = self.db.execute("SELECT * FROM jobs WHERE NOT ignored ORDER BY first_seen DESC LIMIT ?", (limit,)).fetchall()
        return [_row(JobRecord, r) for r in rows]

    def session_jobs(self, session_id: str) -> list[JobRecord]:
        """Jobs the agent submitted in a conversation."""
        rows = self.db.execute("SELECT * FROM jobs WHERE session_id=? AND by_agent=1 AND NOT ignored ORDER BY first_seen",
                               (session_id,)).fetchall()
        return [_row(JobRecord, r) for r in rows]

    def project_jobs(self, project: str | None, limit: int = 15) -> list[JobRecord]:
        q = ("SELECT * FROM jobs WHERE NOT ignored AND " + ("project=?" if project else "project IS NULL")
             + " ORDER BY first_seen DESC LIMIT ?")
        rows = self.db.execute(q, (project, limit) if project else (limit,)).fetchall()
        return [_row(JobRecord, r) for r in rows]

    # ---------------------------------------------------------------- Telegram threads <-> conversations
    def thread_session(self, chat_id: int, thread_id: int | None) -> str | None:
        if not thread_id:
            return None
        row = self.db.execute("SELECT session_id FROM threads WHERE chat_id=? AND thread_id=?",
                              (chat_id, thread_id)).fetchone()
        return row["session_id"] if row else None

    def bind_thread(self, chat_id: int, thread_id: int, session_id: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO threads VALUES (?,?,?,?)", (chat_id, thread_id, session_id, time.time()))
        self.db.commit()

    def unbind_thread(self, chat_id: int, thread_id: int) -> None:
        self.db.execute("DELETE FROM threads WHERE chat_id=? AND thread_id=?", (chat_id, thread_id))
        self.db.commit()

    def session_thread(self, session_id: str, chat_id: int) -> int | None:
        """The Telegram thread a conversation lives in (the latest one, if it moved)."""
        row = self.db.execute("SELECT thread_id FROM threads WHERE session_id=? AND chat_id=? ORDER BY created DESC "
                              "LIMIT 1", (session_id, chat_id)).fetchone()
        return row["thread_id"] if row else None

    # ---------------------------------------------------------------- projects
    def get_project(self, name: str | None) -> ProjectRecord | None:
        if not name:
            return None
        row = self.db.execute("SELECT * FROM projects WHERE name=?", (name,)).fetchone()
        return _row(ProjectRecord, row) if row else None

    def upsert_project(self, p: ProjectRecord) -> None:
        self._upsert("projects", "name", p)

    def list_projects(self, include_archived: bool = False) -> list[ProjectRecord]:
        q = "SELECT * FROM projects" + ("" if include_archived else " WHERE archived=0") + " ORDER BY name"
        return [_row(ProjectRecord, r) for r in self.db.execute(q).fetchall()]

    def project_for_dir(self, path: str) -> ProjectRecord | None:
        """The (non-archived) project whose folder contains `path` - the deepest one if they nest."""
        path = (path or "").rstrip("/")
        best = None
        for p in self.list_projects():
            root = p.dir.rstrip("/")
            if root and (path == root or path.startswith(root + "/")) and (best is None or len(root) > len(best.dir)):
                best = p
        return best

    def project_for_thread(self, chat_id: int, thread_id: int | None) -> ProjectRecord | None:
        if not thread_id:
            return None
        for p in self.list_projects(include_archived=True):
            if p.thread_map.get(str(chat_id)) == int(thread_id):
                return p
        return None

    # ---------------------------------------------------------------- autopilot
    def get_autopilot(self, session_id: str) -> AutopilotRecord | None:
        row = self.db.execute("SELECT * FROM autopilot WHERE session_id=?", (session_id,)).fetchone()
        return _row(AutopilotRecord, row) if row else None

    def upsert_autopilot(self, a: AutopilotRecord) -> None:
        a.updated = time.time()
        self._upsert("autopilot", "session_id", a)

    def list_autopilots(self, active_only: bool = False) -> list[AutopilotRecord]:
        q = "SELECT * FROM autopilot" + (" WHERE active=1" if active_only else "") + " ORDER BY updated DESC"
        return [_row(AutopilotRecord, r) for r in self.db.execute(q).fetchall()]

    # ---------------------------------------------------------------- watches
    def add_watch(self, w: WatchRecord) -> WatchRecord:
        w.created = w.created or time.time()
        d = {k: v for k, v in w.__dict__.items() if k != "id"}
        cur = self.db.execute(f"INSERT INTO watches ({','.join(d)}) VALUES ({','.join('?' * len(d))})",
                              tuple(d.values()))
        self.db.commit()
        w.id = cur.lastrowid
        return w

    def update_watch(self, w: WatchRecord) -> None:
        self._upsert("watches", "id", w)

    def get_watch(self, wid: int) -> WatchRecord | None:
        row = self.db.execute("SELECT * FROM watches WHERE id=?", (wid,)).fetchone()
        return _row(WatchRecord, row) if row else None

    def watches(self, session_id: str | None = None, state: str | None = "active") -> list[WatchRecord]:
        q, args = "SELECT * FROM watches WHERE 1=1", []
        if session_id:
            q, args = q + " AND session_id=?", args + [session_id]
        if state:
            q, args = q + " AND state=?", args + [state]
        return [_row(WatchRecord, r) for r in self.db.execute(q + " ORDER BY id", args).fetchall()]

    def job_watches(self, job_id: str) -> list[WatchRecord]:
        return [w for w in self.watches() if w.kind == "jobs" and job_id in w.job_ids]

    # ---------------------------------------------------------------- token usage
    def add_usage(self, *, session_id: str, project: str | None, backend: str, model: str | None, origin: str,
                  input: int, cached: int, output: int, cost: float | None) -> None:
        now = time.time()
        self.db.execute("INSERT INTO usage VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (now, time.strftime("%Y-%m-%d", time.localtime(now)), session_id, project, backend, model or "",
                         origin, input, cached, output, cost))
        self.db.commit()

    def usage_rows(self, since: float, until: float | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM usage WHERE ts >= ?", [since]
        if until is not None:
            q, args = q + " AND ts < ?", args + [until]
        return self.db.execute(q, args).fetchall()

    # ---------------------------------------------------------------- agent sessions
    def get_session(self, sid: str) -> SessionRecord | None:
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return _row(SessionRecord, row) if row else None

    def upsert_session(self, s: SessionRecord) -> None:
        s.updated = time.time()
        self._upsert("sessions", "id", s)

    def list_sessions(self, limit: int = 20) -> list[SessionRecord]:
        rows = self.db.execute("SELECT * FROM sessions ORDER BY updated DESC LIMIT ?", (limit,)).fetchall()
        return [_row(SessionRecord, r) for r in rows]

    def session_for_job(self, job_id: str) -> SessionRecord | None:
        row = self.db.execute("SELECT * FROM sessions WHERE job_id=? ORDER BY updated DESC LIMIT 1", (job_id,)).fetchone()
        return _row(SessionRecord, row) if row else None

    # ---------------------------------------------------------------- message routing
    def link_message(self, chat_id: int, message_id: int, job_id: str | None, session_id: str | None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO messages VALUES (?,?,?,?)", (chat_id, message_id, job_id, session_id)
        )
        self.db.commit()

    def message_link(self, chat_id: int, message_id: int) -> tuple[str | None, str | None]:
        row = self.db.execute(
            "SELECT job_id, session_id FROM messages WHERE chat_id=? AND message_id=?", (chat_id, message_id)
        ).fetchone()
        return (row["job_id"], row["session_id"]) if row else (None, None)

    # ---------------------------------------------------------------- kv
    def get(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, value))
        self.db.commit()

    def unset(self, key: str) -> None:
        self.db.execute("DELETE FROM kv WHERE key=?", (key,))
        self.db.commit()
