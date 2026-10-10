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
    info TEXT
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
"""


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
        rows = self.db.execute("SELECT * FROM jobs WHERE ended_at IS NULL ORDER BY first_seen").fetchall()
        return [_row(JobRecord, r) for r in rows]

    def recent_jobs(self, limit: int = 15) -> list[JobRecord]:
        rows = self.db.execute("SELECT * FROM jobs ORDER BY first_seen DESC LIMIT ?", (limit,)).fetchall()
        return [_row(JobRecord, r) for r in rows]

    # ---------------------------------------------------------------- agent sessions
    def get_session(self, sid: str) -> SessionRecord | None:
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return _row(SessionRecord, row) if row else None

    def upsert_session(self, s: SessionRecord) -> None:
        s.updated = time.time()
        self._upsert("sessions", "id", s)

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
