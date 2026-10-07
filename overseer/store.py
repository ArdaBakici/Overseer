"""Local SQLite state (lives on the homelab, not on the HPC)."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    name TEXT, state TEXT, partition TEXT,
    workdir TEXT, stdout TEXT, stderr TEXT,
    batch_script TEXT,
    overseer INTEGER DEFAULT 0,
    session_id TEXT, parent_job TEXT, proposal_id TEXT,
    first_seen REAL, last_status_at REAL, ended_at REAL,
    missing_polls INTEGER DEFAULT 0,
    report TEXT,
    info TEXT
);
CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    session_id TEXT, kind TEXT,
    script_path TEXT, script_sha TEXT, workdir TEXT, run_dir TEXT,
    target_job TEXT, parent_job TEXT,
    summary TEXT, warnings TEXT, created REAL,
    status TEXT, job_id TEXT
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
    overseer: int = 0
    session_id: str | None = None
    parent_job: str | None = None
    proposal_id: str | None = None
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
class Proposal:
    id: str
    session_id: str
    kind: str  # "submit" | "cancel"
    script_path: str = ""
    script_sha: str = ""
    workdir: str = ""
    run_dir: str = ""
    target_job: str = ""
    parent_job: str = ""
    summary: str = ""
    warnings: str = ""
    created: float = 0.0
    status: str = "pending"  # pending | approved | rejected | submitted | failed
    job_id: str = ""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---------------------------------------------------------------- jobs
    def get_job(self, job_id: str) -> JobRecord | None:
        row = self.db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return JobRecord(**dict(row)) if row else None

    def upsert_job(self, rec: JobRecord) -> None:
        d = rec.__dict__
        cols = ",".join(d)
        qs = ",".join("?" * len(d))
        updates = ",".join(f"{k}=excluded.{k}" for k in d if k != "job_id")
        self.db.execute(
            f"INSERT INTO jobs ({cols}) VALUES ({qs}) ON CONFLICT(job_id) DO UPDATE SET {updates}",
            tuple(d.values()),
        )
        self.db.commit()

    def active_jobs(self) -> list[JobRecord]:
        rows = self.db.execute("SELECT * FROM jobs WHERE ended_at IS NULL ORDER BY first_seen").fetchall()
        return [JobRecord(**dict(r)) for r in rows]

    def recent_jobs(self, limit: int = 15) -> list[JobRecord]:
        rows = self.db.execute("SELECT * FROM jobs ORDER BY first_seen DESC LIMIT ?", (limit,)).fetchall()
        return [JobRecord(**dict(r)) for r in rows]

    def count_active_overseer_jobs(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM jobs WHERE overseer=1 AND ended_at IS NULL").fetchone()[0]

    # ---------------------------------------------------------------- proposals
    def add_proposal(self, p: Proposal) -> None:
        d = p.__dict__
        self.db.execute(f"INSERT INTO proposals ({','.join(d)}) VALUES ({','.join('?' * len(d))})", tuple(d.values()))
        self.db.commit()

    def get_proposal(self, pid: str) -> Proposal | None:
        row = self.db.execute("SELECT * FROM proposals WHERE id=?", (pid,)).fetchone()
        return Proposal(**dict(row)) if row else None

    def update_proposal(self, pid: str, **fields: Any) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE proposals SET {sets} WHERE id=?", (*fields.values(), pid))
        self.db.commit()

    def claim_proposal(self, pid: str, new_status: str) -> bool:
        """Atomically move a pending proposal to ``new_status``. False if it was not pending."""
        cur = self.db.execute("UPDATE proposals SET status=? WHERE id=? AND status='pending'", (new_status, pid))
        self.db.commit()
        return cur.rowcount == 1

    def pending_proposals(self) -> list[Proposal]:
        rows = self.db.execute("SELECT * FROM proposals WHERE status='pending' ORDER BY created").fetchall()
        return [Proposal(**dict(r)) for r in rows]

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


def now() -> float:
    return time.time()
