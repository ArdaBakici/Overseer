"""Search across past work: conversations, and what the domain keeps (for the HPC: results, findings and jobs).

"What did we try with ratio 64 last month?" - for agents (`ov history search`, then `ov history show` to read a
past conversation) and for you (/search). A full-text index (SQLite FTS5, in state/search.db, rebuilt from the
sources whenever it's missing):
- conversations: from their transcripts (state/transcripts/<id>.jsonl), read incrementally - each request, reply,
  in-between note and command the agent ran (not command output or reasoning);
- the domain's records (Domain.search_rows; for the HPC: results, findings and jobs): re-read at each search, since
  they change.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass

from .config import Config
from .store import Store

MAX_BODY = 6000  # characters of one event indexed (a wake-up's prompt carries whole job reports)
SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
    body, title, kind UNINDEXED, ref UNINDEXED, session UNINDEXED, project UNINDEXED, ts UNINDEXED, what UNINDEXED,
    tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, offset INTEGER);
"""


class SearchError(ValueError):
    pass


@dataclass
class Hit:
    kind: str  # conversation | result | finding | job
    ref: str  # conversation id, result id, finding id, job id
    session: str
    project: str
    ts: float
    what: str  # conversation: request | reply | note | command
    title: str
    snippet: str


def fts_query(text: str) -> str:
    """Free text -> an FTS5 query: every word must appear (the last one as a prefix); "quoted phrases" stay
    phrases; OR between words is kept."""
    parts = re.findall(r'"([^"]+)"|(\S+)', text)
    terms: list[str] = []
    for phrase, word in parts:
        if word and word.upper() == "OR" and terms:
            terms.append("OR")
            continue
        token = " ".join(re.findall(r"\w+", phrase or word, re.UNICODE))
        if token:
            terms.append(f'"{token}"')
    while terms and terms[-1] == "OR":
        terms.pop()
    if not terms:
        raise SearchError("search for what? e.g. ov history search ratio 64")
    if not (parts[-1][0]) and terms[-1] != "OR":  # the last bare word may be the start of one
        terms[-1] += "*"
    return " ".join(terms)


def parse_since(text: str | None) -> float:
    """'30d', '2w', '12h' -> a timestamp; None/'' -> 0."""
    if not text:
        return 0.0
    m = re.fullmatch(r"(\d+)\s*([hdwm])", text.strip().lower())
    if not m:
        raise SearchError("--since: like 12h, 7d, 2w or 3m (months)")
    n, unit = int(m.group(1)), m.group(2)
    return time.time() - n * {"h": 3600, "d": 86400, "w": 7 * 86400, "m": 30 * 86400}[unit]


class Search:
    """The index. Reading and searching it happens in a worker thread (a first indexing of long transcripts takes
    a while); Overseer's records are read in the event loop first (the store isn't shared across threads)."""

    def __init__(self, cfg: Config, store: Store, domain=None):
        self.cfg = cfg
        self.store = store
        self.domain = domain  # its records (search_rows) and how a hit of theirs reads (search_line)
        self.path = cfg.local.state("search.db")
        self._lock = threading.Lock()
        self.db = self._open()

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            db = sqlite3.connect(self.path, check_same_thread=False)
            db.executescript(SCHEMA)
        except sqlite3.DatabaseError:  # damaged: it's only an index, made again from the sources
            self.path.unlink(missing_ok=True)
            db = sqlite3.connect(self.path, check_same_thread=False)
            db.executescript(SCHEMA)
        os.chmod(self.path, 0o600)
        return db

    async def asearch(self, text: str, **kw) -> tuple[list["Hit"], int]:
        rows, projects = self._record_rows(), self._projects()
        return await asyncio.to_thread(self.search, text, rows=rows, projects=projects, **kw)

    async def aconversation(self, sid: str, last: int = 5, grep: str = "") -> list[dict]:
        return await asyncio.to_thread(self.conversation, sid, last, grep)

    # ------------------------------------------------------------------ indexing
    def _projects(self) -> dict[str, str]:
        return {s.id: s.project or "" for s in self.store.list_sessions(100_000)}

    def update(self, rows: list[tuple] | None = None, projects: dict[str, str] | None = None) -> None:
        with self._lock:
            self._transcripts(self._projects() if projects is None else projects)
            self.db.execute("DELETE FROM docs WHERE kind != 'conversation'")
            self.db.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?)",
                                self._record_rows() if rows is None else rows)
            self.db.commit()

    def _transcripts(self, projects: dict[str, str]) -> None:
        d = self.cfg.local.state("transcripts")
        if not d.is_dir():
            return
        done = dict(self.db.execute("SELECT path, offset FROM files").fetchall())
        for path in d.glob("*.jsonl"):
            sid = path.stem
            offset = done.get(path.name, 0)
            try:
                size = path.stat().st_size
                if size < offset:  # (replaced): from the start
                    self.db.execute("DELETE FROM docs WHERE kind='conversation' AND session=?", (sid,))
                    offset = 0
                if size == offset:
                    continue
                with open(path, "rb") as fh:
                    fh.seek(offset)
                    data = fh.read()
            except OSError:
                continue
            end = data.rfind(b"\n") + 1  # a line being written is read next time
            project = projects.get(sid, "")
            for line in data[:end].decode(errors="replace").splitlines():
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                doc = self._event(ev)
                if doc:
                    what, body = doc
                    self.db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?)",
                                    (body[:MAX_BODY], f"conversation {sid}", "conversation", sid, sid,
                                     ev.get("project") or project, float(ev.get("ts") or 0), what))
            self.db.execute("INSERT OR REPLACE INTO files VALUES (?, ?)", (path.name, offset + end))

    @staticmethod
    def _event(ev: dict) -> tuple[str, str] | None:
        kind = ev.get("kind")
        if kind == "turn" and ev.get("text"):
            return "request", ev["text"]
        if kind == "end" and ev.get("status") == "done" and ev.get("text"):
            return "reply", ev["text"]
        if kind == "message" and ev.get("text"):
            return "note", ev["text"]
        if kind == "command" and ev.get("command"):
            return "command", ev["command"]
        return None

    def _record_rows(self) -> list[tuple]:
        """The domain's records (they change, so they're indexed again at each search; they're small): rows of
        (body, title, kind, ref, session, project, ts, what)."""
        return self.domain.search_rows() if self.domain else []

    # ------------------------------------------------------------------ searching
    def search(self, text: str, *, project: str | None | bool = False, kinds: tuple[str, ...] = (),
               since: float = 0.0, limit: int = 12, per_conversation: int = 2, rows: list[tuple] | None = None,
               projects: dict[str, str] | None = None) -> tuple[list[Hit], int]:
        """The best matches (and how many there are). `project`: False = everywhere, None = the main chat.
        (In a worker thread: through asearch, which reads Overseer's records first.)"""
        query = fts_query(text)
        self.update(rows, projects)
        q = ("SELECT kind, ref, session, project, ts, what, title, "
             "snippet(docs, 0, '«', '»', '…', 18), bm25(docs, 1.0, 0.5) FROM docs WHERE docs MATCH ?")
        args: list = [query]
        if project is not False:
            q, args = q + " AND project=?", args + [project or ""]
        if kinds:
            q, args = q + f" AND kind IN ({','.join('?' * len(kinds))})", args + list(kinds)
        if since:
            q, args = q + " AND ts >= ?", args + [since]
        try:
            with self._lock:
                rows = self.db.execute(q + " ORDER BY bm25(docs, 1.0, 0.5) LIMIT 500", args).fetchall()
        except sqlite3.OperationalError as e:
            raise SearchError(f"can't search for that: {e}") from e
        # The best first; a little newer wins a near tie. Few hits per conversation, so one long chat doesn't
        # fill the list.
        now = time.time()
        rows.sort(key=lambda r: r[8] * (1 + 0.5 / (1 + (now - float(r[4] or 0)) / (30 * 86400))))
        hits, seen = [], {}
        for kind, ref, session, proj, ts, what, title, snippet, _ in rows:
            if kind == "conversation":
                seen[ref] = seen.get(ref, 0) + 1
                if seen[ref] > per_conversation:
                    continue
            hits.append(Hit(kind, ref, session, proj, float(ts or 0), what, title, " ".join(snippet.split())))
            if len(hits) >= limit:
                break
        return hits, len(rows)

    # ------------------------------------------------------------------ a past conversation
    def conversation(self, sid: str, last: int = 5, grep: str = "") -> list[dict]:
        """Its turns, oldest first: {ts, request, reply, commands, notes}; the last `last`, or those mentioning
        `grep` (and the ones right after them)."""
        path = self.cfg.local.state("transcripts") / f"{sid}.jsonl"
        if not re.fullmatch(r"[\w.-]+", sid) or not path.is_file():
            raise SearchError(f"no conversation {sid!r} (the ids are in ov history search's results)")
        turns: list[dict] = []
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                kind = ev.get("kind")
                if kind == "turn":
                    turns.append({"ts": ev.get("ts", 0), "request": ev.get("text", ""), "reply": "", "commands": [],
                                  "notes": [], "status": ""})
                elif turns and kind == "end":
                    turns[-1]["status"] = ev.get("status", "")
                    turns[-1]["reply"] = ev.get("text", "") or ""
                elif turns and kind == "command":
                    turns[-1]["commands"].append(ev.get("command", ""))
                elif turns and kind == "message":
                    turns[-1]["notes"].append(ev.get("text", ""))
        if grep:
            words = [w.lower() for w in re.findall(r"\w+", grep)]
            keep = set()
            for i, t in enumerate(turns):
                blob = " ".join([t["request"], t["reply"], *t["commands"], *t["notes"]]).lower()
                if all(w in blob for w in words):
                    keep.update({i, i + 1})
            return [t for i, t in enumerate(turns) if i in keep][-max(last, 1) * 2:]
        return turns[-max(last, 1):]


    def day(self, start: float, end: float) -> list[tuple[str, str, list[dict]]]:
        """Every conversation's turns between start and end: [(conversation, project, turns)], the earliest first."""
        folder = self.cfg.local.state("transcripts")
        projects = self._projects()
        out = []
        for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
            try:
                if path.stat().st_mtime < start:
                    continue  # (nothing written to it since)
            except OSError:
                continue
            turns = [t for t in self.conversation(path.stem, last=100_000) if start <= t["ts"] < end]
            if turns:
                out.append((path.stem, projects.get(path.stem, ""), turns))
        return sorted(out, key=lambda c: c[2][0]["ts"])

    async def aday(self, start: float, end: float) -> list[tuple[str, str, list[dict]]]:
        return await asyncio.to_thread(self.day, start, end)


def render_day(convs: list[tuple[str, str, list[dict]]], limit: int = 40_000) -> str:
    """A day's conversations for the agent (e.g. for a nightly reflection): requests and replies, condensed."""
    if not convs:
        return "no conversations that day"
    parts = []
    for sid, project, turns in convs:
        parts.append(f"=== conversation {sid}" + (f" (project {project})" if project else ""))
        for t in turns:
            req = " ".join(t["request"].split())
            rep = " ".join((t["reply"] or f"({t['status'] or 'no reply'})").split())
            parts.append(f"[{time.strftime('%H:%M', time.localtime(t['ts']))}] user/trigger: "
                         f"{req[:700]}{'…' if len(req) > 700 else ''}\n  reply: {rep[:900]}{'…' if len(rep) > 900 else ''}")
    text = "\n".join(parts)
    if len(text) > limit:
        text = text[:limit] + f"\n[... cut: {len(text) - limit} more characters; read a conversation with ov history show]"
    return text


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "?"


def render_hits(hits: list[Hit], total: int, text: str, html: bool = False, domain=None) -> str:
    from .notify import code, esc
    e = esc if html else (lambda x: x)
    c = code if html else (lambda x: x)
    if not hits:
        return f"nothing found for {e(text)!s}" if not html else f"🔎 Nothing found for <b>{esc(text)}</b>."
    head = (f"🔎 <b>{esc(text)}</b>: {total} match(es)" if html else f"{total} match(es) for {text!r}") + (
        f", the best {len(hits)}:" if total > len(hits) else ":")
    lines = [head]
    for h in hits:
        where = f" · 📁 {e(h.project)}" if h.project else ""
        snippet = e(h.snippet).replace("«", "<b>").replace("»", "</b>") if html else h.snippet
        if h.kind == "conversation":
            lines.append(f"💬 {c(h.ref)}{where} · {e(_when(h.ts))} · {e(h.what)}: {snippet}")
            if not html:
                lines.append(f"   → ov history show {h.ref} --grep '{text[:40]}'")
        else:  # the domain's records
            line = domain.search_line(h, where, snippet, html) if domain else None
            lines.append(line or f"• {e(h.kind)} {c(h.ref)}{where} · {e(_when(h.ts)[:10])}: {snippet}")
    return "\n".join(lines)


def render_turns(sid: str, turns: list[dict], limit: int = 2500) -> str:
    if not turns:
        return f"conversation {sid}: no turns (with that text)"
    out = [f"conversation {sid}, {len(turns)} turn(s):"]
    for t in turns:
        req = " ".join(t["request"].split())
        out.append(f"\n── {_when(t['ts'])} " + ("(" + t["status"] + ")" if t["status"] not in ("", "done") else ""))
        out.append(f"👤 {req[:800]}" + ("…" if len(req) > 800 else ""))
        if t["commands"]:
            shown = t["commands"][-8:]
            out.append(f"$ ({len(t['commands'])} commands" + (", the last 8" if len(t["commands"]) > 8 else "") + ")")
            out += [f"  $ {' '.join(x.split())[:200]}" for x in shown]
        if t["reply"]:
            reply = t["reply"].strip()
            out.append("🤖 " + (reply[:limit] + "…" if len(reply) > limit else reply))
    return "\n".join(out)

