"""Tasks in the vault, in the Obsidian Tasks plugin's format.

A task is a checklist line anywhere in the vault: `- [ ] Submit the review 📅 2026-10-20 ⏫ #work`. Its dates:
📅 due, ⏳ scheduled, 🛫 start, ✅ done, ❌ cancelled, ➕ created; priority 🔺 ⏫ 🔼 🔽 ⏬; 🔁 a recurrence
("every week", "every 2 days when done"). Status: [ ] to do, [/] in progress, [x] done, [-] cancelled.

Completing one marks it [x] with ✅ today; a recurring task gets its next occurrence on the line above, like the
plugin does (for the recurrences it understands; others are left to Obsidian). Overseer only rewrites the task's own
line, after checking it is still what was read, and never follows links out of the vault.
"""

from __future__ import annotations

import calendar
import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

LINE_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)\[(.)\]\s+(.*)$")
DATE_FIELDS = {"📅": "due", "⏳": "scheduled", "🛫": "start", "✅": "done", "❌": "cancelled", "➕": "created"}
PRIORITIES = {"🔺": "highest", "⏫": "high", "🔼": "medium", "🔽": "low", "⏬": "lowest"}
PRIORITY_ORDER = {"highest": 0, "high": 1, "medium": 2, "": 3, "low": 4, "lowest": 5}
FIELD_RE = re.compile(r"(📅|⏳|🛫|✅|❌|➕)️?\s*(\d{4}-\d{2}-\d{2})")
RECUR_RE = re.compile(r"🔁️?\s*([^📅⏳🛫✅❌➕🔺⏫🔼🔽⏬#]+)")
TAG_RE = re.compile(r"(?<![\w/])#[\w/-]+")
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class TaskError(ValueError):
    pass


@dataclass
class Task:
    path: str  # relative to the vault
    line: int  # 0-based
    raw: str
    status: str  # " ", "/", "x", "-"
    text: str  # without the fields
    due: str = ""
    scheduled: str = ""
    start: str = ""
    done: str = ""
    cancelled: str = ""
    created: str = ""
    priority: str = ""
    recurrence: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.path}\n{self.raw.strip()}".encode()).hexdigest()[:6]

    @property
    def open(self) -> bool:
        return self.status in (" ", "/")

    @property
    def when(self) -> str:
        """The date that puts it on a day: due, else scheduled."""
        return self.due or self.scheduled

    def to_dict(self) -> dict:
        return {"id": self.id, "path": self.path, "line": self.line + 1, "status": self.status, "text": self.text,
                "due": self.due, "scheduled": self.scheduled, "start": self.start, "done": self.done,
                "priority": self.priority, "recurrence": self.recurrence, "tags": self.tags}


def parse_line(line: str, path: str = "", n: int = 0) -> Task | None:
    m = LINE_RE.match(line)
    if not m:
        return None
    body = m.group(3)
    t = Task(path=path, line=n, raw=line.rstrip("\n"), status=m.group(2).lower() if m.group(2) in "xX" else m.group(2),
             text="")
    for emoji, value in FIELD_RE.findall(body):
        setattr(t, DATE_FIELDS[emoji], value)
    r = RECUR_RE.search(body)
    if r:
        t.recurrence = r.group(1).strip()
    for emoji, name in PRIORITIES.items():
        if emoji in body:
            t.priority = name
            break
    t.tags = TAG_RE.findall(body)
    text = RECUR_RE.sub("", FIELD_RE.sub("", body))
    for emoji in PRIORITIES:
        text = text.replace(emoji, "")
    t.text = " ".join(text.split())
    return t


def scan(vault) -> list[Task]:
    """Every task in the vault's notes."""
    out = []
    for p in vault.notes():
        try:
            if p.is_symlink():
                continue
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        rel = vault.rel(p)
        for n, line in enumerate(lines):
            if "[" in line and (t := parse_line(line, rel, n)):
                out.append(t)
    return out


def select(tasks: list[Task], when: str = "open", tag: str = "", search: str = "", today: date | None = None
           ) -> list[Task]:
    """when: open | today (due or scheduled today, or overdue) | overdue | week (the next 7 days, and overdue) |
    upcoming (dated, after today) | undated | done | all."""
    today = today or date.today()
    t0, t7 = today.isoformat(), (today + timedelta(days=7)).isoformat()
    keep = {
        "open": lambda t: t.open,
        "today": lambda t: t.open and t.when and t.when <= t0,
        "overdue": lambda t: t.open and t.due and t.due < t0,
        "week": lambda t: t.open and t.when and t.when <= t7,
        "upcoming": lambda t: t.open and t.when and t.when > t0,
        "undated": lambda t: t.open and not t.when,
        "done": lambda t: t.status == "x",
        "all": lambda t: True,
    }.get(when)
    if keep is None:
        raise TaskError("when: open, today, overdue, week, upcoming, undated, done or all")
    out = [t for t in tasks if keep(t)]
    if tag:
        tag = tag if tag.startswith("#") else "#" + tag
        out = [t for t in out if any(x == tag or x.startswith(tag + "/") for x in t.tags)]
    if search:
        words = search.lower().split()
        out = [t for t in out if all(w in (t.text + " " + t.path).lower() for w in words)]
    return sorted(out, key=lambda t: (t.when or "9999", PRIORITY_ORDER.get(t.priority, 3), t.path, t.line))


def line_for(text: str, due: str = "", scheduled: str = "", priority: str = "", recurrence: str = "") -> str:
    text = " ".join(text.split())
    if not text:
        raise TaskError("what is the task?")
    for d in (due, scheduled):
        if d and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            raise TaskError(f"dates are YYYY-MM-DD, not {d!r}")
    prio = {v: k for k, v in PRIORITIES.items()}
    if priority and priority not in prio:
        raise TaskError("priority: highest, high, medium, low or lowest")
    parts = [f"- [ ] {text}"]
    if recurrence:
        parts.append(f"🔁 {recurrence.strip()}")
    if priority:
        parts.append(prio[priority])
    if scheduled:
        parts.append(f"⏳ {scheduled}")
    if due:
        parts.append(f"📅 {due}")
    return " ".join(parts)


# ---------------------------------------------------------------------------- writing (one line, safely)
def _safe_path(vault, rel: str) -> Path:
    p = (vault.root / rel)
    if ".." in Path(rel).parts or Path(rel).is_absolute() or not rel.endswith(".md"):
        raise TaskError(f"{rel!r}: a note's path inside the vault (…/name.md)")
    cur = vault.root
    for part in Path(rel).parts:  # no links on the way (the agent can write the vault)
        cur = cur / part
        if cur.is_symlink():
            raise TaskError(f"{rel!r} goes through a link: not touching it")
    if not p.resolve().is_relative_to(vault.root):
        raise TaskError(f"{rel!r} is outside the vault")
    return p


def _write(p: Path, text: str) -> None:
    tmp = p.with_name(f".{p.name}.overseer-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, p)


def add(vault, line: str, rel: str | None = None) -> str:
    """Append a task line to a note (default: the tasks file); returns the note's path."""
    rel = rel or vault.rel(vault.tasks_file)
    p = _safe_path(vault, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = p.read_text() if p.exists() else ""
    _write(p, old + ("" if not old or old.endswith("\n") else "\n") + line + "\n")
    return rel


def find(vault, tid: str) -> Task:
    matches = [t for t in scan(vault) if t.id == tid.lower().lstrip("#")]
    if not matches:
        raise TaskError(f"no task {tid!r} (ids change when a task's line changes: list them again)")
    return matches[0]


def complete(vault, tid: str, today: date | None = None) -> tuple[Task, str]:
    """Mark a task done (✅ today); a recurring one gets its next occurrence above it. (the task, a remark)"""
    today = today or date.today()
    t = find(vault, tid)
    if not t.open:
        raise TaskError(f"task {t.id} is already {'done' if t.status == 'x' else 'cancelled'}")
    p = _safe_path(vault, t.path)
    lines = p.read_text().split("\n")
    if t.line >= len(lines) or lines[t.line].rstrip() != t.raw.rstrip():
        raise TaskError("the note changed meanwhile: list the tasks again")
    m = LINE_RE.match(t.raw)
    done = f"{m.group(1)}[x] {m.group(3).rstrip()} ✅ {today.isoformat()}"
    new = [done]
    remark = ""
    if t.recurrence:
        nxt = next_occurrence(t, today)
        if nxt is None:
            raise TaskError(f"its recurrence ({t.recurrence!r}) isn't one Overseer understands: complete it in "
                            "Obsidian, which creates the next one")
        new = [nxt, done]
        remark = "next: " + (parse_line(nxt).when or "(undated)")
    lines[t.line:t.line + 1] = new
    _write(p, "\n".join(lines))
    return t, remark


# ---------------------------------------------------------------------------- recurrence
def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def step(rule: str, d: date) -> date | None:
    """The next date after d for a rule like "every 2 weeks", "every weekday", "every monday, thursday"."""
    r = rule.lower().replace("when done", "").strip()
    m = re.fullmatch(r"every\s+(?:(\d+)\s+)?(day|week|month|year)s?(?:\s+on\s+.*)?", r)
    if m:
        n = int(m.group(1) or 1)
        unit = m.group(2)
        if unit == "day":
            return d + timedelta(days=n)
        if unit == "week":
            return d + timedelta(weeks=n)
        if unit == "month":
            return _add_months(d, n)
        return _add_months(d, 12 * n)
    if r == "every weekday":
        nd = d + timedelta(days=1)
        while nd.weekday() >= 5:
            nd += timedelta(days=1)
        return nd
    m = re.fullmatch(r"every\s+((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s*(?:,|and)\s*"
                     r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))*)", r)
    if m:
        days = {WEEKDAYS.index(x) for x in re.findall(r"[a-z]+day", m.group(1))}
        nd = d + timedelta(days=1)
        while nd.weekday() not in days:
            nd += timedelta(days=1)
        return nd
    return None


def next_occurrence(t: Task, today: date) -> str | None:
    """The next instance's line: every date moved by the same step as its base date (due, else scheduled, else
    start; "when done": from today)."""
    base_name = next((f for f in ("due", "scheduled", "start") if getattr(t, f)), "")
    when_done = "when done" in t.recurrence.lower()
    if not base_name:
        return None
    base = date.fromisoformat(getattr(t, base_name))
    origin = today if when_done else base
    nxt = step(t.recurrence, origin)
    if nxt is None:
        return None
    shift = nxt - base
    line = t.raw
    for emoji, name in DATE_FIELDS.items():
        if name in ("due", "scheduled", "start") and getattr(t, name):
            moved = (date.fromisoformat(getattr(t, name)) + shift).isoformat()
            line = re.sub(rf"{emoji}️?\s*{getattr(t, name)}", f"{emoji} {moved}", line)
    line = re.sub(r"\s*(✅|❌)️?\s*\d{4}-\d{2}-\d{2}", "", line)
    return LINE_RE.sub(lambda m: f"{m.group(1)}[ ] {m.group(3)}", line)


def describe(t: Task, today: date | None = None) -> str:
    """One line for a list: id, text, when, where."""
    today = today or date.today()
    when = ""
    if t.when:
        d = date.fromisoformat(t.when)
        delta = (d - today).days
        rel = ("today" if delta == 0 else "tomorrow" if delta == 1 else f"{-delta}d overdue" if delta < 0
               else d.strftime("%a %d %b"))
        when = f" · {'📅' if t.due else '⏳'} {rel}"
    prio = {"highest": " 🔺", "high": " ⏫", "low": " 🔽", "lowest": " ⏬"}.get(t.priority, "")
    status = "🔄 " if t.status == "/" else "✅ " if t.status == "x" else ""
    return f"{t.id} {status}{t.text}{prio}{when} · {t.path}" + (f" · 🔁 {t.recurrence}" if t.recurrence else "")


def _epoch_today() -> date:
    return date.fromtimestamp(time.time())
