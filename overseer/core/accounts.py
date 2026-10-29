"""Several logins per agent CLI (agent.claude_code.accounts, agent.codex.accounts), for the whole bot.

One account of each CLI is in use at a time (the first one at first; /account switches). When a run hits the
account's usage limit, the account is marked out until its reset and the bot moves on: to the CLI's next account
that isn't out, then to the other CLI (every conversation, until the researcher switches back), and the request runs
again there. A conversation keeps its history either way: a CLI's session is copied into the account it continues
in (Claude Code and Codex keep sessions in their home folder), and a conversation moving to the other CLI starts with
a summary of what was said and done (from Overseer's transcripts: `handover`).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

from .config import BACKENDS, Config, accounts_of
from .store import Store

log = logging.getLogger(__name__)

ACTIVE_KEY = "account:"  # + backend: the account in use
OUT_KEY = "account_out:"  # + backend:name -> until when it is out of usage (epoch)
FORCED_KEY = "backend_forced"  # the CLI every conversation runs on after a failover, until /backend or /account
NAMES = {"claude_code": "Claude Code", "codex": "Codex"}
DEFAULT_OUT = 3600  # out for this long when the CLI doesn't say when its limit resets
MAX_OUT = 8 * 86400


# ============================================================================ the accounts
class Accounts:
    def __init__(self, cfg: Config, store: Store):
        self.cfg, self.store = cfg, store
        self.homes = {b: accounts_of(getattr(cfg.agent, b)) for b in BACKENDS}
        # (sessions from before there were accounts: in the CLI's own home, looked for too)
        self.legacy = {b: getattr(cfg.agent, b).home for b in BACKENDS}
        for b in BACKENDS:
            self.apply(b)

    def names(self, backend: str) -> list[str]:
        return list(self.homes[backend])

    def several(self, backend: str) -> bool:
        return len(self.homes[backend]) > 1

    def active(self, backend: str) -> str:
        v = self.store.get(ACTIVE_KEY + backend)
        return v if v in self.homes[backend] else self.names(backend)[0]

    def home(self, backend: str, name: str | None = None) -> Path:
        return self.cfg.local.resolve(self.homes[backend][name or self.active(backend)])

    def apply(self, backend: str) -> None:
        """The CLI's runs use the active account's home from now on (runners read it at each run)."""
        getattr(self.cfg.agent, backend).home = self.homes[backend][self.active(backend)]

    def switch(self, backend: str, name: str) -> None:
        if name not in self.homes[backend]:
            raise ValueError(f"{NAMES[backend]} has no account {name!r} (it has: {', '.join(self.names(backend))})")
        self.store.set(ACTIVE_KEY + backend, name)
        self.apply(backend)

    def find(self, text: str) -> list[tuple[str, str]]:
        """The accounts a name (or "<cli> <name>", "<cli>:<name>") means: [(backend, name)]."""
        words = [w for w in re.split(r"[\s:/]+", text.strip()) if w]
        if len(words) == 2:
            cli = {"claude": "claude_code", "claude_code": "claude_code", "codex": "codex"}.get(words[0].lower())
            return [(cli, words[1])] if cli and words[1] in self.homes[cli] else []
        if len(words) == 1 and words[0].lower() in ("claude", "claude_code", "codex") \
                and not any(words[0] in self.homes[b] for b in BACKENDS):
            b = "codex" if words[0].lower() == "codex" else "claude_code"
            return [(b, self.active(b))]
        return [(b, words[0]) for b in BACKENDS if len(words) == 1 and words[0] in self.homes[b]]

    # ---- out of usage
    def out_until(self, backend: str, name: str) -> float:
        try:
            until = float(self.store.get(f"{OUT_KEY}{backend}:{name}") or 0)
        except ValueError:
            return 0
        return until if until > time.time() else 0

    def mark_out(self, backend: str, name: str, until: float | None) -> float:
        now = time.time()
        until = min(max(until or now + DEFAULT_OUT, now + 60), now + MAX_OUT)
        self.store.set(f"{OUT_KEY}{backend}:{name}", str(int(until)))
        return until

    def clear_out(self, backend: str, name: str) -> None:
        self.store.unset(f"{OUT_KEY}{backend}:{name}")

    def next_after_limit(self, backend: str, name: str) -> tuple[str, str] | None:
        """Where to go when (backend, name) is out: the CLI's next account that isn't out, then the other CLI's
        account in use (or its next one that isn't out)."""
        own = self.names(backend)
        i = own.index(name) if name in own else -1
        for n in own[i + 1:] + own[:max(i, 0)]:
            if n != name and not self.out_until(backend, n):
                return backend, n
        for b in BACKENDS:
            if b == backend:
                continue
            for n in [self.active(b)] + [x for x in self.names(b) if x != self.active(b)]:
                if not self.out_until(b, n):
                    return b, n
        return None

    def earliest_back(self) -> float:
        times = [self.out_until(b, n) for b in BACKENDS for n in self.names(b)]
        return min((t for t in times if t), default=0)

    # ---- the CLI every conversation runs on after a failover to the other CLI
    @property
    def forced(self) -> str | None:
        v = self.store.get(FORCED_KEY)
        return v if v in BACKENDS else None

    def force(self, backend: str | None) -> None:
        if backend:
            self.store.set(FORCED_KEY, backend)
        else:
            self.store.unset(FORCED_KEY)

    # ---- every home (sessions, skills, /reset)
    def all_homes(self, backend: str) -> list[Path]:
        out = [self.cfg.local.resolve(h) for h in self.homes[backend].values()]
        legacy = self.cfg.local.resolve(self.legacy[backend])
        return out + ([legacy] if legacy not in out else [])

    def label(self, backend: str, name: str | None = None) -> str:
        """e.g. "Claude Code" or, with several accounts, "Claude Code (work)"."""
        name = name or self.active(backend)
        return f"{NAMES[backend]} ({name})" if self.several(backend) else NAMES[backend]

    def describe(self) -> str:
        """The /account overview (HTML)."""
        from .notify import esc

        lines = []
        for b in BACKENDS:
            items = []
            for n in self.names(b):
                out = self.out_until(b, n)
                mark = "▶ " if n == self.active(b) else ""
                items.append(f"{mark}<b>{esc(n)}</b>" if mark else esc(n))
                if out:
                    items[-1] += f" (out of usage until {esc(when(out))})"
            lines.append(f"{NAMES[b]}: " + " · ".join(items))
        if self.forced:
            lines.append(f"\nEvery conversation runs on <b>{NAMES[self.forced]}</b> for now (it took over when the "
                         "other ran out). /backend or /account switches back.")
        return "\n".join(lines)


def all_homes(cfg: Config, backend: str) -> list[Path]:
    """Every home a CLI's sessions and skills may be in (its accounts' and its own)."""
    out = [cfg.local.resolve(h) for h in accounts_of(getattr(cfg.agent, backend)).values()]
    own = cfg.local.resolve(getattr(cfg.agent, backend).home)
    return out + ([own] if own not in out else [])


def when(ts: float) -> str:
    t = datetime.fromtimestamp(ts)
    now = datetime.now()
    if t.date() == now.date():
        return t.strftime("%H:%M")
    if t.date() == (now + timedelta(days=1)).date():
        return t.strftime("tomorrow %H:%M")
    return t.strftime("%a %d %b %H:%M")


# ============================================================================ usage limits
LIMIT_RE = re.compile(
    r"usage limit|limit reached|hit your (?:\w+ )?limit|out of (?:extra )?usage|usage_limit_reached|"
    r"insufficient_quota|exceeded your current quota|credit balance is too low|quota exceeded", re.I)
# A run that "answered" with only this is the CLI reporting the limit (Claude Code did so as its result).
LIMIT_ANSWER_RE = re.compile(r"\s*(?:Claude AI usage limit reached|You[’']ve hit your (?:\w+ )?limit|"
                             r"[\w -]*limit reached\s*[∙·|])", re.I)
_UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def usage_limit(text: str) -> tuple[bool, float | None]:
    """Is this error a usage limit (out of the plan's usage or the API's credit)? (yes, when it resets if said)."""
    if not text or not LIMIT_RE.search(text):
        return False, None
    return True, reset_time(text)


def reset_time(text: str, now: float | None = None) -> float | None:
    now = time.time() if now is None else now
    m = re.search(r"\|\s*(\d{10})\b", text)  # Claude Code: "Claude AI usage limit reached|1761300000"
    if m:
        return float(m.group(1))
    m = re.search(r"resets?_(?:in_seconds|after_seconds)\W+(\d+)", text)  # Codex's error details
    if m:
        return now + int(m.group(1))
    m = re.search(r"resets?_at\W+(\d{10})\b", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:try again in|resets? in|available in)\s+((?:\d+\s*[a-z]+[\s,]*(?:and\s+)?)+)", text, re.I)
    if m:
        total = sum(int(n) * _UNITS[u[0].lower()] for n, u in re.findall(r"(\d+)\s*([a-z]+)", m.group(1), re.I)
                    if u[0].lower() in _UNITS)
        if total:
            return now + total
    m = re.search(r"(?:resets?|try again)(?: at| on)?\s+(?:(?P<mon>[A-Z][a-z]{2})[a-z]*\s+(?P<day>\d{1,2}),?\s+"
                  r"(?:at\s+)?)?(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>[ap]\.?m\.?)?"
                  r"(?:\s*\((?P<tz>[\w/+-]+)\))?", text, re.I)
    if m and (m.group("ap") or m.group("m")):
        return _clock(m, now)
    return None


def _clock(m: re.Match, now: float) -> float | None:
    """"3pm", "3:30pm (Asia/Dubai)", "Oct 28, 9am": the next such time."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    tz = None
    if m.group("tz"):
        try:
            tz = ZoneInfo(m.group("tz"))
        except (ZoneInfoNotFoundError, ValueError):
            tz = None
    base = datetime.fromtimestamp(now, tz) if tz else datetime.fromtimestamp(now)
    h, minute = int(m.group("h")), int(m.group("m") or 0)
    ap = (m.group("ap") or "").lower()[:1]
    if ap == "p" and h < 12:
        h += 12
    elif ap == "a" and h == 12:
        h = 0
    if h > 23 or minute > 59:
        return None
    t = base.replace(hour=h, minute=minute, second=0, microsecond=0)
    if m.group("mon"):
        try:
            month = datetime.strptime(m.group("mon")[:3].title(), "%b").month
            t = t.replace(month=month, day=int(m.group("day")))
        except ValueError:
            return None
        if t.timestamp() < now - 86400:
            t = t.replace(year=t.year + 1)
        return t.timestamp()
    if t.timestamp() <= now:
        t += timedelta(days=1)
    return t.timestamp()


# ============================================================================ carrying a CLI session across accounts
def session_files(backend: str, home: Path, sid: str) -> list[Path]:
    """A CLI session's files in one home: Claude Code's projects/<cwd>/<id>.jsonl (and its <id>/ folder: subagents,
    big tool results), Codex's sessions/<y>/<m>/<d>/rollout-...-<id>.jsonl."""
    if not re.fullmatch(r"[\w-]+", sid or "") or not home.is_dir():
        return []
    if backend == "claude_code":
        return [p for p in home.glob(f"projects/*/{sid}.jsonl")] + [p for p in home.glob(f"projects/*/{sid}")
                                                                      if p.is_dir()]
    return list(home.glob(f"sessions/*/*/*/rollout-*{sid}.jsonl"))


def _newest(files: list[Path]) -> float:
    return max((f.stat().st_mtime for f in files if f.is_file()), default=0)


def carry_session(backend: str, sid: str, homes: list[Path], to: Path) -> bool:
    """Make the CLI session `sid` resumable from the home `to`: copy it there from the home that has its latest
    state (when that isn't `to`). False if no home has it."""
    here = session_files(backend, to, sid)
    best, best_t = None, _newest(here)
    for h in homes:
        if h == to:
            continue
        files = session_files(backend, h, sid)
        t = _newest(files)
        if files and t > best_t:
            best, best_t = (h, files), t
    if best is None:
        return bool(here)
    src, files = best
    for f in files:
        dst = to / f.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.is_dir():
            shutil.copytree(f, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(f, dst)
    log.info("%s session %s copied from %s to %s", backend, sid, src, to)
    return True


# ============================================================================ a conversation moving to the other CLI
HANDOVER = """<conversation_so_far>
This conversation started with another AI agent ({before}); you are taking it over ({reason}). You don't have its
memory: below is what was said and done so far, from Overseer's records ({shown}). Continue from here as if you had
been in it - don't greet or recap; check the state of things (files, jobs) yourself before relying on it.

{body}
</conversation_so_far>"""
RECENT_CHARS = 24_000  # the latest turns in full, up to this much
OLDER_LINES = 60  # the ones before: a line each, up to this many


def read_turns(path: Path) -> list[dict]:
    """A transcript's turns, oldest first: {ts, backend, request, reply, status, commands, files}."""
    turns: list[dict] = []
    try:
        fh = open(path, errors="replace")
    except OSError:
        return []
    with fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind = ev.get("kind")
            if kind == "turn":
                turns.append({"ts": ev.get("ts", 0), "backend": ev.get("backend", ""), "request": ev.get("text", ""),
                              "reply": "", "status": "", "commands": [], "files": []})
            elif not turns:
                continue
            elif kind == "end":
                turns[-1]["status"] = ev.get("status", "")
                turns[-1]["reply"] = ev.get("text", "") or ""
            elif kind == "command":
                turns[-1]["commands"].append(ev.get("command", ""))
            elif kind == "steer":  # (the researcher wrote while it worked)
                turns[-1]["request"] += "\n\n(while it worked) " + ev.get("text", "").split("\n", 1)[-1]
            elif kind == "file" and ev.get("path"):
                turns[-1]["files"].append(ev["path"])
    return turns


def _one_line(text: str, n: int) -> str:
    t = " ".join((text or "").split())
    return t[:n] + ("…" if len(t) > n else "")


def handover(cfg: Config, sid: str, before: str, reason: str) -> str:
    """What a conversation moving to the other CLI starts with: its turns so far (the latest in full, the older ones
    a line each). "" if there's nothing to hand over."""
    turns = [t for t in read_turns(cfg.local.state("transcripts") / f"{sid}.jsonl")
             if not t["request"].startswith("/compact")]
    if not turns:
        return ""
    recent: list[str] = []
    used = 0
    i = len(turns)
    while i > 0:
        t = turns[i - 1]
        block = _turn_block(t)
        if recent and used + len(block) > RECENT_CHARS:
            break
        if not recent and len(block) > RECENT_CHARS:
            block = block[:RECENT_CHARS // 3] + "\n[...]\n" + block[-RECENT_CHARS // 2:]
        recent.insert(0, block)
        used += len(block)
        i -= 1
    older = turns[:i]
    lines = []
    if older:
        dropped = max(0, len(older) - OLDER_LINES)
        if dropped:
            lines.append(f"({dropped} earlier turn(s) not shown: `ov history` finds them)")
        for t in older[dropped:]:
            lines.append(f"- {_stamp(t['ts'])} researcher: {_one_line(t['request'], 160)}"
                         + (f" → agent: {_one_line(t['reply'], 200)}" if t["reply"] else ""))
        lines.append("")
    shown = (f"the last {len(recent)} turn(s) in full" + (f", {len(older)} before them in short" if older else ""))
    return HANDOVER.format(before=before, reason=reason, shown=shown, body="\n".join(lines + recent).strip())


def _turn_block(t: dict) -> str:
    out = [f"── {_stamp(t['ts'])}" + (f" ({t['status']})" if t["status"] not in ("", "done") else ""),
           f"Researcher: {t['request'].strip()[:6000]}"]
    if t["commands"]:
        cmds = t["commands"][-12:]
        more = f" (the last 12 of {len(t['commands'])})" if len(t["commands"]) > 12 else ""
        out.append(f"Commands it ran{more}:\n" + "\n".join(f"  $ {_one_line(c, 240)}" for c in cmds))
    if t["files"]:
        out.append("Files it changed: " + ", ".join(dict.fromkeys(t["files"][-12:])))
    if t["status"] in ("error", "stopped"):
        why = f": {_one_line(t['reply'], 300)}" if t["status"] == "error" and t["reply"] else ""
        out.append(f"(The turn didn't finish - {'it was stopped' if t['status'] == 'stopped' else 'an error'}{why})")
    elif t["reply"]:
        out.append(f"Agent: {t['reply'].strip()[:8000]}")
    return "\n".join(out) + "\n"


def _stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
