"""The Obsidian vault: the assistant's memory, and its history.

The vault is a folder of Markdown notes (kept in sync with your devices by Syncthing). The agent reads and edits it
directly. Overseer keeps a history of it in a git repository in the bot's state folder - outside the vault, so no
`.git` folder is synced and the agent can't touch the history: before each run what changed meanwhile (your edits)
is recorded, after it what the agent changed, as its own commit. That commit is what ↩️ Undo reverts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Not part of the history: Obsidian's per-device UI state, Syncthing's own folders, the trash.
EXCLUDE = """.obsidian/workspace*.json
.obsidian/cache
.trash/
.stfolder
.stversions/
.stignore
"""
CONFLICT_RE = re.compile(r"\.sync-conflict-\d{8}-\d{6}")
DEFAULT_PERSONA = """# Persona

Who the assistant is: its name, personality and way of talking. Not set yet - tell it who you'd like it to be (or
write it here, e.g. "- Name: ...").
"""
DEFAULT_ABOUT = """# About me

What the assistant knows about me: routines, preferences, people, ongoing things. It adds what it learns that
lasts; I can edit or delete anything.
"""
DEFAULT_WRITING = """---
assistant: index
description: how I write, informal and formal - kept up to date; for other AI agents to write like me
---
# My writing style

Kept by the assistant from how I write. Give this note to another AI agent to have it write like me.

## Informal
(chats, messages)

## Formal
(emails, documents)
"""
DEFAULT_PERSONALITY = """---
assistant: index
description: who I am - temperament, values, interests, how I decide; for other AI agents to understand me
---
# My personality

Kept by the assistant from what it learns about me. Give this note to another AI agent to have it understand me.
"""
NOTES = (("Persona.md", DEFAULT_PERSONA), ("About me.md", DEFAULT_ABOUT), ("My writing style.md", DEFAULT_WRITING),
         ("My personality.md", DEFAULT_PERSONALITY))

# Obsidian's (moment.js) date tokens -> strftime, for daily note names.
MOMENT = [("YYYY", "%Y"), ("YY", "%y"), ("MMMM", "%B"), ("MMM", "%b"), ("MM", "%m"), ("DD", "%d"),
          ("dddd", "%A"), ("ddd", "%a"), ("Do", "%-d"), ("D", "%-d"), ("M", "%-m")]


def moment_to_strftime(fmt: str) -> str:
    out, i = "", 0
    while i < len(fmt):
        if fmt[i] == "[":  # [literal]
            j = fmt.find("]", i)
            out += fmt[i + 1:j] if j > 0 else fmt[i + 1:]
            i = j + 1 if j > 0 else len(fmt)
            continue
        for tok, rep in MOMENT:
            if fmt.startswith(tok, i):
                out += rep
                i += len(tok)
                break
        else:
            out += fmt[i]
            i += 1
    return out


def _frontmatter_flag(p: Path) -> tuple[str, str] | None:
    """(always | index, description) from a note's YAML frontmatter, if it has the `assistant:` flag."""
    try:
        with open(p, errors="replace") as fh:
            head = fh.read(4000)
    except OSError:
        return None
    if not head.startswith("---"):
        return None
    end = head.find("\n---", 3)
    if end < 0:
        return None
    try:
        import yaml
        meta = yaml.safe_load(head[3:end]) or {}
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    flag = meta.get("assistant")
    flags = {str(x).lower() for x in flag} if isinstance(flag, list) else {str(flag).lower()} if flag else set()
    mode = "always" if "always" in flags or "always-load" in flags else "index" if "index" in flags else ""
    return (mode, " ".join(str(meta.get("description") or "").split())[:300]) if mode else None


@dataclass
class Change:
    commit: str
    when: float
    message: str
    files: list[tuple[str, str]] = field(default_factory=list)  # (status A/M/D/R, path)


class Vault:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root: Path = cfg.vault_dir
        self.git_dir: Path = cfg.vault_git_dir
        self._lock = asyncio.Lock()
        self.history = bool(cfg.vault.history and shutil.which("git"))
        self.extra_exclude: list[str] = []  # more for the history to leave out (e.g. the HPC bot's graphs)

    # ------------------------------------------------------------------ the assistant's files
    @property
    def assistant_dir(self) -> Path:
        return self.root / self.cfg.vault.assistant_dir

    @property
    def tasks_file(self) -> Path:
        rel = self.cfg.vault.tasks_file or f"{self.cfg.vault.assistant_dir}/Tasks.md"
        return self.root / rel

    def ensure_files(self) -> None:
        """The assistant's own notes (persona, about me, writing style, personality), with a starting text, if they
        don't exist yet."""
        d = self.assistant_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / "Memory").mkdir(exist_ok=True)
        for name, text in NOTES:
            if not (d / name).exists():
                (d / name).write_text(text)

    def read_note(self, rel: str, limit: int = 20_000) -> str:
        p = self.root / rel
        try:
            if p.is_symlink() or not p.resolve().is_relative_to(self.root):
                return ""
            return p.read_text(errors="replace")[:limit]
        except OSError:
            return ""

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))

    def _daily_settings(self) -> dict:
        try:
            st = json.loads((self.root / ".obsidian" / "daily-notes.json").read_text())
        except (OSError, ValueError):
            return {}
        return st if isinstance(st, dict) else {}

    def daily_template(self) -> str:
        """The daily notes' template in Obsidian's settings (relative path), "" if none."""
        t = str(self._daily_settings().get("template") or "").strip().strip("/")
        return "" if not t else t if t.endswith(".md") else t + ".md"

    def append_under(self, rel: str, heading: str, line: str) -> bool:
        """Add a line at the end of a section of an existing note (the section is added if missing). False if the
        note doesn't exist (not created here: Obsidian applies its template when it creates the note)."""
        from .tasks import _safe_path, _write
        p = _safe_path(self, rel)
        if not p.exists():
            return False
        lines = p.read_text().split("\n")
        at = next((i for i, x in enumerate(lines) if x.strip().lower() == heading.lower()), None)
        if at is None:
            while lines and not lines[-1].strip():
                lines.pop()
            lines += ["", heading, line, ""]
        else:
            end = at + 1
            while end < len(lines) and not lines[end].startswith("#"):
                end += 1
            while end > at + 1 and not lines[end - 1].strip():
                end -= 1
            lines.insert(end, line)
        _write(p, "\n".join(lines))
        return True

    def daily_note(self, day: float | None = None) -> str:
        """Today's daily note (relative path), by Obsidian's settings (or the config's)."""
        folder, fmt = self.cfg.vault.daily_folder, self.cfg.vault.daily_format
        if folder is None or fmt is None:
            st = self._daily_settings()
            folder = st.get("folder", "") if folder is None else folder
            fmt = st.get("format") or "YYYY-MM-DD" if fmt is None else fmt
        name = time.strftime(moment_to_strftime(fmt), time.localtime(day or time.time()))
        return str(Path(folder.strip("/")) / f"{name}.md") if folder.strip("/") else f"{name}.md"

    def notes(self) -> list[Path]:
        """Every note (Markdown file) in the vault, not following links, skipping dot-folders."""
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            out += [Path(dirpath) / f for f in filenames if f.endswith(".md")]
        return out

    def flagged(self) -> list[tuple[str, str, str]]:
        """Notes whose frontmatter says `assistant: always` (their text goes into the assistant's instructions) or
        `assistant: index` (just their name and `description:`): [(path, always|index, description)]."""
        out = []
        cache = self.__dict__.setdefault("_flag_cache", {})
        for p in self.notes():
            try:
                st = p.stat()
                if p.is_symlink():
                    continue
            except OSError:
                continue
            rel = self.rel(p)
            hit = cache.get(rel)
            if hit is None or hit[0] != (st.st_mtime, st.st_size):
                hit = ((st.st_mtime, st.st_size), _frontmatter_flag(p))
                cache[rel] = hit
            if hit[1]:
                out.append((rel, *hit[1]))
        return sorted(out)

    def conflicts(self) -> list[str]:
        """Syncthing's conflict copies: two devices changed a note at once."""
        return sorted(self.rel(p) for p in self.notes() if CONFLICT_RE.search(p.name))

    # ------------------------------------------------------------------ the history (git, outside the vault)
    async def _git(self, *args: str, check: bool = True) -> tuple[int, str]:
        env = {**{k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "TZ", "HOME")},
               "GIT_DIR": str(self.git_dir), "GIT_WORK_TREE": str(self.root), "GIT_TERMINAL_PROMPT": "0"}
        proc = await asyncio.create_subprocess_exec(
            "git", "-c", "core.quotepath=off", "-c", "user.name=Overseer", "-c", "user.email=overseer@localhost",
            "-c", "commit.gpgsign=false", *args, cwd=str(self.root), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        text = out.decode(errors="replace")
        if check and proc.returncode:
            raise RuntimeError(f"git {' '.join(args[:2])}: {text.strip()[:500]}")
        return proc.returncode, text

    async def init(self) -> None:
        if not self.history:
            return
        if not (self.git_dir / "HEAD").exists():
            self.git_dir.parent.mkdir(parents=True, exist_ok=True)
            await self._git("init", "-q")
            (self.git_dir / "info").mkdir(exist_ok=True)
            self._write_exclude()
            await self.snapshot("Vault history starts")
        else:
            self._write_exclude()

    def _write_exclude(self) -> None:
        text = EXCLUDE + "".join(f"{p}\n" for p in self.extra_exclude)
        f = self.git_dir / "info" / "exclude"
        try:
            if f.read_text() != text:
                f.write_text(text)
        except OSError:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)

    async def snapshot(self, message: str) -> Change | None:
        """Record what changed since the last snapshot as one commit (None if nothing did)."""
        if not self.history:
            return None
        async with self._lock:
            await self._git("add", "-A")
            rc, _ = await self._git("diff", "--cached", "--quiet", check=False)
            if rc == 0:
                return None
            await self._git("commit", "-q", "--no-verify", "-m", message)
            return await self._change("HEAD")

    async def _change(self, ref: str) -> Change:
        _, out = await self._git("show", "--no-renames", "--name-status", "--format=%H%x00%ct%x00%s", ref)
        lines = out.strip().splitlines()
        commit, when, message = lines[0].split("\x00")
        files = [tuple(x.split("\t", 1)) for x in lines[1:] if "\t" in x]
        return Change(commit, float(when), message, files)  # type: ignore[arg-type]

    async def log(self, limit: int = 30) -> list[Change]:
        if not self.history or not (self.git_dir / "HEAD").exists():
            return []
        rc, out = await self._git("log", f"-{limit}", "--format=%H", check=False)
        return [await self._change(h) for h in out.split()] if rc == 0 else []

    async def undo(self, commit: str) -> tuple[bool, str]:
        """Revert one turn's changes; your edits since are kept (recorded first). (ok, what happened)"""
        if not self.history or not re.fullmatch(r"[0-9a-f]{7,40}", commit):
            return False, "no such change"
        await self.snapshot("Your edits")
        async with self._lock:
            rc, out = await self._git("revert", "--no-edit", commit, check=False)
            if rc:
                await self._git("revert", "--abort", check=False)
                if "conflict" in out.lower():
                    return False, "those notes changed since: undo it by hand (or ask the assistant)"
                return False, out.strip()[:300]
        change = await self._change("HEAD")
        return True, ", ".join(p for _, p in change.files) or "nothing to undo"
