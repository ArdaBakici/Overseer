"""The personal assistant domain: a daily assistant whose memory is your Obsidian vault.

What it adds to the platform: the agent's instructions (its persona and what it knows about you, from notes in the
vault; the vault's conventions; Obsidian Tasks), the context of a request (the date, tasks due and overdue), a
sandbox that can write the vault and nothing else of yours (not Obsidian's settings, not the HPC bot's anything),
the vault's history with ↩️ Undo for each reply's changes, tasks and reminders (`ov task`, `ov remind`, /tasks,
/remind), and its dashboard, digest and search.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.domain import Domain, RunNote
from ..core.notify import code
from ..core.runners import Activity, Sandbox
from . import prompts as P
from . import tasks as T
from .reminders import Reminders
from .vault import Vault

if TYPE_CHECKING:
    from ..core.agent import Session

log = logging.getLogger(__name__)

SKILL = Path(__file__).parent / "skills" / "assistant" / "SKILL.md"  # how to use ov, for the agent
NAME_RE = re.compile(r"^\s*[-*]?\s*\**name\**\s*:\s*\**(.+?)\**\s*$", re.I | re.M)
PROTECTED = (".obsidian", ".stfolder", ".stversions", ".stignore")  # Obsidian's settings, Syncthing's markers


def skills() -> dict[str, str]:
    return {"assistant": SKILL.read_text()}


def make_domain(cfg, store, notifier) -> AssistantDomain:
    """The app's way in (core/app.py)."""
    return AssistantDomain(cfg, store, notifier)


def _write(path: Path, content: str, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def files_line(files: list[tuple[str, str]], limit: int = 8) -> str:
    mark = {"A": "+", "D": "−", "M": ""}
    names = [f"{mark.get(st[:1], '')}{p}" for st, p in files]
    more = f" (+{len(names) - limit} more)" if len(names) > limit else ""
    return ", ".join(code(n) for n in names[:limit]) + more


class AssistantDomain(Domain):
    name = "assistant"
    command = "ov"
    planner = False  # no planner mode: nothing here needs subagents

    def __init__(self, cfg, store, notifier):
        super().__init__(cfg, store, notifier)
        self.vault = Vault(cfg)
        self.reminders = Reminders(store, notifier)
        self._dashboard = None

    # ------------------------------------------------------------------ the app
    def prepare(self) -> None:
        from ..core.ctl import write_ov_tool, write_skills
        from ..core.runners import agent_bin_dir
        self.vault.ensure_files()  # the persona and "about me" notes, if they don't exist yet
        write_ov_tool(self.cfg, agent_bin_dir(self.cfg), _write)
        write_skills(self.cfg, skills())

    def services(self) -> list[tuple[str, object]]:
        return [("reminders", self.reminders.loop())]

    async def start(self, notifier, projects) -> None:
        try:
            await self.vault.init()  # the history starts with the vault as it is
        except Exception:
            log.warning("could not start the vault's history", exc_info=True)

    async def check(self) -> list[tuple[str, bool, str]]:
        root = self.vault.root
        notes = await asyncio.to_thread(self.vault.notes)
        out = [("vault", os.access(root, os.W_OK), f"{root} · {len(notes)} notes"
                + ("" if os.access(root, os.W_OK) else " · not writable by this user")),
               ("vault history", self.vault.history or not self.cfg.vault.history,
                "on (git)" if self.vault.history else "git not found" if self.cfg.vault.history else "off")]
        if not (root / ".obsidian").is_dir():
            out.append(("obsidian", True, "no .obsidian folder: is this the vault's top folder?"))
        return out

    # ------------------------------------------------------------------ the sandbox
    def sandbox(self) -> Sandbox:
        root = self.cfg.vault_dir
        return Sandbox(writable=[root], deny_write=[root / p for p in PROTECTED if (root / p).exists()],
                       skills=skills())

    # ------------------------------------------------------------------ instructions
    def persona_name(self) -> str:
        m = NAME_RE.search(self.vault.read_note(self._rel("Persona.md"), 4000))
        return m.group(1).strip()[:40] if m else "the assistant"

    # ------------------------------------------------------------------ projects: a topic each, maybe a vault folder
    def check_project_dir(self, path: str) -> str:
        """A project's folder in the vault (optional): a relative path, e.g. Projects/Japan trip."""
        path = (path or "").strip().rstrip("/")
        if not path:
            return ""
        parts = Path(path).parts
        if path.startswith(("/", "~")) or ".." in parts or any(x.startswith(".") for x in parts):
            raise ValueError(f"{path!r}: give a folder inside the vault, e.g. Projects/Japan (no .., no dot-folders)")
        return str(Path(*parts))

    def _project_section(self, s: Session | None) -> str:
        proj = self.store.get_project(s.project) if s and s.project else None
        if not proj:
            return ""
        about = f"\nWhat it's about, from the user:\n{proj.context.strip()}" if proj.context.strip() else ""
        where = (f"Its notes live in `{proj.dir}/` in the vault: read them first, keep them up to date."
                 if proj.dir else f"It has no folder in the vault yet: when there's something worth keeping, make one "
                 f"(e.g. `Projects/{proj.name}/`) and set it: `ov project set {proj.name} --dir <folder>`.")
        return P.PROJECT_SECTION.format(name=proj.name, about=about, where=where)

    def _rel(self, name: str) -> str:
        return f"{self.cfg.vault.assistant_dir.strip('/')}/{name}" if self.cfg.vault.assistant_dir.strip("/") else name

    def system_prompt(self, s: Session | None) -> str:
        cfg = self.cfg
        domains = cfg.agent.allowed_domains
        persona = self.vault.read_note(self._rel("Persona.md"), 6000).strip() or "(none yet)"
        about = self.vault.read_note(self._rel("About me.md"), 12000).strip() or "(nothing yet)"
        prompt = P.SYSTEM_PROMPT.format(
            name=self.persona_name(), persona=persona, about=about, vault=self.vault.root,
            persona_path=self._rel("Persona.md"), about_path=self._rel("About me.md"),
            assistant_dir=cfg.vault.assistant_dir.strip("/") or ".", daily=self.vault.daily_note(),
            tasks_file=self.vault.rel(self.vault.tasks_file),
            domains=f" (except {', '.join(domains)})" if domains else "",
            web=self.agent.web_line(s.project if s else None) if self.agent else "")
        if self.agent and self.agent.ctl:
            prompt += P.OV_SECTION
        return prompt + self._project_section(s)

    def wake_instructions(self, s: Session) -> str:
        return P.WAKE_INSTRUCTIONS

    def auto_prompt(self, goal: str) -> str:
        return P.AUTO_PROMPT.format(goal=goal)

    # ------------------------------------------------------------------ context
    async def context(self, s: Session, fresh: bool) -> list[str]:
        return [await self.now_context()]

    async def wake_context(self, s: Session) -> list[str]:
        return [await self.now_context()]

    async def now_context(self) -> str:
        """The date and time (the agent can't know them), what's due, and Syncthing's conflicts."""
        now = datetime.now()
        lines = [f"<now>{now:%A %d %B %Y, %H:%M} {time.strftime('%Z')}</now>"]
        try:
            all_tasks = await asyncio.to_thread(T.scan, self.vault)
        except OSError:
            all_tasks = []
        today = now.date()
        due = T.select(all_tasks, "today", today=today)
        if due:
            shown = "\n".join(T.describe(t, today) for t in due[:15])
            more = f"\n(+{len(due) - 15} more: ov task list --when today)" if len(due) > 15 else ""
            lines.append(f"<tasks_due_today_or_overdue>\n{shown}{more}\n</tasks_due_today_or_overdue>")
        conflicts = await asyncio.to_thread(self.vault.conflicts)
        if conflicts:
            lines.append("<sync_conflicts>" + ", ".join(conflicts[:10]) + "</sync_conflicts>")
        return "\n".join(lines)

    # ------------------------------------------------------------------ a run: the vault's changes
    async def before_run(self, s: Session) -> Any:
        try:
            await self.vault.init()
            await self.vault.snapshot("Your edits")  # (what changed since: yours, from Obsidian, or synced)
        except Exception:
            log.warning("vault history: could not record the changes before the run", exc_info=True)
        return None

    async def after_run(self, s: Session, activity: Activity, state: Any, reply: str) -> RunNote | None:
        summary = " ".join((reply or "").split())[:72] or "(no answer)"
        try:
            change = await self.vault.snapshot(f"{self.persona_name()} · {s.id}: {summary}")
        except Exception:
            log.warning("vault history: could not record the run's changes", exc_info=True)
            return None
        if not change:
            return None
        return RunNote(lines=[f"📝 {files_line(change.files)}"],
                       buttons=[[("↩️ Undo", f"vundo:{change.commit[:12]}")]])

    async def record_before(self) -> None:
        """Before Overseer changes a note for you (a task ticked off in Telegram): your edits so far, on their own."""
        await self.record("Your edits")

    async def record(self, message: str):
        """...and after: that change, as a commit of its own (undoable)."""
        try:
            await self.vault.init()
            return await self.vault.snapshot(message)
        except Exception:
            log.warning("vault history: could not record %r", message, exc_info=True)
            return None

    # ------------------------------------------------------------------ the agent's ov command, Telegram
    def ctl_commands(self, ctl):
        from .commands import AssistantCommands
        return AssistantCommands(ctl, self)

    def bot_commands(self, bot):
        from .telegram import AssistantBot
        return AssistantBot(bot, self)

    # ------------------------------------------------------------------ the morning digest
    async def digest_sections(self, since: float, now: float) -> list[str]:
        from .digest import sections
        return await sections(self, since, now)

    async def digest_closing(self) -> list[str]:
        conflicts = await asyncio.to_thread(self.vault.conflicts)
        if not conflicts:
            return []
        return [f"⚠️ Sync conflicts in the vault: {', '.join(code(c) for c in conflicts[:5])}"
                + (f" (+{len(conflicts) - 5})" if len(conflicts) > 5 else "")]

    # ------------------------------------------------------------------ the dashboard
    def dashboard_js(self) -> bytes:
        return (Path(__file__).parent / "dashboard.js").read_bytes()

    async def dashboard_api(self, parts: list[str], query: dict[str, str]):
        from .dashboard import route
        return await route(self, parts, query)

    # ------------------------------------------------------------------ search
    def search_rows(self) -> list[tuple]:
        from .search import search_rows
        return search_rows(self)

    def search_line(self, hit, where: str, snippet: str, html: bool) -> str | None:
        from .search import search_line
        return search_line(hit, where, snippet, html)

    # ------------------------------------------------------------------ helpers for commands
    def today(self) -> date:
        return date.today()

    def task_lines(self, items: list[T.Task], limit: int = 40) -> str:
        today = self.today()
        out = [T.describe(t, today) for t in items[:limit]]
        if len(items) > limit:
            out.append(f"(+{len(items) - limit} more)")
        return "\n".join(out)
