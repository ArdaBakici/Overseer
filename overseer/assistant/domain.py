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
from .calendar import Calendar
from .reminders import Reminders, when_text
from .scheduler import Scheduler
from .vault import Vault

if TYPE_CHECKING:
    from ..core.agent import Session

log = logging.getLogger(__name__)

SKILL = Path(__file__).parent / "skills" / "assistant" / "SKILL.md"  # how to use ov, for the agent
NAME_RE = re.compile(r"^[ \t]*[-*]?[ \t]*\**name\**[ \t]*:[ \t]*\**(.+?)\**[ \t]*$", re.I | re.M)
PROTECTED = (".obsidian", ".stfolder", ".stversions", ".stignore")  # Obsidian's settings, Syncthing's markers
ALWAYS_BUDGET = 24_000  # characters of `assistant: always` notes in the instructions, all together
ALWAYS_NOTE_MAX = 12_000  # ... and of one of them


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
    reset_prefixes = ("assistant:",)  # follow-ups, day logs, the first-run and routines flags

    def __init__(self, cfg, store, notifier):
        super().__init__(cfg, store, notifier)
        self.vault = Vault(cfg)
        self.reminders = Reminders(store, notifier)
        self.reminders.on_stuck = self._stuck
        self.scheduler = Scheduler(self)  # Routines.md, and the first hello
        self.calendar = Calendar(self)  # Google Calendar (if turned on)
        self._sync: asyncio.Task | None = None

    # ------------------------------------------------------------------ the app
    def prepare(self) -> None:
        from ..core.ctl import write_ov_tool, write_skills
        from ..core.runners import agent_bin_dir
        self.vault.ensure_files()  # its own notes (persona, about me, ...), if they don't exist yet
        write_ov_tool(self.cfg, agent_bin_dir(self.cfg), _write)
        write_skills(self.cfg, skills())

    def services(self) -> list[tuple[str, object]]:
        out = [("follow-ups", self.reminders.loop()), ("routines", self.scheduler.loop())]
        if self.calendar.enabled:
            out.append(("calendar", self.calendar.loop()))
        return out

    def schedule_sync(self) -> None:
        """Read the calendar again soon (after a change)."""
        if self.calendar.enabled and not (self._sync and not self._sync.done()):
            try:
                self._sync = asyncio.get_running_loop().create_task(self.calendar.sync())
            except RuntimeError:
                pass

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
        from .media import local_tools
        tools = await asyncio.to_thread(local_tools)
        missing = [n for n, ok in tools.items() if not ok]
        have = ", ".join(n for n, ok in tools.items() if ok) or "none"
        out.append(("image tools", True, have + (f" (not installed: {', '.join(missing)} - e.g. sudo apt install "
                                                 "python3-pil ffmpeg imagemagick)" if missing else "")))
        if self.calendar.enabled:
            c = self.calendar
            await c.sync()
            out.append(("calendar", not c.error, c.error or "Google · " + ", ".join(
                f"{x.name} ({x.mode})" for x in c.cals) + f" · {len(c.events)} events in the next two weeks"))
        else:
            out.append(("calendar", True, "off"))
        m = self.cfg.media
        out.append(("media", True, f"ov fetch {'on' if m.fetch else 'off'} (up to {m.max_mb} MB) · ov gif "
                    + ("on" if m.giphy_key_env and os.environ.get(m.giphy_key_env) else
                       f"off ({m.giphy_key_env} not set)" if m.giphy_key_env else "off")))
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

    def persona_set(self) -> bool:
        """Whether the user gave the assistant a persona yet (the placeholder note has no name in it)."""
        return bool(NAME_RE.search(self.vault.read_note(self._rel("Persona.md"), 4000)))

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
            persona_unset="" if self.persona_set() else P.PERSONA_UNSET,
            writing_path=self._rel("My writing style.md"), personality_path=self._rel("My personality.md"),
            system_path=self._rel("System.md"), someday_path=self._rel("Someday.md"), silent=P.SILENT,
            assistant_dir=cfg.vault.assistant_dir.strip("/") or ".", daily=self.vault.daily_note(),
            tasks_file=self.vault.rel(self.vault.tasks_file),
            template=f", from the template `{t}`" if (t := self.vault.daily_template()) else "",
            routines_path=self.scheduler.rel, routines_state=self.scheduler.state_line(),
            domains=f" (except {', '.join(domains)})" if domains else "",
            web=self.agent.web_line(s.project if s else None) if self.agent else "")
        prompt += self._flagged_section()
        if self.calendar.enabled:
            prompt += P.CALENDAR_SECTION.format(calendars=self._calendars_line())
        if self.agent and self.agent.ctl:
            prompt += P.OV_SECTION
        return prompt + self._project_section(s)

    def _calendars_line(self) -> str:
        c = self.calendar
        if not c.cals:
            return "(not read yet" + (f": {c.error}" if c.error else "") + " - `ov cal calendars`)"
        what = {"read": "read-only", "ask": "changes after they tap ✅", "auto": "changes at once, with ↩️ Undo"}
        return "; ".join(f"{x.name}{' (primary)' if x.primary else ''}: {what[x.mode]}" for x in c.cals)

    def _flagged_section(self) -> str:
        """Notes flagged `assistant: always` (their text, within a budget) and `assistant: index` (name and
        description), besides the persona and "about me"."""
        skip = {self._rel("Persona.md"), self._rel("About me.md")}
        always, index, left_out = [], [], []
        budget = ALWAYS_BUDGET
        for rel, mode, desc in self.vault.flagged():
            if rel in skip:
                continue
            if mode == "always":
                text = self.vault.read_note(rel, ALWAYS_NOTE_MAX + 4001)
                if text.startswith("---") and (end := text.find("\n---", 3)) > 0:  # (its frontmatter: not needed)
                    text = text[end + 4:]
                text = text.strip()
                if len(text) > ALWAYS_NOTE_MAX:  # (too long to keep in mind whole: its beginning, and say so)
                    text = text[:ALWAYS_NOTE_MAX] + "\n[... cut here: the note goes on - read it for the rest, and keep it shorter]"
                if len(text) > budget:
                    left_out.append(rel)
                    continue
                budget -= len(text)
                always.append(f"<note path='{rel}'>\n{text}\n</note>")
            else:
                index.append(f"- {rel}" + (f": {desc}" if desc else ""))
        if left_out:
            index += [f"- {rel}: (flagged always, left out: over the budget - read it when it matters)" for rel in left_out]
        out = ""
        if always:
            out += "\n\n## Notes you always keep in mind\n" + "\n".join(always)
        if index:
            out += "\n\n## Notes to read when they matter\n" + "\n".join(index)
        return out

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
        if self.calendar.enabled:
            lines.append(self.calendar.context())
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
        rs = self.scheduler.load()
        if rs is not None:
            from .routines import today_lines
            later = today_lines(rs, now)
            if later:
                lines.append("<routines_later_today>" + "; ".join(later) + "</routines_later_today>")
        lines += self.followup_context()
        return "\n".join(lines)

    def followup_context(self) -> list[str]:
        """What's waiting for the user's answer, what's scheduled, and today's outcomes (done, skipped, ...)."""
        out = []
        rem = self.reminders
        kinds = {"reminder": "reminder", "say": "routine", "commit": "check-back", "focus": "focus"}
        waiting = [f"{r['id']} {kinds.get(r.get('kind'), '')}: {r['text']} (sent {when_text(r['at'])}"
                   + (f", {r['attempts']} follow-up(s)" if r.get("attempts") else "") + ")" for r in rem.open()]
        if waiting:
            out.append("<waiting_for_their_answer>\n" + "\n".join(waiting) + "\n</waiting_for_their_answer>")
        focus = rem.focus()
        if focus:
            out.append(f"<focus_session>{focus['id']}: {focus['text']} until "
                       f"{datetime.fromtimestamp(focus['at']):%H:%M}</focus_session>")
        planned = [f"{r['id']} {when_text(r['at'])} {kinds.get(r.get('kind'), '')}: {r['text']}"
                   for r in rem.pending() if r.get("kind") != "focus"][:15]
        if planned:
            out.append("<scheduled>\n" + "\n".join(planned) + "\n</scheduled>")
        today = rem.log()
        if today:
            out.append("<today_so_far>\n" + "\n".join(
                f"{datetime.fromtimestamp(e['at']):%H:%M} {e['icon']} {e['state']}: {e['text']}" for e in today[-25:])
                       + "\n</today_so_far>")
        return out

    async def _stuck(self, item: dict) -> None:
        """🧱 Stuck: the assistant is woken to help, in the conversation the follow-up came from."""
        agent = self.agent
        if not agent or not agent.autopilot or not agent.cfg.agent.enabled:
            return
        s = agent.get(item.get("session")) or agent.general_session()
        agent.autopilot.wake(s, "\n\n".join([P.STUCK_PROMPT.format(text=item["text"]), await self.now_context()]),
                             kind="stuck", by_user=True)  # (they pressed 🧱 Stuck)

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
        lines = [f"📝 {files_line(change.files)}"]
        if any(path == self.scheduler.rel for _, path in change.files):
            rs = self.scheduler.load()
            if rs is not None and rs.errors:
                lines.append("⚠️ Routines: " + "; ".join(code(e) for e in rs.errors[:5]))
        return RunNote(lines=lines,
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
