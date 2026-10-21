"""The assistant's `ov` commands: tasks (Obsidian Tasks, anywhere in the vault) and reminders. The platform's control
server (core/ctl.py) runs them for the agent."""

from __future__ import annotations

import asyncio

from ..core.ctl import CtlError
from ..core.notify import esc
from . import tasks as T
from .reminders import when_text
from .when import parse_day, parse_moment

AREAS = ("task", "remind")

HELP = """ov - talk to Overseer (the system that runs you and sends your replies to the user on Telegram)

Tasks (checklist lines anywhere in the vault, Obsidian Tasks format):
  ov task list [--when open|today|overdue|week|upcoming|undated|done|all] [--tag t] [--search words] [--limit N]
                                   ids are the first column (they change when a task's line changes)
  ov task add "text" [--due D] [--scheduled D] [--priority highest|high|medium|low|lowest] [--every "week"]
              [--note path/in/vault.md]       (default: the tasks note)
  ov task done <id>                mark it done (✅ today); a recurring task gets its next occurrence
                                   D: YYYY-MM-DD, today, tomorrow, a weekday (the next one), +3d, +2w
Reminders (a message to the user then; you are not woken):
  ov remind <when> "<text>"        when: 17:30, "tomorrow 9am", "friday 18:00", "2026-10-20 09:00", 2h, 45m
  ov remind list                   ov remind rm <id>

Projects (a lasting subject - a trip, a move, a course: its own Telegram topic, conversations, notes folder):
  ov project new <name> --about "what it is" [--dir <vault folder>] [--move-here]
  ov project list | set <name> [--about ...] [--dir ...] | archive <name>
  ov conversation move <project|main>

  ov notify <text>                 send the user a message now
  ov status                        this conversation: what will wake you
  ov history search <text> [--since 30d] [--limit N]   your past conversations (and the vault's notes)
  ov history show <conversation> [--last 5] [--grep text]
  ov wake in 90m ["what to do then"]   ov wake every 1d [--for 7d] ["..."]   ov watches   ov unwatch <w12 | all>
  ov feedback "<what limited you>" [--kind limitation|bug|idea]   ov feedback list
  ov skill propose skill.md | list | show <name>
"""


class AssistantCommands:
    """Registered with the control server by AssistantDomain."""

    help = HELP
    errors = (T.TaskError,)

    def __init__(self, ctl, domain):
        self.ctl = ctl
        self.domain = domain

    def __getattr__(self, name):  # store, agent, notifier, cfg, _announce, ...
        return getattr(self.ctl, name)

    @property
    def vault(self):
        return self.domain.vault

    def register(self, sub) -> None:
        task = sub.add_parser("task", add_help=False).add_subparsers(dest="action")
        a = task.add_parser("list", add_help=False)
        a.add_argument("--when", default="open")
        a.add_argument("--tag", default="")
        a.add_argument("--search", default="")
        a.add_argument("--limit", type=int, default=40)
        a = task.add_parser("add", add_help=False)
        a.add_argument("text", nargs="+")
        a.add_argument("--due", default="")
        a.add_argument("--scheduled", default="")
        a.add_argument("--priority", default="")
        a.add_argument("--every", default="")
        a.add_argument("--note", default="")
        task.add_parser("done", add_help=False).add_argument("id")
        a = sub.add_parser("remind", add_help=False)
        a.add_argument("words", nargs="+")

    async def dispatch(self, session, args) -> str | None:
        """The answer, or None when the command isn't the assistant's."""
        if args.area not in AREAS:
            return None
        if args.area == "remind":
            return await self.remind(session, args.words)
        if not args.action:
            raise CtlError("ov task list | add | done (ov help)")
        return await getattr(self, f"task_{args.action}")(session, args)

    def status_lines(self, session) -> list[str]:
        mine = [r for r in self.domain.reminders.pending() if r.get("session") == session.id]
        return [f"reminders set in this conversation: {len(mine)}" + (f" (next: {when_text(mine[0]['at'])})"
                                                                      if mine else "")]

    # ------------------------------------------------------------------ tasks
    def _day(self, text: str) -> str:
        return parse_day(text, self.domain.today()).isoformat() if text else ""

    async def task_list(self, session, args) -> str:
        items = await asyncio.to_thread(T.scan, self.vault)
        hits = T.select(items, args.when, args.tag, args.search, self.domain.today())
        if not hits:
            return f"no tasks ({args.when}{', #' + args.tag.lstrip('#') if args.tag else ''})"
        return self.domain.task_lines(hits, max(1, min(args.limit, 200)))

    async def task_add(self, session, args) -> str:
        line = T.line_for(" ".join(args.text), due=self._day(args.due), scheduled=self._day(args.scheduled),
                          priority=args.priority.lower(),
                          recurrence=("every " + args.every.removeprefix("every ").strip()) if args.every else "")
        rel = T.add(self.vault, line, args.note or None)
        return f"✓ added to {rel}: {line[6:]}"

    async def task_done(self, session, args) -> str:
        t, remark = T.complete(self.vault, args.id, self.domain.today())
        return f"✓ done: {t.text}" + (f" ({remark})" if remark else "")

    # ------------------------------------------------------------------ reminders
    async def remind(self, session, words: list[str]) -> str:
        if words[0] == "list":
            items = self.domain.reminders.pending()
            return "\n".join(f"{r['id']} {when_text(r['at'])}: {r['text']}" for r in items) or "no reminders"
        if words[0] in ("rm", "remove", "delete") and len(words) == 2:
            r = self.domain.reminders.remove(words[1])
            if not r:
                raise CtlError(f"no reminder {words[1]} (ov remind list)")
            await self._announce(session, f"🗑 Reminder removed: {esc(r['text'])}")
            return f"✓ removed {r['id']}"
        if len(words) < 2:
            raise CtlError('ov remind <when> "<text>"')
        when, text = self._split_when(words)
        r = self.domain.reminders.add(when.timestamp(), text, session.id, session.project)
        await self._announce(session, f"⏰ Reminder set for {esc(when_text(r['at']))}: {esc(r['text'])}",
                             buttons=[[("✖ Cancel", f"rrm:{r['id']}")]])
        return f"✓ {r['id']}: {when_text(r['at'])}"

    def _split_when(self, words: list[str]):
        """The longest start of the words that reads as a moment ("tomorrow 9am"), the rest is the text."""
        last_error = None
        for n in (3, 2, 1):
            if len(words) > n:
                try:
                    return parse_moment(" ".join(words[:n])), " ".join(words[n:])
                except T.TaskError as e:
                    last_error = e
        raise CtlError(str(last_error) if last_error else 'ov remind <when> "<text>"')
