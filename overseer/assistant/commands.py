"""The assistant's `ov` commands: tasks (Obsidian Tasks, anywhere in the vault), reminders, check-backs on
commitments, focus sessions, the follow-ups waiting for an answer, the routines note, and the calendar. The platform's control
server (core/ctl.py) runs them for the agent."""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlsplit

import httpx

from ..core.ctl import CtlError
from ..core.notify import Outgoing, esc
from . import calendar as C
from . import media
from . import tasks as T
from . import routines as R
from .reminders import exact_text, when_text
from .when import parse_day, parse_moment

AREAS = ("task", "remind", "commit", "focus", "followup", "routine", "fetch", "gif", "capabilities", "cal")
ACTIONS = ("done", "later", "skip", "stuck", "cancel")

HELP = """ov - talk to Overseer (the system that runs you and sends your replies to the user on Telegram)

Tasks (checklist lines anywhere in the vault, Obsidian Tasks format):
  ov task list [--when open|today|overdue|week|upcoming|undated|someday|done|all] [--tag t] [--search words]
               [--limit N]
                                   ids are the first column (they change when a task's line changes)
  ov task add "text" [--due D] [--scheduled D] [--priority highest|high|medium|low|lowest] [--every "week"]
              [--note path/in/vault.md]       (default: the tasks note)
              [--someday]                     something to do sometime: tagged #someday, into the someday note
  ov task done <id>                mark it done (✅ today); a recurring task gets its next occurrence
                                   D: YYYY-MM-DD, today, tomorrow, a weekday (the next one), +3d, +2w
Reminders and follow-ups (messages to the user with buttons; again until answered only with --follow-up; you
are not woken):
  ov remind <when> "<text>" [--follow-up <policy>]
                                   when: 17:30, "tomorrow 9am", "friday 18:00", "2026-10-20 09:00", 2h, 45m
  ov commit <when> "<what they'll do>" [--follow-up <policy>]
                                   a check-back on a commitment: "how did it go?" ✅ / 💤 / 🧱 Stuck (wakes you)
  ov focus <25m> "<the step>"      a focus session: a quiet countdown, then a ping    ov focus stop | ov focus
  ov followup list                 waiting for an answer, and scheduled (ids)
  ov followup done|later|skip|stuck|cancel <id>   when they answer in words instead of a button
                                   <policy>: a name from the routines note's # Follow-ups, "again after 10m, up
                                   to 3 more times", or none
  ov routine check                 the routines note: problems, and when each runs next
Calendar (Google, and read-only iCal links; if turned on - ids are the first column of a list):
  ov cal list [today|tomorrow|week|next week|<day>] [--days N] [--calendar c] [--search words]
  ov cal free [<day>] [--from 09:00] [--to 21:00] [--min 30m]   free stretches between events
  ov cal calendars                 the calendars, and which ones you may change
  ov cal add "<title>" <start> [<length>|<end>] [--calendar c] [--where p] [--note t]
                                   start: "thursday 14:00", "tomorrow 9am"; length: 2h, 45m, or an end, 16:00
  ov cal add "<title>" <day> --all-day [--days N]
  ov cal move <id> <start> [<length>] · ov cal edit <id> [--title t] [--where p] [--note t] · ov cal delete <id>
                                   changes go through the user: a card to tap ✅ on (or done at once, with
                                   ↩️ Undo, in calendars they allowed); never events with other people
Pictures, GIFs and videos from the internet (your shell has none: Overseer downloads them, public addresses only):
  ov fetch <url> [--name n] [--keep]   into ./outbox/ (sent with your reply), --keep: into ./inbox/ to work on
  ov gif <query> [--pick N]        a GIF from Giphy into ./outbox/ (if turned on)
  ov react <emoji>                 react to the user's last message (👍 ❤ 🔥 🎉 👏 🙏 💯 🤝 ...)
  ov capabilities                  what you can send and do here

Projects (a lasting subject - a trip, a move, a course: its own Telegram topic, conversations, notes folder):
  ov project new <name> --about "what it is" [--dir <vault folder>] [--move-here]
  ov project list | set <name> [--about ...] [--dir ...] | archive <name>
  ov conversation move <project|main>

  ov ask "<question>" [--options "Yes|Another time|No"]   a question with tap buttons; their tap comes back to
                                   you as their answer (2-6 choices)
  ov notify <text>                 send the user a message now
  ov status                        this conversation: what will wake you
  ov history search <text> [--since 30d] [--limit N]   your past conversations (and the vault's notes)
  ov history show <conversation> [--last 5] [--grep text]
  ov history day [today|yesterday|YYYY-MM-DD]   every conversation of that day (e.g. for a nightly reflection)
  ov wake in 90m ["what to do then"]   ov wake every 1d [--for 7d] ["..."]   ov watches   ov unwatch <w12 | all>
  ov feedback "<what limited you>" [--kind limitation|bug|idea]   ov feedback list
  ov skill propose skill.md | list | show <name>
"""


def _clock(text: str):
    from .when import _clock as clock
    try:
        return clock(text)
    except T.TaskError as e:
        raise CtlError(str(e)) from None


def _length(text: str) -> int:
    """30m, 2h, 1h30m, 90 (minutes) -> seconds."""
    import re
    t = text.strip().lower().replace(" ", "")
    if t.isdigit():
        return int(t) * 60
    parts = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m(?:in)?)?", t)
    if not t or not parts or not any(parts.groups()):
        raise CtlError(f"{text!r}: a length is like 30m, 2h or 1h30m (or an end time, 16:00)")
    seconds = int(parts.group(1) or 0) * 3600 + int(parts.group(2) or 0) * 60
    if not 0 < seconds <= 7 * 86400:
        raise CtlError(f"{text!r}: too long")
    return seconds


class AssistantCommands:
    """Registered with the control server by AssistantDomain."""

    help = HELP
    errors = (T.TaskError, C.CalendarError)

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
        a.add_argument("--someday", action="store_true")
        task.add_parser("done", add_help=False).add_argument("id")
        for name in ("remind", "commit"):
            a = sub.add_parser(name, add_help=False)
            a.add_argument("words", nargs="+")
            a.add_argument("--follow-up", dest="follow_up", default="")
        sub.add_parser("focus", add_help=False).add_argument("words", nargs="*")
        a = sub.add_parser("followup", add_help=False)
        a.add_argument("action", nargs="?", default="list")
        a.add_argument("id", nargs="?")
        sub.add_parser("routine", add_help=False).add_argument("action", nargs="?", default="check")
        a = sub.add_parser("fetch", add_help=False)
        a.add_argument("url")
        a.add_argument("--name", default="")
        a.add_argument("--keep", action="store_true", help="into ./inbox (to work on) instead of the reply")
        sub.add_parser("capabilities", add_help=False)
        cal = sub.add_parser("cal", add_help=False).add_subparsers(dest="action")
        a = cal.add_parser("list", add_help=False)
        a.add_argument("when", nargs="*")
        a.add_argument("--days", type=int, default=0)
        a.add_argument("--calendar", default="")
        a.add_argument("--search", default="")
        a = cal.add_parser("free", add_help=False)
        a.add_argument("day", nargs="?", default="today")
        a.add_argument("--from", dest="from_", default="09:00")
        a.add_argument("--to", default="21:00")
        a.add_argument("--min", default="30m")
        cal.add_parser("calendars", add_help=False)
        a = cal.add_parser("add", add_help=False)
        a.add_argument("title")
        a.add_argument("start")
        a.add_argument("length", nargs="?", default="")
        a.add_argument("--calendar", default="")
        a.add_argument("--where", default="")
        a.add_argument("--note", default="")
        a.add_argument("--all-day", dest="all_day", action="store_true")
        a.add_argument("--days", type=int, default=1)
        a = cal.add_parser("move", add_help=False)
        a.add_argument("id")
        a.add_argument("start")
        a.add_argument("length", nargs="?", default="")
        a = cal.add_parser("edit", add_help=False)
        a.add_argument("id")
        a.add_argument("--title")
        a.add_argument("--where")
        a.add_argument("--note")
        cal.add_parser("delete", add_help=False).add_argument("id")
        a = sub.add_parser("gif", add_help=False)
        a.add_argument("query", nargs="+")
        a.add_argument("--pick", type=int, default=1)

    async def dispatch(self, session, args) -> str | None:
        """The answer, or None when the command isn't the assistant's."""
        if args.area not in AREAS:
            return None
        if args.area in ("remind", "commit"):
            return await self.remind(session, args.words, args.follow_up, kind=args.area)
        if args.area == "focus":
            return await self.focus(session, args.words)
        if args.area == "followup":
            return await self.followup(session, args.action, args.id)
        if args.area == "capabilities":
            m = self.cfg.media
            return await asyncio.to_thread(media.capabilities, self.cfg,
                                           bool(m.giphy_key_env and os.environ.get(m.giphy_key_env)))
        if args.area == "fetch":
            return await self.fetch(session, args.url, args.name, args.keep)
        if args.area == "gif":
            return await self.gif(session, " ".join(args.query), args.pick)
        if args.area == "cal":
            return await self.cal(session, args)
        if args.area == "routine":
            if args.action not in ("check", "list"):
                raise CtlError("ov routine check (edit the routines note itself to change them)")
            return self.domain.scheduler.check_text()
        if not args.action:
            raise CtlError("ov task list | add | done (ov help)")
        return await getattr(self, f"task_{args.action}")(session, args)

    def status_lines(self, session) -> list[str]:
        mine = [r for r in self.domain.reminders.pending() if r.get("session") == session.id]
        return [f"reminders set in this conversation: {len(mine)}" + (f" (next: {when_text(mine[0]['at'])})"
                                                                      if mine else ""),
                f"waiting for the user's answer: {len(self.domain.reminders.open())}"]

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
        text = " ".join(args.text)
        if args.someday and T.SOMEDAY not in text.split():
            text += " " + T.SOMEDAY
        line = T.line_for(text, due=self._day(args.due), scheduled=self._day(args.scheduled),
                          priority=args.priority.lower(),
                          recurrence=("every " + args.every.removeprefix("every ").strip()) if args.every else "")
        rel = T.add(self.vault, line, args.note or (self.domain._rel("Someday.md") if args.someday else None))
        return f"✓ added to {rel}: {line[6:]}"

    async def task_done(self, session, args) -> str:
        t, remark = T.complete(self.vault, args.id, self.domain.today())
        return f"✓ done: {t.text}" + (f" ({remark})" if remark else "")

    # ------------------------------------------------------------------ the calendar
    async def cal(self, session, args) -> str:
        c = self.domain.calendar
        if not c.enabled:
            raise CtlError("the calendar is off (calendar.enabled in the bot's config)")
        action = args.action or "list"
        if action == "list" and not hasattr(args, "when"):
            args.when, args.days, args.calendar, args.search = [], 0, "", ""
        return await getattr(self, f"cal_{action}")(c, session, args)

    async def cal_calendars(self, c, session, args) -> str:
        if not c.cals:
            await c.sync()
        if not c.cals:
            raise C.CalendarError(c.error or "no calendars")
        what = {"read": "read-only", "ask": "changes after they tap ✅", "auto": "changes on your own (with ↩️ Undo)"}
        return "\n".join(f"{x.name}{' (primary)' if x.primary else ''}{' (iCal link)' if x.kind == 'ics' else ''}: "
                         f"{what[x.mode]}" for x in c.cals)

    def _range(self, words: list[str], days: int):
        today = self.domain.today()
        text = " ".join(words).strip().lower() or "today"
        if text in ("week", "this week"):
            start, n = today, 7
        elif text in ("next week",):
            start, n = today + C.timedelta(days=7 - today.weekday()), 7
        else:
            start, n = parse_day(text, today), 1
        n = days or n
        if not 1 <= n <= 62:
            raise CtlError("--days: 1 to 62")
        begin = C.local(C.datetime.combine(start, C.dtime(0, 0)))
        return begin, begin + C.timedelta(days=n)

    async def cal_list(self, c, session, args) -> str:
        start, end = self._range(args.when, args.days)
        events = await c.between(start, end, args.calendar, args.search)
        if not events:
            return f"no events {start:%a %d %b}" + (f" – {end - C.timedelta(days=1):%a %d %b}"
                                                     if (end - start).days > 1 else "")
        today = self.domain.today()
        lines = [e.line(today, show_cal=len(c.cals) > 1) for e in events[:150]]
        return "\n".join(lines + ([f"(+{len(events) - 150} more)"] if len(events) > 150 else []))

    async def cal_free(self, c, session, args) -> str:
        day = parse_day(args.day, self.domain.today())
        frm, to = (C.local(C.datetime.combine(day, _clock(x))) for x in (args.from_, args.to))
        now = C.local(C.datetime.now()).replace(second=0, microsecond=0)
        frm = max(frm, now) if day == now.date() else frm
        if to <= frm:
            return f"nothing left of {day:%a %d %b} between {args.from_} and {args.to}"
        minimum = _length(args.min)
        events = await c.between(frm, to)
        slots = c.free_slots(events, frm, to, minimum)
        busy = [e for e in c.busy(events)]
        head = f"{day:%a %d %b} {frm:%H:%M}–{to:%H:%M}: {len(busy)} event(s)"
        if not slots:
            return head + f"; no free stretch of {C.duration_text(minimum)} or more"
        return head + "; free: " + ", ".join(f"{a:%H:%M}–{b:%H:%M} ({C.duration_text((b - a).total_seconds())})"
                                              for a, b in slots)

    def _when(self, start_text: str, length: str, all_day: bool, days: int = 1):
        if all_day:
            start = parse_day(start_text, self.domain.today())
            if not 1 <= days <= 60:
                raise CtlError("--days: 1 to 60")
            return start, start + C.timedelta(days=days)
        try:
            start = parse_moment(start_text)
        except T.TaskError as e:
            raise CtlError(f"{e} (an event's start: e.g. \"thursday 14:00\", \"tomorrow 9am\", 2026-11-03 10:00)") from None
        start = C.local(start)
        if not length:
            return start, start + C.timedelta(hours=1)
        if C.re_clock(length):
            end = C.local(C.datetime.combine(start.date(), _clock(length)))
            if end <= start:
                raise CtlError(f"it would end ({length}) before it starts ({start:%H:%M})")
            return start, end
        return start, start + C.timedelta(seconds=_length(length))

    async def cal_add(self, c, session, args) -> str:
        start, end = self._when(args.start, args.length, args.all_day, args.days)
        return await c.propose_add(session, args.title.strip()[:200] or "(no title)", start, end, args.all_day,
                                   args.calendar, args.where[:300], args.note[:2000])

    async def cal_move(self, c, session, args) -> str:
        cal_id, eid = c.find(args.id)
        ev = next((e for e in c.events if (e.cal, e.id) == (cal_id, eid)), None)
        length = args.length or (C.duration_text((ev.ends() - ev.starts()).total_seconds())
                                 if ev and not ev.all_day else "")
        if ev and ev.all_day:
            raise CtlError("an all-day event: say what to change in Google Calendar, or delete it and add it again")
        start, end = self._when(args.start, length, False)
        return await c.propose_change(session, args.id, start=start, end=end)

    async def cal_edit(self, c, session, args) -> str:
        return await c.propose_change(session, args.id, title=args.title, where=args.where, note=args.note)

    async def cal_delete(self, c, session, args) -> str:
        return await c.propose_delete(session, args.id)

    # ------------------------------------------------------------------ reminders, check-backs, focus
    def _policy(self, spec: str):
        if not spec:
            return None  # the default of the routines note (or Overseer's)
        rs = self.domain.scheduler.load() or R.Routines()
        if spec.lower() not in rs.policies and R.parse_policy(spec) is None:
            raise CtlError(f"--follow-up {spec!r}: one of {', '.join(rs.policies) or '(no named ones yet)'}, "
                           "\"again after 10m, up to 3 more times\", or none")
        return rs.policy(spec)

    def _default_policy(self):
        rs = self.domain.scheduler.load()
        return rs.policy("") if rs else None

    async def remind(self, session, words: list[str], follow_up: str = "", kind: str = "remind") -> str:
        rem = self.domain.reminders
        if words[0] == "list":
            return await self.followup(session, "list", None)
        if words[0] in ("rm", "remove", "delete") and len(words) == 2:
            r = rem.remove(words[1])
            if not r:
                raise CtlError(f"no reminder {words[1]} waiting (ov followup list)")
            await self._announce(session, f"🗑 Removed: {esc(r['text'])}")
            return f"✓ removed {r['id']}"
        if len(words) < 2:
            raise CtlError(f'ov {kind} <when> "<text>"')
        when, text = self._split_when(words)
        policy = self._policy(follow_up) or self._default_policy()
        r = rem.add(when.timestamp(), text, session.id, session.project, policy=policy,
                    kind="commit" if kind == "commit" else "reminder")
        what = "🎯 I'll check in" if kind == "commit" else "⏰ Reminder set for"
        await self._announce(session, f"{what} {esc(when_text(r['at']))}: {esc(r['text'])}",
                             buttons=[[("✖ Cancel", f"fu:{r['id']}:cancel")]])
        return (f"✓ {r['id']}: {exact_text(r['at'])}; if unanswered: "
                f"{R.describe_policy(r['policy'])}")

    async def focus(self, session, words: list[str]) -> str:
        rem = self.domain.reminders
        if not words or words[0] in ("status", "show"):
            f = rem.focus()
            return (f"{f['id']}: {f['text']}, until {when_text(f['at'])}" if f else "no focus session running")
        if words[0] in ("stop", "end", "cancel"):
            f = rem.focus()
            if not f:
                raise CtlError("no focus session running")
            await rem.act(f["id"], "cancel")
            return f"✓ stopped {f['id']}"
        from ..core.autopilot import parse_duration
        secs = parse_duration(words[0] if words[0][-1:].isalpha() else words[0] + "m")
        if not secs or len(words) < 2:
            raise CtlError('ov focus <minutes, e.g. 25m> "<the step>"')
        try:
            f = await rem.start_focus(secs / 60, " ".join(words[1:]), session.id, session.project,
                                      policy=self._default_policy())
        except ValueError as e:
            raise CtlError(str(e)) from None
        return f"✓ {f['id']}: focus until {exact_text(f['at'])} (a countdown message is up; a ping at the end)"

    async def followup(self, session, action: str, rid: str | None) -> str:
        rem = self.domain.reminders
        if action == "list":
            kinds = {"reminder": "reminder", "say": "routine", "commit": "check-back", "focus": "focus"}
            lines = [f"{r['id']} waiting since {when_text(r['at'])} · {kinds.get(r.get('kind'), '')}: {r['text']}"
                     for r in rem.open()]
            lines += [f"{r['id']} at {when_text(r['at'])} · {kinds.get(r.get('kind'), '')}: {r['text']}"
                      for r in rem.pending()]
            return "\n".join(lines) or "nothing waiting or scheduled"
        if action not in ACTIONS or not rid:
            raise CtlError("ov followup list | done|later|skip|stuck|cancel <id>")
        r = await rem.act(rid, action)
        if not r:
            raise CtlError(f"no follow-up {rid} waiting (ov followup list)")
        return f"✓ {r['id']} {r['state']}" + (f", again {when_text(r['at'])}" if r["state"] == "pending" else "")

    # ------------------------------------------------------------------ media from the internet
    def _folder(self, session, keep: bool):
        workdir = self.agent.session_runner(session).workdir(session.id)
        return workdir / ("inbox" if keep else "outbox")

    async def _notice(self, session, text: str) -> None:
        """A small line in the 🌐 web topic (or quietly in the chat without topics): what was downloaded."""
        await self.notifier.send(Outgoing(text, session_id=session.id, silent=True, topic="web"))

    async def fetch(self, session, url: str, name: str = "", keep: bool = False, label: str = "") -> str:
        m = self.cfg.media
        if not m.fetch:
            raise CtlError("fetching from the internet is off (media.fetch in the config)")
        try:
            f = await media.fetch(url, self._folder(session, keep), m.max_mb * 2**20, name)
        except media.FetchError as e:
            raise CtlError(f"not fetched: {e}") from None
        except httpx.HTTPError as e:
            raise CtlError(f"not fetched: {type(e).__name__}: {str(e)[:200]}") from None
        host = urlsplit(f.url).hostname or ""
        await self._notice(session, f"🌐 {esc(label or 'Fetched')} <code>{esc(f.path.name)}</code> · "
                                    f"{f.size / 2**20:.1f} MB · from {esc(host)}")
        where = "./inbox/ (not sent: work on it, then put the result in ./outbox/)" if keep else \
            "./outbox/: it's sent with your reply"
        return f"✓ {f.path.name} ({f.kind}, {f.size / 2**20:.2f} MB) in {where}"

    async def gif(self, session, query: str, pick: int = 1) -> str:
        m = self.cfg.media
        key = os.environ.get(m.giphy_key_env, "") if m.giphy_key_env else ""
        if not key:
            raise CtlError("GIF search is off (media.giphy_key_env in the config, with the key in .env); "
                           "`ov fetch <url>` works for a GIF you found")
        try:
            results = await media.giphy_search(query, key, m.giphy_rating)
        except media.FetchError as e:
            raise CtlError(str(e)) from None
        if not results:
            raise CtlError(f"no GIFs for {query!r}")
        if not 1 <= pick <= len(results):
            raise CtlError(f"--pick 1 to {len(results)}")
        chosen = results[pick - 1]
        done = await self.fetch(session, chosen["url"], query, label=f"GIF for “{query}”:")
        others = "\n".join(f"  {i}. {r['title']}" for i, r in enumerate(results, 1) if i != pick)
        return f"{done} - “{chosen['title']}” (Powered by GIPHY)\nothers (--pick N):\n{others}"

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
