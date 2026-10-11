"""Autopilot and watches: the agent is woken by what it chose to wait for.

The agent decides what wakes it, with `ov` (ctl.py):

* `ov watch <job ids>` - woken when those jobs have ended (all of them, or as soon as one fails),
  with optional check-ins while they run (`--every 2h`);
* `ov wake in 90m` / `ov wake every 2h` - woken by a timer.

Waking itself repeatedly needs the researcher's approval: autopilot. The agent asks with
`ov auto request` (an approval card in Telegram whose limits the researcher can change), or the
researcher starts it with `/auto <goal>`. An approval has a goal, a number of wake-ups (watched jobs
ending and timers; check-ins don't count), a time limit and a check-in interval floor.

Without autopilot the agent can still wake itself once ("look at job X when it ends", "in 2h"):
a one-off. One-offs are capped per conversation, and after a few in a row without a message from
the researcher the agent has to ask for autopilot instead.

Guardrails: the wake-up budget and time limit, at most a few check-ins per job, `/auto off`,
`/stop`, cancel buttons, and the loop pauses on any failed or stopped wake-up.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING

from .config import Config
from .diagnose import extract_metrics, tqdm_cleanup
from .hpc import HPC
from .monitor import progress_log
from .notify import Notifier, Outgoing, code, esc
from .slurm import SUCCESS_STATES, Job
from .store import AutopilotRecord, JobRecord, Store, WatchRecord

if TYPE_CHECKING:
    from .agent import Agent, Session

log = logging.getLogger(__name__)

MAX_REPORT = 12_000  # chars of a finished job's report given to the agent
MAX_WATCHES = 10  # active watches and timers per conversation
WAKEUP_STEPS = [1, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]  # the approval card's +/- buttons
HOUR_STEPS = [2, 6, 12, 24, 48, 72, 96, 168, 336, 720]
CHECKIN_STEPS = [0, 15 * 60, 30 * 60, 3600, 2 * 3600, 4 * 3600, 8 * 3600]

SYSTEM_SECTION = """

## Autopilot is ON in this conversation
The researcher approved you working on your own toward this goal:
<goal>
{goal}
</goal>
Limits: at most {wakeups} wake-ups (a watched batch of jobs ending, or a timer, is one; each wake-up message says \
which one it is) until {until}; check-ins on running jobs {checkins}. Nothing runs between wake-ups, so:
- Each wake-up: read the results, update the research log, decide the next experiment, submit it, then \
`ov watch <job ids>` and end your reply. Don't wait for jobs, poll in a loop or sleep: Overseer wakes you. \
(Jobs you submit but forget to watch are watched for you, as a safety net.)
- `ov watch <ids> --every 2h` adds check-ins while they run; `ov wake in 3h "<what to look at>"` wakes you at a \
time instead; `ov watches` lists what will wake you, `ov unwatch` removes it.
- Research log: keep RESEARCH_LOG.md in the project folder on the HPC. Create it at the start (goal, baseline, \
plan) and say where it is; then add an entry per wake-up: what you changed and why, the job ids, the key \
results, the next idea. Read it first whenever you wake up.
- Change one thing at a time unless the goal says otherwise. Give every run its own name and output folder, and \
never delete or overwrite earlier results.
- Only cancel or change jobs you submitted in this conversation.
- When the goal is reached, or nothing sensible is left to try: `ov auto done "<one-line conclusion>"`. When you \
need the researcher's decision or you're stuck: `ov auto help "<question>"`, then end your reply."""


class WatchError(ValueError):
    """A watch or timer the agent may not set (the message tells it why)."""


def parse_duration(text: str) -> float | None:
    """'2h', '90m', '1.5h', '1d', '2' (hours) -> seconds; 'off'/'0' -> 0; None if not a duration."""
    t = text.strip().lower()
    if t in ("off", "none", "no", "0"):
        return 0.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(m|min|mins|minutes?|h|hr|hrs|hours?|d|days?)?", t)
    if not m:
        return None
    unit = (m.group(2) or "h")[0]
    return float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[unit]


def human_interval(seconds: float) -> str:
    seconds = max(60, round(seconds / 60) * 60)  # to the minute
    if seconds >= 86400 and seconds % 86400 == 0:
        return f"{int(seconds // 86400)}d"
    if seconds % 3600 == 0:
        return f"{int(seconds // 3600)}h"
    if seconds >= 3600:
        h, m = divmod(int(seconds) // 60, 60)
        return f"{h}h{m:02d}"
    return f"{max(1, round(seconds / 60))}m"


def _ids(jobs: list[JobRecord]) -> str:
    return ", ".join(j.job_id for j in jobs)


def _step(steps: list[float], value: float, up: bool, cap: float) -> float:
    steps = [x for x in steps if x <= cap] or [cap]
    if up:
        return next((x for x in steps if x > value), steps[-1])
    return next((x for x in reversed(steps) if x < value), steps[0])


class Autopilot:
    def __init__(self, cfg: Config, store: Store, agent: Agent, notifier: Notifier, hpc: HPC):
        self.cfg = cfg
        self.store = store
        self.agent = agent
        self.notifier = notifier
        self.hpc = hpc
        self._tasks: set[asyncio.Task] = set()
        self._outcome: dict[str, tuple[str, str]] = {}  # conversation -> (DONE | NEED_HELP, detail) from this turn
        agent.autopilot = self
        self._upgrade()

    @property
    def c(self):
        return self.cfg.agent.autopilot

    def _upgrade(self) -> None:
        """Autopilots started before watches existed: give them a time limit and watch their running jobs."""
        for a in self.store.list_autopilots(active_only=True):
            if a.expires:
                continue
            a.hours = a.hours or self.c.default_hours
            a.expires = time.time() + a.hours * 3600
            self.store.upsert_autopilot(a)
            running = [j.job_id for j in self.store.session_jobs(a.session_id) if not j.ended_at]
            if running and not self.store.watches(a.session_id):
                self.store.add_watch(WatchRecord(session_id=a.session_id, kind="jobs", jobs=" ".join(running)))

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self) -> None:
        """Wait for wake-ups in flight (tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------------ state
    def active(self, session_id: str | None) -> AutopilotRecord | None:
        a = self.store.get_autopilot(session_id) if session_id else None
        if not a or not a.active or (a.expires and time.time() > a.expires):
            return None
        return a

    def _counts(self, w: WatchRecord) -> bool:
        """Whether a watch can wake its conversation now (one-offs always; others while autopilot is on)."""
        return bool(w.oneoff or self.active(w.session_id))

    def handles(self, rec: JobRecord) -> bool:
        """A job the agent analyses itself (so no one-shot summary): its end wakes the agent, or it belongs to a
        conversation on autopilot (e.g. a run the agent cancelled)."""
        return (any(self._counts(w) for w in self.store.job_watches(rec.job_id))
                or bool(rec.by_agent and self.active(rec.session_id)))

    def system_section(self, session_id: str) -> str:
        a = self.active(session_id)
        if not a:
            return ""
        # Stable for the whole approval: the system prompt heads the provider's prompt cache.
        until = time.strftime("%a %d %b %H:%M", time.localtime(a.expires)) if a.expires else "it's turned off"
        return SYSTEM_SECTION.format(goal=a.goal or "(see the conversation)", wakeups=a.max_rounds, until=until,
                                     checkins=self._checkin_text(a))

    def _checkin_text(self, a: AutopilotRecord) -> str:
        return f"at most every {human_interval(a.checkin_every)}" if a.checkin_every else "are off"

    def _left(self, a: AutopilotRecord) -> str:
        return human_interval(max(60.0, a.expires - time.time())) if a.expires else "?"

    # ------------------------------------------------------------------ asking for approval
    def request(self, s: Session, goal: str, wakeups: int | None = None, hours: float | None = None,
                checkin: float | None = None) -> AutopilotRecord:
        """The agent asks to run on its own: an open request, shown as an approval card."""
        if self.active(s.id):
            raise WatchError("autopilot is already on in this conversation (ov auto status)")
        if not goal.strip():
            raise WatchError("give the goal: ov auto request --goal \"...\"")
        a = self.store.get_autopilot(s.id) or AutopilotRecord(session_id=s.id, created=time.time())
        a.active, a.status, a.goal, a.rounds = 0, "requested", goal.strip(), 0
        a.max_rounds = max(1, min(int(wakeups or self.c.default_wakeups), self.c.max_wakeups))
        a.hours = max(1.0, min(float(hours or self.c.default_hours), self.c.max_hours))
        a.checkin_every = self._checkin_floor(self.c.checkin_minutes * 60 if checkin is None else checkin)
        a.expires = 0
        self.store.upsert_autopilot(a)
        return a

    def _checkin_floor(self, every: float) -> float:
        return 0.0 if every <= 0 else max(every, self.c.min_checkin_minutes * 60)

    def card(self, a: AutopilotRecord) -> tuple[str, list[list[tuple[str, str]]]]:
        """The approval card: the request, and buttons to start, decline or change its limits."""
        text = (f"🤖 <b>The agent asks to work on its own</b>\nGoal: {esc(a.goal)}\n"
                f"• up to <b>{a.max_rounds}</b> wake-ups (a watched job ending, or a timer)\n"
                f"• for <b>{human_interval(a.hours * 3600)}</b>\n"
                f"• check-ins on running jobs: <b>{self._checkin_text(a)}</b>\n"
                "It can't wake itself beyond these. /auto off stops it any time.")
        sid = a.session_id
        buttons = [[("✅ Start", f"ap:ok:{sid}"), ("❌ No", f"ap:no:{sid}")],
                   [("➖ wake-ups", f"ap:w-:{sid}"), ("➕ wake-ups", f"ap:w+:{sid}")],
                   [("➖ time", f"ap:h-:{sid}"), ("➕ time", f"ap:h+:{sid}")],
                   [("➖ check-ins", f"ap:c-:{sid}"), ("➕ check-ins", f"ap:c+:{sid}")]]
        return text, buttons

    def adjust(self, session_id: str, what: str) -> AutopilotRecord | None:
        """A +/- button on the card (only while the request is open)."""
        a = self.store.get_autopilot(session_id)
        if not a or a.status != "requested":
            return None
        if what in ("w+", "w-"):
            a.max_rounds = int(_step(WAKEUP_STEPS, a.max_rounds, what == "w+", self.c.max_wakeups))
        elif what in ("h+", "h-"):
            a.hours = float(_step(HOUR_STEPS, a.hours, what == "h+", self.c.max_hours))
        elif what in ("c+", "c-"):  # more check-ins = a shorter interval
            steps = [0.0] + [x for x in CHECKIN_STEPS if x >= self.c.min_checkin_minutes * 60]
            order = [0.0] + sorted(steps[1:], reverse=True)  # off, 8h, 4h, ... 15m
            i = order.index(a.checkin_every) if a.checkin_every in order else 0
            a.checkin_every = order[min(i + 1, len(order) - 1)] if what == "c+" else order[max(i - 1, 0)]
        self.store.upsert_autopilot(a)
        return a

    def decline(self, session_id: str) -> bool:
        a = self.store.get_autopilot(session_id)
        if not a or a.status != "requested":
            return False
        a.status = "declined"
        self.store.upsert_autopilot(a)
        self.agent.add_note(session_id, "The researcher declined your autopilot request (the approval card in "
                                        "Telegram). Don't start a loop on your own; ask what they'd like instead.")
        return True

    async def approve(self, session_id: str) -> str | None:
        """✅ on the card: start with the card's limits. Returns the confirmation (HTML), or None if not open."""
        a = self.store.get_autopilot(session_id)
        s = self.agent.get(session_id)
        if not a or a.status != "requested" or not s:
            return None
        return await self.start(s, a.max_rounds, a.goal, a.checkin_every, a.hours, approved=True)

    # ------------------------------------------------------------------ turning it on and off
    async def start(self, s: Session, wakeups: int, goal: str = "", checkin_every: float | None = None,
                    hours: float | None = None, approved: bool = False) -> str:
        """Turn autopilot on (or extend it) for a conversation. With a goal (or from an approval) the agent
        starts at once. Returns the confirmation (HTML)."""
        a = self.store.get_autopilot(s.id)
        if a is None:
            a = AutopilotRecord(session_id=s.id, created=time.time(), checkin_every=self.c.checkin_minutes * 60)
        if checkin_every is not None:
            a.checkin_every = self._checkin_floor(checkin_every)
        resume = a.status == "paused" and not goal and not approved
        a.max_rounds = max(1, min(int(wakeups), self.c.max_wakeups))
        a.rounds = 0
        a.hours = max(1.0, min(float(hours or a.hours or self.c.default_hours), self.c.max_hours))
        a.expires = time.time() + a.hours * 3600
        a.active, a.status = 1, "waiting"
        a.goal = goal or a.goal
        self.store.upsert_autopilot(a)

        lines = [f"🔁 <b>Autopilot on</b> for {code(s.id)}: up to {a.max_rounds} wake-ups for "
                 f"{human_interval(a.hours * 3600)}, check-ins {self._checkin_text(a)}."]
        if a.goal:
            lines.append(f"Goal: {esc(a.goal)}")
        watches = self.store.watches(s.id)
        if goal or approved:
            lines.append(f"{esc(self.agent.backend_of(s.project))} is planning the first experiment…")
            self._spawn(self._run(s, self._kickoff_prompt(a), notice=None, kind="autopilot"))
        elif watches:
            lines.append("Waiting for: " + esc(self._describe_watches(watches)) + ".")
            self._spawn(self.check(s.id))
        else:
            lines.append("Nothing is watched yet. Reply to this message with what to try; the agent watches the "
                         "jobs it submits and continues on its own.")
        if resume:
            lines.append("(resumed after the pause)")
        lines.append("/auto shows progress · /auto off stops it · /stop stops the current run")
        return "\n".join(lines)

    def set_checkins(self, session_id: str, every: float) -> str:
        a = self.active(session_id)
        if not a:
            return "Autopilot isn't on in this conversation."
        a.checkin_every = self._checkin_floor(every)
        self.store.upsert_autopilot(a)
        for w in self.store.watches(session_id):  # watches can't check in more often than allowed now
            if w.kind == "jobs" and w.every and not w.oneoff:
                w.every = max(w.every, a.checkin_every) if a.checkin_every else 0.0
                self.store.update_watch(w)
        return (f"🔎 Check-ins at most every {human_interval(a.checkin_every)} for {code(session_id)}."
                if a.checkin_every else f"Check-ins off for {code(session_id)}.")

    def stop(self, session_id: str | None = None, status: str = "off") -> list[str]:
        """Turn autopilot off for one conversation, or all; its watches go too (one-offs stay).
        Returns the conversations stopped."""
        targets = [self.store.get_autopilot(session_id)] if session_id else self.store.list_autopilots(True)
        stopped = []
        for a in targets:
            if a and (a.active or a.status == "requested"):
                a.active, a.status = 0, status
                self.store.upsert_autopilot(a)
                stopped.append(a.session_id)
                for w in self.store.watches(a.session_id):
                    if not w.oneoff:
                        w.state = "cancelled"
                        self.store.update_watch(w)
        return stopped

    def finish(self, session_id: str, detail: str) -> AutopilotRecord:
        """`ov auto done`: the goal is reached (announced with the agent's reply)."""
        a = self.active(session_id)
        if not a:
            raise WatchError("autopilot isn't on in this conversation")
        self.stop(session_id, "done")
        self._outcome[session_id] = ("DONE", detail)
        return a

    def need_help(self, session_id: str, question: str) -> None:
        """`ov auto help`: the researcher is asked (with a notification); watches stay."""
        a = self.active(session_id)
        if not a:
            raise WatchError("autopilot isn't on in this conversation: just ask in your reply")
        a.status = "need_help"
        self.store.upsert_autopilot(a)
        self._outcome[session_id] = ("NEED_HELP", question)

    def describe(self) -> str:
        rows = [a for a in self.store.list_autopilots() if a.status != "declined"][:8]
        oneoffs = [w for w in self.store.watches() if w.oneoff]
        if not rows and not oneoffs:
            return ("Autopilot is off everywhere.\n\nThe agent asks for it when a task needs several runs (an approval "
                    "card), or start it yourself with /auto [wake-ups] [every 2h] [for 48h] &lt;goal&gt; - as a reply "
                    "to a job or agent message to use that conversation.")
        lines = []
        for a in rows:
            mark = "▶️" if self.active(a.session_id) else "⏹"
            bits = [f"{mark} {code(a.session_id)} · {esc(a.status or '-')} · wake-up {a.rounds}/{a.max_rounds}"]
            if self.active(a.session_id):
                bits.append(f"{self._left(a)} left")
                if a.checkin_every:
                    bits.append(f"check-ins ≥ {human_interval(a.checkin_every)}")
                watches = self.store.watches(a.session_id)
                if watches:
                    bits.append("waiting for " + esc(self._describe_watches(watches)))
            lines.append(" · ".join(bits) + (f"\n    goal: {esc(a.goal[:200])}" if a.goal else ""))
        if oneoffs:
            lines.append("⏰ one-off wake-ups: " + "; ".join(
                f"{code(w.session_id)} {esc(self._describe_watches([w]))}" for w in oneoffs))
        return "\n".join(lines)

    # ------------------------------------------------------------------ watches and timers
    def _describe_watches(self, watches: list[WatchRecord]) -> str:
        parts = []
        now = time.time()
        for w in watches:
            if w.kind == "timer":
                p = f"⏰ in {human_interval(max(60.0, w.due - now))}"
                if w.every:
                    p += f", then every {human_interval(w.every)}"
            else:
                p = f"job{'s' if len(w.job_ids) > 1 else ''} {', '.join(w.job_ids)}"
                if w.every:
                    p += f" (check-ins every {human_interval(w.every)})"
            parts.append(p)
        return "; ".join(parts)

    def watch_lines(self, session_id: str) -> str:
        """What will wake the conversation, for the agent (`ov watches`)."""
        ws = self.store.watches(session_id)
        if not ws:
            return "Nothing will wake you: no watches or timers."
        out = []
        for w in ws:
            what = self._describe_watches([w])
            out.append(f"w{w.id}  {what}" + (f"  - {w.note}" if w.note else "") + ("  [one-off]" if w.oneoff else ""))
        return "\n".join(out)

    def _oneoff_check(self, s: Session) -> None:
        pending = [w for w in self.store.watches(s.id) if w.oneoff]
        if len(pending) >= self.c.oneoff_max_pending:
            raise WatchError(f"you already have {len(pending)} one-off wake-ups pending (the limit without autopilot): "
                             "ov watches / ov unwatch, or ask for autopilot (ov auto request)")
        chain = int(self.store.get(f"oneoff_chain:{s.id}") or 0)
        if chain >= self.c.oneoff_chain:
            raise WatchError(f"you've woken yourself {chain} times in a row without the researcher writing. To keep "
                             "going on your own, ask for autopilot: ov auto request --goal \"...\"")

    def add_watch(self, s: Session, job_ids: list[str], every: float = 0.0, note: str = "") -> tuple[WatchRecord, str]:
        """Wake the conversation when these jobs have ended. Returns the watch and a remark for the agent."""
        a = self.active(s.id)
        remark = ""
        if a is None:
            if every:
                raise WatchError("check-ins need autopilot (ov auto request). Without it you can be woken once, when "
                                 "the jobs end: drop --every")
            self._oneoff_check(s)
        elif every:
            if not a.checkin_every:
                raise WatchError("check-ins are off for this autopilot (the researcher's choice): drop --every")
            if every < a.checkin_every:
                remark = f" (check-ins at most every {human_interval(a.checkin_every)}: that's this autopilot's limit)"
            every = max(every, a.checkin_every)
        mine = self.store.watches(s.id)
        if len(mine) >= MAX_WATCHES:
            raise WatchError(f"{len(mine)} watches and timers are pending (the limit): ov watches / ov unwatch")
        for w in mine:  # the latest grouping wins: a job is in one watch of this conversation
            if w.kind == "jobs" and set(w.job_ids) & set(job_ids):
                rest = [j for j in w.job_ids if j not in job_ids]
                w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
                self.store.update_watch(w)
        w = self.store.add_watch(WatchRecord(session_id=s.id, kind="jobs", jobs=" ".join(job_ids), every=every,
                                             note=note.strip(), oneoff=int(a is None)))
        return w, remark

    def add_timer(self, s: Session, delay: float, every: float = 0.0, note: str = "") -> WatchRecord:
        a = self.active(s.id)
        if delay < 60:
            raise WatchError("the shortest timer is 1 minute")
        if a is None:
            if every:
                raise WatchError("repeating wake-ups need autopilot (ov auto request); `ov wake in` works once")
            if delay > self.c.oneoff_max_hours * 3600:
                raise WatchError(f"one-off wake-ups can be at most {human_interval(self.c.oneoff_max_hours * 3600)} ahead")
            self._oneoff_check(s)
        else:
            if a.expires and time.time() + delay > a.expires:
                raise WatchError(f"that's after this autopilot ends (in {self._left(a)})")
            if every and every < self.c.min_checkin_minutes * 60:
                raise WatchError(f"repeat at most every {self.c.min_checkin_minutes} minutes")
        if len(self.store.watches(s.id)) >= MAX_WATCHES:
            raise WatchError(f"{MAX_WATCHES} watches and timers are pending (the limit): ov watches / ov unwatch")
        return self.store.add_watch(WatchRecord(session_id=s.id, kind="timer", due=time.time() + delay, every=every,
                                                note=note.strip(), oneoff=int(a is None)))

    def unwatch(self, session_id: str, ref: str) -> list[WatchRecord]:
        """Remove a watch (w12), a job from the watches (its id), or everything (all)."""
        changed = []
        for w in self.store.watches(session_id):
            if ref == "all" or ref.lower() == f"w{w.id}":
                w.state = "cancelled"
            elif w.kind == "jobs" and ref in w.job_ids:
                rest = [j for j in w.job_ids if j != ref]
                w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
            else:
                continue
            self.store.update_watch(w)
            changed.append(w)
        return changed

    def cancel_watch(self, wid: int) -> WatchRecord | None:
        """The researcher's ✖ button on a one-off wake-up."""
        w = self.store.get_watch(wid)
        if not w or w.state != "active":
            return None
        w.state = "cancelled"
        self.store.update_watch(w)
        self.agent.add_note(w.session_id, f"The researcher cancelled your wake-up w{w.id} "
                                          f"({self._describe_watches([w])}).")
        return w

    def forget_job(self, job_id: str) -> None:
        """The job isn't tracked any more (ov job delete): it wakes nobody."""
        for w in self.store.job_watches(job_id):
            rest = [j for j in w.job_ids if j != job_id]
            w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
            self.store.update_watch(w)

    # ------------------------------------------------------------------ events from the monitor and agent
    def user_turn(self, s: Session) -> None:
        """The researcher wrote in this conversation: one-off wake-ups may start a new chain."""
        if self.store.get(f"oneoff_chain:{s.id}") not in (None, "0"):
            self.store.set(f"oneoff_chain:{s.id}", "0")

    async def job_finished(self, rec: JobRecord, report: str) -> None:
        watches = [w for w in self.store.job_watches(rec.job_id) if self._counts(w)]
        if not watches:
            return
        rec = self.store.get_job(rec.job_id) or rec
        rec.report = report[-MAX_REPORT:]
        self.store.upsert_job(rec)
        for sid in dict.fromkeys(w.session_id for w in watches):
            if rec.state in SUCCESS_STATES:
                waiting = [j for w in watches if w.session_id == sid for j in w.job_ids
                           if j != rec.job_id and (r := self.store.get_job(j)) and not r.ended_at]
                if waiting:
                    await self.notifier.send(Outgoing(
                        f"👁 Job {esc(rec.job_id)} finished; waking the agent when {esc(', '.join(waiting))} "
                        f"{'have' if len(waiting) > 1 else 'has'} finished too.", session_id=sid, silent=True))
            await self.check(sid)

    async def check(self, session_id: str) -> None:
        """Wake the conversation if watches or timers call for it (all that are due, in one turn)."""
        s = self.agent.get(session_id)
        if not s or self.agent.is_running(s):
            return  # a running turn re-checks when it ends (turn_ended)
        a = self.active(session_id)
        now = time.time()
        fired: list[tuple[WatchRecord, list[JobRecord], list[JobRecord]]] = []  # (watch, ended, still running)
        for w in self.store.watches(session_id):
            if not (w.oneoff or a):
                continue
            if w.kind == "timer":
                if w.due <= now:
                    fired.append((w, [], []))
                continue
            recs = [r for j in w.job_ids if (r := self.store.get_job(j)) and not r.ignored]
            ended = [r for r in recs if r.ended_at]
            running = [r for r in recs if not r.ended_at]
            if not recs:
                w.state = "cancelled"
                self.store.update_watch(w)
            elif not running or any(r.state not in SUCCESS_STATES for r in ended):
                fired.append((w, ended, running))
        if not fired:
            return
        autopilot = any(not w.oneoff for w, _, _ in fired)
        if autopilot:
            if a.rounds >= a.max_rounds:
                self.stop(session_id, "limit")
                ended = [r for _, e, _ in fired for r in e]
                await self.notifier.send(Outgoing(
                    f"🔁 <b>Autopilot stopped</b>: it used all {a.max_rounds} wake-ups"
                    + (f". Job(s) {esc(_ids(ended))} finished (reports above)" if ended else "")
                    + ". Reply to this message to continue by hand, or send /auto &lt;wake-ups&gt; as a reply to "
                      "give it more.", session_id=session_id))
                return
            a.rounds += 1
            a.status = "working"
            self.store.upsert_autopilot(a)
        else:
            chain = int(self.store.get(f"oneoff_chain:{session_id}") or 0)
            self.store.set(f"oneoff_chain:{session_id}", str(chain + 1))
        before: list[tuple] = [(w.id, w.state, w.due, w.fired) for w, _, _ in fired]
        for w, ended, running in fired:
            w.fired = now
            if w.kind == "timer" and w.every and not w.oneoff:
                w.due = now + w.every  # repeating: stays active
            else:
                w.state = "fired"
            self.store.update_watch(w)
            if running:  # woken early because one failed: the rest stay watched
                rest = self.store.add_watch(WatchRecord(session_id=session_id, kind="jobs", every=w.every, note=w.note,
                                                        jobs=" ".join(r.job_id for r in running), oneoff=w.oneoff))
                before.append((rest.id, "cancelled", rest.due, rest.fired))  # undone if this wake-up fails
        prompt, notice = self._wake_prompt(a if autopilot else None, fired)
        self._spawn(self._run(s, prompt, notice, kind="autopilot" if autopilot else "oneoff", undo=before))

    def turn_ended(self, s: Session) -> None:
        """After any turn: jobs may have ended or timers come due while it ran."""
        if self.store.watches(s.id):
            async def recheck():
                async with s.lock:
                    pass
                await self.check(s.id)
            self._spawn(recheck())

    def after_turn(self, s: Session, cancelled: list[str], started: float) -> tuple[str, str | None]:
        """After a successful turn: (a line for the reply (HTML), DONE | NEED_HELP | None)."""
        for jid in cancelled:  # the agent knows why it cancelled these: their end wakes nobody
            for w in self.store.job_watches(jid):
                if w.session_id == s.id:
                    rest = [j for j in w.job_ids if j != jid]
                    w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
                    self.store.update_watch(w)
        outcome = self._outcome.pop(s.id, None)
        if outcome and outcome[0] == "DONE":
            a = self.store.get_autopilot(s.id)
            used = f" after {a.rounds} wake-up(s)" if a else ""
            return f"🏁 <b>Autopilot finished</b>{used}: {esc(outcome[1] or 'goal reached')}", "DONE"
        a = self.active(s.id)
        if not a:
            return "", None
        # Safety net: jobs submitted in this turn that the agent didn't watch.
        watched = {j for w in self.store.watches(s.id) if w.kind == "jobs" for j in w.job_ids}
        new = [j for j in self.store.session_jobs(s.id) if not j.ended_at and j.job_id not in watched
               and j.first_seen >= started]
        extra = ""
        if new:
            self.store.add_watch(WatchRecord(session_id=s.id, kind="jobs", jobs=" ".join(j.job_id for j in new)))
            extra = f" (auto-watched {esc(_ids(new))}: the agent didn't watch {'them' if len(new) > 1 else 'it'})"
        if outcome and outcome[0] == "NEED_HELP":
            return (f"🙋 <b>Autopilot needs you</b>: {esc(outcome[1] or 'see above')}\nReply to this message to "
                    "answer." + extra, "NEED_HELP")
        watches = self.store.watches(s.id)
        if watches:
            a.status = "waiting"
            self.store.upsert_autopilot(a)
            return (f"🔁 Autopilot: waiting for {esc(self._describe_watches(watches))}{extra} · wake-up "
                    f"{a.rounds}/{a.max_rounds} used · {self._left(a)} left"), None
        a.status = "idle"
        self.store.upsert_autopilot(a)
        return ("⚠️ Autopilot: nothing is watched, so nothing will wake the agent. Reply to this message to continue, "
                "or /auto off."), None

    # ------------------------------------------------------------------ the minute tick: expiry, timers, check-ins
    async def tick(self, live: dict[str, Job]) -> None:
        """Called by the monitor after each poll."""
        now = time.time()
        for a in self.store.list_autopilots(active_only=True):
            if a.expires and now > a.expires:
                s = self.agent.get(a.session_id)
                if s and self.agent.is_running(s):
                    continue  # ends after the current turn
                self.stop(a.session_id, "expired")
                await self.notifier.send(Outgoing(
                    f"⌛ <b>Autopilot ended</b>: its {human_interval(a.hours * 3600)} are up, after {a.rounds} "
                    f"wake-up(s). Jobs still running are reported as usual. Reply to continue by hand, or /auto "
                    "&lt;wake-ups&gt; as a reply to give it more time.", session_id=a.session_id))
        for sid in dict.fromkeys(w.session_id for w in self.store.watches() if w.kind == "timer" and w.due <= now):
            await self.check(sid)
        await self._checkins(live, now)

    async def _checkins(self, live: dict[str, Job], now: float) -> None:
        by_session: dict[str, list[tuple[JobRecord, Job, float]]] = {}
        for w in self.store.watches():
            if w.kind != "jobs" or not w.every or w.oneoff:
                continue
            a = self.active(w.session_id)
            if not a or not a.checkin_every:
                continue
            every = max(w.every, a.checkin_every)
            for jid in w.job_ids:
                j = self.store.get_job(jid)
                if not j or j.ended_at or jid not in live or live[jid].state != "RUNNING":
                    continue
                if not j.last_checkin:  # the clock starts when the job starts running
                    j.last_checkin = now
                    self.store.upsert_job(j)
                by_session.setdefault(w.session_id, []).append((j, live[jid], every))
        cap = self.c.max_checkins_per_job
        for sid, items in by_session.items():
            s = self.agent.get(sid)
            if not s or self.agent.is_running(s):
                continue
            if not any(j.checkins < cap and now - j.last_checkin >= every for j, _, every in items):
                continue
            # One check-in covers every running job that is at least halfway to its next one.
            due = [(j, job) for j, job, every in items if j.checkins < cap and now - j.last_checkin >= every / 2]
            for j, _ in due:
                j.checkins += 1
                j.last_checkin = now
                self.store.upsert_job(j)
            self._spawn(self._checkin(s, due))

    async def _checkin(self, s: Session, due: list[tuple[JobRecord, Job]]) -> None:
        parts = []
        for rec, job in due:
            tail = tqdm_cleanup(await self.hpc.try_tail(rec.stdout, 40)) if rec.stdout else ""
            log_path = await progress_log(self.hpc, rec)
            if log_path:  # the program's own log, where the training progress is
                self.store.upsert_job(rec)
                tail += f"\n--- {log_path} ---\n" + tqdm_cleanup(await self.hpc.try_tail(log_path, 80))
            err = ""
            if rec.stderr and rec.stderr != rec.stdout:
                err = tqdm_cleanup(await self.hpc.try_tail(rec.stderr, 30))
            metrics = extract_metrics(tail)
            parts.append("\n".join(x for x in [
                f"<job id='{rec.job_id}' name='{rec.name}' state='RUNNING' elapsed='{job.elapsed}' "
                f"time_limit='{job.time_limit}' node='{job.reason}' stdout='{rec.stdout}' stderr='{rec.stderr}'>",
                f"latest metrics: {metrics}" if metrics else "",
                f"<stdout_tail>\n{tail[-6000:]}\n</stdout_tail>" if tail.strip() else "(no output yet)",
                f"<stderr_tail>\n{err[-3000:]}\n</stderr_tail>" if err.strip() else "",
                f"(check-in {rec.checkins} of at most {self.c.max_checkins_per_job} for this job)",
                "</job>",
            ] if x))
        prompt = (
            "[Autopilot check-in - not a wake-up; these jobs are still running]\n\n" + "\n\n".join(parts) + "\n\n"
            "Check whether they are healthy (read more of the logs or metrics files if you need to). If they are, "
            "change nothing: reply in one or two sentences. If a run is clearly broken, diverging or wasting GPU "
            "time, scancel it, note why in the research log, fix and resubmit, and `ov watch` the new job.")
        elapsed = ", ".join(f"{rec.job_id} running {job.elapsed}" for rec, job in due)
        await self._run(s, prompt, notice=f"🔎 Autopilot check-in: {esc(elapsed)}", kind="checkin", quiet=True)

    # ------------------------------------------------------------------ running a wake-up
    async def _run(self, s: Session, prompt: str, notice: str | None, kind: str,
                   undo: list[tuple] | None = None, quiet: bool = False) -> None:
        if notice:
            await self.notifier.send(Outgoing(notice, session_id=s.id, silent=quiet))
        ok = await self.agent.handle(s, prompt, quiet=quiet, origin=kind)
        if ok or kind == "checkin":
            return
        if kind == "oneoff":
            await self.notifier.send(Outgoing("⚠️ The wake-up the agent asked for didn't complete (see above). Reply to "
                                              "this message to continue.", session_id=s.id))
            return
        a = self.active(s.id)
        if not a:
            return
        # A failed or stopped wake-up pauses the loop: it never keeps going unattended after a problem.
        # Updated before any await, so the re-check after the turn sees it paused. The watches that fired
        # are put back, so resuming (/auto as a reply) retries this wake-up.
        a.active, a.status = 0, "paused"
        if undo:
            a.rounds = max(0, a.rounds - 1)
            for wid, state, due, fired in undo:
                w = self.store.get_watch(wid)
                if w:
                    w.state, w.due, w.fired = state, due, fired
                    self.store.update_watch(w)
        self.store.upsert_autopilot(a)
        await self.notifier.send(Outgoing(
            "⏸ <b>Autopilot paused</b>: that run didn't complete (see above). Send /auto as a reply to this message to "
            "retry, or reply to continue by hand.", session_id=s.id))

    # ------------------------------------------------------------------ prompts
    def _kickoff_prompt(self, a: AutopilotRecord) -> str:
        return (f"[Autopilot started: up to {a.max_rounds} wake-ups for {human_interval(a.hours * 3600)}]\n\n"
                f"Goal: {a.goal}\n\n"
                "Look at the project and any earlier results, create the research log, plan the first experiment, "
                "submit it, `ov watch` it, and end your reply.")

    def _wake_prompt(self, a: AutopilotRecord | None,
                     fired: list[tuple[WatchRecord, list[JobRecord], list[JobRecord]]]) -> tuple[str, str]:
        ended = list({r.job_id: r for _, e, _ in fired for r in e}.values())
        running = list({r.job_id: r for _, _, rn in fired for r in rn}.values())
        timers = [w for w, _, _ in fired if w.kind == "timer"]
        notes = [f"w{w.id}: {w.note}" for w, _, _ in fired if w.note and w.kind == "jobs"]
        head = (f"[Autopilot wake-up {a.rounds} of {a.max_rounds} · {self._left(a)} left]" if a else
                "[The wake-up you asked for (one-off)]")
        parts = [head]
        if ended:
            parts.append("Job(s) you watched have finished:")
            parts.append("\n\n".join(
                f"<job id='{j.job_id}' name='{j.name}' state='{j.state}' stdout='{j.stdout}' stderr='{j.stderr}'>\n"
                f"{(j.report or '(no final report: read its logs)').strip()}\n</job>" for j in ended))
        for w in timers:
            ago = human_interval(max(60.0, time.time() - w.created))
            parts.append(f"Your timer w{w.id} (set {ago} ago): {w.note or '(no note)'}")
        if notes:
            parts.append("Your notes for these: " + "; ".join(notes))
        if running:
            parts.append(f"Still running: {_ids(running)} (still watched; one of the jobs failed, which is why "
                         "you're woken now).")
        if a and a.rounds >= a.max_rounds:
            parts.append("This is the LAST wake-up: don't start new experiments. Write the final summary in the research "
                         "log and in your reply, and run `ov auto done \"<conclusion>\"`.")
        elif a:
            parts.append("Read the results (open the logs or metrics files for more), update the research log, then "
                         "submit the next experiment and `ov watch` it - or `ov auto done` / `ov auto help`.")
        else:
            parts.append("Look at it now and tell the researcher what you found. If something needs following up, "
                         "say so in your reply.")
        what = ", ".join(f"{j.job_id} {j.state}" for j in ended) or ", ".join(
            f"timer{(': ' + w.note[:80]) if w.note else ''}" for w in timers)
        if a:
            notice = (f"🔁 <b>Autopilot wake-up {a.rounds}/{a.max_rounds}</b>: {esc(what)}. "
                      f"{esc(self.agent.backend_of(self.agent.get(a.session_id).project))} is analysing…")
        else:
            notice = f"⏰ <b>Waking the agent</b> as it asked: {esc(what)}"
        return "\n\n".join(parts), notice
