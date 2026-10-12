"""Wake-ups: the agent decides what wakes it, without asking.

With `ov` (ctl.py) the agent sets up what wakes it:

* `ov watch <job ids>` - woken when those jobs have ended (all of them, or as soon as one fails), with
  their final reports; `--every 30m` adds check-ins while they run;
* `ov wake in 90m` - woken once, later; `ov wake every 2h [--for 12h]` - repeatedly, until a set end.

That is how it works on its own over many runs ("auto research"): submit, watch, read the results when
woken, submit the next run. Each watch is announced in the conversation's topic with a cancel button.
Guardrails: check-ins and repeating timers at least `agent.autopilot.min_checkin_minutes` apart,
repeating timers end (by default after `repeat_hours`), `/auto off` and `/stop`, and a wake-up that
fails pauses the conversation's wake-ups until the researcher writes there.

In projects the agent keeps a research log (RESEARCH_LOG.md in the project folder); its beginning and
latest entries are shown to the agent whenever it is woken.
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
from .store import JobRecord, Store, WatchRecord

if TYPE_CHECKING:
    from .agent import Agent, Session

log = logging.getLogger(__name__)

MAX_REPORT = 12_000  # chars of a finished job's report given to the agent


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


class Autopilot:
    """The conversations' watches and timers, and the wake-ups they cause."""

    def __init__(self, cfg: Config, store: Store, agent: Agent, notifier: Notifier, hpc: HPC):
        self.cfg = cfg
        self.store = store
        self.agent = agent
        self.notifier = notifier
        self.hpc = hpc
        self._tasks: set[asyncio.Task] = set()
        agent.autopilot = self

    @property
    def c(self):
        return self.cfg.agent.autopilot

    @property
    def min_interval(self) -> float:
        return self.c.min_checkin_minutes * 60

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
    def paused(self, session_id: str) -> bool:
        return bool(self.store.get(f"wakeups_paused:{session_id}"))

    def handles(self, rec: JobRecord) -> bool:
        """A job whose end wakes the agent (so the agent analyses it, not a one-shot summary)."""
        return bool(self.store.job_watches(rec.job_id))

    def describe_watches(self, watches: list[WatchRecord]) -> str:
        parts = []
        now = time.time()
        for w in watches:
            if w.kind == "timer":
                p = f"⏰ in {human_interval(max(60.0, w.due - now))}"
                if w.every:
                    p += f", then every {human_interval(w.every)}"
                    if w.until:
                        p += f" for {human_interval(max(60.0, w.until - now))}"
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
        lines = [f"w{w.id}  {self.describe_watches([w])}" + (f"  - {w.note}" if w.note else "") for w in ws]
        if self.paused(session_id):
            lines.append("(paused after a failed wake-up: they resume when the researcher writes to you)")
        return "\n".join(lines)

    def describe(self, session_ids: list[str] | None = None) -> str:
        """Pending wake-ups, by conversation (HTML, for /auto)."""
        by_session: dict[str, list[WatchRecord]] = {}
        for w in self.store.watches():
            if session_ids is None or w.session_id in session_ids:
                by_session.setdefault(w.session_id, []).append(w)
        if not by_session:
            return ("Nothing will wake the agent: no watches or timers.\n\nThe agent sets them itself (ov watch / "
                    "ov wake) when it waits for results. /auto &lt;goal&gt; asks it to work on something on its own.")
        lines = []
        for sid, ws in by_session.items():
            sess = self.store.get_session(sid)
            where = f" · 📁 {esc(sess.project)}" if sess and sess.project else ""
            paused = " · ⏸ paused (write to it to resume)" if self.paused(sid) else ""
            lines.append(f"{code(sid)}{where}{paused}\n    " + "\n    ".join(
                f"w{w.id} {esc(self.describe_watches([w]))}" + (f" — {esc(w.note[:120])}" if w.note else "")
                for w in ws))
        return "⏰ <b>Pending wake-ups</b>\n" + "\n".join(lines) + "\n\n/auto off cancels them"

    # ------------------------------------------------------------------ watches and timers
    def add_watch(self, s: Session, job_ids: list[str], every: float = 0.0, note: str = "") -> tuple[WatchRecord, str]:
        """Wake the conversation when these jobs have ended. Returns the watch and a remark for the agent."""
        remark = ""
        if every and every < self.min_interval:
            remark = f" (check-ins are at least {human_interval(self.min_interval)} apart)"
            every = self.min_interval
        for w in self.store.watches(s.id):  # the latest grouping wins: a job is in one watch of this conversation
            if w.kind == "jobs" and set(w.job_ids) & set(job_ids):
                rest = [j for j in w.job_ids if j not in job_ids]
                w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
                self.store.update_watch(w)
        w = self.store.add_watch(WatchRecord(session_id=s.id, kind="jobs", jobs=" ".join(job_ids), every=every,
                                             note=note.strip()))
        return w, remark

    def add_timer(self, s: Session, delay: float, every: float = 0.0, note: str = "",
                  hours: float | None = None) -> tuple[WatchRecord, str]:
        if delay < 60:
            raise WatchError("the shortest timer is 1 minute")
        remark = ""
        if every and every < self.min_interval:
            remark = f" (repeating wake-ups are at least {human_interval(self.min_interval)} apart)"
            every = delay = max(delay, self.min_interval)
        until = time.time() + (hours if hours else self.c.repeat_hours) * 3600 if every else 0.0
        w = self.store.add_watch(WatchRecord(session_id=s.id, kind="timer", due=time.time() + delay, every=every,
                                             note=note.strip(), until=until))
        return w, remark

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
        """The researcher's ✖ button on a watch's announcement."""
        w = self.store.get_watch(wid)
        if not w or w.state != "active":
            return None
        w.state = "cancelled"
        self.store.update_watch(w)
        self.agent.add_note(w.session_id, f"The researcher cancelled your wake-up w{w.id} ({self.describe_watches([w])}).")
        return w

    def cancel_all(self, session_ids: list[str] | None = None) -> list[WatchRecord]:
        """/auto off: cancel the pending wake-ups of these conversations (or all). The agents are told."""
        cancelled = []
        for w in self.store.watches():
            if session_ids is None or w.session_id in session_ids:
                w.state = "cancelled"
                self.store.update_watch(w)
                cancelled.append(w)
        for sid in dict.fromkeys(w.session_id for w in cancelled):
            self.agent.add_note(sid, "The researcher cancelled all your pending wake-ups (/auto off). Don't set new "
                                     "ones unless they ask you to continue.")
        return cancelled

    def forget_job(self, job_id: str) -> None:
        """The job isn't tracked any more (ov job delete): it wakes nobody."""
        for w in self.store.job_watches(job_id):
            rest = [j for j in w.job_ids if j != job_id]
            w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
            self.store.update_watch(w)

    # ------------------------------------------------------------------ events from the monitor and agent
    def user_turn(self, s: Session) -> None:
        """The researcher wrote in this conversation: wake-ups paused after a failure resume (after this turn)."""
        if self.paused(s.id):
            self.store.set(f"wakeups_paused:{s.id}", "")

    async def job_finished(self, rec: JobRecord, report: str) -> None:
        watches = self.store.job_watches(rec.job_id)
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
        if not s or self.agent.is_running(s) or self.paused(session_id):
            return  # a running turn re-checks when it ends (turn_ended); a paused one when the researcher writes
        now = time.time()
        fired: list[tuple[WatchRecord, list[JobRecord], list[JobRecord]]] = []  # (watch, ended, still running)
        for w in self.store.watches(session_id):
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
        undo: list[tuple] = [(w.id, w.state, w.due, w.fired) for w, _, _ in fired]
        for w, ended, running in fired:
            w.fired = now
            if w.kind == "timer" and w.every and (not w.until or now + w.every <= w.until):
                w.due = now + w.every  # repeating: stays active until its end
            else:
                w.state = "fired"
            self.store.update_watch(w)
            if running:  # woken early because one failed: the rest stay watched
                rest = self.store.add_watch(WatchRecord(session_id=session_id, kind="jobs", every=w.every, note=w.note,
                                                        jobs=" ".join(r.job_id for r in running)))
                undo.append((rest.id, "cancelled", rest.due, rest.fired))  # undone if this wake-up fails
        prompt = await self._wake_prompt(s, fired)
        what = ", ".join(f"{r.job_id} {r.state}" for _, e, _ in fired for r in e) or ", ".join(
            f"timer{(': ' + w.note[:80]) if w.note else ''}" for w, _, _ in fired if w.kind == "timer")
        self._spawn(self._run(s, prompt, f"⏰ <b>Waking the agent</b>: {esc(what)}", "wakeup", undo=undo))

    def turn_ended(self, s: Session) -> None:
        """After any turn: jobs may have ended or timers come due while it ran."""
        if self.store.watches(s.id):
            async def recheck():
                async with s.lock:
                    pass
                await self.check(s.id)
            self._spawn(recheck())

    def after_turn(self, s: Session, cancelled: list[str]) -> str:
        """After a successful turn: a line for the reply (HTML) saying what will wake the agent, if anything."""
        for jid in cancelled:  # the agent knows why it cancelled these: their end wakes nobody
            for w in self.store.job_watches(jid):
                if w.session_id == s.id:
                    rest = [j for j in w.job_ids if j != jid]
                    w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
                    self.store.update_watch(w)
        watches = self.store.watches(s.id)
        return f"⏰ Waiting for: {esc(self.describe_watches(watches))}" if watches else ""

    # ------------------------------------------------------------------ the minute tick: timers, check-ins
    async def tick(self, live: dict[str, Job]) -> None:
        """Called by the monitor after each poll."""
        now = time.time()
        for sid in dict.fromkeys(w.session_id for w in self.store.watches() if w.kind == "timer" and w.due <= now):
            await self.check(sid)
        await self._checkins(live, now)

    async def _checkins(self, live: dict[str, Job], now: float) -> None:
        by_session: dict[str, list[tuple[JobRecord, Job, float]]] = {}
        for w in self.store.watches():
            if w.kind != "jobs" or not w.every or self.paused(w.session_id):
                continue
            every = max(w.every, self.min_interval)
            for jid in w.job_ids:
                j = self.store.get_job(jid)
                if not j or j.ended_at or jid not in live or live[jid].state != "RUNNING":
                    continue
                if not j.last_checkin:  # the clock starts when the job starts running
                    j.last_checkin = now
                    self.store.upsert_job(j)
                by_session.setdefault(w.session_id, []).append((j, live[jid], every))
        for sid, items in by_session.items():
            s = self.agent.get(sid)
            if not s or self.agent.is_running(s):
                continue
            if not any(now - j.last_checkin >= every for j, _, every in items):
                continue
            # One check-in covers every running job that is at least halfway to its next one.
            due = [(j, job) for j, job, every in items if now - j.last_checkin >= every / 2]
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
            tasks = job.extra.get("tasks")
            parts.append("\n".join(x for x in [
                f"<job id='{rec.job_id}' name='{rec.name}' state='RUNNING' elapsed='{job.elapsed}' "
                f"time_limit='{job.time_limit}' node='{job.reason}' stdout='{rec.stdout}' stderr='{rec.stderr}'>",
                f"job array tasks: {tasks}" if tasks else "",
                f"latest metrics: {metrics}" if metrics else "",
                f"<stdout_tail>\n{tail[-6000:]}\n</stdout_tail>" if tail.strip() else "(no output yet)",
                f"<stderr_tail>\n{err[-3000:]}\n</stderr_tail>" if err.strip() else "",
                "</job>",
            ] if x))
        prompt = (
            "[Check-in you asked for - these jobs are still running]\n\n" + "\n\n".join(parts) + "\n\n"
            "Check whether they are healthy (read more of the logs or metrics files if you need to). If they are, "
            "change nothing: reply in one or two sentences. If a run is clearly broken, diverging or wasting GPU "
            "time, scancel it, note why in the research log, fix and resubmit, and `ov watch` the new job.")
        elapsed = ", ".join(f"{rec.job_id} running {job.elapsed}" for rec, job in due)
        await self._run(s, prompt, notice=f"🔎 Check-in: {esc(elapsed)}", kind="checkin", quiet=True)

    # ------------------------------------------------------------------ running a wake-up
    async def _run(self, s: Session, prompt: str, notice: str | None, kind: str,
                   undo: list[tuple] | None = None, quiet: bool = False) -> None:
        if notice:
            await self.notifier.send(Outgoing(notice, session_id=s.id, silent=quiet))
        ok = await self.agent.handle(s, prompt, quiet=quiet, origin=kind)
        if ok:
            return
        # A failed or stopped wake-up pauses this conversation's wake-ups: nothing keeps going unattended after a
        # problem. The watches that fired are put back, so they fire again once the researcher writes here.
        self.store.set(f"wakeups_paused:{s.id}", "1")
        for wid, state, due, fired in undo or []:
            w = self.store.get_watch(wid)
            if w:
                w.state, w.due, w.fired = state, due, fired
                self.store.update_watch(w)
        await self.notifier.send(Outgoing(
            "⏸ <b>Wake-ups paused</b> for this conversation: that run didn't complete (see above). They resume when "
            "you write to the agent here; /auto off cancels them.", session_id=s.id))

    # ------------------------------------------------------------------ prompts
    async def _wake_prompt(self, s: Session,
                           fired: list[tuple[WatchRecord, list[JobRecord], list[JobRecord]]]) -> str:
        ended = list({r.job_id: r for _, e, _ in fired for r in e}.values())
        running = list({r.job_id: r for _, _, rn in fired for r in rn}.values())
        timers = [w for w, _, _ in fired if w.kind == "timer"]
        notes = [f"w{w.id}: {w.note}" for w, _, _ in fired if w.note and w.kind == "jobs"]
        parts = ["[Wake-up you set up]"]
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
        if s.project:
            parts.append(await self.agent.research_log(s.project))
        parts.append("Read the results (open the logs or metrics files for more) and update the research log. If the "
                     "work goes on, submit the next run and `ov watch` it; if the goal is reached or nothing sensible is "
                     "left, say so and set no more wake-ups; if you need the researcher's decision, ask.")
        return "\n\n".join(p for p in parts if p)
