"""Wake-ups: the agent decides what wakes it, without asking.

With its control command (ctl.py) the agent sets up what wakes it:

* timers - `ov wake in 90m`: woken once, later; `ov wake every 2h [--for 12h]`: repeatedly, until a set end;
* watches of a kind its domain registers (a WatchKind) - e.g. the HPC's `ov watch <job ids>`: woken when those
  jobs have ended (hpc/watches.py).

That is how it works on its own over many runs: start something, watch it, read the results when woken, decide the
next step. Each watch is announced in the conversation's topic with a cancel button. Guardrails: repeating timers
(and check-ins) at least `agent.autopilot.min_checkin_minutes` apart, repeating timers end (by default after
`repeat_hours`), `/auto off` and `/stop`, and a wake-up that fails pauses the conversation's wake-ups until the
researcher writes there.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .config import Config
from .notify import Notifier, Outgoing, code, esc
from .store import Store, WatchRecord

if TYPE_CHECKING:
    from .agent import Agent, Session

log = logging.getLogger(__name__)

REDELIVERED = "rewake:"  # + watch id: put back after a wake-up run that didn't finish (its next delivery says so)


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


class WatchKind:
    """Something besides timers that wakes a conversation, registered by its domain (e.g. the HPC's jobs). A watch
    of the kind names its subjects (WatchRecord.jobs: ids, space-separated)."""

    name = ""

    def describe(self, w: WatchRecord) -> str:
        return f"{self.name} {w.jobs}"

    def remove(self, w: WatchRecord, ref: str) -> bool:
        """`ov unwatch <ref>` where ref isn't a watch id: take it out of this watch (True if it was in it)."""
        if ref not in w.job_ids:
            return False
        rest = [j for j in w.job_ids if j != ref]
        w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
        return True

    def evaluate(self, w: WatchRecord) -> tuple[Any, list[str]] | None:
        """None while the watch isn't due; else (what the wake-up prompt needs, the subjects still to watch)."""
        return None

    async def wake_parts(self, s: Session, fired: list[tuple[WatchRecord, Any]]) -> list[str]:
        """The wake-up prompt's parts about the watches of this kind that fired."""
        return []

    def after_turn(self, s: Session, activity) -> None:
        """After a successful turn (e.g. the jobs the agent cancelled itself wake nobody)."""


class Autopilot:
    """The conversations' watches and timers, and the wake-ups they cause."""

    def __init__(self, cfg: Config, store: Store, agent: Agent, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.agent = agent
        self.notifier = notifier
        self.kinds: dict[str, WatchKind] = {}
        self._tasks: set[asyncio.Task] = set()
        agent.autopilot = self
        agent.domain.attach_autopilot(self)  # its watch kinds

    def register(self, kind: WatchKind) -> None:
        self.kinds[kind.name] = kind

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
            await asyncio.sleep(0)  # (finished tasks leave the set in a callback: let it run)

    # ------------------------------------------------------------------ state
    def paused(self, session_id: str) -> bool:
        return bool(self.store.get(f"wakeups_paused:{session_id}"))

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
            elif w.kind in self.kinds:
                p = self.kinds[w.kind].describe(w)
            else:
                p = f"{w.kind} {w.jobs}"
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
    def add_watch(self, s: Session, kind: str, subjects: list[str], every: float = 0.0,
                  note: str = "") -> tuple[WatchRecord, str]:
        """Wake the conversation when these subjects (of a registered kind, e.g. jobs) call for it. Returns the
        watch and a remark for the agent."""
        remark = ""
        if every and every < self.min_interval:
            remark = f" (check-ins are at least {human_interval(self.min_interval)} apart)"
            every = self.min_interval
        for w in self.store.watches(s.id):  # the latest grouping wins: a subject is in one watch of this conversation
            if w.kind == kind and set(w.job_ids) & set(subjects):
                rest = [j for j in w.job_ids if j not in subjects]
                w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
                self.store.update_watch(w)
        w = self.store.add_watch(WatchRecord(session_id=s.id, kind=kind, jobs=" ".join(subjects), every=every,
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
        """Remove a watch (w12), a subject from the watches (e.g. a job id), or everything (all)."""
        changed = []
        for w in self.store.watches(session_id):
            if ref == "all" or ref.lower() == f"w{w.id}":
                w.state = "cancelled"
            elif not (w.kind in self.kinds and self.kinds[w.kind].remove(w, ref)):
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

    # ------------------------------------------------------------------ events from the monitor and agent
    def user_turn(self, s: Session) -> None:
        """The researcher wrote in this conversation: wake-ups paused after a failure resume (after this turn)."""
        if self.paused(s.id):
            self.store.set(f"wakeups_paused:{s.id}", "")

    async def check(self, session_id: str) -> None:
        """Wake the conversation if watches or timers call for it (all that are due, in one turn)."""
        s = self.agent.get(session_id)
        if not s or self.agent.is_running(s) or self.paused(session_id):
            return  # a running turn re-checks when it ends (turn_ended); a paused one when the researcher writes
        now = time.time()
        fired: list[tuple[WatchRecord, Any, list[str]]] = []  # (watch, what its kind found, subjects still to watch)
        for w in self.store.watches(session_id):
            if w.kind == "timer":
                if w.due <= now:
                    fired.append((w, None, []))
                continue
            kind = self.kinds.get(w.kind)
            if kind is None:
                continue  # (a kind of another domain: not this bot's)
            state = w.state
            due = kind.evaluate(w)
            if w.state != state:  # (e.g. nothing left to watch)
                self.store.update_watch(w)
            if due is not None:
                fired.append((w, *due))
        if not fired:
            return
        undo: list[tuple] = [(w.id, w.state, w.due, w.fired) for w, _, _ in fired]
        for w, _, rest_ids in fired:
            w.fired = now
            if w.kind == "timer" and w.every and (not w.until or now + w.every <= w.until):
                w.due = now + w.every  # repeating: stays active until its end
            else:
                w.state = "fired"
            self.store.update_watch(w)
            if rest_ids:  # woken early (e.g. one job failed): the rest stay watched
                rest = self.store.add_watch(WatchRecord(session_id=session_id, kind=w.kind, every=w.every, note=w.note,
                                                        jobs=" ".join(rest_ids)))
                undo.append((rest.id, "cancelled", rest.due, rest.fired))  # undone if this wake-up fails
        prompt = await self._wake_prompt(s, fired)
        self._spawn(self._run(s, prompt, None, "wakeup", undo=undo))  # (the agent's reply is the notice)

    def turn_ended(self, s: Session) -> None:
        """After any turn: jobs may have ended or timers come due while it ran."""
        if self.store.watches(s.id):
            async def recheck():
                async with s.lock:
                    pass
                await self.check(s.id)
            self._spawn(recheck())

    def after_turn(self, s: Session, activity) -> str:
        """After a successful turn: a line for the reply (HTML) saying what will wake the agent, if anything."""
        for kind in self.kinds.values():
            kind.after_turn(s, activity)
        watches = self.store.watches(s.id)
        return f"⏰ Waiting for: {esc(self.describe_watches(watches))}" if watches else ""

    # ------------------------------------------------------------------ the tick: timers
    async def tick(self) -> None:
        """Timers that came due (every poll of a domain's monitor, and the platform's own loop)."""
        now = time.time()
        for sid in dict.fromkeys(w.session_id for w in self.store.watches() if w.kind == "timer" and w.due <= now):
            await self.check(sid)

    async def loop(self, every: float = 30) -> None:
        while True:
            await asyncio.sleep(every)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("timer tick failed")

    def wake(self, s: Session, prompt: str, kind: str = "wakeup", quiet: bool = False, by_user: bool = False) -> None:
        """Wake the conversation now with this prompt (a domain's alert), unless its wake-ups are paused.
        `by_user`: something the user set up (e.g. their routine, a button they pressed), not the agent - it runs
        even while the agent's own wake-ups are paused, and its failure doesn't pause them."""
        if self.paused(s.id) and not by_user:
            return
        self._spawn(self._run(s, prompt, None, kind, quiet=quiet, by_user=by_user))

    # ------------------------------------------------------------------ running a wake-up
    async def _run(self, s: Session, prompt: str, notice: str | None, kind: str,
                   undo: list[tuple] | None = None, quiet: bool = False, by_user: bool = False) -> None:
        if notice:
            await self.notifier.send(Outgoing(notice, session_id=s.id, silent=quiet))
        started = time.time()
        try:
            ok = await self.agent.handle(s, prompt, quiet=quiet, origin=kind)
        except Exception:
            if self.agent.shutting_down:  # cut off by Overseer stopping: offered again at the next start
                return
            raise
        if ok or self.agent.shutting_down or by_user:  # (the user's own schedule isn't the agent's to pause)
            return
        # A failed or stopped wake-up pauses this conversation's wake-ups: nothing keeps going unattended after a
        # problem. The watches that fired are put back, so they fire again once the researcher writes here - unless
        # the run had already acted on them (it set new wake-ups: e.g. it resubmitted and watches the new job), then
        # the same news isn't handed to it again (FB5). Put back, they come with a warning that they were delivered.
        self.store.set(f"wakeups_paused:{s.id}", "1")
        fired_ids = [wid for wid, state, _, _ in undo or [] if state != "cancelled"]
        new = [w for w in self.store.watches(s.id) if w.created >= started]
        if new and fired_ids:
            self.agent.add_note(s.id, f"Your wake-up run for {', '.join(f'w{i}' for i in fired_ids)} didn't finish "
                                "(an error or a stop), but it had already set new wake-ups ("
                                + ", ".join(f"w{w.id}" for w in new) + "), so those aren't handed to you again. "
                                "Check that nothing was left half-done.")
        else:
            for wid, state, due, fired in undo or []:
                w = self.store.get_watch(wid)
                if w:
                    w.state, w.due, w.fired = state, due, fired
                    self.store.update_watch(w)
                    if wid in fired_ids:
                        self.store.set(f"{REDELIVERED}{wid}", str(started))
        await self.notifier.send(Outgoing(
            "⏸ <b>Wake-ups paused</b> for this conversation: that run didn't complete (see above). They resume when "
            "you write to the agent here; /auto off cancels them.", session_id=s.id))

    # ------------------------------------------------------------------ prompts
    async def _wake_prompt(self, s: Session, fired: list[tuple[WatchRecord, Any, list[str]]]) -> str:
        parts = ["[Wake-up you set up]"]
        before = [(w, self.store.get(f"{REDELIVERED}{w.id}")) for w, _, _ in fired]
        before = [(w, float(t)) for w, t in before if t]
        if before:  # (handed over before, by a run that didn't finish: some of it may be done)
            for w, _ in before:
                self.store.unset(f"{REDELIVERED}{w.id}")
            when = datetime.fromtimestamp(min(t for _, t in before)).strftime("%H:%M")
            parts.append(f"[Again: {', '.join(f'w{w.id}' for w, _ in before)} came to you before, at {when}, but that "
                         "run didn't finish (an error, a stop or a restart). Some of it may already be done: check "
                         "first (`ov watches`, `ov job list`, `ov results`, the research log), then do only what's "
                         "left - don't resubmit what's already running.]")
        by_kind: dict[str, list[tuple[WatchRecord, Any]]] = {}
        for w, info, _ in fired:
            if w.kind != "timer":
                by_kind.setdefault(w.kind, []).append((w, info))
        for name, items in by_kind.items():
            parts += await self.kinds[name].wake_parts(s, items)
        for w, _, _ in fired:
            if w.kind == "timer":
                ago = human_interval(max(60.0, time.time() - w.created))
                parts.append(f"Your timer w{w.id} (set {ago} ago): {w.note or '(no note)'}")
        parts += await self.agent.domain.wake_context(s)  # e.g. the research log, the latest results
        parts.append(self.agent.domain.wake_instructions(s))
        return "\n\n".join(p for p in parts if p)
