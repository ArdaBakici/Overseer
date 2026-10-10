"""Autopilot: the agent runs experiments on its own, woken by its jobs.

Turned on per conversation (`/auto 10 every 2h <goal>`). Then:

* when the jobs the agent submitted in that conversation finish, Overseer resumes the
  conversation with their final reports (one round). Jobs submitted together are reported
  together: the round starts when all of them have finished, or as soon as one fails;
* while they run, optional check-ins show the agent their progress so it can cancel a run
  that is clearly broken or diverging (check-ins don't count as rounds);
* the agent ends every reply with a status line - continue, done, or need help - and keeps
  a research log (RESEARCH_LOG.md) in the project folder on the HPC.

Guardrails: a hard round limit, `/auto off`, `/stop`, at most a few check-ins per job, the
loop pauses on any failed or stopped round, and it only ever wakes for the agent's own jobs
in that conversation.
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
from .notify import Notifier, Outgoing, code, esc
from .slurm import SUCCESS_STATES, Job
from .store import AutopilotRecord, JobRecord, Store

if TYPE_CHECKING:
    from .agent import Agent, Session

log = logging.getLogger(__name__)

MAX_REPORT = 12_000  # chars of a finished job's report given to the agent
STATUS_RE = re.compile(r"^[ \t>*_`]*AUTOPILOT\s*:\s*(CONTINUE|DONE|NEED[ _-]?HELP)\b[ \t:*_`—–-]*(.*)$", re.I | re.M)

SYSTEM_SECTION = """

## Autopilot is ON in this conversation
The researcher is away. You run experiments toward this goal on your own:
<goal>
{goal}
</goal>
Overseer wakes you up when the jobs you submit in this conversation finish (at most {max_rounds} rounds; each \
wake-up message says which round it is){checkins}. Nothing runs between wake-ups, so:
- Each round: read the results, update the research log, decide the next experiment, submit it with sbatch, \
then end your reply. Don't wait for jobs, poll in a loop or sleep: Overseer wakes you.
- Research log: keep RESEARCH_LOG.md in the project folder on the HPC. Create it in your first autopilot reply \
(goal, baseline, plan) and say where it is; then add an entry per round: what you changed and why, the job ids, \
the key results, the next idea. Read it first whenever you wake up.
- Change one thing at a time unless the goal says otherwise. Give every run its own name and output folder, and \
never delete or overwrite earlier results.
- Only cancel or change jobs you submitted in this conversation.
- End every reply with exactly one status line:
  AUTOPILOT: CONTINUE - jobs are queued or running and you want to be woken when they finish
  AUTOPILOT: DONE <one-line conclusion> - the goal is reached, or nothing sensible is left to try
  AUTOPILOT: NEED_HELP <question> - you need the researcher's decision, or you're stuck"""


def parse_status(text: str) -> tuple[str, str | None, str]:
    """(text without the status line, CONTINUE|DONE|NEED_HELP or None, detail)."""
    matches = list(STATUS_RE.finditer(text or ""))
    if not matches:
        return text, None, ""
    m = matches[-1]
    status = "NEED_HELP" if m.group(1).upper().startswith("NEED") else m.group(1).upper()
    detail = m.group(2).strip().strip("*_`").strip()
    return (text[:m.start()] + text[m.end():]).strip(), status, detail


def parse_duration(text: str) -> float | None:
    """'2h', '90m', '1.5h', '1d', '2' (hours) -> seconds; 'off'/'0' -> 0; None if not a duration."""
    t = text.strip().lower()
    if t in ("off", "none", "no", "0"):
        return 0.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(m|min|mins|h|hr|hrs|hours?|d|days?)?", t)
    if not m:
        return None
    unit = (m.group(2) or "h")[0]
    return float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[unit]


def human_interval(seconds: float) -> str:
    if seconds % 3600 == 0:
        return f"{int(seconds // 3600)}h"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{int(seconds // 60)}m"


def _ids(jobs: list[JobRecord]) -> str:
    return ", ".join(j.job_id for j in jobs)


class Autopilot:
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

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self) -> None:
        """Wait for wake-ups in flight (tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def active(self, session_id: str | None) -> AutopilotRecord | None:
        a = self.store.get_autopilot(session_id) if session_id else None
        return a if a and a.active else None

    def handles(self, rec: JobRecord) -> bool:
        return bool(rec.by_agent and self.active(rec.session_id))

    def system_section(self, session_id: str) -> str:
        a = self.active(session_id)
        if not a:
            return ""
        checkins = (f", and every {human_interval(a.checkin_every)} for a check-in while they run"
                    if a.checkin_every else "")
        # Stable across rounds: the system prompt heads the provider's prompt cache.
        return SYSTEM_SECTION.format(goal=a.goal or "(see the conversation)", max_rounds=a.max_rounds,
                                     checkins=checkins)

    # ------------------------------------------------------------------ turning it on and off
    async def start(self, s: Session, rounds: int, goal: str = "", checkin_every: float | None = None) -> str:
        """Turn autopilot on (or extend it) for a conversation. Returns the confirmation (HTML)."""
        rounds = max(1, min(rounds, self.c.max_rounds))
        a = self.store.get_autopilot(s.id)
        if a is None:
            a = AutopilotRecord(session_id=s.id, created=time.time(), checkin_every=self.c.checkin_minutes * 60)
            for j in self.store.session_jobs(s.id):  # results from before autopilot aren't a round
                if j.ended_at and not j.agent_reported:
                    j.agent_reported = 1
                    self.store.upsert_job(j)
        if checkin_every is not None:
            a.checkin_every = 0.0 if checkin_every <= 0 else max(checkin_every, self.c.min_checkin_minutes * 60)
        a.active, a.max_rounds, a.rounds, a.status = 1, rounds, 0, "waiting"
        a.goal = goal or a.goal
        self.store.upsert_autopilot(a)

        jobs = self.store.session_jobs(s.id)
        running = [j for j in jobs if not j.ended_at]
        pending = [j for j in jobs if j.ended_at and not j.agent_reported]
        checkins = f"check-ins every {human_interval(a.checkin_every)}" if a.checkin_every else "no check-ins"
        lines = [f"🔁 <b>Autopilot on</b> for {code(s.id)}: up to {rounds} rounds, {checkins}."]
        if a.goal:
            lines.append(f"Goal: {esc(a.goal)}")
        if goal:
            lines.append(f"{esc(self.agent.backend)} is planning the first experiment…")
            self._spawn(self._run(s, self._kickoff_prompt(a), notice=None))
        elif pending:
            lines.append(f"Job(s) {esc(_ids(pending))} already finished: starting a round now.")
            self._spawn(self.check(s.id))
        elif running:
            lines.append(f"Waiting for job(s) {esc(_ids(running))}.")
        else:
            lines.append("No job of this conversation is running. Reply to this message with what to try; "
                         "once the agent submits a job, it continues on its own.")
        lines.append("/auto shows progress · /auto off stops it · /stop stops the current run")
        return "\n".join(lines)

    def set_checkins(self, session_id: str, every: float) -> str:
        a = self.active(session_id)
        if not a:
            return "Autopilot isn't on in this conversation."
        a.checkin_every = 0.0 if every <= 0 else max(every, self.c.min_checkin_minutes * 60)
        self.store.upsert_autopilot(a)
        return (f"🔎 Check-ins every {human_interval(a.checkin_every)} for {code(session_id)}." if a.checkin_every
                else f"Check-ins off for {code(session_id)}.")

    def stop(self, session_id: str | None = None, status: str = "off") -> list[str]:
        """Turn autopilot off for one conversation, or all. Returns the conversations stopped."""
        targets = [self.store.get_autopilot(session_id)] if session_id else self.store.list_autopilots(True)
        stopped = []
        for a in targets:
            if a and a.active:
                a.active, a.status = 0, status
                self.store.upsert_autopilot(a)
                stopped.append(a.session_id)
        return stopped

    def describe(self) -> str:
        rows = self.store.list_autopilots()[:8]
        if not rows:
            return ("Autopilot is off everywhere.\n\nStart it with /auto [rounds] [every 2h] &lt;goal&gt; - as a reply "
                    "to a job or agent message to use that conversation.")
        lines = []
        for a in rows:
            jobs = self.store.session_jobs(a.session_id)
            running = [j for j in jobs if not j.ended_at]
            mark = "▶️" if a.active else "⏹"
            bits = [f"{mark} {code(a.session_id)} · {esc(a.status or '-')} · round {a.rounds}/{a.max_rounds}"]
            if a.checkin_every:
                bits.append(f"check-ins every {human_interval(a.checkin_every)}")
            if running:
                bits.append(f"jobs {esc(_ids(running))}")
            lines.append(" · ".join(bits) + (f"\n    goal: {esc(a.goal[:200])}" if a.goal else ""))
        return "\n".join(lines)

    # ------------------------------------------------------------------ events from the monitor and agent
    async def job_finished(self, rec: JobRecord, report: str) -> None:
        if not self.handles(rec):
            return
        rec = self.store.get_job(rec.job_id) or rec
        rec.report = report[-MAX_REPORT:]
        self.store.upsert_job(rec)
        await self.check(rec.session_id)

    async def check(self, session_id: str) -> None:
        """Start a round if the conversation's finished jobs call for one."""
        a = self.active(session_id)
        s = self.agent.get(session_id)
        if not a or not s or self.agent.is_running(s):
            return  # a running turn re-checks when it ends (turn_ended)
        jobs = self.store.session_jobs(session_id)
        done = [j for j in jobs if j.ended_at and not j.agent_reported]
        running = [j for j in jobs if not j.ended_at]
        if not done:
            return
        failed = [j for j in done if j.state not in SUCCESS_STATES]
        if running and not failed:
            await self.notifier.send(Outgoing(
                f"🔁 Autopilot: job {esc(_ids(done))} finished; waiting for {esc(_ids(running))} before the next round.",
                session_id=session_id, silent=True))
            return
        if a.rounds >= a.max_rounds:
            a.active, a.status = 0, "limit"
            self.store.upsert_autopilot(a)
            await self.notifier.send(Outgoing(
                f"🔁 <b>Autopilot stopped</b>: it used all {a.max_rounds} rounds. Job(s) {esc(_ids(done))} finished "
                "(reports above). Reply to this message to continue by hand, or send /auto &lt;rounds&gt; as a reply "
                "to give it more rounds.", session_id=session_id))
            return
        a.rounds += 1
        a.status = "working"
        self.store.upsert_autopilot(a)
        for j in done:
            j.agent_reported = 1
            self.store.upsert_job(j)
        states = ", ".join(f"{j.job_id} {j.state}" for j in done)
        notice = (f"🔁 <b>Autopilot round {a.rounds}/{a.max_rounds}</b>: {esc(states)}. "
                  f"{esc(self.agent.backend)} is analysing…")
        self._spawn(self._run(s, self._round_prompt(a, done, running), notice, round_jobs=[j.job_id for j in done]))

    def turn_ended(self, s: Session) -> None:
        """After any turn in an autopilot conversation: jobs may have finished while it ran."""
        if self.active(s.id):
            async def recheck():
                async with s.lock:
                    pass
                await self.check(s.id)
            self._spawn(recheck())

    def after_turn(self, s: Session, status: str | None, detail: str, cancelled: list[str]) -> str:
        """Update the autopilot after a successful turn. Returns a line for the reply (HTML)."""
        a = self.active(s.id)
        if not a:
            return ""
        for jid in cancelled:  # the agent knows why it cancelled these: their end isn't a new round
            j = self.store.get_job(jid)
            if j and j.session_id == s.id and not j.agent_reported:
                j.agent_reported = 1
                self.store.upsert_job(j)
        running = [j for j in self.store.session_jobs(s.id) if not j.ended_at]
        if status == "DONE":
            a.active, a.status = 0, "done"
            line = f"🏁 <b>Autopilot finished</b> after {a.rounds} round(s): {esc(detail or 'goal reached')}"
        elif status == "NEED_HELP":
            a.status = "need_help"
            line = (f"🙋 <b>Autopilot needs you</b>: {esc(detail or 'see above')}\nReply to this message to answer; "
                    "it continues on its own once the agent submits the next job.")
        elif running:
            a.status = "waiting"
            line = f"🔁 Autopilot: waiting for job(s) {esc(_ids(running))} · round {a.rounds}/{a.max_rounds} used"
        else:
            a.status = "idle"
            line = ("⚠️ Autopilot: no job is running, so nothing will wake the agent. Reply to this message to "
                    "continue, or /auto off.")
        self.store.upsert_autopilot(a)
        return line

    # ------------------------------------------------------------------ check-ins
    async def poll_running(self, live: dict[str, Job]) -> None:
        """Called by the monitor after each poll: start check-ins that are due."""
        now = time.time()
        for a in self.store.list_autopilots(active_only=True):
            if a.checkin_every <= 0:
                continue
            s = self.agent.get(a.session_id)
            if not s or self.agent.is_running(s):
                continue
            running = []
            for j in self.store.session_jobs(a.session_id):
                if j.ended_at or j.job_id not in live or live[j.job_id].state != "RUNNING":
                    continue
                if not j.last_checkin:  # the clock starts when the job starts running
                    j.last_checkin = now
                    self.store.upsert_job(j)
                running.append(j)
            cap = self.c.max_checkins_per_job
            if not any(j.checkins < cap and now - j.last_checkin >= a.checkin_every for j in running):
                continue
            # One check-in covers every running job that is at least halfway to its next one.
            due = [j for j in running if j.checkins < cap and now - j.last_checkin >= a.checkin_every / 2]
            for j in due:
                j.checkins += 1
                j.last_checkin = now
                self.store.upsert_job(j)
            self._spawn(self._checkin(s, a, [(j, live[j.job_id]) for j in due]))

    async def _checkin(self, s: Session, a: AutopilotRecord, due: list[tuple[JobRecord, Job]]) -> None:
        parts = []
        for rec, job in due:
            tail = tqdm_cleanup(await self.hpc.try_tail(rec.stdout, 80)) if rec.stdout else ""
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
            "[Autopilot check-in - not a new round; these jobs are still running]\n\n" + "\n\n".join(parts) + "\n\n"
            "Check whether they are healthy (read more of the logs or metrics files if you need to). If they are, "
            "change nothing: reply in one or two sentences and end with AUTOPILOT: CONTINUE. If a run is clearly "
            "broken, diverging or wasting GPU time, scancel it, note why in the research log, fix and resubmit, "
            "and end with the status line as usual.")
        elapsed = ", ".join(f"{rec.job_id} running {job.elapsed}" for rec, job in due)
        await self._run(s, prompt, notice=f"🔎 Autopilot check-in: {esc(elapsed)}", quiet=True)

    # ------------------------------------------------------------------ running a wake-up
    async def _run(self, s: Session, prompt: str, notice: str | None, round_jobs: list[str] | None = None,
                   quiet: bool = False) -> None:
        if notice:
            await self.notifier.send(Outgoing(notice, session_id=s.id, silent=quiet))
        ok = await self.agent.handle(s, prompt, quiet=quiet)
        if ok:
            return
        a = self.active(s.id)
        if not a:
            return
        # A failed or stopped wake-up pauses the loop: it never keeps going unattended after a problem.
        # Updated before any await, so the re-check after the turn sees it paused.
        a.active, a.status = 0, "paused"
        if round_jobs:
            a.rounds = max(0, a.rounds - 1)
            for jid in round_jobs:
                j = self.store.get_job(jid)
                if j:
                    j.agent_reported = 0
                    self.store.upsert_job(j)
        self.store.upsert_autopilot(a)
        await self.notifier.send(Outgoing(
            "⏸ <b>Autopilot paused</b>: that run didn't complete (see above). Send /auto &lt;rounds&gt; as a reply to "
            "this message to retry, or reply to continue by hand.", session_id=s.id))

    # ------------------------------------------------------------------ prompts
    def _kickoff_prompt(self, a: AutopilotRecord) -> str:
        return (f"[Autopilot started: up to {a.max_rounds} rounds]\n\nGoal: {a.goal}\n\n"
                "Look at the project and any earlier results, create the research log, plan the first experiment, "
                "submit it, and end with the status line.")

    def _round_prompt(self, a: AutopilotRecord, done: list[JobRecord], running: list[JobRecord]) -> str:
        reports = "\n\n".join(
            f"<job id='{j.job_id}' name='{j.name}' state='{j.state}' stdout='{j.stdout}' stderr='{j.stderr}'>\n"
            f"{j.report.strip()}\n</job>" for j in done)
        parts = [f"[Autopilot round {a.rounds} of {a.max_rounds}] Job(s) you submitted have finished:", reports]
        if running:
            parts.append(f"Still running: {_ids(running)} (you'll be woken again when they finish; one of the "
                         "finished jobs failed, which is why you're woken now).")
        if a.rounds >= a.max_rounds:
            parts.append("This is the LAST round: don't start new experiments. Write the final summary in the research "
                         "log and in your reply, and end with AUTOPILOT: DONE.")
        else:
            parts.append("Read the results (open the logs or metrics files for more), update the research log, then "
                         "submit the next experiment or stop. End with the status line.")
        return "\n\n".join(parts)
