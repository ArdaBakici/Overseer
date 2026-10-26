"""Job watches: the HPC's kind of wake-up (`ov watch <job ids>`).

The agent is woken when the jobs it watches have ended (all of them, or as soon as one fails), with their final
reports; `--every 30m` adds check-ins while they run. A job it cancelled itself (scancel) wakes nobody. The monitor
also wakes the conversations watching a job when an alert they set fires (`ov metrics set --wake-on`) or its GPUs
sit idle.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..core.autopilot import Autopilot, WatchKind, human_interval
from ..core.store import JobRecord, WatchRecord
from .diagnose import extract_metrics, tqdm_cleanup
from .monitor import progress_log
from .slurm import SUCCESS_STATES, Job

if TYPE_CHECKING:
    from ..core.agent import Session

MAX_REPORT = 12_000  # chars of a finished job's report given to the agent


def _ids(jobs: list[JobRecord]) -> str:
    return ", ".join(j.job_id for j in jobs)


class JobWatches(WatchKind):
    name = "jobs"

    def __init__(self, domain, autopilot: Autopilot):
        self.domain = domain
        self.ap = autopilot
        self.cfg = domain.cfg
        self.store = domain.store
        self.agent = autopilot.agent

    @property
    def hpc(self):
        return self.domain.hpc

    async def drain(self) -> None:
        await self.ap.drain()

    # ------------------------------------------------------------------ the kind
    def describe(self, w: WatchRecord) -> str:
        p = f"job{'s' if len(w.job_ids) > 1 else ''} {', '.join(w.job_ids)}"
        if w.every:
            p += f" (check-ins every {human_interval(w.every)})"
        return p

    def evaluate(self, w: WatchRecord) -> tuple[Any, list[str]] | None:
        recs = [r for j in w.job_ids if (r := self.store.get_job(j)) and not r.ignored]
        ended = [r for r in recs if r.ended_at]
        running = [r for r in recs if not r.ended_at]
        if not recs:
            w.state = "cancelled"
            return None
        if not running or any(r.state not in SUCCESS_STATES for r in ended):
            return (ended, running), [r.job_id for r in running]
        return None

    async def wake_parts(self, s: Session, fired: list[tuple[WatchRecord, Any]]) -> list[str]:
        ended = list({r.job_id: r for _, (e, _) in fired for r in e}.values())
        running = list({r.job_id: r for _, (_, rn) in fired for r in rn}.values())
        notes = [f"w{w.id}: {w.note}" for w, _ in fired if w.note]
        parts = []
        if ended:
            parts.append("Job(s) you watched have finished:")
            parts.append("\n\n".join(
                f"<job id='{j.job_id}' name='{j.name}' state='{j.state}' stdout='{j.stdout}' stderr='{j.stderr}'>\n"
                f"{(j.report or '(no final report: read its logs)').strip()}\n</job>" for j in ended))
        if notes:
            parts.append("Your notes for these: " + "; ".join(notes))
        if running:
            parts.append(f"Still running: {_ids(running)} (still watched; one of the jobs failed, which is why "
                         "you're woken now).")
        return parts

    def after_turn(self, s: Session, activity) -> None:
        for jid in getattr(activity, "cancelled", []):  # the agent knows why it cancelled these: their end wakes nobody
            for w in self.store.job_watches(jid):
                if w.session_id == s.id:
                    rest = [j for j in w.job_ids if j != jid]
                    w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
                    self.store.update_watch(w)

    # ------------------------------------------------------------------ events from the monitor and ov
    def forget_job(self, job_id: str) -> None:
        """The job isn't tracked any more (ov job delete): it wakes nobody."""
        for w in self.store.job_watches(job_id):
            rest = [j for j in w.job_ids if j != job_id]
            w.jobs, w.state = " ".join(rest), ("active" if rest else "cancelled")
            self.store.update_watch(w)

    async def job_finished(self, rec: JobRecord, report: str) -> None:
        watches = self.store.job_watches(rec.job_id)
        if not watches:
            return
        rec = self.store.get_job(rec.job_id) or rec
        rec.report = report[-MAX_REPORT:]
        self.store.upsert_job(rec)
        for sid in dict.fromkeys(w.session_id for w in watches):
            await self.ap.check(sid)

    async def tick(self, live: dict[str, Job]) -> None:
        """After each poll of the monitor: timers, and check-ins on the watched jobs that run."""
        await self.ap.tick()
        await self._checkins(live, time.time())

    async def _checkins(self, live: dict[str, Job], now: float) -> None:
        by_session: dict[str, list[tuple[JobRecord, Job, float]]] = {}
        for w in self.store.watches():
            if w.kind != "jobs" or not w.every or self.ap.paused(w.session_id):
                continue
            every = max(w.every, self.ap.min_interval)
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
            self.ap._spawn(self._checkin(s, due))

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
                (f"job array tasks: {tasks}" + (f"; the logs above are task {job.extra['task']}'s, each task "
                                                f"writes {rec.info_dict['stdout_tpl']} (%a = its id)"
                                                if job.extra.get("task") and rec.info_dict.get("stdout_tpl") else ""))
                if tasks else "",
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
        await self.ap._run(s, prompt, notice=None, kind="checkin", quiet=True)

    async def alert_wake(self, session_id: str, rec: JobRecord, alert: str, value: str) -> None:
        """An alert the agent set (`ov metrics set --wake-on`) matched: wake its conversation about the job."""
        s = self.agent.get(session_id)
        if not s or self.ap.paused(session_id):
            return
        prompt = (f"[Alert you set up] Job {rec.job_id} ({rec.name}): `{alert}` matched (now {value}).\n\n"
                  f"stdout: {rec.stdout}\n\nLook at the job now (its logs or metrics) and decide: let it run, or "
                  "scancel it, fix and resubmit. Note it in the research log.")
        self.ap.wake(s, prompt)

    async def idle_wake(self, session_id: str, rec: JobRecord, idle_for: str, state: dict) -> None:
        """A job this conversation watches has left its GPUs idle (monitor.py): wake it to find out why."""
        s = self.agent.get(session_id)
        if not s or self.ap.paused(session_id):
            return
        mem, total = state.get("mem", 0) / 1024, state.get("total", 0) / 1024
        hint = ("Nothing is on the GPU: the program may be running on the CPU (a device bug), or still in a long "
                "CPU phase (data preparation)." if mem < 0.5 else
                "The GPU holds data, but nothing runs on it: a hang or deadlock (e.g. a distributed collective), "
                "waiting on data loading or I/O, or a long CPU phase (evaluation, checkpointing).")
        prompt = (f"[Idle GPU] Job {rec.job_id} ({rec.name}), which you watch, has been running with its "
                  f"{state.get('n', 1)} GPU(s) idle for {idle_for}: at most {self.cfg.slurm.idle_gpu_percent}% busy "
                  f"(now {state.get('util', 0):.0f}%), {mem:.1f} of {total:.0f} GB GPU memory in use. {hint}\n\n"
                  f"stdout: {rec.stdout}\n\nFind out why from its logs (is it still making progress?). If it's stuck "
                  "or wasting the GPU, scancel it, fix the cause, resubmit and `ov watch` the new job; if the idle "
                  "phase is expected, say so briefly (you won't be woken again unless it's busy and then idle again). "
                  "Note what you found in the research log.")
        self.ap.wake(s, prompt)
