"""Polls SLURM and turns state changes into Telegram notifications."""

from __future__ import annotations

import asyncio
import dataclasses
import fnmatch
import json
import logging
import shutil
import time

from .compute import WORKER_NAME, is_helper
from ..core.config import Config
from .diagnose import (Reporter, all_metrics, extract_traceback, find_issues, find_log_paths, rank_metrics,
                       tqdm_cleanup)
from .remote import HPC
from .metrics import check_alerts, fired_alerts, resolve, run_command
from .metrics import lines as metric_lines
from ..core.notify import Notifier, Outgoing, code, esc
from .quota import parse_quota
from .plots import resolve_plot, run_plots
from .results import record_job, snapshot
from ..core.notify import human_seconds
from ..core.richtext import fold_text
from .cluster import Cluster, until
from .slurm import (NO_VAL, STATE_EMOJI, SUCCESS_STATES, Job, array_parent, format_gpu_stats,
                    has_gpus, log_template, parse_gpu_stats, slurm_seconds, summarize_steps,
                    task_states)
from .ssh import SSHError
from ..core.store import JobRecord, Store

log = logging.getLogger(__name__)

MISSING_POLLS_BEFORE_GIVING_UP = 10
ESTIMATE_SECONDS = 300  # how often queued jobs' estimated starts are read


def when(start: str, clock: str, relative: bool = False) -> str:
    """'today 14:20' ('today 14:20 (in ~1h05m)' with `relative`), 'tomorrow 03:00', 'Mon 20 Oct 09:00': an
    estimated start, in the cluster's time. (The card shows no "in ...": it would change at every look.)"""
    from datetime import datetime
    try:
        t, now = datetime.fromisoformat(start), datetime.fromisoformat(clock)
    except ValueError:
        return start
    days = (t.date() - now.date()).days
    day = "today" if days <= 0 else "tomorrow" if days == 1 else t.strftime("%a %d %b")
    rel = until(start, clock) if days <= 0 and relative else ""
    return f"{day} {t.strftime('%H:%M')}" + (f" ({rel})" if rel else "")


async def progress_log(hpc: HPC, rec: JobRecord) -> str:
    """The log file the job's program actually writes to, when its SLURM stdout names one
    (e.g. "output: .../train_<jobid>.log"). Remembered in rec.info; the caller saves rec."""
    info = rec.info_dict
    if info.get("progress_log"):
        return info["progress_log"]
    if not rec.stdout:
        return ""
    paths = find_log_paths(await hpc.try_head(rec.stdout, 80), rec.job_id, exclude=(rec.stdout, rec.stderr))
    if not paths:
        return ""
    info["progress_log"] = paths[0]
    rec.info = json.dumps(info)
    return paths[0]


class Monitor:
    def __init__(self, cfg: Config, hpc: HPC, store: Store, notifier: Notifier, reporter: Reporter):
        self.cfg = cfg
        self.hpc = hpc
        self.store = store
        self.notifier = notifier
        self.reporter = reporter
        self.last_ok: float = 0.0
        self.last_error: str = ""
        self._error_notified = False
        self._wake = asyncio.Event()
        self.autopilot = None  # set by Autopilot's owner: told about the agent's jobs finishing / running
        self.worker: Job | None = None  # hpc-run's compute worker, if one is queued or running
        self._quota_checked = 0.0
        self._bg: set[asyncio.Task] = set()  # graphs and card moves: never in the way of the poll
        self._plotting: set[str] = set()
        self._sampling: set[str] = set()  # jobs whose GPUs are being sampled
        self.cluster = Cluster(hpc)  # what's free, when queued jobs start (ov cluster, /cluster, queued cards)
        self._estimated = 0.0

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def drain(self) -> None:
        """Wait for graphs and card moves in progress (tests)."""
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)
            await asyncio.sleep(0)  # (finished tasks leave _bg in a callback: let it run)

    def poke(self) -> None:
        """Poll right away (e.g. after a submission)."""
        self._wake.set()

    async def run_forever(self) -> None:
        first = True
        while True:
            try:
                await self.poll(first=first)
                first = False
                self.last_ok = time.time()
                self.last_error = ""
                if self._error_notified:
                    self._error_notified = False
                    await self.notifier.send(Outgoing("🟢 Connection to the HPC restored.", silent=True))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("poll failed")
                down_for = time.time() - (self.last_ok or time.time())
                if not self._error_notified and (self.last_ok == 0 or down_for > 10 * 60):
                    self._error_notified = True
                    await self.notifier.send(Outgoing(f"🔴 Cannot poll the HPC: {code(self.last_error[:500])}"))
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.cfg.slurm.poll_interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    # ------------------------------------------------------------------ polling
    def _ignored(self, job: Job, rec: JobRecord | None) -> bool:
        if is_helper(job.name):  # hpc-run's jobs (its compute worker: shown in /health): not reported as jobs
            if job.name == WORKER_NAME and (job.state == "RUNNING" or self.worker is None):
                self.worker = job
            return True
        if rec and rec.ignored:  # deleted from Overseer (ov job delete)
            return True
        if rec and rec.by_agent:
            return False
        if self.cfg.slurm.track == "agent":
            return True
        return any(fnmatch.fnmatch(job.name, g) for g in self.cfg.slurm.ignore_name_globs)

    async def poll(self, first: bool = False) -> None:
        await self.hpc.ensure_connected()
        self._merge_array_tasks()
        live = await self.hpc.squeue()
        live_by_id = {j.job_id: j for j in live}
        self.worker = None
        # Jobs that started and finished between polls never show up in squeue; sacct catches them.
        window = max(10, int(self.cfg.slurm.poll_interval * 3 / 60) + 10)
        recent = await self.hpc.sacct_recent(window if not first else 60)
        recent_by_id = {j.job_id: j for j in recent}

        adopted = []
        for job in live:
            if "[" in job.job_id:  # pending array placeholder like 123_[0-9]
                continue
            rec = self.store.get_job(job.job_id)
            if self._ignored(job, rec):
                continue
            if rec is None:
                rec = await self._register(job)
                if first:
                    adopted.append(rec)
                if job.state == "RUNNING" or self.cfg.slurm.notify_queued:
                    rec.last_status_at = time.time()
                    self.store.upsert_job(rec)
                    await self.refresh_card(rec, job)
                continue
            if not rec.stdout:  # submitted by the agent: fill in details on first sight
                await self._fill(rec, job)
            self._follow_task(rec, job)
            prev = rec.state
            rec.state = job.state
            rec.missing_polls = 0
            if prev != "RUNNING" and job.state == "RUNNING":  # started: its card says so
                self.store.unset(f"eststart:{rec.job_id}")
                rec.last_status_at = time.time()
                self.store.upsert_job(rec)
                await self.refresh_card(rec, job)
            elif job.state == "RUNNING":
                await self._maybe_status(rec, job)
            if job.state == "RUNNING":
                self._maybe_sample_gpus(rec, job)
            self.store.upsert_job(rec)

        await self._estimate_starts(live)
        if adopted:
            await self.notifier.send(Outgoing(
                f"👀 <b>Overseer online</b>: tracking {len(adopted)} job(s) - {', '.join(r.job_id for r in adopted[:8])}"
                + (" …" if len(adopted) > 8 else "") + ".", silent=True))

        # Jobs we know about that left the queue.
        for rec in self.store.active_jobs():
            if rec.job_id in live_by_id:
                continue
            if is_helper(rec.name):  # hpc-run's job, registered by mistake: never reported
                rec.ended_at, rec.ignored = time.time(), 1
                self.store.upsert_job(rec)
                continue
            final = recent_by_id.get(rec.job_id)
            if final is None or "tasks" in final.extra:  # (the recent window misses an array's earlier tasks)
                final = await self.hpc.sacct_job(rec.job_id) or final
            if final and final.is_terminal:
                if not rec.stdout:
                    await self._fill(rec, final)
                self._follow_task(rec, final)
                await self._on_finished(rec, final)
            else:
                rec.missing_polls += 1
                if rec.missing_polls >= MISSING_POLLS_BEFORE_GIVING_UP:
                    rec.ended_at = time.time()
                    rec.state = final.state if final else "UNKNOWN"
                    self.store.upsert_job(rec)
                    await self.notifier.send(Outgoing(
                        f"❔ Job {code(rec.job_id)} {esc(rec.name)} left the queue but SLURM accounting has no final state.",
                        job_id=rec.job_id))
                    await self._tell_autopilot(rec, f"Job {rec.job_id} left the queue, but SLURM accounting has no "
                                                    "final state for it. Check its log files.")
                self.store.upsert_job(rec)

        try:
            await self.check_quota()
        except Exception:
            log.exception("quota check failed")
        if self.autopilot:  # timers, check-ins on watched jobs, autopilots running out of time
            try:
                await self.autopilot.tick(live_by_id)
            except Exception:
                log.exception("autopilot tick failed")

        # Finished jobs we never saw in the queue (short jobs between polls).
        if not first:
            for job in recent:
                if job.is_terminal and self.store.get_job(job.job_id) is None and not self._ignored(job, None):
                    rec = await self._register(job)
                    await self._on_finished(rec, job)

    def _merge_array_tasks(self) -> None:
        """Records of single array tasks (123_4), from before arrays were tracked as one job: squeue and sacct
        now only show the array (123), so they would never be seen again. Each becomes its array's record,
        and watches on a task watch the array instead."""
        now = time.time()
        for rec in self.store.active_jobs():
            if "_" not in rec.job_id:
                continue
            parent = array_parent(rec.job_id)
            if self.store.get_job(parent) is None:
                self.store.upsert_job(dataclasses.replace(rec, job_id=parent, missing_polls=0, state="PENDING"))
            rec.ended_at, rec.ignored = now, 1  # silently: the array's record reports on it
            self.store.upsert_job(rec)
            for w in self.store.job_watches(rec.job_id):
                w.jobs = " ".join(dict.fromkeys(array_parent(j) for j in w.job_ids))
                self.store.update_watch(w)
            log.info("job %s is a task of array %s: tracked as the array", rec.job_id, parent)

    async def _register(self, job: Job) -> JobRecord:
        rec = JobRecord(job_id=job.job_id, state=job.state, first_seen=time.time(), last_status_at=time.time())
        await self._fill(rec, job)
        # A job you submitted by hand belongs to the project whose folder it runs in (/assign moves it).
        proj = self.store.project_for_dir(rec.workdir)
        rec.project = proj.name if proj else None
        self.store.upsert_job(rec)
        return rec

    def _assign_buttons(self, rec: JobRecord) -> list[list[tuple[str, str]]]:
        """For a job outside every project's folder: one tap moves it to a project."""
        if rec.project or rec.by_agent:
            return []
        names = [p.name for p in self.store.list_projects()][:6]
        buttons = [(f"📁 {n}", f"assign:{rec.job_id}:{n}") for n in names]
        return [buttons[i:i + 3] for i in range(0, len(buttons), 3)]

    async def _fill(self, rec: JobRecord, job: Job) -> None:
        """Look up a job's name, paths and batch script (scontrol only knows recent jobs)."""
        records = await self.hpc.scontrol_jobs(job.job_id)
        # An array's id shows each task, and its pending tasks as one record whose %a is NO_VAL: one task's is real.
        info = next((r for r in records if r.get("ArrayTaskId", "").isdigit()), records[0] if records else {})
        rec.name = rec.name or info.get("JobName") or job.name
        rec.partition = rec.partition or job.partition or info.get("Partition", "")
        rec.workdir = info.get("WorkDir") or job.workdir or rec.workdir
        rec.stdout = info.get("StdOut") or (f"{rec.workdir}/slurm-{job.job_id}.out" if rec.workdir else "")
        rec.stderr = info.get("StdErr") or rec.stdout
        rec.batch_script = rec.batch_script or await self.hpc.batch_script(job.job_id)
        keep = {k: info[k] for k in ("TimeLimit", "TRES", "NumNodes", "Command", "SubmitTime") if k in info}
        if "tasks" in job.extra or info.get("ArrayJobId"):  # each task writes its own logs: remember their pattern
            array_id = array_parent(job.job_id)
            for key, field in (("stdout_tpl", "StdOut"), ("stderr_tpl", "StdErr")):
                tpl = next((t for r in records if (t := log_template(r.get(field, ""), array_id,
                                                                     r.get("ArrayTaskId", "")))), "")
                if not tpl and not records and rec.workdir:  # scontrol forgot it: SLURM's default name
                    tpl = f"{rec.workdir}/slurm-{array_id}_%a.out"
                if tpl:
                    keep[key] = tpl
        rec.info = json.dumps(keep)
        self._follow_task(rec, job)

    def _follow_task(self, rec: JobRecord, job: Job | None) -> None:
        """An array's logs are those of the task its card shows: the running one, at the end the worst one."""
        info = rec.info_dict
        if not info.get("stdout_tpl") and NO_VAL in (rec.stdout or ""):  # recorded before task logs were followed
            info["stdout_tpl"] = rec.stdout.replace(NO_VAL, "%a")
            info["stderr_tpl"] = (rec.stderr or rec.stdout).replace(NO_VAL, "%a")
        tpl = info.get("stdout_tpl")
        own = task_states(job)
        task = (job.extra.get("task") if job else "") or (own[0][0] if own else "")
        if tpl and task:
            out, err = tpl.replace("%a", task), (info.get("stderr_tpl") or tpl).replace("%a", task)
            if out != rec.stdout:
                rec.stdout, rec.stderr = out, err
                info.pop("progress_log", None)  # (found in the task's stdout)
        if info != rec.info_dict:
            rec.info = json.dumps(info)

    def task_logs(self, rec: JobRecord, job: Job | None) -> list[tuple[str, str, str, str]]:
        """An array's tasks that have run or run now: (task id, state, stdout, stderr)."""
        info = rec.info_dict
        tpl = info.get("stdout_tpl")
        if not tpl:
            return []
        err = info.get("stderr_tpl") or tpl
        return [(t, st, tpl.replace("%a", t), err.replace("%a", t)) for t, st in task_states(job) if st != "PENDING"]

    async def _task_report(self, rec: JobRecord, final: Job) -> str:
        """For the agent, about an array: each task's state and logs, and the end of the other tasks' logs (failed
        ones first; the task the card shows is in the stdout tail already)."""
        tasks = self.task_logs(rec, final)
        if len(tasks) < 2:
            return ""
        lines = ["Its tasks (state, stdout" + (", stderr" if any(o != e for _, _, o, e in tasks) else "") + "):"]
        lines += [f"  task {t}: {st} {o}" + (f" {e}" if e != o else "") for t, st, o, e in tasks[:40]]
        if len(tasks) > 40:
            lines.append(f"  … {len(tasks) - 40} more: {rec.info_dict['stdout_tpl']} with %a = the task id")
        bad = [x for x in tasks if x[1] not in SUCCESS_STATES]
        others = [x for x in bad + [x for x in tasks if x not in bad] if x[2] != rec.stdout][:6]
        for t, st, o, e in others:
            tail = tqdm_cleanup(await self.hpc.try_tail(o, 60 if st not in SUCCESS_STATES else 25))
            if e != o and st not in SUCCESS_STATES:
                tail += "\n" + tqdm_cleanup(await self.hpc.try_tail(e, 30))
            lines.append(f"--- task {t} ({st}) tail of {o} ---\n{tail.strip()[-2500:] or '(empty or unreadable)'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ events
    def _header(self, rec: JobRecord, emoji: str, verb: str) -> str:
        origin = " (submitted by the agent)" if rec.by_agent else ""
        return f"{emoji} <b>Job {esc(rec.job_id)}</b> {esc(rec.name)} {verb}{origin}"

    # ------------------------------------------------------------------ the live job card
    async def _view(self, rec: JobRecord) -> dict:
        """What a card or report needs: the (training) log's tail, the metrics found in it, the agent's metric
        spec and the values of its command."""
        log_path = await progress_log(self.hpc, rec)
        self.store.upsert_job(rec)
        source = log_path or rec.stdout
        text = tqdm_cleanup(await self.hpc.try_tail(source, 200)) if source else ""
        found = all_metrics(text)  # everything the log shows: configured metrics are looked up here
        auto = rank_metrics(found)  # (the most informative, for a card without configured metrics)
        spec = resolve(self.store, rec)
        values, err = (await run_command(self.hpc, rec, spec.command)) if spec and spec.command else ({}, "")
        return {"text": text, "log_path": log_path, "auto": auto, "spec": spec, "values": {**found, **values},
                "err": err}

    def _card_text(self, rec: JobRecord, job: Job | None, view: dict, final: Job | None = None,
                   problem: str = "") -> str:
        state = final.state if final else (job.state if job else rec.state)
        head = f"{STATE_EMOJI.get(state, '•')} <b>{esc(rec.job_id)}</b> {esc(rec.name)}"
        if final:
            success = final.state in SUCCESS_STATES and final.exit_code.startswith("0")
            head += f" · {'finished' if success else esc(final.state) + ' after'} {esc(final.elapsed)}"
        elif job and state == "RUNNING":
            head += f" · {esc(job.reason)}"
        elif job:
            head += f" · ⏳ {esc(state.lower())}: {esc(job.reason)}"
        lines = [head]
        if job and not final and state == "PENDING" and self.store.get(f"eststart:{rec.job_id}"):
            lines.append(f"🕐 {esc(self.store.get(f'eststart:{rec.job_id}'))} (SLURM's estimate)")
        critical, important, progress = metric_lines(view["spec"], view["values"], view["auto"])
        if job and not final and state == "RUNNING":
            elapsed, limit = slurm_seconds(job.elapsed), slurm_seconds(job.time_limit)
            timing = f"⏱ {esc(job.elapsed)} / {esc(job.time_limit)}"
            if elapsed is not None and limit:
                timing += f" ({100 * elapsed // limit}%, {human_seconds(max(0, limit - elapsed))} left)"
            lines.append(timing + (f" · {esc(progress)}" if progress else ""))
        if critical:
            lines.append(f"🎯 <b>{esc(critical)}</b>")
        if important:
            lines.append(f"📈 {esc(important)}")
        tasks = (final or job).extra.get("tasks") if (final or job) else None
        if tasks:
            lines.append(f"📦 array: {esc(tasks)}")
        if not final:
            watches = self.store.job_watches(rec.job_id)
            if watches:
                every = min((w.every for w in watches if w.every), default=0)
                lines.append("⏰ agent watching" + (f" (check-ins every {human_seconds(int(every))})" if every else ""))
            if view["err"]:
                lines.append(f"⚠️ metrics command: {esc(view['err'][:160])}")
            idle = self.idle_state(rec.job_id)
            if idle.get("alerted") and state == "RUNNING":
                lines.append(f"💤 <b>GPU idle</b> for {human_seconds(int(time.time() - idle['since']))} "
                             f"(≤{self.cfg.slurm.idle_gpu_percent}% busy)")
        if problem:
            lines.append(problem)
        for a in fired_alerts(self.store, rec)[:3]:
            lines.append(f"🚨 {esc(a)}")
        return "\n".join(lines)

    def _card_buttons(self, rec: JobRecord, ended: bool = False, failed: bool = False) -> list[list[tuple[str, str]]]:
        row = [("📜 Log", f"log:{rec.job_id}"), ("📈 Plot", f"plot:{rec.job_id}")]
        if failed:
            row.insert(0, ("🔧 Ask agent to fix", f"fix:{rec.job_id}"))
        if not ended:
            row.insert(0, ("📊 Status", f"status:{rec.job_id}"))
            watches = self.store.job_watches(rec.job_id)
            if watches:
                row.append(("✖ watch", f"uw:{watches[0].id}"))
        return [row] + self._assign_buttons(rec)

    async def refresh_card(self, rec: JobRecord, job: Job | None = None) -> None:
        """Update the job's live card (in its jobs topic, silently), and act on the agent's alerts."""
        try:
            view = await self._view(rec)
        except (SSHError, OSError) as e:
            log.warning("job %s: could not read its logs for the card (%s)", rec.job_id, e)
            view = {"text": "", "log_path": "", "auto": {}, "spec": resolve(self.store, rec), "values": {}, "err": ""}
        if job and job.state == "RUNNING":
            await self._check_alerts(rec, job, view)
        text = self._card_text(rec, job, view)
        if self.cfg.slurm.gpu_stats and job and job.state == "RUNNING":
            stats = await self.hpc.gpu_stats(rec.job_id)
            text += "".join(f"\n🎮 {esc(g)}" for g in format_gpu_stats(stats))
        await self._show_card(rec, text, self._card_buttons(rec))

    async def refresh_card_by_id(self, job_id: str) -> None:
        """After something outside the poll changed what the card shows (a watch was set or cancelled)."""
        rec = self.store.get_job(job_id)
        if not rec or rec.ignored:
            return
        if rec.ended_at:
            return  # an ended job's card is final
        try:
            live = {j.job_id: j for j in await self.hpc.squeue()}
        except (SSHError, OSError):
            live = {}
        await self.refresh_card(rec, live.get(job_id))

    async def _show_card(self, rec: JobRecord, text: str, buttons: list[list[tuple[str, str]]]) -> None:
        if hasattr(self.notifier, "card"):
            await self.notifier.card(rec.job_id, text, buttons)
        else:  # a notifier without cards: a (silent) message each time
            await self.notifier.send(Outgoing(text, job_id=rec.job_id, buttons=buttons, silent=True, jobs=True))

    async def _check_alerts(self, rec: JobRecord, job: Job, view: dict) -> None:
        fired = check_alerts(self.store, rec, view["spec"], view["values"], slurm_seconds(job.elapsed))
        for alert, value in fired:
            await self.notifier.send(Outgoing(
                f"🚨 <b>{esc(rec.job_id)}</b> {esc(rec.name)}: {code(alert.text)} (now {esc(value)})"
                + (" - waking the agent" if alert.wake else ""),
                job_id=rec.job_id, jobs=True, reply_to=self.store.card_refs(rec.job_id)))
            if alert.wake and self.autopilot:
                session = (view["spec"].session_id if view["spec"] else None) or rec.session_id
                if session:
                    await self.autopilot.alert_wake(session, rec, alert.text, value)

    async def _maybe_status(self, rec: JobRecord, job: Job) -> None:
        interval = self.cfg.slurm.status_interval_minutes * 60
        if interval <= 0 or time.time() - rec.last_status_at < interval:
            return
        rec.last_status_at = time.time()
        await self.refresh_card(rec, job)
        every = self.cfg.slurm.plot_interval_minutes * 60  # its graphs, refreshed in place
        if every > 0 and time.time() - float(self.store.get(f"plot_at:{rec.job_id}") or 0) >= every \
                and resolve_plot(self.store, rec):
            self._spawn(self.plots(rec))

    # ------------------------------------------------------------------ queued jobs: when they should start
    async def _estimate_starts(self, live: list[Job]) -> None:
        """Every few minutes, SLURM's estimated start of the tracked jobs that wait in the queue: on their cards."""
        pending = {j.job_id: j for j in live if j.state == "PENDING"}
        tracked = {r.job_id: r for r in self.store.active_jobs() if r.job_id in pending}
        if not tracked or time.time() - self._estimated < ESTIMATE_SECONDS:
            return
        self._estimated = time.time()
        try:
            estimates, clock = await self.cluster.estimates()
        except (SSHError, OSError):
            return
        for jid, rec in tracked.items():
            start = estimates.get(jid, ("", ""))[0]
            line = f"starts {when(start, clock)}" if start[:1].isdigit() and clock else ""
            if line != (self.store.get(f"eststart:{jid}") or ""):
                if line:
                    self.store.set(f"eststart:{jid}", line)
                else:
                    self.store.unset(f"eststart:{jid}")
                await self.refresh_card(rec, pending[jid])

    # ------------------------------------------------------------------ idle GPUs
    def idle_state(self, job_id: str) -> dict:
        """{since: first idle sample, util, mem, total, alerted: when it was reported} while a job's GPUs are idle."""
        try:
            return json.loads(self.store.get(f"gpuidle:{job_id}") or "{}")
        except ValueError:
            return {}

    def _maybe_sample_gpus(self, rec: JobRecord, job: Job) -> None:
        c = self.cfg.slurm
        if c.idle_gpu_minutes <= 0 or is_helper(rec.name) or rec.job_id in self._sampling:
            return
        if not has_gpus(job, rec.info_dict.get("TRES", "")):
            return
        key = f"gpusample_at:{rec.job_id}"
        if time.time() - float(self.store.get(key) or 0) < c.idle_gpu_sample_minutes * 60:
            return
        self.store.set(key, str(time.time()))
        self._sampling.add(rec.job_id)
        self._spawn(self._sample_gpus(rec.job_id, job))

    async def _sample_gpus(self, job_id: str, job: Job) -> None:
        """One look at a running job's GPUs (nvidia-smi inside it). All of them at most idle_gpu_percent busy for
        idle_gpu_minutes: reported once (card, message, the watching conversations woken) until they're busy again."""
        try:
            task = job.extra.get("task")  # an array: the task the card shows
            gpus = parse_gpu_stats(await self.hpc.gpu_stats(f"{job_id}_{task}" if task else job_id, readings=5))
        except (SSHError, OSError) as e:
            log.info("job %s: no GPU sample (%s)", job_id, e)
            return
        finally:
            self._sampling.discard(job_id)
        rec = self.store.get_job(job_id)
        if not gpus or not rec or rec.ended_at:
            return  # (no conclusion from a failed sample)
        c = self.cfg.slurm
        state = self.idle_state(job_id)
        util = max(u for u, _, _ in gpus)
        if util > c.idle_gpu_percent:
            if state:
                self.store.unset(f"gpuidle:{job_id}")
                if state.get("alerted"):  # busy again: the card says so no more
                    log.info("job %s: GPUs busy again (%s%%)", job_id, util)
                    await self.refresh_card(rec, job)
            return
        now = time.time()
        state.setdefault("since", now)
        state.update(util=util, mem=sum(m for _, m, _ in gpus), total=sum(t for _, _, t in gpus), n=len(gpus))
        due = not state.get("alerted") and now - state["since"] >= c.idle_gpu_minutes * 60
        if due:
            state["alerted"] = now
        self.store.set(f"gpuidle:{job_id}", json.dumps(state))
        if due:
            await self._report_idle(rec, job, state)

    async def _report_idle(self, rec: JobRecord, job: Job, state: dict) -> None:
        sessions = list(dict.fromkeys(w.session_id for w in self.store.job_watches(rec.job_id)))
        idle_for = human_seconds(int(time.time() - state["since"]))
        mem = f"{state['mem'] / 1024:.1f} of {state['total'] / 1024:.0f} GB GPU memory in use"
        await self.notifier.send(Outgoing(
            f"💤 <b>{esc(rec.job_id)}</b> {esc(rec.name)}: its {state['n']} GPU(s) have been idle for {idle_for} "
            f"(now {state['util']:.0f}% busy, {esc(mem)})"
            + (" - waking the agent watching it" if sessions and self.autopilot else " - no agent is watching it"),
            job_id=rec.job_id, jobs=True, reply_to=self.store.card_refs(rec.job_id)))
        await self.refresh_card(rec, job)
        if self.autopilot:
            for sid in sessions:
                await self.autopilot.idle_wake(sid, rec, idle_for, state)

    # ------------------------------------------------------------------ graphs (plots.py) and card order
    async def _after_end(self, rec: JobRecord) -> None:
        """A job ended: its graphs under its final card, then running cards above it move to the bottom."""
        try:
            await self.plots(rec)
        finally:
            await self._restack(rec)

    async def plots(self, rec: JobRecord) -> tuple[int, str]:
        """Run the job's plotting command (if the agent set one) and show the graphs under its card, replacing the
        previous ones: (how many images, what went wrong)."""
        spec = resolve_plot(self.store, rec)
        if not spec:
            return 0, "no plotting command set for this job (ov plots set)"
        if rec.job_id in self._plotting:
            return 0, "its graphs are being made right now: try again in a minute"
        self._plotting.add(rec.job_id)
        self.store.set(f"plot_at:{rec.job_id}", str(time.time()))
        try:
            images, err = await run_plots(self.hpc, self.store, rec, spec.plot)
        except (SSHError, OSError) as e:
            return 0, f"couldn't run the plotting command: {e}"
        finally:
            self._plotting.discard(rec.job_id)
        self.store.set(f"plot_err:{rec.job_id}", err)
        if images:
            cache = self.cfg.local.state("plots", rec.job_id)
            shutil.rmtree(cache, ignore_errors=True)
            cache.mkdir(parents=True, exist_ok=True)
            for i, (name, data) in enumerate(images):
                (cache / f"{i:02d}-{name}").write_bytes(data)
            if hasattr(self.notifier, "show_plots"):
                await self.notifier.show_plots(rec.job_id, images)
        return len(images), err

    def cached_plots(self, job_id: str) -> list[tuple[str, bytes]]:
        cache = self.cfg.local.state("plots", job_id)
        if not cache.is_dir():
            return []
        return [(p.name.split("-", 1)[-1], p.read_bytes()) for p in sorted(cache.glob("*.png"))]

    async def _restack(self, finished: JobRecord) -> None:
        """Running jobs' cards stay below finished ones in the jobs topic: a running card above the one that just
        ended moves to the bottom (a photo card with its graphs)."""
        if not hasattr(self.notifier, "move_card"):
            return
        last = self.notifier.last_message_ids(finished.job_id)  # {chat: id of its card or graphs, the lowest down}
        running = sorted((r for r in self.store.active_jobs() if r.project == finished.project
                          and r.job_id != finished.job_id and not is_helper(r.name)), key=lambda r: r.first_seen)
        for r in running:
            refs = self.store.card_refs(r.job_id)
            if any(chat in last and mid < last[chat] for chat, mid in refs.items()):
                await self.notifier.move_card(r.job_id, self.cached_plots(r.job_id))

    async def status_message(self, rec: JobRecord, job: Job | None = None, gpu: bool | None = None) -> Outgoing:
        """On request (/job): the card's content plus the last lines of the (training) log. `gpu`: query
        nvidia-smi on the node (default: slurm.gpu_stats)."""
        view = await self._view(rec)
        text = view["text"]
        state = job.state if job else rec.state
        parts = [self._card_text(rec, job, view)]
        if (self.cfg.slurm.gpu_stats if gpu is None else gpu) and state == "RUNNING":
            stats = await self.hpc.gpu_stats(rec.job_id)
            parts += [f"🎮 {esc(g)}" for g in format_gpu_stats(stats)] if stats else ["🎮 (GPU usage unavailable)"]
        if rec.stderr and rec.stderr != rec.stdout and state == "RUNNING":
            err = tqdm_cleanup(await self.hpc.try_tail(rec.stderr, 60))
            for f in find_issues(err)[:3]:  # only real problems; warnings are noise here
                parts.append(f"⚠️ stderr: <b>{esc(f.category)}</b> {code(f.evidence[:200])}")
        shown = "\n".join(text.rstrip("\n").split("\n")[-self.cfg.slurm.log_tail_lines:])
        label = code(view["log_path"].rsplit("/", 1)[-1]) if view["log_path"] else "stdout"
        if shown.strip():
            parts.append(f"📄 {label}, last lines:" + fold_text(shown, 1500))
        return Outgoing("\n".join(p for p in parts if p), job_id=rec.job_id, silent=True,
                        buttons=[[("📈 Plot", f"plot:{rec.job_id}"), ("📜 Log", f"log:{rec.job_id}")]])

    async def check_quota(self, force: bool = False) -> None:
        """At startup and every quota_check_hours: the disk quota (space and files), kept for the agent; warns you once when a
        filesystem passes quota_warn_percent (again after it went back below)."""
        c = self.cfg.slurm
        if not c.quota_command.strip() or c.quota_check_hours <= 0:
            return
        if not force and time.time() - self._quota_checked < c.quota_check_hours * 3600:
            return
        self._quota_checked = time.time()
        rows = parse_quota(await self.hpc.quota(c.quota_command))
        if not rows:
            log.debug("no quota table from %r", c.quota_command)
            return
        self.store.set("quota", json.dumps({"checked": time.time(), "rows": [q.to_dict() for q in rows]}))
        warn = c.quota_warn_percent
        for q in rows:
            for kind, pct, what in (("files", q.files_pct, f"{q.files:,} of {q.files_limit:,} files (inodes)"),
                                    ("space", q.pct, f"{q.used} of {q.limit}")):
                key = f"quota_warned:{q.fs}:{kind}"
                if kind == "files" and not q.files_limit:
                    continue
                if pct >= warn and not self.store.get(key):
                    self.store.set(key, "1")
                    tip = (" Many small files are the usual cause: caches (pip, uv, conda, __pycache__), per-step outputs "
                           "and checkpoints, wandb runs, unpacked datasets - tar them or delete them."
                           if kind == "files" else "")
                    await self.notifier.send(Outgoing(
                        f"💾 <b>{esc(q.fs)} is at {pct}% of its quota</b>: {esc(what)}.{esc(tip)}\nThe agent is told "
                        "too, at the start of its next request."))
                elif pct < warn - 5 and self.store.get(key):
                    self.store.set(key, "")

    async def _on_finished(self, rec: JobRecord, final: Job) -> None:
        rec.state = final.state
        rec.ended_at = time.time()
        self.store.upsert_job(rec)  # mark ended first so a crash below can't cause duplicate reports
        for key in (f"gpuidle:{rec.job_id}", f"gpusample_at:{rec.job_id}", f"eststart:{rec.job_id}"):
            self.store.unset(key)
        watched_here = bool(rec.session_id) and any(w.session_id == rec.session_id
                                                    for w in self.store.job_watches(rec.job_id))
        try:
            context, card, failed = await self.final_report(rec, final)
        except (SSHError, OSError) as e:
            if not is_helper(rec.name):
                record_job(self.store, rec, final, {})
            failed = not (final.state in SUCCESS_STATES and final.exit_code.startswith("0"))
            card = (f"{STATE_EMOJI.get(final.state, '🏁')} <b>{esc(rec.job_id)}</b> {esc(rec.name)} · {esc(final.state)}"
                    f"\n(couldn't read its logs: {esc(e)})")
            context = (f"Job {rec.job_id} final state {final.state}, exit code {final.exit_code}, elapsed "
                       f"{final.elapsed} of {final.time_limit}. (Overseer couldn't read its logs: {e})")
        # The card's final version is the result (no separate report); then its graphs, and running cards below it.
        await self._show_card(rec, card, self._card_buttons(rec, ended=True, failed=failed))
        self._spawn(self._after_end(rec))
        if rec.by_agent and rec.session_id and not watched_here:  # results the agent isn't following
            success = final.state in SUCCESS_STATES and final.exit_code.startswith("0")
            await self.notifier.send(Outgoing(
                f"{STATE_EMOJI.get(final.state, '🏁')} Job {esc(rec.job_id)} {esc(rec.name)} "
                f"{'finished' if success else esc(final.state)}",
                session_id=rec.session_id, job_id=rec.job_id, silent=True))
        await self._tell_autopilot(rec, context)

    async def _tell_autopilot(self, rec: JobRecord, report: str) -> None:
        if self.autopilot:
            try:
                await self.autopilot.job_finished(rec, report)
            except Exception:
                log.exception("autopilot wake-up failed")

    async def final_report(self, rec: JobRecord, final: Job) -> tuple[str, str, bool]:
        """(the facts as plain text for the agent, the card's final text, did it fail)."""
        steps = summarize_steps(await self.hpc.sacct_steps(rec.job_id))
        seff = await self.hpc.seff(rec.job_id)
        out_tail = tqdm_cleanup(await self.hpc.try_tail(rec.stdout, 200))
        err_tail = tqdm_cleanup(await self.hpc.try_tail(rec.stderr, 120)) if rec.stderr and rec.stderr != rec.stdout else ""
        view = await self._view(rec)
        log_path = view["log_path"]
        if log_path:  # the program's own log: its metrics and its traceback, if it crashed
            out_tail = f"{out_tail}\n--- {log_path} ---\n{view['text']}"
        combined = out_tail + "\n" + err_tail
        success = final.state in SUCCESS_STATES and final.exit_code.startswith("0")
        findings = [] if success else find_issues(combined, final.state)
        tb = "" if success else extract_traceback(combined)
        metrics = view["values"]
        critical, important, _ = metric_lines(view["spec"], metrics, view["auto"])
        if not is_helper(rec.name):  # its row in the project's results (the agent adds what it means)
            record_job(self.store, rec, final, snapshot(view["spec"], metrics))

        tasks = final.extra.get("tasks")
        shown = final.extra.get("task")
        task_report = await self._task_report(rec, final) if tasks else ""
        context = "\n".join(x for x in [
            f"Job {rec.job_id} '{rec.name}' final state {final.state}, exit code {final.exit_code}",
            (f"It is a job array of {final.extra.get('task_count')} tasks: {tasks}. The state, exit code and times are "
             f"those of its worst task{f' (task {shown})' if shown else ''}; the stdout tail below is "
             f"{f'task {shown}' if shown else 'one task'}'s") if tasks else "",
            f"Elapsed {final.elapsed} of limit {final.time_limit}; partition {final.partition}; nodes {final.nodelist}",
            f"Allocated TRES: {final.alloc_tres}",
            f"Step summary: {json.dumps(steps)}",
            f"seff:\n{seff}" if seff else "",
            f"Heuristic findings: {[f.category + ': ' + f.evidence for f in findings]}",
            task_report,
            f"Batch script:\n{rec.batch_script[:6000]}" if rec.batch_script else "",
            f"--- stdout tail ({rec.stdout}) ---\n{out_tail[-15000:]}",
            f"--- stderr tail ({rec.stderr}) ---\n{err_tail[-10000:]}" if err_tail else "",
        ] if x)
        # For the agent: the facts first, then the end of the logs; it can read the rest itself.
        for_agent = "\n".join(x for x in [
            context.split("\nBatch script:")[0].split("\n--- stdout")[0],
            f"Your metrics: {critical}" + (f" · {important}" if important else "") if critical or important else "",
            f"Last metrics: {dict(list(metrics.items())[:12])}" if metrics else "",
            f"--- stdout tail ({rec.stdout}) ---\n{out_tail[-7000:]}",
            f"--- stderr tail ({rec.stderr}) ---\n{err_tail[-3000:]}" if err_tail.strip() else "",
        ] if x)
        rec.report = "; ".join(f.category for f in findings)
        self.store.upsert_job(rec)
        # No separate report: the card's final version is the result - with, for a failure, what went wrong.
        problem = ""
        if not success:
            decisive = findings[0].evidence if findings else (tb.strip().splitlines()[-1] if tb.strip() else "")
            mem = (f" (peak RAM {steps['max_rss']}" + (f" of {steps['req_mem']}" if steps.get("req_mem") else "") + ")"
                   if steps.get("max_rss") and "OUT_OF_MEMORY" in final.state else "")
            problem = (f"⚠️ <b>{esc(findings[0].category)}</b>{esc(mem)}" if findings else "") + (
                ("\n" if findings else "⚠️ ") + code(decisive[:300]) if decisive else "")
        card = self._card_text(rec, None, view, final=final, problem=problem)
        return for_agent, card, not success

