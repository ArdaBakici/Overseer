"""Polls SLURM and turns state changes into Telegram notifications."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import time

from .compute import WORKER_NAME
from .config import Config
from .diagnose import Reporter, extract_metrics, extract_traceback, find_issues, find_log_paths, tqdm_cleanup
from .hpc import HPC
from .notify import Notifier, Outgoing, code, esc, pre
from .quota import parse_quota
from .richtext import fold_text
from .slurm import (STATE_EMOJI, SUCCESS_STATES, Job, format_gpu_stats, gpus_from_tres, human_seconds,
                    slurm_seconds, summarize_steps)
from .ssh import SSHError
from .store import JobRecord, Store

log = logging.getLogger(__name__)

MISSING_POLLS_BEFORE_GIVING_UP = 10


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
        if job.name == WORKER_NAME:  # hpc-run's compute worker: shown in /health, not reported as a job
            if job.state == "RUNNING" or self.worker is None:
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
                elif job.state == "RUNNING":
                    await self._on_started(rec, job)
                elif self.cfg.slurm.notify_queued:
                    await self._on_queued(rec, job)
                continue
            if not rec.stdout:  # submitted by the agent: fill in details on first sight
                await self._fill(rec, job)
            prev = rec.state
            rec.state = job.state
            rec.missing_polls = 0
            if prev != "RUNNING" and job.state == "RUNNING":
                await self._on_started(rec, job)
            elif job.state == "RUNNING":
                await self._maybe_status(rec, job)
            self.store.upsert_job(rec)

        if adopted:
            lines = [f"{STATE_EMOJI.get(r.state, '•')} {code(r.job_id)} {esc(r.name)} — {esc(r.state)}" for r in adopted]
            await self.notifier.send(Outgoing("👀 <b>Overseer online.</b> Tracking:\n" + "\n".join(lines), silent=True))

        # Jobs we know about that left the queue.
        for rec in self.store.active_jobs():
            if rec.job_id in live_by_id:
                continue
            if rec.name == WORKER_NAME:  # hpc-run's worker, registered by mistake: never reported
                rec.ended_at, rec.ignored = time.time(), 1
                self.store.upsert_job(rec)
                continue
            final = recent_by_id.get(rec.job_id) or await self.hpc.sacct_job(rec.job_id)
            if final and final.is_terminal:
                if not rec.stdout:
                    await self._fill(rec, final)
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
        info = await self.hpc.scontrol_job(job.job_id)
        rec.name = rec.name or info.get("JobName") or job.name
        rec.partition = rec.partition or job.partition or info.get("Partition", "")
        rec.workdir = info.get("WorkDir") or job.workdir or rec.workdir
        rec.stdout = info.get("StdOut") or (f"{rec.workdir}/slurm-{job.job_id}.out" if rec.workdir else "")
        rec.stderr = info.get("StdErr") or rec.stdout
        rec.batch_script = rec.batch_script or await self.hpc.batch_script(job.job_id)
        rec.info = json.dumps({k: info[k] for k in ("TimeLimit", "TRES", "NumNodes", "Command", "SubmitTime") if k in info})

    # ------------------------------------------------------------------ events
    def _header(self, rec: JobRecord, emoji: str, verb: str) -> str:
        origin = " (submitted by the agent)" if rec.by_agent else ""
        return f"{emoji} <b>Job {esc(rec.job_id)}</b> {esc(rec.name)} {verb}{origin}"

    async def _on_queued(self, rec: JobRecord, job: Job) -> None:
        gres = f" · {esc(job.gres)}" if job.gres and job.gres != "N/A" else ""
        await self.notifier.send(Outgoing(
            f"{self._header(rec, '📥', 'queued')}\npartition {esc(job.partition)} · limit {esc(job.time_limit)}{gres}\nreason: {esc(job.reason)}",
            job_id=rec.job_id, silent=True, buttons=self._assign_buttons(rec)))

    async def _on_started(self, rec: JobRecord, job: Job) -> None:
        rec.state = "RUNNING"
        rec.last_status_at = time.time()
        self.store.upsert_job(rec)
        gres = f" · {esc(job.gres)}" if job.gres and job.gres != "N/A" else ""
        text = (f"{self._header(rec, '🚀', 'started')}\nnode(s) {esc(job.reason)} · partition {esc(job.partition)} · "
                f"limit {esc(job.time_limit)}{gres}\nlog: {code(rec.stdout)}")
        await self.notifier.send(Outgoing(text, job_id=rec.job_id,
                                          buttons=[[("📜 Log", f"log:{rec.job_id}"), ("📊 Status", f"status:{rec.job_id}"),
                                                    ("📈 Plot", f"plot:{rec.job_id}")]] + self._assign_buttons(rec)))

    async def _maybe_status(self, rec: JobRecord, job: Job) -> None:
        interval = self.cfg.slurm.status_interval_minutes * 60
        if interval <= 0 or time.time() - rec.last_status_at < interval:
            return
        rec.last_status_at = time.time()
        await self.notifier.send(await self.status_message(rec, job))

    async def status_message(self, rec: JobRecord, job: Job | None = None, gpu: bool | None = None) -> Outgoing:
        """Where the job stands: time used/left, metrics, GPU use, and the last few lines of its
        (training) log. `gpu`: query nvidia-smi on the node (default: slurm.gpu_stats)."""
        log_path = await progress_log(self.hpc, rec)
        self.store.upsert_job(rec)
        source = log_path or rec.stdout
        text = tqdm_cleanup(await self.hpc.try_tail(source, max(200, self.cfg.slurm.log_tail_lines)))
        shown = "\n".join(text.rstrip("\n").split("\n")[-self.cfg.slurm.log_tail_lines:])
        state = job.state if job else rec.state
        parts = [f"📊 <b>Job {esc(rec.job_id)}</b> {esc(rec.name)} — {esc(state)}"]
        if job:
            elapsed, limit = slurm_seconds(job.elapsed), slurm_seconds(job.time_limit)
            timing = f"⏱ {esc(job.elapsed)} of {esc(job.time_limit)}"
            if elapsed is not None and limit:
                timing += f" ({100 * elapsed // limit}%, {human_seconds(max(0, limit - elapsed))} left)"
            where = f"node {esc(job.reason)}" if state == "RUNNING" else f"waiting: {esc(job.reason)}"
            parts.append(f"{timing} · {where}")
        metrics = extract_metrics(text)
        if metrics:
            parts.append("📈 " + " · ".join(f"{esc(k)} {esc(v)}" for k, v in metrics.items()))
        line = await self.reporter.status_line(text)
        if line:
            parts.append(f"🧠 {esc(line)}")
        if (self.cfg.slurm.gpu_stats if gpu is None else gpu) and state == "RUNNING":
            stats = await self.hpc.gpu_stats(rec.job_id)
            parts += [f"🎮 {esc(g)}" for g in format_gpu_stats(stats)] if stats else ["🎮 (GPU usage unavailable)"]
        if rec.stderr and rec.stderr != rec.stdout and state == "RUNNING":
            err = tqdm_cleanup(await self.hpc.try_tail(rec.stderr, 60))
            for f in find_issues(err)[:3]:  # only real problems; warnings are noise here
                parts.append(f"⚠️ stderr: <b>{esc(f.category)}</b> {code(f.evidence[:200])}")
        label = code(log_path.rsplit("/", 1)[-1]) if log_path else "stdout"
        if shown.strip():
            parts.append(f"📄 {label}, last lines:" + pre(shown, 1500))
        return Outgoing("\n".join(p for p in parts if p), job_id=rec.job_id, silent=True,
                        buttons=[[("📈 Plot", f"plot:{rec.job_id}"), ("📜 Log", f"log:{rec.job_id}")]])

    async def check_quota(self, force: bool = False) -> None:
        """Every quota_check_minutes: the disk quota (space and files), kept for the agent; warns you once when a
        filesystem passes quota_warn_percent (again after it went back below)."""
        c = self.cfg.slurm
        if not c.quota_command.strip() or c.quota_check_minutes <= 0:
            return
        if not force and time.time() - self._quota_checked < c.quota_check_minutes * 60:
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
        # A watched job is analysed by the agent itself right after: skip the one-shot LLM summary.
        analyze = not (self.autopilot and self.autopilot.handles(rec))
        try:
            msg, context = await self.final_report(rec, final, analyze=analyze)
        except (SSHError, OSError) as e:
            msg = Outgoing(f"{self._header(rec, STATE_EMOJI.get(final.state, '🏁'), final.state)}\n"
                           f"(could not build full report: {esc(e)})", job_id=rec.job_id)
            context = (f"Job {rec.job_id} final state {final.state}, exit code {final.exit_code}, elapsed "
                       f"{final.elapsed} of {final.time_limit}. (Overseer couldn't read its logs: {e})")
        await self.notifier.send(msg)
        await self._tell_autopilot(rec, context)

    async def _tell_autopilot(self, rec: JobRecord, report: str) -> None:
        if self.autopilot:
            try:
                await self.autopilot.job_finished(rec, report)
            except Exception:
                log.exception("autopilot wake-up failed")

    async def final_report(self, rec: JobRecord, final: Job, analyze: bool = True) -> tuple[Outgoing, str]:
        """The Telegram report, and the same facts as compact plain text (for the agent)."""
        steps = summarize_steps(await self.hpc.sacct_steps(rec.job_id))
        seff = await self.hpc.seff(rec.job_id)
        out_tail = tqdm_cleanup(await self.hpc.try_tail(rec.stdout, 200))
        err_tail = tqdm_cleanup(await self.hpc.try_tail(rec.stderr, 120)) if rec.stderr and rec.stderr != rec.stdout else ""
        log_path = await progress_log(self.hpc, rec)
        if log_path:  # the program's own log: its metrics and its traceback, if it crashed
            out_tail = (f"{out_tail}\n--- {log_path} ---\n"
                        + tqdm_cleanup(await self.hpc.try_tail(log_path, 200)))
        combined = out_tail + "\n" + err_tail
        success = final.state in SUCCESS_STATES and final.exit_code.startswith("0")
        findings = [] if success else find_issues(combined, final.state)
        tb = "" if success else extract_traceback(combined)
        metrics = extract_metrics(out_tail)

        context = "\n".join([
            f"Job {rec.job_id} '{rec.name}' final state {final.state}, exit code {final.exit_code}",
            f"Elapsed {final.elapsed} of limit {final.time_limit}; partition {final.partition}; nodes {final.nodelist}",
            f"Allocated TRES: {final.alloc_tres}",
            f"Step summary: {json.dumps(steps)}",
            f"seff:\n{seff}" if seff else "",
            f"Heuristic findings: {[f.category + ': ' + f.evidence for f in findings]}",
            f"Batch script:\n{rec.batch_script[:6000]}" if rec.batch_script else "",
            f"--- stdout tail ({rec.stdout}) ---\n{out_tail[-15000:]}",
            f"--- stderr tail ({rec.stderr}) ---\n{err_tail[-10000:]}" if err_tail else "",
        ])
        analysis = await self.reporter.final_analysis(context) if analyze else ""
        # For the agent (autopilot): the facts first, then the end of the logs; it can read the rest itself.
        for_agent = "\n".join(x for x in [
            context.split("\nBatch script:")[0].split("\n--- stdout")[0],
            f"Last metrics: {metrics}" if metrics else "",
            f"--- stdout tail ({rec.stdout}) ---\n{out_tail[-7000:]}",
            f"--- stderr tail ({rec.stderr}) ---\n{err_tail[-3000:]}" if err_tail.strip() else "",
        ] if x)
        rec.report = analysis or "; ".join(f.category for f in findings)
        self.store.upsert_job(rec)

        emoji = STATE_EMOJI.get(final.state, "🏁")
        verb = "finished" if success else f"ended: {final.state}"
        gpus = gpus_from_tres(final.alloc_tres)
        parts = [
            self._header(rec, emoji, verb),
            f"exit {esc(final.exit_code)} · elapsed {esc(final.elapsed)} / {esc(final.time_limit)}"
            + (f" · {gpus} GPU" if gpus else "")
            + (f" · peak RAM {esc(steps['max_rss'])}" if steps.get("max_rss") else "")
            + (f" (req {esc(steps['req_mem'])})" if steps.get("req_mem") else ""),
        ]
        if metrics:
            parts.append("last metrics: " + ", ".join(f"{esc(k)}={esc(v)}" for k, v in metrics.items()))
        if analysis:
            parts.append(f"\n🧠 {esc(analysis)}")
        elif findings:
            parts.append("\n".join(f"⚠️ <b>{esc(f.category)}</b>: {esc(f.hint)}" for f in findings))
        # Logs fold away: the first lines show, the rest on tap (the full tail is attached on failure).
        if tb:
            parts.append("Traceback:" + fold_text(tb, 2500))
        elif combined.strip():
            parts.append("End of the log:" + fold_text(combined.strip(), 2500))
        buttons = [[("📜 Full log", f"log:{rec.job_id}"), ("📈 Plot", f"plot:{rec.job_id}")]]
        if not success:
            buttons[0].insert(0, ("🔧 Ask agent to fix", f"fix:{rec.job_id}"))
        buttons += self._assign_buttons(rec)
        files = []
        if out_tail.strip() and not success:
            files.append((f"job-{rec.job_id}-stdout-tail.log", out_tail.encode()))
        return Outgoing("\n".join(parts), job_id=rec.job_id, buttons=buttons, files=files), for_agent
