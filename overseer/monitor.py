"""Polls SLURM and turns state changes into Telegram notifications."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import time

from .config import Config
from .diagnose import Reporter, extract_metrics, extract_traceback, find_issues, tqdm_cleanup
from .hpc import HPC
from .notify import Notifier, Outgoing, code, esc, pre
from .slurm import STATE_EMOJI, SUCCESS_STATES, Job, gpus_from_tres, summarize_steps
from .ssh import SSHError
from .store import JobRecord, Store

log = logging.getLogger(__name__)

MISSING_POLLS_BEFORE_GIVING_UP = 10


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
        if rec and rec.by_agent:
            return False
        if self.cfg.slurm.track == "agent":
            return True
        return any(fnmatch.fnmatch(job.name, g) for g in self.cfg.slurm.ignore_name_globs)

    async def poll(self, first: bool = False) -> None:
        await self.hpc.ensure_connected()
        live = await self.hpc.squeue()
        live_by_id = {j.job_id: j for j in live}
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
                    await self.notifier.send(Outgoing(
                        f"❔ Job {code(rec.job_id)} {esc(rec.name)} left the queue but SLURM accounting has no final state.",
                        job_id=rec.job_id))
                self.store.upsert_job(rec)

        # Finished jobs we never saw in the queue (short jobs between polls).
        if not first:
            for job in recent:
                if job.is_terminal and self.store.get_job(job.job_id) is None and not self._ignored(job, None):
                    rec = await self._register(job)
                    await self._on_finished(rec, job)

    async def _register(self, job: Job) -> JobRecord:
        rec = JobRecord(job_id=job.job_id, state=job.state, first_seen=time.time(), last_status_at=time.time())
        await self._fill(rec, job)
        self.store.upsert_job(rec)
        return rec

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
            job_id=rec.job_id, silent=True))

    async def _on_started(self, rec: JobRecord, job: Job) -> None:
        rec.state = "RUNNING"
        rec.last_status_at = time.time()
        self.store.upsert_job(rec)
        gres = f" · {esc(job.gres)}" if job.gres and job.gres != "N/A" else ""
        text = (f"{self._header(rec, '🚀', 'started')}\nnode(s) {esc(job.reason)} · partition {esc(job.partition)} · "
                f"limit {esc(job.time_limit)}{gres}\nlog: {code(rec.stdout)}")
        await self.notifier.send(Outgoing(text, job_id=rec.job_id,
                                          buttons=[[("📜 Log", f"log:{rec.job_id}"), ("📊 Status", f"status:{rec.job_id}")]]))

    async def _maybe_status(self, rec: JobRecord, job: Job) -> None:
        interval = self.cfg.slurm.status_interval_minutes * 60
        if interval <= 0 or time.time() - rec.last_status_at < interval:
            return
        rec.last_status_at = time.time()
        await self.notifier.send(await self.status_message(rec, job))

    async def status_message(self, rec: JobRecord, job: Job | None = None) -> Outgoing:
        tail = tqdm_cleanup(await self.hpc.try_tail(rec.stdout, self.cfg.slurm.log_tail_lines))
        err_tail = ""
        if rec.stderr and rec.stderr != rec.stdout:
            err_tail = tqdm_cleanup(await self.hpc.try_tail(rec.stderr, 15))
        parts = [f"📊 <b>Job {esc(rec.job_id)}</b> {esc(rec.name)} — {esc(job.state if job else rec.state)}"]
        if job:
            parts.append(f"elapsed {esc(job.elapsed)} / {esc(job.time_limit)} · {esc(job.reason)}")
        metrics = extract_metrics(tail)
        if metrics:
            parts.append("metrics: " + ", ".join(f"{esc(k)}={esc(v)}" for k, v in metrics.items()))
        line = await self.reporter.status_line(tail)
        if line:
            parts.append(f"🧠 {esc(line)}")
        if self.cfg.slurm.gpu_stats and job and job.state == "RUNNING":
            gpu = await self.hpc.gpu_stats(rec.job_id)
            if gpu:
                parts.append("GPU:" + pre(gpu, 800))
        parts.append(pre(tail, 1800))
        if err_tail.strip():
            parts.append("stderr:" + pre(err_tail, 800))
        return Outgoing("\n".join(p for p in parts if p), job_id=rec.job_id, silent=True)

    async def _on_finished(self, rec: JobRecord, final: Job) -> None:
        rec.state = final.state
        rec.ended_at = time.time()
        self.store.upsert_job(rec)  # mark ended first so a crash below can't cause duplicate reports
        try:
            msg = await self.final_report(rec, final)
        except (SSHError, OSError) as e:
            msg = Outgoing(f"{self._header(rec, STATE_EMOJI.get(final.state, '🏁'), final.state)}\n"
                           f"(could not build full report: {esc(e)})", job_id=rec.job_id)
        await self.notifier.send(msg)

    async def final_report(self, rec: JobRecord, final: Job) -> Outgoing:
        steps = summarize_steps(await self.hpc.sacct_steps(rec.job_id))
        seff = await self.hpc.seff(rec.job_id)
        out_tail = tqdm_cleanup(await self.hpc.try_tail(rec.stdout, 200))
        err_tail = tqdm_cleanup(await self.hpc.try_tail(rec.stderr, 120)) if rec.stderr and rec.stderr != rec.stdout else ""
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
        analysis = await self.reporter.final_analysis(context)
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
        if tb:
            parts.append(pre(tb, 1500))
        else:
            parts.append(pre(combined.strip(), 1200))
        buttons = [[("📜 Full log", f"log:{rec.job_id}")]]
        if not success:
            buttons[0].insert(0, ("🔧 Ask agent to fix", f"fix:{rec.job_id}"))
        files = []
        if out_tail.strip() and not success:
            files.append((f"job-{rec.job_id}-stdout-tail.log", out_tail.encode()))
        return Outgoing("\n".join(parts), job_id=rec.job_id, buttons=buttons, files=files)
