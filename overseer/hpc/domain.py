"""The HPC domain: Overseer's SLURM jobs on the university cluster.

What it adds to the platform: the agent's instructions (its environment on the cluster, `ov` for jobs, metrics,
graphs and results, the disk quota, uv), the project's notes and research log (files on the HPC), the job a
conversation is about, the sandbox's way to the HPC (the SSH master's socket, with Overseer's SSH folder unreadable),
and the jobs the agent submits: seen in its commands (sbatch's output) or asked of SLURM after the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.autopilot import human_interval
from ..core.domain import Domain
from ..core.notify import Outgoing, esc
from ..core.richtext import fold_text
from ..core.runners import Sandbox
from ..core.store import JobRecord
from . import prompts as P
from .activity import JobActivity
from .compute import WORKER_NAME
from .quota import Quota, limits_line, over
from .results import context as results_context
from .slurm import array_parent
from .ssh import SSHError
from .sshlink import ensure_master

if TYPE_CHECKING:
    from ..core.agent import Session
    from .remote import HPC

log = logging.getLogger(__name__)


SKILL = Path(__file__).parent / "skills" / "overseer" / "SKILL.md"  # how to use ov, for the agent


def skills() -> dict[str, str]:
    return {"overseer": SKILL.read_text()}


def hpc_sandbox(cfg) -> Sandbox:
    """The HPC is reached through the unix socket of Overseer's SSH master; its key folder stays unreadable."""
    return Sandbox(deny_read=[cfg.ssh_dir], unix_sockets=[cfg.control_socket], skills=skills())


class HpcDomain(Domain):
    name = "hpc"
    command = "ov"

    def __init__(self, cfg, store, notifier, hpc: HPC):
        super().__init__(cfg, store, notifier)
        self.hpc = hpc
        self._monitor = None
        self.jobs = None  # watches.JobWatches, once the wake-up engine exists
        self._dashboard = None  # dashboard.HpcDashboard, made on first use

    @property
    def monitor(self):
        """The SLURM monitor (set by the app): poked when the agent submits a job, tells the job watches."""
        return self._monitor

    @monitor.setter
    def monitor(self, monitor) -> None:
        self._monitor = monitor
        if monitor is not None and self.jobs is not None:
            monitor.autopilot = self.jobs

    # ------------------------------------------------------------------ the agent's ov command
    def ctl_commands(self, ctl):
        from .commands import HpcCommands
        return HpcCommands(ctl, self)

    # ------------------------------------------------------------------ search
    def search_rows(self) -> list[tuple]:
        from .search import search_rows
        return search_rows(self.store)

    def search_line(self, hit, where: str, snippet: str, html: bool) -> str | None:
        from .search import search_line
        return search_line(hit, where, snippet, html)

    # ------------------------------------------------------------------ the dashboard
    def dashboard_js(self) -> bytes:
        from .dashboard import page_js
        return page_js()

    async def dashboard_api(self, parts: list[str], query: dict[str, str]):
        if self._dashboard is None:
            from .dashboard import HpcDashboard
            self._dashboard = HpcDashboard(self)
        return await self._dashboard.route(parts, query)

    # ------------------------------------------------------------------ the morning digest
    async def digest_sections(self, since: float, now: float) -> list[str]:
        from .digest import HpcDigest
        return await HpcDigest(self).sections(since, now)

    async def digest_closing(self) -> list[str]:
        from .digest import HpcDigest
        return await HpcDigest(self).closing()

    # ------------------------------------------------------------------ Telegram
    def bot_commands(self, bot):
        from .telegram import HpcBot
        return HpcBot(bot, self)

    def auto_prompt(self, goal: str) -> str:
        from .telegram import AUTO_PROMPT
        return AUTO_PROMPT.format(goal=goal)

    async def project_notes_file(self, project) -> tuple[str, str] | None:
        path = f"{project.dir.rstrip('/')}/{P.NOTES_FILE}"
        return path, await self.hpc.try_head(self.remote_path(path), 200)

    # ------------------------------------------------------------------ wake-ups
    def attach_autopilot(self, autopilot) -> None:
        from .watches import JobWatches
        self.jobs = JobWatches(self, autopilot)
        autopilot.register(self.jobs)
        if self._monitor is not None:
            self._monitor.autopilot = self.jobs

    def wake_instructions(self, s: Session) -> str:
        return ("Read the results (open the logs or metrics files for more), record what each showed (`ov result "
                "<job> --verdict ... --conclusion ...`; a lasting lesson: `ov finding`) and update the research log. "
                "If the work goes on, submit the next run and `ov watch` it; if the goal is reached or nothing "
                "sensible is left, say so and set no more wake-ups; if you need the researcher's decision, ask.")

    def ledger_context(self, project: str | None, session_id: str | None = None) -> str:
        return results_context(self.store, project, session_id)

    # ------------------------------------------------------------------ the sandbox
    def sandbox(self) -> Sandbox:
        return hpc_sandbox(self.cfg)

    # ------------------------------------------------------------------ instructions
    def system_prompt(self, s: Session | None) -> str:
        cfg = self.cfg
        domains = cfg.agent.allowed_domains
        setup = cfg.slurm.setup.strip()
        cpu = cfg.slurm.compute.login_cpu_seconds
        ctl = self.agent.ctl if self.agent else None
        prompt = P.SYSTEM_PROMPT.format(
            login_limit=(f" Commands run directly on the login node are stopped after {cpu} CPU seconds." if cpu else ""),
            host=cfg.ssh.host,
            setup=f"{setup}; " if setup else "",
            domains=f" (except {', '.join(domains)})" if domains else "",
            web=self.agent.web_line(s.project if s else None) if self.agent else "",
            ov_rule=" Then register it: `ov job add <id>`." if ctl else "",
        )
        if ctl:
            prompt += P.OV_SECTION.format(min_gap=human_interval(cfg.agent.autopilot.min_checkin_minutes * 60),
                                          repeat=human_interval(cfg.agent.autopilot.repeat_hours * 3600))
        prompt += self.storage_section() + P.PYTHON_SECTION
        proj = self.store.get_project(s.project) if s else None
        if proj:
            worker = bool(s and s.rec.is_worker)
            context = f"\n\nAbout this project, from the researcher:\n{proj.context.strip()}" if proj.context.strip() else ""
            log_path = f"{proj.dir.rstrip('/')}/{P.LOG_FILE}"
            prompt += P.PROJECT_SECTION.format(name=proj.name, dir=proj.dir, context=context,
                                               notes=f"{proj.dir.rstrip('/')}/{P.NOTES_FILE}",
                                               log_rule=(P.WORKER_LOG_RULE if worker else P.LOG_RULE).format(log=log_path))
        return prompt

    def _quota_rows(self) -> list:
        try:
            return [Quota(**r) for r in json.loads(self.store.get("quota") or "{}").get("rows", [])]
        except (ValueError, TypeError):
            return []

    def storage_section(self) -> str:
        """The quota limits (stable: they're in the system prompt) and how to check the usage."""
        cmd = self.cfg.slurm.quota_command.strip()
        rows = self._quota_rows()
        limits = (f"{limits_line(rows)} (from `{cmd}`)" if rows else
                  "not known to Overseer yet" + (f" (`{cmd}` shows them)" if cmd else " (`quota -s` may show them)"))
        return P.STORAGE_SECTION.format(limits=limits, check=f"`{cmd}`" if cmd else "the cluster's quota command")

    def storage_warning(self) -> str:
        """For the start of a request, while a filesystem is nearly full (the usage changes, so not in the system prompt)."""
        full = over(self._quota_rows(), self.cfg.slurm.quota_warn_percent)
        if not full:
            return ""
        return ("<storage_warning>The HPC disk quota is nearly full: " + "; ".join(full) + ". Don't create more files "
                "there than needed; clean up what you can (your own files, caches) and tell the researcher.</storage_warning>")

    # ------------------------------------------------------------------ context
    async def context(self, s: Session, fresh: bool) -> list[str]:
        parts = []
        if fresh and s.project:  # a new CLI session: give it the project's notes
            parts.append(await self.project_notes(s.project))
        if fresh and s.job_id:
            parts.append(await self.job_context(s.job_id))
        if fresh:  # the project's findings and latest results
            parts.append(results_context(self.store, s.project, s.id))
        parts.append(self.storage_warning())
        return [p for p in parts if p]

    async def wake_context(self, s: Session) -> list[str]:
        parts = [await self.research_log(s.project)] if s.project else []
        return parts + [results_context(self.store, s.project, s.id)]

    def remote_path(self, path: str) -> str:
        """'~/x' -> '/home/me/x' (paths are passed quoted, so the remote shell won't expand ~)."""
        if path.startswith("~/") and self.hpc.home:
            return self.hpc.home.rstrip("/") + path[1:]
        return path

    async def project_notes(self, project: str) -> str:
        proj = self.store.get_project(project)
        if not proj or not proj.dir:
            return ""
        path = f"{proj.dir.rstrip('/')}/{P.NOTES_FILE}"
        notes = await self.hpc.try_head(self.remote_path(path), 400)
        if not notes.strip():
            notes = (f"<project_notes path='{path}'>(no notes yet - create the file when there is something "
                     "worth keeping)</project_notes>")
        else:
            notes = f"<project_notes path='{path}'>\n{notes[:12000].strip()}\n</project_notes>"
        return notes + "\n\n" + await self.research_log(project)

    async def research_log(self, project: str) -> str:
        """The project's research log for the agent: its beginning (the goal) and its latest entries."""
        proj = self.store.get_project(project)
        if not proj or not proj.dir:
            return ""
        path = f"{proj.dir.rstrip('/')}/{P.LOG_FILE}"
        remote = self.remote_path(path)
        try:
            res = await self.hpc.run(["stat", "-c", "%s", remote])
            size = int(res.stdout.strip()) if res.ok and res.stdout.strip().isdigit() else -1
        except SSHError:
            size = -1
        if size <= 0:
            return (f"<research_log path='{path}'>(none yet - start it when you run experiments in this "
                    "project)</research_log>")

        async def part(argv: list[str]) -> str:  # bytes, not lines: a line can be huge
            try:
                res = await self.hpc.run(argv, max_output=20_000)
            except SSHError:
                return ""
            return res.stdout if res.ok else ""

        if size <= 12_000:
            body = await part(["head", "-c", "12000", "--", remote])
        else:
            head = await part(["head", "-c", "4000", "--", remote])
            tail = await part(["tail", "-c", "8000", "--", remote])
            body = f"{head.rstrip()}\n[... {size // 1000} KB in all; the latest entries: ...]\n{tail}"
        note = (f"\n(The log is {size // 1000} KB - too long: condense the older entries into a short summary.)"
                if size > P.LOG_CONDENSE_BYTES else "")
        return f"<research_log path='{path}'>\n{body.strip()}\n</research_log>{note}"

    async def job_context(self, job_id: str) -> str:
        rec = self.store.get_job(job_id)
        if not rec:
            return f"(Job {job_id} is not known to Overseer; look it up with sacct.)"
        tail = await self.hpc.try_tail(rec.stdout, 60) if rec.stdout else ""
        return "\n".join(x for x in [
            f"<job id='{rec.job_id}' name='{rec.name}' state='{rec.state}' workdir='{rec.workdir}' "
            f"stdout='{rec.stdout}' stderr='{rec.stderr}'>",
            f"<overseer_analysis>\n{rec.report}\n</overseer_analysis>" if rec.report else "",
            f"<batch_script>\n{rec.batch_script[:8000]}\n</batch_script>" if rec.batch_script else "",
            f"<stdout_tail>\n{tail[-6000:]}\n</stdout_tail>" if tail else "",
            "</job>",
        ] if x)

    # ------------------------------------------------------------------ a run: the jobs it submits
    def new_activity(self, s: Session) -> JobActivity:
        activity = JobActivity(login_warn_seconds=self.cfg.slurm.compute.login_warn_seconds)

        def on_job(job_id: str, command: str) -> None:
            job_id = array_parent(job_id)
            known = self.store.get_job(job_id)
            if known and known.by_agent and known.session_id == s.id and not known.ignored:
                return  # already registered (ov job add) and announced
            self.record_job(s, job_id)
            asyncio.get_running_loop().create_task(self.notifier.send(Outgoing(
                f"🚀 Agent submitted job {esc(job_id)}" + fold_text(command, 600),
                job_id=job_id, session_id=s.id, silent=True)))

        activity.on_job = on_job
        return activity

    async def before_run(self, s: Session) -> Any:
        await ensure_master(self.cfg)
        return {"since": await self._cluster_clock(), "started": time.monotonic()}

    async def after_run(self, s: Session, activity: JobActivity, state: Any, reply: str) -> None:
        await self.find_submissions(activity, state["since"], state["started"], reply)

    async def _cluster_clock(self) -> str:
        try:
            await self.hpc.ensure_connected()
            return await self.hpc.clock()
        except Exception:
            log.debug("could not read the cluster's clock", exc_info=True)
            return ""

    async def find_submissions(self, activity: JobActivity, since: str, started: float, reply: str) -> None:
        """Jobs the agent submitted in ways the output parsing can't see (a script or CLI that calls sbatch):
        your jobs that SLURM says were submitted during this turn, whose id appears in the agent's command
        output or reply. Old jobs it merely looked at weren't submitted during the turn, so they don't count."""
        if not since or not activity.on_job:
            return
        try:
            new = await self.hpc.submitted_since(since, int((time.monotonic() - started) / 60) + 3,
                                                 skip_names=(WORKER_NAME,))  # hpc-run's worker isn't a job of yours
        except Exception:
            log.debug("could not ask SLURM for new jobs", exc_info=True)
            return
        known = {j for j, _ in activity.jobs}
        in_reply = set(re.findall(r"(?<![\w.])\d{3,12}(?![\w.])", reply or ""))
        for jid in new:
            if jid in known or (jid not in activity.numbers and jid not in in_reply):
                continue
            rec = self.store.get_job(jid)
            if rec and rec.by_agent:
                continue
            command = activity.numbers.get(jid) or "(the job id is in the agent's reply)"
            activity.jobs.append((jid, command))
            activity.on_job(jid, command)

    def record_job(self, s: Session, job_id: str) -> None:
        job_id = array_parent(job_id)  # a job array is tracked under the id sbatch printed
        rec = self.store.get_job(job_id) or JobRecord(job_id=job_id, state="PENDING", first_seen=time.time(),
                                                      last_status_at=time.time())
        rec.by_agent = 1
        rec.session_id = s.id
        rec.project = s.project
        self.store.upsert_job(rec)
        if self.monitor:
            self.monitor.poke()

    def move_job(self, job_id: str, project: str | None) -> JobRecord | None:
        """Reassign a job (and the conversation about it) to another project, or the main chat."""
        rec = self.store.get_job(job_id)
        if not rec:
            return None
        rec.project = project
        self.store.upsert_job(rec)
        sess = self.store.session_for_job(job_id)
        if sess and sess.job_id == job_id:
            sess.project = project
            self.store.upsert_session(sess)
            if self.agent and sess.id in self.agent.sessions:
                self.agent.sessions[sess.id].rec.project = project
        return rec
