"""Proposed actions (job submissions / cancellations) and their approval flow.

The agent can only *propose*. A proposal is shown in Telegram with the full script and
the diff of every code change, and nothing reaches SLURM until it is approved. On approval
the script is re-read and its hash compared, so it can't change between review and submit.
"""

from __future__ import annotations

import logging
import posixpath
import time
import uuid
from typing import TYPE_CHECKING

from .config import Config
from .hpc import HPC, sha256_text
from .notify import Notifier, Outgoing, code, esc, pre
from .policy import PolicyError, check_script, scan_diff
from .store import JobRecord, Proposal, Store

if TYPE_CHECKING:
    from .agent import Session
    from .monitor import Monitor

log = logging.getLogger(__name__)


def diffstat(diff: str) -> str:
    files, add, rem = 0, 0, 0
    for line in diff.splitlines():
        if line.startswith("diff "):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            add += 1
        elif line.startswith("-") and not line.startswith("---"):
            rem += 1
    return f"{files} file(s) changed, +{add} −{rem}" if files else "no code changes vs. staged originals"


class Submitter:
    def __init__(self, cfg: Config, hpc: HPC, store: Store, notifier: Notifier):
        self.cfg = cfg
        self.hpc = hpc
        self.store = store
        self.notifier = notifier
        self.monitor: Monitor | None = None

    # ------------------------------------------------------------------ proposing
    async def propose_submit(self, session: Session, script_path: str, summary: str) -> str:
        script_real = await self.hpc.check_write(script_path, base=session.work)
        script = (await self.hpc.read_file(script_real, max_bytes=500_000)).text
        rep = check_script(script, self.hpc.root, self.cfg.limits)
        if not rep.ok:
            raise PolicyError("script rejected by policy:\n- " + "\n- ".join(rep.errors))
        active = self.store.count_active_overseer_jobs()
        if active >= self.cfg.limits.max_active_overseer_jobs:
            raise PolicyError(f"{active} Overseer jobs are already active (limit {self.cfg.limits.max_active_overseer_jobs})")

        diff = await self.hpc.diff(session.orig, session.work)
        warnings = rep.warnings + scan_diff(diff, self.hpc.root)
        pid = uuid.uuid4().hex[:8]
        p = Proposal(
            id=pid, session_id=session.id, kind="submit", script_path=script_real,
            script_sha=sha256_text(script), workdir=session.work,
            run_dir=posixpath.join(session.root, "runs", pid), parent_job=session.job_id or "",
            summary=summary, warnings="\n".join(warnings), created=time.time(),
        )
        self.store.add_proposal(p)

        text = [
            f"📝 <b>Proposed submission</b> {code(pid)}" + (f" (follow-up to job {esc(session.job_id)})" if session.job_id else ""),
            esc(summary),
            f"resources: {esc(rep.resources_line())}",
            f"script: {code(script_real)}",
            f"changes: {esc(diffstat(diff))}",
        ]
        if warnings:
            text.append("⚠️ <b>Warnings</b>\n" + "\n".join(f"• {esc(w)}" for w in warnings[:12]))
        text.append(pre(script, 2500, keep="head"))
        files = [(posixpath.basename(script_real), script.encode())]
        if diff.strip():
            files.append((f"changes-{pid}.diff", diff.encode()))
        auto = not self.cfg.limits.require_approval
        buttons = [] if auto else [[("✅ Approve & submit", f"approve:{pid}"), ("❌ Reject", f"reject:{pid}")]]
        await self.notifier.send(Outgoing("\n".join(text), job_id=session.job_id, session_id=session.id,
                                          buttons=buttons, files=files))
        if auto:
            result = await self.approve(pid)
            return f"Proposal {pid} auto-approved (approval disabled in config). {result}"
        return (f"Proposal {pid} was sent to the user with the full script and diff and is AWAITING APPROVAL. "
                "Do not assume it was submitted; you will be told the job id if it is approved.")

    async def propose_cancel(self, session: Session, job_id: str, reason: str) -> str:
        rec = self.store.get_job(job_id)
        if not rec or not rec.overseer:
            raise PolicyError("only jobs submitted by Overseer can be cancelled")
        if rec.ended_at:
            raise PolicyError(f"job {job_id} already ended ({rec.state})")
        pid = uuid.uuid4().hex[:8]
        self.store.add_proposal(Proposal(id=pid, session_id=session.id, kind="cancel", target_job=job_id,
                                         summary=reason, created=time.time()))
        await self.notifier.send(Outgoing(
            f"🛑 <b>Proposed cancellation</b> of job {code(job_id)} {esc(rec.name)}\n{esc(reason)}",
            job_id=job_id, session_id=session.id,
            buttons=[[("✅ Cancel the job", f"approve:{pid}"), ("❌ Keep it running", f"reject:{pid}")]]))
        return f"Cancellation proposal {pid} is awaiting user approval."

    # ------------------------------------------------------------------ deciding
    async def reject(self, pid: str) -> str:
        if not self.store.claim_proposal(pid, "rejected"):
            return "This proposal was already handled."
        return f"Proposal {pid} rejected."

    async def approve(self, pid: str) -> str:
        p = self.store.get_proposal(pid)
        if not p:
            return "Unknown proposal."
        if not self.store.claim_proposal(pid, "approved"):
            return f"Proposal {pid} was already {p.status}."
        try:
            if p.kind == "cancel":
                return await self._do_cancel(p)
            return await self._do_submit(p)
        except Exception as e:
            log.exception("approved action failed")
            self.store.update_proposal(pid, status="failed")
            return f"❌ Failed: {e}"

    async def _do_submit(self, p: Proposal) -> str:
        script = (await self.hpc.read_file(p.script_path, max_bytes=500_000)).text
        if sha256_text(script) != p.script_sha:
            self.store.update_proposal(p.id, status="failed")
            return "❌ The script changed after it was proposed; not submitting. Ask the agent to propose again."
        rep = check_script(script, self.hpc.root, self.cfg.limits)
        if not rep.ok:
            self.store.update_proposal(p.id, status="failed")
            return "❌ Script no longer passes policy: " + "; ".join(rep.errors)
        if self.store.count_active_overseer_jobs() >= self.cfg.limits.max_active_overseer_jobs:
            self.store.update_proposal(p.id, status="failed")
            return "❌ Too many active Overseer jobs; not submitting."
        session_root = posixpath.dirname(posixpath.dirname(p.run_dir))
        logdir = posixpath.join(session_root, "logs")
        await self.hpc.mkdir(posixpath.join(p.run_dir, "tmp"))
        await self.hpc.mkdir(logdir)
        env = {
            "OVERSEER_RUN_DIR": p.run_dir,
            "OVERSEER_WORKSPACE": p.workdir,
            "OVERSEER_PROPOSAL": p.id,
            "TMPDIR": posixpath.join(p.run_dir, "tmp"),
        }
        job_id = await self.hpc.sbatch(p.script_path, p.workdir, logdir, env)
        out = f"{logdir}/slurm-{job_id}.out"
        self.store.upsert_job(JobRecord(
            job_id=job_id, name=rep.directives.get("job-name", posixpath.basename(p.script_path)), state="PENDING",
            partition=rep.directives.get("partition", ""), workdir=p.workdir, stdout=out,
            stderr=f"{logdir}/slurm-{job_id}.err", batch_script=script, overseer=1, session_id=p.session_id,
            parent_job=p.parent_job or None, proposal_id=p.id, first_seen=time.time(), last_status_at=time.time(),
        ))
        self.store.update_proposal(p.id, status="submitted", job_id=job_id)
        if self.monitor:
            self.monitor.poke()
        return (f"🚀 Submitted job {job_id} (proposal {p.id}, script sha256 {p.script_sha[:12]}).\n"
                f"Outputs: {p.run_dir}\nLog: {out}")

    async def _do_cancel(self, p: Proposal) -> str:
        rec = self.store.get_job(p.target_job)
        if not rec or not rec.overseer:
            return "❌ Refusing: not an Overseer job."
        await self.hpc.scancel(p.target_job)
        self.store.update_proposal(p.id, status="submitted")
        return f"🛑 Cancelled job {p.target_job}."
