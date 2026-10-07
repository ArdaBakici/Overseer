"""Actions that need the user's approval in Telegram.

* ``submit``  - sbatch a script. Shown with the full script and a diff of every file the agent
  changed; the script is re-read and its hash compared at approval time.
* ``cancel``  - scancel a job Overseer submitted.
* ``access``  - make a new HPC folder readable or writable.
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
        if line.startswith("+++ "):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            add += 1
        elif line.startswith("-") and not line.startswith("---"):
            rem += 1
    return f"{files} file(s) changed, +{add} −{rem}" if files else "no file changes by the agent"


class Submitter:
    def __init__(self, cfg: Config, hpc: HPC, store: Store, notifier: Notifier):
        self.cfg = cfg
        self.hpc = hpc
        self.store = store
        self.notifier = notifier
        self.monitor: Monitor | None = None

    # ------------------------------------------------------------------ proposing
    async def propose_submit(self, session: Session, script_path: str, summary: str, workdir: str = "") -> str:
        script_real = await self.hpc.check_write(script_path)
        workdir_real = await self.hpc.check_write(workdir or posixpath.dirname(script_real), allow_root=True)
        script = (await self.hpc.read_file(script_real, max_bytes=500_000)).text
        rep = check_script(script, self.hpc.write_roots, self.hpc.home, self.cfg.limits)
        if not rep.ok:
            raise PolicyError("script rejected by policy:\n- " + "\n- ".join(rep.errors))
        active = self.store.count_active_overseer_jobs()
        if active >= self.cfg.limits.max_active_overseer_jobs:
            raise PolicyError(f"{active} Overseer jobs are already active (limit {self.cfg.limits.max_active_overseer_jobs})")

        diff = await session.changes.diff(self.hpc)
        warnings = rep.warnings + scan_diff(diff, self.hpc.write_roots, self.hpc.home)
        pid = uuid.uuid4().hex[:8]
        self.store.add_proposal(Proposal(
            id=pid, session_id=session.id, kind="submit", script_path=script_real,
            script_sha=sha256_text(script), workdir=workdir_real, parent_job=session.job_id or "",
            summary=summary, warnings="\n".join(warnings), created=time.time(),
        ))
        text = [
            f"📝 <b>Proposed submission</b> {code(pid)}" + (f" (follow-up to job {esc(session.job_id)})" if session.job_id else ""),
            esc(summary),
            f"resources: {esc(rep.resources_line())}",
            f"script: {code(script_real)}\nworkdir: {code(workdir_real)}",
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

    async def propose_access(self, session: Session | None, path: str, mode: str, reason: str) -> str:
        real = await self.hpc.resolve_root_request(path, mode)
        if self.hpc.has_access(real, mode):
            return f"You already have {mode} access to {real}."
        pid = uuid.uuid4().hex[:8]
        self.store.add_proposal(Proposal(id=pid, session_id=session.id if session else "", kind="access",
                                         target_path=real, mode=mode, summary=reason, created=time.time()))
        verb = "write to (create/modify/delete files in)" if mode == "write" else "read"
        await self.notifier.send(Outgoing(
            f"🔐 <b>Access request</b>: the agent wants to {verb}\n{code(real)}\n\nWhy: {esc(reason)}",
            session_id=session.id if session else None,
            buttons=[[(f"✅ Grant {mode}", f"approve:{pid}"), ("❌ Deny", f"reject:{pid}")]]))
        return (f"Access request {pid} for {mode} access to {real} was sent to the user. END YOUR TURN NOW and wait; "
                "you will be resumed automatically with the user's decision.")

    async def grant(self, path: str, mode: str, reason: str) -> str:
        """Grant access directly (used by /grant and approved access proposals)."""
        real = await self.hpc.resolve_root_request(path, mode)
        self.store.add_grant(real, mode, reason)
        await self.hpc.grant(real, mode)
        return f"🔓 Granted {mode} access to {real}."

    async def revoke(self, path: str) -> str:
        real = self.hpc.expand(path)
        if real not in self.hpc.grants:
            real = await self.hpc.realpath(real)
        self.store.remove_grant(real)
        if await self.hpc.revoke(real):
            return f"🔒 Revoked access to {real}."
        return f"{real} was not a runtime grant (folders from the config file can only be removed there)."

    # ------------------------------------------------------------------ deciding
    async def reject(self, pid: str) -> str:
        if not self.store.claim_proposal(pid, "rejected"):
            return "This proposal was already handled."
        p = self.store.get_proposal(pid)
        if p and p.kind == "access":
            return f"Denied {p.mode} access to {p.target_path}."
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
            if p.kind == "access":
                result = await self.grant(p.target_path, p.mode, p.summary)
                self.store.update_proposal(p.id, status="done")
                return result
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
        rep = check_script(script, self.hpc.write_roots, self.hpc.home, self.cfg.limits)
        if not rep.ok:
            self.store.update_proposal(p.id, status="failed")
            return "❌ Script no longer passes policy: " + "; ".join(rep.errors)
        if self.store.count_active_overseer_jobs() >= self.cfg.limits.max_active_overseer_jobs:
            self.store.update_proposal(p.id, status="failed")
            return "❌ Too many active Overseer jobs; not submitting."
        job_id = await self.hpc.sbatch(p.script_path, p.workdir)
        info = await self.hpc.scontrol_job(job_id)
        out = info.get("StdOut") or f"{p.workdir}/slurm-{job_id}.out"
        err = info.get("StdErr") or out
        self.hpc.allow_job_paths(out, err)
        self.store.upsert_job(JobRecord(
            job_id=job_id, name=info.get("JobName") or rep.directives.get("job-name", posixpath.basename(p.script_path)),
            state="PENDING", partition=rep.directives.get("partition", ""), workdir=p.workdir, stdout=out,
            stderr=err, batch_script=script, overseer=1, session_id=p.session_id,
            parent_job=p.parent_job or None, proposal_id=p.id, first_seen=time.time(), last_status_at=time.time(),
        ))
        self.store.update_proposal(p.id, status="submitted", job_id=job_id)
        if self.monitor:
            self.monitor.poke()
        return f"🚀 Submitted job {job_id} (proposal {p.id}, script sha256 {p.script_sha[:12]}).\nLog: {out}"

    async def _do_cancel(self, p: Proposal) -> str:
        rec = self.store.get_job(p.target_job)
        if not rec or not rec.overseer:
            return "❌ Refusing: not an Overseer job."
        await self.hpc.scancel(p.target_job)
        self.store.update_proposal(p.id, status="submitted")
        return f"🛑 Cancelled job {p.target_job}."
