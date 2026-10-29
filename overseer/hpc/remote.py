"""Read-only queries Overseer itself runs on the HPC (for monitoring and reports).

The agent does not use this: it has its own shell and reaches the HPC with `ssh hpc ...`.
"""

from __future__ import annotations

import re

import logging

from .slurm import (
    fold_arrays,
    SACCT_LIST_FORMAT,
    SACCT_STEP_FORMAT,
    SQUEUE_FORMAT,
    Job,
    parse_sacct_list,
    parse_sacct_steps,
    parse_scontrol,
    parse_squeue,
)
from .ssh import SSHError, SSHRunner

log = logging.getLogger(__name__)


PATH_MARK = "__OVERSEER_PATH__="
GPU_STEP = "overseer-gpu"  # the name of Overseer's nvidia-smi steps inside jobs: left out of the jobs' step summaries


class HPC:
    def __init__(self, runner: SSHRunner, setup: str = ""):
        self.run = runner.run
        self.setup = setup  # slurm.setup, e.g. "module load slurm"
        self.user = ""
        self.home = ""
        self.path = ""  # PATH of a login shell (+ setup); used for SLURM commands

    async def connect(self) -> None:
        self.user = (await self.run(["whoami"], check=True)).stdout.strip()
        self.home = (await self.run(["printenv", "HOME"], check=True)).stdout.strip() or f"/home/{self.user}"
        await self.discover_path()

    async def discover_path(self) -> None:
        """Commands over SSH run in a non-login shell, where clusters often don't have SLURM on PATH
        (it's set up by /etc/profile.d or `module load`). Read PATH once from a login shell."""
        script = (f"{self.setup}\n" if self.setup else "") + f'echo "{PATH_MARK}$PATH"'
        try:
            res = await self.run(["bash", "-lc", script], timeout=60)
        except SSHError as e:
            log.warning("could not read PATH from a login shell: %s", e)
            return
        # Login profiles may print banners; only trust our marker line.
        for line in res.stdout.splitlines():
            if line.startswith(PATH_MARK):
                self.path = line[len(PATH_MARK):].strip()
        if not self.path:
            log.warning("could not read PATH from a login shell (exit %s): %s", res.returncode, res.stderr.strip()[:300])

    def _slurm(self, argv: list[str]) -> list[str]:
        return ["env", f"PATH={self.path}", *argv] if self.path else argv

    async def ensure_connected(self) -> None:
        if not self.user:
            await self.connect()

    async def try_head(self, path: str, lines: int) -> str:
        """First lines of a file, or "" if it can't be read."""
        if not path:
            return ""
        try:
            res = await self.run(["head", "-n", str(int(lines)), "--", path], max_output=100_000)
        except SSHError:
            return ""
        return res.stdout if res.ok else ""

    async def try_tail(self, path: str, lines: int) -> str:
        """Last lines of a (log) file, or a short explanation if it can't be read."""
        if not path:
            return ""
        try:
            res = await self.run(["tail", "-n", str(int(lines)), "--", path], max_output=200_000)
        except SSHError as e:
            return f"<could not read {path}: {e}>"
        if not res.ok:
            return f"<could not read {path}: {res.stderr.strip()[:300]}>"
        return res.stdout

    # ------------------------------------------------------------------ SLURM
    async def squeue(self) -> list[Job]:
        res = await self.run(self._slurm(["squeue", "-h", "-u", self.user, "-o", SQUEUE_FORMAT]), check=True)
        return fold_arrays(parse_squeue(res.stdout))  # an array's tasks are one job

    async def sacct_recent(self, minutes: int, fold: bool = True) -> list[Job]:
        res = await self.run(
            self._slurm(["sacct", "-X", "-n", "-P", "-u", self.user, "-S", f"now-{int(minutes)}minutes",
                         "-o", SACCT_LIST_FORMAT]),
            check=True,
        )
        jobs = parse_sacct_list(res.stdout)
        return fold_arrays(jobs) if fold else jobs

    async def quota(self, command: str) -> str:
        """The output of the cluster's quota command (e.g. myquota), or "" if it fails."""
        try:
            res = await self.run(self._slurm(["sh", "-c", command]), timeout=60, max_output=100_000)
        except SSHError:
            return ""
        return res.stdout if res.ok else ""

    async def clock(self) -> str:
        """The cluster's local time (to compare with SLURM's timestamps, which are in that time zone)."""
        res = await self.run(["date", "+%Y-%m-%dT%H:%M:%S"])
        return res.stdout.strip() if res.ok else ""

    async def submitted_since(self, since: str, minutes: int, skip=lambda name: False) -> list[str]:
        """Ids of your jobs submitted at or after `since` (cluster time), looking back `minutes`."""
        res = await self.run(self._slurm(["sacct", "-X", "-n", "-P", "-u", self.user, "-S", f"now-{int(minutes)}minutes",
                                          "-o", "JobID,Submit,JobName"]))
        ids: list[str] = []
        for line in res.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 3 and skip(parts[2]):
                continue
            if len(parts) >= 2 and parts[1] >= since and re.fullmatch(r"\d+(_.+)?", parts[0]):
                jid = parts[0].split("_")[0]
                if jid not in ids:
                    ids.append(jid)
        return ids

    async def sacct_job(self, job_id: str) -> Job | None:
        res = await self.run(self._slurm(["sacct", "-X", "-n", "-P", "-j", job_id, "-o", SACCT_LIST_FORMAT]))
        # (for an array's id, sacct lists its tasks: 123_1, 123_[2-9]; folded into one job 123)
        rows = parse_sacct_list(res.stdout)
        jobs = [j for j in fold_arrays(rows) if j.job_id == job_id] or [j for j in rows if j.job_id == job_id]
        return jobs[0] if jobs else None

    async def sacct_steps(self, job_id: str) -> list[dict[str, str]]:
        res = await self.run(self._slurm(["sacct", "-n", "-P", "-j", job_id, "-o", SACCT_STEP_FORMAT]))
        return [r for r in parse_sacct_steps(res.stdout) if r.get("name") != GPU_STEP]  # (Overseer's own looks)

    async def scontrol_job(self, job_id: str) -> dict[str, str]:
        jobs = await self.scontrol_jobs(job_id)
        return jobs[0] if jobs else {}

    async def scontrol_jobs(self, job_id: str) -> list[dict[str, str]]:
        """`scontrol show job`: one record, or for an array's id one per task (and one for its pending tasks)."""
        res = await self.run(self._slurm(["scontrol", "show", "job", "-o", job_id]))
        if not res.ok:
            return []
        return [d for d in (parse_scontrol(line) for line in res.stdout.splitlines() if line.strip()) if d]

    async def batch_script(self, job_id: str) -> str:
        res = await self.run(self._slurm(["scontrol", "write", "batch_script", job_id, "-"]), max_output=500_000)
        return res.stdout if res.ok else ""

    # ---- the cluster (cluster.py): what's free, when your queued jobs start, your fairshare
    async def _lines(self, argv: list[str], timeout: float = 60) -> str:
        try:
            res = await self.run(self._slurm(argv), timeout=timeout, max_output=5_000_000)
        except SSHError:
            return ""
        return res.stdout if res.ok else ""

    async def nodes(self) -> list[dict[str, str]]:
        out = await self._lines(["scontrol", "show", "node", "-o"])
        return [d for d in (parse_scontrol(line) for line in out.splitlines() if line.strip()) if d.get("NodeName")]

    async def sinfo_partitions(self) -> str:
        return await self._lines(["sinfo", "-h", "-o", "%P|%a|%l"])

    async def pending_starts(self) -> str:
        return await self._lines(["squeue", "-h", "-u", self.user, "-t", "PENDING", "--start", "-o", "%i|%S|%r"])

    async def sshare(self) -> str:
        return await self._lines(["sshare", "-h", "-P", "-U", "-u", self.user, "-o", "Account,FairShare"], timeout=30)

    async def seff(self, job_id: str) -> str:
        try:
            res = await self.run(self._slurm(["seff", job_id]), timeout=30)
        except SSHError:
            return ""
        return res.stdout.strip() if res.ok else ""

    async def gpu_stats(self, job_id: str, readings: int = 1) -> str:
        """nvidia-smi inside a running job: index, name, busy %, memory used and total per GPU (CSV). With
        `readings` > 1, that many readings a second apart (one line per GPU each)."""
        query = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader"]
        if readings > 1:
            query = ["sh", "-c", f"for i in $(seq {int(readings)}); do {' '.join(query)}; sleep 1; done"]
        try:
            res = await self.run(
                self._slurm(["srun", f"--jobid={job_id}", "--overlap", "--ntasks=1", f"--job-name={GPU_STEP}",
                             *query]),
                timeout=45 + readings,
            )
        except SSHError:
            return ""
        return res.stdout.strip() if res.ok else ""

