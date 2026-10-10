"""Read-only queries Overseer itself runs on the HPC (for monitoring and reports).

The agent does not use this: it has its own shell and reaches the HPC with `ssh hpc ...`.
"""

from __future__ import annotations

import logging

from .slurm import (
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


class HPC:
    def __init__(self, runner: SSHRunner):
        self.run = runner.run
        self.user = ""
        self.home = ""

    async def connect(self) -> None:
        self.user = (await self.run(["whoami"], check=True)).stdout.strip()
        self.home = (await self.run(["printenv", "HOME"], check=True)).stdout.strip() or f"/home/{self.user}"

    async def ensure_connected(self) -> None:
        if not self.user:
            await self.connect()

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
        res = await self.run(["squeue", "-h", "-u", self.user, "-o", SQUEUE_FORMAT], check=True)
        return parse_squeue(res.stdout)

    async def sacct_recent(self, minutes: int) -> list[Job]:
        res = await self.run(
            ["sacct", "-X", "-n", "-P", "-u", self.user, "-S", f"now-{int(minutes)}minutes", "-o", SACCT_LIST_FORMAT],
            check=True,
        )
        return parse_sacct_list(res.stdout)

    async def sacct_job(self, job_id: str) -> Job | None:
        res = await self.run(["sacct", "-X", "-n", "-P", "-j", job_id, "-o", SACCT_LIST_FORMAT])
        jobs = [j for j in parse_sacct_list(res.stdout) if j.job_id == job_id]
        return jobs[0] if jobs else None

    async def sacct_steps(self, job_id: str) -> list[dict[str, str]]:
        res = await self.run(["sacct", "-n", "-P", "-j", job_id, "-o", SACCT_STEP_FORMAT])
        return parse_sacct_steps(res.stdout)

    async def scontrol_job(self, job_id: str) -> dict[str, str]:
        res = await self.run(["scontrol", "show", "job", "-o", job_id])
        return parse_scontrol(res.stdout) if res.ok else {}

    async def batch_script(self, job_id: str) -> str:
        res = await self.run(["scontrol", "write", "batch_script", job_id, "-"], max_output=500_000)
        return res.stdout if res.ok else ""

    async def seff(self, job_id: str) -> str:
        try:
            res = await self.run(["seff", job_id], timeout=30)
        except SSHError:
            return ""
        return res.stdout.strip() if res.ok else ""

    async def gpu_stats(self, job_id: str) -> str:
        try:
            res = await self.run(
                ["srun", f"--jobid={job_id}", "--overlap", "--ntasks=1", "nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader"],
                timeout=45,
            )
        except SSHError:
            return ""
        return res.stdout.strip() if res.ok else ""

