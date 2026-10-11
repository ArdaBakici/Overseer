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


PATH_MARK = "__OVERSEER_PATH__="


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
        return parse_squeue(res.stdout)

    async def sacct_recent(self, minutes: int) -> list[Job]:
        res = await self.run(
            self._slurm(["sacct", "-X", "-n", "-P", "-u", self.user, "-S", f"now-{int(minutes)}minutes",
                         "-o", SACCT_LIST_FORMAT]),
            check=True,
        )
        return parse_sacct_list(res.stdout)

    async def sacct_job(self, job_id: str) -> Job | None:
        res = await self.run(self._slurm(["sacct", "-X", "-n", "-P", "-j", job_id, "-o", SACCT_LIST_FORMAT]))
        jobs = [j for j in parse_sacct_list(res.stdout) if j.job_id == job_id]
        return jobs[0] if jobs else None

    async def sacct_steps(self, job_id: str) -> list[dict[str, str]]:
        res = await self.run(self._slurm(["sacct", "-n", "-P", "-j", job_id, "-o", SACCT_STEP_FORMAT]))
        return parse_sacct_steps(res.stdout)

    async def scontrol_job(self, job_id: str) -> dict[str, str]:
        res = await self.run(self._slurm(["scontrol", "show", "job", "-o", job_id]))
        return parse_scontrol(res.stdout) if res.ok else {}

    async def batch_script(self, job_id: str) -> str:
        res = await self.run(self._slurm(["scontrol", "write", "batch_script", job_id, "-"]), max_output=500_000)
        return res.stdout if res.ok else ""

    async def seff(self, job_id: str) -> str:
        try:
            res = await self.run(self._slurm(["seff", job_id]), timeout=30)
        except SSHError:
            return ""
        return res.stdout.strip() if res.ok else ""

    async def gpu_stats(self, job_id: str) -> str:
        try:
            res = await self.run(
                self._slurm(["srun", f"--jobid={job_id}", "--overlap", "--ntasks=1", "nvidia-smi",
                             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                             "--format=csv,noheader"]),
                timeout=45,
            )
        except SSHError:
            return ""
        return res.stdout.strip() if res.ok else ""

