"""What the HPC domain watches in the agent's commands: the jobs it submits (sbatch's output), the jobs it cancels
(scancel), and heavy work it runs on the login node instead of a compute node."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..core.notify import code
from ..core.runners import Activity
from .compute import DISPATCHED, LOGIN_NODE_CMD

SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")
PARSABLE_RE = re.compile(r"^\s*(\d+)(?:;\S+)?\s*$", re.M)
# `sbatch` run as a command - not a folder, a `train.sbatch` file or sacct's --parsable2.
_SBATCH = r"(?:^|[\s;&|(`'\"/])sbatch(?![\w.-])"
SBATCH_RE = re.compile(_SBATCH)
SBATCH_PARSABLE_RE = re.compile(_SBATCH + r"[^;&|\n]*\s--parsable(?![\w-])")  # the flag of that same sbatch
SCANCEL_RE = re.compile(r"(?:^|[\s;&|(`'\"/])scancel(?![\w.-])([^;&|\n]*)")
SLURM_QUERY_RE = re.compile(r"(?:^|[\s;&|(`'\"/])(?:sacct|squeue|scontrol|seff|sinfo)(?![\w.-])")


def submitted_jobs(cmd: str, output: str) -> list[str]:
    """Job ids a command submitted: only when it actually runs sbatch (reading a log that says
    "Submitted batch job N", or sacct output, is not a submission)."""
    if not SBATCH_RE.search(cmd):
        return []
    ids = SUBMITTED_RE.findall(output)
    if not ids and SBATCH_PARSABLE_RE.search(cmd):
        ids = PARSABLE_RE.findall(output)
        if SLURM_QUERY_RE.search(cmd):  # other numbers in the output may be sacct/squeue job ids
            ids = ids[:1]
    return ids


def duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


@dataclass
class JobActivity(Activity):
    jobs: list[tuple[str, str]] = field(default_factory=list)  # (job_id, command)
    cancelled: list[str] = field(default_factory=list)  # job ids the agent ran scancel on
    # (command, seconds) that ran on the login node for at least login_warn_seconds (0 = don't check)
    login_node: list[tuple[str, float]] = field(default_factory=list)
    login_warn_seconds: float = 0
    # Numbers seen in command output -> the command (to recognise job ids submitted by scripts)
    numbers: dict[str, str] = field(default_factory=dict)
    on_job: Callable[[str, str], None] | None = None

    def shell_command(self, cmd: str) -> None:
        for args in SCANCEL_RE.findall(cmd):
            for jid in re.findall(r"(?<![\w-])(\d+)(?:_\d+)?(?![\w.])", args):
                if jid not in self.cancelled:
                    self.cancelled.append(jid)

    def shell_output(self, cmd: str, output: str, took: float | None) -> None:
        for n in re.findall(r"(?<![\w.])\d{3,12}(?![\w.])", output[-20_000:]):
            if len(self.numbers) < 5000:
                self.numbers.setdefault(n, cmd)
        if took is not None and self.login_warn_seconds and took >= self.login_warn_seconds \
                and LOGIN_NODE_CMD.search(cmd) and not DISPATCHED.search(cmd):
            self.login_node.append((cmd, took))
        for jid in submitted_jobs(cmd, output):
            if all(jid != j for j, _ in self.jobs):
                self.jobs.append((jid, cmd))
                if self.on_job:
                    self.on_job(jid, cmd)

    def report_lines(self) -> list[str]:
        # heavy work that should have gone to a compute node
        return [f"⚠️ ran {duration(took)} on the login node: {code(cmd[:200])}" for cmd, took in self.login_node[:5]]
