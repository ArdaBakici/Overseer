"""The HPC domain's configuration: its sections of config.yaml (ssh, slurm) and the paths derived from them.

The platform's Config (core/config.py) builds these sections for `domain: hpc` (the default) and reaches them as
cfg.ssh, cfg.slurm, cfg.ssh_dir, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import UNIX_SOCKET_MAX, ConfigError


@dataclass
class SSHConfig:
    """How to reach the HPC. Overseer keeps its own key, ssh config and known_hosts in ``dir``
    (inside the Overseer folder) and never uses ~/.ssh. Set it up with `overseer ssh setup`."""

    host: str = "hpc"  # the name the agent uses: `ssh hpc '...'`
    hostname: str = ""  # e.g. login.cluster.edu
    user: str = ""  # your HPC username
    port: int = 22
    proxy_jump: str | None = None  # e.g. "netid@gateway.cluster.edu" (uses the same key)
    dir: str = "./state/ssh"  # key, ssh config, known_hosts
    key_name: str = "id_ed25519"
    extra_options: dict[str, str] = field(default_factory=dict)  # extra ssh_config options for the host
    connect_timeout: int = 15
    command_timeout: int = 90
    max_concurrent: int = 4


@dataclass
class ComputeConfig:
    """Where the agent runs anything heavier than a quick command (`hpc-run` on the HPC side).

    A shared "worker" job holds a compute node so commands start without queueing; it ends
    itself after `idle_minutes` unused. Commands asking for other resources (or a GPU) get an
    allocation of their own."""

    partition: str = "compute"
    cpus: int = 8
    mem: str = "32G"
    worker_hours: float = 8  # time limit of the worker job
    idle_minutes: int = 45  # the worker ends itself after this long unused (clusters cancel idle jobs too)
    wait_minutes: int = 15  # how long hpc-run waits for a node before giving up
    gpu_partition: str = "nvidia"  # for `hpc-run --gpu`
    gpu_gres: str = "gpu:1"
    sbatch_args: list[str] = field(default_factory=list)  # extra flags for every allocation, e.g. ["--account=x"]
    # Guardrail against mistakes (not a security boundary): commands the agent runs directly on
    # the login node get this many CPU seconds (ulimit -t); 0 = no limit. Commands that hand work
    # to SLURM (sbatch, srun, salloc, hpc-run) aren't limited - SLURM would pass the limit on to the job.
    login_cpu_seconds: int = 120
    login_warn_seconds: int = 120  # flag commands that ran this long on the login node


@dataclass
class SlurmConfig:
    poll_interval: int = 60  # seconds between squeue/sacct polls
    # How often a running job's live card is refreshed (edited in place, silently) and the agent's alerts on it
    # are checked; 0 = only when it starts and ends.
    status_interval_minutes: int = 10
    # Graphs from the agent's plotting command (ov plots set): refreshed in place this often while a job runs.
    plot_interval_minutes: int = 60
    # "all": every job of your user. "agent": only jobs the agent submitted.
    track: str = "all"
    notify_queued: bool = True
    log_tail_lines: int = 12  # lines of the (training) log shown in status messages
    # Run `srun --overlap nvidia-smi` inside running jobs for GPU utilisation in status updates.
    gpu_stats: bool = False
    # Idle GPUs: a running job whose GPUs all stay at most idle_gpu_percent busy for idle_gpu_minutes gets a 💤 line
    # on its card, you're told, and the conversations watching it (ov watch) are woken. 0 minutes = off.
    # Its GPUs are sampled every idle_gpu_sample_minutes (srun --overlap nvidia-smi inside the job).
    idle_gpu_minutes: float = 30
    idle_gpu_percent: int = 5
    idle_gpu_sample_minutes: float = 10
    # Jobs whose names match any of these globs are ignored (e.g. interactive sessions).
    ignore_name_globs: list[str] = field(default_factory=lambda: ["interactive", "sys/dashboard*", "ood-*"])
    # SSH commands run in a non-login shell, where SLURM is often not on PATH. Overseer picks up
    # PATH from a login shell (`bash -lc`) once per connection. If SLURM still isn't found, put
    # what you'd type to get it here, e.g. "module load slurm" - it runs before PATH is read.
    setup: str = ""
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    # Disk quota: the command that prints it ("" = don't check). Overseer reads it at startup and every quota_check_hours,
    # gives the agent the limits (space and number of files), and warns you and the agent when a filesystem
    # passes quota_warn_percent. Understood: tables like `myquota`'s (size, quota (%), files, quota (%)).
    quota_command: str = "myquota"
    quota_check_hours: float = 24
    quota_warn_percent: int = 80


SECTIONS = {"ssh": SSHConfig, "slurm": SlurmConfig}


# Paths derived from the sections (all inside the allowed local folders), as cfg.<name>.
def _ssh_dir(cfg) -> Path:
    return cfg.local.resolve(cfg.ssh.dir)


DERIVED = {
    "ssh_dir": _ssh_dir,
    "ssh_config_file": lambda cfg: _ssh_dir(cfg) / "config",
    "ssh_key": lambda cfg: _ssh_dir(cfg) / cfg.ssh.key_name,
    "control_socket": lambda cfg: cfg.local.tmp("ssh", "master.sock"),
    # Lives outside ssh_dir: the agent may read this one (it holds no secrets).
    "agent_ssh_config": lambda cfg: cfg.local.tmp("agent-ssh", "config"),
}


def validate(cfg) -> None:
    if not cfg.ssh.hostname:
        raise ConfigError("ssh.hostname is required (the HPC login node, e.g. login.cluster.edu)")
    if not cfg.ssh.host or any(c.isspace() for c in cfg.ssh.host) or cfg.ssh.host in ("*", "localhost"):
        raise ConfigError("ssh.host must be a simple alias such as 'hpc'")
    cfg.ssh_dir  # noqa: B018 - validates it is inside allowed roots
    if len(str(cfg.control_socket)) > UNIX_SOCKET_MAX:
        raise ConfigError(f"local.tmp_dir is too long for a unix socket path ({cfg.control_socket}); use e.g. /tmp/overseer")
    if cfg.slurm.track not in ("all", "agent"):
        raise ConfigError("slurm.track must be 'all' or 'agent'")
    for name, proj in cfg.projects.items():
        if not proj.dir.startswith(("/", "~")):
            raise ConfigError(f"projects.{name}.dir must be an absolute path on the HPC")
