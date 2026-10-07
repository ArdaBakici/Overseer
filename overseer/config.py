"""Configuration loading.

Secrets (Telegram token, Anthropic key) come from environment variables, never from
the YAML file, so the config can be committed or shared safely.
"""

from __future__ import annotations

import os
import posixpath
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class LocalConfig:
    """Where Overseer may write on the machine it runs on.

    Relative paths are resolved against the folder containing config.yaml (Overseer's own
    folder), so "." means "next to the config file".
    """

    # Overseer never writes outside these folders on this machine.
    allowed_roots: list[str] = field(default_factory=lambda: [".", "/tmp"])
    # Persistent state: SQLite DB (jobs, proposals, folder grants), audit log, backups of
    # HPC files the agent edits. Keep it out of /tmp if grants/backups should survive reboots.
    state_dir: str = "./state"
    # Scratch: SSH control socket and other throwaway files.
    tmp_dir: str = "/tmp/overseer"
    # Set by load_config to the config file's folder; relative paths resolve against it.
    base_dir: str = ""

    def _resolve(self, p: str) -> Path:
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = Path(self.base_dir or os.getcwd()) / path
        return path.resolve()

    @property
    def roots(self) -> list[Path]:
        return [self._resolve(r) for r in self.allowed_roots]

    def check(self, path: Path) -> Path:
        """Raise unless ``path`` is inside one of the allowed local folders."""
        p = path.resolve()
        if not any(_path_within(p, r) for r in self.roots):
            raise ConfigError(f"{p} is outside the allowed local folders ({', '.join(map(str, self.roots))})")
        return p

    def state(self, *parts: str) -> Path:
        base = self._resolve(self.state_dir)
        p = self.check(base.joinpath(*parts))
        if not _path_within(p, base):
            raise ConfigError(f"{p} escapes the state dir {base}")
        return p

    def tmp(self, *parts: str) -> Path:
        base = self._resolve(self.tmp_dir)
        p = self.check(base.joinpath(*parts))
        if not _path_within(p, base):
            raise ConfigError(f"{p} escapes the tmp dir {base}")
        return p

    @property
    def db_path(self) -> Path:
        return self.state("overseer.db")

    @property
    def audit_log(self) -> Path:
        return self.state("audit.log")

    @property
    def control_path(self) -> str:
        # Unix socket paths are limited to ~100 chars, so keep the socket in the (short) tmp dir.
        return str(self.tmp("ssh")) + "/cm-%C"

    def sessions_dir(self, sid: str) -> Path:
        return self.state("sessions", sid)


@dataclass
class SSHConfig:
    # Host alias from ~/.ssh/config or user@host. Pubkey auth only (BatchMode=yes).
    host: str = ""
    extra_args: list[str] = field(default_factory=list)  # e.g. ["-J", "jumphost"]
    connect_timeout: int = 15
    command_timeout: int = 90
    control_persist: int = 600
    max_concurrent: int = 4


DEFAULT_DENY_READ = [
    "*/.ssh", "*/.ssh/*", "*/.gnupg", "*/.gnupg/*", "*/.netrc", "*/.git-credentials",
    "*/.pgpass", "*/.aws", "*/.aws/*", "*/.kube", "*/.kube/*", "*/.docker/config.json",
    "*/.config/gcloud", "*/.config/gcloud/*", "*/.cache/huggingface/token",
    "*/.huggingface/token", "*/.env", "*.pem", "*.key", "*/id_rsa*", "*/id_ed25519*",
    "*/id_ecdsa*", "*/.bash_history", "*/.zsh_history",
]
# Files that would make something *run* later (login scripts, cron...) are never writable,
# even inside a write root.
DEFAULT_DENY_WRITE_EXTRA = [
    "*/.bashrc", "*/.bash_profile", "*/.bash_login", "*/.bash_logout", "*/.profile", "*/.zshrc",
    "*/.zprofile", "*/.zshenv", "*/.zlogin", "*/.login", "*/.cshrc", "*/.tcshrc", "*/.config/autostart",
    "*/.config/autostart/*", "*/.config/systemd", "*/.config/systemd/*", "*/.forward", "*/.git/hooks/*",
]


@dataclass
class HPCAccessConfig:
    # Directories on the HPC Overseer may create/modify/delete files in and submit jobs from.
    # More can be granted at runtime from Telegram (the agent asks; you tap Grant).
    write_roots: list[str] = field(default_factory=list)
    # Directories the agent may read (to diagnose failures). Write roots are readable too.
    read_roots: list[str] = field(default_factory=lambda: ["~"])
    deny_read_globs: list[str] = field(default_factory=lambda: list(DEFAULT_DENY_READ))
    deny_write_globs: list[str] = field(default_factory=lambda: list(DEFAULT_DENY_READ + DEFAULT_DENY_WRITE_EXTRA))
    max_copy_mb: int = 200  # max size of a copy_path call
    max_edit_mb: int = 2  # files larger than this can't be modified (they're backed up locally first)


@dataclass
class SlurmConfig:
    poll_interval: int = 60  # seconds between squeue/sacct polls
    status_interval_minutes: int = 60  # periodic progress update for running jobs; 0 = off
    # "all": every job of your user. "overseer": only jobs Overseer submitted.
    track: str = "all"
    notify_queued: bool = True
    log_tail_lines: int = 40
    # Run `srun --overlap nvidia-smi` inside running jobs for GPU utilisation in status updates.
    gpu_stats: bool = False
    # Jobs whose names match any of these globs are ignored (e.g. interactive sessions).
    ignore_name_globs: list[str] = field(default_factory=lambda: ["interactive", "sys/dashboard*", "ood-*"])


@dataclass
class LimitsConfig:
    require_approval: bool = True  # every sbatch/scancel needs a tap on "Approve" in Telegram
    max_gpus_per_job: int = 4
    max_nodes_per_job: int = 1
    max_time: str = "2-00:00:00"
    max_active_overseer_jobs: int = 4
    allowed_partitions: list[str] = field(default_factory=list)  # empty = any
    max_array_tasks: int = 0  # 0 = job arrays not allowed for Overseer-submitted jobs


@dataclass
class TelegramConfig:
    token_env: str = "TELEGRAM_BOT_TOKEN"
    allowed_chat_ids: list[int] = field(default_factory=list)

    @property
    def token(self) -> str:
        tok = os.environ.get(self.token_env, "")
        if not tok:
            raise ConfigError(f"environment variable {self.token_env} is not set")
        return tok


@dataclass
class LLMConfig:
    enabled: bool = True
    model: str = "claude-opus-5-5"
    report_effort: str | None = "medium"  # effort for job reports/diagnoses
    agent_effort: str | None = "high"  # effort for the fixing agent
    adaptive_thinking: bool = True  # set false for models without adaptive thinking (e.g. Haiku 4.5)
    fallbacks: bool = True  # server-side refusal fallback (Claude API only)
    max_tokens: int = 16000
    max_agent_steps: int = 40
    llm_status_updates: bool = False  # one-line LLM summary in periodic status updates


@dataclass
class Config:
    local: LocalConfig = field(default_factory=LocalConfig)
    ssh: SSHConfig = field(default_factory=SSHConfig)
    hpc: HPCAccessConfig = field(default_factory=HPCAccessConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def validate(self) -> None:
        if not self.ssh.host:
            raise ConfigError("ssh.host is required")
        if not self.local.allowed_roots:
            raise ConfigError("local.allowed_roots must list at least one folder")
        for r in self.local.roots:
            if r == Path("/") or r == Path.home():
                raise ConfigError(f"local.allowed_roots entry {r} is too broad; use specific folders")
        self.local.state("x")  # both dirs must sit inside an allowed root
        self.local.tmp("x")
        for r in self.hpc.write_roots:
            check_root_spec(r)
        for r in self.hpc.read_roots:
            if not (r.startswith("/") or r.startswith("~")):
                raise ConfigError(f"read root {r!r} must be absolute or start with ~")
        if self.slurm.track not in ("all", "overseer"):
            raise ConfigError("slurm.track must be 'all' or 'overseer'")
        if not self.telegram.allowed_chat_ids:
            raise ConfigError("telegram.allowed_chat_ids must list at least one chat id")


def check_root_spec(r: str) -> None:
    """Validate a write-root as written in config or a grant request (before ~ expansion)."""
    if not isinstance(r, str) or not (r.startswith("/") or r.startswith("~")):
        raise ConfigError(f"write root {r!r} must be absolute or start with ~")
    if "\n" in r or "\x00" in r:
        raise ConfigError("write root contains control characters")
    if posixpath.normpath(r) in ("/", "~"):
        raise ConfigError(f"write root {r!r} is too broad")


def _path_within(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def _build(cls: type, data: dict[str, Any] | None, path: str) -> Any:
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"unknown keys in {path or 'config'}: {', '.join(sorted(unknown))}")
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{path}.{name}".strip("."))
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | os.PathLike) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(Config, raw, "")
    cfg.local.base_dir = str(Path(path).resolve().parent)
    cfg.validate()
    return cfg
