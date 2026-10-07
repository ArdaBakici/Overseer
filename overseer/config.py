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
class SSHConfig:
    # Host alias from ~/.ssh/config or user@host. Pubkey auth only (BatchMode=yes).
    host: str = ""
    extra_args: list[str] = field(default_factory=list)  # e.g. ["-J", "jumphost"]
    connect_timeout: int = 15
    command_timeout: int = 90
    # Local path for the multiplexed connection socket. %C is a hash of the connection.
    control_path: str = "~/.ssh/overseer-%C"
    control_persist: int = 600
    max_concurrent: int = 4


@dataclass
class SandboxConfig:
    # The ONLY directory on the HPC that Overseer may create, modify, delete or run from.
    root: str = "/tmp/overseer"
    # Directories the agent may *read* (to diagnose failures and stage code). "~" is the HPC home.
    read_roots: list[str] = field(default_factory=lambda: ["~"])
    # Never readable, even inside read_roots (fnmatch, '*' also matches '/').
    deny_read_globs: list[str] = field(
        default_factory=lambda: [
            "*/.ssh", "*/.ssh/*", "*/.gnupg", "*/.gnupg/*", "*/.netrc", "*/.git-credentials",
            "*/.pgpass", "*/.aws", "*/.aws/*", "*/.kube", "*/.kube/*", "*/.docker/config.json",
            "*/.config/gcloud", "*/.config/gcloud/*", "*/.cache/huggingface/token",
            "*/.huggingface/token", "*/.env", "*.pem", "*.key", "*/id_rsa*", "*/id_ed25519*",
            "*/id_ecdsa*", "*/.bash_history", "*/.zsh_history",
        ]
    )
    # Max size of code copied into the sandbox per staging call.
    max_stage_mb: int = 200


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
class StateConfig:
    dir: str = "./state"

    @property
    def db_path(self) -> Path:
        return Path(self.dir).expanduser() / "overseer.db"

    @property
    def audit_log(self) -> Path:
        return Path(self.dir).expanduser() / "audit.log"


@dataclass
class Config:
    ssh: SSHConfig = field(default_factory=SSHConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    state: StateConfig = field(default_factory=StateConfig)

    def validate(self) -> None:
        if not self.ssh.host:
            raise ConfigError("ssh.host is required")
        root = self.sandbox.root
        if not root.startswith("/") or posixpath.normpath(root) in ("/", "/tmp", "/home", "/scratch"):
            raise ConfigError(
                "sandbox.root must be an absolute, dedicated directory (e.g. /tmp/overseer), "
                f"not {root!r}"
            )
        if "," in root or any(c.isspace() for c in root):
            raise ConfigError("sandbox.root must not contain commas or whitespace")
        if self.slurm.track not in ("all", "overseer"):
            raise ConfigError("slurm.track must be 'all' or 'overseer'")
        if not self.telegram.allowed_chat_ids:
            raise ConfigError("telegram.allowed_chat_ids must list at least one chat id")
        self.sandbox.root = posixpath.normpath(root)


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
    cfg.validate()
    return cfg
