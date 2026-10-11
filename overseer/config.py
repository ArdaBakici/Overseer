"""Configuration loading.

Secrets (Telegram token, API keys) come from environment variables, never from the YAML
file, so the config can be committed or shared safely.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

UNIX_SOCKET_MAX = 107  # sun_path limit on Linux, minus the terminating NUL


class ConfigError(ValueError):
    pass


def _path_within(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


@dataclass
class LocalConfig:
    """Where Overseer (and the agents it runs) may write on the machine it runs on.

    Relative paths are resolved against the folder containing config.yaml (Overseer's own
    folder), so "." means "next to the config file".
    """

    # Nothing on this machine is written outside these folders.
    allowed_roots: list[str] = field(default_factory=lambda: [".", "/tmp"])
    # Persistent state: SQLite DB, audit log, the SSH key/config, the CLIs' logins and sessions.
    state_dir: str = "./state"
    # Scratch: SSH control socket and the agents' working directories.
    tmp_dir: str = "/tmp/overseer"
    # Secrets file (TELEGRAM_BOT_TOKEN=..., API keys) loaded at startup. Variables already set in
    # the real environment win. The agent can never read this file.
    env_file: str = ".env"
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

    def resolve(self, p: str) -> Path:
        """Resolve a configured local path and make sure it is inside allowed_roots."""
        return self.check(self._resolve(p))

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
    def env_path(self) -> Path:
        return self._resolve(self.env_file)

    @property
    def db_path(self) -> Path:
        return self.state("overseer.db")

    @property
    def audit_log(self) -> Path:
        return self.state("audit.log")


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
    status_interval_minutes: int = 60  # periodic progress update for running jobs; 0 = off
    # "all": every job of your user. "agent": only jobs the agent submitted.
    track: str = "all"
    notify_queued: bool = True
    log_tail_lines: int = 12  # lines of the (training) log shown in status messages
    # Run `srun --overlap nvidia-smi` inside running jobs for GPU utilisation in status updates.
    gpu_stats: bool = False
    # Jobs whose names match any of these globs are ignored (e.g. interactive sessions).
    ignore_name_globs: list[str] = field(default_factory=lambda: ["interactive", "sys/dashboard*", "ood-*"])
    # SSH commands run in a non-login shell, where SLURM is often not on PATH. Overseer picks up
    # PATH from a login shell (`bash -lc`) once per connection. If SLURM still isn't found, put
    # what you'd type to get it here, e.g. "module load slurm" - it runs before PATH is read.
    setup: str = ""
    compute: ComputeConfig = field(default_factory=ComputeConfig)


@dataclass
class TelegramConfig:
    token_env: str = "TELEGRAM_BOT_TOKEN"
    allowed_chat_ids: list[int] = field(default_factory=list)
    # Telegram's native rich messages (Bot API 10.1+) for agent replies: real tables, collapsible
    # sections, formulas, pictures inside the reply. Falls back to regular formatted messages.
    rich_messages: bool = True
    # Stream the agent's progress as a live draft with a Stop button (private chats), instead of
    # a message that is edited every few seconds.
    streaming: bool = True

    @property
    def token(self) -> str:
        tok = os.environ.get(self.token_env, "")
        if not tok:
            raise ConfigError(f"environment variable {self.token_env} is not set")
        return tok


PROXY_ENV = ["HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
             "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE"]


@dataclass
class ClaudeCodeConfig:
    command: str = "claude"
    model: str | None = None  # e.g. "opus", "sonnet", "claude-opus-5-5"; None = Claude Code's default
    effort: str | None = None  # low | medium | high | xhigh | max
    # CLAUDE_CONFIG_DIR (login, sessions, settings) for Overseer's runs. Log in once with:
    #   overseer login claude_code
    home: str = "./state/claude"
    # Environment variables passed through (everything else, e.g. the Telegram token, is stripped).
    pass_env: list[str] = field(default_factory=lambda: ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", *PROXY_ENV])
    # Merged into the settings Overseer passes with --settings (sandbox keys can't be loosened).
    extra_settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodexConfig:
    command: str = "codex"
    model: str | None = None  # e.g. "gpt-5.5-codex"; None = codex default
    reasoning_effort: str | None = None  # low | medium | high | xhigh
    # CODEX_HOME for Overseer's runs. Log in once with: overseer login codex
    home: str = "./state/codex"
    pass_env: list[str] = field(default_factory=lambda: ["OPENAI_API_KEY", "CODEX_API_KEY", *PROXY_ENV])
    web_search: bool = True  # the model's own web search (runs on OpenAI's side)
    extra_config: list[str] = field(default_factory=list)  # extra `-c key=value` overrides


@dataclass
class AutopilotConfig:
    """Autopilot: the agent works on its own, woken by the jobs it watches and the timers it sets.
    It asks first (an approval card in Telegram), or you start it with /auto."""

    default_wakeups: int = 20  # wake-ups (a watched job ending, or a timer) per approval, unless changed
    max_wakeups: int = 100  # upper bound, also for the approval card's +/- buttons
    default_hours: float = 48  # how long an approval lasts
    max_hours: float = 336
    checkin_minutes: int = 60  # check-in interval the card offers when the agent doesn't ask for one
    min_checkin_minutes: int = 15  # check-ins on running jobs: never more often than this
    max_checkins_per_job: int = 30
    # One-off self-wakes, without autopilot: "wake me when job X ends" / "in 2h".
    oneoff_max_pending: int = 3  # per conversation
    oneoff_max_hours: float = 168  # how far ahead a one-off timer may be
    oneoff_chain: int = 3  # one-off wake-ups in a row without a message from you; then it must ask for autopilot


@dataclass
class AgentConfig:
    enabled: bool = True
    # claude_code or codex. Can be switched at runtime from Telegram with /backend.
    backend: str = "claude_code"
    timeout_minutes: int = 60  # max length of one agent run
    # How the commands the agent runs are reported in Telegram:
    #   summary - one message after each request; live - batched every few seconds; off
    report_commands: str = "summary"
    # While the agent works, one Telegram message that updates itself every few seconds with its
    # latest reasoning summary, the running command and the elapsed time (removed when it answers).
    live_progress: bool = True
    # Internet hosts the agent's local shell may reach (it reaches the HPC through Overseer's
    # SSH connection, which needs no internet access). Empty = no internet from the shell.
    allowed_domains: list[str] = field(default_factory=list)
    # Extra local paths the agent's commands may never read (the SSH key folder, ~/.ssh,
    # Overseer's state and well-known daemon sockets are always blocked).
    deny_read: list[str] = field(default_factory=list)
    # LLM summary of finished jobs (root cause for failures) using the same backend.
    job_reports: bool = True
    llm_status_updates: bool = False  # one-line LLM progress summary in periodic status messages
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    autopilot: AutopilotConfig = field(default_factory=AutopilotConfig)


BACKENDS = ("claude_code", "codex")
PROJECT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
RESERVED_PROJECT_NAMES = {"main", "all", "general", "none"}


@dataclass
class ProjectConfig:
    """A project: its own Telegram topic, conversations, jobs and agent settings."""

    dir: str = ""  # project root on the HPC; hand-submitted jobs running under it belong to the project
    context: str = ""  # standing instructions for the agent in this project
    # Unset = inherited from the main chat when the project is first created; then /backend,
    # /model and /effort in the project's topic change them. Changing them here applies at the next start.
    backend: str | None = None
    model: str | None = None
    effort: str | None = None


@dataclass
class Config:
    local: LocalConfig = field(default_factory=LocalConfig)
    ssh: SSHConfig = field(default_factory=SSHConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    projects: dict[str, ProjectConfig] = field(default_factory=dict)

    # ---- derived SSH paths (all inside the allowed local folders)
    @property
    def ctl_socket(self) -> Path:
        """Overseer's control socket for the agent's `ov` command."""
        return self.local.tmp("ctl.sock")

    @property
    def ssh_dir(self) -> Path:
        return self.local.resolve(self.ssh.dir)

    @property
    def ssh_config_file(self) -> Path:
        return self.ssh_dir / "config"

    @property
    def ssh_key(self) -> Path:
        return self.ssh_dir / self.ssh.key_name

    @property
    def control_socket(self) -> Path:
        return self.local.tmp("ssh", "master.sock")

    @property
    def agent_ssh_config(self) -> Path:
        # Lives outside ssh_dir: the agent may read this one (it holds no secrets).
        return self.local.tmp("agent-ssh", "config")

    def validate(self) -> None:
        for name, proj in self.projects.items():
            if not PROJECT_NAME_RE.match(name) or name in RESERVED_PROJECT_NAMES:
                raise ConfigError(f"project name {name!r}: use lowercase letters, digits, - and _ (not main/all)")
            if not proj.dir.startswith(("/", "~")):
                raise ConfigError(f"projects.{name}.dir must be an absolute path on the HPC")
            if proj.backend and proj.backend not in BACKENDS:
                raise ConfigError(f"projects.{name}.backend must be one of {', '.join(BACKENDS)}")
        if not self.ssh.hostname:
            raise ConfigError("ssh.hostname is required (the HPC login node, e.g. login.cluster.edu)")
        if not self.ssh.host or any(c.isspace() for c in self.ssh.host) or self.ssh.host in ("*", "localhost"):
            raise ConfigError("ssh.host must be a simple alias such as 'hpc'")
        if not self.local.allowed_roots:
            raise ConfigError("local.allowed_roots must list at least one folder")
        for r in self.local.roots:
            if r == Path("/") or r == Path.home():
                raise ConfigError(f"local.allowed_roots entry {r} is too broad; use specific folders")
        self.local.state("x")  # both dirs must sit inside an allowed root
        self.local.tmp("x")
        self.ssh_dir  # noqa: B018 - validates it is inside allowed roots
        if len(str(self.control_socket)) > UNIX_SOCKET_MAX:
            raise ConfigError(f"local.tmp_dir is too long for a unix socket path ({self.control_socket}); use e.g. /tmp/overseer")
        if self.agent.backend not in BACKENDS:
            raise ConfigError(f"agent.backend must be one of {', '.join(BACKENDS)}")
        if self.agent.report_commands not in ("summary", "live", "off"):
            raise ConfigError("agent.report_commands must be summary, live or off")
        self.local.resolve(self.agent.claude_code.home)
        self.local.resolve(self.agent.codex.home)
        if self.slurm.track not in ("all", "agent"):
            raise ConfigError("slurm.track must be 'all' or 'agent'")
        if not self.telegram.allowed_chat_ids:
            raise ConfigError("telegram.allowed_chat_ids must list at least one chat id")


RENAMED = {"AutopilotConfig": {"default_rounds": "default_wakeups", "max_rounds": "max_wakeups"}}


def _build(cls: type, data: dict[str, Any] | None, path: str) -> Any:
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    known = {f.name: f for f in fields(cls)}
    for old_name, new_name in RENAMED.get(cls.__name__, {}).items():  # settings renamed since: still accepted
        if old_name in data and new_name not in data:
            data = {**{k: v for k, v in data.items() if k != old_name}, new_name: data[old_name]}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"unknown keys in {path or 'config'}: {', '.join(sorted(unknown))}")
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{path}.{name}".strip("."))
        elif cls is Config and name == "projects":
            if not isinstance(value or {}, dict):
                raise ConfigError("projects must be a mapping of name -> {dir, context, ...}")
            kwargs[name] = {str(k): _build(ProjectConfig, v, f"projects.{k}") for k, v in (value or {}).items()}
        else:
            kwargs[name] = value
    return cls(**kwargs)


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env_file(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines (optional `export`, quotes and # comments), like a minimal dotenv."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            raise ConfigError(f"bad line in env file: {line.strip()[:60]!r}")
        key, value = m.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0]
        out[key] = value
    return out


def load_env_file(path: Path) -> list[str]:
    """Load ``path`` into os.environ (without overriding variables that are already set).
    Returns the names that were loaded."""
    if not path.is_file():
        return []
    if path.stat().st_mode & 0o077:
        log.warning("%s is readable by other users; run: chmod 600 %s", path, path)
    loaded = []
    for key, value in parse_env_file(path.read_text()).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load_config(path: str | os.PathLike) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(Config, raw, "")
    cfg.local.base_dir = str(Path(path).resolve().parent)
    cfg.validate()
    load_env_file(cfg.local.env_path)
    return cfg
