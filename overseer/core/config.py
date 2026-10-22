"""Configuration loading.

Secrets (Telegram token, API keys) come from environment variables, never from the YAML
file, so the config can be committed or shared safely.
"""

from __future__ import annotations

import hashlib
import importlib
import json
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
    # The time zone this bot thinks in - its times, reminders, routines, digest - and the agent's `date`, if not this
    # machine's: e.g. Asia/Dubai. "" = this machine's.
    timezone: str = ""
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

    # ------------------------------------------------------------------ the bots on this machine
    @property
    def bots_dir(self) -> Path:
        """Where each of Overseer's bots notes its folders (in ./state, which no agent can read)."""
        return Path(self.base_dir or os.getcwd()).resolve() / "state" / "bots"

    def other_bots(self) -> list[dict]:
        """The other bots that ran from this folder: {"domain", "state", "tmp"}."""
        own = str(self._resolve(self.tmp_dir))
        out = []
        for f in sorted(self.bots_dir.glob("*.json")) if self.bots_dir.is_dir() else []:
            try:
                info = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(info, dict) and info.get("tmp") != own:
                out.append(info)
        return out

    def register(self, domain: str) -> None:
        """Note this bot's folders, so the other bots' agents are kept out of them. Its folders must not lie inside
        another bot's (or hold one): an agent can't be kept out of a folder its own is in."""
        state, tmp = self._resolve(self.state_dir), self._resolve(self.tmp_dir)
        shared = self.bots_dir.parent  # ./state: the HPC bot's, and the other bots' folders below it are fine
        mine = [tmp] + ([state] if state != shared else [])
        for other in self.other_bots():
            for key in ("tmp", "state"):
                p = Path(other.get(key) or "/nonexistent")
                if p != shared and any(_path_within(m, p) or _path_within(p, m) for m in mine):
                    raise ConfigError(f"local.state_dir/tmp_dir overlap the {other.get('domain')} bot's {key} folder "
                                      f"{p}: give each bot folders of its own")
        self.bots_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        name = hashlib.sha1(str(tmp).encode()).hexdigest()[:12]
        (self.bots_dir / f"{name}.json").write_text(json.dumps({"domain": domain, "state": str(state),
                                                                 "tmp": str(tmp)}))

    def secrets(self) -> list[Path]:
        """What no agent of any of Overseer's bots may read: every bot's state (./state and the bots' folders below
        it), the other bots' tmp folders (their sockets: e.g. the HPC's SSH master), and every .env file in Overseer's
        folder (the bots may share one)."""
        base = Path(self.base_dir or os.getcwd()).resolve()
        others = [Path(p) for b in self.other_bots() for p in (b.get("state"), b.get("tmp"))
                  if p and Path(p).is_absolute() and Path(p) != Path("/")]
        return [base / "state", self._resolve(self.state_dir), self.env_path, *sorted(base.glob(".env*")), *others]

    @property
    def db_path(self) -> Path:
        return self.state("overseer.db")

    @property
    def audit_log(self) -> Path:
        return self.state("audit.log")


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
    # Topics in the chat with the bot (Telegram's threaded mode for bots, switched on in @BotFather): "general",
    # a topic per project, ... - Overseer makes them; null = whatever the bot's mode in @BotFather is, true = topics
    # (warns if the bot's topic mode is off), false = one conversation in the plain chat, even if it's on.
    topics: bool | None = None
    # With topics: a message written outside them (Telegram's "All" view) starts a new conversation of its own.
    # Off: it's refused, with a note to write in a project's topic or in "general".
    new_threads: bool = False

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
    # The model's own web search (runs on OpenAI's side): live, cached (OpenAI's index only) or disabled.
    web_search: str = "live"
    extra_config: list[str] = field(default_factory=list)  # extra `-c key=value` overrides


@dataclass
class AutopilotConfig:
    """Wake-ups: the agent watches jobs (`ov watch`) and sets timers (`ov wake`) on its own; no approval."""

    min_checkin_minutes: int = 15  # check-ins on running jobs and repeating timers: at least this far apart
    repeat_hours: float = 24  # a repeating timer stops after this, unless the agent gives `--for`


@dataclass
class SubagentsConfig:
    """Planner mode: a conversation plans and delegates to subagents (cheaper models), each in its own thread.
    Off unless you turn it on (/subagents on) or approve an agent's proposal (`ov propose`)."""
    max_running: int = 3  # subagents running at once; more are queued
    # The planner's model when the mode is on, per backend (the conversation keeps its backend).
    planner: dict[str, str] = field(default_factory=lambda: {"claude_code": "claude-opus-5-5", "codex": "gpt-6-astra"})
    # The models subagents may use, per backend; the first is the default.
    workers: dict[str, list[str]] = field(default_factory=lambda: {
        "claude_code": ["claude-sonnet-5-5"], "codex": ["gpt-6.1-sol", "gpt-6-luna"]})
    worker_effort: str | None = None  # None = the model's default


@dataclass
class AgentConfig:
    enabled: bool = True
    # claude_code or codex. Can be switched at runtime from Telegram with /backend.
    backend: str = "claude_code"
    timeout_minutes: int = 60  # max length of one agent run
    # Update the agent CLIs (`claude update`, `codex update`) once a day, while no agent run is in progress.
    # New models often need a recent CLI. /update in Telegram (or `overseer update`) does it now.
    auto_update: bool = True
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
    llm_status_updates: bool = False  # one-line LLM progress summary in periodic status messages
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    autopilot: AutopilotConfig = field(default_factory=AutopilotConfig)
    subagents: SubagentsConfig = field(default_factory=SubagentsConfig)


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
class DigestConfig:
    """The morning digest (digest.py): once a day in the general topic, what happened since the last one."""
    enabled: bool = True
    time: str = "08:00"  # local time of this machine (HH:MM)


@dataclass
class DashboardConfig:
    """The dashboard Mini App (dashboard.py): served on this machine; make it reachable on your tailnet with
    `tailscale serve --bg <port>` and put the https://<machine>.<tailnet>.ts.net address it prints in `url`."""
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    url: str = ""  # what Telegram opens (https, e.g. https://homelab.tail1234.ts.net); "" = not reachable yet


def domain_module(name: str):
    """A domain's configuration module (overseer/<domain>/config.py): its SECTIONS of config.yaml, the DERIVED values
    reached as cfg.<name>, and validate(cfg). Loaded by name, so the platform never imports a domain itself."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
        raise ConfigError(f"domain {name!r}: a lowercase name such as hpc")
    try:
        return importlib.import_module(f"overseer.{name}.config")
    except ModuleNotFoundError as e:
        raise ConfigError(f"unknown domain {name!r} (there is no overseer/{name}/config.py)") from e


@dataclass
class Config:
    local: LocalConfig = field(default_factory=LocalConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    digest: DigestConfig = field(default_factory=DigestConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    projects: dict[str, ProjectConfig] = field(default_factory=dict)
    domain: str = "hpc"  # which bot this is: hpc (Overseer's SLURM jobs), ...
    sections: dict[str, Any] = field(default_factory=dict)  # the domain's own config (hpc: ssh, slurm)

    def __post_init__(self) -> None:
        for name, cls in domain_module(self.domain).SECTIONS.items():
            self.sections.setdefault(name, cls())

    def __getattr__(self, name: str) -> Any:
        """cfg.ssh, cfg.slurm, cfg.ssh_dir, ...: the domain's sections and the values derived from them."""
        if name.startswith("__") or "sections" not in self.__dict__:
            raise AttributeError(name)
        if name in self.sections:
            return self.sections[name]
        derived = getattr(domain_module(self.domain), "DERIVED", {})
        if name in derived:
            return derived[name](self)
        raise AttributeError(f"config has no {name!r} (domain {self.domain})")

    @property
    def ctl_socket(self) -> Path:
        """Overseer's control socket for the agent's `ov` command."""
        return self.local.tmp("ctl.sock")

    def validate(self) -> None:
        for name, proj in self.projects.items():
            if not PROJECT_NAME_RE.match(name) or name in RESERVED_PROJECT_NAMES:
                raise ConfigError(f"project name {name!r}: use lowercase letters, digits, - and _ (not main/all)")
            if proj.backend and proj.backend not in BACKENDS:
                raise ConfigError(f"projects.{name}.backend must be one of {', '.join(BACKENDS)}")
        if not self.local.allowed_roots:
            raise ConfigError("local.allowed_roots must list at least one folder")
        for r in self.local.roots:
            if r == Path("/") or r == Path.home():
                raise ConfigError(f"local.allowed_roots entry {r} is too broad; use specific folders")
        if self.local.timezone:
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
            try:
                ZoneInfo(self.local.timezone)
            except (ZoneInfoNotFoundError, ValueError):
                raise ConfigError(f"local.timezone {self.local.timezone!r}: a time zone name like Asia/Dubai") from None
        self.local.state("x")  # both dirs must sit inside an allowed root
        self.local.tmp("x")
        if len(str(self.ctl_socket)) > UNIX_SOCKET_MAX:
            raise ConfigError(f"local.tmp_dir is too long for a unix socket path ({self.ctl_socket}); use e.g. /tmp/overseer")
        if self.agent.backend not in BACKENDS:
            raise ConfigError(f"agent.backend must be one of {', '.join(BACKENDS)}")
        if self.agent.report_commands not in ("summary", "live", "off"):
            raise ConfigError("agent.report_commands must be summary, live or off")
        self.local.resolve(self.agent.claude_code.home)
        self.local.resolve(self.agent.codex.home)
        ws = self.agent.codex.web_search  # true/false in older config files
        self.agent.codex.web_search = {True: "live", False: "disabled"}.get(ws, ws) if isinstance(ws, bool) else ws
        if self.agent.codex.web_search not in ("live", "cached", "disabled"):
            raise ConfigError("agent.codex.web_search must be live, cached or disabled")
        if self.dashboard.url and not self.dashboard.url.startswith("https://"):
            raise ConfigError("dashboard.url must be https:// (Telegram only opens Mini Apps over HTTPS): the address "
                              "`tailscale serve` prints")
        if self.digest.time and not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", self.digest.time.strip()):
            raise ConfigError("digest.time must be HH:MM, e.g. 08:00")
        if not self.telegram.allowed_chat_ids:
            raise ConfigError("telegram.allowed_chat_ids must list at least one chat id")
        domain_module(self.domain).validate(self)


RENAMED: dict[str, dict[str, str]] = {}
# Settings that no longer exist: ignored (with a warning) rather than refused, so old config files keep working.
DROPPED = {"AgentConfig": {"job_reports"},  # finished jobs: the card is the result, no LLM report
           "AutopilotConfig": {"default_rounds", "max_rounds", "default_wakeups", "max_wakeups", "default_hours",
                               "max_hours", "checkin_minutes", "max_checkins_per_job", "oneoff_max_pending",
                               "oneoff_max_hours", "oneoff_chain"}}


def _build(cls: type, data: dict[str, Any] | None, path: str) -> Any:
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    if cls is Config:  # the domain's sections (e.g. ssh, slurm) are built by its own classes
        if "sections" in data:
            raise ConfigError("unknown keys in config: sections")
        domain = str(data.get("domain") or "hpc")
        section_types = domain_module(domain).SECTIONS
        sections = {k: _build(t, data.get(k), k) for k, t in section_types.items()}
        data = {k: v for k, v in data.items() if k not in section_types}
        cfg = _build_fields(cls, data, path)
        cfg.sections.update(sections)
        return cfg
    return _build_fields(cls, data, path)


def _build_fields(cls: type, data: dict[str, Any], path: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    for old_name, new_name in RENAMED.get(cls.__name__, {}).items():  # settings renamed since: still accepted
        if old_name in data and new_name not in data:
            data = {**{k: v for k, v in data.items() if k != old_name}, new_name: data[old_name]}
    old = set(data) & DROPPED.get(cls.__name__, set())
    if old:
        logging.getLogger(__name__).warning("%s: no longer used, ignored: %s", path or "config", ", ".join(sorted(old)))
        data = {k: v for k, v in data.items() if k not in old}
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


def apply_timezone(name: str) -> None:
    """This process (and the agents it starts: TZ is passed on) thinks in this time zone."""
    if not name:
        return
    os.environ["TZ"] = name
    import time
    time.tzset()


def load_config(path: str | os.PathLike) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(Config, raw, "")
    cfg.local.base_dir = str(Path(path).resolve().parent)
    cfg.validate()
    load_env_file(cfg.local.env_path)
    apply_timezone(cfg.local.timezone)
    return cfg
