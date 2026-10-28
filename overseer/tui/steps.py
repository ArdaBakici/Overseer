"""The setup wizard's work, apart from its screens (setup.py): what's installed, the secrets file, the config file,
Telegram, the cluster. Plain functions, so they can be tested without a terminal."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..core import configtool
from ..core.config import ConfigError, accounts_of, read_yaml

TELEGRAM_API = "https://api.telegram.org"
CLI_PACKAGES = {"claude_code": ("claude", "@anthropic-ai/claude-code"), "codex": ("codex", "@openai/codex")}
SYSTEM_TOOLS = {"ssh": "the SSH client", "bwrap": "bubblewrap (the agents' sandbox)",
                "socat": "socat (Claude Code's sandbox)", "npm": "Node.js and npm (to install the agent CLIs)"}
# package names per package manager: ssh, bwrap, socat, npm
PACKAGES = {
    "apt": {"ssh": "openssh-client", "bwrap": "bubblewrap", "socat": "socat", "npm": "npm"},
    "dnf": {"ssh": "openssh-clients", "bwrap": "bubblewrap", "socat": "socat", "npm": "nodejs-npm"},
    "pacman": {"ssh": "openssh", "bwrap": "bubblewrap", "socat": "socat", "npm": "npm"},
    "zypper": {"ssh": "openssh-clients", "bwrap": "bubblewrap", "socat": "socat", "npm": "npm"},
}
INSTALL = {"apt": ["sudo", "apt-get", "install", "-y"], "dnf": ["sudo", "dnf", "install", "-y"],
           "pacman": ["sudo", "pacman", "-S", "--needed", "--noconfirm"], "zypper": ["sudo", "zypper", "install", "-y"]}


# ============================================================================ what's installed
@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    optional: bool = False


def package_manager() -> str | None:
    for pm in ("apt", "dnf", "pacman", "zypper"):
        if shutil.which("apt-get" if pm == "apt" else pm):
            return pm
    return None


def userns_restricted() -> bool:
    """Ubuntu 24.04+: AppArmor stops unprivileged user namespaces, which bubblewrap needs."""
    try:
        return Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns").read_text().strip() == "1"
    except OSError:
        return False


def prerequisites() -> list[Check]:
    out = [Check(tool, bool(shutil.which(tool)), what) for tool, what in SYSTEM_TOOLS.items()]
    for backend, (cmd, pkg) in CLI_PACKAGES.items():
        path = shutil.which(cmd)
        out.append(Check(cmd, bool(path), f"{'Claude Code' if cmd == 'claude' else 'Codex'} ({pkg})" +
                         (f": {path}" if path else ""), optional=True))
    if platform.system() != "Linux":
        out.append(Check("linux", False, "Overseer runs on Linux (the agents' sandbox needs it)"))
    if userns_restricted():
        out.append(Check("userns", False, "AppArmor blocks the sandbox (Ubuntu 24.04+): see docs/setup.md, "
                                          "Troubleshooting"))
    return out


def install_commands(checks: list[Check], clis: list[str]) -> list[list[str]]:
    """What to run for the missing tools (the system packages, then the agent CLIs chosen)."""
    missing = {c.name for c in checks if not c.ok}
    cmds = []
    pm = package_manager()
    pkgs = [PACKAGES[pm][t] for t in SYSTEM_TOOLS if t in missing] if pm else []
    if pkgs:
        cmds.append([*INSTALL[pm], *pkgs])
    npm_pkgs = [pkg for b, (cmd, pkg) in CLI_PACKAGES.items() if b in clis and cmd in missing]
    if npm_pkgs:
        cmds.append(["npm", "install", "-g", *npm_pkgs])
    return cmds


# ============================================================================ the secrets file (.env)
def read_env(path: Path) -> dict[str, str]:
    out = {}
    try:
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                out[k.strip().removeprefix("export ").strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


def set_env(path: Path, key: str, value: str) -> None:
    """Set KEY=value in the secrets file (kept private: 600), leaving its other lines as they are."""
    lines = path.read_text().splitlines() if path.exists() else []
    done = False
    for i, line in enumerate(lines):
        k = line.strip().removeprefix("export ").partition("=")[0].strip()
        if k == key and not line.lstrip().startswith("#"):
            lines[i], done = f"{key}={value}", True
    if not done:
        lines.append(f"{key}={value}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(path, 0o600)


# ============================================================================ the config file
class Draft:
    """The config file being set up: its values (only those set), written as the example with its comments."""

    def __init__(self, path: Path, domain: str):
        self.path = path
        self.raw: dict = read_yaml(path) if path.exists() else {}
        self.domain = str(self.raw.get("domain") or domain)
        self.raw["domain"] = self.domain
        self.backed_up = False

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in dotted.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def set(self, dotted: str, value: Any) -> None:
        keys = dotted.split(".")
        node = self.raw
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    def text(self) -> str:
        guide = configtool.defaults(self.domain)
        mine = {p: v for p, v in configtool.leaves(self.raw, guide).items() if configtool._known(p, guide)}
        return configtool.merge(configtool.example_path(self.domain).read_text(), mine,
                                configtool.leaves(guide, guide))

    def save(self) -> Path | None:
        """Write the file (the first time this session, the old one is kept as <file>.bak-<date>). The backup's path."""
        text = self.text()
        backup = None
        if self.path.exists() and not self.backed_up:
            if self.path.read_text() == text:
                return None
            backup = self.path.with_name(f"{self.path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
            shutil.copy2(self.path, backup)
            self.backed_up = True
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(text)
        tmp.replace(self.path)
        self.backed_up = True  # (a file this session wrote needs no backup)
        return backup

    def validate(self) -> str:
        """"" if it loads, else why not."""
        from ..core.config import Config, _build
        try:
            cfg = _build(Config, yaml.safe_load(self.text()), "")
            cfg.local.base_dir = str(self.path.parent.resolve())
            cfg.validate()
        except ConfigError as e:
            return str(e)
        return ""


def detect_timezone() -> str:
    try:
        link = os.readlink("/etc/localtime")
        if "zoneinfo/" in link:
            return link.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    try:
        return Path("/etc/timezone").read_text().strip()
    except OSError:
        return ""


# ============================================================================ Telegram
class TelegramError(RuntimeError):
    pass


def telegram(token: str, method: str, http_timeout: float = 15, **params) -> Any:
    import httpx

    try:
        r = httpx.get(f"{TELEGRAM_API}/bot{token}/{method}", params=params, timeout=http_timeout)
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise TelegramError(f"couldn't reach Telegram: {type(e).__name__}") from None
    if not data.get("ok"):
        desc = data.get("description") or "error"
        if r.status_code == 401 or "Unauthorized" in desc:
            raise TelegramError("Telegram doesn't know this token: copy it again from @BotFather")
        if r.status_code == 409:
            raise TelegramError("the bot is running (it reads its own messages): stop it first, or type your id")
        raise TelegramError(desc)
    return data["result"]


def bot_info(token: str) -> dict:
    """{"username", "name", "topics"} - topics: threaded mode is on."""
    me = telegram(token, "getMe")
    return {"username": me.get("username", ""), "name": me.get("first_name", ""),
            "topics": bool(me.get("has_topics_enabled"))}


def wait_for_chat(token: str, wait: float = 60, since: float | None = None) -> tuple[int, str] | None:
    """The private chat that writes to the bot next (within `wait` s): (chat id, the person's name). Messages
    aren't consumed: the bot still gets them when it starts."""
    since = time.time() - 120 if since is None else since
    end = time.time() + wait
    while time.time() < end:
        updates = telegram(token, "getUpdates", http_timeout=40, allowed_updates='["message"]',
                           timeout=min(25, max(1, int(end - time.time()))))
        for u in reversed(updates):
            m = u.get("message") or {}
            chat = m.get("chat") or {}
            if chat.get("type") == "private" and m.get("date", 0) >= since:
                name = " ".join(x for x in (chat.get("first_name"), chat.get("last_name")) if x)
                return int(chat["id"]), name or chat.get("username", "")
    return None


# ============================================================================ the cluster
@dataclass
class Partition:
    name: str
    default: bool = False
    up: bool = True
    limit: str = ""
    gres: list[str] = field(default_factory=list)
    nodes: int = 0

    @property
    def gpus(self) -> bool:
        return any(g.startswith("gpu") for g in self.gres)


def parse_sinfo(text: str) -> list[Partition]:
    """`sinfo -h -o '%P|%a|%l|%G|%D'` (one line per partition and node kind) -> partitions."""
    parts: dict[str, Partition] = {}
    for line in text.splitlines():
        cols = line.strip().split("|")
        if len(cols) < 5 or not cols[0]:
            continue
        name = cols[0].rstrip("*")
        p = parts.setdefault(name, Partition(name, default=cols[0].endswith("*"), up=cols[1] == "up", limit=cols[2]))
        for g in cols[3].split(","):
            g = re.sub(r"\(.*?\)", "", g).strip()
            if g and g != "(null)" and g not in p.gres:
                p.gres.append(g)
        p.nodes += int(cols[4]) if cols[4].isdigit() else 0
    return list(parts.values())


def gpu_gres(p: Partition) -> str:
    """"gpu:1", or "gpu:a100:1" when the partition names one GPU type."""
    types = {g.split(":")[1] for g in p.gres if g.startswith("gpu:") and len(g.split(":")) >= 3}
    return f"gpu:{types.pop()}:1" if len(types) == 1 else "gpu:1"


def cluster_info(config: Path) -> tuple[list[Partition], str, str]:
    """Over Overseer's own connection: (partitions, the quota command found or "", an error)."""
    out = overseer(config, "ssh", "test")
    if out.returncode != 0:
        return [], "", out.stdout.strip() or out.stderr.strip() or "the SSH test failed"
    from .data import peek_config
    cfg = peek_config(config)
    agent_ssh = cfg.local.tmp("agent-bin", "ssh")
    probe = ("sinfo -h -o '%P|%a|%l|%G|%D' 2>/dev/null; echo '@@'; "
             "for q in myquota quota_check checkquota lfs; do command -v $q >/dev/null 2>&1 && echo $q && break; done")
    res = subprocess.run([str(agent_ssh), cfg.ssh.host, probe], capture_output=True, text=True, timeout=90)
    sinfo, _, quota = res.stdout.partition("@@")
    q = quota.strip().splitlines()[0] if quota.strip() else ""
    return parse_sinfo(sinfo), ("" if q == "lfs" else q), ("" if res.returncode == 0 else res.stderr.strip()[-500:])


# ============================================================================ the agents
def logged_in(backend: str, home: Path, env: dict[str, str]) -> bool:
    """A login (or a key in .env) for this CLI account - by the files the CLIs keep, not a model call."""
    if backend == "claude_code":
        if env.get("ANTHROPIC_API_KEY") or env.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return True
        if (home / ".credentials.json").exists():
            return True
        try:
            return bool(json.loads((home / ".claude.json").read_text()).get("oauthAccount"))
        except (OSError, ValueError):
            return False
    return (home / "auth.json").exists() or bool(env.get("OPENAI_API_KEY") or env.get("CODEX_API_KEY"))


def accounts(draft: Draft, backend: str, base: Path) -> dict[str, Path]:
    from ..core.config import ClaudeCodeConfig, CodexConfig
    c = (ClaudeCodeConfig if backend == "claude_code" else CodexConfig)()
    c.accounts = dict(draft.get(f"agent.{backend}.accounts") or {})
    c.home = draft.get(f"agent.{backend}.home") or (
        f"./state/assistant/{'claude' if backend == 'claude_code' else 'codex'}" if draft.domain == "assistant"
        else c.home)
    return {n: (base / h).resolve() if not Path(h).is_absolute() else Path(h) for n, h in accounts_of(c).items()}


# ============================================================================ running Overseer's own commands
def overseer_argv(config: Path, *args: str) -> list[str]:
    return [sys.executable, "-m", "overseer", *args, "-c", str(config)]


def overseer(config: Path, *args: str, timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(overseer_argv(config, *args), capture_output=True, text=True, timeout=timeout,
                          cwd=str(config.parent), stdin=subprocess.DEVNULL)
