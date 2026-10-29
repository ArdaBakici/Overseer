"""Run the agent with Claude Code (`claude -p`) or OpenAI Codex (`codex exec`).

The agent gets a real shell, and reaches the HPC with `ssh hpc ...` through the master
connection Overseer keeps open (see sshlink.py). On this machine each CLI runs inside its
own OS-level sandbox, configured by Overseer:

* writes: only the session's scratch directory (inside local.tmp_dir)
* reads: everything except Overseer's SSH key folder, ~/.ssh, Overseer's state dir, its .env
  file (Telegram token, API keys) and well-known daemon sockets (docker, podman, session bus)
* network: none from the shell (except agent.allowed_domains); the HPC is reached through
  the unix socket of Overseer's SSH master connection
* environment: scrubbed - only PATH/locale and the API-key variables listed in pass_env;
  HOME / CLAUDE_CONFIG_DIR / CODEX_HOME point inside local.state_dir

Claude Code: sandbox with failIfUnavailable + allowUnsandboxedCommands=false (no escape hatch).
Codex: a named permission profile (filesystem read/write/deny + network) and approvals off.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .accounts import LIMIT_ANSWER_RE, usage_limit
from .config import Config, accounts_of

log = logging.getLogger(__name__)

MAX_PROMPT = 200_000
BASE_ENV = ["LANG", "LC_ALL", "TZ"]
# Claude Code's answer when a tool call needed an interactive approval that -p mode can't give.
PERMISSION_DENIED_RE = re.compile(r"requires approval|permission to use .* (?:was|has been) denied|haven't granted", re.I)

# The agent CLIs running now (process group ids): ended with Overseer, so a run that a restart cut off can't go on
# by itself while the researcher is offered to continue it.
_RUNNING: set[int] = set()


def _load_prctl():
    try:
        import ctypes
        return ctypes.CDLL(None, use_errno=True).prctl
    except Exception:  # not Linux
        return None


_PRCTL = _load_prctl()  # looked up here: the child, between fork and exec, must not import or load anything


def _die_with_overseer() -> None:
    """In the CLI's process, before it starts: killed if Overseer dies, even by SIGKILL (Linux)."""
    if _PRCTL is not None:
        _PRCTL(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def kill_running() -> int:
    """Stop every agent CLI that is running (Overseer is shutting down). How many there were."""
    n = 0
    for pgid in list(_RUNNING):
        try:
            os.killpg(pgid, signal.SIGKILL)
            n += 1
        except ProcessLookupError:
            pass
        _RUNNING.discard(pgid)
    return n


# Claude Code tools that can't work in a run that ends with the turn: wake-ups and schedules (Overseer's `ov wake` /
# `ov watch` do that), background workflows, messages to other sessions, and tools for its own UIs. (Cron tools
# and background subagents/commands are turned off with environment variables, see ClaudeCodeRunner.env.)
CLAUDE_DISALLOWED_TOOLS = ["ScheduleWakeup", "CronCreate", "CronDelete", "CronList", "Workflow", "SendMessage",
                           "ListAgents", "TaskStop", "Monitor", "PushNotification", "RemoteTrigger", "EnterWorktree",
                           "ExitWorktree", "ReportFindings"]

# Codex features with tools that act outside the shell sandbox or aren't useful headless.
CODEX_DISABLED_FEATURES = [
    "browser_use", "browser_use_external", "browser_use_full_cdp_access", "in_app_browser", "computer_use",
    "image_generation", "apps", "plugins", "remote_plugin", "in_app_local_automation", "tool_suggest",
    "skill_mcp_dependency_install", "workspace_dependencies", "memories", "goals", "hooks",
    # Subagents: always asynchronous (spawn, then wait), so a turn that ends first loses them.
    "multi_agent",
]


class AgentRunError(RuntimeError):
    pass


class UsageLimitError(AgentRunError):
    """The account ran out of usage (its plan's limit, or the API key's credit): `until` when it resets, if said."""

    def __init__(self, message: str, until: float | None = None):
        super().__init__(message)
        self.until = until


def _check_limit(message: str) -> None:
    hit, until = usage_limit(message)
    if hit:
        raise UsageLimitError(message, until)


@dataclass
class RunResult:
    text: str
    session_id: str | None
    is_error: bool = False
    model: str | None = None  # the model that actually ran, when the CLI reports it
    effort: str | None = None  # the reasoning effort that actually applied, when known
    input_tokens: int = 0  # all input, including the cached part below
    cached_tokens: int = 0  # input served from the provider's prompt cache (much cheaper)
    output_tokens: int = 0
    cost_usd: float | None = None  # Claude Code reports it; Codex doesn't
    usage_is_total: bool = False  # Codex: the token counts are the whole conversation's running total
    # The CLI compacted the conversation (summarised older turns to free context): one dict each,
    # {"trigger": "auto" | "manual", "pre_tokens": int | None, "post_tokens": int | None}
    compactions: list[dict[str, Any]] = field(default_factory=list)


class Steering:
    """Messages for a run in progress - the researcher (or a planner) wrote while the agent worked. The runner hands
    them to the running turn, and marks the ones the agent took (it reads them at its next step). send() is False once
    the run can't take any more; untaken() are the ones it ended without: they're the next request."""

    def __init__(self) -> None:
        self.closed = False
        self.queued: list[str] = []  # not handed to the CLI yet (it wasn't ready)
        self.given: list[str] = []  # handed to the CLI, not taken yet
        self.taken: list[str] = []
        self.instant = False  # the CLI takes a message as it's handed over (codex's turn/steer): no separate ack
        self._deliver: Callable[[str], Any] | None = None

    async def send(self, text: str) -> bool:
        if self.closed:
            return False
        if self._deliver is None:
            self.queued.append(text)  # (handed over as soon as the CLI is ready)
            return True
        if not await self._deliver(text):
            return False
        (self.taken if self.instant else self.given).append(text)
        return True

    async def ready(self, deliver: Callable[[str], Any]) -> None:
        """The runner can take messages now: deliver(text) -> handed over."""
        self._deliver = deliver
        while self.queued and not self.closed:
            text = self.queued[0]
            if not await deliver(text):
                break
            self.queued.pop(0)
            (self.taken if self.instant else self.given).append(text)

    def took(self, text: str) -> bool:
        if text in self.given:
            self.given.remove(text)
            self.taken.append(text)
            return True
        return False

    def untaken(self) -> list[str]:
        return self.queued + self.given

    def close(self) -> None:
        self.closed = True


@dataclass
class Sandbox:
    """What a domain adds to the agent's sandbox (on top of the platform's own rules, below), and the skills its
    agent gets."""
    deny_read: list[Path] = field(default_factory=list)  # more local paths its commands must never read
    # Paths only its shell commands are kept out of (the CLIs' sandboxes can deny a folder yet allow a writable one
    # inside it; Claude Code's own file tools can't - their deny rules always win - so for those, list the folder's
    # other entries in deny_read). E.g. the vault around the HPC bot's folder.
    deny_read_shell: list[Path] = field(default_factory=list)
    unix_sockets: list[Path] = field(default_factory=list)  # local sockets its commands may connect to
    writable: list[Path] = field(default_factory=list)  # local folders it may write besides its scratch folder
    deny_write: list[Path] = field(default_factory=list)  # folders inside those it must not write (e.g. .obsidian)
    skills: dict[str, str] = field(default_factory=dict)  # skills that come with the domain: name -> SKILL.md


def agent_bin_dir(cfg: Config) -> Path:
    """The agent's own commands (its control command, and whatever its domain adds): first on its PATH."""
    return cfg.local.tmp("agent-bin")


@dataclass
class Activity:
    """What the agent did during one request (for Telegram reports and the live view). A domain's subclass
    watches the commands too (shell_command, shell_output), e.g. for the jobs they submit.

    Every step is also passed to `on_progress` as a dict with a "kind": thought (reasoning summary),
    message (text the agent wrote between steps), command, output, tool, file, session, compacting or
    compact (the conversation was compacted)."""

    commands: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    blocked: list[tuple[str, str]] = field(default_factory=list)  # (command, reason) refused by the CLI itself
    _started: dict[str, float] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)  # extra environment for this run (e.g. the ov token)
    on_command: Callable[[str], None] | None = None
    on_progress: Callable[[dict[str, Any]], None] | None = None

    # ---- what a domain's subclass watches
    def shell_command(self, cmd: str) -> None:
        """A command the agent started (as displayed)."""

    def shell_output(self, cmd: str, output: str, took: float | None) -> None:
        """Its output, and how long it ran (None if its start wasn't seen)."""

    def report_lines(self) -> list[str]:
        """Extra lines (HTML) for the summary of the commands it ran."""
        return []

    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_progress:
            try:
                self.on_progress({"kind": kind, **data})
            except Exception:
                log.exception("progress handler failed")

    def thought(self, text: str) -> None:
        if text and text.strip():
            self._emit("thought", text=text.strip())

    def message(self, text: str) -> None:
        if text and text.strip():
            self._emit("message", text=text.strip())

    def tool(self, name: str, detail: str = "") -> None:
        self._emit("tool", name=name, detail=detail)

    def session(self, session_id: str) -> None:
        self._emit("session", id=session_id)

    def steered(self, text: str) -> None:
        """A message sent while it worked reached the agent (it reads it at its next step)."""
        self._emit("steer", text=text)

    def compacting(self) -> None:
        self._emit("compacting")

    def compacted(self, info: dict[str, Any]) -> None:
        """The CLI summarised the conversation's older turns to free context."""
        self._emit("compact", **info)

    def command(self, cmd: str) -> None:
        cmd = display_command(cmd)
        self.commands.append(cmd)
        self._started[cmd] = time.monotonic()
        self.shell_command(cmd)
        if self.on_command:
            self.on_command(cmd)
        self._emit("command", command=cmd)

    def output(self, cmd: str, output: str, exit_code: int | None = None) -> None:
        shown = display_command(cmd)
        self._emit("output", command=shown, output=output or "", exit_code=exit_code)
        started = self._started.pop(shown, None)
        self.shell_output(shown, output or "", time.monotonic() - started if started is not None else None)

    def file(self, path: str) -> None:
        if not path:
            return
        self._emit("file", path=path)
        if path not in self.files:
            self.files.append(path)

    def block(self, cmd: str, reason: str) -> None:
        self.blocked.append((display_command(cmd), reason.strip()[:300]))


def display_command(cmd: str) -> str:
    """Strip the `/bin/bash -lc '...'` wrapper codex reports around commands."""
    m = re.match(r"^(?:\S*/)?(?:ba|z)?sh\s+-l?c\s+(.*)$", cmd.strip(), re.S)
    if m:
        try:
            parts = shlex.split(m.group(1))
            if len(parts) == 1:
                return parts[0]
        except ValueError:
            pass
    return cmd.strip()


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _toml(value: Any) -> str:
    """Serialize a value as a TOML literal for `codex -c key=value`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {_toml(v)}" for k, v in value.items()) + "}"
    raise TypeError(type(value))


CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
CODEX_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


class Runner:
    name = ""
    efforts: tuple[str, ...] = ()

    def __init__(self, cfg: Config, sandbox: Sandbox | None = None):
        self.cfg = cfg
        self.sandbox = sandbox or Sandbox()  # what the domain adds to the sandbox
        self.timeout = cfg.agent.timeout_minutes * 60
        # Set from Telegram (/model, /effort); None = use config.yaml (where null = the CLI's default).
        self.model_override: str | None = None
        self.effort_override: str | None = None

    @property
    def model(self) -> str | None:
        return self.model_override or self.c.model

    @property
    def effort(self) -> str | None:
        return self.effort_override or self._configured_effort()

    def _configured_effort(self) -> str | None:
        raise NotImplementedError

    # ------------------------------------------------------------------ sandbox inputs
    def workdir(self, key: str) -> Path:
        d = self.cfg.local.tmp("agents", key)
        (d / ".tmp").mkdir(mode=0o700, parents=True, exist_ok=True)
        return d

    def deny_read(self, shell: bool = False) -> list[str]:
        """Local paths the agent's commands must never read (only existing ones are listed); `shell`: for its shell
        commands' sandbox (with the domain's deny_read_shell)."""
        home = Path.home()
        uid = os.getuid()
        candidates = [
            *self.sandbox.deny_read, *(self.sandbox.deny_read_shell if shell else []), home / ".ssh",
            *self.cfg.local.secrets(),
            home / ".gnupg", home / ".docker", home / ".aws", home / ".config" / "gcloud",
            Path("/var/run/docker.sock"), Path("/run/docker.sock"), Path("/run/containerd"),
            Path("/run/podman"), Path("/var/run/podman"), Path(f"/run/user/{uid}"),
        ]
        if os.environ.get("SSH_AUTH_SOCK"):
            candidates.append(Path(os.environ["SSH_AUTH_SOCK"]).parent)
        candidates += [Path(p).expanduser() for p in self.cfg.agent.deny_read]
        # Every login of the CLIs (accounts kept outside the state dir too)
        candidates += [self.cfg.local._resolve(h) for b in ("claude_code", "codex")
                       for h in accounts_of(getattr(self.cfg.agent, b)).values()]
        # Real paths only: the sandboxes can't mask a path through a symlinked dir (e.g. /var/run -> /run),
        # and both names of the same socket must collapse into one entry.
        paths = []
        for p in candidates:
            if p.exists() or p.is_symlink():
                paths.append(p.resolve())
        # Drop entries already covered by a denied parent (bubblewrap can't mount inside a masked dir).
        out = [p for p in paths if not any(q != p and q in p.parents for q in paths)]
        return list(dict.fromkeys(str(p) for p in out))

    def base_env(self, pass_env: list[str], home: Path, workdir: Path | None) -> dict[str, str]:
        env = {k: os.environ[k] for k in BASE_ENV + pass_env if k in os.environ}
        env["PATH"] = os.pathsep.join([str(agent_bin_dir(self.cfg)), os.environ.get("PATH", "/usr/bin:/bin")])
        # The agent's commands can't read the state dir, so during runs HOME is a scratch dir in
        # the session's (writable) workdir; for `overseer login` it is inside the backend's home.
        h = workdir / ".home" if workdir else home / "home"
        h.mkdir(mode=0o700, parents=True, exist_ok=True)
        env["HOME"] = str(h)
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"
        if workdir:  # the sandboxes only let commands write the scratch dir, so temp files go there too
            tmp = str(workdir / ".tmp")
            env["TMPDIR"] = env["TMP"] = env["TEMP"] = tmp
            env["TMPPREFIX"] = f"{tmp}/zsh"  # zsh's here-documents (it ignores TMPDIR; /tmp/zsh is read-only)
        return env

    # ------------------------------------------------------------------ process handling
    async def _exec(self, argv: list[str], env: dict[str, str], cwd: Path, stdin: str,
                    on_event: Callable[[dict[str, Any]], None]) -> str:
        """Run the CLI with `stdin` as its input, each JSON line of its output to on_event: its stderr if it failed."""
        async def talk(send, close, events):
            await send(stdin)
            close()
            async for event in events:
                _handle(on_event, event)
        return await self._converse(argv, env, cwd, talk)

    async def _converse(self, argv: list[str], env: dict[str, str], cwd: Path, talk) -> str:
        """Run the CLI and talk to it: talk(send(text), close(), events) writes to its stdin and reads its JSON output
        lines until it's done. Its stderr if it failed; the run's timeout, /stop and Overseer stopping kill it."""
        if not shutil.which(argv[0], path=env.get("PATH")):
            raise AgentRunError(f"{argv[0]!r} not found on PATH; install it or set agent.{self.name}.command")
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=32 * 1024 * 1024,
            start_new_session=True,  # own process group, so /stop and timeouts kill everything it started
            preexec_fn=_die_with_overseer,
        )
        _RUNNING.add(proc.pid)
        err_task = asyncio.create_task(proc.stderr.read())

        async def send(text: str) -> bool:
            if proc.stdin.is_closing():
                return False
            try:
                proc.stdin.write(text.encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return False
            return True

        def close() -> None:
            if not proc.stdin.is_closing():
                proc.stdin.close()

        async def events():
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("%s: %s", self.name, line[:500])
                    continue
                yield event

        async def pump():
            try:
                await talk(send, close, events())
            finally:
                close()
            await proc.wait()

        def kill():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        try:
            await asyncio.wait_for(pump(), timeout=self.timeout)
        except asyncio.TimeoutError:
            kill()
            await proc.wait()
            raise AgentRunError(f"{self.name} did not finish within {self.timeout // 60} minutes; stopped it")
        except BaseException:  # (cancelled: /stop, Overseer stopping; or the conversation went wrong)
            kill()
            await proc.wait()
            raise
        finally:
            _RUNNING.discard(proc.pid)
            stderr = (await err_task).decode(errors="replace")
        if proc.returncode is not None and proc.returncode < 0:  # killed (out of memory, Overseer stopping, ...)
            tail = "\n".join(stderr.strip().splitlines()[-5:])
            raise AgentRunError(f"{self.name} was killed (signal {-proc.returncode}) before it finished"
                                + (f": {tail}" if tail else ""))
        if proc.returncode != 0:
            return stderr
        return ""

    # ------------------------------------------------------------------ API
    async def run(self, prompt: str, *, system: str, session_id: str | None, key: str,
                  activity: Activity, steering: Steering | None = None) -> RunResult:
        """One request. With `steering`, messages sent while it works go into the running turn."""
        raise NotImplementedError

    async def text(self, *, system: str, prompt: str) -> str:
        """One-shot answer without tools (job reports)."""
        raise NotImplementedError

    async def compact(self, *, session_id: str, key: str, system: str, activity: Activity, focus: str = "") -> RunResult:
        """Compact the conversation now (summarise its older turns to free context)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ the CLI itself
    def installed(self) -> bool:
        return bool(shutil.which(self.c.command))

    async def version(self) -> str:
        """`<cli> --version`, e.g. "2.1.296 (Claude Code)" / "codex-cli 0.160.1"; "" if it can't be run."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.c.command, "--version", stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except (OSError, asyncio.TimeoutError):
            return ""
        lines = out.decode(errors="replace").strip().splitlines()
        return lines[0].strip() if lines else ""

    async def update(self) -> tuple[bool, str]:
        """The CLI's own updater (`claude update` / `codex update`). It runs with your normal environment, not the
        agent's: the installers find the installation through HOME, which Overseer's runs point elsewhere."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.c.command, "update", stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=600)
        except asyncio.TimeoutError:
            return False, "timed out after 10 minutes"
        except OSError as e:
            return False, str(e)
        return proc.returncode == 0, out.decode(errors="replace").strip()[-1500:]

    def login_command(self, extra: list[str]) -> tuple[list[str], dict[str, str]]:
        raise NotImplementedError

    def interactive_command(self, session_id: str, key: str) -> tuple[list[str], dict[str, str], Path]:
        """argv, env and cwd to open an agent session in the CLI's own interactive UI, with the same
        sandbox, ssh wrappers and isolated home as Overseer's runs (`overseer open`)."""
        raise NotImplementedError


# ============================================================================ Claude Code

class ClaudeCodeRunner(Runner):
    name = "claude_code"
    efforts = CLAUDE_EFFORTS

    def _configured_effort(self) -> str | None:
        return self.c.effort

    @property
    def c(self):
        return self.cfg.agent.claude_code

    def home(self) -> Path:
        h = self.cfg.local.resolve(self.c.home)
        h.mkdir(mode=0o700, parents=True, exist_ok=True)
        return h

    def env(self, workdir: Path | None) -> dict[str, str]:
        env = self.base_env(self.c.pass_env, self.home(), workdir)
        env["CLAUDE_CONFIG_DIR"] = str(self.home())
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["DISABLE_AUTOUPDATER"] = "1"
        # The CLI exits when the turn ends, so nothing it runs "in the background" survives to report back.
        # Subagents and commands run in the foreground; follow-ups go through Overseer (`ov wake`, `ov watch`).
        env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
        env["CLAUDE_CODE_DISABLE_CRON"] = "1"
        return env

    def settings(self, workdir: Path) -> dict[str, Any]:
        deny = self.deny_read()
        enforced = {
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,  # never run commands unsandboxed
                "allowUnsandboxedCommands": False,  # no dangerouslyDisableSandbox escape hatch
                "autoAllowBashIfSandboxed": True,
                "excludedCommands": [],
                "filesystem": {"allowWrite": [str(workdir), *map(str, self.sandbox.writable)],
                               "denyWrite": [str(p) for p in self.sandbox.deny_write],
                               "denyRead": self.deny_read(shell=True)},
                # Overseer's control socket, and the domain's (e.g. the SSH master that reaches the HPC). On Linux the
                # sandbox can't tell sockets apart by path (seccomp), so it's all or none: other bots' sockets are kept
                # out by their folders being unreadable (LocalConfig.secrets).
                "network": {"allowAllUnixSockets": True, "allowedDomains": list(self.cfg.agent.allowed_domains)},
            },
            "permissions": {
                "defaultMode": "acceptEdits",
                # Every Bash command runs inside the sandbox above (no unsandboxed escape), so allow them
                # all. Without this, Claude Code still asks for approval on some sandboxed commands (e.g.
                # `ssh hpc 'bash -lc "echo $HOME"'`), and in -p mode nobody can answer, so they're refused.
                "allow": ["Bash", "WebSearch", "WebFetch"],
                "deny": [rule for p in deny for rule in (f"Read(/{p}/**)", f"Read(/{p})", f"Edit(/{p}/**)")]
                + [f"Edit(/{p}/**)" for p in self.sandbox.deny_write],
                # The domain's writable folders (e.g. the vault) are the agent's to edit, like its scratch folder.
                "additionalDirectories": [str(p) for p in self.sandbox.writable],
            },
        }
        return _merge(self.c.extra_settings, enforced)

    def _common(self, workdir: Path) -> list[str]:
        argv = [self.c.command, "-p", "--settings", json.dumps(self.settings(workdir)), "--strict-mcp-config",
                "--disallowedTools", ",".join(CLAUDE_DISALLOWED_TOOLS)]
        if self.model:
            argv += ["--model", self.model]
        if self.effort:
            argv += ["--effort", self.effort]
        return argv

    async def run(self, prompt: str, *, system: str, session_id: str | None, key: str,
                  activity: Activity, steering: Steering | None = None) -> RunResult:
        workdir = self.workdir(key)
        argv = self._common(workdir) + ["--output-format", "stream-json", "--verbose", "--append-system-prompt", system]
        if session_id:
            argv += ["--resume", session_id]
        pending: dict[str, str] = {}
        result = RunResult(text="", session_id=session_id)
        texts: list[str] = []

        def on_event(ev: dict[str, Any]) -> None:
            t = ev.get("type")
            if ev.get("session_id") and ev["session_id"] != result.session_id:
                result.session_id = ev["session_id"]
                activity.session(result.session_id)
            if t == "system" and ev.get("subtype") == "init" and ev.get("model"):
                result.model = ev["model"]
            if t == "system" and ev.get("subtype") == "status" and ev.get("status") == "compacting":
                activity.compacting()
            if t == "system" and ev.get("subtype") == "compact_boundary":
                meta = ev.get("compact_metadata") or {}
                info = {"trigger": meta.get("trigger") or "auto", "pre_tokens": meta.get("pre_tokens"),
                        "post_tokens": meta.get("post_tokens")}
                result.compactions.append(info)
                activity.compacted(info)
            if t == "assistant":
                for block in (ev.get("message") or {}).get("content") or []:
                    kind = block.get("type")
                    if kind == "thinking":
                        activity.thought(block.get("thinking") or "")
                    elif kind == "text" and block.get("text"):
                        texts.append(block["text"])
                        activity.message(block["text"])
                    elif kind == "tool_use":
                        name, inp = block.get("name"), block.get("input") or {}
                        if name == "Bash" and inp.get("command"):
                            pending[block.get("id")] = inp["command"]
                            activity.command(inp["command"])
                        elif name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
                            activity.file(inp.get("file_path") or inp.get("notebook_path") or "")
                        elif name:
                            activity.tool(name, _tool_detail(inp))
            elif t == "user":
                for block in (ev.get("message") or {}).get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                        content = block.get("content")
                        if isinstance(content, list):
                            content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
                        cmd = pending.pop(block["tool_use_id"])
                        text = str(content or "")
                        if block.get("is_error") and PERMISSION_DENIED_RE.search(text):
                            activity.block(cmd, text)
                        activity.output(cmd, text)
            elif t == "result":
                result.is_error = bool(ev.get("is_error"))
                if ev.get("result"):
                    answers.append(str(ev["result"]))
                    result.text = "\n\n".join(dict.fromkeys(answers))  # (a steered message may have made more turns)
                # (its totals are the whole process's - every turn of this run so far)
                result.input_tokens = result.cached_tokens = result.output_tokens = 0
                _claude_usage(ev, result)

        answers: list[str] = []
        env = {**self.env(workdir), **activity.env}
        if steering is None:
            stderr = await self._exec(argv, env, workdir, prompt[:MAX_PROMPT], on_event)
        else:
            stderr = await self._converse(argv + ["--input-format", "stream-json", "--replay-user-messages"], env,
                                          workdir, self._steered_talk(prompt, steering, activity, on_event, result))
        if result.is_error:  # e.g. an API error: Claude Code reports it as the result, not as an answer
            message = "claude: " + (result.text or stderr.strip() or "the run failed")[-1500:]
            _check_limit(message)
            raise AgentRunError(message)
        if result.text and len(result.text) < 400 and LIMIT_ANSWER_RE.match(result.text):
            _check_limit("claude: " + result.text)  # (the limit, reported as if it were the answer)
        if not result.text:
            result.text = texts[-1] if texts else ""
        if stderr and not result.text:
            raise AgentRunError("claude: " + "\n".join(stderr.strip().splitlines()[-15:])[-1500:])
        return result

    def _steered_talk(self, prompt: str, steering: Steering, activity: Activity, on_event, result: RunResult):
        """Claude Code with streaming input: the request, then the messages sent while it works - it takes them at its
        next step, and says so (--replay-user-messages). Its input is closed once a turn has ended with every
        message taken; a message it hadn't taken by then gets a turn of its own first."""
        def line(text: str) -> str:
            return json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"

        async def talk(send, close, events):
            async def deliver(text: str) -> bool:
                return not steering.closed and await send(line(text))

            await send(line(prompt[:MAX_PROMPT]))
            await steering.ready(deliver)
            async for ev in events:
                if ev.get("type") == "user" and ev.get("isReplay"):
                    content = (ev.get("message") or {}).get("content")
                    text = content if isinstance(content, str) else "".join(
                        b.get("text", "") for b in content or [] if isinstance(b, dict))
                    if steering.took(text):
                        activity.steered(text)
                    continue
                _handle(on_event, ev)
                if ev.get("type") == "result" and (not steering.untaken() or result.is_error):
                    steering.close()  # (what comes from now on is the next request)
                    close()
        return talk

    async def compact(self, *, session_id: str, key: str, system: str, activity: Activity, focus: str = "") -> RunResult:
        """Compact the conversation now: Claude Code's own /compact, in the same session."""
        res = await self.run(("/compact " + focus).strip(), system=system, session_id=session_id, key=key,
                             activity=activity)
        if not res.compactions:
            raise AgentRunError("claude: the conversation wasn't compacted" + (f" ({res.text[:300]})" if res.text else ""))
        return res

    async def text(self, *, system: str, prompt: str) -> str:
        workdir = self.workdir("oneshot")
        argv = self._common(workdir) + ["--output-format", "json", "--no-session-persistence",
                                        "--system-prompt", system, "--tools", ""]
        out: dict[str, Any] = {}
        stderr = await self._exec(argv, self.env(workdir), workdir, prompt[-MAX_PROMPT:],
                                  lambda ev: out.update(ev) if ev.get("type") == "result" else None)
        if not out:
            raise AgentRunError("claude: " + stderr.strip()[-1000:])
        return str(out.get("result") or "")

    def login_command(self, extra: list[str]) -> tuple[list[str], dict[str, str]]:
        return [self.c.command, *(extra or ["auth", "login"])], self.env(None)

    def trust_folder(self, folder: Path) -> None:
        """Mark one of Overseer's own scratch folders as trusted in Overseer's Claude config, so the
        interactive UI doesn't stop at "Do you trust this folder?" (its default answer exits)."""
        cfg_file = self.home() / ".claude.json"
        try:
            data = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        except (OSError, json.JSONDecodeError):
            return  # never break the user's Claude config; they'll just see the question
        project = data.setdefault("projects", {}).setdefault(str(folder), {})
        if project.get("hasTrustDialogAccepted"):
            return
        project["hasTrustDialogAccepted"] = True
        tmp = cfg_file.with_name(f".claude.json.overseer-{uuid.uuid4().hex[:6]}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, cfg_file)

    def interactive_command(self, session_id: str, key: str) -> tuple[list[str], dict[str, str], Path]:
        workdir = self.workdir(key)  # Claude Code finds sessions by the folder they ran in
        self.trust_folder(workdir)
        argv = [self.c.command, "--resume", session_id, "--settings", json.dumps(self.settings(workdir)),
                "--strict-mcp-config"]
        if self.model:
            argv += ["--model", self.model]
        if self.effort:
            argv += ["--effort", self.effort]
        return argv, self.env(workdir), workdir


# ============================================================================ Codex

class CodexRunner(Runner):
    name = "codex"
    efforts = CODEX_EFFORTS

    def __init__(self, cfg: Config, sandbox: Sandbox | None = None):
        super().__init__(cfg, sandbox)
        self._catalog: dict[str, str] | None = None  # model -> its default reasoning effort

    def _configured_effort(self) -> str | None:
        return self.c.reasoning_effort

    @property
    def c(self):
        return self.cfg.agent.codex

    def home(self) -> Path:
        h = self.cfg.local.resolve(self.c.home)
        h.mkdir(mode=0o700, parents=True, exist_ok=True)
        return h

    def env(self, workdir: Path | None) -> dict[str, str]:
        env = self.base_env(self.c.pass_env, self.home(), workdir)
        env["CODEX_HOME"] = str(self.home())
        return env

    def overrides(self, workdir: Path, *, system: str | None, writable: bool, network: bool) -> list[str]:
        fs: dict[str, str] = {"/": "read"}
        if writable:
            fs[str(workdir)] = "write"
            for p in self.sandbox.writable:
                fs[str(p)] = "write"
            for p in self.sandbox.deny_write:
                fs[str(p)] = "read"
        for p in self.deny_read(shell=True):
            fs[p] = "deny"
        net: dict[str, Any] = {"enabled": network}
        if network:
            net.update({
                "mode": "limited",
                "domains": {d: "allow" for d in self.cfg.agent.allowed_domains},
                "unix_sockets": {**{str(p): "allow" for p in self.sandbox.unix_sockets},
                                 str(self.cfg.ctl_socket): "allow"},
            })
        out = [
            'default_permissions="overseer"',
            f"permissions.overseer.filesystem={_toml(fs)}",
            f"permissions.overseer.network={_toml(net)}",
            'approval_policy="never"',
            "project_doc_max_bytes=0",  # ignore AGENTS.md files in the scratch dir
            # The proxy settings passed for codex's own API traffic are not handed to the agent's commands.
            'shell_environment_policy.exclude=["*PROXY*", "*proxy*", "ALL_PROXY", "all_proxy"]',
            *[f"features.{f}=false" for f in CODEX_DISABLED_FEATURES],
        ]
        out.append(f"web_search={_toml(self.c.web_search)}")
        if system:
            out.append(f"developer_instructions={_toml(system)}")
        if self.effort:
            out.append(f"model_reasoning_effort={_toml(self.effort)}")
        return out + list(self.c.extra_config)

    def _argv(self, *, session_id: str | None, out_file: Path, overrides: list[str], ephemeral: bool) -> list[str]:
        argv = [self.c.command, "exec"]
        if session_id:
            argv.append("resume")
        argv += ["--json", "--skip-git-repo-check", "-o", str(out_file)]
        if ephemeral:
            argv.append("--ephemeral")
        if self.model:
            argv += ["-m", self.model]
        for o in overrides:
            argv += ["-c", o]
        if session_id:
            argv.append(session_id)
        return argv + ["-"]  # prompt on stdin

    async def _run(self, prompt: str, *, system: str | None, session_id: str | None, key: str,
                   activity: Activity, writable: bool, network: bool, ephemeral: bool) -> RunResult:
        workdir = self.workdir(key)
        if writable and self.sandbox.skills:  # the domain's skills, where codex finds them and its sandbox can read them
            from .addons import write_skills
            write_skills(workdir, (".agents", "skills"), self.sandbox.skills, prune=False)
        out_file = workdir / f".last-{uuid.uuid4().hex[:8]}.txt"
        argv = self._argv(session_id=session_id, out_file=out_file, ephemeral=ephemeral,
                          overrides=self.overrides(workdir, system=system, writable=writable, network=network))
        # codex reports the thread's running token total, not this run's; the caller subtracts the last one
        result = RunResult(text="", session_id=session_id, model=self.model, effort=self.effort, usage_is_total=True)
        compacted_before = self.compaction_count(session_id) if session_id else 0
        messages: list[str] = []
        errors: list[str] = []

        def on_event(ev: dict[str, Any]) -> None:
            t = ev.get("type", "")
            if t == "thread.started":
                result.session_id = ev.get("thread_id") or result.session_id
                if result.session_id:
                    activity.session(result.session_id)
            elif t in ("item.started", "item.completed"):
                item = ev.get("item") or {}
                kind = item.get("type")
                if kind == "command_execution":
                    if t == "item.started":
                        activity.command(item.get("command", ""))
                    else:
                        activity.output(item.get("command", ""), item.get("aggregated_output", ""), item.get("exit_code"))
                elif t != "item.completed":
                    pass
                elif kind == "file_change":
                    for ch in item.get("changes") or []:
                        activity.file(ch.get("path", ""))
                elif kind == "agent_message":
                    messages.append(item.get("text", ""))
                    activity.message(item.get("text", ""))
                elif kind == "reasoning":
                    activity.thought(item.get("text", ""))
                elif kind == "web_search":
                    activity.tool("web_search", item.get("query", ""))
                elif kind == "mcp_tool_call":
                    activity.tool(f"{item.get('server', '')}.{item.get('tool', '')}".strip("."))
            elif t == "turn.completed":
                usage = ev.get("usage") or {}
                result.input_tokens += int(usage.get("input_tokens") or 0)  # includes the cached part
                result.cached_tokens += int(usage.get("cached_input_tokens") or 0)
                result.output_tokens += int(usage.get("output_tokens") or 0)
            elif t in ("turn.failed", "error"):
                err = ev.get("error") or {}
                errors.append(err.get("message") if isinstance(err, dict) else str(ev.get("message", err)))

        stderr = await self._exec(argv, {**self.env(workdir), **activity.env}, workdir, prompt[:MAX_PROMPT], on_event)
        text = out_file.read_text().strip() if out_file.exists() else ""
        out_file.unlink(missing_ok=True)
        result.text = text or (messages[-1] if messages else "")
        if errors:  # out of usage part-way through a turn: not an answer, even with some text before it
            _check_limit("codex: " + "; ".join(e for e in errors if e)[-1500:])
        if not result.text and (errors or stderr):
            detail = "; ".join(e for e in errors if e) or "\n".join(stderr.strip().splitlines()[-15:])
            _check_limit("codex: " + detail[-1500:] + "\n" + stderr[-1500:])
            raise AgentRunError("codex: " + detail[-1500:])
        if not ephemeral and result.session_id:
            await self._fill_actual_settings(result)
            # codex's --json events don't report compactions either; its session log does
            for _ in range(self.compaction_count(result.session_id) - compacted_before):
                info = {"trigger": "auto", "pre_tokens": None, "post_tokens": None}
                result.compactions.append(info)
                activity.compacted(info)
        return result

    def _rollout(self, thread_id: str) -> Path | None:
        logs = sorted(self.home().glob(f"sessions/*/*/*/rollout-*{thread_id}.jsonl"), key=lambda f: f.stat().st_mtime)
        return logs[-1] if logs else None

    def compaction_count(self, thread_id: str) -> int:
        """How often the thread was compacted so far (its session log has a "compacted" entry for each)."""
        path = self._rollout(thread_id)
        if not path:
            return 0
        n = 0
        try:
            with path.open(errors="replace") as fh:
                for line in fh:
                    if '"compacted"' in line:
                        try:
                            n += json.loads(line).get("type") == "compacted"
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
        return n

    async def compact(self, *, session_id: str, key: str, system: str, activity: Activity, focus: str = "") -> RunResult:
        """Compact the conversation now. `codex exec` has no way to, so this asks codex's app server
        (JSON-RPC over stdio): resume the thread, then thread/compact/start. `focus` isn't supported."""
        workdir = self.workdir(key)
        argv = [self.c.command, "app-server"]
        if self.model:
            argv += ["-c", f"model={_toml(self.model)}"]
        for o in self.overrides(workdir, system=system, writable=True, network=True):
            argv += ["-c", o]
        before = self.compaction_count(session_id)
        activity.compacting()
        proc = await asyncio.create_subprocess_exec(
            *argv, env=self.env(workdir), cwd=str(workdir), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=32 * 1024 * 1024)

        async def send(msg: dict[str, Any]) -> None:
            proc.stdin.write((json.dumps(msg) + "\n").encode())
            await proc.stdin.drain()

        async def converse() -> None:
            await send({"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "overseer", "version": "1"}}})
            await send({"method": "initialized"})
            await send({"id": 2, "method": "thread/resume", "params": {"threadId": session_id}})
            await send({"id": 3, "method": "thread/compact/start", "params": {"threadId": session_id}})
            compacted = False
            while True:
                line = await proc.stdout.readline()
                if not line:
                    raise AgentRunError("codex app-server stopped before the compaction finished")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("error"):
                    err = msg["error"]
                    raise AgentRunError("codex: " + str(err.get("message") if isinstance(err, dict) else err)[:800])
                method = msg.get("method")
                item = (msg.get("params") or {}).get("item") or {}
                if method == "item/completed" and item.get("type") == "contextCompaction":
                    compacted = True
                elif method == "turn/completed":
                    turn = (msg.get("params") or {}).get("turn") or {}
                    if turn.get("status") not in (None, "completed"):
                        raise AgentRunError(f"codex: the compaction {turn.get('status')}: {turn.get('error')}"[:800])
                    if compacted:
                        return

        try:
            await asyncio.wait_for(converse(), timeout=self.cfg.agent.timeout_minutes * 60)
        except asyncio.TimeoutError:
            raise AgentRunError("codex: the compaction timed out")
        finally:
            if proc.returncode is None:
                proc.stdin.close()
                try:
                    await asyncio.wait_for(proc.wait(), 10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
        result = RunResult(text="", session_id=session_id, model=self.model, effort=self.effort)
        if self.compaction_count(session_id) <= before:
            raise AgentRunError("codex: the conversation wasn't compacted")
        info = {"trigger": "manual", "pre_tokens": None, "post_tokens": None}
        result.compactions.append(info)
        activity.compacted(info)
        return result

    # codex's --json events don't say which model ran, but its session log does.
    def session_settings(self, thread_id: str) -> tuple[str | None, str | None]:
        """(model, effort) of the latest turn, from the session log in CODEX_HOME/sessions.

        The effort is None when none was configured (the model's default applied)."""
        logs = sorted(self.home().glob(f"sessions/*/*/*/rollout-*{thread_id}.jsonl"), key=lambda f: f.stat().st_mtime)
        model = effort = None
        if not logs:
            return None, None
        try:
            with logs[-1].open(errors="replace") as fh:
                for line in fh:
                    if '"turn_context"' not in line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") != "turn_context":
                        continue
                    ctx = ev.get("payload") or {}
                    settings = (ctx.get("collaboration_mode") or {}).get("settings") or {}
                    model = ctx.get("model") or settings.get("model") or model
                    effort = settings.get("reasoning_effort") or ctx.get("effort")
        except OSError as e:
            log.debug("codex session log %s: %s", logs[-1], e)
        return model, effort

    async def default_effort(self, model: str) -> str | None:
        """The model's default reasoning effort, from codex's model catalog (`codex debug models`)."""
        if self._catalog is None:
            self._catalog = {}
            try:
                proc = await asyncio.create_subprocess_exec(
                    self.c.command, "debug", "models", env=self.env(None), cwd=str(self.home()),
                    stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                for m in json.loads(out or b"{}").get("models") or []:
                    if m.get("slug") and m.get("default_reasoning_level"):
                        self._catalog[m["slug"]] = m["default_reasoning_level"]
            except (OSError, asyncio.TimeoutError, ValueError, AttributeError) as e:
                log.debug("codex model catalog unavailable: %s", e)
        return self._catalog.get(model)

    async def _fill_actual_settings(self, result: RunResult) -> None:
        model, effort = self.session_settings(result.session_id)
        result.model = model or result.model
        if effort:
            result.effort = effort
        elif not result.effort and result.model:
            default = await self.default_effort(result.model)
            result.effort = f"{default} (model default)" if default else None

    async def run(self, prompt: str, *, system: str, session_id: str | None, key: str,
                  activity: Activity, steering: Steering | None = None) -> RunResult:
        if steering is None:
            return await self._run(prompt, system=system, session_id=session_id, key=key, activity=activity,
                                   writable=True, network=True, ephemeral=False)
        return await self._app_run(prompt, system=system, session_id=session_id, key=key, activity=activity,
                                   steering=steering)

    async def _app_run(self, prompt: str, *, system: str, session_id: str | None, key: str, activity: Activity,
                       steering: Steering) -> RunResult:
        """A request through codex's app server (JSON-RPC over stdio) rather than `codex exec`: the same thread,
        sandbox and settings, and messages sent while it works go into the running turn (turn/steer)."""
        workdir = self.workdir(key)
        if self.sandbox.skills:
            from .addons import write_skills
            write_skills(workdir, (".agents", "skills"), self.sandbox.skills, prune=False)
        argv = [self.c.command, "app-server"]
        for o in self.overrides(workdir, system=system, writable=True, network=True):
            argv += ["-c", o]
        if self.model:  # (last, and in thread/start: like exec's -m, it wins over the config's model)
            argv += ["-c", f"model={_toml(self.model)}"]
        # codex reports the thread's running token total, not this run's; the caller subtracts the last one
        result = RunResult(text="", session_id=session_id, model=self.model, effort=self.effort, usage_is_total=True)
        messages: list[str] = []
        errors: list[str] = []
        done = asyncio.Event()
        turn: dict[str, str] = {}

        def notified(method: str, params: dict[str, Any]) -> None:
            item = params.get("item") or {}
            kind = item.get("type")
            if method == "item/started" and kind == "commandExecution":
                activity.command(item.get("command", ""))
            elif method == "item/completed":
                if kind == "commandExecution":
                    activity.output(item.get("command", ""), item.get("aggregatedOutput") or "", item.get("exitCode"))
                elif kind == "fileChange":
                    for ch in item.get("changes") or []:
                        activity.file(ch.get("path", ""))
                elif kind == "agentMessage":
                    messages.append(item.get("text", ""))
                    activity.message(item.get("text", ""))
                elif kind == "reasoning":
                    activity.thought("\n".join(str(x) for x in item.get("summary") or []))
                elif kind == "webSearch":
                    activity.tool("web_search", item.get("query", ""))
                elif kind == "mcpToolCall":
                    activity.tool(f"{item.get('server', '')}.{item.get('tool', '')}".strip("."))
                elif kind == "contextCompaction":
                    info = {"trigger": "auto", "pre_tokens": None, "post_tokens": None}
                    result.compactions.append(info)
                    activity.compacted(info)
            elif method == "thread/tokenUsage/updated":
                total = (params.get("tokenUsage") or {}).get("total") or {}
                result.input_tokens = int(total.get("inputTokens") or 0)  # includes the cached part
                result.cached_tokens = int(total.get("cachedInputTokens") or 0)
                result.output_tokens = int(total.get("outputTokens") or 0)
            elif method == "error" and not params.get("willRetry"):
                err = params.get("error") or {}
                errors.append(f"{err.get('message', '')} {json.dumps(err.get('codexErrorInfo') or '')}".strip())
            elif method == "turn/completed" and (params.get("turn") or {}).get("id") == turn.get("id"):
                t = params["turn"]
                if t.get("status") not in (None, "completed"):
                    err = t.get("error") or {}
                    errors.append(err.get("message") or f"the turn {t.get('status')}")
                done.set()

        async def talk(send, close, events):
            replies: dict[int, asyncio.Future] = {}
            ids = iter(range(1, 1_000_000))

            async def read():
                async for msg in events:
                    if "id" in msg and "method" not in msg:
                        fut = replies.pop(msg["id"], None)
                        if fut and not fut.done():
                            fut.set_result(msg)
                    elif "id" in msg:  # a request of codex's: approvals are off, so this shouldn't come - decline
                        answer = ({"decision": "decline"} if msg["method"].lower().endswith("approval")
                                  else None)
                        await send(json.dumps({"id": msg["id"], **({"result": answer} if answer else
                                              {"error": {"code": -32601, "message": "not supported by Overseer"}})})
                                   + "\n")
                    else:
                        try:
                            notified(msg.get("method", ""), msg.get("params") or {})
                        except Exception:
                            log.exception("codex app-server notification")
                for fut in replies.values():  # (it ended: nothing more will be answered)
                    if not fut.done():
                        fut.set_exception(AgentRunError("codex app-server stopped"))
                done.set()

            async def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
                i = next(ids)
                replies[i] = asyncio.get_running_loop().create_future()
                if not await send(json.dumps({"id": i, "method": method, "params": params}) + "\n"):
                    raise AgentRunError("codex app-server stopped")
                msg = await replies[i]
                if msg.get("error"):
                    err = msg["error"]
                    raise AgentRunError("codex: " + str(err.get("message") if isinstance(err, dict) else err)[:800])
                return msg.get("result") or {}

            reader = asyncio.create_task(read())
            try:
                await call("initialize", {"clientInfo": {"name": "overseer", "version": "1"}})
                await send(json.dumps({"method": "initialized"}) + "\n")
                which = {"cwd": str(workdir), **({"model": self.model} if self.model else {})}
                if session_id:
                    started = await call("thread/resume", {"threadId": session_id, **which})
                else:
                    started = await call("thread/start", which)
                result.session_id = (started.get("thread") or {}).get("id") or session_id
                result.model = started.get("model") or result.model
                result.effort = started.get("reasoningEffort") or result.effort
                if result.session_id:
                    activity.session(result.session_id)
                t = await call("turn/start", {"threadId": result.session_id,
                                              "input": [{"type": "text", "text": prompt[:MAX_PROMPT]}]})
                turn["id"] = (t.get("turn") or {}).get("id", "")

                async def deliver(text: str) -> bool:
                    if steering.closed or done.is_set():
                        return False
                    try:
                        await call("turn/steer", {"threadId": result.session_id, "expectedTurnId": turn["id"],
                                                  "input": [{"type": "text", "text": text}]})
                    except AgentRunError:  # (the turn had just ended: it's the next request)
                        return False
                    activity.steered(text)  # (the turn has it)
                    return True

                steering.instant = True
                await steering.ready(deliver)
                await done.wait()
            finally:
                steering.close()
                close()
                await reader

        stderr = await self._converse(argv, {**self.env(workdir), **activity.env}, workdir, talk)
        result.text = messages[-1] if messages else ""
        if result.session_id and not result.effort:
            await self._fill_actual_settings(result)  # (the model's own default effort, from its session log)
        if errors:  # out of usage part-way through a turn: not an answer, even with some text before it
            _check_limit("codex: " + "; ".join(e for e in errors if e)[-1500:])
        if not result.text and (errors or stderr):
            detail = "; ".join(e for e in errors if e) or "\n".join(stderr.strip().splitlines()[-15:])
            _check_limit("codex: " + detail[-1500:] + "\n" + stderr[-1500:])
            raise AgentRunError("codex: " + detail[-1500:])
        return result

    async def text(self, *, system: str, prompt: str) -> str:
        res = await self._run(f"{system}\n\n<input>\n{prompt[-MAX_PROMPT:]}\n</input>", system=None, session_id=None,
                              key="oneshot", activity=Activity(), writable=False, network=False, ephemeral=True)
        return res.text

    def login_command(self, extra: list[str]) -> tuple[list[str], dict[str, str]]:
        # Overseer usually runs on a headless server, where codex's browser login can't work:
        # default to the device-code flow (open the printed URL on any device, enter the code).
        return [self.c.command, "login", *(extra or ["--device-auth"])], self.env(None)

    def interactive_command(self, session_id: str, key: str) -> tuple[list[str], dict[str, str], Path]:
        workdir = self.workdir(key)
        argv = [self.c.command, "resume"]
        if self.model:
            argv += ["-c", f"model={_toml(self.model)}"]
        for o in self.overrides(workdir, system=None, writable=True, network=True):
            argv += ["-c", o]
        return argv + [session_id], self.env(workdir), workdir


def _handle(on_event: Callable[[dict[str, Any]], None], event: dict[str, Any]) -> None:
    """One of the CLI's events (a handler that fails is logged: the run goes on)."""
    try:
        on_event(event)
    except Exception:
        log.exception("event handler failed")


def _tool_detail(inp: dict[str, Any]) -> str:
    """The interesting argument of a Claude Code tool call (Read path, Grep pattern, fetched URL...)."""
    for key in ("file_path", "path", "pattern", "url", "query", "description", "prompt"):
        if inp.get(key):
            return str(inp[key])[:300]
    return ""


def _claude_usage(ev: dict[str, Any], result: RunResult) -> None:
    """Token counts and cost from Claude Code's final `result` event."""
    per_model = ev.get("modelUsage") or {}
    if per_model:
        for name, u in per_model.items():
            result.input_tokens += int(u.get("inputTokens", 0)) + int(u.get("cacheReadInputTokens", 0)) \
                + int(u.get("cacheCreationInputTokens", 0))
            result.cached_tokens += int(u.get("cacheReadInputTokens", 0))
            result.output_tokens += int(u.get("outputTokens", 0))
        if not result.model:
            result.model = max(per_model, key=lambda n: per_model[n].get("outputTokens", 0))
    elif isinstance(ev.get("usage"), dict):
        u = ev["usage"]
        result.input_tokens += int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)) \
            + int(u.get("cache_creation_input_tokens", 0))
        result.cached_tokens += int(u.get("cache_read_input_tokens", 0))
        result.output_tokens += int(u.get("output_tokens", 0))
    cost = ev.get("total_cost_usd")
    if cost is None and per_model:
        cost = sum(float(u.get("costUSD", 0)) for u in per_model.values())
    if cost is not None:
        result.cost_usd = float(cost)


def make_runner(cfg: Config, backend: str, sandbox: Sandbox | None = None) -> Runner:
    return ClaudeCodeRunner(cfg, sandbox) if backend == "claude_code" else CodexRunner(cfg, sandbox)
