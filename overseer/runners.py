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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .sshlink import agent_bin_dir

log = logging.getLogger(__name__)

MAX_PROMPT = 200_000
BASE_ENV = ["LANG", "LC_ALL", "TZ"]
SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")
PARSABLE_RE = re.compile(r"^\s*(\d+)(?:;\S+)?\s*$", re.M)

# Codex features with tools that act outside the shell sandbox or aren't useful headless.
CODEX_DISABLED_FEATURES = [
    "browser_use", "browser_use_external", "browser_use_full_cdp_access", "in_app_browser", "computer_use",
    "image_generation", "apps", "plugins", "remote_plugin", "in_app_local_automation", "tool_suggest",
    "skill_mcp_dependency_install", "workspace_dependencies", "memories", "goals", "hooks",
]


class AgentRunError(RuntimeError):
    pass


@dataclass
class RunResult:
    text: str
    session_id: str | None
    is_error: bool = False
    model: str | None = None  # the model that actually ran, when the CLI reports it
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # Claude Code reports it; Codex doesn't


@dataclass
class Activity:
    """What the agent did during one request (for Telegram reports and job tracking)."""

    commands: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    jobs: list[tuple[str, str]] = field(default_factory=list)  # (job_id, command)
    on_job: Callable[[str, str], None] | None = None
    on_command: Callable[[str], None] | None = None

    def command(self, cmd: str) -> None:
        cmd = display_command(cmd)
        self.commands.append(cmd)
        if self.on_command:
            self.on_command(cmd)

    def output(self, cmd: str, output: str) -> None:
        ids = SUBMITTED_RE.findall(output or "")
        if not ids and "sbatch" in cmd and "--parsable" in cmd:
            ids = PARSABLE_RE.findall(output or "")[-1:]
        for jid in ids:
            if all(jid != j for j, _ in self.jobs):
                self.jobs.append((jid, display_command(cmd)))
                if self.on_job:
                    self.on_job(jid, display_command(cmd))

    def file(self, path: str) -> None:
        if path and path not in self.files:
            self.files.append(path)


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

    def __init__(self, cfg: Config):
        self.cfg = cfg
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

    def deny_read(self) -> list[str]:
        """Local paths the agent's commands must never read (only existing ones are listed)."""
        home = Path.home()
        uid = os.getuid()
        candidates = [
            self.cfg.ssh_dir, home / ".ssh", self.cfg.local.state(), self.cfg.local.env_path,
            home / ".gnupg", home / ".docker", home / ".aws", home / ".config" / "gcloud",
            Path("/var/run/docker.sock"), Path("/run/docker.sock"), Path("/run/containerd"),
            Path("/run/podman"), Path("/var/run/podman"), Path(f"/run/user/{uid}"),
        ]
        if os.environ.get("SSH_AUTH_SOCK"):
            candidates.append(Path(os.environ["SSH_AUTH_SOCK"]).parent)
        candidates += [Path(p).expanduser() for p in self.cfg.agent.deny_read]
        paths = []
        for p in candidates:
            if p.exists() or p.is_symlink():
                paths.append(p if p.is_socket() else p.resolve())
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
        if workdir:
            env["TMPDIR"] = str(workdir / ".tmp")
        return env

    # ------------------------------------------------------------------ process handling
    async def _exec(self, argv: list[str], env: dict[str, str], cwd: Path, stdin: str,
                    on_event: Callable[[dict[str, Any]], None]) -> str:
        if not shutil.which(argv[0], path=env.get("PATH")):
            raise AgentRunError(f"{argv[0]!r} not found on PATH; install it or set agent.{self.name}.command")
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=32 * 1024 * 1024,
            start_new_session=True,  # own process group, so /stop and timeouts kill everything it started
        )
        err_task = asyncio.create_task(proc.stderr.read())

        async def pump():
            proc.stdin.write(stdin.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("%s: %s", self.name, line[:500])
                    continue
                try:
                    on_event(event)
                except Exception:
                    log.exception("event handler failed")
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
        except asyncio.CancelledError:
            kill()
            await proc.wait()
            raise
        finally:
            stderr = (await err_task).decode(errors="replace")
        if proc.returncode != 0:
            return stderr
        return ""

    # ------------------------------------------------------------------ API
    async def run(self, prompt: str, *, system: str, session_id: str | None, key: str,
                  activity: Activity) -> RunResult:
        raise NotImplementedError

    async def text(self, *, system: str, prompt: str) -> str:
        """One-shot answer without tools (job reports)."""
        raise NotImplementedError

    def login_command(self, extra: list[str]) -> tuple[list[str], dict[str, str]]:
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
                "filesystem": {"allowWrite": [str(workdir)], "denyRead": deny},
                # The HPC is reached through the unix socket of Overseer's SSH master.
                "network": {"allowAllUnixSockets": True, "allowedDomains": list(self.cfg.agent.allowed_domains)},
            },
            "permissions": {
                "defaultMode": "acceptEdits",
                "allow": ["WebSearch", "WebFetch"],
                "deny": [rule for p in deny for rule in (f"Read(/{p}/**)", f"Read(/{p})", f"Edit(/{p}/**)")],
            },
        }
        return _merge(self.c.extra_settings, enforced)

    def _common(self, workdir: Path) -> list[str]:
        argv = [self.c.command, "-p", "--settings", json.dumps(self.settings(workdir)), "--strict-mcp-config"]
        if self.model:
            argv += ["--model", self.model]
        if self.effort:
            argv += ["--effort", self.effort]
        return argv

    async def run(self, prompt: str, *, system: str, session_id: str | None, key: str,
                  activity: Activity) -> RunResult:
        workdir = self.workdir(key)
        argv = self._common(workdir) + ["--output-format", "stream-json", "--verbose", "--append-system-prompt", system]
        if session_id:
            argv += ["--resume", session_id]
        pending: dict[str, str] = {}
        result = RunResult(text="", session_id=session_id)
        texts: list[str] = []

        def on_event(ev: dict[str, Any]) -> None:
            t = ev.get("type")
            if ev.get("session_id"):
                result.session_id = ev["session_id"]
            if t == "system" and ev.get("subtype") == "init" and ev.get("model"):
                result.model = ev["model"]
            if t == "assistant":
                for block in (ev.get("message") or {}).get("content") or []:
                    if block.get("type") == "text" and block.get("text"):
                        texts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        name, inp = block.get("name"), block.get("input") or {}
                        if name == "Bash" and inp.get("command"):
                            pending[block.get("id")] = inp["command"]
                            activity.command(inp["command"])
                        elif name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
                            activity.file(inp.get("file_path") or inp.get("notebook_path") or "")
            elif t == "user":
                for block in (ev.get("message") or {}).get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                        content = block.get("content")
                        if isinstance(content, list):
                            content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
                        activity.output(pending.pop(block["tool_use_id"]), str(content or ""))
            elif t == "result":
                result.is_error = bool(ev.get("is_error"))
                if ev.get("result"):
                    result.text = str(ev["result"])
                _claude_usage(ev, result)

        stderr = await self._exec(argv, self.env(workdir), workdir, prompt[:MAX_PROMPT], on_event)
        if not result.text:
            result.text = texts[-1] if texts else ""
        if stderr and not result.text:
            raise AgentRunError("claude: " + "\n".join(stderr.strip().splitlines()[-15:])[-1500:])
        return result

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


# ============================================================================ Codex

class CodexRunner(Runner):
    name = "codex"
    efforts = CODEX_EFFORTS

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
        for p in self.deny_read():
            fs[p] = "deny"
        net: dict[str, Any] = {"enabled": network}
        if network:
            net.update({
                "mode": "limited",
                "domains": {d: "allow" for d in self.cfg.agent.allowed_domains},
                "unix_sockets": {str(self.cfg.control_socket): "allow"},
            })
        out = [
            'default_permissions="overseer"',
            f"permissions.overseer.filesystem={_toml(fs)}",
            f"permissions.overseer.network={_toml(net)}",
            'approval_policy="never"',
            "project_doc_max_bytes=0",  # ignore AGENTS.md files in the scratch dir
            *[f"features.{f}=false" for f in CODEX_DISABLED_FEATURES],
        ]
        if not self.c.web_search:
            out.append('web_search="disabled"')
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
        out_file = workdir / f".last-{uuid.uuid4().hex[:8]}.txt"
        argv = self._argv(session_id=session_id, out_file=out_file, ephemeral=ephemeral,
                          overrides=self.overrides(workdir, system=system, writable=writable, network=network))
        result = RunResult(text="", session_id=session_id, model=self.model)  # codex doesn't report its model
        messages: list[str] = []
        errors: list[str] = []

        def on_event(ev: dict[str, Any]) -> None:
            t = ev.get("type", "")
            if t == "thread.started":
                result.session_id = ev.get("thread_id") or result.session_id
            elif t in ("item.started", "item.completed"):
                item = ev.get("item") or {}
                kind = item.get("type")
                if kind == "command_execution":
                    if t == "item.started":
                        activity.command(item.get("command", ""))
                    else:
                        activity.output(item.get("command", ""), item.get("aggregated_output", ""))
                elif kind == "file_change" and t == "item.completed":
                    for ch in item.get("changes") or []:
                        activity.file(ch.get("path", ""))
                elif kind == "agent_message" and t == "item.completed":
                    messages.append(item.get("text", ""))
            elif t == "turn.completed":
                usage = ev.get("usage") or {}
                result.input_tokens += int(usage.get("input_tokens") or 0)
                result.output_tokens += int(usage.get("output_tokens") or 0)
            elif t in ("turn.failed", "error"):
                err = ev.get("error") or {}
                errors.append(err.get("message") if isinstance(err, dict) else str(ev.get("message", err)))

        stderr = await self._exec(argv, self.env(workdir), workdir, prompt[:MAX_PROMPT], on_event)
        text = out_file.read_text().strip() if out_file.exists() else ""
        out_file.unlink(missing_ok=True)
        result.text = text or (messages[-1] if messages else "")
        if not result.text and (errors or stderr):
            detail = "; ".join(e for e in errors if e) or "\n".join(stderr.strip().splitlines()[-15:])
            raise AgentRunError("codex: " + detail[-1500:])
        return result

    async def run(self, prompt: str, *, system: str, session_id: str | None, key: str,
                  activity: Activity) -> RunResult:
        return await self._run(prompt, system=system, session_id=session_id, key=key, activity=activity,
                               writable=True, network=True, ephemeral=False)

    async def text(self, *, system: str, prompt: str) -> str:
        res = await self._run(f"{system}\n\n<input>\n{prompt[-MAX_PROMPT:]}\n</input>", system=None, session_id=None,
                              key="oneshot", activity=Activity(), writable=False, network=False, ephemeral=True)
        return res.text

    def login_command(self, extra: list[str]) -> tuple[list[str], dict[str, str]]:
        # Overseer usually runs on a headless server, where codex's browser login can't work:
        # default to the device-code flow (open the printed URL on any device, enter the code).
        return [self.c.command, "login", *(extra or ["--device-auth"])], self.env(None)


def _claude_usage(ev: dict[str, Any], result: RunResult) -> None:
    """Token counts and cost from Claude Code's final `result` event."""
    per_model = ev.get("modelUsage") or {}
    if per_model:
        for name, u in per_model.items():
            result.input_tokens += int(u.get("inputTokens", 0)) + int(u.get("cacheReadInputTokens", 0)) \
                + int(u.get("cacheCreationInputTokens", 0))
            result.output_tokens += int(u.get("outputTokens", 0))
        if not result.model:
            result.model = max(per_model, key=lambda n: per_model[n].get("outputTokens", 0))
    elif isinstance(ev.get("usage"), dict):
        u = ev["usage"]
        result.input_tokens += int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)) \
            + int(u.get("cache_creation_input_tokens", 0))
        result.output_tokens += int(u.get("output_tokens", 0))
    cost = ev.get("total_cost_usd")
    if cost is None and per_model:
        cost = sum(float(u.get("costUSD", 0)) for u in per_model.values())
    if cost is not None:
        result.cost_usd = float(cost)


def make_runner(cfg: Config, backend: str) -> Runner:
    return ClaudeCodeRunner(cfg) if backend == "claude_code" else CodexRunner(cfg)
