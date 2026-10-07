"""Run the agent with an external coding-agent CLI: OpenAI Codex (`codex exec`) or opencode (`opencode run`).

Both CLIs come with their own shell / file-editing tools that act on *this* machine, which
holds your SSH key to the HPC. Overseer therefore runs them locked down:

* their built-in tools are disabled (codex: shell/exec/browser/etc. features off, read-only
  sandbox, no approvals, no web search; opencode: every built-in tool off and denied),
* the only tools they get are Overseer's, through a local MCP server (see bridge.py), so
  every action goes through the same policy checks, backups, approvals and Telegram reports,
* they run with a scrubbed environment (no Telegram token, no SSH agent, only the API-key
  variables you list), in a scratch dir under local.tmp_dir, with HOME/CODEX_HOME/XDG dirs
  inside local.state_dir - so even the CLI's own files stay inside the allowed local folders.

Any built-in tool use that still shows up in the CLI's event stream is reported to Telegram.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config

log = logging.getLogger(__name__)

MAX_PROMPT = 100_000
BASE_ENV = ["PATH", "LANG", "LC_ALL", "TZ"]

# Codex features that would give the model tools acting on this machine (or beyond Overseer).
CODEX_DISABLED_FEATURES = [
    "shell_tool", "unified_exec", "unified_exec_tty", "shell_snapshot", "view_image", "browser_use",
    "browser_use_external", "browser_use_full_cdp_access", "in_app_browser", "computer_use",
    "image_generation", "apps", "plugins", "remote_plugin", "multi_agent", "multi_agent_v2",
    "skill_mcp_dependency_install", "workspace_dependencies", "tool_suggest", "memories",
    "in_app_local_automation", "code_mode_host", "worktrees", "hooks", "goals",
]


class AgentCLIError(RuntimeError):
    pass


@dataclass
class CLIResult:
    text: str
    session_id: str | None
    native_actions: list[str] = field(default_factory=list)


def _toml(value: Any) -> str:
    """Serialize a value as a TOML literal for `codex -c key=value` (JSON strings are valid TOML)."""
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


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class CLIAgent:
    name = "cli"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.timeout = cfg.llm.cli_timeout_minutes * 60

    @property
    def enabled(self) -> bool:
        return self.cfg.llm.enabled

    # ------------------------------------------------------------------ environment
    def _mcp_env(self, bridge_env: dict[str, str]) -> dict[str, str]:
        pkg_parent = str(Path(__file__).resolve().parent.parent)
        return {**bridge_env, "PYTHONPATH": pkg_parent, "PYTHONDONTWRITEBYTECODE": "1"}

    def _mcp_command(self) -> list[str]:
        return [sys.executable, "-m", "overseer.mcp_server"]

    def _env(self, pass_env: list[str], extra: dict[str, str]) -> dict[str, str]:
        env = {k: os.environ[k] for k in BASE_ENV + pass_env if k in os.environ}
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"
        env.update(extra)
        return env

    def _workdir(self, key: str) -> Path:
        d = self.cfg.local.tmp("agents", key)
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------ running
    async def _exec(self, argv: list[str], env: dict[str, str], cwd: Path, on_event) -> None:
        if not shutil.which(argv[0], path=env.get("PATH")):
            raise AgentCLIError(f"{argv[0]!r} not found on PATH; install it or set llm.{self.name}.command")
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=16 * 1024 * 1024,
            start_new_session=True,
        )
        err_task = asyncio.create_task(proc.stderr.read())

        async def pump():
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("%s: %s", self.name, line[:500])
                    continue
                on_event(event)
            await proc.wait()

        try:
            await asyncio.wait_for(pump(), timeout=self.timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            proc.kill()
            await proc.wait()
            raise AgentCLIError(f"{self.name} did not finish within {self.timeout // 60} minutes")
        finally:
            stderr = (await err_task).decode(errors="replace")
        if proc.returncode != 0:
            tail = "\n".join(stderr.strip().splitlines()[-15:])
            raise AgentCLIError(f"{self.name} exited with code {proc.returncode}: {tail[-1500:]}")

    async def run(self, prompt: str, *, session_id: str | None, session_key: str, actions: list[str],
                  bridge_env: dict[str, str] | None) -> CLIResult:
        raise NotImplementedError

    def login_command(self, extra: list[str]) -> tuple[list[str], dict[str, str]]:
        """argv + env to log the CLI in, using the same isolated home Overseer runs it with."""
        raise NotImplementedError

    async def text(self, *, system: str, prompt: str, effort: str | None = None, max_tokens: int = 4000) -> str:
        """One-shot completion without tools (job reports, status lines)."""
        full = f"{system}\n\nDo not use any tools; answer directly.\n\n<input>\n{prompt[-MAX_PROMPT:]}\n</input>"
        res = await self.run(full, session_id=None, session_key=f"oneshot-{uuid.uuid4().hex[:8]}",
                             actions=[], bridge_env=None)
        return res.text


class CodexAgent(CLIAgent):
    name = "codex"

    def argv(self, prompt: str, session_id: str | None, out_file: Path, bridge_env: dict[str, str] | None) -> list[str]:
        c = self.cfg.llm.codex
        overrides = [
            'sandbox_mode="read-only"',
            'approval_policy="never"',
            'web_search="disabled"',
            'shell_environment_policy.inherit="none"',
            "project_doc_max_bytes=0",  # ignore AGENTS.md files
            *[f"features.{f}=false" for f in CODEX_DISABLED_FEATURES],
        ]
        if bridge_env is not None:
            overrides += [
                f"mcp_servers.overseer.command={_toml(self._mcp_command()[0])}",
                f"mcp_servers.overseer.args={_toml(self._mcp_command()[1:])}",
                f"mcp_servers.overseer.env={_toml(self._mcp_env(bridge_env))}",
                "mcp_servers.overseer.startup_timeout_sec=60",
                # Overseer's tools are safe to run without codex prompting: they enforce their own
                # policy, and everything consequential (sbatch, scancel, new folders) is approved in Telegram.
                'mcp_servers.overseer.default_tools_approval_mode="approve"',
                f"mcp_servers.overseer.tool_timeout_sec={max(600, self.timeout)}",
            ]
        if c.reasoning_effort:
            overrides.append(f"model_reasoning_effort={_toml(c.reasoning_effort)}")
        overrides += c.extra_config
        argv = [c.command, "exec"]
        if session_id:
            argv += ["resume"]
        argv += ["--json", "--skip-git-repo-check", "-o", str(out_file)]
        if session_id is None and bridge_env is None:
            argv.append("--ephemeral")  # one-shot report: don't keep a session
        if c.model:
            argv += ["-m", c.model]
        for o in overrides:
            argv += ["-c", o]
        if session_id:
            argv.append(session_id)
        argv.append(prompt[:MAX_PROMPT])
        return argv

    def env(self) -> dict[str, str]:
        c = self.cfg.llm.codex
        home = self.cfg.local.resolve(c.home)
        (home / "home").mkdir(mode=0o700, parents=True, exist_ok=True)
        return self._env(c.pass_env, {"CODEX_HOME": str(home), "HOME": str(home / "home")})

    def login_command(self, extra: list[str]) -> tuple[list[str], dict[str, str]]:
        return [self.cfg.llm.codex.command, "login", *extra], self.env()

    async def run(self, prompt: str, *, session_id: str | None, session_key: str, actions: list[str],
                  bridge_env: dict[str, str] | None) -> CLIResult:
        cwd = self._workdir(session_key)
        out_file = cwd / f"last-{uuid.uuid4().hex[:8]}.txt"
        env = self.env()
        result = CLIResult(text="", session_id=session_id)
        messages: list[str] = []
        errors: list[str] = []

        def on_event(ev: dict[str, Any]) -> None:
            t = ev.get("type", "")
            if t == "thread.started":
                result.session_id = ev.get("thread_id") or result.session_id
            elif t in ("item.completed", "item.started"):
                item = ev.get("item") or {}
                kind = item.get("type")
                if t == "item.completed" and kind == "agent_message":
                    messages.append(item.get("text", ""))
                elif t == "item.started" and kind == "command_execution":
                    result.native_actions.append(f"⚠️ codex ran a local command: {item.get('command', '')!r}")
                elif t == "item.completed" and kind == "file_change":
                    result.native_actions.append(f"⚠️ codex tried to change local files: {item.get('changes')}")
                elif t == "item.completed" and kind == "web_search":
                    result.native_actions.append(f"🌐 codex web search: {item.get('query', '')!r}")
            elif t in ("turn.failed", "error"):
                err = ev.get("error") or {}
                errors.append(err.get("message") if isinstance(err, dict) else str(ev.get("message", err)))

        await self._exec(self.argv(prompt, session_id, out_file, bridge_env), env, cwd, on_event)
        if errors and not messages:
            raise AgentCLIError("codex: " + "; ".join(e for e in errors if e)[:1500])
        text = out_file.read_text().strip() if out_file.exists() else ""
        out_file.unlink(missing_ok=True)
        result.text = text or (messages[-1] if messages else "")
        return result


class OpencodeAgent(CLIAgent):
    name = "opencode"

    def config(self, bridge_env: dict[str, str] | None) -> dict[str, Any]:
        o = self.cfg.llm.opencode
        cfg: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "snapshot": False,
        }
        cfg = _deep_merge(cfg, o.extra_config)
        # Applied last so extra_config can't loosen them.
        cfg["tools"] = {"*": False, "overseer_*": bridge_env is not None}
        cfg["permission"] = {"edit": "deny", "bash": "deny", "webfetch": "deny", "external_directory": "deny"}
        cfg["mcp"] = {}
        if bridge_env is not None:
            cfg["mcp"]["overseer"] = {
                "type": "local", "enabled": True, "command": self._mcp_command(),
                "environment": self._mcp_env(bridge_env), "timeout": 60_000,
            }
        return cfg

    def argv(self, prompt: str, session_id: str | None) -> list[str]:
        o = self.cfg.llm.opencode
        argv = [o.command, "run", "--format", "json", "--pure"]
        if session_id:
            argv += ["--session", session_id]
        if o.model:
            argv += ["-m", o.model]
        if o.variant:
            argv += ["--variant", o.variant]
        return argv + ["--", prompt[:MAX_PROMPT]]

    def env(self, bridge_env: dict[str, str] | None) -> dict[str, str]:
        o = self.cfg.llm.opencode
        home = self.cfg.local.resolve(o.home)
        dirs = {k: home / k.lower() for k in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")}
        for d in [*dirs.values(), home / "home"]:
            d.mkdir(mode=0o700, parents=True, exist_ok=True)
        return self._env(o.pass_env, {
            **{k: str(v) for k, v in dirs.items()},
            "HOME": str(home / "home"),
            "OPENCODE_CONFIG_CONTENT": json.dumps(self.config(bridge_env)),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_SHARE": "1",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
        })

    def login_command(self, extra: list[str]) -> tuple[list[str], dict[str, str]]:
        return [self.cfg.llm.opencode.command, "auth", "login", *extra], self.env(None)

    async def run(self, prompt: str, *, session_id: str | None, session_key: str, actions: list[str],
                  bridge_env: dict[str, str] | None) -> CLIResult:
        cwd = self._workdir(session_key)
        result = CLIResult(text="", session_id=session_id)
        texts: list[str] = []
        errors: list[str] = []

        def on_event(ev: dict[str, Any]) -> None:
            sid = ev.get("sessionID") or (ev.get("part") or {}).get("sessionID")
            if sid:
                result.session_id = sid
            t = ev.get("type", "")
            part = ev.get("part") or {}
            if t == "text" and part.get("text"):
                texts.append(part["text"])
            elif t == "tool_use":
                texts.clear()  # the answer is the text after the last tool call
                tool = str(part.get("tool", ""))
                if not tool.startswith("overseer"):
                    result.native_actions.append(f"⚠️ opencode used built-in tool {tool!r}")
            elif t == "error":
                err = ev.get("error") or {}
                data = err.get("data") if isinstance(err, dict) else None
                errors.append((data or {}).get("message") if isinstance(data, dict) else str(err))

        await self._exec(self.argv(prompt, session_id), self.env(bridge_env), cwd, on_event)
        if errors and not texts:
            raise AgentCLIError("opencode: " + "; ".join(e for e in errors if e)[:1500])
        result.text = "\n".join(texts).strip()
        return result


def make_cli_agent(cfg: Config) -> CLIAgent | None:
    if cfg.llm.backend == "codex":
        return CodexAgent(cfg)
    if cfg.llm.backend == "opencode":
        return OpencodeAgent(cfg)
    return None
