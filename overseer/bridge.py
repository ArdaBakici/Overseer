"""Local bridge between external agent CLIs (codex, opencode) and Overseer's tools.

The CLI starts ``python -m overseer.mcp_server`` as an MCP server. That process holds no
state and no credentials: it forwards every tool call over a Unix socket (mode 600, in
``local.tmp_dir``) to the running Overseer, which executes it with exactly the same policy
checks, backups and Telegram reporting as the built-in Claude agent.

Each agent run gets a random token that is only valid while that run is in progress and
is bound to one conversation (session).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import Agent, Session

log = logging.getLogger(__name__)

MAX_LINE = 16 * 1024 * 1024


@dataclass
class _Run:
    session: Session
    actions: list[str]
    calls: int = 0
    tools: list[str] = field(default_factory=list)


class ToolBridge:
    def __init__(self, agent: Agent, socket_path: Path):
        self.agent = agent
        self.socket_path = socket_path
        self._runs: dict[str, _Run] = {}
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        if self._server:
            return
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            self.socket_path.unlink()
        old = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(self._handle, path=str(self.socket_path), limit=MAX_LINE)
        finally:
            os.umask(old)
        os.chmod(self.socket_path, 0o600)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @asynccontextmanager
    async def run(self, session: Session, actions: list[str]):
        """Register one agent run; yields the env vars the MCP server needs."""
        await self.start()
        token = secrets.token_urlsafe(32)
        self._runs[token] = _Run(session, actions)
        try:
            yield {"OVERSEER_BRIDGE_SOCKET": str(self.socket_path), "OVERSEER_BRIDGE_TOKEN": token}
        finally:
            self._runs.pop(token, None)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                try:
                    req = json.loads(line)
                    resp = await self._dispatch(req)
                except Exception as e:  # malformed request: answer, don't crash the server
                    log.exception("bridge request failed")
                    resp = {"error": f"{type(e).__name__}: {e}"}
                writer.write(json.dumps(resp).encode() + b"\n")
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def _dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        from .agent import TOOLS

        run = self._runs.get(str(req.get("token", "")))
        if run is None:
            return {"error": "invalid or expired token"}
        op = req.get("op")
        if op == "list":
            return {"tools": TOOLS}
        if op == "call":
            run.calls += 1
            name = str(req.get("tool", ""))
            run.tools.append(name)
            content, is_error = await self.agent._run_tool(run.session, name, req.get("args") or {}, run.actions)
            return {"content": content, "is_error": is_error}
        return {"error": f"unknown op {op!r}"}
