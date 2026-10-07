"""Stdio MCP server that exposes Overseer's tools to codex / opencode.

Started by the agent CLI (never by hand). It is a thin, stateless proxy: tools/list and
tools/call are forwarded to the running Overseer over the bridge socket (see bridge.py),
identified by the per-run token in $OVERSEER_BRIDGE_TOKEN.

Implements the small subset of MCP that tool servers need: initialize, ping, tools/list,
tools/call (newline-delimited JSON-RPC 2.0 over stdin/stdout).
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any

SERVER_INFO = {"name": "overseer", "version": "0.3.0"}
DEFAULT_PROTOCOL = "2025-06-18"


class Bridge:
    def __init__(self, path: str, token: str):
        self.path = path
        self.token = token
        self.sock: socket.socket | None = None
        self.buf = b""

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.sock is None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.path)
        self.sock.sendall(json.dumps({**payload, "token": self.token}).encode() + b"\n")
        while b"\n" not in self.buf:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                self.sock = None
                raise ConnectionError("Overseer closed the bridge connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)


def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle(bridge: Bridge, msg: dict[str, Any]) -> dict[str, Any] | None:
    method = msg.get("method")
    msg_id = msg.get("id")
    if msg_id is None:  # notification (e.g. notifications/initialized)
        return None
    params = msg.get("params") or {}
    if method == "initialize":
        return _result(msg_id, {
            "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": "Overseer tools: the only way to read or change anything on the HPC.",
        })
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        resp = bridge.request({"op": "list"})
        if "error" in resp:
            return _error(msg_id, -32603, resp["error"])
        tools = [{"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]} for t in resp["tools"]]
        return _result(msg_id, {"tools": tools})
    if method == "tools/call":
        resp = bridge.request({"op": "call", "tool": params.get("name"), "args": params.get("arguments") or {}})
        if "error" in resp:
            return _result(msg_id, {"content": [{"type": "text", "text": resp["error"]}], "isError": True})
        return _result(msg_id, {"content": [{"type": "text", "text": resp["content"]}], "isError": bool(resp["is_error"])})
    if method in ("resources/list", "prompts/list"):
        return _result(msg_id, {method.split("/")[0]: []})
    return _error(msg_id, -32601, f"method not found: {method}")


def main() -> int:
    path = os.environ.get("OVERSEER_BRIDGE_SOCKET")
    token = os.environ.get("OVERSEER_BRIDGE_TOKEN")
    if not path or not token:
        print("overseer.mcp_server must be started by Overseer (bridge env vars missing)", file=sys.stderr)
        return 2
    bridge = Bridge(path, token)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            out = _error(None, -32700, "parse error")
        else:
            try:
                out = handle(bridge, msg)
            except Exception as e:
                out = _error(msg.get("id"), -32603, f"{type(e).__name__}: {e}")
        if out is not None:
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
