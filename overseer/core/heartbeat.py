"""What a running bot shows the TUI (`overseer tui`): a status file in its state dir, rewritten every few seconds.

The TUI only reads: this file, the bot's database (read-only), its transcripts and its log. A status file that stops
being rewritten, or whose process is gone, means the bot isn't running.

    {"pid", "started", "updated", "version", "domain", "config", "bot", "telegram": {"ok", "waiting"},
     "runs": [{"session", "project", "backend", "account", "origin", "started"}],
     "accounts": {backend: {"active", "names", "out": {name: until}}}, "forced", "clis": {backend: version},
     "domain": {"lines": [[label, text, ok]], "tables": [{"title", "columns", "rows", "details"}]}}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from importlib import metadata
from pathlib import Path

from .config import BACKENDS, Config

log = logging.getLogger(__name__)

STATUS_FILE = "status.json"
EVERY = 5  # seconds between rewrites
DOMAIN_EVERY = 30  # the domain's part (it may scan files) is refreshed this often
CLIS_EVERY = 600  # the CLIs' versions (a subprocess each)
STALE = 60  # a status file older than this: the bot isn't running (or is stuck)


def version() -> str:
    try:
        return metadata.version("overseer")
    except metadata.PackageNotFoundError:
        return "?"


def status_path(state_dir: Path) -> Path:
    return state_dir / STATUS_FILE


def write_json(path: Path, data: dict) -> None:
    """Atomically (a reader never sees half a file), private to this user."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, default=str)
    tmp.replace(path)


class Heartbeat:
    def __init__(self, cfg: Config, agent, notifier, domain):
        self.cfg, self.agent, self.notifier, self.domain = cfg, agent, notifier, domain
        self.started = time.time()
        self.path = status_path(cfg.local.state())
        self.bot_name = ""
        self._domain: dict = {}
        self._domain_at = 0.0
        self._clis: dict[str, str] = {}
        self._clis_at = 0.0

    async def snapshot(self) -> dict:
        now = time.time()
        if now - self._domain_at >= DOMAIN_EVERY:
            self._domain_at = now
            try:
                self._domain = await self.domain.status()
            except Exception as e:  # (never let the domain's part stop the heartbeat)
                log.warning("domain status: %s", e)
                self._domain = {"lines": [["status", f"{type(e).__name__}: {e}", False]], "tables": []}
        if now - self._clis_at >= CLIS_EVERY:
            self._clis_at = now
            for b in BACKENDS:
                r = self.agent.runner(b)
                if r.installed():
                    self._clis[b] = await r.version() or "?"
        acc = self.agent.accounts
        return {
            "pid": os.getpid(), "started": self.started, "updated": now, "version": version(),
            "domain_name": self.cfg.domain, "config": getattr(self.cfg, "source", ""), "bot": self.bot_name,
            "telegram": {"ok": not getattr(self.notifier, "_backlog", None),
                         "waiting": len(getattr(self.notifier, "_backlog", []) or [])},
            "runs": [{"session": s.id, "project": s.project or "", "backend": s.ran_on[0], "account": s.ran_on[1],
                      "origin": s.origin, "started": s.started} for s in self.agent.running_sessions()],
            "accounts": {b: {"active": acc.active(b), "names": acc.names(b),
                             "out": {n: acc.out_until(b, n) for n in acc.names(b) if acc.out_until(b, n)}}
                         for b in BACKENDS},
            "forced": acc.forced, "backend": self.agent.backend, "clis": self._clis,
            "domain": self._domain,
        }

    async def write(self) -> None:
        data = await self.snapshot()
        await asyncio.to_thread(write_json, self.path, data)

    async def loop(self) -> None:
        while True:
            try:
                await self.write()
            except Exception:
                log.exception("heartbeat")
            await asyncio.sleep(EVERY)

    def stopped(self) -> None:
        """At a clean stop: the file says so (the TUI shows "stopped" at once, not after STALE)."""
        try:
            data = json.loads(self.path.read_text())
            data.update({"updated": time.time(), "stopped": time.time(), "runs": []})
            write_json(self.path, data)
        except (OSError, ValueError):
            pass
