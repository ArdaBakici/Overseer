"""Noticing an agent run that is stuck (runs have no time limit: agent.timeout_minutes 0).

A watchdog follows each run: every event of the CLI (a thought, a message, a command, its output) is a sign of life,
and so is a running command's own processes using CPU (a compile, a copy). After `agent.stuck.quiet_minutes` without
any, it acts - and tells you afterwards:

* a command that's quiet and idle (no output, no CPU - e.g. `ssh hpc 'python ...'` in the foreground, a prompt
  waiting for input) is stopped: just that command, so the agent gets it back as failed and carries on, with a note
  saying why;
* a CLI gone quiet with no command running (its connection stalled) is restarted: the request runs again, resuming
  the conversation (once per request);
* the same command with the same result again and again (agent.stuck.loop_repeats): the agent gets a note to try
  something else or ask you. It's never stopped for that. A file the agent edits in between starts the count over,
  and a command that worked without printing anything (an upload, a copy) isn't counted: its "result" shows nothing.

A remote command's own CPU isn't seen (ssh's is ~0 here): a long foreground command on the HPC counts as idle - long
work belongs in sbatch or hpc-run, which returns after a few minutes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from .notify import esc

log = logging.getLogger(__name__)

# The CLI process of the run in progress, told by the runner as it starts one (runners.py: _converse)
PROCESS_STARTED: ContextVar[Callable[[int], None] | None] = ContextVar("PROCESS_STARTED", default=None)

SAMPLE_SECONDS = 60  # how often a run is looked at
BUSY_SHARE = 0.05  # a command's processes using at least this share of a CPU since the last look: it's working
_TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


@dataclass
class _Proc:
    pid: int
    ppid: int
    started: float  # seconds since boot
    cpu: float  # seconds (user + system)


def _processes() -> dict[int, _Proc]:
    out = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
        except OSError:
            continue
        rest = stat.rsplit(")", 1)[-1].split()  # (the name, in parentheses, may hold spaces)
        try:
            out[int(d.name)] = _Proc(int(d.name), int(rest[1]), int(rest[19]) / _TICK,
                                     (int(rest[11]) + int(rest[12])) / _TICK)
        except (IndexError, ValueError):
            continue
    return out


def _descendants(root: int, procs: dict[int, _Proc]) -> list[_Proc]:
    children: dict[int, list[_Proc]] = {}
    for p in procs.values():
        children.setdefault(p.ppid, []).append(p)
    found, todo = [], [root]
    while todo:
        for c in children.get(todo.pop(), []):
            found.append(c)
            todo.append(c.pid)
    return found


def _since_boot() -> float:
    return time.clock_gettime(time.CLOCK_BOOTTIME) if hasattr(time, "CLOCK_BOOTTIME") else time.monotonic()


def _silent_success(ev: dict) -> bool:
    """A command that worked and printed nothing (an upload, a copy, mkdir): the same "result" again says nothing
    about being stuck - what it acted on may have changed (FB13: a script revised between uploads)."""
    out = (ev.get("output") or "").strip().lower()
    return ev.get("exit_code") in (0, None) and (not out or out in ("(no output)", "(bash completed with no output)"))


@dataclass
class Watchdog:
    quiet_minutes: float
    stop_command: bool
    restart_cli: bool
    loop_repeats: int
    running: Callable[[], dict[str, float]]  # the commands running now: their start (time.monotonic())
    note: Callable[[str], Awaitable[bool]]  # a note into the running turn (steering)
    tell: Callable[[str], Awaitable[None]]  # a message for the researcher (afterwards)
    pid: int = 0  # the CLI's process (its own process group)
    restarted: bool = False  # it restarted the CLI: the request runs again
    last_life: float = field(default_factory=time.monotonic)
    _looked: float = field(default_factory=time.monotonic)
    _cpu: dict[str, float] = field(default_factory=dict)  # command -> its processes' CPU at the last look
    _stopped: set[tuple[str, float]] = field(default_factory=set)  # (command, its start)
    _results: dict[str, int] = field(default_factory=dict)
    _task: asyncio.Task | None = None

    # ---- what the run tells it
    def started(self, pid: int) -> None:
        self.pid, self.last_life = pid, time.monotonic()

    def event(self, ev: dict) -> None:
        self.last_life = time.monotonic()
        if ev.get("kind") == "file":  # (it changed something: what follows isn't the same attempt again)
            self._results.clear()
        elif ev.get("kind") == "output" and self.loop_repeats and not _silent_success(ev):
            key = hashlib.sha1(f"{ev.get('command')}\0{ev.get('exit_code')}\0{ev.get('output', '')[-4000:]}"
                               .encode(errors="replace")).hexdigest()
            self._results[key] = n = self._results.get(key, 0) + 1
            if n == self.loop_repeats:
                self._spawn(self._looping(str(ev.get("command") or ""), n))

    # ---- looking
    def start(self) -> None:
        if self.quiet_minutes > 0 or self.loop_repeats:
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    def _spawn(self, coro) -> None:
        asyncio.get_running_loop().create_task(coro)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(SAMPLE_SECONDS)
            try:
                await self.look()
            except Exception:
                log.exception("stuck check failed")

    def _command_procs(self, procs: dict[int, _Proc], since: float) -> list[_Proc]:
        """The processes a command started: the CLI's descendants that began with it or after."""
        began = _since_boot() - (time.monotonic() - since) - 1.0
        return [p for p in _descendants(self.pid, procs) if p.started >= began]

    async def look(self) -> None:
        if not self.pid or self.quiet_minutes <= 0:
            return
        now, quiet = time.monotonic(), self.quiet_minutes * 60
        since_look, self._looked = now - self._looked, now
        running = {c: t for c, t in self.running().items() if (c, t) not in self._stopped}
        procs = _processes() if running else {}
        for cmd, since in running.items():  # (a command that uses CPU is working, output or not)
            mine = self._command_procs(procs, since)
            cpu = sum(p.cpu for p in mine)
            if cmd in self._cpu and cpu - self._cpu[cmd] >= BUSY_SHARE * since_look:
                self.last_life = now
            self._cpu[cmd] = cpu
        if now - self.last_life < quiet:
            return
        mins = int((now - self.last_life) // 60)
        if running:
            if not self.stop_command:
                return
            for cmd, since in running.items():
                self._stopped.add((cmd, since))
                await self._kill([p.pid for p in self._command_procs(procs, since)])
                await self.note(f"Your command `{cmd[:300]}` had shown nothing for {mins} min - no output, and no CPU "
                                "used here - so Overseer stopped it. If it was meant to take long, run it as a job "
                                "(sbatch, then `ov watch`) or in the background and check on it; if it was waiting for "
                                "input or hanging, find out why before running it again.")
                await self.tell(f"⏹ Stopped a command that had been quiet for {mins} min (no output, no CPU here): "
                                f"<code>{esc(cmd[:300])}</code>. The agent was told and carries on.")
            self.last_life = now
        elif self.restart_cli and not self.restarted:
            self.restarted = True
            await self.tell(f"↻ The agent's CLI had gone quiet for {mins} min with no command running: Overseer "
                            "restarted it, and the request runs again from where the conversation was.")
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    async def _kill(self, pids: list[int]) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):  # (a moment to end cleanly, then for good)
            for pid in pids:
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            if sig == signal.SIGTERM:
                end = time.monotonic() + 3
                while time.monotonic() < end and any(Path(f"/proc/{p}").exists() for p in pids):
                    await asyncio.sleep(0.1)

    async def _looping(self, cmd: str, n: int) -> None:
        await self.note(f"You've run `{cmd[:300]}` {n} times now with the same result. If you're stuck, try "
                        "something materially different, or ask the researcher (`ov ask`); if something it depends on "
                        "changed in between, carry on.")
        await self.tell(f"🔁 The agent ran <code>{esc(cmd[:300])}</code> {n}× with the same result: it was asked to "
                        "try something else.")

