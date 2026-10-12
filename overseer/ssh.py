"""SSH transport to the HPC login node.

Uses the system ``ssh`` with Overseer's own config (see sshlink.py), multiplexed over one
master connection so polling does not open a new SSH session every time.

Commands are always passed as argv lists and quoted with ``shlex.quote``. Every command is
appended to the audit log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from dataclasses import dataclass

from .config import Config
from .sshlink import base_argv

log = logging.getLogger(__name__)


class SSHError(RuntimeError):
    pass


@dataclass
class Result:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class SSHRunner:
    """Runs Overseer's own (monitoring) commands on the HPC over the shared master connection."""

    def __init__(self, cfg: Config):
        self.full_cfg = cfg
        self.cfg = cfg.ssh
        self.audit_path = cfg.local.audit_log
        self._sem = asyncio.Semaphore(cfg.ssh.max_concurrent)

    def build_command(self, argv: list[str]) -> list[str]:
        return [*base_argv(self.full_cfg), self.cfg.host, "--", self.remote_command(argv)]

    @staticmethod
    def remote_command(argv: list[str]) -> str:
        return " ".join(shlex.quote(a) for a in argv)

    async def run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout: float | None = None,
        max_output: int = 2_000_000,
        check: bool = False,
    ) -> Result:
        cmd = self.build_command(argv)
        timeout = timeout or self.cfg.command_timeout
        started = time.time()
        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err, truncated = await asyncio.wait_for(
                    self._communicate(proc, stdin, max_output), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                self._audit(argv, -1, time.time() - started, note="timeout")
                raise SSHError(f"timed out after {timeout}s: {argv[0]}")
        rc = proc.returncode if proc.returncode is not None else -1
        if truncated:
            rc = 0 if rc in (-9, 255, 141) else rc  # we killed it on purpose
        self._audit(argv, rc, time.time() - started, note="truncated" if truncated else "")
        res = Result(argv, rc, out.decode(errors="replace"), err.decode(errors="replace"), truncated)
        if rc == 255 and not truncated:
            raise SSHError(f"ssh failed: {res.stderr.strip()[:500]}")
        if check and not res.ok:
            raise SSHError(f"{argv[0]} exited {rc}: {res.stderr.strip()[:500]}")
        return res

    @staticmethod
    async def _communicate(proc, stdin: bytes | None, max_output: int):
        if stdin is not None:
            proc.stdin.write(stdin)
            await proc.stdin.drain()
            proc.stdin.close()
        out = bytearray()
        truncated = False
        err_task = asyncio.create_task(proc.stderr.read())
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            out.extend(chunk)
            if len(out) > max_output:
                del out[max_output:]
                truncated = True
                proc.kill()
                break
        if truncated:  # a grandchild may still hold stderr open: don't wait for it
            try:
                err = await asyncio.wait_for(err_task, 5)
            except asyncio.TimeoutError:
                err_task.cancel()
                err = b""
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                pass
            return bytes(out), err, truncated
        err = await err_task
        await proc.wait()
        return bytes(out), err, truncated

    def _audit(self, argv: list[str], rc: int, dur: float, note: str = "") -> None:
        log.debug("remote %s rc=%s %.2fs %s", self.remote_command(argv)[:200], rc, dur, note)
        if not self.audit_path:
            return
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_path, "a") as fh:
                fh.write(json.dumps({"ts": time.time(), "argv": argv, "rc": rc, "sec": round(dur, 2), "note": note}) + "\n")
        except OSError:
            log.exception("could not write audit log")
