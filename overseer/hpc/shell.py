"""/x (or /bash): the researcher's own shell on the HPC, from Telegram.

The command runs as you through Overseer's SSH connection, in an interactive bash (`bash -ic`): it reads ~/.bashrc,
so your aliases, functions, modules and PATH are there as in your terminal. It gets a terminal (ssh -tt), so ⏹ Stop
and the time limit end it on the HPC too (the hang-up reaches it); pagers are off (PAGER=cat), colours are stripped.

Its output is one message that updates itself while it runs (with ⏹ Stop), then the end of the output with the exit
code and how long it took - all of it as a file when it's long. It starts in your home folder, or in a project's
topic, in the project's folder. The agents don't see it: it's yours, not a conversation.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import re
import shlex
import time

from ..core.notify import Outgoing, esc
from .sshlink import base_argv

log = logging.getLogger(__name__)

UPDATE_SECONDS = 3  # the message's refresh while it runs
SHOWN = 3500  # characters of output in the message (Telegram's limit is 4096); more: the whole of it as a file
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][A-Za-z0-9]|\x1b[=>]")
NOISE = re.compile(r"^bash: (cannot set terminal process group|no job control in this shell).*$\n?", re.M)


def clean(raw: bytes) -> str:
    """What a terminal would show, as text: no colours or cursor codes, \\n line ends, overwritten lines (\\r,
    progress bars) reduced to what stays."""
    text = ANSI.sub("", raw.decode(errors="replace")).replace("\r\n", "\n")
    text = "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))
    return NOISE.sub("", text)


class Shell:
    _ids = itertools.count(1)

    def __init__(self, cfg, notifier):
        self.cfg = cfg
        self.notifier = notifier
        self.running: dict[int, asyncio.subprocess.Process] = {}
        self.stopped: set[int] = set()

    def argv(self, remote: str) -> list[str]:
        """The local command: ssh through Overseer's connection, with a terminal (the hang-up ends the command)."""
        return [*base_argv(self.cfg), "-tt", self.cfg.ssh.host, "--", remote]

    def remote(self, command: str, folder: str = "") -> str:
        setup = self.cfg.slurm.setup.strip()
        prefix = "export PAGER=cat GIT_PAGER=cat SYSTEMD_PAGER= MANPAGER=cat; " + (f"{setup}; " if setup else "")
        if folder:
            prefix += f"cd {shlex.quote(folder)} || exit 1; "
        return "bash -ic " + shlex.quote(prefix + command)

    def stop(self, rid: int) -> bool:
        proc = self.running.get(rid)
        if not proc:
            return False
        self.stopped.add(rid)
        proc.kill()  # (ssh: its terminal on the HPC hangs up, and the command gets SIGHUP)
        return True

    async def run(self, command: str, folder: str = "", project: str | None = None) -> int:
        """Run it, showing its output as it comes; its exit code (-1: stopped or timed out)."""
        rid = next(self._ids)
        limit = self.cfg.ssh.shell_timeout_minutes * 60
        head = f"<b>$</b> <code>{esc(command[:300])}</code>" + (f"  <i>in {esc(folder)}</i>" if folder else "")
        stop = [[("⏹ Stop", f"xstop:{rid}")]]
        refs = await self.notifier.post(Outgoing(f"{head}\n<i>running…</i>", buttons=stop, project=project))
        started = time.time()
        proc = await asyncio.create_subprocess_exec(*self.argv(self.remote(command, folder)),
                                                    stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        self.running[rid] = proc
        out = bytearray()
        shown = ""

        async def read():
            while chunk := await proc.stdout.read(65536):
                out.extend(chunk)

        reader = asyncio.create_task(read())
        timed_out = False
        try:
            while not reader.done():
                await asyncio.wait({reader}, timeout=UPDATE_SECONDS)
                if time.time() - started > limit:
                    timed_out = True
                    proc.kill()
                    break
                text = clean(bytes(out))[-SHOWN:]
                if not reader.done() and text != shown:
                    shown = text
                    await self.notifier.edit(refs, f"{head}\n<pre>{esc(text)}</pre>\n<i>running for "
                                                   f"{int(time.time() - started)}s…</i>", stop)
            await reader
            rc = await proc.wait()
        finally:
            self.running.pop(rid, None)
        took = int(time.time() - started)
        text = clean(bytes(out))
        if rid in self.stopped or timed_out:
            self.stopped.discard(rid)
            rc, end = -1, (f"⏹ stopped after {took}s" if not timed_out else
                           f"⏱ stopped: it ran past {self.cfg.ssh.shell_timeout_minutes} min (longer work: sbatch)")
        elif rc == 255 and not text.strip():
            end = "✗ couldn't reach the HPC (the SSH connection is down: it reopens on the next poll)"
        else:
            end = f"{'✓' if rc == 0 else '✗'} exit code {rc} · {took}s"
        body = f"<pre>{esc(text[-SHOWN:])}</pre>\n" if text.strip() else ""
        cut = len(text) > SHOWN
        await self.notifier.edit(refs, f"{head}\n{body}<i>{esc(end)}" + (" · the whole output is attached"
                                                                             if cut else "") + "</i>")
        if cut:
            await self.notifier.send(Outgoing(f"<code>{esc(command[:200])}</code>: the whole output",
                                              files=[("output.txt", text.encode())], project=project, silent=True))
        self._audit(command, folder, rc, took)
        return rc

    def _audit(self, command: str, folder: str, rc: int, took: int) -> None:
        """In the audit log with Overseer's other commands on the HPC: what you ran from Telegram."""
        path = self.cfg.local.audit_log
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as fh:
                fh.write(json.dumps({"ts": time.time(), "argv": ["/x", command], "cwd": folder, "rc": rc,
                                     "sec": took, "note": "researcher's shell"}) + "\n")
        except OSError:
            log.exception("could not write audit log")
