"""Watching the agent while it works.

* Transcript: every step of every agent run (reasoning summaries, messages, commands and their
  output) is appended to state/transcripts/<conversation>.jsonl. `overseer watch` follows these.
* Progress: one Telegram message per request that updates itself every few seconds with the
  latest reasoning, the running command and the elapsed time; deleted when the answer arrives.

Both CLIs only expose summaries of the model's reasoning (OpenAI and Anthropic don't return the
raw chain of thought), plus whatever the agent writes between steps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator

from .config import Config
from .notify import Outgoing, TG_LIMIT, esc, pre

log = logging.getLogger(__name__)

PROGRESS_SECONDS = 5  # Telegram progress message refresh
MAX_OUTPUT = 20_000  # chars of each command's output kept in the transcript (the end of it)


# ============================================================================ transcript

def transcript_dir(cfg: Config) -> Path:
    return cfg.local.state("transcripts")


class Transcript:
    """Append-only JSON-lines log of one conversation."""

    def __init__(self, cfg: Config, conversation: str):
        d = transcript_dir(cfg)
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = d / f"{conversation}.jsonl"

    def write(self, event: dict[str, Any]) -> None:
        ev = {"ts": time.time(), **event}
        out = ev.get("output")
        if isinstance(out, str) and len(out) > MAX_OUTPUT:
            ev["output"] = out[-MAX_OUTPUT:]
            ev["truncated"] = len(out) - MAX_OUTPUT
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as fh:
                fh.write(json.dumps(ev) + "\n")
        except OSError as e:  # the live log is a convenience; never fail a run over it
            log.warning("transcript %s: %s", self.path, e)


# ============================================================================ Telegram progress

def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


class Progress:
    """The self-updating "⏳ working" message for one request."""

    def __init__(self, notifier, backend: str, session_id: str | None, job_id: str | None,
                 interval: float | None = None):
        self.notifier = notifier
        self.backend = backend
        self.session_id, self.job_id = session_id, job_id
        self.interval = interval or PROGRESS_SECONDS
        self.t0 = time.monotonic()
        self.steps: deque[tuple[str, str]] = deque(maxlen=3)  # (icon, text): latest thoughts/messages/tools
        self.commands = 0
        self.running: tuple[str, float] | None = None  # (command, started)
        self.last: tuple[str, int | None] | None = None  # (command, exit code) of the last finished one
        self.handle: Any = None
        self.shown = ""
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    def on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("kind")
        if kind == "thought":
            self.steps.append(("💭", ev["text"]))
        elif kind == "message":
            self.steps.append(("💬", ev["text"]))
        elif kind == "tool":
            self.steps.append(("🔧", f"{ev.get('name')} {ev.get('detail') or ''}".strip()))
        elif kind == "file":
            self.steps.append(("✏️", ev.get("path", "")))
        elif kind == "command":
            self.commands += 1
            self.running = (ev["command"], time.monotonic())
        elif kind == "output":
            self.running = None
            self.last = (ev.get("command", ""), ev.get("exit_code"))

    def render(self) -> str:
        now = time.monotonic()
        n = self.commands
        head = (f"⏳ <b>{esc(self.backend)} is working</b> · {_duration(now - self.t0)}"
                f" · {n} command{'s' if n != 1 else ''}")
        lines = [head]
        for icon, text in self.steps:
            body = esc(_clip(text, 600))
            lines.append(f"{icon} <i>{body}</i>" if icon == "💭" else f"{icon} {body}")
        if self.running:
            cmd, started = self.running
            lines.append(f"▶️ running for {_duration(now - started)}:\n{pre(_clip(cmd, 400), 400)}")
        elif self.last:
            cmd, rc = self.last
            status = "✔️ last command" if rc in (0, None) else f"✖️ last command exited {rc}"
            lines.append(f"{status}:\n{pre(_clip(cmd, 300), 300)}")
        lines.append("<i>/stop cancels</i>")
        text = "\n".join(lines)
        return text if len(text) <= TG_LIMIT else head + "\n…"

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval)
                return
            except asyncio.TimeoutError:
                pass
            await self._flush()

    async def _flush(self) -> None:
        text = self.render()
        if text == self.shown:
            return
        try:
            if self.handle is None:
                self.handle = await self.notifier.post_live(
                    Outgoing(text, session_id=self.session_id, job_id=self.job_id, silent=True))
            else:
                await self.notifier.edit_live(self.handle, text)
            self.shown = text
        except Exception:
            log.debug("progress message update failed", exc_info=True)

    async def close(self) -> None:
        """Stop updating and remove the message (the answer follows)."""
        self._stop.set()
        if self._task:
            try:
                await self._task
            except Exception:
                pass
        if self.handle is not None:
            try:
                await self.notifier.delete_live(self.handle)
            except Exception:
                log.debug("progress message delete failed", exc_info=True)


# ============================================================================ `overseer watch`

class Style:
    def __init__(self, color: bool):
        self.color = color

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text


def render_event(ev: dict[str, Any], st: Style, *, full: bool = False, output_lines: int = 12,
                 last_message: str | None = None) -> list[str]:
    """Terminal lines for one transcript event. `last_message`: the agent's previous message, which
    is usually its final answer already (not repeated at the end)."""
    kind = ev.get("kind")
    clock = time.strftime("%H:%M:%S", time.localtime(ev.get("ts") or 0))
    if kind == "turn":
        meta = " · ".join(x for x in [ev.get("project") and f"📁 {ev['project']}", ev.get("backend"), ev.get("model"),
                                      ev.get("effort") and f"effort {ev['effort']}"] if x)
        return ["", st(f"── {clock} · new request · {meta} " + "─" * 20, "1;36"), st("👤 ", "1") + ev.get("text", "")]
    if kind == "thought":
        return [st("💭 " + _indent(ev.get("text", ""), 3), "2;3")]
    if kind == "message":
        return ["💬 " + _indent(ev.get("text", ""), 3)]
    if kind == "command":
        return [st("$ " + _indent(ev.get("command", ""), 2), "1;33")]
    if kind == "output":
        out = (ev.get("output") or "").rstrip("\n")
        lines = out.split("\n") if out else []
        shown = lines if full else lines[-output_lines:]
        res = []
        if len(shown) < len(lines) or ev.get("truncated"):
            res.append(st(f"  │ … {len(lines) - len(shown)} earlier lines (--full shows all)", "2"))
        res += [st("  │ " + line, "2") for line in shown]
        rc = ev.get("exit_code")
        if rc not in (None, 0):
            res.append(st(f"  └ exit {rc}", "31"))
        return res
    if kind == "tool":
        return [st(f"🔧 {ev.get('name')} {ev.get('detail') or ''}".rstrip(), "35")]
    if kind == "file":
        return [st(f"✏️  {ev.get('path')}", "35")]
    if kind == "end":
        status = ev.get("status")
        out = []
        if status == "done":
            if ev.get("text") and ev["text"].strip() != (last_message or "").strip():
                out.append(st("🤖 ", "1") + _indent(ev["text"], 3))
            if ev.get("footer"):
                out.append(st("   " + ev["footer"], "2"))
            out.append(st(f"── {clock} · done", "36"))
        elif status == "stopped":
            out.append(st(f"── {clock} · ⏹ stopped", "33"))
        else:
            out.append(st(f"── {clock} · ⚠️ {ev.get('text') or 'failed'}", "31"))
        return out
    return []


def _indent(text: str, n: int) -> str:
    return text.strip().replace("\n", "\n" + " " * n)


def read_events(path: Path, start: int = 0) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """(start, end offset of the line, event) for each complete line from `start`."""
    with path.open("rb") as fh:
        fh.seek(start)
        offset = start
        for raw in fh:
            if not raw.endswith(b"\n"):
                break  # still being written
            end = offset + len(raw)
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                ev = None
            if isinstance(ev, dict):
                yield offset, end, ev
            offset = end


def running_turn_offset(path: Path) -> int | None:
    """Offset of the last request in this transcript if it hasn't finished yet."""
    start = None
    for offset, _, ev in read_events(path):
        if ev.get("kind") == "turn":
            start = offset
        elif ev.get("kind") == "end":
            start = None
    return start


def watch(cfg: Config, conversation: str | None, *, full: bool = False, follow: bool = True,
          color: bool | None = None, out=None, poll: float = 0.5) -> int:
    """Print a conversation's transcript and follow it, or (no conversation) follow all of them."""
    out = out or sys.stdout
    st = Style(out.isatty() and not os.environ.get("NO_COLOR") if color is None else color)
    d = transcript_dir(cfg)
    offsets: dict[Path, int] = {}
    if conversation:
        matches = sorted(d.glob(f"{conversation}*.jsonl")) if d.exists() else []
        exact = d / f"{conversation}.jsonl"
        if exact in matches:
            matches = [exact]
        if len(matches) > 1:
            print(f"{conversation!r} matches several conversations: "
                  f"{', '.join(m.stem for m in matches)}", file=sys.stderr)
            return 1
        if not matches:
            print(f"No live log for {conversation!r} yet: it starts with the next request in that conversation "
                  "(conversations from before this feature have none; `overseer open` shows their history).",
                  file=sys.stderr)
            if not follow:
                return 1
        targets = matches or [exact]
        offsets = {p: 0 for p in targets}
        print(st(f"Conversation {targets[0].stem} (Ctrl-C to stop)", "1"), file=out)
    else:
        files = sorted(d.glob("*.jsonl"), key=lambda f: f.stat().st_mtime) if d.exists() else []
        running = []
        for f in files:
            off = running_turn_offset(f)
            offsets[f] = off if off is not None else f.stat().st_size
            if off is not None:
                running.append(f.stem)
        now = ", ".join(running) if running else "nothing right now"
        print(st(f"Watching all agent conversations (Ctrl-C to stop). Working on: {now}", "1"), file=out)
    out.flush()
    current = None
    last_message: dict[Path, str | None] = {}
    try:
        while True:
            if not conversation and d.exists():
                for f in d.glob("*.jsonl"):
                    offsets.setdefault(f, 0)  # a conversation that started after we did
            for path in sorted(offsets, key=lambda p: p.stat().st_mtime if p.exists() else 0):
                if not path.exists() or path.stat().st_size <= offsets[path]:
                    continue
                for _, end, ev in read_events(path, offsets[path]):
                    if not conversation and current != path.stem:
                        current = path.stem
                        print(st(f"\n▌ {path.stem}", "1;34"), file=out)
                    for line in render_event(ev, st, full=full, last_message=last_message.get(path)):
                        print(line, file=out)
                    if ev.get("kind") in ("message", "turn"):
                        last_message[path] = ev.get("text") if ev["kind"] == "message" else None
                    offsets[path] = end
                out.flush()
            if not follow:
                return 0
            time.sleep(poll)
    except KeyboardInterrupt:
        return 0

