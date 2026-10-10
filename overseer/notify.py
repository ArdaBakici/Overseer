"""Notifier interface and Telegram-HTML formatting helpers."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Protocol

TG_LIMIT = 4096


@dataclass
class Outgoing:
    text: str  # Telegram HTML
    job_id: str | None = None
    session_id: str | None = None
    buttons: list[list[tuple[str, str]]] = field(default_factory=list)  # rows of (label, callback_data)
    files: list[tuple[str, bytes]] = field(default_factory=list)  # (filename, content)
    silent: bool = False


class Notifier(Protocol):
    async def send(self, msg: Outgoing) -> None: ...

    # A message that is edited in place (the agent's live progress). The handle is opaque.
    async def post_live(self, msg: Outgoing) -> object: ...

    async def edit_live(self, handle: object, text: str) -> None: ...

    async def delete_live(self, handle: object) -> None: ...


def esc(s: object) -> str:
    return html.escape(str(s), quote=False)


def pre(text: str, limit: int = 2500, keep: str = "tail") -> str:
    text = text.rstrip()
    if len(text) > limit:
        text = ("…\n" + text[-limit:]) if keep == "tail" else (text[:limit] + "\n…")
    return f"<pre>{esc(text)}</pre>" if text else ""


def code(text: object) -> str:
    return f"<code>{esc(text)}</code>"


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Split on line boundaries without breaking <pre> blocks (re-opened in each chunk)."""
    if len(text) <= limit:
        return [text]
    chunks, cur, in_pre = [], "", False
    for line in text.split("\n"):
        while len(line) > limit - 20:
            head, line = line[: limit - 40], line[limit - 40:]
            chunks.append(cur + head + ("</pre>" if in_pre else ""))
            cur = "<pre>" if in_pre else ""
        if len(cur) + len(line) + 1 > limit - 10:
            chunks.append(cur + ("</pre>" if in_pre else ""))
            cur = "<pre>" if in_pre else ""
        cur += line + "\n"
        in_pre = (in_pre or "<pre>" in line) and "</pre>" not in line.split("<pre>")[-1]
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip() and c.strip() not in ("<pre>", "<pre></pre>")]
