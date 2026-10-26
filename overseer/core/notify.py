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
    photos: list[tuple[str, bytes]] = field(default_factory=list)  # images, sent as pictures (an album if several)
    # GIFs and videos, sent after the message as playable ones: (animation | video, filename, content)
    media: list[tuple[str, str, bytes]] = field(default_factory=list)
    silent: bool = False
    project: str | None = None  # the project topic to post in; by default the conversation's or job's
    # Telegram Rich Markdown version of the message, used when rich messages are available
    # (`text` is the fallback). Pictures and files are embedded in it.
    markdown: str | None = None
    jobs: bool = False  # job news: goes to the jobs topic of the job's project (or the general jobs topic)
    reply_to: dict[str, int] = field(default_factory=dict)  # chat id -> message to reply to (e.g. the job's card)
    # "feedback": the 🛠 feedback topic (agents' feedback about Overseer); a side topic's name (SIDE_TOPICS, e.g.
    # "web": what the agent fetched from the internet) - made when first needed
    topic: str | None = None


SILENT = "[silent]"  # a reply to a wake-up that is exactly this isn't sent: nothing worth saying this time

# Side topics: small notices that shouldn't land in the conversation. name -> (title, intro)
SIDE_TOPICS = {
    "web": ("🌐 web", "🌐 <b>web</b> - what the agent downloaded from the internet (ov fetch, ov gif): one short line "
                     "each, so you can see it. Writing here talks to the main conversation."),
}


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
    """Split Telegram HTML into messages, keeping every tag balanced in each one."""
    from .richtext import split_html
    return split_html(text, limit)


def human_seconds(sec: int) -> str:
    """'45m', '3h05m', '2d4h'."""
    if sec < 3600:
        return f"{sec // 60}m"
    if sec < 86400:
        return f"{sec // 3600}h{sec % 3600 // 60:02d}m"
    return f"{sec // 86400}d{sec % 86400 // 3600}h"

