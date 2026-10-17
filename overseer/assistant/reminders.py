"""Reminders: a message to the user at a set time, without waking the agent (`ov remind`, /remind)."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime

from ..core.notify import Outgoing, esc

log = logging.getLogger(__name__)

KEY = "assistant:reminders"
SENT_KEY = "assistant:reminders_sent"  # the last ones sent, for ⏰ Snooze
MAX_PENDING = 200


def when_text(ts: float) -> str:
    d = datetime.fromtimestamp(ts)
    today = datetime.now().date()
    day = ("today" if d.date() == today else "tomorrow" if (d.date() - today).days == 1
           else d.strftime("%a %d %b") if (d.date() - today).days < 7 else d.strftime("%d %b %Y"))
    return f"{day} {d:%H:%M}"


class Reminders:
    def __init__(self, store, notifier):
        self.store = store
        self.notifier = notifier
        self._wake = asyncio.Event()

    def _load(self, key: str = KEY) -> list[dict]:
        try:
            items = json.loads(self.store.get(key) or "[]")
        except ValueError:
            return []
        return [r for r in items if isinstance(r, dict) and "at" in r and "id" in r]

    def _save(self, items: list[dict], key: str = KEY) -> None:
        self.store.set(key, json.dumps(items))

    def pending(self) -> list[dict]:
        return sorted(self._load(), key=lambda r: r["at"])

    def add(self, at: float, text: str, session_id: str | None = None, project: str | None = None) -> dict:
        text = " ".join(text.split())[:1000]
        if not text:
            raise ValueError("what should the reminder say?")
        items = self._load()
        if len(items) >= MAX_PENDING:
            raise ValueError(f"{MAX_PENDING} reminders are pending already: remove some first")
        r = {"id": "r" + secrets.token_hex(3), "at": float(at), "text": text, "session": session_id,
             "project": project, "made": time.time()}
        self._save(items + [r])
        self._wake.set()
        return r

    def remove(self, rid: str) -> dict | None:
        items = self._load()
        hit = next((r for r in items if r["id"] == rid), None)
        if hit:
            self._save([r for r in items if r["id"] != rid])
        return hit

    def snooze(self, rid: str, seconds: float) -> dict | None:
        """A reminder that was sent, again later."""
        sent = self._load(SENT_KEY)
        hit = next((r for r in sent if r["id"] == rid), None)
        if not hit:
            return None
        self._save([r for r in sent if r["id"] != rid], SENT_KEY)
        return self.add(time.time() + seconds, hit["text"], hit.get("session"), hit.get("project"))

    async def fire_due(self, now: float | None = None) -> int:
        now = now or time.time()
        items = self._load()
        due = [r for r in items if r["at"] <= now]
        if not due:
            return 0
        self._save([r for r in items if r["at"] > now])
        self._save((self._load(SENT_KEY) + due)[-50:], SENT_KEY)
        for r in due:
            await self.notifier.send(Outgoing(
                f"⏰ {esc(r['text'])}", session_id=r.get("session"), project=r.get("project"),
                buttons=[[("💤 1h", f"rsnz:{r['id']}:3600"), ("💤 Tomorrow", f"rsnz:{r['id']}:86400")]]))
        return len(due)

    async def loop(self) -> None:
        while True:
            try:
                await self.fire_due()
            except Exception:
                log.exception("reminders")
            nxt = min((r["at"] for r in self._load()), default=time.time() + 3600)
            self._wake.clear()
            try:  # (until the next one is due, a new one is added, or a minute at most: the clock may jump)
                await asyncio.wait_for(self._wake.wait(), max(1.0, min(60.0, nxt - time.time())))
            except asyncio.TimeoutError:
                pass
