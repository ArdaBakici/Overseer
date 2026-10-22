"""Follow-ups: messages to the user at a set time that come back until answered, without waking the agent.

Four kinds, all with buttons (a tap, never a command):
- reminder: `ov remind` ("remind me at 5 to call the bank")                      ✅ Done · 💤 Later · ⏭ Skip
- say:      a `say:` routine from Routines.md ("Meds 💊")                         ✅ Done · 💤 Later · ⏭ Skip
- commit:   a check-back on something you said you'd do (`ov commit`)             ✅ Done · 💤 Later · 🧱 Stuck
- focus:    a focus session (`ov focus 25m ...`): a quiet countdown, then a ping  ✅ Done · ➕ 5 min · 🧱 Stuck

Unanswered, it comes back by its follow-up policy (Routines.md's # Follow-ups: "again after 20m, up to 2 more
times"), then counts as unanswered. Every outcome goes into the day's log (and today's daily note, if it exists),
which the assistant sees, e.g. in the evening shutdown. 🧱 Stuck wakes the assistant to help.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime

from ..core.notify import Outgoing, esc

log = logging.getLogger(__name__)

KEY = "assistant:followups"
LOG_KEY = "assistant:log:"  # + YYYY-MM-DD: the day's outcomes
MAX_ITEMS = 300
GIVE_UP = 3600  # after the last follow-up, how long until it counts as unanswered (at least)
LATER = 30 * 60  # 💤 Later, when its policy has no follow-ups
FOCUS_MORE = 5 * 60
DEFAULT_POLICY = {"after": 30 * 60, "times": 1}
ACTIVE = ("pending", "open")
ICONS = {"done": "✅", "skipped": "⏭", "unanswered": "🤷", "stuck": "🧱", "cancelled": "✖", "later": "💤",
         "started": "🎯"}


def when_text(ts: float) -> str:
    d = datetime.fromtimestamp(ts)
    today = datetime.now().date()
    day = ("today" if d.date() == today else "tomorrow" if (d.date() - today).days == 1
           else d.strftime("%a %d %b") if (d.date() - today).days < 7 else d.strftime("%d %b %Y"))
    return f"{day} {d:%H:%M}"


def _minutes(seconds: float) -> int:
    return max(0, int((seconds + 59) // 60))


class Reminders:
    def __init__(self, store, notifier):
        self.store = store
        self.notifier = notifier
        self._wake = asyncio.Event()
        self.on_log = None  # (entry) -> awaitable: e.g. into today's daily note
        self.on_stuck = None  # (item) -> awaitable: wake the assistant to help

    # ------------------------------------------------------------------ storage
    def _load(self) -> list[dict]:
        try:
            items = json.loads(self.store.get(KEY) or "[]")
        except ValueError:
            return []
        return [r for r in items if isinstance(r, dict) and "at" in r and "id" in r]

    def _save(self, items: list[dict]) -> None:
        # Finished ones are kept for a day (their buttons may still be tapped), then dropped.
        cutoff = time.time() - 86400
        items = [r for r in items if r.get("state", "pending") in ACTIVE or r.get("closed", r["at"]) > cutoff]
        self.store.set(KEY, json.dumps(items[-MAX_ITEMS:]))

    def _update(self, item: dict) -> None:
        self._save([item if r["id"] == item["id"] else r for r in self._load()])

    def get(self, rid: str) -> dict | None:
        return next((r for r in self._load() if r["id"] == rid), None)

    def pending(self) -> list[dict]:
        """Scheduled, not sent yet (reminders, check-backs, focus sessions running)."""
        return sorted((r for r in self._load() if r.get("state", "pending") == "pending"), key=lambda r: r["at"])

    def open(self) -> list[dict]:
        """Sent and not answered yet."""
        return sorted((r for r in self._load() if r.get("state") == "open"), key=lambda r: r["at"])

    def focus(self) -> dict | None:
        return next((r for r in self.pending() if r.get("kind") == "focus"), None)

    # ------------------------------------------------------------------ adding
    def add(self, at: float, text: str, session_id: str | None = None, project: str | None = None, *,
            kind: str = "reminder", policy: dict | None = None, routine: str = "") -> dict:
        text = " ".join(text.split())[:1000]
        if not text:
            raise ValueError("what should it say?")
        items = self._load()
        if sum(1 for r in items if r.get("state", "pending") in ACTIVE) >= MAX_ITEMS // 2:
            raise ValueError(f"{MAX_ITEMS // 2} follow-ups are pending already: remove some first")
        r = {"id": "r" + secrets.token_hex(3), "kind": kind, "at": float(at), "text": text, "session": session_id,
             "project": project, "made": time.time(), "state": "pending", "attempts": 0,
             "policy": policy or DEFAULT_POLICY, "routine": routine}
        self._save(items + [r])
        self._wake.set()
        return r

    async def start_focus(self, minutes: float, text: str, session_id: str | None = None,
                          project: str | None = None, policy: dict | None = None) -> dict:
        """A focus session: one quiet message counting down (edited in place), then a ping."""
        if self.focus():
            raise ValueError("a focus session is running already (ov focus stop ends it)")
        if not 1 <= minutes <= 240:
            raise ValueError("a focus session is 1 to 240 minutes")
        r = self.add(time.time() + minutes * 60, text, session_id, project, kind="focus", policy=policy)
        r["minutes"] = minutes
        if hasattr(self.notifier, "post"):
            r["live"] = await self.notifier.post(Outgoing(self._focus_text(r), session_id=session_id, project=project,
                                                          silent=True, buttons=self._focus_buttons(r)))
            r["edited"] = time.time()
        self._update(r)
        await self._log(r, "started", f"focus {int(minutes)} min: {r['text']}")
        return r

    # ------------------------------------------------------------------ the messages
    def _buttons(self, r: dict) -> list[list[tuple[str, str]]]:
        i = r["id"]
        if r.get("kind") == "focus":
            return [[("✅ Done", f"fu:{i}:done"), ("➕ 5 min", f"fu:{i}:more"), ("🧱 Stuck", f"fu:{i}:stuck")]]
        third = ("🧱 Stuck", f"fu:{i}:stuck") if r.get("kind") == "commit" else ("⏭ Skip", f"fu:{i}:skip")
        return [[("✅ Done", f"fu:{i}:done"), ("💤 Later", f"fu:{i}:later"), third]]

    def _text(self, r: dict) -> str:
        again = "🔁 " if r.get("attempts") else ""
        if r.get("kind") == "commit":
            return f"{again}🎯 <b>Check-in</b>: {esc(r['text'])} - how did it go?"
        if r.get("kind") == "focus":
            return f"{again}⏱ <b>Focus time's up</b>: {esc(r['text'])}. How did it go?"
        return f"{again}⏰ {esc(r['text'])}"

    def _focus_text(self, r: dict) -> str:
        left = _minutes(r["at"] - time.time())
        return f"🎯 <b>Focus</b>: {esc(r['text'])}\n<i>{left} min left · until {datetime.fromtimestamp(r['at']):%H:%M}</i>"

    def _focus_buttons(self, r: dict) -> list[list[tuple[str, str]]]:
        return [[("✅ Done early", f"fu:{r['id']}:done"), ("⏹ Stop", f"fu:{r['id']}:cancel")]]

    async def _send(self, r: dict) -> None:
        await self.notifier.send(Outgoing(self._text(r), session_id=r.get("session"), project=r.get("project"),
                                          buttons=self._buttons(r)))

    # ------------------------------------------------------------------ what you do with it
    async def act(self, rid: str, action: str) -> dict | None:
        """done | later | skip | stuck | more (focus: 5 more minutes) | cancel. None: no such follow-up, or it's
        over already."""
        r = self.get(rid)
        if not r or r.get("state", "pending") not in ACTIVE:
            return None
        now = time.time()
        if action == "later":
            p = r.get("policy") or DEFAULT_POLICY
            r.update(state="pending", at=now + (p.get("after") or LATER) if p.get("times") else now + LATER,
                     attempts=0, next=None)
        elif action == "more":
            r.update(state="pending", at=now + FOCUS_MORE, attempts=0, next=None)
            if r.get("kind") == "focus" and hasattr(self.notifier, "post"):
                r["live"] = await self.notifier.post(Outgoing(self._focus_text(r), session_id=r.get("session"),
                                                              project=r.get("project"), silent=True,
                                                              buttons=self._focus_buttons(r)))
                r["edited"] = now
        elif action in ("done", "skip", "stuck", "cancel"):
            r.update(state={"skip": "skipped", "cancel": "cancelled"}.get(action, action), closed=now)
        else:
            raise ValueError(f"unknown action {action!r}")
        self._update(r)
        if r.get("kind") == "focus" and r.get("live") and action in ("done", "cancel", "stuck"):
            await self._end_live(r, {"done": "✅ done", "cancel": "⏹ stopped", "stuck": "🧱 stuck"}[action])
        if action in ("done", "skip", "stuck", "cancel", "later"):
            state = {"skip": "skipped", "cancel": "cancelled"}.get(action, action)
            await self._log(r, state)
        if action == "stuck" and self.on_stuck:
            await self.on_stuck(r)
        self._wake.set()
        return r

    def remove(self, rid: str) -> dict | None:
        """Cancel one that hasn't fired (or is waiting for an answer), without a log entry."""
        items = self._load()
        hit = next((r for r in items if r["id"] == rid and r.get("state", "pending") in ACTIVE), None)
        if hit:
            hit.update(state="cancelled", closed=time.time())
            self._save(items)
        return hit

    async def _end_live(self, r: dict, how: str) -> None:
        if hasattr(self.notifier, "edit"):
            await self.notifier.edit(r["live"], f"🎯 <b>Focus</b>: {esc(r['text'])}\n<i>{how}</i>")

    # ------------------------------------------------------------------ the day's log
    def log(self, day: str | None = None) -> list[dict]:
        try:
            return json.loads(self.store.get(LOG_KEY + (day or datetime.now().date().isoformat())) or "[]")
        except ValueError:
            return []

    async def _log(self, r: dict, state: str, text: str | None = None) -> None:
        entry = {"at": time.time(), "state": state, "icon": ICONS.get(state, "•"), "kind": r.get("kind"),
                 "text": text or r["text"], "id": r["id"]}
        day = datetime.now().date().isoformat()
        self.store.set(LOG_KEY + day, json.dumps((self.log(day) + [entry])[-200:]))
        if self.on_log:
            try:
                await self.on_log(entry)
            except Exception:
                log.warning("could not write the log entry to the daily note", exc_info=True)

    # ------------------------------------------------------------------ the clock
    async def fire_due(self, now: float | None = None) -> int:
        """Send what's due, follow up what's unanswered, count down focus sessions. How many messages it sent."""
        now = now or time.time()
        sent = 0
        for r in self._load():
            state = r.get("state", "pending")
            p = r.get("policy") or DEFAULT_POLICY
            if state == "pending" and r.get("kind") == "focus" and r["at"] > now and r.get("live"):
                if now - r.get("edited", 0) >= 60 and hasattr(self.notifier, "edit"):  # the countdown, quietly
                    r["edited"] = now
                    self._update(r)
                    await self.notifier.edit(r["live"], self._focus_text(r), self._focus_buttons(r))
                continue
            if state == "pending" and r["at"] <= now:
                if r.get("kind") == "focus" and r.get("live"):
                    await self._end_live(r, "⏱ time's up")
                r.update(state="open", attempts=0)
            elif state == "open" and r.get("next") and r["next"] <= now:
                if r["attempts"] >= p.get("times", 0):
                    r.update(state="unanswered", closed=now)
                    self._update(r)
                    await self._log(r, "unanswered")
                    continue
                r["attempts"] += 1
            else:
                continue
            last = r["attempts"] >= p.get("times", 0)
            r["next"] = now + (max(p.get("after") or 0, GIVE_UP) if last else p["after"])
            self._update(r)
            await self._send(r)
            sent += 1
        return sent

    def _next_due(self) -> float:
        now = time.time()
        times = []
        for r in self._load():
            if r.get("state", "pending") == "pending":
                times.append(r["at"])
                if r.get("kind") == "focus" and r.get("live"):
                    times.append(r.get("edited", now) + 60)
            elif r.get("state") == "open" and r.get("next"):
                times.append(r["next"])
        return min(times, default=now + 3600)

    async def loop(self) -> None:
        while True:
            try:
                await self.fire_due()
            except Exception:
                log.exception("follow-ups")
            self._wake.clear()
            try:  # (until the next one is due, one is added, or a minute at most: the clock may jump)
                await asyncio.wait_for(self._wake.wait(), max(1.0, min(60.0, self._next_due() - time.time())))
            except asyncio.TimeoutError:
                pass
