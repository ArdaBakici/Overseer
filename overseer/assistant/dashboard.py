"""The assistant's dashboard tabs (core/dashboard.py serves them): today (events, what's due, reminders), every task, and the
vault's history (what changed, by whom). Read-only: ticking tasks off and undoing are in the chat."""

from __future__ import annotations

import asyncio
import time

from ..core.dashboard import json_answer
from . import tasks as T


async def route(domain, parts: list[str], query: dict[str, str]):
    if parts == ["today"]:
        return json_answer(await today(domain))
    if parts == ["tasks"]:
        return json_answer(await tasks(domain, query.get("when", "open") or "open", query.get("q", "")))
    if parts == ["changes"]:
        return json_answer(await changes(domain, int(query.get("limit", "40") or 40)))
    return None


async def today(domain) -> dict:
    items = await asyncio.to_thread(T.scan, domain.vault)
    day = domain.today()
    return {"date": day.isoformat(), "name": domain.persona_name(), "daily": domain.vault.daily_note(),
            "due": [t.to_dict() for t in T.select(items, "today", today=day)[:100]],
            "upcoming": [t.to_dict() for t in T.select(items, "upcoming", today=day)[:15]],
            "reminders": [{"at": r["at"], "text": r["text"], "kind": r.get("kind", "reminder")}
                          for r in domain.reminders.pending()[:30]],
            "waiting": [{"at": r["at"], "text": r["text"], "kind": r.get("kind", "reminder")}
                        for r in domain.reminders.open()[:30]],
            "log": [{"at": e["at"], "icon": e["icon"], "text": e["text"]} for e in domain.reminders.log()[-40:]],
            "routines": routines_today(domain),
            "events": [e.to_dict() for e in domain.calendar.cached_day(day)] if domain.calendar.enabled else None,
            "conflicts": await asyncio.to_thread(domain.vault.conflicts)}


def routines_today(domain) -> list[dict]:
    from datetime import datetime, time as dtime, timedelta

    rs = domain.scheduler.load()
    if rs is None:
        return []
    now = datetime.now()
    start, end = datetime.combine(now.date(), dtime(0, 0)), datetime.combine(now.date(), dtime(23, 59))
    return [{"at": when.timestamp(), "name": r.name, "kind": "do" if r.do else "say", "past": when <= now}
            for r in rs.routines for when in r.occurrences(start - timedelta(seconds=1), end)]


async def tasks(domain, when: str, q: str) -> dict:
    items = await asyncio.to_thread(T.scan, domain.vault)
    try:
        hits = T.select(items, when, search=q[:200], today=domain.today())
    except T.TaskError as e:
        return {"error": str(e), "tasks": []}
    return {"tasks": [t.to_dict() for t in hits[:500]], "total": len(hits)}


async def changes(domain, limit: int) -> dict:
    log = await domain.vault.log(max(1, min(limit, 100)))
    return {"history": domain.vault.history, "now": time.time(),
            "changes": [{"commit": c.commit[:12], "when": c.when, "message": c.message,
                         "files": [[st, p] for st, p in c.files[:50]], "more": max(0, len(c.files) - 50)}
                        for c in log]}
