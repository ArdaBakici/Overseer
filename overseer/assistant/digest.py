"""The assistant's sections of the morning digest (core/digest.py): today's events, today's tasks and the overdue
ones, today's reminders, and what changed in the vault since the last digest."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from ..core.notify import esc
from . import tasks as T
from .domain import files_line
from .reminders import when_text


async def sections(domain, since: float, now: float) -> list[str]:
    out = []
    day = domain.today()
    cal = domain.calendar
    if cal.enabled:
        if not cal.synced or time.time() - cal.synced > 600:
            await cal.sync()
        events = cal.cached_day(day) if cal.synced else []
        if events:
            lines = [f"📅 <b>Calendar</b> ({len(events)} event{'s' if len(events) != 1 else ''})"]
            lines += [f"• {'all day' if e.all_day else f'{e.starts():%H:%M}–{e.ends():%H:%M}'} {esc(e.title or '(no title)')}"
                      + (f" <i>@ {esc(e.where[:40])}</i>" if e.where else "") for e in events[:12]]
            out.append("\n".join(lines))
        elif cal.synced:
            out.append("📅 <b>Calendar</b>: nothing today.")
    items = await asyncio.to_thread(T.scan, domain.vault)
    due = T.select(items, "today", today=day)
    if due:
        lines = [f"✅ <b>Today</b> ({len(due)} task{'s' if len(due) != 1 else ''}"
                 + (f", {sum(1 for t in due if t.due and t.due < day.isoformat())} overdue" if any(
                     t.due and t.due < day.isoformat() for t in due) else "") + ")"]
        lines += [f"• {esc(T.describe(t, day).split(' ', 1)[1])}" for t in due[:12]]
        if len(due) > 12:
            lines.append(f"(+{len(due) - 12} more: /tasks today)")
        out.append("\n".join(lines))
    else:
        out.append("✅ <b>Today</b>: no tasks due.")
    end_of_day = datetime.combine(day, datetime.max.time()).timestamp()
    rem = [r for r in domain.reminders.pending() if r["at"] <= end_of_day]
    if rem:
        out.append("⏰ <b>Reminders today</b>\n" + "\n".join(f"• {esc(when_text(r['at']).split(' ')[-1])} "
                                                            f"{esc(r['text'])}" for r in rem[:10]))
    changes = [c for c in await domain.vault.log(100) if since <= c.when <= now]
    mine = [c for c in changes if c.message not in ("Your edits", "Vault history starts")]
    if mine:
        files = {p: st for c in reversed(mine) for st, p in c.files}
        out.append(f"📝 <b>Vault</b>: {len(mine)} change{'s' if len(mine) != 1 else ''} by the assistant · "
                   + files_line(sorted((st, p) for p, st in files.items()), 6))
    return out
