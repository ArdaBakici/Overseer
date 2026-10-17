"""Days and moments as people write them: for task dates and reminders. Local time (this machine's time zone)."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

from .tasks import WEEKDAYS, TaskError

SHORT_DAYS = {d[:3]: i for i, d in enumerate(WEEKDAYS)}


def parse_day(text: str, today: date | None = None) -> date:
    """YYYY-MM-DD, today, tomorrow, a weekday (the next one: today's weekday means a week from now), +3d, +2w."""
    today = today or date.today()
    t = text.strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        try:
            return date.fromisoformat(t)
        except ValueError:
            pass
    elif t in ("today", "tod"):
        return today
    elif t in ("tomorrow", "tmr", "tom"):
        return today + timedelta(days=1)
    elif m := re.fullmatch(r"\+?(\d+)\s*(d|days?|w|weeks?)", t):
        n = int(m.group(1))
        return today + timedelta(days=n * (7 if m.group(2)[0] == "w" else 1))
    else:
        wd = SHORT_DAYS.get(t[:3]) if len(t) >= 3 and any(d.startswith(t) for d in WEEKDAYS) else None
        if wd is not None:
            return today + timedelta(days=(wd - today.weekday() - 1) % 7 + 1)
    raise TaskError(f"{text!r}: a day is YYYY-MM-DD, today, tomorrow, a weekday or +3d")


def _clock(text: str) -> time:
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text.strip().lower())
    if not m:
        raise TaskError(f"{text!r}: a time is like 17:30, 9am or 9:15pm")
    h, mi, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm:
        if not 1 <= h <= 12:
            raise TaskError(f"{text!r}: not a time")
        h = h % 12 + (12 if ampm == "pm" else 0)
    elif m.group(2) is None:
        raise TaskError(f"{text!r}: a time is like 17:30, 9am or 9:15pm")
    if h > 23 or mi > 59:
        raise TaskError(f"{text!r}: not a time")
    return time(h, mi)


def parse_moment(text: str, now: datetime | None = None) -> datetime:
    """When a reminder is due: "in 2h" / "45m" / "1h30m", "17:30" (today, or tomorrow if past), "tomorrow 9am",
    "friday 18:00", "2026-10-20 09:00", or a day alone (09:00 then)."""
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    t = " ".join(text.strip().lower().split())
    t = re.sub(r"^(at|on)\s+", "", t)
    m = re.fullmatch(r"(?:in\s+)?((?:\d+\s*(?:d|h|m|min|mins|minutes?|hours?|days?)\s*)+)", t)
    if m:
        total = 0
        for n, unit in re.findall(r"(\d+)\s*(d|h|m)", m.group(1)):
            total += int(n) * {"d": 86400, "h": 3600, "m": 60}[unit]
        if total <= 0:
            raise TaskError("a reminder needs a time in the future")
        return now + timedelta(seconds=total)
    parts = t.split(" ")
    try:  # "17:30" alone
        clock = _clock(t)
        when = datetime.combine(now.date(), clock)
        return when if when > now else when + timedelta(days=1)
    except TaskError:
        pass
    if len(parts) == 2 and parts[1].startswith("at"):
        parts = parts[:1]
    if len(parts) == 3 and parts[1] == "at":
        parts = [parts[0], parts[2]]
    day = parse_day(parts[0], now.date())
    clock = _clock(parts[1]) if len(parts) == 2 else time(9, 0)
    if len(parts) > 2:
        raise TaskError(f"{text!r}: say e.g. 17:30, tomorrow 9am, friday 18:00 or in 2h")
    when = datetime.combine(day, clock)
    if when <= now:
        raise TaskError(f"{when:%a %d %b %H:%M} has passed")
    return when
