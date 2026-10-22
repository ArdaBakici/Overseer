"""Routines: Assistant/Routines.md in the vault, edited by you and the assistant, run by Overseer at their times.

    # Routines

    ## Morning planning
    - when: weekdays 09:00
    - do: Help me pick my top 3 for today.        (the assistant is woken and starts a conversation)

    ## Meds
    - when: every day 09:30 and 21:30
    - say: Meds 💊                                (just this message: no model involved)
    - follow-up: persistent
    - paused until: 2026-11-01                     (optional; also `- off: yes`)

    # Follow-ups
    - default: again after 20m, up to 2 more times
    - persistent: again after 10m, up to 5 more times

`when`: every day / daily / weekdays / weekends / weekday names ("monday, thursday", "sun"), then one or more
times (09:00, 9am, 21:30). The note's other text is ignored, so it can hold explanations. Overseer decides when a
routine is due (never the model); a routine missed while Overseer was down runs at the next start if it was due
less than CATCH_UP ago, else it's skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from .tasks import WEEKDAYS

CATCH_UP = 3 * 3600
DEFAULT_POLICY = {"after": 30 * 60, "times": 1}

TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b|\b(\d{1,2})\s*(am|pm)\b", re.I)
FIELD_RE = re.compile(r"^\s*[-*]\s+\**([A-Za-z][A-Za-z -]*?)\**\s*:\s*(.*)$")
POLICY_RE = re.compile(r"(?:again\s+)?(?:after|every)\s+(\d+)\s*(m|min|mins|minutes?|h|hours?)\s*,?\s*"
                       r"(?:up\s+to\s+)?(\d+)\s*(?:more\s+)?times?", re.I)

EXAMPLE = """# Routines

What I do, and when. The assistant reads and edits this note; so can I.

## Morning planning
- when: weekdays 09:00
- do: Help me pick my top 3 for today, size them, and put them in today's note. One question at a time.

## Evening shutdown
- when: every day 21:00
- do: What got done, what moves to tomorrow, my commitments and anything I didn't answer today. No guilt.

## Weekly review
- when: sunday 18:00
- do: Look back at the week and plan the next one with me.

## Nightly reflection
- when: every day 23:30
- do: Read today's conversations (ov history day) and update what you keep about me - About me, and the notes
  flagged `assistant: always` - with anything lasting you learned. Then one short line saying what you updated.

# Follow-ups

How often an unanswered reminder comes back.

- default: again after 20m, up to 2 more times
- persistent: again after 10m, up to 5 more times
"""


@dataclass
class Routine:
    name: str
    days: set[int] = field(default_factory=set)  # 0 = Monday
    times: list[time] = field(default_factory=list)
    do: str = ""
    say: str = ""
    follow_up: str = ""
    paused_until: date | None = None
    off: bool = False

    @property
    def key(self) -> str:
        return " ".join(self.name.lower().split())

    def active(self, day: date) -> bool:
        return not self.off and not (self.paused_until and day <= self.paused_until)

    def occurrences(self, start: datetime, end: datetime) -> list[datetime]:
        """Its times in (start, end], oldest first."""
        out = []
        d = start.date()
        while d <= end.date():
            if d.weekday() in self.days and self.active(d):
                out += [m for t in self.times if start < (m := datetime.combine(d, t)) <= end]
            d += timedelta(days=1)
        return sorted(out)

    def next_time(self, after: datetime) -> datetime | None:
        nxt = self.occurrences(after, after + timedelta(days=60))
        return nxt[0] if nxt else None

    def describe_when(self) -> str:
        if self.days == set(range(7)):
            days = "every day"
        elif self.days == set(range(5)):
            days = "weekdays"
        elif self.days == {5, 6}:
            days = "weekends"
        else:
            days = ", ".join(WEEKDAYS[d][:3] for d in sorted(self.days))
        return f"{days} {' and '.join(t.strftime('%H:%M') for t in self.times)}"


@dataclass
class Routines:
    routines: list[Routine] = field(default_factory=list)
    policies: dict[str, dict] = field(default_factory=dict)  # name -> {"after": seconds, "times": n}
    errors: list[str] = field(default_factory=list)
    refs: list[tuple[str, str]] = field(default_factory=list)  # (routine, follow-up it names): checked at the end

    def policy(self, spec: str | None) -> dict:
        """A follow-up policy: a name from # Follow-ups, an inline one, "none"; default otherwise."""
        spec = (spec or "").strip()
        if spec.lower() in self.policies:
            return self.policies[spec.lower()]
        if spec:
            parsed = parse_policy(spec)
            if parsed is not None:
                return parsed
        return self.policies.get("default", DEFAULT_POLICY)

    def get(self, name: str) -> Routine | None:
        key = " ".join(name.lower().split())
        return next((r for r in self.routines if r.key == key), None)


def parse_policy(text: str) -> dict | None:
    t = text.strip().lower()
    if t in ("none", "no", "off", "never", "0"):
        return {"after": 0, "times": 0}
    m = POLICY_RE.search(t)
    if not m:
        return None
    n, unit, times = int(m.group(1)), m.group(2)[0], int(m.group(3))
    after = n * (3600 if unit == "h" else 60)
    if after < 60 or times > 20:
        return None
    return {"after": after, "times": times}


def describe_policy(p: dict) -> str:
    if not p.get("times"):
        return "no follow-up"
    return f"again after {p['after'] // 60}m, up to {p['times']} more time{'s' if p['times'] != 1 else ''}"


def parse_when(text: str) -> tuple[set[int], list[time]]:
    t = text.lower().replace("·", " ")
    times = []
    for m in TIME_RE.finditer(t):
        if m.group(1) is not None:
            h, mi = int(m.group(1)), int(m.group(2))
        else:
            h, mi = int(m.group(3)) % 12 + (12 if m.group(4).lower() == "pm" else 0), 0
        if h > 23 or mi > 59:
            raise ValueError(f"{m.group(0)!r} isn't a time")
        times.append(time(h, mi))
    if not times:
        raise ValueError("no time in it (e.g. 09:00 or 9am)")
    rest = TIME_RE.sub(" ", t)
    words = [w for w in re.split(r"[\s,/&+]+", rest) if w and w not in ("every", "and", "on", "at", "each", "")]
    days: set[int] = set()
    for w in words:
        if w in ("day", "days", "daily"):
            days |= set(range(7))
        elif w in ("weekday", "weekdays"):
            days |= set(range(5))
        elif w in ("weekend", "weekends"):
            days |= {5, 6}
        elif len(w) >= 3 and any(d.startswith(w.rstrip("s")) for d in WEEKDAYS):
            days.add(next(i for i, d in enumerate(WEEKDAYS) if d.startswith(w.rstrip("s"))))
        else:
            raise ValueError(f"didn't understand {w!r} (say e.g. every day, weekdays, weekends, monday, sun)")
    return days or set(range(7)), sorted(set(times))


def parse(text: str) -> Routines:
    out = Routines()
    section = ""
    cur: Routine | None = None
    last_field = ""
    raw: dict[str, str] = {}

    def finish():
        if cur is None:
            return
        try:
            if "when" not in raw:
                raise ValueError("no `when:`")
            cur.days, cur.times = parse_when(raw["when"])
            if bool(cur.do) == bool(cur.say):
                raise ValueError("needs either `do:` (the assistant does it with you) or `say:` (just a message)")
            if cur.follow_up and cur.follow_up.lower() != "default":
                out.refs.append((cur.name, cur.follow_up))
            if raw.get("paused until"):
                try:
                    cur.paused_until = date.fromisoformat(raw["paused until"].strip()[:10])
                except ValueError:
                    raise ValueError("`paused until:` is a date, YYYY-MM-DD") from None
            if out.get(cur.name):
                raise ValueError("there are two routines with this name")
            out.routines.append(cur)
        except ValueError as e:
            out.errors.append(f"{cur.name}: {e}")

    for line in text.splitlines():
        if re.match(r"^#\s", line):
            finish()
            cur, raw, last_field = None, {}, ""
            title = line[2:].strip().lower()
            section = "followups" if title.startswith("follow") else "routines" if title.startswith("routine") else ""
            continue
        if section == "routines" and re.match(r"^##\s", line):
            finish()
            cur, raw, last_field = Routine(name=line[3:].strip()), {}, ""
            continue
        m = FIELD_RE.match(line)
        if section == "followups" and m:
            name, value = m.group(1).strip().lower(), m.group(2).strip()
            p = parse_policy(value)
            if p is None:
                out.errors.append(f"Follow-ups: {name}: say e.g. `again after 20m, up to 2 more times` (or none)")
            else:
                out.policies[name] = p
            continue
        if section != "routines" or cur is None:
            continue
        if m:
            key, value = m.group(1).strip().lower().replace("-", " "), m.group(2).strip()
            key = {"follow up": "follow up", "followup": "follow up", "paused": "paused until"}.get(key, key)
            raw[key] = value
            last_field = key
            if key == "do":
                cur.do = value
            elif key == "say":
                cur.say = value
            elif key == "follow up":
                cur.follow_up = value
            elif key in ("off", "active"):
                yes = value.lower() in ("yes", "true", "y", "on", "1")
                cur.off = yes if key == "off" else not yes
        elif line.startswith(("  ", "\t")) and line.strip() and last_field in ("do", "say"):
            setattr(cur, last_field, (getattr(cur, last_field) + " " + line.strip()).strip())  # a long `do:`
    finish()
    for name, spec in out.refs:  # (follow-up names may be defined below the routines)
        if spec.lower() not in out.policies and parse_policy(spec) is None:
            out.errors.append(f"{name}: follow-up {spec!r} isn't one of # Follow-ups ({', '.join(out.policies) or 'none'})"
                              " or like `again after 10m, up to 3 more times`")
    return out


def today_lines(rs: Routines, now: datetime) -> list[str]:
    """What's still to come today, for the agent's context."""
    end = datetime.combine(now.date(), time(23, 59))
    items = sorted((when, r) for r in rs.routines for when in r.occurrences(now, end))
    return [f"{when:%H:%M} {r.name} ({'do' if r.do else 'say'})" for when, r in items]
