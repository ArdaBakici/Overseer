"""Token usage of the agent's runs (/usage): every run's tokens and cost are recorded by the agent."""

from __future__ import annotations

import time

from .notify import code, esc
from .store import Store

ORIGINS = {"user": "from you", "wakeup": "wake-ups", "checkin": "check-ins", "compact": "compactions",
           "subagent": "subagent tasks"}


def _n(x: float) -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    return f"{x / 1000:.1f}k" if x >= 1000 else str(int(x))


def _period(period: str) -> tuple[float, float | None, str] | None:
    now = time.time()
    lt = time.localtime(now)
    midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    return {
        "today": (midnight, None, "today"),
        "yesterday": (midnight - 86400, midnight, "yesterday"),
        "week": (now - 7 * 86400, None, "in the last 7 days"),
        "month": (now - 30 * 86400, None, "in the last 30 days"),
    }.get(period)


def usage_report(store: Store, period: str = "today") -> str:
    """Tokens and cost of the agent's runs in a period (HTML)."""
    p = _period(period)
    if p is None:
        return "Usage: /usage [today|yesterday|week|month]"
    since, until, label = p
    rows = store.usage_rows(since, until)
    if not rows:
        return f"📊 No agent runs {label}."
    tokens = sum(r["input"] + r["output"] for r in rows)
    cost = sum(r["cost"] or 0 for r in rows)
    unpriced = sum(1 for r in rows if r["cost"] is None)
    lines = [
        f"📊 <b>Agent usage {label}</b>",
        f"<b>{_n(tokens)} tokens</b>: {_n(sum(r['input'] for r in rows))} input "
        f"({_n(sum(r['cached'] for r in rows))} of it cached) + {_n(sum(r['output'] for r in rows))} output"
        + (f" · <b>${cost:.2f}</b>" if cost else ""),
    ]
    origins: dict[str, int] = {}
    for r in rows:
        origins[r["origin"]] = origins.get(r["origin"], 0) + 1
    lines.append(f"{len(rows)} run(s): " + ", ".join(f"{n} {ORIGINS.get(o, o)}" for o, n in
                                                       sorted(origins.items(), key=lambda kv: -kv[1])))
    backends: dict[str, list] = {}
    for r in rows:
        backends.setdefault(r["backend"], []).append(r)
    if len(backends) > 1 or unpriced:
        lines.append("by backend: " + " · ".join(
            f"{esc(b)} {_n(sum(x['input'] + x['output'] for x in rs))}"
            + (f" ${sum(x['cost'] or 0 for x in rs):.2f}" if any(x["cost"] is not None for x in rs) else "")
            for b, rs in backends.items()))
    workers = {sid for sid in {r["session_id"] for r in rows} if (s := store.get_session(sid)) and s.is_worker}
    if workers:  # planner mode: what the subagents cost, apart from your conversations (and their planners)
        def part(rs: list) -> str:
            c = sum(x["cost"] or 0 for x in rs)
            return f"{_n(sum(x['input'] + x['output'] for x in rs))}" + (f" ${c:.2f}" if c else "")
        sub_rows = [r for r in rows if r["session_id"] in workers]
        lines.append(f"subagents ({len(workers)}): {part(sub_rows)} · conversations and planners: "
                     f"{part([r for r in rows if r['session_id'] not in workers])}")
    if unpriced:
        lines.append("(Codex doesn't report a cost: its runs are in the tokens only)")
    convs: dict[str, list] = {}
    for r in rows:
        convs.setdefault(r["session_id"], []).append(r)
    if len(convs) > 1:
        top = sorted(convs.items(), key=lambda kv: -sum(x["input"] + x["output"] for x in kv[1]))[:5]
        lines.append("top conversations:")
        for sid, rs in top:
            proj = rs[0]["project"]
            c = sum(x["cost"] or 0 for x in rs)
            lines.append(f"    {code(sid)}" + (f" 📁 {esc(proj)}" if proj else "")
                         + f" · {_n(sum(x['input'] + x['output'] for x in rs))}" + (f" · ${c:.2f}" if c else "")
                         + f" · {len(rs)} run(s)")
    return "\n".join(lines)
