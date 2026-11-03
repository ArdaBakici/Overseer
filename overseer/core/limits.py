"""The CLIs' subscription limits: how much of each usage window an account has used, and when it resets.

Both CLIs report them during runs - Claude Code as `rate_limit_event`s in its output (the 5-hour and weekly windows,
and a model's own weekly bucket where the plan has one), Codex as `account/rateLimits/updated` (its primary and
secondary windows, and per-model limits). Overseer keeps the latest per account, with when it was seen: numbers are
only as fresh as the account's last run. An account on an API key has no such limits (its cost is what counts).

They show in each reply's footer, /usage and /account. At WARN % of a window you're told once; at SWITCH % the
bot moves on to another account of the same CLI that has room, before running.
"""

from __future__ import annotations

import json
import time
from typing import Any

WARN = 80
SWITCH = 95
KEY = "limits:"  # + backend:account -> {"seen", "windows": [{"name", "used", "resets"}], "status"}
CLAUDE_NAMES = {"five_hour": "5h", "seven_day": "week", "seven_day_opus": "week (Opus)",
                "seven_day_sonnet": "week (Sonnet)", "seven_day_overage_included": "week (incl. extra)",
                "overage": "extra usage"}


def _window(name: str, used: float, resets: Any) -> dict:
    try:
        resets = float(resets or 0)
    except (TypeError, ValueError):
        resets = 0.0
    return {"name": name, "used": round(float(used), 1), "resets": resets}


def _merge(before: dict | None, windows: list[dict], status: str = "") -> dict:
    """A newer report over an older one of the same run: windows by name (a report may carry only one)."""
    out = {w["name"]: w for w in (before or {}).get("windows", [])}
    out.update({w["name"]: w for w in windows})
    return {"windows": list(out.values()), "status": status or (before or {}).get("status", "")}


def from_claude(info: dict, before: dict | None = None) -> dict:
    """Claude Code's rate_limit_info (utilization: the fraction of the window used)."""
    windows = []
    for key, w in (info.get("unifiedWindows") or {}).items():
        if isinstance(w, dict) and w.get("utilization") is not None:
            windows.append(_window(CLAUDE_NAMES.get(key, key), w["utilization"] * 100, w.get("resetsAt")))
    kind = info.get("rateLimitType")
    if kind and info.get("utilization") is not None and CLAUDE_NAMES.get(kind, kind) not in {w["name"] for w in windows}:
        windows.append(_window(CLAUDE_NAMES.get(kind, kind), info["utilization"] * 100, info.get("resetsAt")))
    return _merge(before, windows, str(info.get("status") or ""))


def _codex_name(w: dict, fallback: str) -> str:
    mins = w.get("windowDurationMins")
    if mins == 300:
        return "5h"
    if mins == 10080:
        return "week"
    if mins:
        return f"{mins // 1440}d" if mins % 1440 == 0 else f"{mins // 60}h" if mins % 60 == 0 else f"{mins}m"
    return fallback


def from_codex(snapshot: dict, by_id: dict | None = None, before: dict | None = None) -> dict:
    """Codex's RateLimitSnapshot (usedPercent: 0-100), and its other limits (e.g. a model's own)."""
    windows = []
    for slot, fallback in (("primary", "5h"), ("secondary", "week")):
        w = snapshot.get(slot)
        if isinstance(w, dict) and w.get("usedPercent") is not None:
            windows.append(_window(_codex_name(w, fallback), w["usedPercent"], w.get("resetsAt")))
    for lid, other in (by_id or {}).items():
        if not isinstance(other, dict) or (other.get("limitId") or lid) == (snapshot.get("limitId") or "codex"):
            continue  # (the main limit again)
        label = other.get("limitName") or lid
        for slot, fallback in (("primary", "5h"), ("secondary", "week")):
            w = other.get(slot)
            if isinstance(w, dict) and w.get("usedPercent") is not None:
                windows.append(_window(f"{_codex_name(w, fallback)} ({label})", w["usedPercent"], w.get("resetsAt")))
    return _merge(before, windows, str(snapshot.get("rateLimitReachedType") or ""))


def ago(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 60}m ago" if s < 3600 else f"{s // 3600}h ago" if s < 86400 else f"{s // 86400}d ago"


class Limits:
    def __init__(self, store):
        self.store = store

    def get(self, backend: str, account: str) -> dict | None:
        try:
            v = json.loads(self.store.get(f"{KEY}{backend}:{account}") or "null")
        except ValueError:
            return None
        if not v:
            return None
        now = time.time()  # (a window that has reset since is back to 0)
        for w in v.get("windows", []):
            if w.get("resets") and w["resets"] <= now:
                w["used"], w["resets"] = 0.0, 0.0
        return v

    def record(self, backend: str, account: str, report: dict) -> list[dict]:
        """What a run reported; the windows that just went past WARN % (to tell the researcher, once each)."""
        if not report or not report.get("windows"):
            return []
        before = self.get(backend, account) or {}
        merged = _merge(before, report["windows"], report.get("status", ""))
        merged["seen"] = time.time()
        self.store.set(f"{KEY}{backend}:{account}", json.dumps(merged))
        crossed = []
        for w in report["windows"]:
            mark = f"limitwarn:{backend}:{account}:{w['name']}:{int(w.get('resets') or 0) // 3600}"
            if w["used"] >= WARN and not self.store.get(mark):
                self.store.set(mark, "1")
                crossed.append(w)
        return crossed

    def fullest(self, backend: str, account: str) -> dict | None:
        """The window closest to its limit."""
        v = self.get(backend, account)
        return max(v["windows"], key=lambda w: w["used"]) if v and v.get("windows") else None

    def nearly_out(self, backend: str, account: str) -> dict | None:
        w = self.fullest(backend, account)
        return w if w and w["used"] >= SWITCH else None

    def line(self, backend: str, account: str, stale_after: float = 600) -> str:
        """For a reply's footer: "5h 42% · week 18%" (with the numbers' age once they're old). "" if none known."""
        v = self.get(backend, account)
        if not v or not v.get("windows"):
            return ""
        order = {"5h": 0, "week": 1}
        parts = [f"{w['name']} {w['used']:.0f}%" for w in sorted(v["windows"], key=lambda w: order.get(w["name"], 2))]
        age = time.time() - float(v.get("seen") or 0)
        return " · ".join(parts) + (f" ({ago(age)})" if age > stale_after else "")

    def windows(self, backend: str, account: str) -> tuple[list[dict], float]:
        v = self.get(backend, account) or {}
        order = {"5h": 0, "week": 1}
        return sorted(v.get("windows", []), key=lambda w: order.get(w["name"], 2)), float(v.get("seen") or 0)


def bar(used: float, width: int = 10) -> str:
    n = max(0, min(width, round(used / 100 * width)))
    return "▓" * n + "░" * (width - n)


def report(accounts, limits: Limits) -> str:
    """/usage: each CLI account's windows, as its last run reported them (HTML)."""
    from .accounts import when
    from .config import BACKENDS
    from .notify import esc

    lines = ["<b>📊 Usage limits</b> (from each account's last run)"]
    for b in BACKENDS:
        for n in accounts.names(b):
            label = esc(accounts.label(b, n)) + (" ▶" if n == accounts.active(b) else "")
            wins, seen = limits.windows(b, n)
            out = accounts.out_until(b, n)
            if not wins:
                lines.append(f"\n<b>{label}</b>: " + (f"out of usage until {esc(when(out))}" if out else
                                                      "nothing reported yet (it comes with its next run; an API key "
                                                      "has no such limits: /usage tokens)"))
                continue
            lines.append(f"\n<b>{label}</b> · {ago(time.time() - seen)}"
                         + (f" · out of usage until {esc(when(out))}" if out else ""))
            for w in wins:
                resets = f" · resets {esc(when(w['resets']))}" if w.get("resets") else ""
                lines.append(f"<code>{bar(w['used'])}</code> {esc(w['name'])} {w['used']:.0f}%{resets}")
    lines.append(f"\nTold at {WARN}%; at {SWITCH}% the bot moves to another account of the same CLI. Tokens and "
                 "cost: /usage tokens [today|week|month]")
    return "\n".join(lines)
