"""The morning digest: once a day, in the general topic, what happened since the last one.

The domain's own sections first (for the HPC: the jobs that ended and the ones running or queued, the GPU time,
the new results and findings; hpc/digest.py), then what each agent did and what it's waiting for, the tokens and
cost, anything that waits for you (feedback, skill proposals), and the domain's closing lines (the HPC's free GPUs).
Put together from records, no model involved, so it costs nothing. /digest shows it at any time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from .config import Config
from .notify import Notifier, Outgoing, code, esc, human_seconds
from .store import Store
from .usage import ORIGINS, _n

log = logging.getLogger(__name__)

TAIL_BYTES = 400_000  # of a conversation's transcript, to find its last reply


def last_reply(cfg: Config, session_id: str, since: float) -> str:
    """The conversation's last answer since `since`, from its transcript ("" if none)."""
    path = cfg.local.state("transcripts") / f"{session_id}.jsonl"
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - TAIL_BYTES))
            lines = fh.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("ts", 0) < since:
            break
        if ev.get("kind") == "end" and ev.get("status") == "done" and (ev.get("text") or "").strip():
            return " ".join(ev["text"].split())
    return ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


class Digest:
    def __init__(self, cfg: Config, store: Store, notifier: Notifier, agent):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.agent = agent

    # ------------------------------------------------------------------ when
    def due(self, now: float | None = None) -> bool:
        """Past today's digest time (local time of this machine) and not sent today."""
        t = self.cfg.digest.time.strip()
        if not self.cfg.digest.enabled or not t:
            return False
        now = now or time.time()
        lt = time.localtime(now)
        try:
            hh, mm = (int(x) for x in t.split(":"))
        except ValueError:
            return False
        today = time.strftime("%Y-%m-%d", lt)
        return (lt.tm_hour, lt.tm_min) >= (hh, mm) and self.store.get("digest_day") != today

    async def loop(self) -> None:
        while True:
            try:
                if self.due():
                    await self.send()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("morning digest failed")
                self.store.set("digest_day", time.strftime("%Y-%m-%d"))  # (not again and again today)
            await asyncio.sleep(60)

    async def send(self) -> None:
        now = time.time()
        since = float(self.store.get("digest_at") or 0) or now - 86400
        since = max(since, now - 7 * 86400)
        text = await self.build(since, now)
        self.store.set("digest_at", str(now))
        self.store.set("digest_day", time.strftime("%Y-%m-%d", time.localtime(now)))
        await self.notifier.send(Outgoing(text))

    # ------------------------------------------------------------------ what
    async def build(self, since: float, now: float | None = None) -> str:
        now = now or time.time()
        hours = (now - since) / 3600
        span = (f"since {time.strftime('%H:%M', time.localtime(since))} yesterday" if 20 <= hours <= 30
                else f"the last {human_seconds(int(now - since))}")
        head = f"☀️ <b>Digest</b> · {time.strftime('%a %d %b', time.localtime(now))} ({span})"
        domain = self.agent.domain
        sections = [head, *await domain.digest_sections(since, now), self._agents(since), self._usage(since),
                    self._waiting(), *await domain.digest_closing()]
        return "\n\n".join(s for s in sections if s)

    def _agents(self, since: float) -> str:
        rows = self.store.usage_rows(since)
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["session_id"], []).append(r)
        waiting = {w.session_id for w in self.store.watches()}
        sids = list(by) + [s for s in waiting if s not in by]
        if not sids:
            return ""
        lines = ["<b>Agents</b>"]
        for sid in sids[:8]:
            rec = self.store.get_session(sid)
            if rec is None:
                continue
            runs = by.get(sid, [])
            label = f"🤖 {esc(rec.name)}" if rec.is_worker else ("📁 " + esc(rec.project) if rec.project else "💬 main")
            what = []
            if runs:
                origins: dict[str, int] = {}
                for r in runs:
                    origins[r["origin"]] = origins.get(r["origin"], 0) + 1
                what.append(f"{len(runs)} run(s)" + (" (" + ", ".join(f"{n} {ORIGINS.get(o, o)}" for o, n in
                                                                      origins.items() if o != "user") + ")"
                                                     if set(origins) - {"user"} else ""))
            watches = self.store.watches(sid)
            ap = getattr(self.agent, "autopilot", None) if self.agent else None
            if watches and ap:
                what.append(f"⏰ {esc(_clip(ap.describe_watches(watches), 120))}")
            lines.append(f"{label} {code(sid)}" + (" · " + " · ".join(what) if what else ""))
            reply = last_reply(self.cfg, sid, since)
            if reply:
                lines.append(f"   <i>{esc(_clip(reply, 220))}</i>")
        if len(sids) > 8:
            lines.append(f"… and {len(sids) - 8} more conversations")
        return "\n".join(lines)

    def _usage(self, since: float) -> str:
        rows = self.store.usage_rows(since)
        if not rows:
            return ""
        tokens = sum(r["input"] + r["output"] for r in rows)
        cost = sum(r["cost"] or 0 for r in rows)
        return f"🪙 <b>Tokens</b>: {_n(tokens)}" + (f" · ${cost:.2f}" if cost else "") + " (/usage)"

    def _waiting(self) -> str:
        """What waits for the researcher."""
        bits = []
        fb = self.store.feedback(("open",))
        if fb:
            bits.append(f"🛠 {len(fb)} open feedback item(s) (/feedback)")
        sk = self.store.skills(("proposed",))
        if sk:
            bits.append(f"🧩 {len(sk)} skill proposal(s) to answer")
        return " · ".join(bits)
