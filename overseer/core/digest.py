"""The morning digest: once a day, in the general topic, what happened since the last one.

The jobs that ended (failed ones first) and the ones still running or queued, the GPU time your jobs used, the new
results and findings, what each agent did and what it's waiting for, the tokens and cost, and anything that waits
for you (feedback, skill proposals). Put together from Overseer's own records and one `sacct`: no model involved,
so it costs nothing. /digest shows it at any time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta

from .config import Config
from .notify import Notifier, Outgoing, code, esc
from ..hpc.slurm import STATE_EMOJI, SUCCESS_STATES, gpus_from_tres, human_seconds
from .store import Store
from .usage import ORIGINS, _n

log = logging.getLogger(__name__)

TAIL_BYTES = 400_000  # of a conversation's transcript, to find its last reply


def gpu_hours(jobs, window_start: str, now: str) -> tuple[float, int]:
    """GPU-hours your jobs used between two cluster times (ISO), from sacct rows (Job with alloc_tres, start_time,
    end_time): (hours, how many jobs used GPUs)."""
    try:
        lo, hi = datetime.fromisoformat(window_start), datetime.fromisoformat(now)
    except ValueError:
        return 0.0, 0
    total, n = 0.0, 0
    for j in jobs:
        gpus = int(gpus_from_tres(j.alloc_tres) or 0)
        if not gpus:
            continue
        try:
            start = datetime.fromisoformat(j.start_time)
        except ValueError:
            continue  # never started (Unknown / None)
        try:
            end = datetime.fromisoformat(j.end_time)
        except ValueError:
            end = hi  # still running
        seconds = (min(end, hi) - max(start, lo)).total_seconds()
        if seconds > 0:
            total += gpus * seconds / 3600
            n += 1
    return total, n


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
    def __init__(self, cfg: Config, store: Store, hpc, notifier: Notifier, agent=None, monitor=None):
        self.cfg = cfg
        self.store = store
        self.hpc = hpc
        self.notifier = notifier
        self.agent = agent
        self.monitor = monitor

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
        sections = [head, self._jobs(since), await self._gpu(since, now), self._results(since),
                    self._agents(since), self._usage(since), self._waiting(), await self._cluster()]
        return "\n\n".join(s for s in sections if s)

    def _jobs(self, since: float) -> str:
        ended = [r for r in self.store.jobs_ended_since(since) if not r.ignored]
        active = [r for r in self.store.active_jobs() if r.name != "overseer-worker"]
        running = [r for r in active if r.state == "RUNNING"]
        queued = [r for r in active if r.state == "PENDING"]
        ok = [r for r in ended if r.state in SUCCESS_STATES]
        bad = [r for r in ended if r not in ok]
        if not (ended or active):
            return "<b>Jobs</b>: nothing ran."
        counts = " · ".join(x for x in [f"✅ {len(ok)} finished" if ok else "", f"❌ {len(bad)} failed" if bad else "",
                                        f"🏃 {len(running)} running" if running else "",
                                        f"⏳ {len(queued)} queued" if queued else ""] if x)
        lines = [f"<b>Jobs</b>: {counts}"]

        def row(r, extra: str = "") -> str:
            where = f" 📁 {esc(r.project)}" if r.project else ""
            return f"{STATE_EMOJI.get(r.state, '•')} {code(r.job_id)} {esc(r.name)}{extra}{where}"
        for r in bad[:8]:
            why = f" · {esc(r.report[:80])}" if r.report else ""
            lines.append(row(r, f" · {esc(r.state)}{why}"))
        for r in ok[:max(0, 10 - len(bad[:8]))]:
            lines.append(row(r))
        if len(ok) + len(bad) > 10:
            lines.append(f"… and {len(ok) + len(bad) - 10} more (/jobs)")
        for r in running[:8]:
            idle = self.monitor.idle_state(r.job_id) if self.monitor else {}
            lines.append(row(r, " · running" + (" · 💤 GPU idle" if idle.get("alerted") else "")))
        return "\n".join(lines)

    async def _gpu(self, since: float, now: float) -> str:
        """GPU time of all your jobs (Overseer's or not), from SLURM's accounting."""
        try:
            clock = await self.hpc.clock()
            rows = await self.hpc.sacct_recent(int((now - since) / 60) + 5, fold=False)
        except Exception as e:
            log.info("digest: no GPU time (%s)", e)
            return ""
        if not clock:
            return ""
        start = datetime.fromisoformat(clock) - timedelta(seconds=now - since)  # (in the cluster's time)
        used, n = gpu_hours(rows, start.isoformat(timespec="seconds"), clock)
        return f"🎮 <b>GPU time</b>: {used:.1f} GPU-hours in {n} job(s)" if n else ""

    def _results(self, since: float) -> str:
        new = [r for r in self.store.results() if r.created >= since]
        judged = [r for r in self.store.results() if r.updated >= since and r.verdict and r.created < since]
        found = [f for f in self.store.findings() if f.created >= since]
        if not (new or judged or found):
            return ""
        lines = []
        if new or judged:
            verdicts: dict[str, int] = {}
            for r in new:
                verdicts[r.verdict or "not judged yet"] = verdicts.get(r.verdict or "not judged yet", 0) + 1
            line = (f"<b>Results</b>: {len(new)} new" + (" (" + ", ".join(f"{n} {esc(v)}" for v, n in
                                                                       sorted(verdicts.items(), key=lambda kv: -kv[1]))
                                                          + ")" if verdicts else ""))
            if judged:
                line += f" · {len(judged)} older one(s) judged"
            lines.append(line + " · /results")
            for r in [r for r in new if r.verdict in ("promising", "broken")][:4]:
                what = f": {esc(_clip(r.conclusion, 140))}" if r.conclusion else ""
                lines.append(f"   {'🌱' if r.verdict == 'promising' else '🔧'} {code(r.id)} {esc(r.name)}{what}")
        for f in found[:6]:
            icon = "🚫" if f.kind == "dead-end" else "💡"
            where = f" ({esc(f.project)})" if f.project else ""
            lines.append(f"{icon}{where} {esc(_clip(f.text, 200))}")
        return "\n".join(lines)

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

    async def _cluster(self) -> str:
        """One line: GPUs free now per GPU partition."""
        if not self.monitor:
            return ""
        try:
            snap = await self.monitor.cluster.snapshot()
        except Exception:
            return ""
        parts = [p for p in snap["partitions"] if p.gpus]
        if not parts:
            return ""
        return "🖥 <b>Free now</b>: " + " · ".join(f"{esc(p.name)} {p.gpus_free}/{p.gpus} GPUs" for p in
                                                  sorted(parts, key=lambda p: -p.gpus)[:5]) + " (/cluster)"
