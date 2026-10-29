"""The HPC's sections of the morning digest (core/digest.py): the jobs that ended (failed ones first) and the ones
still running or queued, the GPU time your jobs used (one sacct), the new results and findings; and at the end, the
GPUs free now."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ..core.digest import _clip
from ..core.notify import code, esc
from .compute import is_helper
from .slurm import STATE_EMOJI, SUCCESS_STATES, gpus_from_tres

log = logging.getLogger(__name__)


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


class HpcDigest:
    def __init__(self, domain):
        self.domain = domain
        self.store = domain.store

    @property
    def hpc(self):
        return self.domain.hpc

    @property
    def monitor(self):
        return self.domain.monitor

    async def sections(self, since: float, now: float) -> list[str]:
        return [self._jobs(since), await self._gpu(since, now), self._results(since)]

    async def closing(self) -> list[str]:
        return [await self._cluster()]

    def _jobs(self, since: float) -> str:
        ended = [r for r in self.store.jobs_ended_since(since) if not r.ignored]
        active = [r for r in self.store.active_jobs() if not is_helper(r.name)]
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
