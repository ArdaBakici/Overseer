"""The HPC's dashboard tabs (core/dashboard.py serves them): jobs with their cards, graphs and logs; the results
table; findings; a timeline of the jobs that ran (SLURM accounting). Read-only."""

from __future__ import annotations

import html
import json
import re
import time
from importlib import resources

from ..core.dashboard import JOB_ID, json_answer
from .slurm import gpus_from_tres


def plain(card_html: str) -> str:
    """A card's Telegram HTML as plain text (the page escapes it again)."""
    return html.unescape(re.sub(r"<[^>]+>", "", card_html or "")).strip()


def page_js() -> bytes:
    return resources.files("overseer.hpc").joinpath("dashboard.js").read_bytes()


class HpcDashboard:
    def __init__(self, domain):
        self.domain = domain
        self.store = domain.store
        self._timeline: tuple[float, int, dict] | None = None

    @property
    def hpc(self):
        return self.domain.hpc

    @property
    def monitor(self):
        return self.domain.monitor

    async def route(self, parts: list[str], query: dict[str, str]):
        if parts == ["overview"]:
            return json_answer(self.overview())
        if parts == ["results"]:
            return json_answer(self.results(query.get("project", "")))
        if parts == ["findings"]:
            return json_answer(self.findings())
        if parts == ["timeline"]:
            return json_answer(await self.timeline(int(query.get("days", "7") or 7)))
        if len(parts) == 3 and parts[0] == "plot" and JOB_ID.match(parts[1]) and parts[2].isdigit():
            plots = self.monitor.cached_plots(parts[1]) if self.monitor else []
            i = int(parts[2])
            if i < len(plots):
                return 200, "image/png", plots[i][1], {}
            return 404, "text/plain", b"no such graph", {}
        if len(parts) == 2 and parts[0] == "log" and JOB_ID.match(parts[1]):
            return json_answer(await self.log(parts[1], int(query.get("lines", "80") or 80)))
        return None

    # ------------------------------------------------------------------ data
    def overview(self) -> dict:
        active = [r for r in self.store.active_jobs() if r.name != "overseer-worker"]
        seen = {r.job_id for r in active}
        recent = [r for r in self.store.recent_jobs(80) if r.job_id not in seen and r.name != "overseer-worker"]
        jobs = []
        for r in active + recent:
            try:
                card = json.loads(self.store.get(f"cardview:{r.job_id}") or "null")
            except ValueError:
                card = None
            idle = self.monitor.idle_state(r.job_id) if self.monitor and not r.ended_at else {}
            jobs.append({
                "id": r.job_id, "name": r.name, "project": r.project or "", "state": r.state,
                "agent": bool(r.by_agent), "seen": r.first_seen, "ended": r.ended_at or 0,
                "card": plain(card[0]) if card else "", "report": r.report or "",
                "plots": len(self.monitor.cached_plots(r.job_id)) if self.monitor else 0,
                "idle": bool(idle.get("alerted")), "start": self.store.get(f"eststart:{r.job_id}") or "",
            })
        return {"jobs": jobs, "projects": [p.name for p in self.store.list_projects()], "now": time.time()}

    def results(self, project: str = "") -> dict:
        scope: str | None | bool = False if project in ("", "all") else (None if project == "main" else project)
        rows, counts = [], {}
        for r in self.store.results(scope):
            try:
                metrics = json.loads(r.metrics or "{}") or {}
            except ValueError:
                metrics = {}
            for k in metrics:
                counts[k] = counts.get(k, 0) + 1
            rows.append({"id": r.id, "name": r.name, "project": r.project or "", "verdict": r.verdict,
                         "state": r.state, "success": bool(r.success), "elapsed": r.elapsed,
                         "ended": r.ended or r.created, "metrics": metrics, "hypothesis": r.hypothesis,
                         "conclusion": r.conclusion, "tags": r.tags})
        keys = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:6]
        return {"rows": rows, "metric_keys": keys}

    def findings(self) -> dict:
        return {"rows": [{"id": f"F{f.id}", "project": f.project or "", "kind": f.kind, "text": f.text,
                          "job": f.job_id, "created": f.created} for f in reversed(self.store.findings())]}

    async def timeline(self, days: int) -> dict:
        """Your jobs that ran in the last `days` (SLURM accounting, the cluster's clock): kept for a minute."""
        days = max(1, min(days, 30))
        if self._timeline and self._timeline[1] == days and time.time() - self._timeline[0] < 60:
            return self._timeline[2]
        try:
            clock = await self.hpc.clock()
            rows = await self.hpc.sacct_recent(days * 1440)
        except Exception as e:
            return {"error": f"couldn't read SLURM's accounting: {e}", "jobs": [], "clock": ""}
        tracked = {r.job_id: r for r in self.store.recent_jobs(2000)}
        jobs = []
        for j in rows:
            if j.name == "overseer-worker" or not j.start_time[:1].isdigit():
                continue  # (never started: nothing to draw)
            rec = tracked.get(j.job_id)
            jobs.append({"id": j.job_id, "name": j.name, "state": j.state, "start": j.start_time,
                         "end": j.end_time if j.end_time[:1].isdigit() else "", "gpus": int(gpus_from_tres(
                             j.alloc_tres) or 0), "project": (rec.project or "") if rec else "",
                         "tasks": j.extra.get("tasks", ""), "elapsed": j.elapsed})
        data = {"jobs": jobs, "clock": clock, "days": days}
        self._timeline = (time.time(), days, data)
        return data

    async def log(self, job_id: str, lines: int) -> dict:
        """The end of a tracked job's log (the path Overseer knows, never one from the request)."""
        rec = self.store.get_job(job_id)
        if not rec or rec.ignored:
            return {"error": "Overseer doesn't track this job", "text": ""}
        path = rec.info_dict.get("progress_log") or rec.stdout
        if not path:
            return {"error": "no log known for this job", "text": ""}
        text = await self.hpc.try_tail(path, max(10, min(lines, 400)))
        return {"path": path, "text": text[-60_000:]}
