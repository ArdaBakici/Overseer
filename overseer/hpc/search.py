"""What the HPC adds to the search (core/search.py): its results, findings and jobs, and how a hit of theirs reads."""

from __future__ import annotations

import json

from ..core.notify import code, esc
from ..core.search import _when


def search_rows(store) -> list[tuple]:
    rows = []
    for r in store.results(hidden=True):
        metrics = ", ".join(f"{k}={v}" for k, v in (json.loads(r.metrics or "{}") or {}).items())
        body = " · ".join(x for x in [r.name, r.verdict, r.hypothesis, r.conclusion, r.tags, metrics, r.state] if x)
        rows.append((body, f"result {r.id} {r.name}", "result", r.id, r.session_id or "", r.project or "",
                     r.ended or r.created, r.verdict or ""))
    for f in store.findings():
        rows.append((f.text, f"{f.kind} F{f.id}", "finding", f"F{f.id}", f.session_id or "", f.project or "",
                     f.created, f.kind))
    for j in store.jobs_for_search():
        body = " · ".join(x for x in [j.name, j.state, j.report, j.workdir] if x)
        rows.append((body, f"job {j.job_id} {j.name}", "job", j.job_id, j.session_id or "", j.project or "",
                     j.ended_at or j.first_seen, j.state))
    return rows


def search_line(h, where: str, snippet: str, html: bool) -> str | None:
    e = esc if html else (lambda x: x)
    c = code if html else (lambda x: x)
    if h.kind == "result":
        return f"📊 result {c(h.ref)}{where} · {e(_when(h.ts)[:10])}: {snippet}"
    if h.kind == "finding":
        return f"{'🚫' if h.what == 'dead-end' else '💡'} {c(h.ref)}{where}: {snippet}"
    if h.kind == "job":
        return f"🧾 job {c(h.ref)}{where} · {e(h.what)} · {e(_when(h.ts)[:10])}: {snippet}"
    return None
