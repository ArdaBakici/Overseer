"""Results and findings: what each experiment showed, and the project's lasting lessons.

Results: a row per finished job, filled in by Overseer when it ends (state, runtime, its key metrics - the agent's
`ov metrics` critical/important ones first, then others found in its log), plus what the agent says it means
(`ov result <job> --hypothesis ... --conclusion ... --verdict promising`). The agent can also record a result that
isn't a job (r1, r2, ...: an evaluation by eye, a qualitative comparison). No numeric goal is needed: the verdicts
(promising, no-gain, worse, ...) carry the judgement, and a run of results without gain is pointed out to the agent.

Findings: insights and dead ends of a project (`ov finding dead-end "..."`). Unlike the research log, which gets
condensed, they are kept whole and shown to the agent whenever it starts or is woken in the project.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import json
import time

from .metrics import number, parse_names, pick
from .slurm import Job, slurm_seconds
from .store import FindingRecord, JobRecord, MetricSpec, ResultRecord, Store

VERDICTS = {"baseline": "📏", "promising": "✅", "no-gain": "➖", "worse": "🔻", "broken": "💥", "dead-end": "⛔",
            "inconclusive": "❔"}
NO_GAIN = {"no-gain", "worse", "broken", "dead-end", "inconclusive"}
FINDING_KINDS = {"insight": "💡", "dead-end": "⛔"}
STALL_AFTER = 3  # results in a row without gain: pointed out to the agent
MAX_METRICS = 12
FIELDS = ("id", "name", "state", "elapsed", "verdict", "hypothesis", "conclusion", "tags", "ended", "job_id")


class ResultError(ValueError):
    pass


# ------------------------------------------------------------------ recording
def snapshot(spec: MetricSpec | None, values: dict[str, str]) -> dict[str, str]:
    """A finished job's key metrics: the agent's critical and important ones (by label), then others from its log."""
    out: dict[str, str] = {}
    if spec:
        for name, label in parse_names(spec.critical) + parse_names(spec.important):
            v = pick(values, name)
            if v is not None:
                out[label] = v
    for k, v in values.items():
        if len(out) >= MAX_METRICS:
            break
        if k not in out and k.split("/")[-1] not in out:
            out[k] = v
    return out


def record_job(store: Store, rec: JobRecord, final: Job, metrics: dict[str, str]) -> ResultRecord:
    """When a job ends: its row (what the agent already wrote about it is kept)."""
    r = store.get_result(rec.job_id) or ResultRecord(id=rec.job_id, job_id=rec.job_id)
    r.project = rec.project
    r.session_id = r.session_id or rec.session_id
    r.name = rec.name or r.name
    r.state = final.state
    r.success = int(final.state == "COMPLETED" and final.exit_code.startswith("0"))
    r.elapsed = final.elapsed
    r.ended = time.time()
    r.metrics = json.dumps({**_metrics(r), **metrics})
    store.save_result(r)
    return r


def annotate(store: Store, rid: str, *, session_id: str | None, project: str | None, job: JobRecord | None = None,
             hypothesis: str | None = None, conclusion: str | None = None, verdict: str | None = None,
             tags: str | None = None, metrics: dict[str, str] | None = None, name: str | None = None,
             hidden: bool | None = None) -> ResultRecord:
    """What the agent says about a result (a job's, before or after it ends, or one without a job)."""
    if verdict is not None and verdict not in VERDICTS:
        raise ResultError(f"verdict must be one of: {', '.join(VERDICTS)}")
    r = store.get_result(rid)
    if r is None:
        if job is None and not rid.startswith("r"):
            raise ResultError(f"no result {rid!r}: give a job id Overseer tracks, or record a new one with "
                              "`ov result add --name ...`")
        r = ResultRecord(id=rid, job_id=job.job_id if job else None, project=job.project if job else project,
                         session_id=session_id, name=(job.name if job else "") or "", state=job.state if job else "")
    for field, value in (("hypothesis", hypothesis), ("conclusion", conclusion), ("verdict", verdict),
                         ("tags", tags), ("name", name)):
        if value is not None:
            setattr(r, field, value.strip())
    if hidden is not None:
        r.hidden = int(hidden)
    if metrics:
        r.metrics = json.dumps({**_metrics(r), **metrics})
    store.save_result(r)
    return r


def add_finding(store: Store, project: str | None, kind: str, text: str, session_id: str | None,
                job_id: str = "") -> FindingRecord:
    if kind not in FINDING_KINDS:
        raise ResultError(f"a finding is an insight or a dead-end, not {kind!r}")
    if not text.strip():
        raise ResultError("a finding needs some text")
    return store.add_finding(FindingRecord(project=project, kind=kind, text=" ".join(text.split()), job_id=job_id,
                                           session_id=session_id))


def parse_metric_args(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not k.strip():
            raise ResultError(f"--metric {it!r}: use name=value, e.g. --metric 'sample quality=good'")
        out[k.strip()] = v.strip()
    return out


def _metrics(r: ResultRecord) -> dict[str, str]:
    try:
        m = json.loads(r.metrics or "{}")
        return m if isinstance(m, dict) else {}
    except ValueError:
        return {}


# ------------------------------------------------------------------ choosing and sorting
def select(results: list[ResultRecord], *, sort: str = "", asc: bool | None = None, verdicts: list[str] = (),
           name: str = "", tag: str = "", state: str = "", since: float | None = None, text: str = "",
           annotated: bool | None = None, limit: int = 0) -> list[ResultRecord]:
    """Filter, then sort: by time (newest first) by default; by a metric (highest first; --asc for loss-like ones),
    or by name, state, verdict, elapsed. Results without the metric go last."""
    rs = list(results)
    if verdicts:
        rs = [r for r in rs if r.verdict in verdicts]
    if name:
        rs = [r for r in rs if fnmatch.fnmatch(r.name, name)]
    if tag:
        rs = [r for r in rs if tag in [t.strip() for t in r.tags.split(",")]]
    if state:
        rs = [r for r in rs if r.state.upper().startswith(state.upper())]
    if since:
        rs = [r for r in rs if (r.ended or r.created) >= since]
    if text:
        low = text.lower()
        rs = [r for r in rs if low in " ".join([r.name, r.hypothesis, r.conclusion, r.tags]).lower()]
    if annotated is not None:
        rs = [r for r in rs if bool(r.verdict or r.conclusion) == annotated]
    key = (sort or "ended").lower()
    if key in ("ended", "time", "date", "newest"):
        rs.sort(key=lambda r: r.ended or r.created, reverse=not asc)
    elif key in ("name", "state", "verdict", "id"):
        order = list(VERDICTS)
        rs.sort(key=lambda r: (order.index(r.verdict) if r.verdict in order else 99) if key == "verdict"
                else getattr(r, key) or "", reverse=bool(asc is False))
    elif key == "elapsed":
        rs.sort(key=lambda r: slurm_seconds(r.elapsed) or 0, reverse=not asc)
    else:  # a metric
        def val(r: ResultRecord):
            v = pick(_metrics(r), sort)
            return number(v)
        have = [r for r in rs if val(r) is not None]
        rs = sorted(have, key=val, reverse=not asc) + [r for r in rs if val(r) is None]
    return rs[:limit] if limit else rs


def stall(results: list[ResultRecord]) -> list[str]:
    """The verdicts of the latest results in a row that brought no gain (since the last promising or baseline)."""
    out = []
    for r in sorted((r for r in results if r.verdict), key=lambda r: r.ended or r.created, reverse=True):
        if r.verdict not in NO_GAIN:
            break
        out.append(r.verdict)
    return out


# ------------------------------------------------------------------ output
def metric_columns(results: list[ResultRecord], wanted: int = 4) -> list[str]:
    """The metrics most results share, in their recorded order (critical ones first)."""
    counts: dict[str, int] = {}
    for r in results:
        for k in _metrics(r):
            counts[k] = counts.get(k, 0) + 1
    order: list[str] = []
    for r in results:
        for k in _metrics(r):
            if k not in order:
                order.append(k)
    return sorted(order, key=lambda k: -counts[k])[:wanted]


def _cell(r: ResultRecord, col: str, width: int = 0) -> str:
    if col == "verdict":
        v = f"{VERDICTS.get(r.verdict, '')} {r.verdict}".strip()
    elif col == "ended":
        v = time.strftime("%m-%d %H:%M", time.localtime(r.ended)) if r.ended else ""
    elif col in FIELDS:
        v = str(getattr(r, col) or "")
    else:
        v = str(pick(_metrics(r), col) or "")
    v = " ".join(v.split())
    return v if not width or len(v) <= width else v[:width - 1] + "…"


def default_columns(results: list[ResultRecord]) -> list[str]:
    return ["id", "name", "state", "verdict", *metric_columns(results), "conclusion"]


def render(results: list[ResultRecord], fmt: str = "table", columns: list[str] | None = None) -> str:
    cols = columns or default_columns(results)
    if fmt == "json":
        return json.dumps([{**{f: getattr(r, f) for f in FIELDS}, "project": r.project, "metrics": _metrics(r)}
                           for r in results], indent=1)
    if fmt == "csv":
        metric_cols = [c for c in metric_columns(results, wanted=50)]
        buf = io.StringIO()
        w = csv.writer(buf)
        base = ["id", "job_id", "name", "state", "elapsed", "ended", "verdict", "hypothesis", "conclusion", "tags"]
        w.writerow(base + metric_cols)
        for r in results:
            w.writerow([_cell(r, c) for c in base] + [_metrics(r).get(c, "") for c in metric_cols])
        return buf.getvalue()
    if fmt == "md":
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in results:
            lines.append("| " + " | ".join(_cell(r, c, 80 if c in ("conclusion", "hypothesis") else 40)
                                           .replace("|", "/") for c in cols) + " |")
        return "\n".join(lines)
    # plain table, fixed width
    rows = [[_cell(r, c, 50 if c in ("conclusion", "hypothesis") else 24) for c in cols] for r in results]
    widths = [max([len(c)] + [len(row[i]) for row in rows]) for i, c in enumerate(cols)]
    out = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(cols)).rstrip()]
    out += ["  ".join(row[i].ljust(widths[i]) for i in range(len(cols))).rstrip() for row in rows]
    return "\n".join(out)


def finding_lines(findings: list[FindingRecord]) -> list[str]:
    return [f"{FINDING_KINDS.get(f.kind, '•')} F{f.id} {f.kind}: {f.text}"
            + (f" (job {f.job_id})" if f.job_id else "") for f in findings]


def context(store: Store, project: str | None, session_id: str | None = None, n: int = 8,
            findings_chars: int = 5000) -> str:
    """For the agent when it starts or is woken in a project: its findings (all, or the newest that fit), its latest
    results, a run of results without gain, and its own results still waiting for a verdict."""
    parts = []
    fs = store.findings(project)
    if fs:
        lines = finding_lines(fs)
        kept: list[str] = []
        for line in reversed(lines):  # dead ends matter most when recent; keep the newest that fit
            if sum(len(x) + 1 for x in kept) + len(line) > findings_chars:
                break
            kept.insert(0, line)
        more = f"\n({len(lines) - len(kept)} older: `ov findings`)" if len(kept) < len(lines) else ""
        parts.append("<findings>\n" + "\n".join(kept) + more + "\n</findings>")
    rs = store.results(project)
    if rs:
        latest = select(rs, limit=n)
        parts.append(f"<results total='{len(rs)}' shown='latest {len(latest)}'>\n"
                     + render(latest, "table") + "\n</results>  (`ov results` sorts and filters them all)")
        run = stall(rs)
        if len(run) >= STALL_AFTER:
            parts.append(f"Your last {len(run)} results brought no gain ({', '.join(run)}). Before another variation "
                         "of the same idea, consider a materially different one, check the findings, or ask the "
                         "researcher.")
        mine = [r for r in rs if session_id and r.session_id == session_id and r.ended and not r.verdict]
        if mine:
            parts.append("Results of yours without a verdict yet: " + ", ".join(f"{r.id} ({r.name})" for r in mine[-8:])
                         + ". Record what each showed: `ov result <id> --verdict ... --conclusion \"...\"`.")
    return "\n\n".join(parts)
