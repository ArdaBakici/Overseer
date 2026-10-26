"""What matters in a job, as the agent says (`ov metrics set`), and alerts on it.

A metric spec belongs to one job, to the jobs whose names match a pattern (a sweep), or to a project
(its default). It names the critical metrics (the job card's first line) and the important ones (its
second line), how to show progress ("step/5000000"), optionally a quick command on the HPC that prints
the values (name=value lines or a JSON object; otherwise they come from the job's log), and alerts:

    score < 1 after 2h        a comparison, checked once the job has run that long
    loss is nan               NaN or inf
    step stalled 30m          the value hasn't changed for that long

An alert fires once per job: it pings the researcher, and with `--wake-on` it also wakes the agent.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
import shlex
import time
from dataclasses import dataclass

from ..core.store import JobRecord, MetricSpec, Store

COMMAND_TIMEOUT = 20  # seconds; the command runs on the login node at every card refresh
COMMAND_OUTPUT = 16_000

_ALERT_RE = re.compile(
    r"^\s*(?P<metric>[\w./:\-\[\]]+)\s*(?:(?P<op><=|>=|==|!=|<|>)\s*(?P<value>[-+\d.eE]+|nan|inf)"
    r"|\s+(?P<isnan>is\s+nan)|\s+stalled\s+(?P<stall>\S+))"
    r"(?:\s+after\s+(?P<after>\S+))?\s*$", re.I)


class SpecError(ValueError):
    pass


def _seconds(text: str) -> float | None:
    from ..core.autopilot import parse_duration
    d = parse_duration(text)
    return d if d and d > 0 else None


@dataclass
class Alert:
    text: str
    metric: str
    op: str = ""  # <, <=, ...; "nan"; "stalled"
    value: float = 0.0
    stall: float = 0.0  # seconds
    after: float = 0.0  # seconds of the job's elapsed time before it's checked
    wake: bool = False


def parse_alert(text: str, wake: bool = False) -> Alert:
    m = _ALERT_RE.match(text or "")
    if not m:
        raise SpecError(f"can't read the alert {text!r}: use e.g. 'score < 1 after 2h', 'loss is nan' or "
                        "'step stalled 30m'")
    a = Alert(text=" ".join(text.split()), metric=m.group("metric"), wake=wake)
    if m.group("after"):
        after = _seconds(m.group("after"))
        if after is None:
            raise SpecError(f"{m.group('after')!r} isn't a duration (e.g. 30m, 2h)")
        a.after = after
    if m.group("isnan"):
        a.op = "nan"
    elif m.group("stall"):
        stall = _seconds(m.group("stall"))
        if stall is None:
            raise SpecError(f"{m.group('stall')!r} isn't a duration (e.g. 30m, 2h)")
        a.op, a.stall = "stalled", stall
    else:
        a.op, a.value = m.group("op"), float(m.group("value"))
    return a


def alerts_of(spec: MetricSpec | None) -> list[Alert]:
    try:
        items = json.loads(spec.alerts or "[]") if spec else []
    except ValueError:
        return []
    out = []
    for it in items:
        try:
            out.append(parse_alert(it.get("when", ""), bool(it.get("wake"))))
        except SpecError:
            continue
    return out


def parse_names(text: str) -> list[tuple[str, str]]:
    """"crl/train_episode/score=score, train/loss/value" -> [(metric, label), ...]."""
    out = []
    for part in (text or "").split(","):
        name, _, label = part.strip().partition("=")
        if name.strip():
            out.append((name.strip(), (label or name.split("/")[-1]).strip()))
    return out


def check_progress(text: str) -> str:
    if text and not re.fullmatch(r"[\w./:\-\[\]]+\s*/\s*[\d.eE+]+", text.strip()):
        raise SpecError(f"--progress {text!r}: use metric/total, e.g. step/5000000")
    return text.strip()


# ------------------------------------------------------------------ which spec applies
def resolve(store: Store, rec: JobRecord) -> MetricSpec | None:
    """The job's own spec, else the newest pattern matching its name, else its project's."""
    spec = store.get_metric_spec("job", rec.job_id)
    if spec:
        return spec
    for p in store.metric_specs("pattern"):  # newest first
        if rec.name and fnmatch.fnmatch(rec.name, p.key):
            return p
    return store.get_metric_spec("project", rec.project) if rec.project else None


# ------------------------------------------------------------------ values
def parse_output(text: str) -> dict[str, str]:
    """A command's output: one JSON object (nested keys joined with /), or `name=value` / `name: value` lines."""
    text = (text or "").strip()
    if not text:
        return {}
    last = text.splitlines()[-1].strip()
    for candidate in (text, last):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            flat: dict[str, str] = {}

            def walk(prefix: str, v) -> None:
                if isinstance(v, dict):
                    for k, x in v.items():
                        walk(f"{prefix}/{k}" if prefix else str(k), x)
                elif not isinstance(v, list):
                    flat[prefix] = _fmt(v)
            walk("", obj)
            return flat
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([\w./:\-\[\] ]+?)\s*[=:]\s*(\S.*?)\s*$", line)
        if m:
            out[m.group(1).strip()] = m.group(2)
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def pick(values: dict[str, str], name: str) -> str | None:
    """A metric's value: by exact name, else by a name ending in it ("score" finds "crl/train_episode/score")."""
    if name in values:
        return values[name]
    low = {k.lower(): v for k, v in values.items()}
    n = name.lower()
    if n in low:
        return low[n]
    for k, v in low.items():
        if k.endswith("/" + n):
            return v
    return None


def number(v: str | None) -> float | None:
    if v is None:
        return None
    m = re.match(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf)", str(v), re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def compact(v: str) -> str:
    """2850000 -> 2.85M (only plain big integers)."""
    x = number(v)
    if x is not None and x.is_integer() and abs(x) >= 10_000 and re.fullmatch(r"\s*[-+]?\d+\s*", str(v)):
        return f"{x / 1e6:.3g}M" if abs(x) >= 1e6 else f"{x / 1e3:.3g}k"
    return str(v)


async def run_command(hpc, rec: JobRecord, command: str) -> tuple[dict[str, str], str]:
    """Run the spec's command on the HPC (placeholders filled in); (values, error)."""
    q = shlex.quote
    filled = (command.replace("{job_id}", q(rec.job_id)).replace("{workdir}", q(rec.workdir or "."))
              .replace("{stdout}", q(rec.stdout or "/dev/null")).replace("{name}", q(rec.name or "")))
    try:
        res = await hpc.run(hpc._slurm(["bash", "-c", filled]), timeout=COMMAND_TIMEOUT, max_output=COMMAND_OUTPUT)
    except Exception as e:  # SSH trouble, timeout
        return {}, f"{type(e).__name__}: {e}"[:300]
    values = parse_output(res.stdout)
    if not res.ok and not values:
        return {}, f"exit {res.returncode}: {(res.stderr or res.stdout).strip()[-300:]}"
    if not values:
        return {}, "no values in its output (expected name=value lines or a JSON object)"
    return values, ""


def lines(spec: MetricSpec | None, values: dict[str, str], auto: dict[str, str]) -> tuple[str, str, str]:
    """(critical line, important line, progress text) for a card, plain text; without a spec, the 3 most
    informative metrics found in the log go on the important line."""
    if not spec or not (spec.critical or spec.important):
        shown = list(auto.items())[:3]
        return "", " · ".join(f"{k.split('/')[-1]} {compact(v)}" for k, v in shown), _progress(spec, {**auto, **values})

    def fmt(names: list[tuple[str, str]]) -> str:
        parts = []
        for name, label in names:
            v = pick(values, name)
            if v is None:
                v = pick(auto, name)
            parts.append(f"{label} {compact(v) if v is not None else '?'}")
        return " · ".join(parts)
    return fmt(parse_names(spec.critical)), fmt(parse_names(spec.important)), _progress(spec, {**auto, **values})


def _progress(spec: MetricSpec | None, values: dict[str, str]) -> str:
    if not spec or not spec.progress:
        return ""
    name, _, total = spec.progress.partition("/")
    v, t = number(pick(values, name.strip())), number(total)
    if v is None or not t:
        return ""
    return f"{name.strip().split('/')[-1]} {compact(str(int(v)) if v.is_integer() else str(v))} / {compact(total.strip())} ({100 * v / t:.0f}%)"


# ------------------------------------------------------------------ alerts
def check_alerts(store: Store, rec: JobRecord, spec: MetricSpec | None, values: dict[str, str],
                 elapsed: float | None) -> list[tuple[Alert, str]]:
    """Alerts that fire now (each at most once per job): [(alert, the value seen)]."""
    alerts = alerts_of(spec)
    if not alerts:
        return []
    key = f"mstate:{rec.job_id}"
    try:
        state = json.loads(store.get(key) or "{}")
    except ValueError:
        state = {}
    fired: list[str] = state.setdefault("fired", [])
    last: dict = state.setdefault("last", {})
    now = time.time()
    out = []
    for a in alerts:
        raw = pick(values, a.metric)
        if a.op == "stalled" and raw is not None:  # remember when each watched value last changed
            prev = last.get(a.metric)
            if not prev or prev[0] != raw:
                last[a.metric] = [raw, now]
        if a.text in fired or raw is None:
            continue
        if a.after and (elapsed is None or elapsed < a.after):
            continue
        x = number(raw)
        hit = False
        if a.op == "nan":
            hit = x is not None and (math.isnan(x) or math.isinf(x))
        elif a.op == "stalled":
            since = last.get(a.metric, [raw, now])[1]
            hit = now - since >= a.stall
        elif x is not None and not math.isnan(x):
            hit = {"<": x < a.value, "<=": x <= a.value, ">": x > a.value, ">=": x >= a.value,
                   "==": x == a.value, "!=": x != a.value}[a.op]
        if hit:
            fired.append(a.text)
            out.append((a, raw))
    store.set(key, json.dumps(state))
    return out


def fired_alerts(store: Store, rec: JobRecord) -> list[str]:
    try:
        return json.loads(store.get(f"mstate:{rec.job_id}") or "{}").get("fired", [])
    except ValueError:
        return []
