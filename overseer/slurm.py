"""Parsing of squeue / sacct / scontrol output. Pure functions, no I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ACTIVE_STATES = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED",
    "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD",
}
SUCCESS_STATES = {"COMPLETED"}
FAILURE_STATES = {
    "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
    "PREEMPTED", "CANCELLED", "REVOKED", "SPECIAL_EXIT",
}
TERMINAL_STATES = SUCCESS_STATES | FAILURE_STATES

STATE_EMOJI = {
    "PENDING": "⏳", "RUNNING": "🏃", "COMPLETING": "🏁", "COMPLETED": "✅", "FAILED": "❌",
    "TIMEOUT": "⌛", "OUT_OF_MEMORY": "💥", "CANCELLED": "🚫", "NODE_FAIL": "🔌",
    "PREEMPTED": "🔁", "BOOT_FAIL": "🔌", "DEADLINE": "⌛",
}

SQUEUE_FIELDS = ["job_id", "name", "state", "partition", "elapsed", "time_limit", "nodes", "reason", "gres", "submit_time", "start_time"]
SQUEUE_FORMAT = "%i|%j|%T|%P|%M|%l|%D|%R|%b|%V|%S"

SACCT_LIST_FIELDS = ["job_id", "name", "state", "exit_code", "elapsed", "time_limit", "start_time", "end_time", "partition", "nodelist", "alloc_tres", "workdir"]
SACCT_LIST_FORMAT = "JobID,JobName%200,State,ExitCode,Elapsed,Timelimit,Start,End,Partition,NodeList,AllocTRES,WorkDir%400"

SACCT_STEP_FIELDS = ["job_id", "name", "state", "exit_code", "elapsed", "max_rss", "req_mem", "alloc_tres", "nodelist"]
SACCT_STEP_FORMAT = "JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES,NodeList"


def normalize_state(state: str) -> str:
    s = state.strip().upper()
    s = s.split(" ", 1)[0]  # "CANCELLED by 1234"
    return s.rstrip("+")


@dataclass
class Job:
    job_id: str
    name: str = ""
    state: str = ""
    partition: str = ""
    elapsed: str = ""
    time_limit: str = ""
    nodes: str = ""
    reason: str = ""  # squeue: pending reason or node list
    gres: str = ""
    submit_time: str = ""
    start_time: str = ""
    end_time: str = ""
    exit_code: str = ""
    nodelist: str = ""
    alloc_tres: str = ""
    workdir: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


def _split(line: str, n_fields: int, name_idx: int) -> list[str] | None:
    """Split a '|'-separated line; extra separators are assumed to belong to the name field."""
    parts = line.split("|")
    if len(parts) < n_fields:
        return None
    extra = len(parts) - n_fields
    if extra:
        merged = "|".join(parts[name_idx: name_idx + extra + 1])
        parts = parts[:name_idx] + [merged] + parts[name_idx + extra + 1:]
    return parts


def parse_squeue(output: str) -> list[Job]:
    jobs = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = _split(line, len(SQUEUE_FIELDS), 1)
        if not parts:
            continue
        d = dict(zip(SQUEUE_FIELDS, (p.strip() for p in parts)))
        d["state"] = normalize_state(d["state"])
        jobs.append(Job(**d))
    return jobs


def parse_sacct_list(output: str) -> list[Job]:
    """Parse ``sacct -X -n -P`` output (allocation lines only)."""
    jobs = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = _split(line, len(SACCT_LIST_FIELDS), 1)
        if not parts:
            continue
        d = dict(zip(SACCT_LIST_FIELDS, (p.strip() for p in parts)))
        if "." in d["job_id"]:
            continue  # a step
        d["state"] = normalize_state(d["state"])
        jobs.append(Job(**d))
    return jobs


def parse_sacct_steps(output: str) -> list[dict[str, str]]:
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < len(SACCT_STEP_FIELDS):
            continue
        row = dict(zip(SACCT_STEP_FIELDS, (p.strip() for p in parts)))
        row["state"] = normalize_state(row["state"])
        rows.append(row)
    return rows


_SCONTROL_KV = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=")


def parse_scontrol(output: str) -> dict[str, str]:
    """Parse ``scontrol show job -o <id>`` (one line of Key=Value pairs, values may contain spaces)."""
    line = output.strip().splitlines()[0] if output.strip() else ""
    out: dict[str, str] = {}
    matches = list(_SCONTROL_KV.finditer(line))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        out[m.group(1)] = line[m.end():end].strip()
    return out


def parse_mem_to_mb(value: str) -> float | None:
    """Parse '1234K', '3.5G', '1200M' (sacct MaxRSS/ReqMem) into MB."""
    m = re.match(r"^([\d.]+)([KMGTP]?)", value.strip())
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2)  # bare numbers are bytes
    mult = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3, "": 1 / 1024 / 1024}[unit]
    return num * mult


def summarize_steps(rows: list[dict[str, str]]) -> dict[str, str]:
    """Pick the interesting numbers out of per-step sacct rows."""
    summary: dict[str, str] = {}
    peak = None
    for r in rows:
        mb = parse_mem_to_mb(r.get("max_rss", "")) if r.get("max_rss") else None
        if mb is not None and (peak is None or mb > peak):
            peak = mb
        if r["job_id"].endswith(".batch") and r.get("exit_code"):
            summary["batch_exit"] = r["exit_code"]
        if "." not in r["job_id"]:
            summary["req_mem"] = r.get("req_mem", "")
            summary["alloc_tres"] = r.get("alloc_tres", "")
    if peak is not None:
        summary["max_rss"] = f"{peak / 1024:.1f}G" if peak >= 1024 else f"{peak:.0f}M"
    failed = [r for r in rows if r["state"] in FAILURE_STATES]
    if failed:
        summary["failed_steps"] = ", ".join(f"{r['job_id']}={r['state']}({r['exit_code']})" for r in failed[:5])
    return summary


def gpus_from_tres(tres: str) -> str:
    m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", tres or "")
    return m.group(1) if m else ""


def slurm_seconds(t: str) -> int | None:
    """SLURM durations ('1:58', '4:00:00', '1-02:00:00') in seconds; None for UNLIMITED etc."""
    t = (t or "").strip()
    if not t or not t[0].isdigit():
        return None
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    try:
        parts = [int(float(x)) for x in t.split(":")]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def human_seconds(sec: int) -> str:
    if sec < 3600:
        return f"{sec // 60}m"
    if sec < 86400:
        return f"{sec // 3600}h{sec % 3600 // 60:02d}m"
    return f"{sec // 86400}d{sec % 86400 // 3600}h"


def format_gpu_stats(csv: str) -> list[str]:
    """nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total CSV -> readable lines."""
    out = []
    for line in csv.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 5:
            out.append(line.strip())
            continue
        idx, name, util, used, total = cols
        try:
            mem = f"{float(used.split()[0]) / 1024:.1f}/{float(total.split()[0]) / 1024:.0f} GB"
        except (ValueError, IndexError):
            mem = f"{used}/{total}"
        out.append(f"GPU {idx} {name.replace('NVIDIA ', '')}: {util.replace(' ', '')} busy · {mem}")
    return out
