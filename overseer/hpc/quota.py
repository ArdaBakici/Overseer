"""Disk quotas on the HPC: space, and the number of files (inodes), which is the one that runs out.

Overseer runs slurm.quota_command (e.g. `myquota`) at startup and every slurm.quota_check_hours. The limits go
into the agent's instructions; when a filesystem's space or file count passes quota_warn_percent,
you get a Telegram warning and the agent is told at the start of its next request.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# "    /scratch      678GB     5000GB ( 14%)       301       500 ( 60%)"
_ROW = re.compile(r"^\s*(/\S*)\s+([\d.]+\s*[KMGTPE]?i?B?)\s+([\d.]+\s*[KMGTPE]?i?B?)\s*\(\s*(\d+)\s*%\s*\)"
                  r"\s+([\d.]+)\s+([\d.]+)\s*\(\s*(\d+)\s*%\s*\)", re.M)


@dataclass
class Quota:
    fs: str
    used: str  # e.g. "678GB"
    limit: str  # e.g. "5000GB"
    pct: int
    files: int  # number of files (inodes) used
    files_limit: int
    files_pct: int

    def to_dict(self) -> dict:
        return asdict(self)


def parse_quota(text: str) -> list[Quota]:
    """The rows of a quota table like `myquota`'s. File counts in thousands ("# FILES (1000's)") are expanded."""
    scale = 1000 if re.search(r"1000'?s|thousands", text or "", re.I) else 1
    rows = []
    for m in _ROW.finditer(text or ""):
        fs, used, limit, pct, files, files_limit, files_pct = m.groups()
        rows.append(Quota(fs, used.replace(" ", ""), limit.replace(" ", ""), int(pct), int(float(files) * scale),
                          int(float(files_limit) * scale), int(files_pct)))
    return rows


def _k(n: int) -> str:
    return f"{n / 1000:g}k" if n >= 1000 else str(n)


def limits_line(rows: list[Quota]) -> str:
    """'/home: 50GB, 500k files; /scratch: ...' (limits only: stable, for the system prompt)."""
    return "; ".join(f"`{q.fs}`: {q.limit} and {_k(q.files_limit)} files" for q in rows if q.files_limit)


def over(rows: list[Quota], percent: int) -> list[str]:
    """What is at or above `percent` of its quota, e.g. "/scratch: 425k of 500k files (85%)"."""
    out = []
    for q in rows:
        if q.files_limit and q.files_pct >= percent:
            out.append(f"{q.fs}: {_k(q.files)} of {_k(q.files_limit)} files ({q.files_pct}%)")
        if q.pct >= percent:
            out.append(f"{q.fs}: {q.used} of {q.limit} ({q.pct}%)")
    return out
