"""What the cluster has free now, and when your queued jobs should start.

For the agents (`ov cluster`: pick a partition and a size that starts soon instead of guessing), for you (/cluster),
and on queued jobs' cards ("starts in ~1h05m", SLURM's own estimate). Read from SLURM itself, nothing is
configured: `scontrol show node` (each node's GPUs, CPUs and memory, and how much of them is allocated), `sinfo`
(partitions and their time limits), `squeue --start` (SLURM's estimated start of your pending jobs) and `sshare`
(your fairshare: how much priority your recent usage left you).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from .notify import code, esc
from .slurm import array_parent, human_seconds, parse_scontrol

# Node states in which nothing new starts on a node.
UNUSABLE = ("DOWN", "DRAIN", "FAIL", "MAINT", "NOT_RESPONDING", "POWERED_DOWN", "POWERING", "REBOOT", "RESERVED",
            "INVAL", "FUTURE", "UNKNOWN")
CACHE_SECONDS = 120


def gpu_count(gres: str) -> tuple[int, list[str]]:
    """'gpu:a100:4(S:0-1),gpu:v100:2' -> (6, ['a100', 'v100']); '(null)' -> (0, [])."""
    n, types = 0, []
    for part in re.sub(r"\([^)]*\)", "", gres or "").split(","):
        bits = part.strip().split(":")
        if not bits or bits[0] != "gpu":
            continue
        try:
            n += int(bits[-1])
        except ValueError:
            n += 1
        if len(bits) >= 3 and bits[1] not in types:
            types.append(bits[1])
    return n, types


def gpus_allocated(tres: str) -> int:
    """'cpu=8,mem=64G,gres/gpu=2,gres/gpu:a100=2' -> 2 (the generic count; else the sum of the typed ones)."""
    m = re.search(r"(?:^|,)gres/gpu=(\d+)", tres or "")
    if m:
        return int(m.group(1))
    return sum(int(x) for x in re.findall(r"gres/gpu:[^=,]+=(\d+)", tres or ""))


def _int(x: str | None) -> int:
    try:
        return int(float(x or 0))
    except ValueError:
        return 0


@dataclass
class Partition:
    name: str
    default: bool = False
    avail: str = "up"
    time_limit: str = ""
    nodes: int = 0
    usable: int = 0  # nodes where something new can start
    idle: int = 0  # usable nodes with nothing running
    gpus: int = 0
    gpus_free: int = 0  # on usable nodes
    max_free_gpus: int = 0  # on a single node: the biggest single-node GPU job that could start now
    gpu_types: list[str] = field(default_factory=list)
    cpus: int = 0
    cpus_free: int = 0
    max_free_mem_gb: float = 0.0  # on a single usable node


def partitions(nodes: list[dict[str, str]], sinfo: str = "") -> list[Partition]:
    """Per partition, from `scontrol show node -o` records and `sinfo -h -o '%P|%a|%l'` lines."""
    parts: dict[str, Partition] = {}
    for line in sinfo.splitlines():
        cols = line.strip().split("|")
        if len(cols) >= 3 and cols[0]:
            name = cols[0].rstrip("*")
            parts[name] = Partition(name=name, default=cols[0].endswith("*"), avail=cols[1], time_limit=cols[2])
    for nd in nodes:
        state = nd.get("State", "").upper()
        usable = not any(s in state for s in UNUSABLE)
        total, types = gpu_count(nd.get("Gres", ""))
        free = max(0, total - gpus_allocated(nd.get("AllocTRES", ""))) if usable else 0
        cpus, cpus_alloc = _int(nd.get("CPUTot")), _int(nd.get("CPUAlloc"))
        mem_free = max(0, _int(nd.get("RealMemory")) - _int(nd.get("AllocMem"))) / 1024 if usable else 0.0
        for name in filter(None, nd.get("Partitions", "").split(",")):
            p = parts.setdefault(name, Partition(name=name))
            p.nodes += 1
            p.usable += usable
            p.idle += usable and cpus_alloc == 0
            p.gpus += total
            p.gpus_free += free
            p.max_free_gpus = max(p.max_free_gpus, free)
            p.gpu_types += [t for t in types if t not in p.gpu_types]
            p.cpus += cpus
            p.cpus_free += max(0, cpus - cpus_alloc) if usable else 0
            p.max_free_mem_gb = max(p.max_free_mem_gb, mem_free)
    return [p for p in parts.values() if p.nodes]


def start_estimates(squeue: str) -> dict[str, tuple[str, str]]:
    """`squeue -h -t PENDING --start -o '%i|%S|%r'` -> {job id (an array's: its own id): (estimated start, reason)};
    for an array, its earliest task."""
    out: dict[str, tuple[str, str]] = {}
    for line in squeue.splitlines():
        cols = line.strip().split("|")
        if len(cols) < 3 or not cols[0]:
            continue
        jid, start, reason = array_parent(cols[0]), cols[1].strip(), cols[2].strip()
        known = out.get(jid)
        if not known or (start[:1].isdigit() and (not known[0][:1].isdigit() or start < known[0])):
            out[jid] = (start, reason)
    return out


def fairshare(sshare: str) -> list[tuple[str, str]]:
    """`sshare -h -P -U -o Account,FairShare` -> [(account, fairshare 0..1)]."""
    out = []
    for line in sshare.splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) >= 2 and cols[0] and cols[1]:
            out.append((cols[0], cols[1]))
    return out


def until(start: str, now: str) -> str:
    """'in 1h05m' from an estimated start and the cluster's clock (both its local time, ISO); "" if unknown."""
    try:
        delta = (datetime.fromisoformat(start) - datetime.fromisoformat(now)).total_seconds()
    except ValueError:
        return ""
    return "any moment now" if delta < 60 else f"in ~{human_seconds(int(delta))}"


class Cluster:
    def __init__(self, hpc):
        self.hpc = hpc
        self._cache: tuple[float, dict] | None = None

    async def snapshot(self, fresh: bool = False) -> dict:
        """{partitions, estimates, fairshare, clock}: kept for a couple of minutes (agents ask often)."""
        if not fresh and self._cache and time.time() - self._cache[0] < CACHE_SECONDS:
            return self._cache[1]
        nodes = await self.hpc.nodes()
        snap = {
            "partitions": partitions(nodes, await self.hpc.sinfo_partitions()),
            "estimates": start_estimates(await self.hpc.pending_starts()),
            "fairshare": fairshare(await self.hpc.sshare()),
            "clock": await self.hpc.clock(),
        }
        self._cache = (time.time(), snap)
        return snap

    async def estimates(self) -> tuple[dict[str, tuple[str, str]], str]:
        """Only the start estimates of your pending jobs (for queued cards), and the cluster's clock."""
        return start_estimates(await self.hpc.pending_starts()), await self.hpc.clock()

    def text(self, snap: dict, names: dict[str, str] | None = None, partition: str | None = None,
             html: bool = False) -> str:
        """The snapshot for an agent (plain) or for Telegram (html); `names`: job id -> name of your tracked jobs."""
        e = esc if html else (lambda x: x)
        b = (lambda x: f"<b>{x}</b>") if html else (lambda x: x)
        parts = [p for p in snap["partitions"] if not partition or p.name == partition]
        if partition and not parts:
            known = ", ".join(p.name for p in snap["partitions"])
            return f"no partition {partition!r}" + (f" (partitions: {known})" if known else "")
        clock = snap.get("clock", "")
        lines = [b("Cluster now") + (f" ({e(clock.replace('T', ' ')[:16])} cluster time)" if clock else "")]
        if not parts:
            lines.append("(SLURM didn't list any nodes: `scontrol show node` may not be allowed here)")
        for p in sorted(parts, key=lambda p: (-p.gpus, p.name)):
            head = f"{b(e(p.name))}{' (default)' if p.default else ''}"
            if p.avail and p.avail != "up":
                head += f" · {e(p.avail)}"
            bits = [f"{p.usable}/{p.nodes} nodes up" + (f", {p.idle} idle" if p.idle else "")]
            if p.gpus:
                kind = "/".join(p.gpu_types)
                bits.append(f"GPUs {p.gpus_free}/{p.gpus} free" + (f" ({e(kind)})" if kind else "")
                            + (f", at most {p.max_free_gpus} on one node" if p.gpus_free else ""))
            bits.append(f"CPUs {p.cpus_free}/{p.cpus} free")
            if p.max_free_mem_gb:
                bits.append(f"≤{p.max_free_mem_gb:.0f} GB RAM free on one node")
            if p.time_limit:
                bits.append(f"max time {e(p.time_limit)}")
            lines.append(f"{head}: " + " · ".join(bits))
        mine = snap.get("estimates", {})
        if mine:
            lines.append(b("Your queued jobs") + " (SLURM's estimated start):")
            for jid, (start, reason) in sorted(mine.items()):
                when = (start.replace("T", " ")[:16] + (f" ({until(start, clock)})" if until(start, clock) else "")
                        if start[:1].isdigit() else "no estimate yet")
                name = (names or {}).get(jid, "")
                lines.append(f"  {code(jid) if html else jid}{' ' + e(name) if name else ''}: {e(when)}"
                             + (f" · waiting for {e(reason)}" if reason and reason != "None" else ""))
        if snap.get("fairshare"):
            fs = ", ".join(f"{e(a)} {e(v)}" for a, v in snap["fairshare"])
            lines.append(f"Fairshare: {fs} (1 = full priority; lower = you used a lot lately, so you wait longer)")
        return "\n".join(lines)
