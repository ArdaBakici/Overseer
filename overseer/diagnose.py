"""Turning raw job data (state, sacct, logs) into human reports.

Heuristics work without any LLM (the decisive error line on a failed job's card, the categories in the
agent's report); an LLM can add a one-line progress status to status messages (llm_status_updates).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from typing import Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass
class Finding:
    category: str
    hint: str
    evidence: str = ""


PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("GPU out of memory", re.compile(r"CUDA out of memory|torch\.(cuda\.)?OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|RESOURCE_EXHAUSTED", re.I),
     "Reduce batch size / sequence length, enable gradient checkpointing or mixed precision, or request a larger GPU."),
    ("Host out of memory", re.compile(r"oom[-_ ]kill|Out Of Memory|Killed\s*$|MemoryError|exceeded memory limit", re.I | re.M),
     "Request more --mem, fewer DataLoader workers, or avoid loading the full dataset into RAM."),
    ("Time limit", re.compile(r"DUE TO TIME LIMIT|TIME LIMIT|time limit exceeded", re.I),
     "Increase --time, resume from the last checkpoint, or make checkpointing more frequent."),
    ("Missing Python module", re.compile(r"ModuleNotFoundError: No module named '([^']+)'|ImportError: .*"),
     "Activate the right environment/module in the batch script or install the package."),
    ("File not found", re.compile(r"FileNotFoundError: .*|No such file or directory", re.I),
     "Check paths relative to the job's working directory; data paths may differ on compute nodes."),
    ("NCCL / distributed error", re.compile(r"NCCL (error|WARN)|ProcessGroupNCCL|Watchdog caught collective operation timeout|torch\.distributed\.\w+Error", re.I),
     "Check world size vs allocated GPUs, NCCL env vars, and that all ranks run the same code path."),
    ("CUDA error", re.compile(r"CUDA error: .*|device-side assert triggered|illegal memory access|no CUDA-capable device", re.I),
     "Run with CUDA_LAUNCH_BLOCKING=1 to localise; often an index out of range in an embedding/loss."),
    ("Disk quota / space", re.compile(r"Disk quota exceeded|No space left on device", re.I),
     "Clean up old checkpoints or write outputs to scratch space."),
    ("NaN / divergence", re.compile(r"\bloss[^\n]{0,20}\bnan\b|\bnan loss\b|Loss is NaN|inf or nan", re.I),
     "Lower the learning rate, add gradient clipping, check for bad data / fp16 overflow."),
    ("Segfault", re.compile(r"Segmentation fault|SIGSEGV|core dumped", re.I),
     "Often a library/driver mismatch or a DataLoader worker crash; try num_workers=0 to isolate."),
    ("Command not found", re.compile(r"command not found", re.I),
     "Load the needed module (module load ...) or activate the environment before calling it."),
    ("Permission denied", re.compile(r"Permission denied", re.I),
     "The job tried to read/write a path it has no access to."),
    ("Preempted / node failure", re.compile(r"PREEMPT|NODE_FAIL|node failure", re.I),
     "Not your code's fault; resubmit (consider --requeue and checkpointing)."),
]

STATE_FINDINGS = {
    "TIMEOUT": Finding("Time limit", "The job hit its --time limit. Increase it or resume from a checkpoint."),
    "OUT_OF_MEMORY": Finding("Host out of memory", "SLURM killed the job for exceeding its memory request (--mem)."),
    "NODE_FAIL": Finding("Preempted / node failure", "The node failed. Resubmitting usually works."),
    "PREEMPTED": Finding("Preempted / node failure", "The job was preempted by a higher-priority job."),
    "BOOT_FAIL": Finding("Preempted / node failure", "The node failed to boot. Resubmit."),
}


def find_issues(log_text: str, state: str = "") -> list[Finding]:
    found: dict[str, Finding] = {}
    if state in STATE_FINDINGS:
        f = STATE_FINDINGS[state]
        found[f.category] = f
    for category, rx, hint in PATTERNS:
        if category in found:
            continue
        m = None
        for m in rx.finditer(log_text):
            pass  # keep the last occurrence; it's usually the fatal one
        if m:
            found[category] = Finding(category, hint, m.group(0).strip()[:200])
    return list(found.values())


def extract_traceback(log_text: str, max_lines: int = 40) -> str:
    idx = log_text.rfind("Traceback (most recent call last):")
    if idx < 0:
        return ""
    lines = log_text[idx:].splitlines()
    out = [lines[0]]
    for ln in lines[1:]:
        out.append(ln)
        # The exception line is the first non-indented line after the frames.
        if ln and not ln.startswith((" ", "\t")):
            break
    if len(out) > max_lines:
        out = out[:3] + ["  ..."] + out[-(max_lines - 4):]
    return "\n".join(out)


_METRIC_WORDS = (r"epoch|step|iter(?:ation)?|loss|acc(?:uracy)?|lr|ppl|perplexity|f1|bleu|reward|return|score"
                 r"|success(?:_rate)?|mse|mae|auc|map|fps")
# A metric name, optionally namespaced like episode/score, train/loss/dyn, val_loss or eval.acc.
METRIC_RE = re.compile(
    r"(?<![\w/.-])((?:[a-z][\w.-]*[/.])*(?:[a-z]+_)?(?:" + _METRIC_WORDS + r")(?:[/.][\w.-]+)*)\b"
    r"\s*[:=]?\s*\[?\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)(?![\w.])(?:\s*/\s*(\d+)(?![\w.]))?",
    re.I,
)
# Shown first: progress, then results, then losses.
_METRIC_RANK = [r"^(epoch|step|iter)", r"score|return|reward|success", r"(^|[/._])(eval|val|test)", r"acc|f1|bleu|auc",
                r"loss|ppl|perplexity|mse|mae"]


def extract_metrics(log_text: str, limit: int = 10) -> dict[str, str]:
    """Last seen value of common training metrics, e.g. {'epoch': '3/10', 'loss': '0.412',
    'episode/score': '3.1'}, at most `limit` of them, most informative first."""
    metrics: dict[str, str] = {}
    for m in METRIC_RE.finditer(log_text[-50_000:]):
        key = m.group(1).lower()
        metrics[key] = m.group(2) + (f"/{m.group(3)}" if m.group(3) else "")

    def rank(key: str) -> int:
        return next((i for i, rx in enumerate(_METRIC_RANK) if re.search(rx, key)), len(_METRIC_RANK))

    return dict(sorted(metrics.items(), key=lambda kv: rank(kv[0]))[:limit])


_PATH_RE = re.compile(r"(?<![\w.-])(/[\w./%+=@-]+\.(?:log|out|txt|err|jsonl|csv))\b")


def find_log_paths(text: str, job_id: str, exclude: tuple[str, ...] = ()) -> list[str]:
    """Log files a job's output points to: absolute paths that contain the job id
    (e.g. "output: /scratch/me/runs/x/train_18743802.log"). Training scripts often
    redirect their real output there, leaving only a preamble in SLURM's stdout file."""
    out = []
    for path in _PATH_RE.findall(text):
        if job_id in path and path not in exclude and path not in out:
            out.append(path)
    return out


def tqdm_cleanup(text: str) -> str:
    """Keep only the final frame of carriage-return progress bars."""
    return "\n".join(line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n"))


STATUS_SYSTEM = """Summarise the progress of a running ML training job in ONE short line for a Telegram status update
(e.g. "epoch 3/10, loss 0.41 ↓, ~2h left at current pace"). Use only what the log shows; if nothing useful, say "no progress info in log"."""


class Reporter:
    """One-line LLM progress summaries for status messages (llm_status_updates), using the agent's current
    backend in one-shot mode (no tools)."""

    def __init__(self, text_fn: Callable[[str, str], Awaitable[str]] | None, enabled: bool, llm_status: bool):
        self.text_fn = text_fn  # async (system, prompt) -> str
        self.enabled = enabled and text_fn is not None
        self.llm_status = llm_status

    async def status_line(self, log_tail: str) -> str:
        if not (self.enabled and self.llm_status and log_tail.strip()):
            return ""
        try:
            return (await self.text_fn(STATUS_SYSTEM, log_tail[-8000:])).strip()
        except Exception:
            log.exception("LLM status failed")
            return ""
