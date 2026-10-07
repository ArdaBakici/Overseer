"""High-level, policy-enforcing operations on the HPC.

Everything that touches the cluster goes through this class. Reads are limited to
``sandbox.read_roots`` (+ files of jobs being monitored) minus a secret denylist; writes,
deletes and submissions are limited to ``sandbox.root``. Paths are resolved with
``realpath`` on the cluster so a symlink can't point a write outside the sandbox.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
import re
from dataclasses import dataclass

from .config import Config
from .policy import PolicyError, is_denied, is_within, require_in_sandbox
from .slurm import (
    SACCT_LIST_FORMAT,
    SACCT_STEP_FORMAT,
    SQUEUE_FORMAT,
    Job,
    parse_sacct_list,
    parse_sacct_steps,
    parse_scontrol,
    parse_squeue,
)
from .ssh import SSHError, SSHRunner

log = logging.getLogger(__name__)

TEXT_LIMIT = 200_000


@dataclass
class FileText:
    path: str
    text: str
    truncated: bool


class HPC:
    def __init__(self, cfg: Config, runner: SSHRunner):
        self.cfg = cfg
        self.run = runner.run
        self.root = cfg.sandbox.root
        self.user = ""
        self.home = ""
        self._read_roots: list[str] = []
        self.extra_read_paths: set[str] = set()  # files/dirs of monitored jobs

    # ------------------------------------------------------------------ setup
    async def connect(self) -> None:
        self.user = (await self.run(["whoami"], check=True)).stdout.strip()
        self.home = (await self.run(["printenv", "HOME"], check=True)).stdout.strip() or f"/home/{self.user}"
        roots = []
        for r in self.cfg.sandbox.read_roots:
            r = r.replace("$USER", self.user)
            roots.append(posixpath.normpath(self.home + r[1:] if r.startswith("~") else r))
        self._read_roots = roots
        await self.ensure_sandbox()

    async def ensure_connected(self) -> None:
        if not self.user:
            await self.connect()

    async def ensure_sandbox(self) -> None:
        """Create the sandbox root with mode 700 and verify we own it (guards against squatting in /tmp)."""
        await self.run(["mkdir", "-p", "--mode=700", self.root], check=True)
        st = await self.run(["stat", "-c", "%U %a %F", self.root], check=True)
        owner, mode, *kind = st.stdout.split()
        if owner != self.user:
            raise PolicyError(f"{self.root} is owned by {owner!r}, not {self.user!r}; refusing to use it")
        if " ".join(kind) != "directory":
            raise PolicyError(f"{self.root} is not a directory")
        if mode[-2:] != "00":
            raise PolicyError(f"{self.root} has mode {mode}; it must not be accessible to other users (chmod 700)")
        real = (await self.run(["realpath", "-e", "--", self.root], check=True)).stdout.strip()
        if posixpath.normpath(real) != self.root:
            raise PolicyError(f"{self.root} resolves to {real}; the sandbox root must not be a symlink")

    # ------------------------------------------------------------------ path policy
    def expand(self, path: str, base: str | None = None) -> str:
        path = path.strip()
        if path.startswith("~"):
            path = self.home + path[1:]
        elif not path.startswith("/"):
            if not base:
                raise PolicyError(f"relative path {path!r} needs a base directory")
            path = posixpath.join(base, path)
        return posixpath.normpath(path)

    async def realpath(self, path: str) -> str:
        res = await self.run(["realpath", "-m", "--", path], check=True)
        return posixpath.normpath(res.stdout.strip())

    async def check_write(self, path: str, base: str | None = None) -> str:
        p = require_in_sandbox(self.expand(path, base), self.root, allow_root=False)
        real = await self.realpath(p)
        if not is_within(real, self.root):
            raise PolicyError(f"{path!r} resolves to {real}, outside the sandbox")
        return real

    def readable_lexically(self, path: str) -> bool:
        roots = [self.root, *self._read_roots]
        if any(is_within(path, r) for r in roots):
            return True
        return any(path == p or is_within(path, p) for p in self.extra_read_paths)

    async def check_read(self, path: str, base: str | None = None) -> str:
        p = self.expand(path, base)
        if is_denied(p, self.cfg.sandbox.deny_read_globs):
            raise PolicyError(f"{path!r} matches the secret denylist")
        real = await self.realpath(p)
        if is_denied(real, self.cfg.sandbox.deny_read_globs):
            raise PolicyError(f"{path!r} matches the secret denylist")
        if not (self.readable_lexically(real)):
            raise PolicyError(
                f"{path!r} is outside the readable roots ({', '.join([self.root, *self._read_roots])})"
            )
        return real

    def allow_job_paths(self, *paths: str) -> None:
        for p in paths:
            if p and p.startswith("/") and p not in ("/", self.home):
                self.extra_read_paths.add(posixpath.normpath(p))

    # ------------------------------------------------------------------ file ops
    async def read_file(self, path: str, *, base: str | None = None, max_bytes: int = TEXT_LIMIT, tail_lines: int | None = None) -> FileText:
        real = await self.check_read(path, base)
        if tail_lines:
            res = await self.run(["tail", "-n", str(int(tail_lines)), "--", real], max_output=max_bytes)
        else:
            res = await self.run(["head", "-c", str(int(max_bytes) + 1), "--", real], max_output=max_bytes)
        if not res.ok:
            raise PolicyError(res.stderr.strip() or f"cannot read {real}")
        text = res.stdout
        truncated = res.truncated or len(text) > max_bytes
        return FileText(real, text[:max_bytes], truncated)

    async def try_tail(self, path: str, lines: int) -> str:
        if not path:
            return ""
        try:
            return (await self.read_file(path, tail_lines=lines, max_bytes=60_000)).text
        except (PolicyError, SSHError) as e:
            return f"<could not read {path}: {e}>"

    async def list_dir(self, path: str, base: str | None = None) -> str:
        real = await self.check_read(path, base)
        res = await self.run(["ls", "-la", "--", real], max_output=50_000)
        if not res.ok:
            raise PolicyError(res.stderr.strip())
        return res.stdout

    async def search(self, pattern: str, path: str, base: str | None = None, include: str | None = None) -> str:
        real = await self.check_read(path, base)
        argv = ["grep", "-rnIE", "--max-count=20"]
        for d in (".ssh", ".gnupg", ".aws", ".kube", ".git", "__pycache__", ".cache", "node_modules", "wandb"):
            argv.append(f"--exclude-dir={d}")
        if include:
            argv.append(f"--include={include}")
        argv += ["-e", pattern, "--", real]
        res = await self.run(argv, max_output=40_000, timeout=60)
        lines = [ln for ln in res.stdout.splitlines() if not is_denied(ln.split(":", 1)[0], self.cfg.sandbox.deny_read_globs)]
        out = "\n".join(lines[:300])
        if res.truncated or len(lines) > 300:
            out += "\n<output truncated>"
        return out or "<no matches>"

    async def write_file(self, path: str, content: str, base: str | None = None) -> str:
        real = await self.check_write(path, base)
        await self.run(["mkdir", "-p", "--mode=700", posixpath.dirname(real)], check=True)
        await self.run(["dd", f"of={real}", "status=none"], stdin=content.encode(), check=True)
        return real

    async def delete(self, path: str, base: str | None = None) -> str:
        real = await self.check_write(path, base)
        await self.run(["rm", "-rf", "--", real], check=True)
        return real

    async def mkdir(self, path: str) -> str:
        real = await self.check_write(path)
        await self.run(["mkdir", "-p", "--mode=700", real], check=True)
        return real

    async def stage(self, src: str, dest_dir: str, base: str | None = None) -> str:
        """Copy code from a readable location into the sandbox (the original is never modified)."""
        src_real = await self.check_read(src, base)
        if src_real in self._read_roots or src_real == self.home:
            raise PolicyError("refusing to stage an entire read root; pick a project directory or file")
        size = await self.run(["du", "-sk", "--", src_real], timeout=120)
        try:
            kb = int(size.stdout.split()[0])
        except (IndexError, ValueError):
            raise PolicyError(f"cannot determine size of {src_real}: {size.stderr.strip()}")
        if kb > self.cfg.sandbox.max_stage_mb * 1024:
            raise PolicyError(
                f"{src_real} is {kb // 1024} MB (> {self.cfg.sandbox.max_stage_mb} MB). "
                "Stage only the code you need (individual files or subdirectories), not data/checkpoints."
            )
        # Never copy secrets into the sandbox.
        probe = await self.run(
            ["find", src_real, "(", "-name", ".ssh", "-o", "-name", ".gnupg", "-o", "-name", "*.pem",
             "-o", "-name", "id_rsa*", "-o", "-name", "id_ed25519*", "-o", "-name", ".netrc",
             "-o", "-name", ".env", ")", "-print", "-quit"],
            timeout=120,
        )
        if probe.stdout.strip():
            raise PolicyError(f"{src_real} contains sensitive files ({probe.stdout.strip()}); stage a narrower path")
        dest_real = await self.check_write(dest_dir)
        await self.run(["mkdir", "-p", "--mode=700", dest_real], check=True)
        res = await self.run(["cp", "-r", "-P", "--", src_real, dest_real + "/"], timeout=300)
        if not res.ok:
            raise PolicyError(f"copy failed: {res.stderr.strip()[:500]}")
        return posixpath.join(dest_real, posixpath.basename(src_real))

    async def diff(self, a: str, b: str) -> str:
        ra, rb = await self.check_read(a), await self.check_read(b)
        res = await self.run(["diff", "-ruN", "--exclude=__pycache__", "--", ra, rb], max_output=400_000)
        return res.stdout

    async def sha256(self, path: str) -> str:
        real = await self.check_read(path)
        res = await self.run(["sha256sum", "--", real], check=True)
        return res.stdout.split()[0]

    # ------------------------------------------------------------------ SLURM
    async def squeue(self) -> list[Job]:
        res = await self.run(["squeue", "-h", "-u", self.user, "-o", SQUEUE_FORMAT], check=True)
        return parse_squeue(res.stdout)

    async def sacct_recent(self, minutes: int) -> list[Job]:
        res = await self.run(
            ["sacct", "-X", "-n", "-P", "-u", self.user, "-S", f"now-{int(minutes)}minutes", "-o", SACCT_LIST_FORMAT],
            check=True,
        )
        return parse_sacct_list(res.stdout)

    async def sacct_job(self, job_id: str) -> Job | None:
        res = await self.run(["sacct", "-X", "-n", "-P", "-j", job_id, "-o", SACCT_LIST_FORMAT])
        jobs = [j for j in parse_sacct_list(res.stdout) if j.job_id == job_id]
        return jobs[0] if jobs else None

    async def sacct_steps(self, job_id: str) -> list[dict[str, str]]:
        res = await self.run(["sacct", "-n", "-P", "-j", job_id, "-o", SACCT_STEP_FORMAT])
        return parse_sacct_steps(res.stdout)

    async def scontrol_job(self, job_id: str) -> dict[str, str]:
        res = await self.run(["scontrol", "show", "job", "-o", job_id])
        return parse_scontrol(res.stdout) if res.ok else {}

    async def batch_script(self, job_id: str) -> str:
        res = await self.run(["scontrol", "write", "batch_script", job_id, "-"], max_output=500_000)
        return res.stdout if res.ok else ""

    async def seff(self, job_id: str) -> str:
        try:
            res = await self.run(["seff", job_id], timeout=30)
        except SSHError:
            return ""
        return res.stdout.strip() if res.ok else ""

    async def gpu_stats(self, job_id: str) -> str:
        try:
            res = await self.run(
                ["srun", f"--jobid={job_id}", "--overlap", "--ntasks=1", "nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader"],
                timeout=45,
            )
        except SSHError:
            return ""
        return res.stdout.strip() if res.ok else ""

    async def sbatch(self, script: str, workdir: str, logdir: str, env: dict[str, str]) -> str:
        script = await self.check_write(script)
        workdir = await self.check_write(workdir)
        logdir = await self.check_write(logdir)
        for k, v in env.items():
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k) or "," in v or "\n" in v:
                raise PolicyError(f"bad environment variable {k}")
        export = ",".join(["ALL", *(f"{k}={v}" for k, v in env.items())])
        res = await self.run(
            ["sbatch", "--parsable", f"--chdir={workdir}", f"--output={logdir}/slurm-%j.out",
             f"--error={logdir}/slurm-%j.err", f"--export={export}", script],
        )
        if not res.ok:
            raise SSHError(f"sbatch failed: {(res.stderr or res.stdout).strip()[:1000]}")
        job_id = res.stdout.strip().split(";")[0]
        if not re.fullmatch(r"\d+", job_id):
            raise SSHError(f"unexpected sbatch output: {res.stdout.strip()[:200]}")
        return job_id

    async def scancel(self, job_id: str) -> None:
        await self.run(["scancel", job_id], check=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
