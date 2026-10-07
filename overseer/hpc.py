"""High-level, policy-enforcing operations on the HPC.

Everything that touches the cluster goes through this class.

* Writes, deletes, copies and job submissions are limited to the *write roots*: the
  folders in ``hpc.write_roots`` plus folders granted from Telegram at runtime.
* Reads are limited to ``hpc.read_roots`` + write roots + files of monitored jobs.
* Secret files are never readable; login scripts/cron/ssh files are never writable.
* Every path is resolved with ``realpath`` *on the cluster* and the resolved path is what
  gets checked and used, so symlinks can't redirect a write.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
import re
from dataclasses import dataclass

from .config import Config, ConfigError, check_root_spec
from .policy import PolicyError, expand_home, is_denied, is_within, require_in_roots
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
SECRET_NAMES = [".ssh", ".gnupg", "*.pem", "id_rsa*", "id_ed25519*", "id_ecdsa*", ".netrc", ".env", ".git-credentials"]


@dataclass
class FileText:
    path: str
    text: str
    truncated: bool


def _dedupe(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(p for p in paths if p))


class HPC:
    def __init__(self, cfg: Config, runner: SSHRunner):
        self.cfg = cfg
        self.run = runner.run
        self.user = ""
        self.home = ""
        self.grants: dict[str, str] = {}  # absolute path -> "read" | "write" (granted at runtime)
        self.write_roots: list[str] = []  # resolved; empty until connected => all writes refused
        self.read_roots: list[str] = []  # resolved, includes write roots
        self.missing_roots: list[str] = []
        self.extra_read_paths: set[str] = set()  # files/dirs of monitored jobs

    # ------------------------------------------------------------------ setup
    async def connect(self) -> None:
        self.user = (await self.run(["whoami"], check=True)).stdout.strip()
        self.home = (await self.run(["printenv", "HOME"], check=True)).stdout.strip() or f"/home/{self.user}"
        await self.refresh_roots()

    async def ensure_connected(self) -> None:
        if not self.user:
            await self.connect()

    def expand(self, path: str) -> str:
        path = path.strip().replace("$USER", self.user or "$USER")
        path = expand_home(path, self.home)
        if not path.startswith("/"):
            raise PolicyError(f"use an absolute path or ~/...: {path!r}")
        return posixpath.normpath(path)

    async def realpath(self, path: str) -> str:
        res = await self.run(["realpath", "-m", "--", path], check=True)
        return posixpath.normpath(res.stdout.strip())

    async def exists(self, path: str) -> bool:
        return (await self.run(["test", "-e", path])).returncode == 0

    async def is_dir(self, path: str) -> bool:
        return (await self.run(["test", "-d", path])).returncode == 0

    async def refresh_roots(self) -> None:
        write_specs = list(self.cfg.hpc.write_roots) + [p for p, m in self.grants.items() if m == "write"]
        read_specs = list(self.cfg.hpc.read_roots) + [p for p, m in self.grants.items() if m == "read"]
        write, read, missing = [], [], []
        for spec in write_specs:
            real = await self.realpath(self.expand(spec))
            if real == "/" or real == self.home:
                log.warning("ignoring overly broad write root %s", spec)
                continue
            write.append(real)
            if not await self.exists(real):
                missing.append(real)
        for spec in read_specs:
            read.append(await self.realpath(self.expand(spec)))
        self.write_roots = _dedupe(write)
        self.read_roots = _dedupe(read + write)
        self.missing_roots = _dedupe(missing)

    async def resolve_root_request(self, path: str, mode: str) -> str:
        """Validate a folder the agent/user wants access to; returns the resolved path."""
        if mode not in ("read", "write"):
            raise PolicyError("mode must be 'read' or 'write'")
        try:
            if mode == "write":
                check_root_spec(path)
            elif not path.startswith(("/", "~")):
                raise ConfigError(f"{path!r} must be absolute or start with ~")
        except ConfigError as e:
            raise PolicyError(str(e)) from None
        real = await self.realpath(self.expand(path))
        if real == "/" or (mode == "write" and real == self.home):
            raise PolicyError(f"{real} is too broad to grant {mode} access to")
        globs = self.cfg.hpc.deny_write_globs if mode == "write" else self.cfg.hpc.deny_read_globs
        if is_denied(real, globs):
            raise PolicyError(f"{real} matches the deny list")
        return real

    def has_access(self, real: str, mode: str) -> bool:
        roots = self.write_roots if mode == "write" else self.read_roots
        return any(is_within(real, r) for r in roots)

    async def grant(self, path: str, mode: str) -> None:
        self.grants[path] = mode
        await self.refresh_roots()

    async def revoke(self, path: str) -> bool:
        if self.grants.pop(path, None) is None:
            return False
        await self.refresh_roots()
        return True

    # ------------------------------------------------------------------ path policy
    async def check_write(self, path: str, *, allow_root: bool = False) -> str:
        p = self.expand(path)
        real = await self.realpath(p)
        for candidate in (p, real):
            if is_denied(candidate, self.cfg.hpc.deny_write_globs):
                raise PolicyError(f"{path!r} is on the never-writable list (login scripts, keys, ...)")
        try:
            return require_in_roots(real, self.write_roots, allow_root=allow_root)
        except PolicyError as e:
            raise PolicyError(f"{e}. Use request_access to ask the user for this folder.") from None

    def readable_lexically(self, path: str) -> bool:
        if any(is_within(path, r) for r in self.read_roots):
            return True
        return any(path == p or is_within(path, p) for p in self.extra_read_paths)

    async def check_read(self, path: str) -> str:
        p = self.expand(path)
        if is_denied(p, self.cfg.hpc.deny_read_globs):
            raise PolicyError(f"{path!r} matches the secret denylist")
        real = await self.realpath(p)
        if is_denied(real, self.cfg.hpc.deny_read_globs):
            raise PolicyError(f"{path!r} matches the secret denylist")
        if not self.readable_lexically(real):
            raise PolicyError(
                f"{path!r} is outside the readable folders ({', '.join(self.read_roots)}). "
                "Use request_access with mode 'read' to ask the user."
            )
        return real

    def allow_job_paths(self, *paths: str) -> None:
        for p in paths:
            if p and p.startswith("/") and p not in ("/", self.home):
                self.extra_read_paths.add(posixpath.normpath(p))

    # ------------------------------------------------------------------ file ops
    async def read_file(self, path: str, *, max_bytes: int = TEXT_LIMIT, tail_lines: int | None = None) -> FileText:
        real = await self.check_read(path)
        if tail_lines:
            res = await self.run(["tail", "-n", str(int(tail_lines)), "--", real], max_output=max_bytes)
        else:
            res = await self.run(["head", "-c", str(int(max_bytes) + 1), "--", real], max_output=max_bytes + 1)
        if not res.ok:
            raise PolicyError(res.stderr.strip() or f"cannot read {real}")
        text = res.stdout
        truncated = res.truncated or len(text.encode()) > max_bytes
        return FileText(real, text[:max_bytes], truncated)

    async def read_for_edit(self, real: str) -> str:
        """Full text of a file that is about to be modified (so it can be backed up and restored)."""
        limit = self.cfg.hpc.max_edit_mb * 1024 * 1024
        ft = await self.read_file(real, max_bytes=limit)
        if ft.truncated:
            raise PolicyError(f"{real} is larger than {self.cfg.hpc.max_edit_mb} MB; refusing to modify it")
        if "\x00" in ft.text or "�" in ft.text:
            raise PolicyError(f"{real} is not a UTF-8 text file; refusing to modify it")
        return ft.text

    async def try_tail(self, path: str, lines: int) -> str:
        if not path:
            return ""
        try:
            return (await self.read_file(path, tail_lines=lines, max_bytes=60_000)).text
        except (PolicyError, SSHError) as e:
            return f"<could not read {path}: {e}>"

    async def list_dir(self, path: str) -> str:
        real = await self.check_read(path)
        res = await self.run(["ls", "-la", "--", real], max_output=50_000)
        if not res.ok:
            raise PolicyError(res.stderr.strip())
        return res.stdout

    async def search(self, pattern: str, path: str, include: str | None = None) -> str:
        real = await self.check_read(path)
        argv = ["grep", "-rnIE", "--max-count=20"]
        for d in (".ssh", ".gnupg", ".aws", ".kube", ".git", "__pycache__", ".cache", "node_modules", "wandb"):
            argv.append(f"--exclude-dir={d}")
        if include:
            argv.append(f"--include={include}")
        argv += ["-e", pattern, "--", real]
        res = await self.run(argv, max_output=40_000, timeout=60)
        lines = [ln for ln in res.stdout.splitlines() if not is_denied(ln.split(":", 1)[0], self.cfg.hpc.deny_read_globs)]
        out = "\n".join(lines[:300])
        if res.truncated or len(lines) > 300:
            out += "\n<output truncated>"
        return out or "<no matches>"

    async def write_file(self, path: str, content: str) -> str:
        real = await self.check_write(path)
        parent = posixpath.dirname(real)
        if not await self.is_dir(parent):
            await self.run(["mkdir", "-p", require_in_roots(parent, self.write_roots)], check=True)
        await self.run(["dd", f"of={real}", "status=none"], stdin=content.encode(), check=True)
        return real

    async def delete(self, path: str) -> str:
        real = await self.check_write(path)
        await self.run(["rm", "-rf", "--", real], check=True)
        return real

    async def mkdir(self, path: str) -> str:
        real = await self.check_write(path, allow_root=True)
        await self.run(["mkdir", "-p", real], check=True)
        return real

    async def copy(self, src: str, dest: str) -> str:
        """Copy a readable file/directory to a new path inside a write root."""
        src_real = await self.check_read(src)
        if src_real in self.read_roots or src_real == self.home:
            raise PolicyError("refusing to copy an entire root folder; pick a project directory or file")
        dest_real = await self.check_write(dest)
        if await self.exists(dest_real):
            raise PolicyError(f"{dest_real} already exists; copy to a new path")
        size = await self.run(["du", "-sk", "--", src_real], timeout=120)
        try:
            kb = int(size.stdout.split()[0])
        except (IndexError, ValueError):
            raise PolicyError(f"cannot determine size of {src_real}: {size.stderr.strip()}")
        if kb > self.cfg.hpc.max_copy_mb * 1024:
            raise PolicyError(
                f"{src_real} is {kb // 1024} MB (> {self.cfg.hpc.max_copy_mb} MB). "
                "Copy only the code you need, not data/checkpoints."
            )
        probe = ["find", src_real, "("]
        for i, name in enumerate(SECRET_NAMES):
            probe += (["-o"] if i else []) + ["-name", name]
        res = await self.run(probe + [")", "-print", "-quit"], timeout=120)
        if res.stdout.strip():
            raise PolicyError(f"{src_real} contains sensitive files ({res.stdout.strip()}); copy a narrower path")
        parent = posixpath.dirname(dest_real)
        if not await self.is_dir(parent):
            await self.run(["mkdir", "-p", require_in_roots(parent, self.write_roots)], check=True)
        res = await self.run(["cp", "-r", "-P", "-T", "--", src_real, dest_real], timeout=300)
        if not res.ok:
            raise PolicyError(f"copy failed: {res.stderr.strip()[:500]}")
        return dest_real

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

    async def sbatch(self, script: str, workdir: str) -> str:
        script = await self.check_write(script)
        workdir = await self.check_write(workdir, allow_root=True)
        res = await self.run(["sbatch", "--parsable", f"--chdir={workdir}", script])
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
