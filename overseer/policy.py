"""Safety policy: the rules that keep Overseer from touching anything outside its sandbox.

Everything here is pure (no I/O) so it can be unit-tested exhaustively.

Layers of protection, from the inside out:

1. ``check_remote_argv`` - every command sent over SSH passes through this guard. Only an
   allowlist of binaries can run; binaries that modify anything (mkdir, cp, rm, dd, sbatch,
   scancel, ...) are only accepted with absolute target paths inside the sandbox and with a
   fixed set of options. There is no way to send a raw shell string.
2. ``HPC`` (hpc.py) additionally resolves every path with ``realpath`` on the cluster before
   use, so symlinks cannot be used to escape the sandbox.
3. ``check_script`` - static analysis of batch scripts the agent wants to submit: resource
   limits, plus heuristic detection of writes outside the sandbox.
4. Human approval in Telegram before any ``sbatch``/``scancel`` (on by default).
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
import shlex
from dataclasses import dataclass, field

from .config import LimitsConfig


class PolicyError(Exception):
    """Raised when an operation would violate the sandbox policy."""


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

def _clean(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise PolicyError("empty path")
    if "\x00" in path or "\n" in path or "\r" in path:
        raise PolicyError("path contains control characters")
    return path


def is_within(path: str, root: str) -> bool:
    """True if ``path`` is ``root`` or below it (purely lexical; callers resolve symlinks first)."""
    path = posixpath.normpath(path)
    root = posixpath.normpath(root)
    if not path.startswith("/") or not root.startswith("/"):
        return False
    return path == root or path.startswith(root.rstrip("/") + "/")


def require_in_sandbox(path: str, sandbox_root: str, *, allow_root: bool = True) -> str:
    path = _clean(path)
    if not path.startswith("/"):
        raise PolicyError(f"path must be absolute: {path!r}")
    norm = posixpath.normpath(path)
    if not is_within(norm, sandbox_root):
        raise PolicyError(f"{path!r} is outside the sandbox {sandbox_root}")
    if not allow_root and norm == posixpath.normpath(sandbox_root):
        raise PolicyError("refusing to operate on the sandbox root itself")
    return norm


def is_denied(path: str, deny_globs: list[str]) -> bool:
    norm = posixpath.normpath(path)
    return any(fnmatch.fnmatchcase(norm, g) for g in deny_globs)


# --------------------------------------------------------------------------------------
# Remote command guard
# --------------------------------------------------------------------------------------

READ_ONLY_BINARIES = frozenset(
    {
        "squeue", "sacct", "sinfo", "seff", "whoami", "printenv", "realpath", "cat", "head",
        "tail", "ls", "stat", "grep", "du", "diff", "sha256sum", "wc", "test", "hostname",
    }
)
FIND_FORBIDDEN = frozenset(
    {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
)
JOB_ID_RE = re.compile(r"^\d+(_\d+)?$")


def _positionals(args: list[str]) -> tuple[list[str], list[str]]:
    opts, pos, after_dd = [], [], False
    for a in args:
        if after_dd:
            pos.append(a)
        elif a == "--":
            after_dd = True
        elif a.startswith("-") and a != "-":
            opts.append(a)
        else:
            pos.append(a)
    return opts, pos


def _only_opts(opts: list[str], allowed: set[str], binary: str) -> None:
    for o in opts:
        key = o.split("=", 1)[0] if o.startswith("--") else o
        if key not in allowed:
            raise PolicyError(f"option {o!r} is not allowed for {binary}")


def check_remote_argv(argv: list[str], sandbox_root: str) -> None:
    """Raise PolicyError unless ``argv`` is an allowed remote command."""
    if not argv:
        raise PolicyError("empty command")
    for a in argv:
        if not isinstance(a, str) or "\x00" in a:
            raise PolicyError("invalid argument")
    binary, args = argv[0], argv[1:]
    if "/" in binary:
        raise PolicyError("binaries must be given by name, not path")

    if binary in READ_ONLY_BINARIES:
        return
    if binary == "find":
        bad = FIND_FORBIDDEN.intersection(args)
        if bad:
            raise PolicyError(f"find action {sorted(bad)[0]} is not allowed")
        return
    if binary == "scontrol":
        if args[:1] == ["show"]:
            return
        if len(args) == 4 and args[:2] == ["write", "batch_script"] and JOB_ID_RE.match(args[2]) and args[3] == "-":
            return
        raise PolicyError("only 'scontrol show' and 'scontrol write batch_script <id> -' are allowed")
    if binary == "scancel":
        if len(args) == 1 and JOB_ID_RE.match(args[0]):
            return
        raise PolicyError("scancel takes exactly one job id")
    if binary == "srun":
        # Only the read-only GPU probe inside an existing allocation.
        if (
            len(args) >= 4
            and re.fullmatch(r"--jobid=\d+(_\d+)?", args[0])
            and args[1:3] == ["--overlap", "--ntasks=1"]
            and args[3] == "nvidia-smi"
            and all(a.startswith("--query-gpu=") or a.startswith("--format=") for a in args[4:])
        ):
            return
        raise PolicyError("srun is only allowed for the nvidia-smi probe")
    if binary == "sbatch":
        opts, pos = _positionals(args)
        _only_opts(opts, {"--parsable", "--chdir", "--output", "--error", "--export"}, binary)
        if len(pos) != 1:
            raise PolicyError("sbatch takes exactly one script path")
        require_in_sandbox(pos[0], sandbox_root)
        given = {o.split("=", 1)[0]: o.split("=", 1)[1] for o in opts if "=" in o}
        for key in ("--chdir", "--output", "--error"):
            if key not in given:
                raise PolicyError(f"sbatch requires {key} inside the sandbox")
            require_in_sandbox(given[key], sandbox_root)
        return
    if binary == "mkdir":
        opts, pos = _positionals(args)
        _only_opts(opts, {"-p", "--mode"}, binary)
        if not pos:
            raise PolicyError("mkdir needs a path")
        for p in pos:
            require_in_sandbox(p, sandbox_root)
        return
    if binary == "cp":
        opts, pos = _positionals(args)
        _only_opts(opts, {"-r", "-R", "-P", "--no-preserve", "--preserve", "-T"}, binary)
        if len(pos) < 2:
            raise PolicyError("cp needs a source and a destination")
        require_in_sandbox(pos[-1], sandbox_root, allow_root=False)
        return
    if binary == "rm":
        opts, pos = _positionals(args)
        _only_opts(opts, {"-r", "-f", "-rf", "-fr"}, binary)
        if not pos:
            raise PolicyError("rm needs a path")
        for p in pos:
            require_in_sandbox(p, sandbox_root, allow_root=False)
        return
    if binary == "dd":
        targets = [a[3:] for a in args if a.startswith("of=")]
        if len(targets) != 1 or any(not (a.startswith("of=") or a == "status=none") for a in args):
            raise PolicyError("dd is only allowed as 'dd of=<sandbox path> status=none'")
        require_in_sandbox(targets[0], sandbox_root, allow_root=False)
        return
    raise PolicyError(f"command {binary!r} is not on the allowlist")


# --------------------------------------------------------------------------------------
# Batch script analysis
# --------------------------------------------------------------------------------------

SHORT_OPTS = {
    "-p": "partition", "-t": "time", "-G": "gpus", "-N": "nodes", "-n": "ntasks",
    "-o": "output", "-e": "error", "-D": "chdir", "-J": "job-name", "-a": "array",
    "-c": "cpus-per-task", "-A": "account", "-q": "qos", "-d": "dependency",
}


def parse_slurm_time(value: str) -> int:
    """Parse a SLURM time spec into seconds. Raises ValueError on bad input."""
    v = value.strip()
    if v.upper() in ("UNLIMITED", "INFINITE"):
        raise ValueError("unlimited time")
    days = 0
    if "-" in v:
        d, v = v.split("-", 1)
        days = int(d)
        parts = [int(p) for p in v.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, s = parts
    else:
        parts = [int(p) for p in v.split(":")]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        elif len(parts) == 3:
            h, m, s = parts
        else:
            raise ValueError(f"bad time {value!r}")
    if min(days, h, m, s) < 0:
        raise ValueError(f"bad time {value!r}")
    return ((days * 24 + h) * 60 + m) * 60 + s


def format_seconds(sec: int) -> str:
    d, rem = divmod(int(sec), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def parse_sbatch_directives(script: str) -> dict[str, str]:
    """Return the #SBATCH options (long names, without dashes) in the order SLURM reads them."""
    opts: dict[str, str] = {}
    lines = script.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if i == 0 and s.startswith("#!"):
            continue
        if not s or (s.startswith("#") and not s.startswith("#SBATCH")):
            continue
        if not s.startswith("#SBATCH"):
            break  # SLURM stops reading directives at the first command
        body = s[len("#SBATCH"):].split(" #", 1)[0].strip()
        try:
            toks = shlex.split(body)
        except ValueError:
            toks = body.split()
        j = 0
        while j < len(toks):
            t = toks[j]
            if t.startswith("--"):
                if "=" in t:
                    k, v = t[2:].split("=", 1)
                elif j + 1 < len(toks) and not toks[j + 1].startswith("-"):
                    k, v = t[2:], toks[j + 1]
                    j += 1
                else:
                    k, v = t[2:], ""
                opts[k] = v
            elif t.startswith("-") and len(t) >= 2:
                k = SHORT_OPTS.get(t[:2], t[1:2])
                if len(t) > 2:
                    v = t[2:].lstrip("=")
                elif j + 1 < len(toks):
                    v = toks[j + 1]
                    j += 1
                else:
                    v = ""
                opts[k] = v
            j += 1
    return opts


def _gres_gpus(spec: str) -> int:
    total = 0
    for item in spec.split(","):
        item = item.strip()
        if not item.startswith("gpu"):
            continue
        parts = item.split(":")
        last = parts[-1]
        if len(parts) > 1 and last.isdigit():
            total += int(last)
        else:
            total += 1
    return total


def _count(spec: str) -> int:
    last = spec.split(":")[-1]
    return int(last) if last.isdigit() else 1


def count_gpus(opts: dict[str, str]) -> int:
    nodes = _int_or(opts.get("nodes", "1").split("-")[-1], 1)
    ntasks = _int_or(opts.get("ntasks", "1"), 1)
    candidates = [0]
    if "gres" in opts:
        candidates.append(_gres_gpus(opts["gres"]) * nodes)
    if "gpus" in opts:
        candidates.append(_count(opts["gpus"]))
    if "gpus-per-node" in opts:
        candidates.append(_count(opts["gpus-per-node"]) * nodes)
    if "gpus-per-task" in opts:
        candidates.append(_count(opts["gpus-per-task"]) * ntasks)
    return max(candidates)


def count_array_tasks(spec: str) -> int:
    spec = spec.split("%", 1)[0]
    total = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, st = part.split(":", 1)
            step = max(int(st), 1)
        if "-" in part:
            a, b = part.split("-", 1)
            total += (int(b) - int(a)) // step + 1
        else:
            total += 1
    return total


def _int_or(v: str, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


SAFE_DEV = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty")
SAFE_VARS = ("$OVERSEER_", "${OVERSEER_", "$TMPDIR", "${TMPDIR", "$SLURM_TMPDIR", "${SLURM_TMPDIR", "$PWD", "${PWD")
HOME_PREFIXES = ("~", "$HOME", "${HOME")

WRITE_ALL_TARGETS = {"rm", "rmdir", "touch", "mkdir", "chmod", "chown", "truncate", "tee", "mv", "unlink", "shred"}
WRITE_LAST_TARGET = {"cp", "rsync", "ln", "install", "scp"}
REDIRECT_RE = re.compile(r"(?:^|[^<>&0-9])(?:[0-9]|&)?>>?\|?\s*([^\s;|&)<>]+)")


@dataclass
class ScriptReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    directives: dict[str, str] = field(default_factory=dict)
    gpus: int = 0
    nodes: int = 1
    time_seconds: int | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def resources_line(self) -> str:
        t = format_seconds(self.time_seconds) if self.time_seconds else "?"
        part = self.directives.get("partition", "default")
        return f"partition={part} nodes={self.nodes} gpus={self.gpus} time={t}"


def _classify_target(target: str, sandbox_root: str) -> str | None:
    """Return a problem description if writing to ``target`` would leave the sandbox."""
    t = target.strip("'\"")
    if not t or t.startswith("-"):
        return None
    if t.startswith(HOME_PREFIXES):
        return f"writes into your home directory ({t})"
    if t.startswith("/"):
        if t in SAFE_DEV or t.startswith("/dev/fd/") or is_within(t, sandbox_root):
            return None
        return f"writes outside the sandbox ({t})"
    return None


def _strip_comment(line: str) -> str:
    s = line.strip()
    if s.startswith("#"):
        return ""
    return re.split(r"\s+#", s, maxsplit=1)[0]


def check_script(script: str, sandbox_root: str, limits: LimitsConfig) -> ScriptReport:
    rep = ScriptReport()
    if not script.startswith("#!"):
        rep.errors.append("script must start with a shebang line (e.g. #!/bin/bash)")
    opts = parse_sbatch_directives(script)
    rep.directives = opts

    # ---- resources
    if "time" not in opts:
        rep.errors.append("set an explicit '#SBATCH --time=...' so the limit can be checked")
    else:
        try:
            rep.time_seconds = parse_slurm_time(opts["time"])
            max_s = parse_slurm_time(limits.max_time)
            if rep.time_seconds > max_s:
                rep.errors.append(f"--time {opts['time']} exceeds the limit {limits.max_time}")
        except ValueError:
            rep.errors.append(f"could not parse --time {opts['time']!r}")
    rep.nodes = _int_or(opts.get("nodes", "1").split("-")[-1], 1)
    if rep.nodes > limits.max_nodes_per_job:
        rep.errors.append(f"{rep.nodes} nodes exceeds the limit of {limits.max_nodes_per_job}")
    rep.gpus = count_gpus(opts)
    if rep.gpus > limits.max_gpus_per_job:
        rep.errors.append(f"{rep.gpus} GPUs exceeds the limit of {limits.max_gpus_per_job}")
    if limits.allowed_partitions and "partition" in opts:
        for p in opts["partition"].split(","):
            if p not in limits.allowed_partitions:
                rep.errors.append(f"partition {p!r} is not in allowed_partitions")
    if "array" in opts:
        try:
            n = count_array_tasks(opts["array"])
        except ValueError:
            n = 10**9
        if n > limits.max_array_tasks:
            rep.errors.append(f"job array with {n} tasks exceeds max_array_tasks={limits.max_array_tasks}")
    for k in ("output", "error", "chdir"):
        if k in opts:
            rep.warnings.append(f"#SBATCH --{k} is ignored; Overseer sets it inside the sandbox")
    if "wrap" in opts:
        rep.errors.append("--wrap is not allowed")

    # ---- body
    for lineno, raw in enumerate(script.splitlines(), 1):
        line = _strip_comment(raw)
        if not line:
            continue
        where = f"line {lineno}"
        if re.search(r"(^|[;&|\s])sudo\s", line):
            rep.errors.append(f"{where}: sudo is not allowed")
        if re.search(r"(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b", line):
            rep.errors.append(f"{where}: piping a download into a shell is not allowed")
        if re.search(r"(^|[;&|\s])(sbatch|scancel|salloc)\b", line):
            rep.errors.append(f"{where}: jobs may not submit or cancel other jobs")
        if re.search(r"(^|[;&|\s])(pip|pip3|conda|mamba|micromamba)\s+(install|create|update|remove|uninstall)\b", line) or re.search(r"-m\s+pip\s+install", line):
            if "--target" not in line and "--prefix" not in line:
                rep.warnings.append(f"{where}: package install modifies an environment outside the sandbox")
        if re.search(r"\bgit\s+(push|commit|reset|checkout|clean)\b", line):
            rep.warnings.append(f"{where}: git command may modify a repository")
        if re.search(r"(^|[;&|\s])(ssh|scp)\s", line):
            rep.warnings.append(f"{where}: opens an SSH connection")
        m = re.match(r"^cd\s+(\S+)", line)
        if m and _classify_target(m.group(1), sandbox_root):
            rep.warnings.append(f"{where}: cd to {m.group(1)} — later relative writes land outside the sandbox")
        for target in REDIRECT_RE.findall(line):
            prob = _classify_target(target, sandbox_root)
            if prob:
                rep.errors.append(f"{where}: redirect {prob}")
        for segment in re.split(r"[;&|]+", line):
            try:
                toks = shlex.split(segment)
            except ValueError:
                toks = segment.split()
            while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
                toks = toks[1:]  # leading VAR=value assignments
            if not toks:
                continue
            cmd = posixpath.basename(toks[0])
            pos = [t for t in toks[1:] if not t.startswith("-")]
            targets: list[str] = []
            if cmd in WRITE_ALL_TARGETS:
                targets = pos
            elif cmd in WRITE_LAST_TARGET and len(pos) >= 2:
                targets = pos[-1:]
            elif cmd == "dd":
                targets = [t[3:] for t in toks if t.startswith("of=")]
            for t in targets:
                prob = _classify_target(t, sandbox_root)
                if prob:
                    rep.errors.append(f"{where}: {cmd} {prob}")
            if cmd in ("rm", "chmod", "chown") and any(t in ("/", "/*", "~", "~/", "$HOME") for t in pos):
                rep.errors.append(f"{where}: destructive {cmd} on a root/home path")
            for t in toks[1:]:
                ref = t.split("=", 1)[1] if t.startswith("-") and "=" in t else t
                if ref.startswith(("/", "~", "$HOME", "${HOME")) and _classify_target(ref, sandbox_root):
                    rep.warnings.append(f"{where}: references {ref} outside the sandbox (OK only if read-only)")
    rep.errors = list(dict.fromkeys(rep.errors))
    rep.warnings = list(dict.fromkeys(rep.warnings))
    return rep


ABS_PATH_LITERAL_RE = re.compile(r"""["'](/[^"'\s]+)["']""")


def scan_diff(diff: str, sandbox_root: str) -> list[str]:
    """Warnings for added lines in a unified diff that reference locations outside the sandbox."""
    warnings: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for path in ABS_PATH_LITERAL_RE.findall(line):
            if not is_within(path, sandbox_root) and not path.startswith("/dev/"):
                warnings.append(f"added code references {path} (outside sandbox; fine if read-only)")
        if "expanduser(" in line or "Path.home()" in line:
            warnings.append("added code uses the home directory")
    return list(dict.fromkeys(warnings))
