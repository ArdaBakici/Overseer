"""Running heavy work on compute nodes instead of the login node.

Overseer generates two tools for the agent's shell (in local.tmp_dir/agent-bin, next to the
ssh/scp/sftp wrappers):

* `ssh`: still only talks to Overseer's connection, and gives commands that run directly on the
  login node a CPU-time limit (`ulimit -t`, slurm.compute.login_cpu_seconds). A guardrail against
  mistakes, not a security boundary: commands that hand work to SLURM (sbatch, srun, salloc) are
  left alone, because SLURM passes the submitting shell's limits on to the job.
* `hpc-run`: runs a command on a compute node. By default in a shared "worker" job that holds a
  node, so commands start without queueing; the worker ends itself after `idle_minutes` unused.
  Other resources (`--cpus/--mem/--time`) or `--gpu` get an allocation of their own.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..core.config import Config

WORKER_NAME = "overseer-worker"
# Remote commands that hand their work to SLURM (or are file transfers): no CPU limit for them.
DISPATCH_RE = r"(?<![\w./-])(sbatch|srun|salloc|rsync --server|scp -[tf]|sftp-server)(?![\w.-])"
# For the "ran N minutes on the login node" warning: an ssh command that didn't hand its work over.
LOGIN_NODE_CMD = re.compile(r"(?<![\w./-])ssh(?![\w.-])")
DISPATCHED = re.compile(DISPATCH_RE + r"|(?<![\w./-])(hpc-run|scp|rsync|sftp)(?![\w.-])")


def _fill(template: str, **values: str) -> str:
    # @NAME@ placeholders: the templates are full of braces and $ (Python and bash), so no str.format.
    for k, v in values.items():
        template = template.replace(f"@{k.upper()}@", v)
    return template


SSH_WRAPPER = r'''#!@PYTHON@ -I
# The agent's ssh: only through Overseer's connection. Commands that run directly on the login
# node get a CPU-time limit - a guardrail against heavy work there (use hpc-run or sbatch).
import os, re, subprocess, sys

REAL, CONFIG, LIMIT = @REAL@, @CONFIG@, @LIMIT@
DISPATCH = re.compile(@DISPATCH@)
VALUE_OPTS = set("BbcDEeFIiJLlmOopQRSWw")  # ssh options that take a value

args = sys.argv[1:]
i = 0
while i < len(args):  # find the host: the first argument that isn't an option or an option's value
    a = args[i]
    if a == "--":
        i += 1
        break
    if a.startswith("-") and len(a) > 1:
        for k, ch in enumerate(a[1:]):
            if ch in VALUE_OPTS:
                if k == len(a) - 2:
                    i += 1  # its value is the next argument
                break
        i += 1
        continue
    break
argv = [REAL, "-F", CONFIG] + args
remote = " ".join(args[i + 1:])
if LIMIT > 0 and remote and not DISPATCH.search(remote):
    # "exit $?" keeps bash from replacing itself with the last command: then a command killed for its CPU
    # time is reported as status 152 (ssh would only say 255 if the session itself were killed)
    # The soft limit sends SIGXCPU (status 152); the hard one, a bit later, SIGKILL for programs that ignore it.
    limits = "ulimit -H -t %d 2>/dev/null; ulimit -S -t %d 2>/dev/null; " % (LIMIT + 10, LIMIT)
    argv = [REAL, "-F", CONFIG] + args[:i + 1] + [limits + remote + "\nexit $?"]
    rc = subprocess.call(argv)
    hint = "Run heavy work on a compute node: hpc-run -- <command>   (or sbatch)"
    if rc == 152:  # SIGXCPU
        sys.stderr.write("\n[overseer] This hit the login node's CPU limit (%d CPU seconds). %s\n" % (LIMIT, hint))
    elif rc == 137:  # SIGKILL: the hard CPU limit - or memory
        sys.stderr.write("\n[overseer] This was killed on the login node - if it computed for a while, it hit the "
                         "CPU limit (%d CPU seconds). %s\n" % (LIMIT, hint))
    sys.exit(rc)
os.execv(REAL, argv)
'''


HPC_RUN = r'''#!@PYTHON@ -I
# hpc-run: run a command on a compute node of the HPC, not on the login node.
import argparse, shlex, subprocess, sys

USAGE = """
  hpc-run [--dir DIR] -- COMMAND...                 on the shared worker node (started if needed)
  hpc-run --cpus 32 --mem 128G --time 120 -- ...    on a node of its own with these resources
  hpc-run --gpu [--time 60] -- ...                  on a GPU node of its own
  hpc-run --status | --stop                         the shared worker

The command runs in a login shell (modules and PATH as usual) in DIR (default: your home), with
MPLBACKEND=Agg so plots can be saved without a display. Its output and exit code are returned.
"""
SSH, HOST, SETUP, NAME = @SSH@, @HOST@, @SETUP@, @NAME@
C = @COMPUTE@

p = argparse.ArgumentParser(prog="hpc-run", description="run a command on a compute node of the HPC",
                            formatter_class=argparse.RawDescriptionHelpFormatter, epilog=USAGE)
p.add_argument("--dir", help="working directory on the HPC (default: home)")
p.add_argument("--cpus", type=int)
p.add_argument("--mem", help="e.g. 64G")
p.add_argument("--time", type=int, help="minutes")
p.add_argument("--gpu", action="store_true")
p.add_argument("--status", action="store_true", help="show the shared worker")
p.add_argument("--stop", action="store_true", help="end the shared worker")
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
cmd = a.command[1:] if a.command[:1] == ["--"] else a.command

q = shlex.quote
pre = SETUP.strip() + "; " if SETUP.strip() else ""
extra = " ".join(q(x) for x in C["sbatch_args"])


def remote(script):
    return subprocess.call([SSH, HOST, "bash -lc " + q(pre + script)])


if a.status:
    sys.exit(remote('out=$(squeue -h -u "$USER" -n ' + NAME + ' -o "%i %T %M %l %N %r"); '
                    '[ -n "$out" ] && echo "worker: $out" || echo "no worker running"'))
if a.stop:
    sys.exit(remote('scancel -u "$USER" -n ' + NAME + ' && echo "worker stopped"'))
if not cmd:
    p.error("give a command after --")

where = q(a.dir) if a.dir else '"$HOME"'
inner = pre + "cd " + where + " && export MPLBACKEND=Agg && " + shlex.join(cmd)

if a.gpu or a.cpus or a.mem or a.time:  # an allocation of its own
    part = C["gpu_partition"] if a.gpu else C["partition"]
    opts = ["-p", q(part), "-c", str(a.cpus or C["cpus"]), "--mem", q(a.mem or C["mem"]), "-t", str(a.time or 60),
            "--ntasks=1", "--immediate=%d" % (C["wait_minutes"] * 60), "--quiet", extra]
    if a.gpu:
        opts.append("--gres=" + q(C["gpu_gres"]))
    sys.exit(remote("srun " + " ".join(x for x in opts if x) + " bash -lc " + q(inner)))

worker = ('f="$HOME/.overseer/worker-$SLURM_JOB_ID.used"; touch "$f"; '
          'while sleep 60; do [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt IDLE ] && exit 0; done'
          ).replace("IDLE", str(C["idle_minutes"] * 60))
script = """
mkdir -p "$HOME/.overseer"
if command -v flock >/dev/null; then exec 9>"$HOME/.overseer/worker.lock"; flock -w 60 9; fi
JOB=$(squeue -h -u "$USER" -n @@NAME -t RUNNING,PENDING -o %i | head -n1)
if [ -z "$JOB" ]; then
  JOB=$(sbatch --parsable -J @@NAME -p @@PART -c @@CPUS --mem @@MEM -t @@MINUTES -o "$HOME/.overseer/worker-%j.log" @@EXTRA --wrap @@WRAP) || exit 75
  JOB=${JOB%%;*}
  echo "[hpc-run] started a compute worker: job $JOB (it ends itself after @@IDLE_MIN min unused; it's Overseer's helper, not a job to register or watch)" >&2
fi
exec 9>&-
end=$((SECONDS + @@WAIT))
while :; do
  ST=$(squeue -h -j "$JOB" -o %T 2>/dev/null)
  [ "$ST" = RUNNING ] && break
  if [ -z "$ST" ]; then echo "[hpc-run] worker job $JOB has ended; run hpc-run again to start a new one" >&2; exit 75; fi
  if [ $SECONDS -ge $end ]; then
    echo "[hpc-run] no compute node yet: worker job $JOB is $ST ($(squeue -h -j "$JOB" -o %r)). It stays in the queue; try again later, or use sbatch." >&2
    exit 75
  fi
  [ -z "$SAID" ] && { echo "[hpc-run] waiting for a compute node (job $JOB, $ST)..." >&2; SAID=1; }
  sleep 5
done
F="$HOME/.overseer/worker-$JOB.used"
touch "$F"; (while sleep 60; do touch "$F"; done) </dev/null >/dev/null 2>&1 & HB=$!  # keeps the worker busy
trap 'kill $HB 2>/dev/null' EXIT
srun --jobid="$JOB" --overlap --ntasks=1 --quiet bash -lc @@INNER
"""
for key, value in [("NAME", NAME), ("PART", q(C["partition"])), ("CPUS", str(C["cpus"])), ("MEM", q(C["mem"])),
                   ("MINUTES", str(int(C["worker_hours"] * 60))), ("EXTRA", extra), ("WRAP", q(worker)),
                   ("IDLE_MIN", str(C["idle_minutes"])), ("WAIT", str(C["wait_minutes"] * 60)), ("INNER", q(inner))]:
    script = script.replace("@@" + key, value)
sys.exit(remote(script))
'''


def write_agent_tools(cfg: Config, bin_dir: Path, real_ssh: str, write) -> None:
    """Write the agent's ssh wrapper (with the login-node CPU limit) and hpc-run."""
    c = cfg.slurm.compute
    py = sys.executable
    write(bin_dir / "ssh", _fill(SSH_WRAPPER, python=py, real=repr(real_ssh), config=repr(str(cfg.agent_ssh_config)),
                                 limit=str(int(c.login_cpu_seconds)), dispatch=repr(DISPATCH_RE)), 0o700)
    compute = {"partition": c.partition, "cpus": int(c.cpus), "mem": c.mem, "worker_hours": float(c.worker_hours),
               "idle_minutes": int(c.idle_minutes), "wait_minutes": int(c.wait_minutes),
               "gpu_partition": c.gpu_partition, "gpu_gres": c.gpu_gres, "sbatch_args": list(c.sbatch_args)}
    write(bin_dir / "hpc-run", _fill(HPC_RUN, python=py, ssh=repr(str(bin_dir / "ssh")), host=repr(cfg.ssh.host),
                                     setup=repr(cfg.slurm.setup), name=repr(WORKER_NAME), compute=repr(compute)), 0o700)
