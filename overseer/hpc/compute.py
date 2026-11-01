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
RUN_NAME = "overseer-run"  # hpc-run's allocations of their own (--cpus/--mem/--time/--gpu): overseer-run-<id>


def is_helper(name: str | None) -> bool:
    """hpc-run's jobs - the shared worker, and a run's allocation of its own: Overseer's helpers, never your jobs."""
    return bool(name) and (name in (WORKER_NAME, RUN_NAME) or name.startswith(RUN_NAME + "-"))


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
  hpc-run --status                                  the shared worker, and runs waiting for or on a node
  hpc-run --log ID / --cancel ID                    a run that was still going: the rest of its output / stop it
  hpc-run --stop                                    end the shared worker

The command runs in a login shell (modules and PATH as usual) in DIR (default: your home), with
MPLBACKEND=Agg so plots can be saved without a display, and OMP/OPENBLAS/MKL_NUM_THREADS set to its
CPUs unless you set them. Its output (stdout and stderr together) and exit code are returned.

A command still running after --return-after minutes (default @@RET) keeps running on its node: hpc-run gives
back the output so far, how long it has run and the CPU time it used, and the run's id (exit code 124). Give
your tool call a timeout above that. hpc-run --log ID follows it again.
"""
SSH, HOST, SETUP, NAME, RUN_NAME = @SSH@, @HOST@, @SETUP@, @NAME@, @RUN_NAME@
C = @COMPUTE@

p = argparse.ArgumentParser(prog="hpc-run", description="run a command on a compute node of the HPC",
                            formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog=USAGE.replace("@@RET", "%g" % C["return_after_minutes"]))
p.add_argument("--dir", help="working directory on the HPC (default: home)")
p.add_argument("--cpus", type=int)
p.add_argument("--mem", help="e.g. 64G")
p.add_argument("--time", type=int, help="minutes")
p.add_argument("--gpu", action="store_true")
p.add_argument("--status", action="store_true", help="show the shared worker and the runs waiting or running")
p.add_argument("--stop", action="store_true", help="end the shared worker")
p.add_argument("--log", metavar="ID", help="a run that was still going: its output since you last looked, and its end")
p.add_argument("--cancel", metavar="ID", help="stop a run that's still going")
p.add_argument("--return-after", type=float, metavar="MIN", default=C["return_after_minutes"],
               help="give the output back after this many minutes if the command is still running (default "
                    "%(default)g); it carries on, and --log follows it")
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
cmd = a.command[1:] if a.command[:1] == ["--"] else a.command

q = shlex.quote
pre = SETUP.strip() + "; " if SETUP.strip() else ""
extra = " ".join(q(x) for x in C["sbatch_args"])
RETURN = max(5, int(a.return_after * 60))

# Every run is a folder in ~/.overseer/runs on the HPC: its command, output (stdout and stderr together), exit code,
# node and SLURM step. It runs detached from hpc-run (setsid), so it isn't lost when hpc-run gives back control or
# is killed (e.g. by the agent's tool timeout); hpc-run follows its output.
RUNS = r"""
R="$HOME/.overseer/runs"
mkdir -p "$R"
still() {  # (a run past --return-after: where it is, and what it has used)
  local D=$1 id node step cpu mins
  id=$(basename "$D"); node=$(cat "$D/node" 2>/dev/null); step=$(cat "$D/step" 2>/dev/null)
  mins=$(( ($(date +%s) - $(cat "$D/start")) / 60 ))
  [ -n "$step" ] && cpu=$(sstat -n -P -j "$step" --format=AveCPU,MaxRSS 2>/dev/null | head -n1)
  if [ -z "$step" ]; then
    echo "[hpc-run] still not started after $mins min: the node never began the command (its login shell or slurm.setup may hang there). It carries on; hpc-run --log $id follows it, hpc-run --cancel $id stops it." >&2
  else
    echo "[hpc-run] still running after $mins min${node:+ on $node} (step $step${cpu:+; CPU time so far ${cpu%%|*}, memory ${cpu##*|}}). It carries on - it hasn't failed, don't start it again. Its output so far is above; hpc-run --log $id gives the rest (waiting up to @@RETURN_MIN min again), hpc-run --cancel $id stops it." >&2
  fi
}
follow() {  # (a run's output from where it was last read, until it ends or it's time to give control back)
  local D=$1 off n i=0 end=$((SECONDS + @@RETURN))
  off=$(cat "$D/seen" 2>/dev/null || echo 0)
  while :; do
    local fin=""; [ -f "$D/rc" ] && fin=1
    n=$(stat -c %s "$D/out" 2>/dev/null || echo 0)
    if [ "$n" -gt "$off" ]; then tail -c +$((off + 1)) "$D/out" | head -c $((n - off)); off=$n; echo "$off" > "$D/seen"; fi
    [ -n "$fin" ] && return "$(cat "$D/rc")"
    if [ $SECONDS -ge $end ]; then still "$D"; return 124; fi
    i=$((i + 1)); [ $i -le 20 ] && sleep 0.5 || sleep 2
  done
}
new_run() {  # (a folder for a new run; old ones go after a week)
  find "$R" -mindepth 1 -maxdepth 1 -mtime +7 -exec rm -rf {} + 2>/dev/null
  ID=$(date +%Y%m%d-%H%M%S)-$$; D="$R/$ID"; mkdir -p "$D"; date +%s > "$D/start"
  printf '%s\n' @@CMDLINE > "$D/cmd"
  export HPC_RUN_DIR="$D"
}
""".replace("@@RETURN_MIN", "%g" % (RETURN / 60)).replace("@@RETURN", str(RETURN)).replace(
    "@@CMDLINE", q(shlex.join(cmd) if cmd else ""))


def remote(script):
    return subprocess.call([SSH, HOST, "bash -lc " + q(pre + script)])


if a.status:  # (squeue's %j is the name; the runs' names have a suffix, so they're picked out with awk)
    sys.exit(remote(RUNS + 'out=$(squeue -h -u "$USER" -o "%i|%j|%T|%M|%l|%N|%r"); '
                    'w=$(echo "$out" | awk -F"|" \'$2=="' + NAME + '"\'); '
                    'r=$(echo "$out" | awk -F"|" \'$2 ~ /^' + RUN_NAME + '(-|$)/\'); '
                    '[ -n "$w" ] && echo "shared worker: $w" | tr "|" " " || echo "shared worker: none running"; '
                    '[ -n "$r" ] && echo "$r" | tr "|" " " | sed "s/^/run on a node of its own: /" '
                    '|| echo "runs on a node of their own: none"; '
                    'for D in "$R"/*/; do [ -f "$D/rc" ] || [ ! -f "$D/cmd" ] && continue; D=${D%/}; '
                    'echo "still going: $(basename "$D") ($(( ($(date +%s) - $(cat "$D/start")) / 60 )) min'
                    '$([ -f "$D/node" ] && echo " on $(cat "$D/node")")): $(cat "$D/cmd")"; done'))
if a.stop:
    sys.exit(remote('scancel -u "$USER" -n ' + NAME + ' && echo "worker stopped"'))
if a.log or a.cancel:
    rid = a.log or a.cancel
    if not rid.replace("-", "").isalnum():
        p.error("not a run id: " + rid)
    look = RUNS + 'D="$R/' + rid + '"; [ -f "$D/cmd" ] || { echo "[hpc-run] no run ' + rid + ' (hpc-run --status lists those still going)" >&2; exit 2; }; '
    if a.cancel:
        sys.exit(remote(look + '[ -f "$D/rc" ] && { echo "[hpc-run] it had already ended (exit code $(cat "$D/rc"))"; exit 0; }; '
                        'S=$(cat "$D/step" 2>/dev/null); N=$(cat "$D/name" 2>/dev/null); '
                        'kill -- -"$(cat "$D/pid" 2>/dev/null)" 2>/dev/null; '  # (started from this login node)
                        'if [ -n "$N" ]; then scancel -n "$N"; elif [ -n "$S" ]; then scancel "$S" 2>/dev/null; fi; '
                        'sleep 1; [ -f "$D/rc" ] || echo 143 > "$D/rc"; echo "[hpc-run] cancelled ' + rid + '"'))
    sys.exit(remote(look + 'follow "$D"; rc=$?; [ $rc -ne 124 ] && echo "[hpc-run] it ended: exit code $rc" >&2; exit $rc'))
if not cmd:
    p.error("give a command after --")

where = q(a.dir) if a.dir else '"$HOME"'
# (the node notes where the run is first - so a hang in the login shell or slurm.setup shows as "not started"; numpy,
# torch & co. get as many threads as the CPUs SLURM gave, unless set: by default they start one per CPU of the node)
inner = ('echo "$SLURM_JOB_ID.${SLURM_STEP_ID:-0}" > "$HPC_RUN_DIR/step"; hostname > "$HPC_RUN_DIR/node"; ' + pre
         + 'n=${SLURM_CPUS_PER_TASK:-1}; export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$n} '
           'OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$n} MKL_NUM_THREADS=${MKL_NUM_THREADS:-$n} '
           'NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$n}; '
         + "cd " + where + " && export MPLBACKEND=Agg && " + shlex.join(cmd))
# (the run: detached, its output and exit code into its folder; HB keeps the shared worker from ending as unused)
LAUNCH = ('setsid bash -c \'D=$1; F=$2; shift 2; echo $$ > "$D/pid"; '
          '[ -n "$F" ] && { (while sleep 60; do touch "$F"; done) </dev/null >/dev/null 2>&1 & HB=$!; }; '
          '"$@" > "$D/out" 2>&1 < /dev/null; echo $? > "$D/rc"; [ -n "$HB" ] && kill $HB\' _ "$D" ')

WAIT = int(C["wait_minutes"] * 60)
if a.gpu or a.cpus or a.mem or a.time:  # an allocation of its own
    # srun waits for it without a word while it's queued: say what SLURM says (the job, its state and why it waits)
    # until the node is there - or give up after the wait, cancelling the request.
    part = C["gpu_partition"] if a.gpu else C["partition"]
    opts = ["-p", q(part), "-c", str(a.cpus or C["cpus"]), "--mem", q(a.mem or C["mem"]), "-t", str(a.time or 60),
            "--ntasks=1", "--quiet", extra]
    if a.gpu:
        opts.append("--gres=" + q(C["gpu_gres"]))
    what = "%s, %s CPU(s), %s, %s min" % (part, a.cpus or C["cpus"], a.mem or C["mem"], a.time or 60)
    script = RUNS + """
new_run
NAME=@@RUN-$ID; echo "$NAME" > "$D/name"
@@LAUNCH "" srun -J "$NAME" @@OPTS bash -lc @@INNER > /dev/null 2>&1 < /dev/null &
job() { squeue -h -u "$USER" -n "$NAME" -o "%i|%T|%r|%N" 2>/dev/null | head -n1; }
# (stopped while still in the queue: the request goes too. Once it runs, it carries on)
trap '[ -z "$got" ] && scancel -n "$NAME" 2>/dev/null; exit 130' INT TERM HUP
end=$((SECONDS + @@WAIT)); said=""; got=""; last=0
while [ ! -f "$D/rc" ]; do
  IFS="|" read -r JOB ST WHY NODE <<EOF_
$(job)
EOF_
  if [ "$ST" = RUNNING ]; then
    [ -n "$said" ] && echo "[hpc-run] got a node after $((SECONDS / 60)) min: $NODE (job $JOB)" >&2 \
      || echo "[hpc-run] running on $NODE (job $JOB)" >&2
    got=1; break
  fi
  if [ -n "$JOB" ] && [ -z "$said" ]; then
    echo "[hpc-run] asked SLURM for a node of its own (@@WHAT): job $JOB is $ST ($WHY); waiting up to @@WAIT_MIN min..." >&2
    said=1; last=$SECONDS
  elif [ -n "$said" ] && [ $((SECONDS - last)) -ge 60 ]; then
    echo "[hpc-run] still waiting: job $JOB is $ST ($WHY), $((SECONDS / 60)) min so far" >&2; last=$SECONDS
  fi
  if [ $SECONDS -ge $end ]; then
    [ -n "$JOB" ] && scancel "$JOB" 2>/dev/null || scancel -n "$NAME" 2>/dev/null
    echo "[hpc-run] no node within @@WAIT_MIN min (job ${JOB:-?} was ${ST:-queued}: ${WHY:-?}), so the request is cancelled. For work that can wait in the queue: sbatch (and ov watch); for small work: hpc-run without --cpus/--mem/--time (the shared worker)." >&2
    exit 75
  fi
  [ $SECONDS -lt 10 ] && sleep 1 || sleep 5  # (soon after asking, then gently: squeue is shared by everyone)
done
trap - INT TERM HUP
follow "$D"; rc=$?
if [ -n "$said" ] && [ -z "$got" ] && [ $rc -ne 124 ]; then  # (it got its node and finished between two looks)
  [ $rc -eq 0 ] && echo "[hpc-run] got a node after $((SECONDS / 60)) min (job $JOB)" >&2 \
    || echo "[hpc-run] job $JOB ended with exit code $rc" >&2
fi
exit $rc
"""
    for key, value in [("RUN", RUN_NAME), ("OPTS", " ".join(x for x in opts if x)), ("INNER", q(inner)),
                       ("WAIT_MIN", "%g" % C["wait_minutes"]), ("WAIT", str(WAIT)), ("WHAT", what),
                       ("LAUNCH", LAUNCH)]:
        script = script.replace("@@" + key, value)
    sys.exit(remote(script))

worker = ('f="$HOME/.overseer/worker-$SLURM_JOB_ID.used"; touch "$f"; '
          'while sleep 60; do [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt IDLE ] && exit 0; done'
          ).replace("IDLE", str(C["idle_minutes"] * 60))
script = RUNS + """
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
new_run
F="$HOME/.overseer/worker-$JOB.used"; touch "$F"
# (a step with the worker's CPUs: without -c it would get one)
@@LAUNCH "$F" srun --jobid="$JOB" --overlap --ntasks=1 -c @@CPUS --quiet bash -lc @@INNER > /dev/null 2>&1 < /dev/null &
echo "[hpc-run] running on $(squeue -h -j "$JOB" -o %N 2>/dev/null) (the shared worker, job $JOB)" >&2
follow "$D"
"""
for key, value in [("NAME", NAME), ("PART", q(C["partition"])), ("CPUS", str(C["cpus"])), ("MEM", q(C["mem"])),
                   ("MINUTES", str(int(C["worker_hours"] * 60))), ("EXTRA", extra), ("WRAP", q(worker)),
                   ("IDLE_MIN", str(C["idle_minutes"])), ("WAIT", str(WAIT)), ("INNER", q(inner)),
                   ("LAUNCH", LAUNCH)]:
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
               "idle_minutes": int(c.idle_minutes), "wait_minutes": float(c.wait_minutes),
               "gpu_partition": c.gpu_partition, "gpu_gres": c.gpu_gres, "sbatch_args": list(c.sbatch_args),
               "return_after_minutes": float(c.return_after_minutes)}
    write(bin_dir / "hpc-run", _fill(HPC_RUN, python=py, ssh=repr(str(bin_dir / "ssh")), host=repr(cfg.ssh.host),
                                     setup=repr(cfg.slurm.setup), name=repr(WORKER_NAME), run_name=repr(RUN_NAME),
                                     compute=repr(compute)), 0o700)
