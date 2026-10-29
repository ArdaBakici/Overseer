#!/usr/bin/env python3
"""Minimal stand-in for squeue/sacct/scontrol/sbatch/scancel used by the integration tests.

State lives in the JSON file named by $FAKE_SLURM_STATE.
"""
import json
import os
import sys

STATE = os.environ["FAKE_SLURM_STATE"]
prog = os.path.basename(sys.argv[0])
args = sys.argv[1:]


def load():  # (a test may be rewriting it right now: read it again)
    import time
    for _ in range(50):
        try:
            with open(STATE) as fh:
                return json.load(fh)
        except ValueError:
            time.sleep(0.02)
    with open(STATE) as fh:
        return json.load(fh)


def save(st):  # atomically: other fake commands may be reading it at the same time
    tmp = f"{STATE}.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(st, fh)
    os.replace(tmp, STATE)


def opt(name):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


st = load()
jobs = st["jobs"]

FIELDS = {"%i": lambda jid, j: jid, "%T": lambda jid, j: j["state"], "%j": lambda jid, j: j["name"],
          "%r": lambda jid, j: j.get("reason", "None"), "%M": lambda jid, j: j.get("elapsed", "0:10"),
          "%l": lambda jid, j: j.get("time_limit", "1:00:00"), "%N": lambda jid, j: j.get("nodelist", "n1")}

import re as _re
_tokens = _re.findall(r"%\w", opt("-o") or "")
if prog == "squeue" and "--start" not in args and _tokens and all(t in FIELDS for t in _tokens):  # (hpc-run's)
    name, states, sel = opt("-n"), (opt("-t") or "PENDING,RUNNING").split(","), opt("-j")
    for jid, j in jobs.items():
        if j["state"] in states and (not name or j["name"] == name) and (not sel or jid == sel):
            line = opt("-o")
            for k, f in FIELDS.items():
                line = line.replace(k, str(f(jid, j)))
            print(line)
elif prog == "squeue" and "--start" in args:  # SLURM's estimated start of pending jobs
    for jid, j in jobs.items():
        if j["state"] == "PENDING":
            print(f"{jid}|{j.get('est_start', 'N/A')}|{j.get('reason', 'Priority')}")
elif prog == "squeue":
    for jid, j in jobs.items():
        if j["state"] in ("PENDING", "RUNNING"):
            where = j.get("nodelist", "n1") if j["state"] == "RUNNING" else "(Priority)"
            print("|".join([jid, j["name"], j["state"], j.get("partition", "gpu"), j.get("elapsed", "0:10"),
                            j.get("time_limit", "1:00:00"), "1", where, "gres/gpu:1", "2026-10-07T10:00:00", "N/A"]))
elif prog == "sacct" and "Submit" in (opt("-o") or ""):  # jobs submitted lately (who submitted what)
    for jid, j in jobs.items():
        print("|".join([jid, j.get("submit", "2026-01-01T00:00:00"), j["name"]]))
elif prog == "sacct":
    sel = opt("-j")
    for jid, j in jobs.items():
        if sel and jid != sel and jid.split("_")[0] != sel:  # like sacct: an array's id lists its tasks
            continue
        if not sel and j.get("ended_long_ago"):  # outside `sacct -S now-N`'s window
            continue
        if "-X" in args:
            print("|".join([jid, j["name"], j["state"], j.get("exit", "0:0"), j.get("elapsed", "00:10:00"),
                            j.get("time_limit", "01:00:00"), j.get("start", "s"), j.get("end", "e"),
                            j.get("partition", "gpu"), "n1", j.get("tres", "cpu=4,gres/gpu=1"), j.get("workdir", "")]))
        else:
            print("|".join([jid, j["name"], j["state"], j.get("exit", "0:0"), "00:10:00", "", "16G", "cpu=4,gres/gpu=1", "n1"]))
            print("|".join([jid + ".batch", "batch", j["state"], j.get("exit", "0:0"), "00:10:00", "2048000K", "", "cpu=4", "n1"]))
elif prog == "scontrol":
    if args[:2] == ["show", "job"]:
        jid = args[-1]
        # like scontrol: an array's id shows each of its tasks; for its pending tasks (123_[2-4]) %a is NO_VAL
        found = [(jid, jobs[jid])] if jid in jobs else [(k, v) for k, v in jobs.items() if k.split("_")[0] == jid]
        found = [(k, v) for k, v in found if not v.get("scontrol_gone")]
        if not found:
            sys.exit(1)
        import getpass
        for k, j in found:
            task = k.split("_", 1)[1] if "_" in k else ""
            a = task if task.isdigit() else "4294967294"
            arr = f"ArrayJobId={k.split('_')[0]} ArrayTaskId={task.strip('[]')} " if task else ""
            print(f"JobId={k} {arr}JobName={j['name']} UserId={j.get('user') or getpass.getuser()}(1000) "
                  f"JobState={j['state']} WorkDir={j.get('workdir', '')} "
                  f"StdErr={j.get('stderr', '').replace('%a', a)} StdOut={j.get('stdout', '').replace('%a', a)} "
                  "TimeLimit=01:00:00")
    elif args[:2] == ["show", "node"]:
        for name, n in st.get("nodes", {}).items():
            print(f"NodeName={name} Arch=x86_64 CoresPerSocket=32 CPUAlloc={n.get('cpu_alloc', 0)} "
                  f"CPUTot={n.get('cpus', 64)} Gres={n.get('gres', '(null)')} NodeAddr={name} "
                  f"OS=Linux 4.18.0 #1 SMP RealMemory={n.get('mem', 512000)} AllocMem={n.get('mem_alloc', 0)} "
                  f"State={n.get('state', 'IDLE')} Partitions={n.get('partitions', 'compute')} "
                  f"CfgTRES=cpu=64 AllocTRES={n.get('alloc_tres', '')}")
    elif args[:2] == ["write", "batch_script"]:
        j = jobs.get(args[2])
        if not j:
            sys.exit(1)
        sys.stdout.write(j.get("script", "#!/bin/bash\n"))
elif prog == "sbatch":
    st["next_id"] += 1
    jid = str(st["next_id"])
    if opt("--wrap"):
        script = "#!/bin/sh\n" + opt("--wrap")
    else:
        with open(os.path.join(os.getcwd(), args[-1])) as fh:
            script = fh.read()
    chdir = opt("--chdir") or os.getcwd()
    out = (opt("--output") or opt("-o") or f"{chdir}/slurm-%j.out").replace("%j", jid)
    err = (opt("--error") or out).replace("%j", jid)
    name = opt("-J") or opt("--job-name") or "submitted"
    # hpc-run's worker gets a node right away here (a real queue may take a while)
    import time
    jobs[jid] = {"name": name, "state": "RUNNING" if name == "overseer-worker" else "PENDING", "workdir": chdir,
                 "stdout": out, "stderr": err, "script": script, "submit": time.strftime("%Y-%m-%dT%H:%M:%S")}
    st.setdefault("submissions", []).append(args)
    save(st)
    print(jid if "--parsable" in args else f"Submitted batch job {jid}")
elif prog == "srun":
    if any("nvidia-smi" in a for a in args):  # srun --jobid=N --overlap ... nvidia-smi --query-gpu=...
        jid = (opt("--jobid") or "").split("_")[0]
        st.setdefault("gpu_samples", []).append(jid)
        save(st)
        util, mem = st.get("gpu_util", {}).get(jid, ["87 %", "31234 MiB"])
        if util == "fail":
            sys.exit(1)
        print(f"0, NVIDIA A100-SXM4-80GB, {util}, {mem}, 81920 MiB")
        sys.exit(0)
    if "bash" not in args:
        sys.exit(1)
    if opt("-J"):  # hpc-run --cpus/--mem/--time/--gpu: an allocation of its own, queued for st["srun_wait"] s
        import time
        st["next_id"] += 1
        jid = str(st["next_id"])
        wait = float(st.get("srun_wait", 0))
        jobs[jid] = {"name": opt("-J"), "state": "PENDING" if wait else "RUNNING", "reason": "Priority",
                     "nodelist": "cn015"}
        save(st)
        end = time.time() + wait
        while time.time() < end:
            time.sleep(0.2)
            if load()["jobs"][jid]["state"] == "CANCELLED":
                print(f"srun: error: Job allocation {jid} has been revoked", file=sys.stderr)
                sys.exit(1)
        st = load()
        st["jobs"][jid]["state"] = "RUNNING"
        save(st)
        jobs = st["jobs"]
    st.setdefault("steps", []).append(args)  # hpc-run: run the command here, as if on the node
    save(st)
    import subprocess
    rc = subprocess.call(args[args.index("bash"):])
    if opt("-J"):
        st = load()
        st["jobs"][jid]["state"] = "COMPLETED"
        save(st)
    sys.exit(rc)
elif prog == "scancel":
    name = opt("-n")
    for jid in ([j for j, v in jobs.items() if v["name"] == name] if name else [args[0]]):
        jobs[jid]["state"] = "CANCELLED"
    save(st)
elif prog == "sinfo":
    for name, p in st.get("partitions", {}).items():
        print(f"{name}|{p.get('avail', 'up')}|{p.get('time', '2-00:00:00')}")
elif prog == "sshare":
    for account, fs in st.get("fairshare", {}).items():
        print(f"{account}|{fs}")
elif prog == "seff":
    sys.exit(1)
