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


def load():
    with open(STATE) as fh:
        return json.load(fh)


def save(st):
    with open(STATE, "w") as fh:
        json.dump(st, fh)


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

if prog == "squeue" and "|" not in (opt("-o") or "|"):  # filtered queries with a simple format (hpc-run)
    name, states, sel = opt("-n"), (opt("-t") or "PENDING,RUNNING").split(","), opt("-j")
    for jid, j in jobs.items():
        if j["state"] in states and (not name or j["name"] == name) and (not sel or jid == sel):
            line = opt("-o")
            for k, f in FIELDS.items():
                line = line.replace(k, str(f(jid, j)))
            print(line)
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
        if "-X" in args:
            print("|".join([jid, j["name"], j["state"], j.get("exit", "0:0"), j.get("elapsed", "00:10:00"),
                            j.get("time_limit", "01:00:00"), "s", "e", j.get("partition", "gpu"), "n1",
                            "cpu=4,gres/gpu=1", j.get("workdir", "")]))
        else:
            print("|".join([jid, j["name"], j["state"], j.get("exit", "0:0"), "00:10:00", "", "16G", "cpu=4,gres/gpu=1", "n1"]))
            print("|".join([jid + ".batch", "batch", j["state"], j.get("exit", "0:0"), "00:10:00", "2048000K", "", "cpu=4", "n1"]))
elif prog == "scontrol":
    if args[:2] == ["show", "job"]:
        jid = args[-1]
        j = jobs.get(jid) or next((v for k, v in jobs.items() if k.split("_")[0] == jid), None)  # an array's tasks
        if not j:
            sys.exit(1)
        import getpass
        print(f"JobId={jid} JobName={j['name']} UserId={j.get('user') or getpass.getuser()}(1000) "
              f"JobState={j['state']} WorkDir={j.get('workdir', '')} "
              f"StdErr={j.get('stderr', '')} StdOut={j.get('stdout', '')} TimeLimit=01:00:00")
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
    if "nvidia-smi" in args:  # srun --jobid=N --overlap ... nvidia-smi --query-gpu=...
        print("0, NVIDIA A100-SXM4-80GB, 87 %, 31234 MiB, 81920 MiB")
        sys.exit(0)
    if "bash" not in args:
        sys.exit(1)
    st.setdefault("steps", []).append(args)  # hpc-run: run the command here, as if on the node
    save(st)
    import subprocess
    sys.exit(subprocess.call(args[args.index("bash"):]))
elif prog == "scancel":
    name = opt("-n")
    for jid in ([j for j, v in jobs.items() if v["name"] == name] if name else [args[0]]):
        jobs[jid]["state"] = "CANCELLED"
    save(st)
elif prog == "seff":
    sys.exit(1)
