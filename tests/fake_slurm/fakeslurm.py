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

if prog == "squeue":
    for jid, j in jobs.items():
        if j["state"] in ("PENDING", "RUNNING"):
            where = j.get("nodelist", "n1") if j["state"] == "RUNNING" else "(Priority)"
            print("|".join([jid, j["name"], j["state"], j.get("partition", "gpu"), j.get("elapsed", "0:10"),
                            j.get("time_limit", "1:00:00"), "1", where, "gres/gpu:1", "2026-10-07T10:00:00", "N/A"]))
elif prog == "sacct":
    sel = opt("-j")
    for jid, j in jobs.items():
        if sel and jid != sel:
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
        j = jobs.get(jid)
        if not j:
            sys.exit(1)
        print(f"JobId={jid} JobName={j['name']} JobState={j['state']} WorkDir={j.get('workdir', '')} "
              f"StdErr={j.get('stderr', '')} StdOut={j.get('stdout', '')} TimeLimit=01:00:00")
    elif args[:2] == ["write", "batch_script"]:
        j = jobs.get(args[2])
        if not j:
            sys.exit(1)
        sys.stdout.write(j.get("script", "#!/bin/bash\n"))
elif prog == "sbatch":
    st["next_id"] += 1
    jid = str(st["next_id"])
    script_path = args[-1]
    with open(os.path.join(os.getcwd(), script_path)) as fh:
        script = fh.read()
    chdir = opt("--chdir") or os.getcwd()
    out = (opt("--output") or f"{chdir}/slurm-%j.out").replace("%j", jid)
    err = (opt("--error") or out).replace("%j", jid)
    jobs[jid] = {"name": "submitted", "state": "PENDING", "workdir": chdir,
                 "stdout": out, "stderr": err, "script": script}
    st.setdefault("submissions", []).append(args)
    save(st)
    print(jid if "--parsable" in args else f"Submitted batch job {jid}")
elif prog == "scancel":
    jobs[args[0]]["state"] = "CANCELLED"
    save(st)
elif prog == "seff":
    sys.exit(1)
