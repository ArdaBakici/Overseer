#!/usr/bin/env python3
"""Stand-in for `claude` and `codex` used by the integration tests.

Logs argv/env/stdin to $FAKE_CLI_LOG (JSON lines) and emits a canned event stream:
the agent runs `ssh hpc ...sbatch...`, sbatch answers with a job id, then a final message.
A prompt containing FORCE_SLEEP makes it hang (for /stop tests); FORCE_FAIL (or a file $FAKE_CLI_LOG.fail) makes it exit with an error;
FORCE_SLOW pauses between events (to watch the live progress).

Autopilot (fake claude, when the system prompt says autopilot is on): every run submits a new job
id and watches it with `ov watch`, like a real agent. Markers in the prompt (the goal, or job logs
quoted in a wake-up's report) steer it: FAKE_DONE (`ov auto done`), FAKE_HELP (`ov auto help`),
FAKE_NOJOB (submit nothing), FAKE_TWO (submit two jobs), FAKE_NOWATCH (submit without watching);
FAKE_EVERY in the goal adds check-ins; in a check-in, FAKE_DIVERGE makes it scancel and resubmit.
Lines "FAKE_OV <command>" make it run those commands (the `ov` tool) and answer with their output.
FAKE_COMPACT: the CLI compacts the conversation on its own during the run; a prompt starting with
/compact (claude) or `codex app-server` + thread/compact/start compacts it on request.
"""
import re
import json
import os
import sys
import time
import uuid

prog = os.path.basename(sys.argv[0])
args = sys.argv[1:]
stdin = "" if sys.stdin.isatty() or args[:1] == ["app-server"] else sys.stdin.read()  # (app-server: a conversation)
with open(os.environ["FAKE_CLI_LOG"], "a") as fh:
    fh.write(json.dumps({"prog": prog, "argv": args, "env": dict(os.environ), "stdin": stdin, "cwd": os.getcwd()}) + "\n")


def emit(ev):
    print(json.dumps(ev), flush=True)
    if "FORCE_SLOW" in stdin:
        time.sleep(0.3)


if "login" in args:
    print("logged in")
    sys.exit(0)

# The CLI's own version and updater (state in $FAKE_CLI_LOG.<prog>.version; $FAKE_CLI_LOG.update-fail fails it).
VERSIONS = {"claude": ("2.1.197 (Claude Code)", "2.1.300 (Claude Code)"), "codex": ("codex-cli 0.150.0", "codex-cli 0.161.0")}
version_file = os.environ["FAKE_CLI_LOG"] + f".{prog}.version"
if args == ["--version"]:
    print(open(version_file).read() if os.path.exists(version_file) else VERSIONS[prog][0])
    sys.exit(0)
if args == ["update"]:
    if os.path.exists(os.environ["FAKE_CLI_LOG"] + ".update-fail"):
        print("Error: could not reach the update server")
        sys.exit(1)
    current = open(version_file).read() if os.path.exists(version_file) else VERSIONS[prog][0]
    if current == VERSIONS[prog][1]:
        print(f"{prog} is up to date ({current})")
    else:
        open(version_file, "w").write(VERSIONS[prog][1])
        print(f"Successfully updated from {current} to {VERSIONS[prog][1]}")
    sys.exit(0)

if prog == "claude" and "FORCE_APIERROR" in stdin:  # an API error, reported as the run's result
    sid = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": "claude-opus-5-5"})
    emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": True,
          "result": "API Error: 400 Claude Code 2.1.197 does not support this model; version 2.1.280 or newer is "
                    "required. Run 'claude update', or update the Claude desktop app, then try again."})
    sys.exit(1)
if "FORCE_SLEEP" in stdin:
    time.sleep(60)
if "FORCE_FAIL" in stdin or os.path.exists(os.environ["FAKE_CLI_LOG"] + ".fail"):
    print("boom: model unavailable", file=sys.stderr)
    sys.exit(3)

def ov(cmd):
    """Run an `ov` command like an agent would (Overseer's generated client, on PATH)."""
    import subprocess
    res = subprocess.run("ov " + cmd, shell=True, capture_output=True, text=True)
    return res.stdout + res.stderr


def fake_submit(jid):
    """sbatch, as far as Overseer can tell: the job exists in (fake) SLURM, and the output says so."""
    state_file = os.environ.get("FAKE_SLURM_STATE")
    if state_file:
        with open(state_file) as fh:
            st = json.load(fh)
        st["jobs"].setdefault(str(jid), {"name": f"run{jid}", "state": "PENDING"})
        with open(state_file, "w") as fh:
            json.dump(st, fh)
    return f"Submitted batch job {jid}\n"


def autopilot_run():
    with open(os.environ["FAKE_CLI_LOG"]) as fh:
        n = sum(1 for _ in fh)  # runs so far: makes job ids unique
    sid = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
    system = " ".join(args)
    every = " --every 1h" if "FAKE_EVERY" in system else ""
    steps, final = [], "Submitted the next experiment."

    def submit_and_watch(count):
        ids = [6000 + n * 10 + i for i in range(count)]
        for jid in ids:
            steps.append(("ssh hpc 'cd ~/proj && sbatch train.sh'", lambda jid=jid: fake_submit(jid)))
        if "FAKE_NOWATCH" not in stdin:
            cmd = "watch " + " ".join(map(str, ids)) + every
            steps.append(("ov " + cmd, lambda: ov(cmd)))

    if "Autopilot check-in" in stdin:
        final = "Looks healthy."
        if "FAKE_DIVERGE" in stdin:
            jid = re.search(r"<job id='(\d+)'", stdin).group(1)
            steps.append((f"ssh hpc scancel {jid}", lambda: ""))
            submit_and_watch(1)
            final = f"Job {jid} diverged: cancelled it and resubmitted with a lower lr."
    elif "FAKE_DONE" in stdin:
        steps.append(("ov auto done best lr is 3e-4", lambda: ov("auto done best lr is 3e-4")))
        final = "Reached the goal."
    elif "FAKE_HELP" in stdin:
        steps.append(("ov auto help should I try a bigger model?", lambda: ov("auto help should I try a bigger model?")))
        final = "Not sure how to continue."
    elif "FAKE_SCRIPT" in stdin:  # submits through its own script: no "sbatch" in the command
        steps = [("ssh hpc 'cd ~/rf && python native7_cli.py submit --gpu a100'",
                  lambda: "native7: building manifest\nSubmitted batch job 7777\n(previous attempt was job 1234)\n")]
        final = "Submitted 7777; also queued 7778 for the seed-1 rerun."
    elif "FAKE_NOJOB" not in stdin:
        submit_and_watch(2 if "FAKE_TWO" in stdin else 1)
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": "claude-fake-default"})
    for i, (cmd, run) in enumerate(steps):
        emit({"type": "assistant", "session_id": sid, "message": {"content": [
            {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": cmd}}]}})
        emit({"type": "user", "session_id": sid, "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": run()}]}})
        time.sleep(0.2)  # a real agent thinks between commands: Overseer has seen the output by then
    emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": False, "result": final})
    sys.exit(0)


if prog == "claude" and any("Autopilot is ON" in a for a in args):
    autopilot_run()

def fake_outbox():
    """FAKE_PLOT: leave files in the outbox like an agent would (plus things that must not be sent)."""
    out = os.path.join(os.getcwd(), "outbox")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "reward.png"), "wb").write(b"\x89PNG\r\n\x1a\n fake image")
    open(os.path.join(out, "results.csv"), "w").write("seed,score\n0,3.1\n")
    os.symlink(os.environ["FAKE_SECRET"], os.path.join(out, "innocent.png"))  # must never be followed
    with open(os.path.join(out, "huge.bin"), "wb") as fh:  # too big for Telegram (sparse, so instant)
        fh.truncate(60 * 1024 * 1024)


if prog == "claude" and "FAKE_PLOT" in stdin:
    fake_outbox()


def ov_run():
    """Lines "FAKE_OV <command>" in the prompt: run them (they use the generated `ov`) like an agent would."""
    import subprocess
    sid = str(uuid.uuid4())
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": "claude-fake-default"})
    outputs = []
    for i, cmd in enumerate(re.findall(r"^FAKE_OV (.+)$", stdin, re.M)):
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = res.stdout + res.stderr + f"[exit {res.returncode}]"
        outputs.append(out)
        emit({"type": "assistant", "session_id": sid, "message": {"content": [
            {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": cmd}}]}})
        emit({"type": "user", "session_id": sid, "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": out}]}})
    emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": False,
          "result": "\n\n@@@\n\n".join(outputs)})
    sys.exit(0)


if prog == "claude" and "FAKE_OV" in stdin:
    ov_run()

if prog == "claude":
    if "--output-format" in args and args[args.index("--output-format") + 1] == "json":
        emit({"type": "result", "subtype": "success", "result": "OK report", "is_error": False})
        sys.exit(0)
    sid = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
    model = args[args.index("--model") + 1] if "--model" in args else "claude-fake-default"
    if stdin.startswith("/compact"):  # Claude Code's own /compact, like in `claude -p`
        emit({"type": "system", "subtype": "status", "status": "compacting", "session_id": sid})
        emit({"type": "system", "subtype": "init", "session_id": sid, "model": model})
        emit({"type": "system", "subtype": "compact_boundary", "session_id": sid,
              "compact_metadata": {"trigger": "manual", "pre_tokens": 150000, "post_tokens": 3000}})
        emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": False, "result": ""})
        sys.exit(0)
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": model})
    if "FAKE_COMPACT" in stdin:  # the context filled up: Claude Code compacted it on its own
        emit({"type": "system", "subtype": "status", "status": "compacting", "session_id": sid})
        emit({"type": "system", "subtype": "compact_boundary", "session_id": sid,
              "compact_metadata": {"trigger": "auto", "pre_tokens": 190000, "post_tokens": 4000}})
    emit({"type": "assistant", "session_id": sid, "message": {"content": [
        {"type": "thinking", "thinking": "The batch size is probably too big for the GPU.", "signature": "x"}]}})
    emit({"type": "assistant", "session_id": sid, "message": {"content": [
        {"type": "text", "text": "Checking the job."},
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ssh hpc 'cd ~/proj && sbatch train.sh'"}},
        {"type": "tool_use", "id": "t2", "name": "Write", "input": {"file_path": "/tmp/x/notes.md", "content": "x"}}]}})
    if "FORCE_BLOCK" in stdin:
        emit({"type": "user", "session_id": sid, "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "This command requires approval"}]}})
        emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": False, "result": "Please approve it."})
        sys.exit(0)
    emit({"type": "user", "session_id": sid, "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "Submitted batch job 4242\n"}]}]}})
    emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": False,
          "result": ("**Fixed** the batch size in `train.py` and resubmitted:\n\n- job **4242**\n- lr `3e-4`"
                     if "FAKE_MD" in stdin else "Fixed the batch size and resubmitted as job 4242."),
          "total_cost_usd": 0.0312,
          "modelUsage": {model: {"inputTokens": 1000, "outputTokens": 200, "cacheReadInputTokens": 400,
                                 "cacheCreationInputTokens": 0, "costUSD": 0.0312}}})
    sys.exit(0)

if prog == "codex" and args[:1] == ["resume"]:  # interactive `overseer open`
    print("codex TUI would open here")
    sys.exit(0)


def codex_rollout(tid):
    log_dir = os.path.join(os.environ["CODEX_HOME"], "sessions", "2026", "10", "10")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"rollout-2026-10-10T00-00-00-{tid}.jsonl")


def codex_compacted(tid):
    with open(codex_rollout(tid), "a") as fh:  # what codex's session log records for a compaction
        fh.write(json.dumps({"type": "compacted", "payload": {"message": "summary of the earlier turns"}}) + "\n")


if prog == "codex" and args[:1] == ["app-server"]:  # JSON-RPC over stdio: only what /compact uses
    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("method") != "thread/compact/start":
            if "id" in msg:
                print(json.dumps({"id": msg["id"], "result": {}}), flush=True)
            continue
        tid = msg["params"]["threadId"]
        if not os.path.exists(codex_rollout(tid)):
            print(json.dumps({"id": msg["id"], "error": {"message": f"no rollout found for thread id {tid}"}}), flush=True)
            continue
        print(json.dumps({"id": msg["id"], "result": {}}), flush=True)
        codex_compacted(tid)
        item = {"type": "contextCompaction", "id": "c1"}
        print(json.dumps({"method": "item/completed", "params": {"threadId": tid, "turnId": "t1", "item": item}}),
              flush=True)
        print(json.dumps({"method": "turn/completed", "params": {"threadId": tid, "turn": {"id": "t1",
                                                                                           "status": "completed"}}}),
              flush=True)
    sys.exit(0)

if prog == "codex" and args[:2] == ["debug", "models"]:
    emit({"models": [{"slug": "gpt-fake-default", "default_reasoning_level": "medium"},
                     {"slug": "gpt-x", "default_reasoning_level": "low"}]})
    sys.exit(0)

if prog == "codex":
    out_file = args[args.index("-o") + 1]
    resumed = len(args) > 1 and args[1] == "resume"
    tid = args[-2] if resumed else str(uuid.uuid4())
    if "--ephemeral" in args:
        open(out_file, "w").write("OK report")
        emit({"type": "turn.completed"})
        sys.exit(0)
    emit({"type": "thread.started", "thread_id": tid})
    # Session log, like codex's own: the model and effort are only recorded here.
    model = args[args.index("-m") + 1] if "-m" in args else "gpt-fake-default"
    effort = next((a.split("=", 1)[1].strip('"') for a in args if a.startswith("model_reasoning_effort=")), None)
    with open(codex_rollout(tid), "a") as fh:
        fh.write(json.dumps({"type": "turn_context", "payload": {"model": model, "collaboration_mode": {
            "mode": "default", "settings": {"model": model, "reasoning_effort": effort}}}}) + "\n")
    if "FAKE_COMPACT" in stdin:  # the context filled up: codex compacted it (only its session log says so)
        codex_compacted(tid)
    cmd = "/bin/bash -lc 'ssh hpc sbatch --parsable ~/proj/train.sh'"
    emit({"type": "item.completed", "item": {"id": "i0", "type": "reasoning", "text": "The batch size is probably too big for the GPU."}})
    emit({"type": "item.completed", "item": {"id": "i0b", "type": "agent_message", "text": "Checking the job."}})
    emit({"type": "item.started", "item": {"id": "i1", "type": "command_execution", "command": cmd, "status": "in_progress"}})
    emit({"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": cmd,
                                              "aggregated_output": "4343\n", "exit_code": 0, "status": "completed"}})
    emit({"type": "item.completed", "item": {"id": "i2", "type": "file_change", "changes": [{"path": "/tmp/x/a.py", "kind": "update"}]}})
    emit({"type": "item.completed", "item": {"id": "i3", "type": "agent_message", "text": "Resubmitted as job 4343."}})
    open(out_file, "w").write("Resubmitted as job 4343.")
    # Like the real codex: usage is the thread's running total (5000/1000/700 more per run).
    counter = os.environ["FAKE_CLI_LOG"] + f".{tid}.runs"
    runs = (int(open(counter).read()) if os.path.exists(counter) else 0) + 1
    open(counter, "w").write(str(runs))
    emit({"type": "turn.completed", "usage": {"input_tokens": 5000 * runs, "cached_input_tokens": 1000 * runs,
                                              "output_tokens": 700 * runs}})
    sys.exit(0)
