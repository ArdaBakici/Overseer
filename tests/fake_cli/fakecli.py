#!/usr/bin/env python3
"""Stand-in for `claude` and `codex` used by the integration tests.

Logs argv/env/stdin to $FAKE_CLI_LOG (JSON lines) and emits a canned event stream:
the agent runs `ssh hpc ...sbatch...`, sbatch answers with a job id, then a final message.
A prompt containing FORCE_SLEEP makes it hang (for /stop tests); FORCE_FAIL (or a file $FAKE_CLI_LOG.fail) makes it exit with an error;
FORCE_SLOW pauses between events (to watch the live progress).

Autopilot (fake claude, when the system prompt says autopilot is on): every run submits a new job
id and ends with a status line. Markers in the prompt (the goal, or job logs quoted in a round's
report) steer it: FAKE_DONE, FAKE_HELP, FAKE_NOJOB (submit nothing), FAKE_TWO (submit two jobs);
in a check-in, FAKE_DIVERGE makes it scancel the job and resubmit.
"""
import re
import json
import os
import sys
import time
import uuid

prog = os.path.basename(sys.argv[0])
args = sys.argv[1:]
stdin = "" if sys.stdin.isatty() else sys.stdin.read()
with open(os.environ["FAKE_CLI_LOG"], "a") as fh:
    fh.write(json.dumps({"prog": prog, "argv": args, "env": dict(os.environ), "stdin": stdin, "cwd": os.getcwd()}) + "\n")


def emit(ev):
    print(json.dumps(ev), flush=True)
    if "FORCE_SLOW" in stdin:
        time.sleep(0.3)


if "login" in args:
    print("logged in")
    sys.exit(0)
if "FORCE_SLEEP" in stdin:
    time.sleep(60)
if "FORCE_FAIL" in stdin or os.path.exists(os.environ["FAKE_CLI_LOG"] + ".fail"):
    print("boom: model unavailable", file=sys.stderr)
    sys.exit(3)

def autopilot_run():
    with open(os.environ["FAKE_CLI_LOG"]) as fh:
        n = sum(1 for _ in fh)  # runs so far: makes job ids unique
    sid = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
    submit = ("ssh hpc 'cd ~/proj && sbatch train.sh'", lambda i: f"Submitted batch job {6000 + n * 10 + i}\n")
    steps, final, status = [], "Submitted the next experiment.", "CONTINUE"
    if "Autopilot check-in" in stdin:
        final = "Looks healthy."
        if "FAKE_DIVERGE" in stdin:
            jid = re.search(r"<job id='(\d+)'", stdin).group(1)
            steps = [(f"ssh hpc scancel {jid}", ""), (submit[0], submit[1](0))]
            final = f"Job {jid} diverged: cancelled it and resubmitted with a lower lr."
    elif "FAKE_DONE" in stdin:
        final, status = "Reached the goal.", "DONE best lr is 3e-4"
    elif "FAKE_HELP" in stdin:
        final, status = "Not sure how to continue.", "NEED_HELP should I try a bigger model?"
    elif "FAKE_NOJOB" not in stdin:
        steps = [(submit[0], submit[1](i)) for i in range(2 if "FAKE_TWO" in stdin else 1)]
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": "claude-fake-default"})
    for i, (cmd, out) in enumerate(steps):
        emit({"type": "assistant", "session_id": sid, "message": {"content": [
            {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": cmd}}]}})
        emit({"type": "user", "session_id": sid, "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": out}]}})
    emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": False,
          "result": f"{final}\n\n**AUTOPILOT: {status}**"})
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

if prog == "claude":
    if "--output-format" in args and args[args.index("--output-format") + 1] == "json":
        emit({"type": "result", "subtype": "success", "result": "OK report", "is_error": False})
        sys.exit(0)
    sid = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
    model = args[args.index("--model") + 1] if "--model" in args else "claude-fake-default"
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": model})
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
    log_dir = os.path.join(os.environ["CODEX_HOME"], "sessions", "2026", "10", "10")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"rollout-2026-10-10T00-00-00-{tid}.jsonl"), "a") as fh:
        fh.write(json.dumps({"type": "turn_context", "payload": {"model": model, "collaboration_mode": {
            "mode": "default", "settings": {"model": model, "reasoning_effort": effort}}}}) + "\n")
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
