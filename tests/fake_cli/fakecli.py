#!/usr/bin/env python3
"""Stand-in for `claude` and `codex` used by the integration tests.

Logs argv/env/stdin to $FAKE_CLI_LOG (JSON lines) and emits a canned event stream:
the agent runs `ssh hpc ...sbatch...`, sbatch answers with a job id, then a final message.
A prompt containing FORCE_SLEEP makes it hang (for /stop tests); FORCE_FAIL makes it exit with an error.
"""
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


if "login" in args:
    print("logged in")
    sys.exit(0)
if "FORCE_SLEEP" in stdin:
    time.sleep(60)
if "FORCE_FAIL" in stdin:
    print("boom: model unavailable", file=sys.stderr)
    sys.exit(3)

if prog == "claude":
    if "--output-format" in args and args[args.index("--output-format") + 1] == "json":
        emit({"type": "result", "subtype": "success", "result": "OK report", "is_error": False})
        sys.exit(0)
    sid = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
    emit({"type": "system", "subtype": "init", "session_id": sid})
    emit({"type": "assistant", "session_id": sid, "message": {"content": [
        {"type": "text", "text": "Checking the job."},
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ssh hpc 'cd ~/proj && sbatch train.sh'"}},
        {"type": "tool_use", "id": "t2", "name": "Write", "input": {"file_path": "/tmp/x/notes.md", "content": "x"}}]}})
    emit({"type": "user", "session_id": sid, "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "Submitted batch job 4242\n"}]}]}})
    emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": False,
          "result": "Fixed the batch size and resubmitted as job 4242."})
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
    cmd = "/bin/bash -lc 'ssh hpc sbatch --parsable ~/proj/train.sh'"
    emit({"type": "item.started", "item": {"id": "i1", "type": "command_execution", "command": cmd, "status": "in_progress"}})
    emit({"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": cmd,
                                              "aggregated_output": "4343\n", "exit_code": 0, "status": "completed"}})
    emit({"type": "item.completed", "item": {"id": "i2", "type": "file_change", "changes": [{"path": "/tmp/x/a.py", "kind": "update"}]}})
    emit({"type": "item.completed", "item": {"id": "i3", "type": "agent_message", "text": "Resubmitted as job 4343."}})
    open(out_file, "w").write("Resubmitted as job 4343.")
    emit({"type": "turn.completed"})
    sys.exit(0)
