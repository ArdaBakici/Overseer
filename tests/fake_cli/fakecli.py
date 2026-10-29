#!/usr/bin/env python3
"""Stand-in for `claude` and `codex` used by the integration tests.

Logs argv/env/stdin to $FAKE_CLI_LOG (JSON lines) and emits a canned event stream:
the agent runs `ssh hpc ...sbatch...`, sbatch answers with a job id, then a final message.
A prompt containing FORCE_SLEEP makes it hang (for /stop tests); FORCE_FAIL (or a file $FAKE_CLI_LOG.fail) makes it exit with an error;
FORCE_SLOW pauses between events (to watch the live progress).

Accounts: a CLI home (CLAUDE_CONFIG_DIR / CODEX_HOME) listed in the file $FAKE_CLI_LOG.limited is out of usage (the
run fails like the real CLIs' usage limits). Sessions are kept in the home like the real CLIs; with a file
$FAKE_CLI_LOG.strict, resuming a session that isn't in the home fails like theirs.

Working alone over many runs (fake claude, for prompts with FAKE_LOOP, and wake-ups and check-ins):
every run submits a new job id and watches it with `ov watch`, like a real agent. Markers in the
prompt (the first message, or job logs quoted in a wake-up's report) steer it: FAKE_DONE (stop),
FAKE_HELP (ask), FAKE_NOJOB (submit nothing), FAKE_TWO (two jobs), FAKE_NOWATCH (submit without
watching), FAKE_EVERY (watch with check-ins); in a check-in, FAKE_DIVERGE makes it scancel and resubmit.

Messages sent while it works (steering): claude reads streaming input (--input-format stream-json) and acknowledges each
message it takes (--replay-user-messages); codex runs through `codex app-server` and takes `turn/steer`. With
FAKE_STEER in the request, either waits up to 5 s after its command for such a message, and its answer ends with
"STEERED: <message>". A message after the answer gets an answer of its own ("LATE: <message>").
"""
import re
import json
import os
import sys
import time
import uuid

import select

prog = os.path.basename(sys.argv[0])
args = sys.argv[1:]


class Lines:
    """stdin, a line at a time, with a timeout (while the run goes on, more may come: steering)."""
    def __init__(self):
        self.buf = b""
        self.eof = False

    def get(self, timeout=None):
        """The next line (str), or None: nothing within `timeout` s (None: wait), or the end of the input."""
        end = None if timeout is None else time.time() + timeout
        while b"\n" not in self.buf and not self.eof:
            left = None if end is None else max(0.0, end - time.time())
            if not select.select([0], [], [], left)[0]:
                return None
            data = os.read(0, 65536)
            if not data:
                self.eof = True
            self.buf += data
        if b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            return line.decode()
        return None


LINES = Lines()
STREAM = prog == "claude" and "stream-json" in args[args.index("--input-format") + 1:args.index("--input-format") + 2] \
    if "--input-format" in args else False
REPLAY = "--replay-user-messages" in args
STEERED = []  # messages taken while it worked


def user_text(line):
    content = json.loads(line)["message"]["content"]
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content)


def log_call(text):
    with open(os.environ["FAKE_CLI_LOG"], "a") as fh:
        fh.write(json.dumps({"prog": prog, "argv": args, "env": dict(os.environ), "stdin": text,
                             "cwd": os.getcwd()}) + "\n")


if STREAM:  # (claude with streaming input: the request is the first message)
    stdin = user_text(LINES.get())
elif sys.stdin.isatty() or args[:1] == ["app-server"]:  # (app-server: a conversation - logged per turn)
    stdin = ""
else:
    stdin = sys.stdin.read()
if args[:1] != ["app-server"]:
    log_call(stdin)


def take_steer(line):
    text = user_text(line)
    STEERED.append(text)
    if REPLAY:
        print(json.dumps({"type": "user", "isReplay": True, "message": {"role": "user", "content": text}}), flush=True)


def emit(ev):
    if STREAM and ev.get("type") == "result" and not ev.get("is_error"):
        while (line := LINES.get(0)) is not None:  # (sent while it worked: taken before it answers)
            take_steer(line)
        if STEERED and ev.get("result"):
            ev = {**ev, "result": ev["result"] + "\nSTEERED: " + " | ".join(STEERED)}
    print(json.dumps(ev), flush=True)
    if "FORCE_SLOW" in stdin:
        time.sleep(0.3)
    if STREAM and ev.get("type") == "result":  # the input stays open: each later message is a turn of its own
        while (line := LINES.get()) is not None:
            if REPLAY:
                print(json.dumps({"type": "user", "isReplay": True,
                                  "message": {"role": "user", "content": user_text(line)}}), flush=True)
            print(json.dumps({**ev, "result": "LATE: " + user_text(line)}), flush=True)


def wait_steer(seconds=5):
    """FAKE_STEER: give a message sent while it works time to come (it's taken at this step)."""
    if STREAM and "FAKE_STEER" in stdin:
        line = LINES.get(seconds)
        if line is not None:
            take_steer(line)


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

CLI_HOME = os.environ.get("CLAUDE_CONFIG_DIR" if prog == "claude" else "CODEX_HOME", "")
STRICT = os.path.exists(os.environ["FAKE_CLI_LOG"] + ".strict")


def limited():
    path = os.environ["FAKE_CLI_LOG"] + ".limited"
    return bool(CLI_HOME) and os.path.exists(path) and CLI_HOME in open(path).read().split("\n")


def claude_session_file(sid):
    return os.path.join(CLI_HOME, "projects", re.sub(r"[^A-Za-z0-9]", "-", os.getcwd()), f"{sid}.jsonl")


if prog == "claude" and limited():
    sid = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": "claude-opus-5-5"})
    emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": True,
          "result": "You've hit your limit · resets 3pm (UTC)"})
    sys.exit(1)
if prog == "codex" and limited() and "exec" in args:
    emit({"type": "thread.started", "thread_id": args[-2] if "resume" in args else str(uuid.uuid4())})
    emit({"type": "turn.failed", "error": {"message": "You've hit your usage limit. Upgrade to Pro "
                                                      "(https://chatgpt.com/explore/pro), or try again in 2 hours 5 minutes."}})
    sys.exit(1)

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
    every = " --every 1h" if "FAKE_EVERY" in stdin else ""
    steps, final = [], "Submitted the next experiment."

    def submit_and_watch(count):
        ids = [6000 + n * 10 + i for i in range(count)]
        for jid in ids:
            steps.append(("ssh hpc 'cd ~/proj && sbatch train.sh'", lambda jid=jid: fake_submit(jid)))
        if "FAKE_NOWATCH" not in stdin:
            cmd = "watch " + " ".join(map(str, ids)) + every
            steps.append(("ov " + cmd, lambda: ov(cmd)))

    if "[Check-in you asked for" in stdin:
        final = "Looks healthy."
        if "FAKE_DIVERGE" in stdin:
            jid = re.search(r"<job id='(\d+)'", stdin).group(1)
            steps.append((f"ssh hpc scancel {jid}", lambda: ""))
            submit_and_watch(1)
            final = f"Job {jid} diverged: cancelled it and resubmitted with a lower lr."
    elif "FAKE_DONE" in stdin:  # done: no more wake-ups
        final = "Reached the goal: best lr is 3e-4."
    elif "FAKE_HELP" in stdin:  # needs a decision: asks, and sets nothing up
        final = "Not sure how to continue: should I try a bigger model?"
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


if prog == "claude" and any(m in stdin for m in ("FAKE_LOOP", "[Wake-up you set up]", "[Check-in you asked for")):
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

if prog == "claude" and "FAKE_SILENT" in stdin:  # a wake-up with nothing worth saying
    sid = str(uuid.uuid4())
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": "claude-fake-default"})
    emit({"type": "result", "subtype": "success", "session_id": sid, "is_error": False, "result": "[silent]"})
    sys.exit(0)

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
    if STRICT and "--resume" in args and not os.path.exists(claude_session_file(sid)):
        emit({"type": "result", "subtype": "error_during_execution", "is_error": True,
              "result": f"No conversation found with session ID: {sid}"})
        sys.exit(1)
    os.makedirs(os.path.dirname(claude_session_file(sid)), exist_ok=True)
    with open(claude_session_file(sid), "a") as fh:
        fh.write(json.dumps({"type": "user", "message": stdin[-2000:]}) + "\n")
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
    wait_steer()
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


def codex_model():
    for i, a in enumerate(args):
        if a == "-m" or (a == "-c" and args[i + 1].startswith("model=")):
            return args[i + 1].split("=", 1)[1].strip('"') if a == "-c" else args[i + 1]
    return "gpt-fake-default"


def codex_effort():
    return next((a.split("=", 1)[1].strip('"') for a in args if a.startswith("model_reasoning_effort=")), None)


if prog == "codex" and args[:1] == ["app-server"]:  # JSON-RPC over stdio: Overseer's runs (and /compact)
    def out(msg):
        print(json.dumps(msg), flush=True)

    def note(method, params):
        out({"method": method, "params": params})
        if "FORCE_SLOW" in stdin:
            time.sleep(0.3)

    def turn_item(kind, item, started=False):
        if started:
            note("item/started", {"item": {**item, "type": kind}})
        note("item/completed", {"item": {**item, "type": kind}})

    tid, active = None, None
    while (line := LINES.get()) is not None:
        msg = json.loads(line)
        m, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if m == "thread/compact/start":
            log_call("")
            tid = params["threadId"]
            if not os.path.exists(codex_rollout(tid)):
                out({"id": rid, "error": {"message": f"no rollout found for thread id {tid}"}})
                continue
            out({"id": rid, "result": {}})
            codex_compacted(tid)
            note("item/completed", {"threadId": tid, "turnId": "t1", "item": {"type": "contextCompaction", "id": "c1"}})
            note("turn/completed", {"threadId": tid, "turn": {"id": "t1", "status": "completed"}})
        elif m == "thread/start":
            tid = str(uuid.uuid4())
            out({"id": rid, "result": {"thread": {"id": tid}, "model": codex_model(), "reasoningEffort": codex_effort()}})
        elif m == "thread/resume":
            tid = params["threadId"]
            if STRICT and not os.path.exists(codex_rollout(tid)):
                out({"id": rid, "error": {"message": f"thread/resume failed: no rollout found for thread id {tid}"}})
                continue
            out({"id": rid, "result": {"thread": {"id": tid}, "model": codex_model(), "reasoningEffort": codex_effort()}})
        elif m == "turn/steer":
            if active and params.get("expectedTurnId") == active:
                STEERED.append(params["input"][0]["text"])
                out({"id": rid, "result": {"turnId": active}})
            else:
                out({"id": rid, "error": {"code": -32600, "message": "no active turn to steer"}})
        elif m == "turn/start":
            stdin = params["input"][0]["text"]
            log_call(stdin)
            turn = active = str(uuid.uuid4())
            out({"id": rid, "result": {"turn": {"id": turn, "status": "inProgress"}}})
            if limited():
                note("error", {"threadId": tid, "turnId": turn, "willRetry": False, "error": {
                    "message": "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), or "
                               "try again in 2 hours 5 minutes.", "codexErrorInfo": "usageLimitExceeded"}})
                note("turn/completed", {"threadId": tid, "turn": {"id": turn, "status": "failed", "error": {
                    "message": "You've hit your usage limit."}}})
                active = None
                continue
            if "FORCE_SLEEP" in stdin:
                time.sleep(60)
            if "FORCE_FAIL" in stdin or os.path.exists(os.environ["FAKE_CLI_LOG"] + ".fail"):
                print("boom: model unavailable", file=sys.stderr)
                sys.exit(3)
            with open(codex_rollout(tid), "a") as fh:  # (its session log: the model and effort are recorded here)
                fh.write(json.dumps({"type": "turn_context", "payload": {"model": codex_model(), "collaboration_mode": {
                    "mode": "default", "settings": {"model": codex_model(), "reasoning_effort": codex_effort()}}}}) + "\n")
            if "FAKE_COMPACT" in stdin:  # the context filled up: codex compacted it
                codex_compacted(tid)
                turn_item("contextCompaction", {"id": "c0"})
            cmd = "/bin/bash -lc 'ssh hpc sbatch --parsable ~/proj/train.sh'"
            turn_item("reasoning", {"id": "i0", "summary": ["The batch size is probably too big for the GPU."],
                                    "content": []})
            turn_item("agentMessage", {"id": "i0b", "text": "Checking the job."})
            note("item/started", {"item": {"id": "i1", "type": "commandExecution", "command": cmd,
                                           "status": "inProgress"}})
            if "FAKE_STEER" in stdin:  # (a message sent while it works comes now: turn/steer)
                end = time.time() + 5
                while not STEERED and time.time() < end and (line := LINES.get(max(0.0, end - time.time()))):
                    msg = json.loads(line)
                    if msg.get("method") == "turn/steer" and msg["params"].get("expectedTurnId") == active:
                        STEERED.append(msg["params"]["input"][0]["text"])
                        out({"id": msg["id"], "result": {"turnId": active}})
            note("item/completed", {"item": {"id": "i1", "type": "commandExecution", "command": cmd,
                                             "aggregatedOutput": "4343\n", "exitCode": 0, "status": "completed"}})
            turn_item("fileChange", {"id": "i2", "changes": [{"path": "/tmp/x/a.py", "kind": "update"}]})
            final = "Resubmitted as job 4343." + ("\nSTEERED: " + " | ".join(STEERED) if STEERED else "")
            turn_item("agentMessage", {"id": "i3", "text": final})
            counter = os.environ["FAKE_CLI_LOG"] + f".{tid}.runs"  # usage: the thread's running total
            runs = (int(open(counter).read()) if os.path.exists(counter) else 0) + 1
            open(counter, "w").write(str(runs))
            note("thread/tokenUsage/updated", {"threadId": tid, "turnId": turn, "tokenUsage": {
                "total": {"inputTokens": 5000 * runs, "cachedInputTokens": 1000 * runs, "outputTokens": 700 * runs},
                "last": {}}})
            active = None
            note("turn/completed", {"threadId": tid, "turn": {"id": turn, "status": "completed"}})
        elif rid is not None:
            out({"id": rid, "result": {}})
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
    if STRICT and resumed and not os.path.exists(codex_rollout(tid)):
        emit({"type": "error", "message": f"thread/resume failed: no rollout found for thread id {tid}"})
        sys.exit(1)
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
