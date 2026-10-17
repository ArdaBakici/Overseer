"""End-to-end: the REAL Claude Code / Codex CLIs, sandboxed, reaching a REAL sshd through Overseer's
SSH master connection, driven by a scripted fake model (tests/fake_model_server.py).

A local sshd plays the HPC login node (fake SLURM on its PATH). The scripted model makes the
agent run shell commands, and we check what actually happened:
  * `ssh hpc ...` works (through Overseer's connection - the agent has no key)
  * `sbatch` on the "HPC" is detected and the job is tracked
  * reading Overseer's SSH key, reading ~/.ssh, writing outside the scratch dir and reaching
    the internet all fail inside the sandbox

Skipped unless sshd, ssh, bubblewrap, socat and the CLIs are installed. Point
OVERSEER_TEST_CLI_DIR at a node_modules/.bin containing `claude` and `codex` if they aren't on PATH.
"""

import asyncio
import getpass
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from fake_model_server import FakeModel
from overseer.core.agent import Agent
from overseer.hpc.domain import HpcDomain
from overseer.core.config import Config
from overseer.hpc.remote import HPC
from overseer.core.runners import _toml
from overseer.hpc.ssh import SSHRunner
from overseer.hpc.sshlink import setup_key, write_configs
from overseer.core.store import Store

FAKE_SLURM = Path(__file__).parent / "fake_slurm"
SSHD = shutil.which("sshd") or ("/usr/sbin/sshd" if os.path.exists("/usr/sbin/sshd") else None)


def find_cli(name):
    extra = os.environ.get("OVERSEER_TEST_CLI_DIR", "")
    return shutil.which(name, path=f"{extra}:{os.environ.get('PATH', '')}" if extra else None)


needs = [SSHD, shutil.which("ssh"), shutil.which("ssh-keygen"), shutil.which("bwrap"), shutil.which("socat")]
pytestmark = pytest.mark.skipif(not all(needs), reason="needs sshd, ssh, ssh-keygen, bubblewrap and socat")


class Notes:
    def __init__(self):
        self.sent = []
        self.live = {}  # handle -> list of versions of a self-updating message
        self.deleted = []

    async def send(self, msg):
        self.sent.append(msg)

    async def post_live(self, msg):
        handle = len(self.live) + 1
        self.live[handle] = [msg.text]
        return handle

    async def edit_live(self, handle, text):
        self.live[handle].append(text)

    async def delete_live(self, handle):
        self.deleted.append(handle)

    def texts(self):
        return "\n".join(m.text for m in self.sent)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def hpc(tmp_path):
    """Overseer config + an sshd on localhost authorising Overseer's freshly generated key."""
    short = tempfile.mkdtemp(prefix="ove-", dir="/tmp")
    app = tmp_path / "app"
    app.mkdir()
    remote = tmp_path / "remote"  # the "HPC" filesystem area
    (remote / "proj").mkdir(parents=True)
    (remote / "proj" / "train.sh").write_text("#!/bin/bash\n#SBATCH --time=00:10:00\necho training\n")
    state = tmp_path / "slurm.json"
    state.write_text(json.dumps({"next_id": 5000, "jobs": {}}))

    port = free_port()
    cfg = Config()
    cfg.ssh.hostname, cfg.ssh.port, cfg.ssh.user = "127.0.0.1", port, getpass.getuser()
    cfg.telegram.allowed_chat_ids = [1]
    cfg.local.base_dir = str(app)
    cfg.local.tmp_dir = short
    cfg.agent.timeout_minutes = 3
    cfg.validate()
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(parents=True, exist_ok=True)
    pub = setup_key(cfg)

    sshd_dir = tmp_path / "sshd"
    sshd_dir.mkdir()
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(sshd_dir / "hostkey")], check=True)
    (sshd_dir / "authorized_keys").write_text(pub + "\n")
    (sshd_dir / "sshd_config").write_text(f"""Port {port}
ListenAddress 127.0.0.1
HostKey {sshd_dir / 'hostkey'}
PidFile {sshd_dir / 'sshd.pid'}
AuthorizedKeysFile {sshd_dir / 'authorized_keys'}
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
StrictModes no
Subsystem sftp internal-sftp
SetEnv PATH={FAKE_SLURM}:/usr/local/bin:/usr/bin:/bin FAKE_SLURM_STATE={state}
""")
    os.makedirs("/run/sshd", exist_ok=True) if os.geteuid() == 0 else None
    proc = subprocess.Popen([SSHD, "-D", "-e", "-f", str(sshd_dir / "sshd_config")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    write_configs(cfg)
    had_overseer_dir = (Path.home() / ".overseer").exists()  # hpc-run's worker files go to the "HPC" home
    try:
        yield cfg, remote, state, app
    finally:
        if not had_overseer_dir:
            shutil.rmtree(Path.home() / ".overseer", ignore_errors=True)
        subprocess.run(["ssh", "-F", str(cfg.ssh_config_file), "-O", "exit", cfg.ssh.host], capture_output=True)
        proc.terminate()
        proc.wait(5)
        shutil.rmtree(short, ignore_errors=True)


def configure_backend(cfg, backend, cli, fm):
    if backend == "claude_code":
        c = cfg.agent.claude_code
        c.command = cli
        os.environ["ANTHROPIC_BASE_URL"] = fm.url
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake"
        c.pass_env = [*c.pass_env, "ANTHROPIC_BASE_URL"]
    else:
        c = cfg.agent.codex
        c.command = cli
        os.environ["FAKE_KEY"] = "sk-fake"
        c.pass_env = [*c.pass_env, "FAKE_KEY"]
        provider = {"name": "fake", "base_url": fm.url + "/v1", "wire_api": "responses", "env_key": "FAKE_KEY",
                    "request_max_retries": 0, "stream_max_retries": 0}
        c.extra_config = ['model_provider="fake"', 'model="fake-model"', f"model_providers.fake={_toml(provider)}"]
        c.reasoning_effort = "high"


@pytest.mark.parametrize("backend,cli_name", [("claude_code", "claude"), ("codex", "codex")])
async def test_agent_end_to_end(hpc, backend, cli_name, monkeypatch):
    cli = find_cli(cli_name)
    if not cli:
        pytest.skip(f"{cli_name} not installed")
    cfg, remote, state, app = hpc
    real_home = Path.home()
    script = [
        "ssh hpc 'echo remote-user:$(whoami)'",
        f"ssh hpc 'cd {remote / 'proj'} && sbatch train.sh'",
        f"cat {cfg.ssh_key} 2>&1 | head -c 60; echo; echo KEYCHECK-DONE",
        f"ls {real_home / '.ssh'} 2>&1; echo HOMESSH-DONE",
        f"touch {app / 'agent-was-here'} 2>&1; echo WRITECHECK-DONE",
        "curl -sS -m 5 https://example.com -o /dev/null 2>&1 && echo INTERNET-REACHABLE || echo NO-INTERNET",
        "echo hello > notes.txt && cat notes.txt",
        f"cat {app / '.env'} 2>&1; echo ENVCHECK-DONE",
        # Variables in nested quotes + pipes: Claude Code used to refuse these as "requires approval".
        "ssh hpc 'bash -lc \"echo HOME=$HOME; echo U=$USER | tail -1\"'",
        # A plot made on the HPC, copied into the outbox: it's sent with the reply.
        f"scp hpc:{remote / 'proj' / 'reward.png'} ./outbox/ && echo SCP-DONE",
        # A service listening on this machine's localhost (here: the test sshd).
        f"(exec 3<>/dev/tcp/127.0.0.1/{cfg.ssh.port}) 2>/dev/null && echo LOOPBACK-OPEN || echo LOOPBACK-CLOSED",
        # Heavy work on a compute node: the first hpc-run starts the shared worker, the second reuses it.
        f"hpc-run --dir {remote / 'proj'} -- python3 -c \"import os; print('ON-COMPUTE', os.getcwd(), os.environ['MPLBACKEND'])\"",
        "hpc-run -- sh -c 'echo AGAIN'",
        "hpc-run --status",
        # ...and directly on the login node it hits the CPU limit (2 s in this test).
        "ssh hpc 'python3 -c \"while True: pass\"'; echo EXIT=$?",
        # Talking to Overseer from inside the sandbox: register the job, create a project.
        "ov job add 5001",
        "ov project new e2eproj --dir ~/e2eproj --about 'made by the agent'",
        # Being woken: check-ins and timers, no approval needed.
        "ov watch 5001 --every 1h --note 'compare with the baseline'",
        "ov wake in 1h check the curves",
        # The overseer skill: Codex has the model read it with a command, so its sandbox must allow that.
        "cat .agents/skills/overseer/SKILL.md 2>&1 | head -2; echo SKILL-DONE",
        # Temp files (FB3: zsh's here-documents went to the read-only /tmp): in the scratch dir.
        "zsh -c 'cat <<X\nZSH-HEREDOC-OK\nX' 2>&1; mktemp >/dev/null && echo MKTEMP-OK; "
        "python3 -c 'import tempfile; tempfile.NamedTemporaryFile(); print(\"PYTMP-OK\")'",
    ]
    (remote / "proj" / "reward.png").write_bytes(b"\x89PNG\r\n\x1a\n made on the HPC")
    # A login shell may reset PATH; on a real cluster this is where "module load slurm" goes.
    cfg.slurm.setup = f"export PATH={FAKE_SLURM}:$PATH"
    cfg.slurm.compute.login_cpu_seconds = 2
    cfg.slurm.compute.login_warn_seconds = 1
    write_configs(cfg)
    (app / ".env").write_text("TELEGRAM_BOT_TOKEN=999:super-secret-token\n")
    (app / ".env").chmod(0o600)
    from overseer.core import live
    monkeypatch.setattr(live, "PROGRESS_SECONDS", 0.2)
    with FakeModel(script, think=True) as fm:
        configure_backend(cfg, backend, cli, fm)
        store = Store(cfg.local.db_path)
        notes = Notes()
        agent = Agent(cfg, store, notes, HpcDomain(cfg, store, notes, HPC(SSHRunner(cfg), cfg.slurm.setup)))
        agent.set_backend(backend)
        from overseer.core.ctl import ControlServer
        from overseer.core.projects import Projects
        ctl = ControlServer(cfg, agent, Projects(cfg, store, agent), notes)
        await ctl.start()
        s = agent.general_session()
        # The wake-up instructions must reach the model through each CLI's own mechanism.
        from overseer.core.autopilot import Autopilot
        autopilot = Autopilot(cfg, store, agent, notes)
        try:
            await asyncio.wait_for(agent.handle(s, "Run the checks."), 240)
        finally:
            await ctl.stop()
        await autopilot.drain()
        texts = notes.texts()
        outs = fm.outputs
        detail = f"\n\nTELEGRAM:\n{texts}\n\nTOOL OUTPUTS:\n" + "\n====\n".join(outs)
        if os.environ.get("OVERSEER_E2E_VERBOSE"):
            print(detail)
        # Web search: Claude's WebSearch/WebFetch tools, Codex's live web_search.
        main = next(r["body"] for r in fm.requests if "Run the checks." in json.dumps(r["body"]))
        tools = main.get("tools") or []
        if os.environ.get("OVERSEER_E2E_DUMP"):
            Path(os.environ["OVERSEER_E2E_DUMP"], f"{backend}.json").write_text(json.dumps(main, indent=1))
        if backend == "codex":
            assert {"type": "web_search", "external_web_access": True} in tools, tools
        else:
            assert {"WebSearch", "WebFetch"} <= {t.get("name") for t in tools}, tools
        assert "search the web" in json.dumps(main), detail
        assert len(outs) >= len(script), detail
        assert f"remote-user:{getpass.getuser()}" in outs[0], detail          # HPC reachable via the master socket
        assert "Submitted batch job 5001" in outs[1], detail
        assert "PRIVATE KEY" not in outs[2] and "KEYCHECK-DONE" in outs[2], detail   # key unreadable
        assert "HOMESSH-DONE" in outs[3], detail
        assert not (app / "agent-was-here").exists(), detail                   # no writes outside the scratch dir
        assert "NO-INTERNET" in outs[5], detail
        assert "hello" in outs[6], detail                                      # scratch dir is writable
        assert "super-secret-token" not in outs[7] and "ENVCHECK-DONE" in outs[7], detail  # .env unreadable
        assert "HOME=/" in outs[8] and "U=" in outs[8] and "requires approval" not in outs[8], detail
        assert "SCP-DONE" in outs[9], detail
        # Known gap: Codex's sandbox doesn't isolate the machine's localhost (Claude Code's does). If this
        # starts failing for codex, Codex closed it: update the README's sandbox table.
        assert ("LOOPBACK-OPEN" if backend == "codex" else "LOOPBACK-CLOSED") in outs[10], detail
        st = json.loads(state.read_text())
        workers = [j for j, v in st["jobs"].items() if v["name"] == "overseer-worker"]
        assert len(workers) == 1, st  # started once, reused by the second hpc-run
        assert f"ON-COMPUTE {remote / 'proj'} Agg" in outs[11] and "started a compute worker" in outs[11], detail
        assert "AGAIN" in outs[12] and "started a compute worker" not in outs[12], detail
        assert [a for a in st["steps"] if f"--jobid={workers[0]}" in a and "--overlap" in a], st["steps"]
        assert f"worker: {workers[0]} RUNNING" in outs[13], detail
        assert "EXIT=152" in outs[14] and "login node's CPU limit" in outs[14], detail
        assert "✓ job 5001 (" in outs[15] and "tracked in this conversation (already was)" in outs[15], detail
        assert "✓ project e2eproj created (~/e2eproj)" in outs[16], detail
        assert store.get_project("e2eproj").context == "made by the agent"
        assert texts.count("Agent submitted job 5001") == 1, detail
        assert f"Agent submitted job {workers[0]}" not in texts, detail  # the worker isn't reported as a job
        assert "on the login node: <code>ssh hpc 'python3 -c" in texts, detail
        assert notes.sent[-1].photos == [("reward.png", b"\x89PNG\r\n\x1a\n made on the HPC")], detail
        assert "refused by" not in texts, detail
        assert "Agent submitted job 5001" in texts, detail
        rec = store.get_job("5001")
        assert rec and rec.by_agent == 1 and rec.session_id == s.id
        assert json.loads(state.read_text())["jobs"]["5001"]["state"] == "PENDING"
        assert "DONE" in notes.sent[-1].text, detail
        assert store.get_session(s.id).cli_session_id
        # The model's reasoning summaries and in-between messages reach the live views.
        events = [json.loads(line) for line in
                  (cfg.local.state("transcripts") / f"{s.id}.jsonl").read_text().splitlines()]
        thoughts = [e["text"] for e in events if e["kind"] == "thought"]
        notes_ = [e["text"] for e in events if e["kind"] == "message"]
        assert "THOUGHT 1: checking step 1" in thoughts and "NOTE 1" in notes_, events
        assert any(e["kind"] == "output" and "Submitted batch job 5001" in e["output"] for e in events)
        assert events[-1]["kind"] == "end" and events[-1]["status"] == "done"
        seen = "\n".join(v for versions in notes.live.values() for v in versions)
        assert "THOUGHT" in seen and notes.deleted, seen
        if backend == "codex":  # read back from codex's own session log (its --json events don't say)
            assert "codex · fake-model · effort high ·" in notes.sent[-1].text, detail
        # Overseer's instructions reached the model.
        tool_requests = [r for r in fm.requests if r["body"].get("tools")]
        assert "You are Overseer's agent" in json.dumps(tool_requests[0]["body"]), "system prompt missing"
        body = json.dumps(tool_requests[0]["body"])
        assert "## Working over time: watches and wake-ups" in body and "ov auto" not in body
        assert "⏰ Waiting for: job 5001 (check-ins every 1h); ⏰ in 1h" in notes.sent[-1].text, detail
        assert "✓ w1: you'll be woken when job 5001 has ended, with check-ins every 1h" in outs[17], detail
        assert "✓ w2: you'll be woken in 1h" in outs[18], detail
        # Each CLI offers the overseer skill to the model; Codex's copy is where its sandbox can read it.
        assert "overseer: How to talk to Overseer with the `ov` command" in body
        if backend == "codex":
            assert "name: overseer" in outs[19], detail
            assert re.search(r"r\d+` = `[^`]*/agents/" + re.escape(s.id) + r"/\.agents/skills`", body), body[:3000]
        if shutil.which("zsh"):
            assert "ZSH-HEREDOC-OK" in outs[20], detail
        assert "MKTEMP-OK" in outs[20] and "PYTMP-OK" in outs[20], detail


@pytest.mark.parametrize("backend,cli_name", [("claude_code", "claude"), ("codex", "codex")])
async def test_compaction_end_to_end(hpc, backend, cli_name):
    """The real CLIs compact on their own when the model reports a full context (Overseer reports it), and
    on request (/compact: Claude Code's own command, codex's app server)."""
    cli = find_cli(cli_name)
    if not cli:
        pytest.skip(f"{cli_name} not installed")
    cfg, remote, state, app = hpc
    # One response reports a context far beyond the limit; the CLI compacts before its next request.
    big, at = (5_000_000, 1) if backend == "claude_code" else (50_000, 2)
    with FakeModel(["echo one", "echo two"], input_tokens=lambda n: big if n == at else 1) as fm:
        configure_backend(cfg, backend, cli, fm)
        if backend == "codex":  # compact above 20k tokens
            cfg.agent.codex.extra_config.append("model_auto_compact_token_limit=20000")
        store = Store(cfg.local.db_path)
        notes = Notes()
        agent = Agent(cfg, store, notes, HpcDomain(cfg, store, notes, HPC(SSHRunner(cfg), cfg.slurm.setup)))
        agent.set_backend(backend)
        s = agent.general_session()
        await asyncio.wait_for(agent.handle(s, "Run the checks."), 180)
        reply = notes.sent[-1].text
        assert "🗜 <i>The conversation was compacted automatically" in reply, notes.texts()

        n = len(fm.requests)
        assert await asyncio.wait_for(agent.compact(s), 180), notes.texts()
        assert "🗜 <b>Compacted the conversation</b>" in notes.sent[-1].text, notes.texts()
        assert len(fm.requests) > n  # the CLI asked the model for the summary
        await asyncio.wait_for(agent.handle(s, "Anything else?"), 180)  # the conversation goes on
        assert notes.sent[-1].text.startswith("🤖"), notes.texts()
        events = [json.loads(line) for line in
                  (cfg.local.state("transcripts") / f"{s.id}.jsonl").read_text().splitlines()]
        assert [e["trigger"] for e in events if e["kind"] == "compact"] == ["auto", "manual"], events


# The tools each CLI offers the model. Anything that would outlive the turn (wake-ups, schedules, background
# commands or subagents, messages to other sessions) is off: the CLI exits when the turn ends, so it would never
# report back. A CLI update that adds a tool fails here, to be looked at.
EXPECTED_TOOLS = {
    "claude_code": {"Agent", "Bash", "Edit", "NotebookEdit", "Read", "Skill", "WebFetch", "WebSearch", "Write"},
    "codex": {"exec_command", "write_stdin", "request_user_input", "view_image", "web_search"},
}


@pytest.mark.parametrize("backend,cli_name", [("claude_code", "claude"), ("codex", "codex")])
async def test_only_tools_that_work_within_a_turn(hpc, backend, cli_name):
    cli = find_cli(cli_name)
    if not cli:
        pytest.skip(f"{cli_name} not installed")
    cfg, remote, state, app = hpc
    if backend == "codex":  # asks the user a question: there's nobody to answer in a headless run
        script = [{"tool": "request_user_input", "args": {"questions": [
            {"id": "lr", "header": "LR", "question": "Which learning rate?",
             "options": [{"label": "1e-4", "description": "safe"}, {"label": "1e-3", "description": "fast"}]}]}},
            "echo STILL-RUNNING"]
    else:
        script = ["echo STILL-RUNNING"]
    with FakeModel(script) as fm:
        configure_backend(cfg, backend, cli, fm)
        store = Store(cfg.local.db_path)
        notes = Notes()
        agent = Agent(cfg, store, notes, HpcDomain(cfg, store, notes, HPC(SSHRunner(cfg), cfg.slurm.setup)))
        agent.set_backend(backend)
        await asyncio.wait_for(agent.handle(agent.general_session(), "Run the checks."), 180)
        main = next(r["body"] for r in fm.requests if "Run the checks." in json.dumps(r["body"]))
        tools = main.get("tools") or []
        names = {t.get("name") or t.get("type") for t in tools}
        assert names == EXPECTED_TOOLS[backend], sorted(names)
        if backend == "claude_code":  # subagents and commands run in the foreground
            for t in tools:
                assert "run_in_background" not in json.dumps(t.get("input_schema")), t["name"]
        outs = fm.outputs
        if backend == "codex":  # refused, and the run goes on
            assert "unavailable in Default mode" in outs[0], outs
        assert "STILL-RUNNING" in outs[-1], outs
        assert notes.sent[-1].text.startswith("🤖"), notes.texts()


@pytest.mark.parametrize("planner,worker,worker_model,planner_model", [
    ("claude_code", "codex", "gpt-6-luna", "claude-opus-5-5"),
    ("codex", "claude_code", "claude-sonnet-5-5", "gpt-6-astra"),
])
async def test_planner_mode_end_to_end(hpc, planner, worker, worker_model, planner_model):
    """With the real CLIs: the agent proposes planner mode (`ov propose`); once approved, a subagent on the other
    backend and a worker model runs its task in the sandbox, and the planner - now on its planner model - is woken
    with its report."""
    clis = {b: find_cli("claude" if b == "claude_code" else "codex") for b in ("claude_code", "codex")}
    if not all(clis.values()):
        pytest.skip("needs both claude and codex")
    cfg, remote, state, app = hpc
    plan = {"goal": "check the cluster from a subagent", "why": "e2e",
            "subagents": [{"name": "grid", "task": "check things", "backend": worker, "model": worker_model}]}
    scripts = [
        ("[Subagents you started have finished]", ["ov agents"]),
        ("[Task from your planner", ["echo WORKER-RAN", "ov status"]),
        ("Run the checks.", [f"ov propose '{json.dumps(plan)}'"]),
    ]
    with FakeModel([], scripts=scripts) as fm:
        for b in ("claude_code", "codex"):
            configure_backend(cfg, b, clis[b], fm)
        store = Store(cfg.local.db_path)
        notes = Notes()
        agent = Agent(cfg, store, notes, HpcDomain(cfg, store, notes, HPC(SSHRunner(cfg), cfg.slurm.setup)))
        agent.set_backend(planner)
        from overseer.core.autopilot import Autopilot
        from overseer.core.ctl import ControlServer
        from overseer.core.projects import Projects
        from overseer.core.subagents import Subagents
        ctl = ControlServer(cfg, agent, Projects(cfg, store, agent), notes)
        await ctl.start()
        Autopilot(cfg, store, agent, notes)
        subs = Subagents(cfg, store, agent, notes)
        s = agent.general_session()
        try:
            await asyncio.wait_for(agent.handle(s, "Run the checks."), 180)
            pending = subs.pending(s.id)
            assert pending, notes.texts()[-3000:]
            card = next(m for m in notes.sent if m.text.startswith("🧭"))
            assert "check the cluster from a subagent" in card.text and worker_model.split("-")[-1] in card.text.lower()
            assert "✓ plan" in " ".join(fm.by_marker.get("Run the checks.", [])), fm.by_marker
            await subs.approve(pending["id"])
            for _ in range(3):  # the subagent's run, then the planner's wake-up
                await asyncio.wait_for(subs.drain(), 240)
        finally:
            await ctl.stop()
        detail = f"\n\nTELEGRAM:\n{notes.texts()[-4000:]}\n\nBY SCRIPT:\n{json.dumps(fm.by_marker, indent=1)[-4000:]}"
        if os.environ.get("OVERSEER_E2E_DUMP"):
            Path(os.environ["OVERSEER_E2E_DUMP"], f"planner-{planner}.json").write_text(json.dumps(
                [r["body"] for r in fm.requests], indent=1))
        worker_outs = fm.by_marker.get("[Task from your planner", [])
        assert len(worker_outs) == 2 and "WORKER-RAN" in worker_outs[0], detail
        assert "you are subagent grid" in worker_outs[1], detail   # `ov` works from the subagent's sandbox
        planner_outs = fm.by_marker.get("[Subagents you started have finished]", [])
        assert planner_outs and "grid: ✅ done" in planner_outs[0] and "planner mode: on" in planner_outs[0], detail
        # Each ran on its model: the subagent on the worker model, the woken planner on the planner model.
        # (models in codex's code mode get their tools in an "additional_tools" input item, not in "tools")
        bodies = [r["body"] for r in fm.requests if r["body"].get("tools") or '"additional_tools"' in json.dumps(r["body"])]
        worker_req = next(b for b in bodies if "[Task from your planner" in json.dumps(b))
        planner_req = next(b for b in bodies if "[Subagents you started have finished]" in json.dumps(b))
        assert worker_req.get("model") == worker_model, worker_req.get("model")
        assert planner_req.get("model") == planner_model, planner_req.get("model")
        rec = store.subagents(parent=s.id)[0]
        assert rec.state == "done" and rec.delivered and "WORKER-RAN" in rec.report, rec


@pytest.mark.parametrize("backend,cli_name", [("claude_code", "claude"), ("codex", "codex")])
async def test_installed_skills_reach_the_cli(hpc, backend, cli_name):
    """A skill the researcher approved (addons.py) is installed into the run's folder, where the real CLI finds it and
    offers it to the model - next to Overseer's own skill."""
    cli = find_cli(cli_name)
    if not cli:
        pytest.skip(f"{cli_name} not installed")
    cfg, remote, state, app = hpc
    from overseer.core.addons import Addons
    from overseer.core.store import SkillRecord
    with FakeModel(["echo hi"]) as fm:
        configure_backend(cfg, backend, cli, fm)
        store = Store(cfg.local.db_path)
        notes = Notes()
        agent = Agent(cfg, store, notes, HpcDomain(cfg, store, notes, HPC(SSHRunner(cfg), cfg.slurm.setup)))
        agent.set_backend(backend)
        Addons(store, agent, notes)
        text = ("---\nname: cluster-long-jobs\ndescription: How to run jobs longer than 24 h on this cluster "
                "(qos long, checkpointing). Use when a job needs more than 24 h.\n---\nUse --qos long.\n")
        store.add_skill(SkillRecord(name="cluster-long-jobs", description="x", content=text, status="installed"))
        await asyncio.wait_for(agent.handle(agent.general_session(), "Run the checks."), 120)
        body = json.dumps(next(r["body"] for r in fm.requests if "Run the checks." in json.dumps(r["body"])))
        assert "cluster-long-jobs" in body and "jobs longer than 24 h on this cluster" in body, body[:3000]


@pytest.mark.parametrize("backend,cli_name", [("claude_code", "claude"), ("codex", "codex")])
async def test_assistant_sandbox_end_to_end(hpc, backend, cli_name, tmp_path):
    """The assistant bot next to the HPC bot, on the same machine: its agent writes the vault (not Obsidian's
    settings), uses `ov`, and can't reach anything of the HPC bot's - its SSH master connection, key or folders."""
    from overseer.assistant.domain import AssistantDomain
    from overseer.core.ctl import ControlServer
    from overseer.core.projects import Projects
    from overseer.hpc.sshlink import ensure_master

    cli = find_cli(cli_name)
    if not cli:
        pytest.skip(f"{cli_name} not installed")
    hcfg, remote, state, app = hpc
    hcfg.local.register("hpc")  # the HPC bot ran from this folder: its folders are noted
    await ensure_master(hcfg)  # ...and its SSH connection to the HPC is open
    sock = hcfg.control_socket
    assert sock.exists()
    vault = tmp_path / "Vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / ".obsidian" / "app.json").write_text("{}")
    (vault / "Journal.md").write_text("# Journal\n")
    short = tempfile.mkdtemp(prefix="ova-", dir="/tmp")
    cfg = Config(domain="assistant")
    cfg.local.base_dir = str(app)
    cfg.local.state_dir = "./state/assistant"
    cfg.local.tmp_dir = short
    cfg.telegram.token_env = "ASSISTANT_BOT_TOKEN"
    cfg.telegram.allowed_chat_ids = [1]
    cfg.vault.path = str(vault)
    cfg.agent.timeout_minutes = 3
    cfg.validate()
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(parents=True, exist_ok=True)
    cfg.local.register("assistant")
    (app / ".env").write_text("ASSISTANT_BOT_TOKEN=999:assistant-secret\n")
    probe = f"import socket; s = socket.socket(socket.AF_UNIX); s.connect({str(sock)!r}); print('MASTER-CONNECTED')"
    script = [
        f"echo '- written by the agent' >> {vault / 'Journal.md'} && echo VAULT-WRITE-OK",
        f"echo hacked > {vault / '.obsidian' / 'app.json'} 2>&1; cat {vault / '.obsidian' / 'app.json'}; echo OBSIDIAN-DONE",
        f"python3 -c \"{probe}\" 2>&1 | tail -1; echo MASTER-DONE",
        f"ssh -S {sock} -O check hpc 2>&1; ssh -S {sock} hpc 'echo REMOTE-REACHED' 2>&1 | tail -1; echo SSH-DONE",
        f"ls {Path(hcfg.local.tmp_dir)} 2>&1; echo HPCTMP-DONE",
        f"cat {hcfg.ssh_key} 2>&1 | head -c 60; echo; echo KEY-DONE",
        f"cat {app / '.env'} 2>&1; echo ENV-DONE",
        f"touch {app / 'assistant-was-here'} 2>&1; echo APPWRITE-DONE",
        "ov task add 'E2E task' --due tomorrow",
        "cat .agents/skills/assistant/SKILL.md 2>&1 | head -2; echo SKILL-DONE",
    ]
    try:
        with FakeModel(script) as fm:
            configure_backend(cfg, backend, cli, fm)
            store = Store(cfg.local.db_path)
            notes = Notes()
            domain = AssistantDomain(cfg, store, notes)
            domain.prepare()
            agent = Agent(cfg, store, notes, domain)
            agent.set_backend(backend)
            ctl = ControlServer(cfg, agent, Projects(cfg, store, agent), notes)
            await ctl.start()
            s = agent.general_session()
            try:
                await asyncio.wait_for(agent.handle(s, "Run the checks."), 240)
            finally:
                await ctl.stop()
        outs = fm.outputs
        detail = f"\n\nTELEGRAM:\n{notes.texts()}\n\nTOOL OUTPUTS:\n" + "\n====\n".join(outs)
        if os.environ.get("OVERSEER_E2E_VERBOSE"):
            print(detail)
        assert len(outs) >= len(script), detail
        assert "VAULT-WRITE-OK" in outs[0] and "written by the agent" in (vault / "Journal.md").read_text(), detail
        assert (vault / ".obsidian" / "app.json").read_text() == "{}", detail       # Obsidian's settings: refused
        assert "MASTER-CONNECTED" not in outs[2] and "MASTER-DONE" in outs[2], detail  # the HPC's SSH master: no
        assert "REMOTE-REACHED" not in outs[3] and "SSH-DONE" in outs[3], detail
        assert "master.sock" not in outs[4] and "agent-ssh" not in outs[4], detail  # the HPC bot's tmp folder
        assert "PRIVATE KEY" not in outs[5] and "KEY-DONE" in outs[5], detail
        assert "assistant-secret" not in outs[6] and "ENV-DONE" in outs[6], detail
        assert not (app / "assistant-was-here").exists(), detail
        assert "✓ added to Assistant/Tasks.md: E2E task 📅" in outs[8], detail
        assert "E2E task" in (vault / "Assistant" / "Tasks.md").read_text()
        if backend == "codex":
            assert "name: assistant" in outs[9], detail
        reply = notes.sent[-1]
        assert reply.buttons and reply.buttons[0][0][0] == "↩️ Undo", detail
        assert "Journal.md" in reply.text and "Assistant/Tasks.md" in reply.text, detail
        body = json.dumps([r["body"] for r in fm.requests if r["body"].get("tools")][0])
        assert "a personal assistant" in body and "You are Overseer's agent" not in body
        assert "assistant: How to work as the user's personal assistant" in body  # its skill, not the HPC's
        assert "overseer: How to talk to Overseer" not in body
    finally:
        shutil.rmtree(short, ignore_errors=True)
