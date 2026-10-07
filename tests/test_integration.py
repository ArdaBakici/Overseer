"""End-to-end tests: real file operations via a local shell instead of SSH, fake SLURM binaries.

Everything above the transport (policy guard, realpath checks, grants, backups/revert,
monitor, agent tools, proposal/approval flow) runs exactly as in production.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from overseer.agent import Agent
from overseer.config import Config, ConfigError
from overseer.diagnose import Reporter
from overseer.hpc import HPC
from overseer.llm import LLM
from overseer.monitor import Monitor
from overseer.policy import PolicyError
from overseer.proposals import Submitter
from overseer.ssh import SSHRunner
from overseer.store import Store

FAKE_BIN = Path(__file__).parent / "fake_slurm"


class LocalRunner(SSHRunner):
    def build_command(self, argv):
        return ["sh", "-c", self.remote_command(argv)]


class FakeNotifier:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    def texts(self):
        return "\n".join(m.text for m in self.sent)

    def last_pid(self):
        return self.sent[-1].buttons[0][0][1].split(":")[1]


def make_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_ed25519").write_text("SECRET")
    (home / ".bashrc").write_text("# rc\n")
    (home / "proj" / "train.py").write_text("BATCH = 512\nprint('training')\n")
    scratch = tmp_path / "scratch" / "exp"
    scratch.mkdir(parents=True)
    state = tmp_path / "slurm.json"
    state.write_text(json.dumps({"next_id": 5000, "jobs": {}}))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{FAKE_BIN}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state))

    cfg = Config()
    cfg.ssh.host = "local"
    cfg.local.root = str(tmp_path / "local")
    cfg.hpc.write_roots = [str(scratch)]
    cfg.hpc.read_roots = ["~"]
    cfg.telegram.allowed_chat_ids = [1]
    cfg.llm.enabled = False
    cfg.validate()
    store = Store(cfg.local.db_path)
    hpc = HPC(cfg, LocalRunner(cfg.ssh, lambda: hpc.write_roots, cfg.local.control_path, cfg.local.audit_log))
    hpc.grants = store.grants()
    notifier = FakeNotifier()
    monitor = Monitor(cfg, hpc, store, notifier, Reporter(LLM(cfg.llm), None, False))
    submitter = Submitter(cfg, hpc, store, notifier)
    submitter.monitor = monitor
    agent = Agent(cfg, hpc, store, LLM(cfg.llm), notifier, submitter)

    def set_jobs(**jobs):
        st = json.loads(state.read_text())
        st["jobs"].update(jobs)
        state.write_text(json.dumps(st))

    return NS(cfg=cfg, home=home, scratch=scratch, state=state, store=store, hpc=hpc, notifier=notifier,
              monitor=monitor, submitter=submitter, agent=agent, tmp=tmp_path, set_jobs=set_jobs,
              slurm=lambda: json.loads(state.read_text()))


@pytest.fixture
def env(tmp_path, monkeypatch):
    return make_env(tmp_path, monkeypatch)


def test_config_validation():
    cfg = Config()
    cfg.ssh.host = "h"
    cfg.telegram.allowed_chat_ids = [1]
    cfg.local.root = "/var/lib/overseer"
    with pytest.raises(ConfigError, match="inside /tmp"):
        cfg.validate()
    cfg.local.root = "/tmp/overseer"
    cfg.hpc.write_roots = ["/"]
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg.hpc.write_roots = ["relative/dir"]
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg.hpc.write_roots = ["~/projects", "/scratch/$USER/exp"]
    cfg.validate()
    with pytest.raises(ConfigError):
        cfg.local.path("..", "etc")


async def test_nothing_writable_before_connect(env):
    with pytest.raises(PolicyError):
        await env.hpc.run(["mkdir", "-p", str(env.scratch / "x")])
    await env.hpc.connect()
    assert env.hpc.write_roots == [str(env.scratch)]
    await env.hpc.mkdir(str(env.scratch / "x"))


async def test_write_and_read_policy(env):
    await env.hpc.connect()
    p = await env.hpc.write_file(str(env.scratch / "a" / "b.txt"), "hello")
    assert Path(p).read_text() == "hello"

    with pytest.raises(PolicyError, match="request_access"):
        await env.hpc.write_file("~/proj/train.py", "pwned")
    with pytest.raises(PolicyError):
        await env.hpc.write_file(str(env.scratch / ".." / "escape.txt"), "x")
    os.symlink(env.home / "proj", env.scratch / "link")  # symlink inside a write root
    with pytest.raises(PolicyError):
        await env.hpc.write_file(str(env.scratch / "link" / "train.py"), "pwned")
    assert "BATCH = 512" in (env.home / "proj" / "train.py").read_text()
    with pytest.raises(PolicyError):
        await env.hpc.delete(str(env.scratch))  # the root itself

    assert "BATCH" in (await env.hpc.read_file("~/proj/train.py")).text
    for bad in ("~/.ssh/id_ed25519", "/etc/hostname"):
        with pytest.raises(PolicyError):
            await env.hpc.read_file(bad)
    with pytest.raises(PolicyError):
        await env.hpc.copy("~", str(env.scratch / "home-copy"))
    with pytest.raises(PolicyError):
        await env.hpc.copy("~/.ssh", str(env.scratch / "keys"))


async def test_grant_rules(env):
    await env.hpc.connect()
    for path, mode in [("/", "write"), ("~", "write"), ("~/.ssh", "read"), ("~/.bashrc", "write"), ("rel", "write")]:
        with pytest.raises(PolicyError):
            await env.submitter.grant(path, mode, "x")
    await env.submitter.grant("~", "read", "x")  # whole home is OK for reading
    await env.submitter.grant("~/proj", "write", "x")
    with pytest.raises(PolicyError, match="never-writable"):
        await env.hpc.write_file("~/proj/.git/hooks/pre-commit", "x")
    # Grants are persisted in the local DB, so they survive a restart.
    assert Store(env.cfg.local.db_path).grants() == {str(env.home / "proj"): "write", str(env.home): "read"}


async def test_access_request_backup_and_revert(env):
    await env.hpc.connect()
    s = env.agent.general_session()
    actions = []

    def call(name, **a):
        return env.agent._run_tool(s, name, a, actions)

    out, err = await call("edit_file", path="~/proj/train.py", old_string="BATCH = 512", new_string="BATCH = 64")
    assert err and "request_access" in out

    out, err = await call("request_access", path="~/proj", mode="write", reason="fix batch size in place")
    assert not err and "END YOUR TURN" in out
    pid = env.notifier.last_pid()
    assert "Access request" in env.notifier.sent[-1].text
    assert "Granted write" in await env.submitter.approve(pid)
    assert env.store.grants() == {str(env.home / "proj"): "write"}

    out, err = await call("edit_file", path="~/proj/train.py", old_string="BATCH = 512", new_string="BATCH = 64")
    assert not err, out
    out, err = await call("write_file", path="~/proj/notes.md", content="new file\n")
    assert not err, out
    assert (env.home / "proj" / "train.py").read_text().startswith("BATCH = 64")

    # Originals are backed up on the local machine, inside local.root (/tmp/...).
    backups = list((Path(env.cfg.local.root) / "sessions" / s.id).glob("*.orig"))
    assert len(backups) == 1 and backups[0].read_text().startswith("BATCH = 512")
    assert oct(backups[0].stat().st_mode & 0o777) == "0o600"

    diff, _ = await call("show_changes")
    assert "-BATCH = 512" in diff and "+BATCH = 64" in diff and "+new file" in diff

    result = await env.agent.revert(s.id)
    assert "restored" in result and "removed" in result
    assert (env.home / "proj" / "train.py").read_text().startswith("BATCH = 512")
    assert not (env.home / "proj" / "notes.md").exists()

    assert "Revoked" in await env.submitter.revoke("~/proj")
    out, err = await call("edit_file", path="~/proj/train.py", old_string="BATCH = 512", new_string="BATCH = 1")
    assert err

    await call("request_access", path="~/proj", mode="write", reason="again")
    assert "Denied" in await env.submitter.reject(env.notifier.last_pid())
    assert env.store.grants() == {}


async def test_monitor_lifecycle(env):
    proj = str(env.home / "proj")
    log = env.home / "proj" / "slurm-77.out"
    env.set_jobs(**{"77": {"name": "train", "state": "PENDING", "workdir": proj, "stdout": str(log), "stderr": str(log),
                           "script": "#!/bin/bash\npython train.py\n"}})
    await env.monitor.poll(first=True)
    assert "Overseer online" in env.notifier.texts()

    env.set_jobs(**{"77": {**env.slurm()["jobs"]["77"], "state": "RUNNING"}})
    await env.monitor.poll()
    assert "🚀" in env.notifier.sent[-1].text

    log.write_text("epoch 1/3 loss 1.0\nTraceback (most recent call last):\n  File \"train.py\", line 1\n"
                   "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4 GiB\n")
    env.set_jobs(**{"77": {**env.slurm()["jobs"]["77"], "state": "FAILED", "exit": "1:0"}})
    await env.monitor.poll()
    final = env.notifier.sent[-1]
    assert "FAILED" in final.text and "GPU out of memory" in final.text
    assert ("🔧 Diagnose & fix", "fix:77") in final.buttons[0]
    rec = env.store.get_job("77")
    assert rec.ended_at and rec.batch_script.startswith("#!/bin/bash")

    env.set_jobs(**{"78": {"name": "quick", "state": "COMPLETED", "workdir": proj, "stdout": str(log)}})
    await env.monitor.poll()
    assert "Job 78" in env.notifier.sent[-1].text and "finished" in env.notifier.sent[-1].text


async def test_agent_fix_and_submission_flow(env):
    proj = str(env.home / "proj")
    env.set_jobs(**{"90": {"name": "train", "state": "FAILED", "exit": "1:0", "workdir": proj,
                           "stdout": f"{proj}/slurm-90.out", "stderr": f"{proj}/slurm-90.out"}})
    await env.monitor.poll(first=True)
    s = env.agent.session_for_job("90")
    actions = []

    def call(name, **a):
        return env.agent._run_tool(s, name, a, actions)

    fix = env.scratch / "fix1"
    out, err = await call("copy_path", src="~/proj", dest=str(fix))
    assert not err, out
    out, err = await call("edit_file", path=str(fix / "train.py"), old_string="BATCH = 512", new_string="BATCH = 128")
    assert not err, out
    assert "BATCH = 512" in (env.home / "proj" / "train.py").read_text()

    bad = "#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --time=01:00:00\npython train.py > ~/out.log\n"
    await call("write_file", path=str(fix / "run.sbatch"), content=bad)
    out, err = await call("propose_submission", script_path=str(fix / "run.sbatch"), summary="x")
    assert err and "redirect" in out

    good = "#!/bin/bash\n#SBATCH --job-name=fixed\n#SBATCH --gres=gpu:1\n#SBATCH --time=01:00:00\npython train.py\n"
    await call("write_file", path=str(fix / "run.sbatch"), content=good)
    out, err = await call("propose_submission", script_path=str(fix / "run.sbatch"), summary="halve batch size")
    assert not err, out
    msg = env.notifier.sent[-1]
    pid = env.notifier.last_pid()
    diff = dict(msg.files)[f"changes-{pid}.diff"].decode()
    assert "# copied ~/proj" in diff and "-BATCH = 512" in diff and "+BATCH = 128" in diff
    assert env.slurm().get("submissions") is None  # nothing submitted before approval

    result = await env.submitter.approve(pid)
    assert "Submitted job 5001" in result, result
    assert env.slurm()["submissions"][0] == ["--parsable", f"--chdir={fix}", str(fix / "run.sbatch")]
    rec = env.store.get_job("5001")
    assert rec.overseer == 1 and rec.session_id == s.id and rec.parent_job == "90"
    assert rec.stdout == f"{fix}/slurm-5001.out"
    assert "already" in await env.submitter.approve(pid)
    assert env.agent.session_for_job("5001") is s

    # Tampering between proposal and approval is detected.
    await call("propose_submission", script_path=str(fix / "run.sbatch"), summary="again")
    pid2 = env.notifier.last_pid()
    (fix / "run.sbatch").write_text(good + "rm -rf ~\n")
    assert "changed" in await env.submitter.approve(pid2)
    assert len(env.slurm()["submissions"]) == 1

    out, err = await call("propose_cancel", job_id="90", reason="x")
    assert err
    out, err = await call("propose_cancel", job_id="5001", reason="wrong lr")
    assert not err
    assert "Cancelled" in await env.submitter.approve(env.notifier.last_pid())
    assert env.slurm()["jobs"]["5001"]["state"] == "CANCELLED"

    # Revert removes the copy entirely.
    await env.agent.revert(s.id)
    assert not fix.exists()

    audit = [json.loads(line) for line in env.cfg.local.audit_log.read_text().splitlines()]
    assert any(a["argv"][0] == "sbatch" for a in audit)


class FakeLLM:
    """Scripted model: first reads a file + tries a secret, then answers."""

    enabled = True

    def __init__(self):
        self.calls = []

    async def create(self, *, system, messages, effort, tools=None, max_tokens=None):
        self.calls.append(messages[:])
        if len(self.calls) == 1:
            return NS(stop_reason="tool_use", content=[
                NS(type="text", text="Looking."),
                NS(type="tool_use", id="t1", name="read_file", input={"path": "~/proj/train.py"}),
                NS(type="tool_use", id="t2", name="read_file", input={"path": "~/.ssh/id_ed25519"}),
            ])
        return NS(stop_reason="end_turn", content=[NS(type="text", text="Batch size too large.")])


async def test_agent_loop_with_scripted_model(env):
    await env.hpc.connect()
    env.agent.llm = FakeLLM()
    s = env.agent.general_session()
    await env.agent.handle(s, "why did my job fail?")
    calls = env.agent.llm.calls
    assert len(calls) == 2
    first_user = calls[0][0]["content"][0]["text"]
    assert "<access" in first_user and str(env.scratch) in first_user and "why did my job fail?" in first_user
    results = calls[1][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[0]["is_error"] is False and results[1]["is_error"] is True and "POLICY" in results[1]["content"]
    texts = env.notifier.texts()
    assert "Batch size too large" in texts and "blocked read_file" in texts

    env.agent.note(s.id, "User approved proposal abc")
    await env.agent.handle(s, "thanks")
    roles = [m["role"] for m in env.agent.llm.calls[-1]]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    assert "User approved proposal abc" in env.agent.llm.calls[-1][-1]["content"][0]["text"]


async def test_bot_routing_access_resume_and_revert(env, monkeypatch):
    from overseer.bot import Bot

    await env.hpc.connect()
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              submitter=env.submitter, notifier=env.notifier)
    started = []

    async def fake_run_agent(chat_id, session, text):
        started.append((session.id, text))

    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)

    async def drain():
        for t in list(bot._tasks):
            await t

    # A reply to a job message is routed to that job's agent session.
    env.set_jobs(**{"55": {"name": "t", "state": "FAILED", "workdir": str(env.home / "proj")}})
    await env.monitor.poll(first=True)
    env.store.link_message(1, 999, "55", None)
    msg = NS(text="use lr 1e-4", reply_to_message=NS(message_id=999))
    await bot.on_text(NS(effective_message=msg, effective_chat=NS(id=1)), None)
    await drain()
    s = env.agent.session_for_job("55")
    assert started[-1] == (s.id, "use lr 1e-4")

    answers = []

    async def answer(*a, **k):
        answers.append(k)

    async def edit_markup(*a, **k):
        pass

    def press(data, chat_id=1):
        q = NS(data=data, answer=answer, edit_message_reply_markup=edit_markup, message=NS(message_id=5))
        return bot.on_button(NS(callback_query=q, effective_chat=NS(id=chat_id)), None)

    # Access approval resumes the waiting agent; strangers can't press buttons.
    await env.submitter.propose_access(s, "~/proj", "write", "edit in place")
    pid = env.notifier.last_pid()
    await press(f"approve:{pid}", chat_id=666)
    assert answers[-1].get("show_alert") and env.store.grants() == {}
    await press(f"approve:{pid}")
    await drain()
    assert env.store.grants() == {str(env.home / "proj"): "write"}
    assert started[-1][0] == s.id and "approved your access request" in started[-1][1]

    # Submission approval via button.
    await env.agent._run_tool(s, "write_file", {"path": str(env.scratch / "run.sbatch"),
                                                "content": "#!/bin/bash\n#SBATCH --time=00:10:00\necho hi\n"}, [])
    await env.submitter.propose_submit(s, str(env.scratch / "run.sbatch"), "test")
    await press(f"approve:{env.notifier.last_pid()}")
    assert len(env.slurm()["submissions"]) == 1 and "Submitted job" in env.notifier.sent[-1].text

    # Revert button asks for confirmation, then reverts.
    await press(f"revert:{s.id}")
    assert "Revert these changes?" in env.notifier.sent[-1].text
    await press(f"revertok:{s.id}")
    assert "Reverted" in env.notifier.sent[-1].text and not (env.scratch / "run.sbatch").exists()


def test_build_app(env, monkeypatch):
    from overseer.__main__ import build

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TEST")
    app = build(env.cfg)
    assert app.handlers
    assert (Path(env.cfg.local.root).stat().st_mode & 0o777) == 0o700
