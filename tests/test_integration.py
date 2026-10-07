"""End-to-end tests: real file operations via a local shell instead of SSH, fake SLURM binaries.

Everything above the transport (policy guard, realpath checks, monitor, agent tools,
proposal/approval flow) runs exactly as in production.
"""

import json
import os
import posixpath
from pathlib import Path

import pytest

from overseer.agent import Agent
from overseer.config import Config
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


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_ed25519").write_text("SECRET")
    (home / "proj" / "train.py").write_text("BATCH = 512\nprint('training')\n")
    (home / "proj" / "run.sh").write_text("#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --time=01:00:00\npython train.py\n")
    state = tmp_path / "slurm.json"
    state.write_text(json.dumps({"next_id": 5000, "jobs": {}}))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{FAKE_BIN}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state))

    cfg = Config()
    cfg.ssh.host = "local"
    cfg.sandbox.root = str(tmp_path / "sandbox")
    cfg.telegram.allowed_chat_ids = [1]
    cfg.llm.enabled = False
    cfg.slurm.poll_interval = 60
    store = Store(tmp_path / "state" / "o.db")
    hpc = HPC(cfg, LocalRunner(cfg.ssh, cfg.sandbox.root, tmp_path / "state" / "audit.log"))
    notifier = FakeNotifier()
    monitor = Monitor(cfg, hpc, store, notifier, Reporter(LLM(cfg.llm), None, False))
    submitter = Submitter(cfg, hpc, store, notifier)
    submitter.monitor = monitor
    agent = Agent(cfg, hpc, store, LLM(cfg.llm), notifier, submitter)

    class Env:
        pass

    e = Env()
    e.__dict__.update(cfg=cfg, home=home, state=state, store=store, hpc=hpc, notifier=notifier,
                      monitor=monitor, submitter=submitter, agent=agent, tmp=tmp_path)

    def set_jobs(**jobs):
        st = json.loads(state.read_text())
        st["jobs"].update(jobs)
        state.write_text(json.dumps(st))

    def slurm():
        return json.loads(state.read_text())

    e.set_jobs, e.slurm = set_jobs, slurm
    return e


async def test_sandbox_created_private(env):
    await env.hpc.connect()
    root = Path(env.cfg.sandbox.root)
    assert root.is_dir() and (root.stat().st_mode & 0o777) == 0o700


async def test_sandbox_with_open_permissions_is_refused(env):
    root = Path(env.cfg.sandbox.root)
    root.mkdir()
    root.chmod(0o777)
    with pytest.raises(PolicyError):
        await env.hpc.connect()


async def test_write_and_read_policy(env):
    await env.hpc.connect()
    sb = env.cfg.sandbox.root
    p = await env.hpc.write_file(f"{sb}/a/b.txt", "hello")
    assert Path(p).read_text() == "hello"

    with pytest.raises(PolicyError):
        await env.hpc.write_file(str(env.home / "proj" / "train.py"), "pwned")
    with pytest.raises(PolicyError):
        await env.hpc.write_file(f"{sb}/../escape.txt", "x")
    # Symlink inside the sandbox pointing at home must not allow writes through it.
    os.symlink(env.home / "proj", f"{sb}/link")
    with pytest.raises(PolicyError):
        await env.hpc.write_file(f"{sb}/link/train.py", "pwned")
    assert "BATCH = 512" in (env.home / "proj" / "train.py").read_text()

    assert "BATCH" in (await env.hpc.read_file("~/proj/train.py")).text
    with pytest.raises(PolicyError):
        await env.hpc.read_file("~/.ssh/id_ed25519")
    with pytest.raises(PolicyError):
        await env.hpc.read_file("/etc/hostname")
    with pytest.raises(PolicyError):
        await env.hpc.stage("~", f"{sb}/w")  # entire home
    with pytest.raises(PolicyError):
        await env.hpc.delete(sb)


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

    # Short job that started and finished between polls is still reported.
    env.set_jobs(**{"78": {"name": "quick", "state": "COMPLETED", "workdir": proj, "stdout": str(log)}})
    await env.monitor.poll()
    assert "Job 78" in env.notifier.sent[-1].text and "finished" in env.notifier.sent[-1].text


async def test_agent_fix_and_approval_flow(env):
    await env.hpc.connect()
    proj = str(env.home / "proj")
    env.set_jobs(**{"90": {"name": "train", "state": "FAILED", "exit": "1:0", "workdir": proj,
                           "stdout": f"{proj}/slurm-90.out", "stderr": f"{proj}/slurm-90.out"}})
    await env.monitor.poll(first=True)
    s = env.agent.session_for_job("90")
    for d in (s.work, s.orig):
        await env.hpc.mkdir(d)
    actions = []
    call = lambda name, **a: env.agent._run_tool(s, name, a, actions)  # noqa: E731

    out, err = await call("stage", src="~/proj/train.py")
    assert not err, out
    out, err = await call("edit_file", path="train.py", old_string="BATCH = 512", new_string="BATCH = 128")
    assert not err, out
    out, err = await call("edit_file", path=f"{proj}/train.py", old_string="BATCH", new_string="X")
    assert err and "POLICY" in out  # the original is untouchable
    assert "BATCH = 512" in (env.home / "proj" / "train.py").read_text()

    bad = "#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --time=01:00:00\npython train.py > ~/out.log\n"
    await call("write_file", path="run.sbatch", content=bad)
    out, err = await call("propose_submission", script_path="run.sbatch", summary="x")
    assert err and "redirect" in out

    good = "#!/bin/bash\n#SBATCH --job-name=fixed\n#SBATCH --gres=gpu:1\n#SBATCH --time=01:00:00\npython train.py --out $OVERSEER_RUN_DIR\n"
    await call("write_file", path="run.sbatch", content=good)
    out, err = await call("propose_submission", script_path="run.sbatch", summary="halve batch size")
    assert not err, out
    assert "AWAITING APPROVAL" in out
    proposal_msg = env.notifier.sent[-1]
    pid = proposal_msg.buttons[0][0][1].split(":")[1]
    assert any(name.endswith(".diff") and b"-BATCH = 512" in data and b"+BATCH = 128" in data for name, data in proposal_msg.files)
    assert env.slurm().get("submissions") is None  # nothing submitted before approval

    result = await env.submitter.approve(pid)
    assert "Submitted job 5001" in result, result
    sub = env.slurm()["submissions"][0]
    assert f"--chdir={s.work}" in sub and all(
        a.split("=", 1)[1].startswith(env.cfg.sandbox.root) for a in sub if a.startswith(("--chdir", "--output", "--error")))
    rec = env.store.get_job("5001")
    assert rec.overseer == 1 and rec.session_id == s.id and rec.parent_job == "90"
    assert "already" in await env.submitter.approve(pid)  # no double submission
    assert env.agent.session_for_job("5001") is s  # follow-ups continue the same conversation

    # Tampering between proposal and approval is detected.
    out, err = await call("propose_submission", script_path="run.sbatch", summary="again")
    pid2 = env.notifier.sent[-1].buttons[0][0][1].split(":")[1]
    Path(posixpath.join(s.work, "run.sbatch")).write_text(good + "rm -rf ~\n")
    assert "changed" in await env.submitter.approve(pid2)
    assert len(env.slurm()["submissions"]) == 1

    # Only Overseer jobs can be cancelled.
    out, err = await call("propose_cancel", job_id="90", reason="x")
    assert err
    out, err = await call("propose_cancel", job_id="5001", reason="wrong lr")
    assert not err
    pid3 = env.notifier.sent[-1].buttons[0][0][1].split(":")[1]
    assert "Cancelled" in await env.submitter.approve(pid3)
    assert env.slurm()["jobs"]["5001"]["state"] == "CANCELLED"

    audit = (env.tmp / "state" / "audit.log").read_text().splitlines()
    assert any(json.loads(line)["argv"][0] == "sbatch" for line in audit)


class FakeLLM:
    """Scripted model: first asks to stage + edit, then answers."""

    enabled = True

    def __init__(self):
        self.calls = []

    async def create(self, *, system, messages, effort, tools=None, max_tokens=None):
        from types import SimpleNamespace as NS
        self.calls.append(messages[:])
        n = len(self.calls)
        if n == 1:
            return NS(stop_reason="tool_use", content=[
                NS(type="text", text="Looking."),
                NS(type="tool_use", id="t1", name="stage", input={"src": "~/proj/train.py"}),
                NS(type="tool_use", id="t2", name="read_file", input={"path": "~/.ssh/id_ed25519"}),
            ])
        return NS(stop_reason="end_turn", content=[NS(type="text", text="Batch size too large; staged train.py.")])


async def test_agent_loop_with_scripted_model(env):
    await env.hpc.connect()
    env.agent.llm = FakeLLM()
    s = env.agent.general_session()
    await env.agent.handle(s, "why did my job fail?")
    calls = env.agent.llm.calls
    assert len(calls) == 2
    first_user = calls[0][0]["content"][0]["text"]
    assert "<workspace" in first_user and "why did my job fail?" in first_user
    results = calls[1][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[0]["is_error"] is False and results[1]["is_error"] is True and "POLICY" in results[1]["content"]
    texts = env.notifier.texts()
    assert "Batch size too large" in texts
    assert "Agent actions" in texts and "blocked read_file" in texts

    # Next turn carries pending events and keeps roles alternating.
    env.agent.note(s.id, "User approved proposal abc")
    await env.agent.handle(s, "thanks")
    roles = [m["role"] for m in env.agent.llm.calls[-1]]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    assert "User approved proposal abc" in env.agent.llm.calls[-1][-1]["content"][0]["text"]


def test_build_app(env, monkeypatch, tmp_path):
    from overseer.__main__ import build

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TEST")
    env.cfg.state.dir = str(tmp_path / "st")
    app = build(env.cfg)
    assert app.handlers


async def test_bot_routing_and_approval_button(env, monkeypatch):
    from types import SimpleNamespace as NS

    from overseer.bot import Bot

    await env.hpc.connect()
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              submitter=env.submitter, notifier=env.notifier)
    started = []

    async def fake_run_agent(chat_id, session, text):
        started.append((session.id, text))

    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)

    # A reply to a job message is routed to that job's agent session.
    env.set_jobs(**{"55": {"name": "t", "state": "FAILED", "workdir": str(env.home / "proj")}})
    await env.monitor.poll(first=True)
    env.store.link_message(1, 999, "55", None)
    msg = NS(text="use lr 1e-4", reply_to_message=NS(message_id=999))
    await bot.on_text(NS(effective_message=msg, effective_chat=NS(id=1)), None)
    for t in list(bot._tasks):
        await t
    assert started[-1] == (env.agent.session_for_job("55").id, "use lr 1e-4")

    # Buttons from an unknown chat are refused; from the allowed chat they approve.
    s = env.agent.session_for_job("55")
    for d in (s.work, s.orig):
        await env.hpc.mkdir(d)
    await env.hpc.write_file(f"{s.work}/run.sbatch", "#!/bin/bash\n#SBATCH --time=00:10:00\necho hi\n")
    await env.submitter.propose_submit(s, "run.sbatch", "test")
    pid = env.notifier.sent[-1].buttons[0][0][1].split(":")[1]

    answers = []

    async def answer(*a, **k):
        answers.append(k)

    async def edit_markup(*a, **k):
        pass

    def query(chat_id):
        q = NS(data=f"approve:{pid}", answer=answer, edit_message_reply_markup=edit_markup, message=NS(message_id=5))
        return NS(callback_query=q, effective_chat=NS(id=chat_id))

    await bot.on_button(query(666), None)
    assert answers[-1].get("show_alert") and "submissions" not in env.slurm()
    await bot.on_button(query(1), None)
    assert len(env.slurm()["submissions"]) == 1
    assert "Submitted job" in env.notifier.sent[-1].text
    assert any("approved proposal" in n for n in s.notes)
