"""Integration tests: monitor, agent bookkeeping, runners and bot, with fake SLURM and fake agent CLIs.

Monitoring runs its HPC commands through a local shell instead of SSH (fake squeue/sacct/...
from tests/fake_slurm). The agent CLIs are tests/fake_cli (canned Claude Code / Codex event
streams), so everything around them - argv, sandbox settings, env, event parsing, job
detection, sessions, Telegram reports - runs for real.
"""

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from overseer.agent import Agent
from overseer.config import Config, ConfigError
from overseer.diagnose import Reporter
from overseer.hpc import HPC
from overseer.monitor import Monitor
from overseer.runners import Activity, ClaudeCodeRunner, CodexRunner, display_command
from overseer.ssh import SSHRunner
from overseer.sshlink import agent_ssh_config, overseer_ssh_config, write_configs
from overseer.notify import Outgoing
from overseer.store import JobRecord, Store, WatchRecord

FAKE_SLURM = Path(__file__).parent / "fake_slurm"
FAKE_CLI = Path(__file__).parent / "fake_cli"


class LocalRunner(SSHRunner):
    def build_command(self, argv):
        return ["sh", "-c", self.remote_command(argv)]


class FakeNotifier:
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


def make_cfg(tmp_path, short_tmp):
    cfg = Config()
    cfg.ssh.hostname = "login.example.edu"
    cfg.ssh.user = "netid"
    cfg.telegram.allowed_chat_ids = [1]
    app = tmp_path / "app"  # stands in for Overseer's own folder (where config.yaml lives)
    app.mkdir(exist_ok=True)
    cfg.local.base_dir = str(app)
    cfg.local.tmp_dir = short_tmp  # unix socket paths must stay short
    cfg.agent.claude_code.command = str(FAKE_CLI / "claude")
    cfg.agent.codex.command = str(FAKE_CLI / "codex")
    cfg.agent.claude_code.pass_env += ["FAKE_CLI_LOG", "FAKE_SLURM_STATE"]
    cfg.agent.codex.pass_env += ["FAKE_CLI_LOG", "FAKE_SLURM_STATE"]
    cfg.validate()
    return cfg


@pytest.fixture
def short_tmp():
    d = tempfile.mkdtemp(prefix="ovt-", dir="/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def env(tmp_path, monkeypatch, short_tmp):
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    state = tmp_path / "slurm.json"
    state.write_text(json.dumps({"next_id": 5000, "jobs": {}}))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{FAKE_SLURM}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state))
    monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "cli.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:SECRET")

    cfg = make_cfg(tmp_path, short_tmp)
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(parents=True, exist_ok=True)
    write_configs(cfg)
    store = Store(cfg.local.db_path)
    runner = LocalRunner(cfg)
    hpc = HPC(runner)
    notifier = FakeNotifier()
    agent = Agent(cfg, store, notifier, hpc)
    reporter = Reporter(lambda system, prompt: agent.runner().text(system=system, prompt=prompt), True, False)
    monitor = Monitor(cfg, hpc, store, notifier, reporter)
    agent.monitor = monitor

    async def no_master(_cfg):
        return None

    monkeypatch.setattr("overseer.agent.ensure_master", no_master)

    def set_jobs(**jobs):
        st = json.loads(state.read_text())
        st["jobs"].update(jobs)
        state.write_text(json.dumps(st))

    def cli_log(lookups: bool = False):
        """Calls to the fake CLIs; `codex debug models` lookups only with lookups=True."""
        p = tmp_path / "cli.jsonl"
        calls = [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
        return calls if lookups else [c for c in calls if c["argv"][:1] != ["debug"]]

    return NS(cfg=cfg, home=home, store=store, hpc=hpc, notifier=notifier, agent=agent, monitor=monitor,
              tmp=tmp_path, set_jobs=set_jobs, cli_log=cli_log,
              slurm=lambda: json.loads(state.read_text()))


# ----------------------------------------------------------------------------- config & ssh files

def test_config_validation(tmp_path, short_tmp):
    cfg = make_cfg(tmp_path, short_tmp)
    assert cfg.local.db_path == (tmp_path / "app" / "state" / "overseer.db").resolve()
    assert cfg.ssh_dir == (tmp_path / "app" / "state" / "ssh").resolve()

    for mutate, needle in [
        (lambda c: setattr(c.ssh, "hostname", ""), "hostname"),
        (lambda c: setattr(c.local, "state_dir", "/var/lib/overseer"), "outside the allowed local folders"),
        (lambda c: setattr(c.local, "allowed_roots", ["/"]), "too broad"),
        (lambda c: setattr(c.ssh, "dir", "/etc/overseer-ssh"), "outside the allowed local folders"),
        (lambda c: setattr(c.local, "tmp_dir", "/tmp/" + "x" * 120), "too long"),
        (lambda c: setattr(c.agent, "backend", "opencode"), "agent.backend"),
        (lambda c: setattr(c.agent, "report_commands", "loud"), "report_commands"),
    ]:
        c = make_cfg(tmp_path, short_tmp)
        mutate(c)
        with pytest.raises(ConfigError, match=needle):
            c.validate()


def test_load_config_resolves_relative_to_config_file(tmp_path):
    from overseer.config import load_config

    (tmp_path / "config.yaml").write_text("ssh: {hostname: login.x.edu}\ntelegram: {allowed_chat_ids: [1]}\n")
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.local.state() == (tmp_path / "state").resolve()
    assert cfg.ssh_key == (tmp_path / "state" / "ssh" / "id_ed25519").resolve()


def test_env_file(tmp_path, monkeypatch):
    from overseer.config import load_config, parse_env_file

    assert parse_env_file('# c\nexport A=1\nB = "two words"\nC=three # note\nD=\'x#y\'\n\n') == \
        {"A": "1", "B": "two words", "C": "three", "D": "x#y"}
    with pytest.raises(ConfigError):
        parse_env_file("not a valid line")

    (tmp_path / "config.yaml").write_text("ssh: {hostname: h.x.edu}\ntelegram: {allowed_chat_ids: [1]}\n")
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=from-dotenv\nOVERSEER_TEST_KEEP=from-dotenv\n")
    env_file.chmod(0o600)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("OVERSEER_TEST_KEEP", "from-real-env")
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.telegram.token == "from-dotenv"
    assert os.environ["OVERSEER_TEST_KEEP"] == "from-real-env"  # the real environment wins
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    assert cfg.local.env_path == env_file.resolve()


def test_ssh_files(env):
    cfg = env.cfg
    own = overseer_ssh_config(cfg)
    assert f"IdentityFile {cfg.ssh_key}" in own and "IdentitiesOnly yes" in own
    assert f"UserKnownHostsFile {cfg.ssh_dir / 'known_hosts'}" in own
    assert f"ControlPath {cfg.control_socket}" in own and "ControlMaster auto" in own
    agent_cfg = agent_ssh_config(cfg)
    assert "IdentityFile none" in agent_cfg and "ControlMaster no" in agent_cfg
    assert str(cfg.ssh_dir) not in agent_cfg  # the agent's config doesn't even mention the key folder
    for name in ("scp", "sftp"):
        w = cfg.local.tmp("agent-bin", name)
        assert os.access(w, os.X_OK) and f"-F {cfg.agent_ssh_config}" in w.read_text()
    w = cfg.local.tmp("agent-bin", "ssh")  # see test_login_node_cpu_limit
    assert os.access(w, os.X_OK) and repr(str(cfg.agent_ssh_config)) in w.read_text()
    assert os.access(cfg.local.tmp("agent-bin", "hpc-run"), os.X_OK)
    assert (cfg.ssh_dir.stat().st_mode & 0o777) == 0o700
    assert (cfg.ssh_config_file.stat().st_mode & 0o777) == 0o600


def test_login_node_cpu_limit(env, tmp_path):
    """The agent's ssh caps the CPU time of commands run directly on the login node; commands that
    hand work to SLURM are left alone (SLURM would pass the limit on to the job)."""
    import subprocess
    from overseer.compute import write_agent_tools
    from overseer.sshlink import _write
    fake = tmp_path / "real-ssh"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$FAKE_SSH_ARGS\"\nexit ${FAKE_SSH_RC:-0}\n")
    fake.chmod(0o755)
    bin_dir = tmp_path / "bin"
    write_agent_tools(env.cfg, bin_dir, str(fake), _write)

    def ssh(*args, rc=0):
        out = tmp_path / "args"
        p = subprocess.run([str(bin_dir / "ssh"), *args], capture_output=True, text=True,
                           env={**os.environ, "FAKE_SSH_ARGS": str(out), "FAKE_SSH_RC": str(rc)})
        return p, out.read_text().splitlines()

    _, argv = ssh("hpc", "python train.py --lr 3e-4")
    assert argv[:2] == ["-F", str(env.cfg.agent_ssh_config)]
    assert argv[2:] == ["hpc", "ulimit -H -t 130 2>/dev/null; ulimit -S -t 120 2>/dev/null; python train.py --lr 3e-4",
                        "exit $?"]
    _, argv = ssh("-o", "ConnectTimeout=5", "-p22", "-t", "hpc", "bash", "-lc", "'make all'")
    assert argv[-2] == "ulimit -H -t 130 2>/dev/null; ulimit -S -t 120 2>/dev/null; bash -lc 'make all'"
    assert argv[-3] == "hpc"  # options and their values skipped correctly
    for dispatching in ["cd ~/proj && sbatch train.sh", "bash -lc 'srun --jobid=5 --overlap nvidia-smi'",
                        "salloc -N1", "rsync --server -vlogDtpre.iLsfxC . ~/x"]:
        assert ssh("hpc", dispatching)[1][-1] == dispatching, dispatching
    assert ssh("hpc")[1] == ["-F", str(env.cfg.agent_ssh_config), "hpc"]  # no command: untouched
    p, _ = ssh("hpc", "python heavy.py", rc=152)  # killed by SIGXCPU: says what to do instead
    assert p.returncode == 152 and "login node's CPU limit (120 CPU seconds)" in p.stderr and "hpc-run" in p.stderr
    env.cfg.slurm.compute.login_cpu_seconds = 0  # switched off
    write_agent_tools(env.cfg, bin_dir, str(fake), _write)
    assert ssh("hpc", "python heavy.py")[1][-1] == "python heavy.py"


def test_ssh_setup_generates_and_imports_key(env, tmp_path):
    from overseer.sshlink import key_permissions_ok, setup_key

    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen not installed")
    pub = setup_key(env.cfg)
    assert pub.startswith("ssh-ed25519 ") and key_permissions_ok(env.cfg)
    assert setup_key(env.cfg) == pub  # idempotent

    other = tmp_path / "mykey"
    os.system(f"ssh-keygen -q -t ed25519 -N '' -f {other}")
    assert setup_key(env.cfg, str(other)) == (tmp_path / "mykey.pub").read_text().strip()
    assert env.cfg.ssh_key.read_text() == other.read_text()


def test_store_migrates_old_database(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, overseer INTEGER)")
    con.commit()
    con.close()
    store = Store(db)
    store.upsert_session(__import__("overseer.store", fromlist=["SessionRecord"]).SessionRecord(id="s1"))
    assert store.get_session("s1").id == "s1"
    assert store.get_job("x") is None

    # A database from before autopilot: the new job columns are added in place.
    db2 = tmp_path / "v4.db"
    con = sqlite3.connect(db2)
    con.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, name TEXT, state TEXT, partition TEXT, workdir TEXT, "
                "stdout TEXT, stderr TEXT, batch_script TEXT, by_agent INTEGER DEFAULT 0, session_id TEXT, "
                "first_seen REAL, last_status_at REAL, ended_at REAL, missing_polls INTEGER DEFAULT 0, report TEXT, "
                "info TEXT)")
    con.execute("INSERT INTO jobs (job_id, name, by_agent) VALUES ('7', 'old', 1)")
    con.commit()
    con.close()
    rec = Store(db2).get_job("7")
    assert rec.name == "old" and rec.agent_reported == 0 and rec.checkins == 0


# ----------------------------------------------------------------------------- activity parsing

def test_job_detection_ignores_lookalikes():
    from overseer.runners import submitted_jobs
    not_submissions = [
        ("ssh hpc 'cd ~/rf-dreamer/sbatch && sacct -u $USER -n -X -o JobID --parsable2 -S now-2days'", "12345\n12346\n"),
        ("ssh hpc 'bash -lc \"sacct -j 12345 --parsable2 -n -o JobID; cat scripts/train.sbatch\"'", "12345\n"),
        ("ssh hpc 'sacct -j 12345 -X -n -o JobID --parsable2; head run.sbatch'", "12345\n#!/bin/bash\n"),
        ("ssh hpc 'tail -5 logs/sweep.out'", "Submitted batch job 12345\n"),
        ("ssh hpc 'cat ~/notes/sbatch-history.txt'", "Submitted batch job 12345\n"),
        ("ssh hpc 'squeue -u $USER -h -o %i'", "12345\n"),
    ]
    for cmd, out in not_submissions:
        assert submitted_jobs(cmd, out) == [], cmd
    submissions = [
        ("ssh hpc 'cd ~/proj && sbatch train.sh'", "Submitted batch job 1\n", ["1"]),
        ("ssh hpc 'bash -lc \"cd ~/proj && sbatch -p gpu --gres=gpu:a100:1 train.sbatch\"'", "Submitted batch job 2\n", ["2"]),
        ("ssh hpc 'JOB=$(sbatch --parsable run.sh); echo $JOB'", "3\n", ["3"]),
        ("ssh hpc 'for s in 1 2; do /usr/bin/sbatch --parsable run.sh $s; done'", "4\n5;cl\n", ["4", "5"]),
        ("ssh hpc 'sbatch --parsable run.sh; sacct -j 77 -n -X -o JobID --parsable2'", "6\n77\n", ["6"]),
    ]
    for cmd, out, ids in submissions:
        assert submitted_jobs(cmd, out) == ids, cmd


def test_activity_and_job_detection():
    seen = []
    a = Activity(on_job=lambda j, c: seen.append(j))
    assert display_command("/bin/bash -lc 'ssh hpc squeue -u me'") == "ssh hpc squeue -u me"
    assert display_command("ssh hpc ls") == "ssh hpc ls"
    a.command("/bin/bash -lc 'ssh hpc sbatch run.sh'")
    a.output("ssh hpc sbatch run.sh", "Submitted batch job 123\n")
    a.output("ssh hpc 'sbatch --parsable a.sh'", "456;cluster\n")
    a.output("ssh hpc 'sbatch --parsable a.sh'", "456\n")  # duplicate
    a.output("ssh hpc squeue", "789\n")  # not an sbatch: ignored
    a.output("ssh hpc 'for f in *.sh; do sbatch $f; done'", "Submitted batch job 1\nSubmitted batch job 2\n")
    assert seen == ["123", "456", "1", "2"]
    assert a.commands == ["ssh hpc sbatch run.sh"]


# ----------------------------------------------------------------------------- runners (fake CLIs)

def covered(path, denied):
    return any(path == Path(d) or Path(d) in path.parents for d in denied)


def test_deny_list_covers_secrets_without_nesting(env, monkeypatch):
    from overseer.runners import Runner

    env.cfg.local.env_path.write_text("TELEGRAM_BOT_TOKEN=x\n")
    denied = Runner(env.cfg).deny_read()
    assert covered(env.cfg.ssh_dir, denied) and covered(env.cfg.local.state(), denied)
    assert str(env.cfg.local.env_path) in denied
    assert not any(Path(a) in Path(b).parents for a in denied for b in denied)  # no nested entries
    # With the key folder outside the state dir it is listed on its own.
    env.cfg.ssh.dir = str(env.tmp / "app" / "keys")
    (env.tmp / "app" / "keys").mkdir()
    assert str(env.cfg.ssh_dir) in Runner(env.cfg).deny_read()


def test_deny_list_uses_real_paths(env):
    """A socket reachable as /var/run/x and /run/x (/var/run -> /run) is listed once, by its real
    path: bubblewrap can't mask a path through a symlinked dir and the agent's shell wouldn't start."""
    import socket
    from overseer.runners import Runner

    run = env.tmp / "run"
    run.mkdir()
    (env.tmp / "var-run").symlink_to(run)
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(run / "d.sock"))
    try:
        env.cfg.agent.deny_read = [str(env.tmp / "var-run" / "d.sock"), str(run / "d.sock")]
        denied = Runner(env.cfg).deny_read()
        assert str(run / "d.sock") in denied and not any("var-run" in d for d in denied)
    finally:
        sock.close()


async def test_claude_runner_sandbox_and_parsing(env):
    r = ClaudeCodeRunner(env.cfg)
    act = Activity()
    res = await r.run("fix job 1", system="SYS", session_id=None, key="s1", activity=act)
    assert res.text == "Fixed the batch size and resubmitted as job 4242." and res.session_id
    assert act.jobs == [("4242", "ssh hpc 'cd ~/proj && sbatch train.sh'")] and act.files == ["/tmp/x/notes.md"]

    call = env.cli_log()[-1]
    argv, e = call["argv"], call["env"]
    settings = json.loads(argv[argv.index("--settings") + 1])
    sb = settings["sandbox"]
    assert sb["enabled"] and sb["failIfUnavailable"] and sb["allowUnsandboxedCommands"] is False
    assert sb["excludedCommands"] == []
    workdir = str(env.cfg.local.tmp("agents", "s1"))
    assert sb["filesystem"]["allowWrite"] == [workdir] and call["cwd"] == workdir
    denied = sb["filesystem"]["denyRead"]
    assert covered(env.cfg.ssh_dir, denied) and str(env.cfg.local.state()) in denied
    assert sb["network"]["allowedDomains"] == [] and sb["network"]["allowAllUnixSockets"] is True
    assert f"Read(/{env.cfg.local.state()}/**)" in settings["permissions"]["deny"]
    assert argv[argv.index("--append-system-prompt") + 1] == "SYS" and call["stdin"] == "fix job 1"
    # Scrubbed environment: no Telegram token, PATH starts with the ssh wrappers, own config dir.
    assert "TELEGRAM_BOT_TOKEN" not in e and e["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert e["PATH"].split(":")[0] == str(env.cfg.local.tmp("agent-bin"))
    assert e["CLAUDE_CONFIG_DIR"] == str(env.cfg.local.resolve(env.cfg.agent.claude_code.home))
    assert e["HOME"].startswith(workdir)

    # extra_settings can't loosen the sandbox.
    env.cfg.agent.claude_code.extra_settings = {"sandbox": {"allowUnsandboxedCommands": True, "excludedCommands": ["ssh *"]},
                                                "model": "sonnet"}
    s2 = r.settings(env.cfg.local.tmp("agents", "s1"))
    assert s2["sandbox"]["allowUnsandboxedCommands"] is False and s2["sandbox"]["excludedCommands"] == []
    assert s2["model"] == "sonnet"

    res2 = await r.run("again", system="SYS", session_id=res.session_id, key="s1", activity=Activity())
    argv2 = env.cli_log()[-1]["argv"]
    assert argv2[argv2.index("--resume") + 1] == res.session_id and res2.session_id == res.session_id
    assert await r.text(system="S", prompt="P") == "OK report"


async def test_codex_runner_profile_and_parsing(env):
    r = CodexRunner(env.cfg)
    act = Activity()
    res = await r.run("fix job 1", system="SYS", session_id=None, key="s2", activity=act)
    assert res.text == "Resubmitted as job 4343." and res.session_id
    assert act.commands == ["ssh hpc sbatch --parsable ~/proj/train.sh"]
    assert [j for j, _ in act.jobs] == ["4343"] and act.files == ["/tmp/x/a.py"]

    call = env.cli_log()[-1]
    argv = call["argv"]
    overrides = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
    assert 'default_permissions="overseer"' in overrides and 'approval_policy="never"' in overrides
    fs = next(o for o in overrides if o.startswith("permissions.overseer.filesystem="))
    assert f'"{env.cfg.local.state()}" = "deny"' in fs and '"/" = "read"' in fs  # covers the ssh key folder
    assert f'"{env.cfg.local.tmp("agents", "s2")}" = "write"' in fs
    net = next(o for o in overrides if o.startswith("permissions.overseer.network="))
    assert f'"{env.cfg.control_socket}" = "allow"' in net and '"enabled" = true' in net
    assert not any(o.startswith("sandbox_mode") for o in overrides)
    assert any(o.startswith("developer_instructions=") for o in overrides)
    assert argv[-1] == "-" and call["stdin"] == "fix job 1"
    assert "TELEGRAM_BOT_TOKEN" not in call["env"] and call["env"]["CODEX_HOME"].endswith("state/codex")

    await r.run("again", system="SYS", session_id=res.session_id, key="s2", activity=Activity())
    argv2 = env.cli_log()[-1]["argv"]
    assert argv2[:2] == ["exec", "resume"] and argv2[-2] == res.session_id
    assert await r.text(system="S", prompt="P") == "OK report"
    one = env.cli_log()[-1]["argv"]
    net1 = next(one[i + 1] for i, a in enumerate(one) if a == "-c" and one[i + 1].startswith("permissions.overseer.network="))
    assert '"enabled" = false' in net1 and "--ephemeral" in one


def test_login_commands(env):
    argv, e = CodexRunner(env.cfg).login_command([])
    assert argv[1:] == ["login", "--device-auth"]  # headless-friendly default
    assert e["CODEX_HOME"].endswith("state/codex") and "TELEGRAM_BOT_TOKEN" not in e
    argv, _ = CodexRunner(env.cfg).login_command(["--with-api-key"])
    assert argv[1:] == ["login", "--with-api-key"]
    argv, e = ClaudeCodeRunner(env.cfg).login_command([])
    assert argv[1:] == ["auth", "login"] and e["CLAUDE_CONFIG_DIR"].endswith("state/claude")
    argv, _ = ClaudeCodeRunner(env.cfg).login_command(["setup-token"])
    assert argv[1:] == ["setup-token"]


async def test_runner_errors_are_reported(env):
    from overseer.runners import AgentRunError

    with pytest.raises(AgentRunError, match="boom"):
        await ClaudeCodeRunner(env.cfg).run("FORCE_FAIL", system="", session_id=None, key="s", activity=Activity())
    env.cfg.agent.codex.command = "definitely-not-installed-codex"
    with pytest.raises(AgentRunError, match="not found"):
        await CodexRunner(env.cfg).run("x", system="", session_id=None, key="s", activity=Activity())


# ----------------------------------------------------------------------------- agent conversations

@pytest.mark.parametrize("backend,job", [("claude_code", "4242"), ("codex", "4343")])
async def test_agent_turn_reports_and_tracks_jobs(env, backend, job):
    env.agent.set_backend(backend)
    s = env.agent.general_session()
    await env.agent.handle(s, "please fix my training run")
    texts = env.notifier.texts()
    assert f"Agent submitted job {job}" in texts
    assert "command(s)" in texts and "sbatch" in texts  # command summary
    assert "🤖 " in env.notifier.sent[-1].text
    rec = env.store.get_job(job)
    assert rec.by_agent == 1 and rec.session_id == s.id
    assert env.agent.session_for_job(job) is s  # replies to the job's reports continue this conversation
    first = env.store.get_session(s.id)
    assert first.backend == backend and first.cli_session_id

    await env.agent.handle(s, "thanks")
    assert env.store.get_session(s.id).cli_session_id == first.cli_session_id  # resumed

    # Switching backend starts a new CLI session but keeps the Overseer conversation.
    other = "codex" if backend == "claude_code" else "claude_code"
    env.agent.set_backend(other)
    await env.agent.handle(s, "continue")
    after = env.store.get_session(s.id)
    assert after.backend == other and after.cli_session_id != first.cli_session_id


async def test_model_effort_and_footer(env):
    s = env.agent.general_session()
    await env.agent.handle(s, "hi")
    reply = env.notifier.sent[-1].text
    assert "<i>claude_code · claude-fake-default · effort default · 1.6k tokens (400 cached) · $0.03</i>" in reply
    assert "model CLI default (last run: claude-fake-default) · effort CLI default" in env.agent.describe("claude_code")

    env.agent.set_model("opus")
    env.agent.set_effort("xhigh")
    await env.agent.handle(s, "again")
    argv = env.cli_log()[-1]["argv"]
    assert argv[argv.index("--model") + 1] == "opus" and argv[argv.index("--effort") + 1] == "xhigh"
    assert "claude_code · opus · effort xhigh" in env.notifier.sent[-1].text
    assert "opus (set via /model)" in env.agent.describe("claude_code")
    with pytest.raises(ValueError, match="must be one of"):
        env.agent.set_effort("minimal")  # a codex level, not a Claude Code one
    with pytest.raises(ValueError):
        env.agent.set_model("two words")

    # Overrides are per backend.
    env.agent.set_backend("codex")
    env.agent.set_effort("minimal")
    env.agent.set_model("gpt-x")
    await env.agent.handle(s, "go")
    argv = env.cli_log()[-1]["argv"]
    assert argv[argv.index("-m") + 1] == "gpt-x" and 'model_reasoning_effort="minimal"' in argv
    assert "<i>codex · gpt-x · effort minimal · 5.7k tokens (1.0k cached)</i>" in env.notifier.sent[-1].text  # no cost

    # codex reports the conversation's running total; the footer shows this request's share.
    await env.agent.handle(s, "go on")
    assert "5.7k tokens (1.0k cached)</i>" in env.notifier.sent[-1].text
    env.store.unset(f"usage:codex:{env.store.get_session(s.id).cli_session_id}")  # total from an older version
    await env.agent.handle(s, "and on")
    assert "17.1k tokens (3.0k cached) in this conversation so far" in env.notifier.sent[-1].text

    env.agent.set_model("default")
    env.agent.set_effort("default")
    await env.agent.handle(s, "go")
    argv = env.cli_log()[-1]["argv"]
    assert "-m" not in argv and not any(a.startswith("model_reasoning_effort") for a in argv)
    # Nothing configured: the model and effort codex actually used come from its session log
    # and its model catalog.
    assert "codex · gpt-fake-default · effort medium (model default)" in env.notifier.sent[-1].text
    assert env.agent.describe("codex") == ("codex: model CLI default (last run: gpt-fake-default) · "
                                           "effort CLI default (last run: medium (model default))")
    await env.agent.handle(s, "go")  # the catalog is looked up once
    assert sum(c["argv"][:2] == ["debug", "models"] for c in env.cli_log(lookups=True)) == 1
    env.cfg.agent.codex.model = "from-config"
    assert "model from-config (last run: gpt-fake-default) ·" in env.agent.describe("codex")
    assert env.agent.runner("claude_code").model == "opus"  # untouched


def test_sessions_and_open_commands(env, monkeypatch, capsys):
    from overseer.__main__ import open_command, sessions_command

    async def no_master(_cfg):
        return None

    monkeypatch.setattr("overseer.sshlink.ensure_master", no_master)
    assert open_command(env.cfg, None) == 1  # nothing yet

    claude_s = env.agent.general_session()
    asyncio.run(env.agent.handle(claude_s, "hi"))
    env.agent.set_backend("codex")
    codex_s = env.agent.general_session(reset=True)
    asyncio.run(env.agent.handle(codex_s, "hi"))

    capsys.readouterr()
    sessions_command(env.cfg)
    listing = capsys.readouterr().out
    assert claude_s.id in listing and codex_s.id in listing and "claude_code" in listing and "codex" in listing

    # Latest = the codex conversation, resumed in codex's interactive UI with Overseer's settings.
    monkeypatch.setenv("TERM", "xterm-kitty")
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert open_command(env.cfg, None) == 0
    call = env.cli_log()[-1]
    # The user's terminal, not the bot's TERM=dumb (codex refuses to start its TUI with that).
    assert call["env"]["TERM"] == "xterm-kitty" and call["env"]["COLORTERM"] == "truecolor"
    assert "NO_COLOR" not in call["env"]
    codex_id = env.store.get_session(codex_s.id).cli_session_id
    assert call["prog"] == "codex" and call["argv"][0] == "resume" and call["argv"][-1] == codex_id
    assert 'default_permissions="overseer"' in call["argv"] and "exec" not in call["argv"]
    assert call["env"]["CODEX_HOME"].endswith("state/codex")
    assert call["env"]["PATH"].split(":")[0] == str(env.cfg.local.tmp("agent-bin"))  # `ssh hpc` works
    assert "TELEGRAM_BOT_TOKEN" not in call["env"]

    # By name (prefix works too): Claude Code, started in the folder its session ran in.
    assert open_command(env.cfg, "chat-") == 1  # ambiguous prefix
    assert open_command(env.cfg, claude_s.id) == 0
    call = env.cli_log()[-1]
    claude_id = env.store.get_session(claude_s.id).cli_session_id
    assert call["prog"] == "claude" and call["argv"][:2] == ["--resume", claude_id] and "-p" not in call["argv"]
    assert call["cwd"] == str(env.cfg.local.tmp("agents", claude_s.id))
    settings = json.loads(call["argv"][call["argv"].index("--settings") + 1])
    assert settings["sandbox"]["failIfUnavailable"] and settings["sandbox"]["allowUnsandboxedCommands"] is False
    claude_json = json.loads((env.cfg.local.resolve(env.cfg.agent.claude_code.home) / ".claude.json").read_text())
    assert claude_json["projects"][call["cwd"]]["hasTrustDialogAccepted"] is True  # no "trust this folder?" stop
    assert open_command(env.cfg, "nope") == 1


async def test_job_context_and_report_modes(env):
    env.set_jobs(**{"77": {"name": "train", "state": "FAILED", "exit": "1:0", "workdir": str(env.home / "proj"),
                           "stdout": str(env.home / "proj" / "slurm-77.out"), "script": "#!/bin/bash\npython t.py\n"}})
    (env.home / "proj" / "slurm-77.out").write_text("CUDA out of memory\n")
    await env.monitor.poll(first=True)
    await env.monitor.poll()  # finished jobs are picked up from sacct after the first poll
    s = env.agent.session_for_job("77")
    await env.agent.handle(s, "why?")
    prompt = env.cli_log()[-1]["stdin"]
    assert "<job id='77'" in prompt and "CUDA out of memory" in prompt and "python t.py" in prompt
    await env.agent.handle(s, "and now?")
    assert "<job id='77'" not in env.cli_log()[-1]["stdin"]  # resumed sessions already have it

    env.cfg.agent.report_commands = "off"
    n = len(env.notifier.sent)
    await env.agent.handle(env.agent.general_session(reset=True), "hi")
    assert not any("command(s)" in m.text for m in env.notifier.sent[n:])

    env.cfg.agent.report_commands = "live"
    n = len(env.notifier.sent)
    await env.agent.handle(env.agent.general_session(reset=True), "hi")
    assert any(m.text.startswith("💻") and "sbatch" in m.text for m in env.notifier.sent[n:])


async def test_stop_kills_the_run(env):
    s = env.agent.general_session()
    task = asyncio.create_task(env.agent.handle(s, "FORCE_SLEEP please"))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if env.agent.is_running(s) and env.cli_log():
            break
    assert await env.agent.stop(s.id)
    await asyncio.wait_for(task, 10)
    assert "Stopped" in env.notifier.sent[-1].text and not env.agent.is_running(s)
    last = (env.cfg.local.state("transcripts") / f"{s.id}.jsonl").read_text().splitlines()[-1]
    assert json.loads(last)["status"] == "stopped"


@pytest.mark.parametrize("backend", ["claude_code", "codex"])
async def test_live_progress_and_transcript(env, monkeypatch, backend):
    from overseer import live
    monkeypatch.setattr(live, "PROGRESS_SECONDS", 0.1)
    env.agent.set_backend(backend)
    s = env.agent.general_session(reset=True)
    task = asyncio.create_task(env.agent.handle(s, "FORCE_SLOW fix it"))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if env.store.get_session(s.id).cli_session_id:
            break
    # The CLI's session id is saved while the run is still going (so `overseer open` finds it).
    assert env.agent.is_running(s) and env.store.get_session(s.id).cli_session_id
    await asyncio.wait_for(task, 20)

    # One message that updated itself while the agent worked, removed once the answer arrived.
    (handle, versions), = env.notifier.live.items()
    assert handle in env.notifier.deleted and len(versions) > 1
    seen = "\n".join(versions)
    assert f"{backend} is working" in seen and "/stop cancels" in seen
    assert "💭 <i>The batch size is probably too big for the GPU.</i>" in seen and "💬 Checking the job." in seen
    assert "running for" in seen and "sbatch" in seen
    assert env.notifier.sent[-1].text.startswith("🤖")

    events = [json.loads(line) for line in
              (env.cfg.local.state("transcripts") / f"{s.id}.jsonl").read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "turn" and events[0]["text"] == "FORCE_SLOW fix it" and events[0]["backend"] == backend
    assert {"session", "thought", "message", "command", "output"} <= set(kinds)
    assert kinds.index("thought") < kinds.index("command") < kinds.index("output")
    assert kinds[-1] == "end" and events[-1]["status"] == "done" and events[-1]["footer"].startswith(backend)
    assert oct((env.cfg.local.state("transcripts") / f"{s.id}.jsonl").stat().st_mode & 0o777) == "0o600"


def test_watch(env):
    import io
    from overseer.live import Transcript, watch

    s = env.agent.general_session()
    asyncio.run(env.agent.handle(s, "hi"))

    def run(conversation, **kw):
        out = io.StringIO()
        rc = watch(env.cfg, conversation, follow=False, color=False, out=out, **kw)
        return rc, out.getvalue()

    rc, text = run(s.id)
    assert rc == 0
    for part in ["new request · claude_code", "👤 hi", "💭 The batch size is probably too big", "💬 Checking the job.",
                 "$ ssh hpc 'cd ~/proj && sbatch train.sh'", "  │ Submitted batch job 4242",
                 "✏️  /tmp/x/notes.md", "🤖 Fixed the batch size", "claude_code · claude-fake-default", "· done"]:
        assert part in text, part
    assert run("nope")[0] == 1
    t = Transcript(env.cfg, "chat-answer")
    for ev in [{"kind": "turn", "text": "q"}, {"kind": "message", "text": "The answer."},
               {"kind": "end", "status": "done", "text": "The answer.", "footer": "codex · m"}]:
        t.write(ev)
    assert run("chat-answer")[1].count("The answer.") == 1  # the final message isn't printed twice

    # All conversations: only requests that are still running are replayed, then new events.
    rc, text = run(None)
    assert "Working on: nothing right now" in text and "👤" not in text
    t = Transcript(env.cfg, "j77-abcdef")
    t.write({"kind": "turn", "backend": "codex", "text": "still going"})
    t.write({"kind": "command", "command": "ls"})
    t.write({"kind": "output", "command": "ls", "output": "\n".join(f"line{i}" for i in range(30)), "exit_code": 2})
    rc, text = run(None)
    assert "Working on: j77-abcdef" in text and "▌ j77-abcdef" in text and "👤 still going" in text
    assert "$ ls" in text and "line29" in text and "line5\n" not in text and "18 earlier lines" in text
    assert "└ exit 2" in text
    assert "line5\n" in run("j77", full=True)[1]  # a unique prefix is enough


async def test_cli_refusals_are_reported_plainly(env):
    settings = ClaudeCodeRunner(env.cfg).settings(env.cfg.local.tmp("agents", "x"))
    assert "Bash" in settings["permissions"]["allow"]  # sandboxed Bash never waits for an approval
    await env.agent.handle(env.agent.general_session(), "FORCE_BLOCK please")
    notice = next(m.text for m in env.notifier.sent if "refused by" in m.text)
    assert "claude_code itself" in notice and "requires approval" in notice and "sbatch train.sh" in notice
    assert "approval" in env.agent.system_prompt() and "Never ask them to approve" in env.agent.system_prompt()


async def test_agent_failure_is_reported(env):
    await env.agent.handle(env.agent.general_session(), "FORCE_FAIL now")
    assert "boom" in env.notifier.sent[-1].text


# ----------------------------------------------------------------------------- monitor

async def test_slurm_found_via_login_shell_and_setup(env, monkeypatch):
    """Clusters often only put SLURM on PATH in login shells (or after `module load slurm`)."""
    plain_path = ":".join(p for p in os.environ["PATH"].split(":") if p != str(FAKE_SLURM))
    monkeypatch.setenv("PATH", plain_path)
    with pytest.raises(Exception, match="not found"):
        await _squeue_without_path(env)

    hpc = HPC(LocalRunner(env.cfg), setup=f"export PATH={FAKE_SLURM}:$PATH")
    await hpc.connect()
    assert hpc.path.startswith(str(FAKE_SLURM))
    env.set_jobs(**{"321": {"name": "x", "state": "RUNNING"}})
    assert [j.job_id for j in await hpc.squeue()] == ["321"]


async def _squeue_without_path(env):
    hpc = HPC(LocalRunner(env.cfg))
    hpc.user = "me"  # skip discovery: plain non-login PATH
    await hpc.squeue()


def test_agent_prompt_mentions_login_shell(env):
    assert "bash -lc" in env.agent.system_prompt()
    env.cfg.slurm.setup = "module load slurm"
    assert 'bash -lc "module load slurm; squeue' in env.agent.system_prompt()


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
    assert "FAILED" in final.text and "OK report" in final.text  # LLM analysis via the agent backend
    assert ("🔧 Ask agent to fix", "fix:77") in final.buttons[0]
    rec = env.store.get_job("77")
    assert rec.ended_at and rec.batch_script.startswith("#!/bin/bash")

    env.set_jobs(**{"78": {"name": "quick", "state": "COMPLETED", "workdir": proj, "stdout": str(log)}})
    await env.monitor.poll()
    assert "Job 78" in env.notifier.sent[-1].text and "finished" in env.notifier.sent[-1].text


async def test_monitor_fills_in_agent_submitted_jobs(env):
    s = env.agent.general_session()
    await env.agent.handle(s, "submit it")  # fake claude "submits" 4242
    proj = str(env.home / "proj")
    env.set_jobs(**{"4242": {"name": "fixed", "state": "RUNNING", "workdir": proj, "stdout": f"{proj}/slurm-4242.out"}})
    await env.monitor.poll()
    started = env.notifier.sent[-1]
    assert "Job 4242" in started.text and "started" in started.text and "submitted by the agent" in started.text
    rec = env.store.get_job("4242")
    assert rec.stdout == f"{proj}/slurm-4242.out" and rec.name == "fixed" and rec.session_id == s.id


async def test_monitor_track_agent_only(env):
    env.cfg.slurm.track = "agent"
    env.set_jobs(**{"90": {"name": "mine", "state": "RUNNING", "workdir": "/x"}})
    await env.monitor.poll()
    assert env.store.get_job("90") is None


# ----------------------------------------------------------------------------- bot

async def test_bot_routing_and_commands(env, monkeypatch):
    from overseer.bot import Bot

    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=env.notifier)
    started = []

    async def fake_run_agent(chat_id, session, text):
        started.append((session.id, text))

    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)
    replies = []

    async def reply(update, text):
        replies.append(text)

    monkeypatch.setattr(bot, "reply", reply)

    async def drain():
        for t in list(bot._tasks):
            await t

    env.set_jobs(**{"55": {"name": "t", "state": "FAILED", "workdir": str(env.home / "proj")}})
    await env.monitor.poll(first=True)
    env.store.link_message(1, 999, "55", None)
    await bot.on_text(NS(effective_message=NS(text="use lr 1e-4", reply_to_message=NS(message_id=999)),
                         effective_chat=NS(id=1)), None)
    await drain()
    assert started[-1] == (env.agent.session_for_job("55").id, "use lr 1e-4")

    await bot.cmd_model(None, NS(args=["sonnet"]))
    assert "claude_code: model sonnet (set via /model)" in replies[-1]
    await bot.cmd_effort(None, NS(args=["turbo"]))
    assert "must be one of" in replies[-1]
    await bot.cmd_effort(None, NS(args=["high"]))
    assert "effort high (set via /effort)" in replies[-1]
    await bot.cmd_backend(None, NS(args=[]))
    assert "▶️ claude_code: model sonnet" in replies[-1] and "▫️ codex: model CLI default" in replies[-1]
    await bot.cmd_backend(None, NS(args=["codex"]))
    assert env.agent.backend == "codex" and "codex" in replies[-1]
    await bot.cmd_effort(None, NS(args=[]))
    assert "minimal | low | medium | high | xhigh" in replies[-1]
    await bot.cmd_backend(None, NS(args=["gpt"]))
    assert "must be one of" in replies[-1] and env.agent.backend == "codex"

    answers = []

    async def answer(*a, **k):
        answers.append(k)

    q = NS(data="fix:55", answer=answer, message=NS(message_id=5))
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=666)), None)
    assert answers[-1].get("show_alert")
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1)), None)
    await drain()
    assert started[-1][0] == env.agent.session_for_job("55").id and "Job 55 failed" in started[-1][1]


def test_build_app(env):
    from overseer.__main__ import build

    app = build(env.cfg)
    assert app.handlers
    assert (env.cfg.local.state().stat().st_mode & 0o777) == 0o700


# ----------------------------------------------------------------------------- autopilot

async def make_autopilot(env):
    """The autopilot, and the ov socket the (fake) agent uses to watch its jobs."""
    from overseer.autopilot import Autopilot
    from overseer.ctl import ControlServer
    from overseer.projects import Projects
    ap = Autopilot(env.cfg, env.store, env.agent, env.notifier, env.hpc)
    env.monitor.autopilot = ap
    env.ctl = ControlServer(env.cfg, env.agent, Projects(env.cfg, env.store, env.agent), env.notifier, env.hpc)
    await env.ctl.start()
    return ap


def agent_jobs(env, s):
    return [j.job_id for j in env.store.session_jobs(s.id)]


def watched(env, s):
    return [(w.kind, w.jobs, w.oneoff) for w in env.store.watches(s.id)]


async def run_job(env, jid, state="COMPLETED", log_text="step 100 loss=0.50\n", finish=True):
    proj = env.home / "proj"
    log = proj / f"slurm-{jid}.out"
    log.write_text(log_text)
    job = {"name": f"run{jid}", "state": "RUNNING", "workdir": str(proj), "stdout": str(log)}
    env.set_jobs(**{jid: job})
    await env.monitor.poll()
    if finish:
        env.set_jobs(**{jid: {**job, "state": state, "exit": "0:0" if state == "COMPLETED" else "1:0"}})
        await env.monitor.poll()
    await env.monitor.autopilot.drain()


def claude_calls(env):
    """The agent's turns (not one-shot job summaries)."""
    return [c for c in env.cli_log() if c["prog"] == "claude" and "--no-session-persistence" not in c["argv"]]


def system_of(call):
    return call["argv"][call["argv"].index("--append-system-prompt") + 1]


async def test_working_alone_over_many_runs(env):
    """No approval, no mode: the agent watches its jobs, is woken with their results, and carries on."""
    await make_autopilot(env)
    s = env.agent.general_session()
    await env.agent.handle(s, "FAKE_LOOP find the best lr")
    call = claude_calls(env)[-1]
    system = system_of(call)
    assert "## Working over time: watches and wake-ups" in system and "at least 15m apart" in system
    assert "Autopilot is ON" not in system and "ov auto" not in system
    (j1,) = agent_jobs(env, s)
    assert watched(env, s) == [("jobs", j1, 0)]
    note = next(m for m in env.notifier.sent if f"will look at job {j1} when it ends" in m.text)
    assert note.silent and note.buttons[0][0][0] == "✖ Cancel"
    assert f"⏰ Waiting for: job {j1}" in env.notifier.sent[-1].text

    # Job 1 finishes: the conversation is woken with its report; no one-shot LLM summary.
    await run_job(env, j1)
    final = next(m.text for m in env.notifier.sent if f"Job {j1}" in m.text and "finished" in m.text)
    assert "OK report" not in final
    assert any(f"Waking the agent</b>: {j1} COMPLETED" in m.text for m in env.notifier.sent)
    call = claude_calls(env)[-1]
    assert "[Wake-up you set up]" in call["stdin"] and f"<job id='{j1}'" in call["stdin"]
    assert "loss=0.50" in call["stdin"] and "final state COMPLETED" in call["stdin"]
    assert call["argv"][call["argv"].index("--resume") + 1] == env.store.get_session(s.id).cli_session_id
    j2 = agent_jobs(env, s)[-1]
    assert j2 != j1 and watched(env, s) == [("jobs", j2, 0)]
    assert system_of(call) == system  # stays cacheable

    # Job 2's results reach the goal: the agent stops and sets nothing up.
    n = len(claude_calls(env))
    await run_job(env, j2, log_text="score 9.1 FAKE_DONE\n")
    assert len(claude_calls(env)) == n + 1
    reply = env.notifier.sent[-1].text
    assert "Reached the goal" in reply and "Waiting for" not in reply and not watched(env, s)


async def test_batches_and_early_wakeup_on_failure(env):
    await make_autopilot(env)
    s = env.agent.general_session()
    await env.agent.handle(s, "FAKE_LOOP sweep FAKE_TWO")
    j1, j2 = agent_jobs(env, s)
    assert watched(env, s) == [("jobs", f"{j1} {j2}", 0)]  # one batch
    n = len(claude_calls(env))
    await run_job(env, j1)  # the first finishes: wait for the other one
    assert len(claude_calls(env)) == n and f"waking the agent when {j2} has finished too" in env.notifier.sent[-1].text
    await run_job(env, j2)
    stdin = claude_calls(env)[-1]["stdin"]
    assert f"<job id='{j1}'" in stdin and f"<job id='{j2}'" in stdin

    # A batch where one fails while the other runs: woken at once; the other stays watched.
    j3 = agent_jobs(env, s)[-1]
    env.agent.autopilot.unwatch(s.id, "all")
    await env.agent.handle(s, "FAKE_LOOP again FAKE_TWO")
    j4, j5 = agent_jobs(env, s)[-2:]
    await run_job(env, j5, finish=False)
    await run_job(env, j4, state="FAILED", log_text="CUDA out of memory\n")
    stdin = claude_calls(env)[-1]["stdin"]
    assert f"<job id='{j4}' name='run{j4}' state='FAILED'" in stdin and f"Still running: {j5}" in stdin
    assert ("jobs", j5, 0) in watched(env, s) and j3


async def test_asking_and_forgetting_to_watch(env):
    await make_autopilot(env)
    s = env.agent.general_session()
    await env.agent.handle(s, "FAKE_LOOP try things FAKE_NOWATCH")
    (j1,) = agent_jobs(env, s)
    assert not watched(env, s) and "Waiting for" not in env.notifier.sent[-1].text  # nothing set up: nothing wakes it
    n = len(claude_calls(env))
    await run_job(env, j1)
    assert len(claude_calls(env)) == n  # reported to you, but nobody is woken

    await env.agent.handle(s, "FAKE_LOOP and now FAKE_HELP")
    reply = env.notifier.sent[-1]
    assert "should I try a bigger model?" in reply.text and not reply.silent and not watched(env, s)


async def test_checkins(env):
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    await env.agent.handle(s, "FAKE_LOOP train FAKE_EVERY")
    (j1,) = agent_jobs(env, s)
    (w,) = env.store.watches(s.id)
    assert w.every == 3600
    await run_job(env, j1, finish=False)
    assert env.store.get_job(j1).last_checkin  # the clock starts when the job runs
    n = len(claude_calls(env))
    await env.monitor.poll()
    await ap.drain()
    assert len(claude_calls(env)) == n  # not due yet

    def make_due():
        rec = env.store.get_job(j1)
        rec.last_checkin -= 3700
        env.store.upsert_job(rec)

    for k in range(1, 4):  # no limit on how many
        make_due()
        await env.monitor.poll()
        await ap.drain()
        assert env.store.get_job(j1).checkins == k
    stdin = claude_calls(env)[-1]["stdin"]
    assert "[Check-in you asked for" in stdin and f"<job id='{j1}'" in stdin and "loss=0.50" in stdin
    reply = env.notifier.sent[-1]
    assert "Looks healthy." in reply.text and reply.silent  # a routine check-in doesn't buzz

    # The run diverges: the agent cancels it and resubmits; the cancelled job's end wakes nobody.
    (env.home / "proj" / f"slurm-{j1}.out").write_text("loss=nan FAKE_DIVERGE\n")
    make_due()
    await env.monitor.poll()
    await ap.drain()
    assert f"Job {j1} diverged: cancelled it and resubmitted" in env.notifier.sent[-1].text
    j2 = agent_jobs(env, s)[-1]
    assert j2 != j1 and [x[1] for x in watched(env, s)] == [j2]
    n = len(claude_calls(env))
    env.set_jobs(**{j1: {"name": f"run{j1}", "state": "CANCELLED", "exit": "0:15", "workdir": str(env.home / "proj"),
                         "stdout": str(env.home / "proj" / f"slurm-{j1}.out")}})
    await env.monitor.poll()
    await ap.drain()
    assert len(claude_calls(env)) == n


async def test_timers_and_their_limits(env):
    await make_autopilot(env)
    s = env.agent.general_session()
    token = env.ctl.open(s.id)
    ok, text = await env.ctl.handle(token, ["watch", "123456", "--every", "5m"])
    assert not ok  # (no such job) - but the interval rule is checked on real jobs below
    env.set_jobs(**{"7001": {"name": "x", "state": "RUNNING"}})
    ok, text = await env.ctl.handle(token, ["watch", "7001", "--every", "5m"])
    assert ok and "check-ins are at least 15m apart" in text and env.store.watches(s.id)[0].every == 900
    ok, text = await env.ctl.handle(token, ["wake", "every", "5m", "look"])
    assert ok and "every 15m for 1d" in text and "at least 15m apart" in text
    ok, text = await env.ctl.handle(token, ["wake", "every", "2h", "--for", "6h", "summarise"])
    assert ok and "every 2h for 6h" in text
    timer = env.store.watches(s.id)[-1]
    assert abs(timer.until - time.time() - 6 * 3600) < 60

    # A repeating timer fires, stays, and stops at its end.
    timer.due = time.time() - 1
    env.store.update_watch(timer)
    await env.monitor.poll()
    await env.monitor.autopilot.drain()
    assert "Your timer" in claude_calls(env)[-1]["stdin"] and "summarise" in claude_calls(env)[-1]["stdin"]
    assert env.store.get_watch(timer.id).state == "active"
    timer = env.store.get_watch(timer.id)
    timer.due, timer.until = time.time() - 1, time.time() + 60
    env.store.update_watch(timer)
    await env.monitor.poll()
    await env.monitor.autopilot.drain()
    assert env.store.get_watch(timer.id).state == "fired"  # the next one would be after its end

    ok, text = await env.ctl.handle(token, ["auto", "request", "--goal", "x"])  # what older conversations try
    assert not ok and "no longer needs approval" in text


async def test_failed_wakeup_pauses_until_you_write(env):
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    await env.agent.handle(s, "FAKE_LOOP train")
    (j1,) = agent_jobs(env, s)
    fail = Path(os.environ["FAKE_CLI_LOG"] + ".fail")
    fail.touch()  # the CLI crashes during the wake-up
    await run_job(env, j1)
    assert "Wake-ups paused" in env.notifier.sent[-1].text and ap.paused(s.id)
    assert watched(env, s) == [("jobs", j1, 0)]  # the watch is back
    n = len(claude_calls(env))
    await env.monitor.poll()
    await ap.drain()
    assert len(claude_calls(env)) == n  # paused

    fail.unlink()
    await env.agent.handle(s, "the GPU is back, go on")  # writing resumes them
    await ap.drain()
    assert not ap.paused(s.id) and "[Wake-up you set up]" in claude_calls(env)[-1]["stdin"]


async def test_auto_command(env, monkeypatch):
    from overseer.bot import Bot
    await make_autopilot(env)
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=env.notifier)
    replies = []

    async def reply(update, text):
        replies.append(text)

    monkeypatch.setattr(bot, "reply", reply)
    await bot.cmd_auto(None, NS(args=[]))
    assert "Nothing will wake the agent" in replies[-1]

    await bot.cmd_auto(None, NS(args=["tune", "the", "lr", "FAKE_LOOP"]))  # a goal: the agent is asked
    for t in list(bot._tasks):
        await t
    s = env.agent.general_session()
    stdin = claude_calls(env)[-1]["stdin"]
    assert "Work on this on your own, over as many runs as it takes: tune the lr" in stdin
    (j1,) = agent_jobs(env, s)
    await bot.cmd_auto(None, NS(args=[]))
    assert f"w1 job {j1}" in replies[-1] and s.id in replies[-1]
    await bot.cmd_auto(None, NS(args=["off"]))
    assert "Cancelled 1 pending wake-up" in replies[-1] and not watched(env, s)
    await env.agent.handle(s, "hi")
    assert "cancelled all your pending wake-ups" in claude_calls(env)[-1]["stdin"]


class FakeQuery:
    """A pressed inline button: records how the message was edited."""
    def __init__(self, data=""):
        self.edits = []
        self.data = data
        self.message = None

    async def answer(self, *a, **kw):
        pass

    async def edit_message_text(self, text, parse_mode=None, reply_markup=None):
        rows = [[(b.text, b.callback_data) for b in row] for row in reply_markup.inline_keyboard] if reply_markup else []
        self.edits.append((text, rows))


async def test_cancel_button_and_old_cards(env):
    from overseer.bot import Bot
    await make_autopilot(env)
    s = env.agent.general_session()
    env.set_jobs(**{"5001": {"name": "train", "state": "RUNNING", "workdir": str(env.home / "proj")}})
    await env.agent.handle(s, "FAKE_OV ov watch 5001 --note 'compare the loss'\nFAKE_OV ov wake in 30m look at the eval")
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=env.notifier)
    w = env.store.watches(s.id)[0]
    q = FakeQuery(f"uw:{w.id}")
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1)), None)
    assert "Cancelled" in q.edits[-1][0] and env.store.get_watch(w.id).state == "cancelled"
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1)), None)
    assert "already happened or was cancelled" in q.edits[-1][0]
    await env.agent.handle(s, "hi")
    assert f"cancelled your wake-up w{w.id}" in claude_calls(env)[-1]["stdin"]
    q = FakeQuery(f"ap:ok:{s.id}")  # an approval card from before
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1)), None)
    assert "no longer used" in q.edits[-1][0]


async def test_research_log(env):
    """In a project, the research log's beginning and latest entries reach the agent when it starts and wakes."""
    await make_autopilot(env)
    projects = make_projects(env)
    pdir = env.home / "dreamer"
    pdir.mkdir()
    projects.create("dreamer", str(pdir))
    s = env.agent.new_conversation("dreamer")
    system = env.agent.system_prompt(s.id)
    assert f"Research log: `{pdir}/RESEARCH_LOG.md`" in system and "well under 200k tokens" in system
    await env.agent.handle(s, "hello")
    assert "<research_log" in claude_calls(env)[-1]["stdin"] and "(none yet" in claude_calls(env)[-1]["stdin"]

    log_lines = ["# Goal: crafter seed 1 above 5"] + [f"- run {i}: lr={i}e-4 score {i / 10}" for i in range(3000)]
    (pdir / "RESEARCH_LOG.md").write_text("\n".join(log_lines) + "\n")
    await env.agent.handle(s, "FAKE_LOOP go on")
    j1 = agent_jobs(env, s)[-1]
    await run_job(env, j1)
    stdin = claude_calls(env)[-1]["stdin"]
    assert "[Wake-up you set up]" in stdin and "# Goal: crafter seed 1 above 5" in stdin
    assert "- run 2999:" in stdin and "- run 1500:" not in stdin  # the beginning and the latest entries
    assert "too long: condense" not in stdin
    (pdir / "RESEARCH_LOG.md").write_text("x" * 700_000)
    assert "too long: condense" in await env.agent.research_log("dreamer")


async def test_usage_report(env):
    from overseer.usage import usage_report
    assert "No agent runs today" in usage_report(env.store)
    s = env.agent.general_session()
    await env.agent.handle(s, "hi")
    env.agent.set_backend("codex")
    await env.agent.handle(env.agent.new_conversation(), "hi")
    report = usage_report(env.store, "today")
    assert "Agent usage today" in report and "2 run(s): 2 from you" in report and "$0.03" in report
    assert "Codex doesn't report a cost" in report and "top conversations" in report and s.id in report
    assert "No agent runs yesterday" in usage_report(env.store, "yesterday")
    assert "Usage:" in usage_report(env.store, "soon")


def test_durations():
    from overseer.autopilot import human_interval, parse_duration
    assert parse_duration("2h") == 7200 and parse_duration("90m") == 5400 and parse_duration("1d") == 86400
    assert parse_duration("1.5") == 5400 and parse_duration("off") == 0 and parse_duration("soon") is None
    assert parse_duration("30 minutes") == 1800
    assert human_interval(5400) == "1h30" and human_interval(172800) == "2d" and human_interval(900) == "15m"


async def test_status_follows_the_training_log(env):
    """SLURM's stdout often only has a launcher preamble; the training output goes to a file it names."""
    from overseer.bot import Bot
    proj = env.home / "proj"
    run = proj / "runs" / "seed0"
    run.mkdir(parents=True)
    train_log = run / "train_18743802.log"
    train_log.write_text("──── Step 11000 ────\nepisode/score 2.4 / episode/length 230\n"
                         "──── Step 12000 ────\nepisode/score 3.1 / episode/length 245 / train/loss/dyn 0.42\n")
    stdout = proj / "slurm-18743802.out"
    stdout.write_text(f"node: cn257 MUJOCO_GL: egl\npygame 2.6.1\ntraining: /venv/bin/python train.py --logdir {run}\n"
                      f"output: {train_log}\n")
    stderr = proj / "slurm-18743802.err"
    stderr.write_text("gymnasium/core.py:311: UserWarning: env.get_frame to get variables is deprecated\n")
    job = {"name": "bench-dreamerv3-mixed-s0", "state": "RUNNING", "workdir": str(proj), "stdout": str(stdout),
           "stderr": str(stderr), "elapsed": "1:00:00", "time_limit": "4:00:00", "nodelist": "cn257"}
    env.set_jobs(**{"18743802": job})
    await env.monitor.poll()

    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=env.notifier)
    await bot._send_status("18743802")
    text = env.notifier.sent[-1].text
    assert "⏱ 1:00:00 of 4:00:00 (25%, 3h00m left) · node cn257" in text
    assert "📈 step 12000 · episode/score 3.1 · train/loss/dyn 0.42" in text
    assert "🎮 GPU 0 A100-SXM4-80GB: 87% busy · 30.5/80 GB" in text
    assert "<code>train_18743802.log</code>, last lines:" in text and "Step 12000" in text
    assert "pygame" not in text and "UserWarning" not in text  # the preamble and harmless warnings stay out
    assert json.loads(env.store.get_job("18743802").info)["progress_log"] == str(train_log)

    # A real problem on stderr is flagged.
    stderr.write_text("Traceback (most recent call last):\n  File \"train.py\"\ntorch.OutOfMemoryError: CUDA out of memory.\n")
    await bot._send_status("18743802")
    assert "⚠️ stderr: <b>" in env.notifier.sent[-1].text

    # 📜 Log shows the training log; SLURM's own files come along as attachments.
    await bot._send_log("18743802", 200)
    msg = env.notifier.sent[-1]
    assert "train_18743802.log" in msg.text and "episode/score 3.1" in msg.text
    assert [name for name, _ in msg.files] == ["train_18743802.log", "job-18743802-stdout.log", "job-18743802-stderr.log"]

    # The final report reads the training log too (that's where a crash's traceback is).
    stderr.write_text("gymnasium/core.py:311: UserWarning: env.get_frame is deprecated\n")
    with train_log.open("a") as fh:
        fh.write("Traceback (most recent call last):\n  File \"dreamer.py\", line 9\nValueError: bad replay ratio\n")
    env.set_jobs(**{"18743802": {**job, "state": "FAILED", "exit": "1:0"}})
    await env.monitor.poll()
    final = next(m for m in env.notifier.sent if "Job 18743802" in m.text and "FAILED" in m.text)
    assert "ValueError: bad replay ratio" in final.text


# ----------------------------------------------------------------------------- projects

def make_projects(env):
    from overseer.projects import Projects
    return Projects(env.cfg, env.store, env.agent)


PICKERS = []  # (text, buttons) the bot replied with directly


def topic_update(text, thread=None, reply_to=None):
    async def reply_text(text, reply_markup=None, **kw):
        rows = reply_markup.inline_keyboard if reply_markup else []
        PICKERS.append((text, [(b.text, b.callback_data) for row in rows for b in row]))
    msg = NS(text=text, reply_to_message=NS(message_id=reply_to) if reply_to else None,
             message_thread_id=thread, is_topic_message=thread is not None, reply_text=reply_text)
    return NS(effective_message=msg, effective_chat=NS(id=1))


async def test_project_settings_inherit_from_main_chat(env):
    env.agent.set_backend("codex")
    env.agent.set_model("gpt-x")
    env.agent.set_effort("high")
    projects = make_projects(env)
    projects.create("dreamer", "/scratch/me/rf-dreamer", "DreamerV3 benchmarks")
    # A new project starts as a copy of the main chat...
    assert env.agent.backend_of("dreamer") == "codex"
    assert env.agent.runner("codex", "dreamer").model == "gpt-x" and env.agent.runner("codex", "dreamer").effort == "high"
    # ...and then the two are independent.
    env.agent.set_model("gpt-y", project="dreamer")
    env.agent.set_model("gpt-main")
    assert env.agent.runner("codex", "dreamer").model == "gpt-y" and env.agent.runner("codex").model == "gpt-main"
    env.agent.set_backend("claude_code")
    assert env.agent.backend_of("dreamer") == "codex"
    env.agent.set_model("default", project="dreamer")  # the config / CLI default, not the main chat's
    assert env.agent.runner("codex", "dreamer").model is None
    # Runs in different projects never share a runner (their model/effort would leak).
    assert env.agent.runner("codex", "dreamer") is not env.agent.runner("codex")

    with pytest.raises(Exception, match="already exists"):
        projects.create("dreamer", "/x")
    with pytest.raises(Exception, match="can't be a project name"):
        projects.create("Main Project", "/x")
    with pytest.raises(Exception, match="absolute path"):
        projects.create("chem", "scratch/chem")


async def test_projects_from_config(env):
    from overseer.config import ProjectConfig
    env.agent.set_model("opus")
    env.cfg.projects = {"chem": ProjectConfig(dir="/scratch/me/chem", context="DFT runs with ORCA", backend="codex")}
    projects = make_projects(env)
    assert projects.sync_config() == ["chem"]
    p = env.store.get_project("chem")
    assert (p.dir, p.context, p.source) == ("/scratch/me/chem", "DFT runs with ORCA", "config")
    assert env.agent.backend_of("chem") == "codex"
    assert env.agent.runner("claude_code", "chem").model == "opus"  # the rest is inherited

    env.agent.set_backend("claude_code", "chem")  # changed from Telegram: kept across restarts...
    assert projects.sync_config() == [] and env.agent.backend_of("chem") == "claude_code"
    # A value that changes in the config applies (to the project's current backend) at the next start.
    env.cfg.projects["chem"].model = "sonnet"
    env.cfg.projects["chem"].context = "now with xtb too"
    projects.sync_config()
    assert env.agent.runner("claude_code", "chem").model == "sonnet"
    assert env.agent.backend_of("chem") == "claude_code" and env.store.get_project("chem").context == "now with xtb too"
    assert env.agent.runner("claude_code").model == "opus"  # the main chat is untouched


async def test_jobs_are_assigned_to_projects_by_folder(env):
    projects = make_projects(env)
    proj = env.home / "proj"
    projects.create("dreamer", str(proj))
    projects.create("dreamer-a100", str(proj / "a100"))
    (proj / "a100").mkdir()
    env.set_jobs(**{"301": {"name": "a", "state": "RUNNING", "workdir": str(proj / "runs")},
                    "302": {"name": "b", "state": "RUNNING", "workdir": str(proj / "a100")},
                    "303": {"name": "c", "state": "RUNNING", "workdir": str(env.home / "elsewhere")}})
    await env.monitor.poll()
    assert env.store.get_job("301").project == "dreamer"
    assert env.store.get_job("302").project == "dreamer-a100"  # the deepest folder wins
    assert env.store.get_job("303").project is None
    started = {m.job_id: m for m in env.notifier.sent if "started" in m.text}
    assert ("📁 dreamer", "assign:303:dreamer") in [b for row in started["303"].buttons for b in row]
    assert all(not d.startswith("assign:") for row in started["301"].buttons for _, d in row)


async def test_project_conversation_context(env):
    projects = make_projects(env)
    proj = env.home / "proj"
    (proj / "OVERSEER.md").write_text("Use the venv in methods/dreamerv3/venv. Crafter seed 1 had a reward bug.\n")
    projects.create("dreamer", str(proj), "DreamerV3 benchmarks on crafter")
    env.agent.set_model("opus", project="dreamer")
    s = env.agent.general_session(project="dreamer")
    assert s.id.startswith("dreamer-") and s.project == "dreamer"
    await env.agent.handle(s, "check seed 1")
    call = env.cli_log()[-1]
    system = call["argv"][call["argv"].index("--append-system-prompt") + 1]
    assert "## Project: dreamer" in system and "DreamerV3 benchmarks on crafter" in system
    assert f"{proj}/OVERSEER.md" in system
    assert "<project_notes" in call["stdin"] and "reward bug" in call["stdin"]  # read on a new CLI session
    assert call["argv"][call["argv"].index("--model") + 1] == "opus"
    assert env.store.get_job("4242").project == "dreamer"  # the job it submitted
    await env.agent.handle(s, "and now?")
    assert "<project_notes" not in env.cli_log()[-1]["stdin"]  # a resumed session already has them
    # The main chat's conversation is separate and has no project section.
    main = env.agent.general_session()
    assert main.id != s.id and "## Project" not in env.agent.system_prompt(main.id)


class FakeTelegram:
    """Just enough of telegram.Bot for TelegramNotifier."""

    def __init__(self, topics=True):
        self.topics = topics
        self.sent = []  # (chat_id, thread, text)
        self.created = []  # (chat_id, name, thread)
        self.dead_threads = set()
        self.can_create = True
        self._next = 100

    async def get_me(self):
        return NS(has_topics_enabled=self.topics)

    async def create_forum_topic(self, chat_id, name):
        from telegram.error import BadRequest
        if not self.can_create:
            raise BadRequest("BOT_FORUM_CREATE_FORBIDDEN")
        self._next += 1
        self.created.append((chat_id, name, self._next))
        return NS(message_thread_id=self._next)

    async def send_message(self, chat_id, text, message_thread_id=None, **kw):
        from telegram.error import BadRequest
        if message_thread_id in self.dead_threads:
            raise BadRequest("Message thread not found")
        self.sent.append((chat_id, message_thread_id, text))
        return NS(message_id=len(self.sent))

    async def send_document(self, chat_id, document, filename, message_thread_id=None, **kw):
        self.sent.append((chat_id, message_thread_id, f"<file {filename}>"))
        return NS(message_id=len(self.sent))

    async def send_photo(self, chat_id, photo, filename, message_thread_id=None, **kw):
        self.sent.append((chat_id, message_thread_id, f"<photo {filename}>"))
        return NS(message_id=len(self.sent))

    api_calls: list = None  # (endpoint, kwargs) of do_api_request
    api_errors: dict = None  # endpoint -> exception to raise

    async def send_message_draft(self, chat_id, draft_id, text=None, message_thread_id=None, parse_mode=None,
                                 api_kwargs=None):
        return await self.do_api_request("sendMessageDraft", {
            "chat_id": chat_id, "draft_id": draft_id, "text": text, "message_thread_id": message_thread_id,
            "parse_mode": parse_mode, **(api_kwargs or {})})

    async def do_api_request(self, endpoint, api_kwargs=None, return_type=None):
        self.api_calls = (self.api_calls or []) + [(endpoint, api_kwargs)]
        err = (self.api_errors or {}).get(endpoint)
        if err:
            raise err
        if endpoint == "sendRichMessage":
            self.sent.append((api_kwargs["chat_id"], api_kwargs.get("message_thread_id"),
                              "<rich>" + api_kwargs["rich_message"]["markdown"]))
            return NS(message_id=len(self.sent))
        return True


async def test_topics_routing(env):
    from overseer.bot import TelegramNotifier
    tg = FakeTelegram()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    projects = make_projects(env)
    projects.create("dreamer", "/scratch/me/rf-dreamer")
    assert await notifier.detect_topics()
    await notifier.ensure_topic("dreamer")
    thread = env.store.get_project("dreamer").thread_map["1"]
    assert tg.created == [(1, "dreamer", thread)] and tg.sent[-1][1] == thread  # with an intro message
    await notifier.ensure_topic("dreamer")
    assert len(tg.created) == 1  # only once

    # A job's messages go to its project's topic; others to the main chat.
    env.store.upsert_job(JobRecord(job_id="501", project="dreamer"))
    env.store.upsert_job(JobRecord(job_id="502"))
    await notifier.send(Outgoing("📊 job 501", job_id="501"))
    await notifier.send(Outgoing("📊 job 502", job_id="502"))
    assert tg.sent[-2] == (1, thread, "📊 job 501") and tg.sent[-1] == (1, None, "📊 job 502")
    # A conversation's messages follow the conversation's project.
    s = env.agent.general_session(project="dreamer")
    await notifier.send(Outgoing("🤖 hi", session_id=s.id))
    assert tg.sent[-1][1] == thread
    notifier.streaming = False  # (drafts are tested in test_streaming_drafts)
    handle = await notifier.post_live(Outgoing("⏳ working", session_id=s.id))
    assert tg.sent[-1][1] == thread and handle["messages"][0][2] == ""

    # The topic was deleted: a new one is created and the message lands there.
    tg.dead_threads.add(thread)
    await notifier.send(Outgoing("✅ job 501 finished", job_id="501"))
    new_thread = env.store.get_project("dreamer").thread_map["1"]
    assert new_thread != thread and tg.sent[-1] == (1, new_thread, "✅ job 501 finished")
    # ...but not again within the hour (a Telegram outage mustn't spawn a topic per message): the main
    # chat gets it, labelled.
    tg.dead_threads.add(new_thread)
    await notifier.send(Outgoing("✅ again", job_id="501"))
    assert tg.sent[-1] == (1, None, "📁 <b>dreamer</b> · ✅ again") and len(tg.created) == 2
    notifier._recreated.clear()
    tg.can_create = False  # and if no topic can be made at all, the same
    await notifier.send(Outgoing("✅ and again", job_id="501"))
    assert tg.sent[-1] == (1, None, "📁 <b>dreamer</b> · ✅ and again")

    # Without topic mode, project messages are labelled in the main chat.
    plain = TelegramNotifier(NS(bot=FakeTelegram(topics=False)), [1], env.store)
    assert not await plain.detect_topics()
    await plain.send(Outgoing("📊 job 501", job_id="501"))
    assert plain.app.bot.sent[-1] == (1, None, "📁 <b>dreamer</b> · 📊 job 501")


async def test_project_commands(env, monkeypatch):
    from overseer.bot import Bot
    projects = make_projects(env)
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=env.notifier, projects=projects)
    replies, started = [], []

    async def reply(update, text):
        replies.append(text)

    async def fake_run_agent(chat_id, session, text):
        started.append((session.id, text))

    monkeypatch.setattr(bot, "reply", reply)
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)

    async def drain():
        for t in list(bot._tasks):
            await t

    await bot.cmd_project(topic_update("/project"), NS(args=[]))
    assert "No projects yet" in replies[-1]
    env.agent.set_model("opus")
    await bot.cmd_project(topic_update("/project new"), NS(args=["new", "dreamer", "/scratch/me/rf-dreamer", "DreamerV3",
                                                                  "benchmarks"]))
    assert "Project <code>dreamer</code> created" in replies[-1] and "claude_code: model opus" in replies[-1]
    assert env.store.get_project("dreamer").context == "DreamerV3 benchmarks"
    # Created inside an existing topic: that topic becomes the project's.
    await bot.cmd_project(topic_update("/project new", thread=77), NS(args=["new", "chem", "/scratch/me/chem"]))
    assert env.store.get_project("chem").thread_map == {"1": 77} and "this topic" in replies[-1]
    await bot.cmd_project(topic_update("/project new", thread=77), NS(args=["new", "other", "/x/y"]))
    assert "already belongs to project chem" in replies[-1]

    # Messages in a project's topic go to that project's conversation; in an unknown topic, a hint.
    await bot.on_text(topic_update("run the DFT sweep", thread=77), None)
    await drain()
    assert started[-1][0].startswith("chem-") and started[-1][1] == "run the DFT sweep"
    await bot.on_text(topic_update("hello?", thread=999), None)  # a thread of its own: which project?
    await drain()
    assert PICKERS[-1][0] == "New conversation. Which project is it about?" and len(started) == 1
    await bot.on_text(topic_update("hi main"), None)
    await drain()
    assert started[-1][0].startswith("chat-")

    # Settings commands act on the topic's project only.
    await bot.cmd_model(topic_update("/model", thread=77), NS(args=["sonnet"]))
    assert "model sonnet" in replies[-1] and "project <code>chem</code>" in replies[-1]
    assert env.agent.runner("claude_code").model == "opus"
    await bot.cmd_backend(topic_update("/backend", thread=77), NS(args=["codex"]))
    assert env.agent.backend_of("chem") == "codex" and env.agent.backend == "claude_code"

    # /jobs in a topic lists that project's jobs; /assign moves jobs (args, reply, in-topic, button).
    env.store.upsert_job(JobRecord(job_id="801", name="dft", state="RUNNING", project="chem"))
    env.store.upsert_job(JobRecord(job_id="802", name="misc", state="RUNNING"))
    await bot.cmd_jobs(topic_update("/jobs", thread=77), NS(args=[]))
    assert "801" in replies[-1] and "802" not in replies[-1]
    await bot.cmd_jobs(topic_update("/jobs"), NS(args=[]))
    assert "801" in replies[-1] and "📁chem" in replies[-1] and "802" in replies[-1]
    await bot.cmd_assign(topic_update("/assign"), NS(args=["802", "dreamer"]))
    assert env.store.get_job("802").project == "dreamer" and "moved to project <code>dreamer</code>" in replies[-1]
    assert "moved here from the main chat" in env.notifier.sent[-1].text and env.notifier.sent[-1].job_id == "802"
    env.store.link_message(1, 555, "802", None)
    await bot.cmd_assign(topic_update("/assign", reply_to=555), NS(args=["main"]))
    assert env.store.get_job("802").project is None
    await bot.cmd_assign(topic_update("/assign", thread=77), NS(args=["802"]))
    assert env.store.get_job("802").project == "chem"
    await bot._assign("802", "nope", None)
    assert "No active project" in replies[-1]
    s = env.agent.session_for_job("801")
    assert s.project == "chem"  # a job's conversation lives in the job's project
    env.agent.move_job("801", "dreamer")
    assert env.agent.get(s.id).project == "dreamer"

    # Context, archive, restore.
    await bot.cmd_project(topic_update("/project", thread=77), NS(args=["context", "ORCA", "6", "on", "the", "cpu", "nodes"]))
    assert env.store.get_project("chem").context == "ORCA 6 on the cpu nodes"
    await bot.cmd_project(topic_update("/project"), NS(args=["list"]))
    assert "📁 <b>chem</b>" in replies[-1] and "ORCA 6" in replies[-1] and "📁 <b>dreamer</b>" in replies[-1]
    await bot.cmd_project(topic_update("/project"), NS(args=["archive", "chem"]))
    assert env.store.get_project("chem").archived and env.store.project_for_dir("/scratch/me/chem/x") is None
    await bot.on_text(topic_update("still there?", thread=77), None)
    assert "is archived" in replies[-1]
    await bot.cmd_project(topic_update("/project"), NS(args=["restore", "chem"]))
    assert not env.store.get_project("chem").archived and "restored" in replies[-1]


# ----------------------------------------------------------------------------- rich messages

async def test_rich_reply_and_outbox(env, monkeypatch):
    secret = env.tmp / "secret.key"
    secret.write_text("PRIVATE KEY MATERIAL")
    monkeypatch.setenv("FAKE_SECRET", str(secret))
    env.cfg.agent.claude_code.pass_env.append("FAKE_SECRET")
    s = env.agent.general_session()
    await env.agent.handle(s, "FAKE_MD FAKE_PLOT plot the reward")
    reply = env.notifier.sent[-1]
    # Markdown arrives formatted.
    assert reply.text.startswith("🤖 <b>Fixed</b> the batch size in <code>train.py</code> and resubmitted:")
    assert "• job <b>4242</b>" in reply.text
    # The outbox: the picture and the CSV are attached; the symlink and the oversized file are not.
    assert reply.photos == [("reward.png", b"\x89PNG\r\n\x1a\n fake image")]
    assert reply.files == [("results.csv", b"seed,score\n0,3.1\n")]
    assert "innocent.png: not sent (not a regular file)" in reply.text and "huge.bin: not sent (60 MB" in reply.text
    assert all(b"PRIVATE" not in data for _, data in reply.photos + reply.files) and secret.exists()
    assert list((env.cfg.local.tmp("agents", s.id) / "outbox").iterdir()) == []  # emptied after sending
    # The commands come folded, and the agent knows about the outbox.
    summary = next(m.text for m in env.notifier.sent if "command(s)" in m.text)
    assert "<blockquote expandable>$ ssh hpc" in summary
    assert "./outbox/" in env.agent.system_prompt(s.id) and "scp hpc:/path/plot.png ./outbox/" in env.agent.system_prompt(s.id)


async def test_plot_button(env, monkeypatch):
    from overseer.bot import Bot
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=env.notifier)
    started = []

    async def fake_run_agent(chat_id, session, text):
        started.append((session.id, text))

    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)
    env.store.upsert_job(JobRecord(job_id="901", name="dreamer", state="RUNNING", stdout="/s/slurm-901.out",
                                   info=json.dumps({"progress_log": "/s/train_901.log"})))
    await bot._start_plot(1, "901")
    for t in list(bot._tasks):
        await t
    sid, text = started[-1]
    assert sid == env.agent.session_for_job("901").id
    assert "/s/train_901.log" in text and "./outbox/" in text and "PNG" in text
    assert "is plotting job" in env.notifier.sent[-1].text


async def test_telegram_pictures_and_fallbacks(env):
    from telegram.error import BadRequest
    from overseer.bot import TelegramNotifier

    class Tg(FakeTelegram):
        reject_photos = False
        reject_html = False

        async def send_message(self, chat_id, text, message_thread_id=None, parse_mode=None, **kw):
            if self.reject_html and parse_mode:
                raise BadRequest("Can't parse entities")
            return await super().send_message(chat_id, text, message_thread_id=message_thread_id, **kw)

        async def send_photo(self, chat_id, photo, filename, message_thread_id=None, **kw):
            if self.reject_photos:
                raise BadRequest("PHOTO_INVALID_DIMENSIONS")
            self.sent.append((chat_id, message_thread_id, f"<photo {filename}>"))
            return NS(message_id=len(self.sent))

        async def send_media_group(self, chat_id, media, message_thread_id=None, **kw):
            self.sent.append((chat_id, message_thread_id, f"<album {len(media)}>"))
            return [NS(message_id=len(self.sent) * 100 + i) for i in range(len(media))]

    tg = Tg()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    s = env.agent.general_session()
    await notifier.send(Outgoing("🤖 one", session_id=s.id, photos=[("a.png", b"1")], files=[("r.csv", b"x")]))
    assert [t for _, _, t in tg.sent[-3:]] == ["🤖 one", "<photo a.png>", "<file r.csv>"]
    await notifier.send(Outgoing("🤖 three", session_id=s.id, photos=[(f"{i}.png", b"1") for i in range(3)]))
    assert tg.sent[-1][2] == "<album 3>"
    assert env.store.message_link(1, tg.sent[-1][0] and len(tg.sent) * 100)[1] == s.id  # replies to it continue
    tg.reject_photos = True
    await notifier.send(Outgoing("🤖 odd", photos=[("wide.png", b"1")]))
    assert tg.sent[-1][2] == "<file wide.png>"  # sent as a file instead
    tg.reject_html = True
    await notifier.send(Outgoing("🤖 <b>bold</b> &amp; <code>x</code>"))
    assert tg.sent[-1][2] == "🤖 bold & x"  # plain text, no stray tags


async def test_threads_are_conversations(env, monkeypatch):
    from overseer.bot import Bot, TelegramNotifier
    tg = FakeTelegram()
    renamed = []

    async def edit_forum_topic(chat_id, message_thread_id, name):
        renamed.append((message_thread_id, name))
    tg.edit_forum_topic = edit_forum_topic
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    projects = make_projects(env)
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=notifier, projects=projects)
    started = []

    async def fake_run_agent(chat_id, session, text):
        started.append((session.id, text))
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)

    async def drain():
        for t in list(bot._tasks):
            await t

    async def press(data, thread):
        q = NS(data=data, message=NS(message_thread_id=thread), answer=_noop,
               edit_message_text=lambda text: _noop())
        await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1), effective_message=q.message), None)
        await drain()

    # Without projects, a new thread simply becomes a new conversation.
    await bot.on_text(topic_update("why is seed 1 at zero?", thread=201), None)
    await drain()
    first = started[-1][0]
    assert env.store.thread_session(1, 201) == first and env.agent.get(first).project is None
    await bot.on_text(topic_update("and seed 2?", thread=201), None)  # same thread, same conversation
    await drain()
    assert started[-1][0] == first
    await bot.on_text(topic_update("unrelated", thread=202), None)  # another thread, another conversation
    await drain()
    assert started[-1][0] not in (first,)

    # With projects: a picker; the choice binds the thread to that project and renames it.
    projects.create("dreamer", "/scratch/me/rf-dreamer")
    projects.create("chem", "/scratch/me/chem")
    n = len(started)
    await bot.on_text(topic_update("plot the reward curves", thread=203), None)
    await drain()
    assert len(started) == n and ("📁 dreamer", "thread:dreamer") in PICKERS[-1][1]
    assert ("💬 No project", "thread:") in PICKERS[-1][1]
    await press("thread:dreamer", 203)
    sid, text = started[-1]
    assert text == "plot the reward curves" and env.agent.get(sid).project == "dreamer"
    assert env.store.thread_session(1, 203) == sid and renamed[-1] == (203, "dreamer · plot the reward curves")
    # "<project>: ..." skips the picker.
    await bot.on_text(topic_update("chem: is the DFT run done?", thread=204), None)
    await drain()
    assert env.agent.get(started[-1][0]).project == "chem" and started[-1][1] == "is the DFT run done?"
    # Commands act on the thread's conversation: /new gives the thread a fresh one, in the same project.
    await bot.cmd_new(topic_update("/new", thread=203), NS(args=[]))
    fresh = env.store.thread_session(1, 203)
    assert fresh != sid and env.agent.get(fresh).project == "dreamer"

    # Replies of a conversation, and reports on the jobs it submitted, go to its thread.
    await notifier.send(Outgoing("🤖 done", session_id=fresh))
    assert tg.sent[-1] == (1, 203, "🤖 done")
    env.store.upsert_job(JobRecord(job_id="777", by_agent=1, session_id=fresh, project="dreamer"))
    await notifier.send(Outgoing("✅ Job 777 finished", job_id="777"))
    assert tg.sent[-1] == (1, 203, "✅ Job 777 finished")
    # The thread was deleted: messages continue in the project's topic (here: labelled main chat).
    tg.dead_threads.add(203)
    await notifier.send(Outgoing("🤖 still there?", session_id=fresh))
    assert tg.sent[-1] == (1, None, "📁 <b>dreamer</b> · 🤖 still there?") and env.store.thread_session(1, 203) is None


async def _noop(*a, **k):
    return None


async def test_native_rich_messages(env):
    from telegram import InputFile
    from telegram.error import BadRequest, EndPointNotFound
    from overseer.bot import TelegramNotifier
    tg = FakeTelegram()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    s = env.agent.general_session()
    msg = Outgoing("🤖 <b>html</b> fallback", markdown="**Done** - a table:\n\n| a | b |\n|---|---|\n| 1 | 2 |",
                   session_id=s.id, photos=[("reward.png", b"PNG")], files=[('res"ults.csv', b"x,y")],
                   buttons=[[("📜 Log", "log:1")]])
    await notifier.send(msg)
    endpoint, kw = tg.api_calls[-1]
    assert endpoint == "sendRichMessage" and kw["chat_id"] == 1
    md = kw["rich_message"]["markdown"]
    assert md.startswith("**Done** - a table:") and '![](tg://photo?id=p0 "reward.png")' in md
    assert '![](tg://document?id=d0 "results.csv")' in md  # quotes can't break the caption
    assert kw["rich_message"]["media"] == [{"id": "p0", "media": {"type": "photo", "media": "attach://file_p0"}},
                                           {"id": "d0", "media": {"type": "document", "media": "attach://file_d0"}}]
    assert isinstance(kw["file_p0"], InputFile) and isinstance(kw["file_d0"], InputFile)
    assert kw["reply_markup"].inline_keyboard[0][0].callback_data == "log:1"
    assert env.store.message_link(1, len(tg.sent))[1] == s.id  # replies to it continue the conversation

    # Telegram rejects it: the regular message goes out instead (with its pictures and files).
    tg.api_errors = {"sendRichMessage": BadRequest("Can't parse rich message")}
    await notifier.send(msg)
    assert [t for _, _, t in tg.sent[-3:]] == ["🤖 <b>html</b> fallback", "<photo reward.png>", '<file res"ults.csv>']
    # A server without rich messages: switched off for good.
    tg.api_errors = {"sendRichMessage": EndPointNotFound("Not Found")}
    await notifier.send(Outgoing("plain", markdown="**rich**"))
    assert not notifier.rich and tg.sent[-1][2] == "plain"
    n = len(tg.api_calls)
    await notifier.send(Outgoing("plain again", markdown="**rich**"))
    assert len(tg.api_calls) == n

    # The agent's replies and command summaries come with a rich version.
    await env.agent.handle(s, "FAKE_MD go")
    reply = env.notifier.sent[-1]
    assert reply.markdown.startswith("**Fixed** the batch size in `train.py`") and "<i>claude_code ·" in reply.markdown
    summary = next(m for m in env.notifier.sent if "command(s)" in m.text)
    assert "<details><summary>Show</summary>" in summary.markdown and "```bash\n$ ssh hpc" in summary.markdown


async def test_streaming_drafts(env, monkeypatch):
    from overseer.bot import Bot, TelegramNotifier
    tg = FakeTelegram()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    s = env.agent.general_session()
    handle = await notifier.post_live(Outgoing("⏳ codex is working", session_id=s.id))
    endpoint, kw = tg.api_calls[-1]
    assert endpoint == "sendMessageDraft" and kw["can_stop"] and kw["parse_mode"] == "HTML"
    assert kw["draft_id"] == handle["draft"] and notifier.draft_session(handle["draft"]) == s.id
    await notifier.edit_live(handle, "⏳ still working")
    assert tg.api_calls[-1][1]["draft_id"] == handle["draft"] and tg.api_calls[-1][1]["text"] == "⏳ still working"

    # The draft's Stop button stops the agent's run.
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent, notifier=notifier)
    stopped = []

    async def stop(sid):
        stopped.append(sid)
        return True
    monkeypatch.setattr(env.agent, "stop", stop)
    update = NS(api_kwargs={"stopped_message_generation": {"chat": {"id": 1, "type": "private"}, "draft_id": handle["draft"]}})
    await bot.on_raw_update(update, None)
    assert stopped == [s.id]
    await bot.on_raw_update(NS(api_kwargs={"stopped_message_generation": {"chat": {"id": 666}, "draft_id": handle["draft"]}}), None)
    assert stopped == [s.id]  # not from a stranger
    await notifier.delete_live(handle)
    assert notifier.draft_session(handle["draft"]) is None

    # Drafts unavailable: back to a message edited in place.
    from telegram.error import EndPointNotFound
    tg.api_errors = {"sendMessageDraft": EndPointNotFound("Not Found")}
    handle = await notifier.post_live(Outgoing("⏳ working", session_id=s.id))
    assert "messages" in handle and not notifier.streaming and tg.sent[-1][2] == "⏳ working"


def test_login_node_warnings(monkeypatch):
    """Commands that ran a while directly on the login node are flagged; work handed to SLURM isn't."""
    import overseer.runners as runners
    clock = [1000.0]
    monkeypatch.setattr(runners.time, "monotonic", lambda: clock[0])
    a = Activity(login_warn_seconds=120)
    for cmd, took in [("ssh hpc 'python make_plots.py'", 200), ("ssh hpc 'ls -la'", 2),
                      ("hpc-run --dir ~/proj -- python make_plots.py", 900), ("ssh hpc 'cd x && sbatch run.sh'", 300),
                      ("python local_thing.py", 500), ("/bin/bash -lc \"ssh hpc 'srun -p compute python x.py'\"", 400)]:
        a.command(cmd)
        clock[0] += took
        a.output(cmd, "")
    assert a.login_node == [("ssh hpc 'python make_plots.py'", 200)]


async def test_login_node_warning_in_summary(env):
    s = env.agent.general_session()
    a = Activity(login_warn_seconds=120)
    a.commands = ["ssh hpc 'python make_plots.py'"]
    a.login_node = [("ssh hpc 'python make_plots.py'", 312)]
    await env.agent._report_activity(s, a, [])
    msg = env.notifier.sent[-1]
    assert "⚠️ ran 5m 12s on the login node: <code>ssh hpc 'python make_plots.py'</code>" in msg.text
    assert "⚠️ ran 5m 12s on the login node" in msg.markdown  # the rich version too


async def test_submissions_through_scripts_are_found(env):
    """A job submitted by a script or CLI (no "sbatch" in the agent's command) is still recognised: SLURM
    says it was submitted during the turn and its id is in the output or reply. Old jobs mentioned aren't."""
    import time as _time
    ap = await make_autopilot(env)
    soon = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime(_time.time() + 60))  # during the turn
    env.set_jobs(**{"7777": {"name": "native7", "state": "PENDING", "submit": soon},
                    "7778": {"name": "seed1", "state": "PENDING", "submit": soon},
                    "1234": {"name": "old", "state": "FAILED", "submit": "2026-01-01T00:00:00"}})
    s = env.agent.general_session()
    await env.agent.handle(s, "FAKE_LOOP rerun the benchmark FAKE_SCRIPT")
    await ap.drain()
    await asyncio.sleep(0.05)  # the announcements are sent in the background
    announced = [m.text for m in env.notifier.sent if "Agent submitted job" in m.text]
    assert any("job 7777</b>" in t and "native7_cli.py submit" in t for t in announced), announced
    assert any("job 7778</b>" in t and "in the agent's reply" in t for t in announced), announced
    assert not any("job 1234" in t for t in announced)
    assert {j.job_id for j in env.store.session_jobs(s.id)} == {"7777", "7778"}
    reply = next(m.text for m in reversed(env.notifier.sent) if m.text.startswith("🤖"))
    assert "Agent submitted" not in reply


async def test_assign_links_a_job_to_the_conversation(env, monkeypatch):
    """A job the agent submitted unnoticed can be handed to its conversation: /assign <id> in the thread."""
    from overseer.bot import Bot
    ap = await make_autopilot(env)
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=env.notifier)
    replies = []

    async def reply(update, text):
        replies.append(text)
    monkeypatch.setattr(bot, "reply", reply)
    s = env.agent.new_conversation()
    env.store.bind_thread(1, 301, s.id)
    env.store.upsert_job(JobRecord(job_id="18746955", name="native7", state="RUNNING"))  # seen, but not linked
    await bot.cmd_assign(topic_update("/assign", thread=301), NS(args=["18746955"]))
    rec = env.store.get_job("18746955")
    assert rec.by_agent == 1 and rec.session_id == s.id
    assert "part of this conversation" in replies[-1] and "the agent is woken when it ends" in replies[-1]
    # When it ends, the conversation is woken.
    proj = env.home / "proj"
    env.set_jobs(**{"18746955": {"name": "native7", "state": "COMPLETED", "workdir": str(proj),
                                 "stdout": str(proj / "x.out")}})
    (proj / "x.out").write_text("done\n")
    await env.monitor.poll()
    await ap.drain()
    assert any("Waking the agent</b>: 18746955 COMPLETED" in m.text for m in env.notifier.sent)


async def test_general_topic(env, monkeypatch):
    """With topics, anything outside every project lives in a "general" topic."""
    from overseer.bot import Bot, TelegramNotifier
    tg = FakeTelegram()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    await notifier.ensure_general_topic()
    gen = notifier.general_thread(1)
    assert tg.created == [(1, "general", gen)] and "general" in tg.sent[-1][2]
    await notifier.ensure_general_topic()
    assert len(tg.created) == 1  # once

    # A job of yours outside every project reports there; a conversation of the main chat answers there.
    env.store.upsert_job(JobRecord(job_id="610"))
    await notifier.send(Outgoing("📥 Job 610 queued", job_id="610"))
    assert tg.sent[-1] == (1, gen, "📥 Job 610 queued")
    main = env.agent.general_session()
    await notifier.send(Outgoing("🤖 hi", session_id=main.id))
    assert tg.sent[-1][1] == gen

    # Messages in the general topic go to the main conversation; /new there starts a fresh one.
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=notifier)
    started, replies = [], []

    async def fake_run_agent(chat_id, session, text):
        started.append((session.id, text))

    async def reply(update, text):
        replies.append(text)
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)
    monkeypatch.setattr(bot, "reply", reply)
    await bot.on_text(topic_update("how much quota do I have?", thread=gen), None)
    for t in list(bot._tasks):
        await t
    assert started[-1] == (main.id, "how much quota do I have?")
    await bot.cmd_new(topic_update("/new", thread=gen), NS(args=[]))
    assert env.agent.general_session().id != main.id and env.store.thread_session(1, gen) is None

    # The general topic was deleted: a new one is made.
    tg.dead_threads.add(gen)
    await notifier.send(Outgoing("✅ Job 610 finished", job_id="610"))
    assert notifier.general_thread(1) != gen and tg.sent[-1] == (1, notifier.general_thread(1), "✅ Job 610 finished")


# ----------------------------------------------------------------------------- ov: the agent talks to Overseer

async def test_ov_interface(env):
    """The agent's `ov` command, run by the (fake) agent CLI through the generated client and the socket."""
    from overseer.ctl import ControlServer
    projects = make_projects(env)
    ctl = ControlServer(env.cfg, env.agent, projects, env.notifier, env.hpc)
    await ctl.start()
    try:
        assert "## Talking to Overseer: `ov`" in env.agent.system_prompt()
        env.set_jobs(**{
            "5001": {"name": "native7", "state": "RUNNING", "workdir": str(env.home / "proj")},
            "5002": {"name": "theirs", "state": "RUNNING", "user": "someone-else"},
        })
        s = env.agent.general_session()
        await env.agent.handle(s, "\n".join([
            "FAKE_OV ov job add 5001",
            "FAKE_OV ov job add 5001 999999 5002",
            "FAKE_OV ov project new dreamer --dir ~/dreamer --about 'world models' --move-here",
            "FAKE_OV ov job move 5001 dreamer",
            "FAKE_OV ov status",
            "FAKE_OV ov job delete 5001",
            "FAKE_OV ov job list",
            "FAKE_OV ov project nonsense",
            "FAKE_OV echo token=$OVERSEER_CTL > token.txt",
        ]))
        answer = env.notifier.sent[-1].text
        outs = answer.split("@@@")
        assert "✓ job 5001 (native7, RUNNING) is tracked in this conversation" in outs[0] and "[exit 0]" in outs[0]
        assert "(already was)" in outs[1] and "doesn't know a job 999999" in outs[1]
        assert "belongs to someone-else" in outs[1]
        assert "✓ project dreamer created" in outs[2] and "now in dreamer" in outs[2]
        assert "✓ job 5001 moved to dreamer" in outs[3]
        assert "project: dreamer" in outs[4] and "5001 (RUNNING)" in outs[4]
        assert "no longer tracked" in outs[5]
        assert "No jobs in this conversation" in outs[6]
        assert "[exit 1]" in outs[7]

        rec = env.store.get_job("5001")
        assert rec.by_agent and rec.session_id == s.id and rec.project == "dreamer" and rec.ignored
        assert env.store.get_job("999999") is None and env.store.get_job("5002") is None
        assert 5001 not in [int(j.job_id) for j in env.store.active_jobs()]
        assert env.store.get_session(s.id).project == "dreamer"
        assert projects.get("dreamer").dir == "~/dreamer" and projects.get("dreamer").context == "world models"
        texts = env.notifier.texts()
        assert texts.count("Agent submitted job 5001") == 1  # announced once
        assert "created project <code>dreamer</code>" in texts and "no longer tracks job <code>5001</code>" in texts

        # The run's token stops working when the run ends.
        token = (env.agent.runner().workdir(s.id) / "token.txt").read_text().split("=", 1)[1].strip()
        assert token and not ctl.tokens
        ok, text = await ctl.handle(token, ["status"])
        assert not ok and "Not connected" in text
    finally:
        await ctl.stop()
    assert not env.cfg.ctl_socket.exists()


async def test_ov_job_add_needs_slurm(env):
    """Jobs that finished long ago are still found in accounting; ids that aren't numbers are refused."""
    from overseer.ctl import ControlServer
    ctl = ControlServer(env.cfg, env.agent, make_projects(env), env.notifier, env.hpc)
    s = env.agent.general_session()
    token = ctl.open(s.id)
    ok, text = await ctl.handle(token, ["job", "add", "12;rm"])
    assert ok and "isn't a job id" in text
    env.set_jobs(**{"5003": {"name": "old", "state": "COMPLETED"}})
    ok, text = await ctl.handle(token, ["job", "add", "5003"])
    assert ok and "✓ job 5003" in text
    ok, text = await ctl.handle(token, ["job", "move", "5003", "nope"])
    assert not ok and "no project 'nope'" in text
    ok, text = await ctl.handle(token, [])
    assert ok and "ov job add" in text


# ----------------------------------------------------------------------------- context compaction

@pytest.mark.parametrize("backend", ["claude_code", "codex"])
async def test_automatic_compaction_is_reported(env, backend):
    """The CLI compacts the conversation on its own when the context fills up: the reply says so."""
    env.agent.set_backend(backend)
    s = env.agent.general_session()
    await env.agent.handle(s, "go on FAKE_COMPACT")
    reply = next(m.text for m in reversed(env.notifier.sent) if m.text.startswith("🤖"))
    assert "🗜 <i>The conversation was compacted automatically, as the context was full" in reply
    if backend == "claude_code":  # Claude Code says how much it freed
        assert "(190.0k → 4.0k tokens)" in reply
    events = [json.loads(line) for line in (env.cfg.local.state("transcripts") / f"{s.id}.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "compact" and e["trigger"] == "auto" for e in events)

    await env.agent.handle(s, "and again, without compacting")  # only compactions of this run count
    reply = next(m.text for m in reversed(env.notifier.sent) if m.text.startswith("🤖"))
    assert "🗜" not in reply


@pytest.mark.parametrize("backend", ["claude_code", "codex"])
async def test_compact_on_request(env, backend, monkeypatch):
    from overseer.bot import Bot
    env.agent.set_backend(backend)
    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=env.notifier)
    replies = []

    async def reply(update, text):
        replies.append(text)
    monkeypatch.setattr(bot, "reply", reply)

    s = env.agent.general_session()
    await env.agent.compact(s)
    assert "Nothing to compact" in env.notifier.sent[-1].text  # no history yet

    await env.agent.handle(s, "hi")
    cli_session = env.store.get_session(s.id).cli_session_id
    await bot.cmd_compact(None, NS(args=["keep", "the", "lr", "results"]))  # in the main chat: its conversation
    for t in list(bot._tasks):
        await t
    done = env.notifier.sent[-1].text
    assert "🗜 <b>Compacted the conversation</b>" in done
    call = env.cli_log()[-1]
    if backend == "claude_code":  # Claude Code's own /compact, in the same session
        assert call["stdin"] == "/compact keep the lr results" and call["argv"][call["argv"].index("--resume") + 1] == cli_session
        assert ": 150.0k → 3.0k tokens" in done
    else:  # codex's app server, on the same thread
        assert call["argv"][0] == "app-server" and "doesn't take a focus" in done
    assert env.store.get_session(s.id).cli_session_id == cli_session  # the conversation continues
    events = [json.loads(line) for line in (env.cfg.local.state("transcripts") / f"{s.id}.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "end" and any(e["kind"] == "compact" and e["trigger"] == "manual" for e in events)
    await env.agent.handle(s, "continue")  # and works afterwards
    assert env.notifier.sent[-1].text.startswith("🤖")


async def test_hpc_run_worker_is_never_a_job(env):
    """The agent mistook hpc-run's worker for one of its jobs: ov refuses it, and the monitor never reports it."""
    await make_autopilot(env)
    ctl = env.ctl
    s = env.agent.general_session()
    token = ctl.open(s.id)
    env.set_jobs(**{"18747467": {"name": "overseer-worker", "state": "PENDING"}})
    ok, text = await ctl.handle(token, ["job", "add", "18747467"])
    assert "✗ job 18747467 is hpc-run's shared compute worker" in text and env.store.get_job("18747467") is None
    ok, text = await ctl.handle(token, ["watch", "18747467"])
    assert not ok and "shared compute worker" in text

    # Registered anyway (an older Overseer, or by hand): when it's cancelled, nothing is reported.
    env.store.upsert_job(JobRecord(job_id="18747467", name="overseer-worker", state="PENDING", by_agent=1,
                                   session_id=s.id, first_seen=time.time()))
    env.set_jobs(**{"18747467": {"name": "overseer-worker", "state": "CANCELLED", "exit": "0:0"}})
    n = len(env.notifier.sent)
    await env.monitor.poll()
    assert len(env.notifier.sent) == n, env.notifier.texts()
    assert env.store.get_job("18747467").ignored


async def test_disk_quota(env, monkeypatch, tmp_path):
    """The quota command's limits reach the agent; nearly full filesystems warn you once and the agent each time."""
    from test_parsing import QUOTA
    table = tmp_path / "quota.txt"
    table.write_text(QUOTA.replace("301       500 ( 60%)", "425       500 ( 85%)"))
    monkeypatch.setenv("FAKE_QUOTA", str(table))
    await env.monitor.poll()
    warning = next(m.text for m in env.notifier.sent if "💾" in m.text)
    assert "<b>/scratch is at 85% of its quota</b>: 425,000 of 500,000 files (inodes)" in warning
    system = env.agent.system_prompt()
    assert "`/home`: 50GB and 500k files; `/scratch`: 5000GB and 500k files" in system and "(from `myquota`)" in system
    assert "## Python on the HPC: uv" in system and "UV_CACHE_DIR" in system

    await env.agent.handle(env.agent.general_session(), "train it")
    stdin = [c for c in env.cli_log() if c["prog"] == "claude"][-1]["stdin"]
    assert "<storage_warning>" in stdin and "/scratch: 425k of 500k files (85%)" in stdin

    n = len([m for m in env.notifier.sent if "💾" in m.text])
    await env.monitor.check_quota(force=True)
    assert len([m for m in env.notifier.sent if "💾" in m.text]) == n  # once, not on every check
    table.write_text(QUOTA)  # cleaned up: below the threshold again
    await env.monitor.check_quota(force=True)
    assert env.agent._storage_warning() == ""
    table.write_text(QUOTA.replace("301       500 ( 60%)", "450       500 ( 90%)"))
    await env.monitor.check_quota(force=True)
    assert len([m for m in env.notifier.sent if "💾" in m.text]) == n + 1  # filled up again: warned again


def test_no_quota_command(env):
    env.cfg.slurm.quota_command = ""
    system = env.agent.system_prompt()
    assert "not known to Overseer yet (`quota -s` may show them)" in system and "Python on the HPC: uv" in system


# ----------------------------------------------------------------------------- CLI errors and updates

async def test_api_error_is_an_error_and_pauses_autopilot(env):
    """An API error Claude Code reports as its result (here: the CLI is too old for the model) is an error, not
    an answer - with a hint to /update - and an autopilot wake-up that hits it pauses the loop."""
    s = env.agent.general_session()
    assert not await env.agent.handle(s, "hi FORCE_APIERROR")
    msg = env.notifier.sent[-1].text
    assert msg.startswith("⚠️ claude: API Error: 400 Claude Code 2.1.197 does not support this model") and "/update" in msg
    assert not any(m.text.startswith("🤖") for m in env.notifier.sent)

    ap = await make_autopilot(env)
    env.store.add_watch(WatchRecord(session_id=s.id, kind="timer", due=time.time() - 1, note="FORCE_APIERROR"))
    await ap.check(s.id)
    await ap.drain()
    assert "Wake-ups paused" in env.notifier.sent[-1].text and ap.paused(s.id)


async def test_updating_the_clis(env, monkeypatch):
    from overseer.bot import Bot
    lines = await env.agent.update_clis()
    assert lines == ["⬆️ Claude Code updated: 2.1.197 (Claude Code) → 2.1.300 (Claude Code)",
                     "⬆️ Codex updated: codex-cli 0.150.0 → codex-cli 0.161.0"]
    assert await env.agent.update_clis() == ["✓ Claude Code 2.1.300 (Claude Code) is up to date",
                                             "✓ Codex codex-cli 0.161.0 is up to date"]
    Path(os.environ["FAKE_CLI_LOG"] + ".update-fail").touch()
    lines = await env.agent.update_clis()
    assert lines[0].startswith("⚠️ Claude Code: the update failed") and "could not reach the update server" in lines[0]
    Path(os.environ["FAKE_CLI_LOG"] + ".update-fail").unlink()

    # Not while the agent is working; and runs wait while an update is going on.
    s = env.agent.general_session()
    env.agent.updating = True
    run = asyncio.create_task(env.agent.handle(s, "hi"))
    await asyncio.sleep(0.3)
    assert not run.done() and not [c for c in env.cli_log() if c["argv"][:1] not in (["--version"], ["update"])]
    env.agent.updating = False
    await run

    bot = Bot(env.cfg, app=None, store=env.store, hpc=env.hpc, monitor=env.monitor, agent=env.agent,
              notifier=env.notifier)
    replies = []

    async def reply(update, text):
        replies.append(text)
    monkeypatch.setattr(bot, "reply", reply)
    monkeypatch.setattr(env.agent, "running_sessions", lambda: [s])
    await bot.cmd_update(None, NS(args=[]))
    assert "The agent is working" in replies[-1]
    assert await env.agent.update_clis() is None
    monkeypatch.setattr(env.agent, "running_sessions", lambda: [])
    await bot.cmd_update(None, NS(args=[]))
    assert replies[-2] == "⬆️ Updating the agent CLIs…" and "is up to date" in replies[-1]


async def test_job_arrays_are_one_job(env):
    """sbatch prints the array's id; SLURM lists its tasks (123_1, 123_[2-4]). The array is one job: running while
    any task is, ended with the worst task state once all are done - never "left the queue" while tasks run."""
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    proj = env.home / "proj"
    (proj / "slurm-6100_1.out").write_text("epoch 3 loss=0.4\n")
    task = {"name": "sweep", "workdir": str(proj), "stdout": str(proj / "slurm-6100_1.out")}
    env.set_jobs(**{"6100_1": {**task, "state": "RUNNING"}, "6100_[2-4]": {**task, "state": "PENDING"}})
    token = env.ctl.open(s.id)
    ok, text = await env.ctl.handle(token, ["watch", "6100_3"])  # a task id means its array
    assert ok and "w1" in text, text
    assert env.store.get_job("6100") and env.store.get_job("6100_1") is None
    for _ in range(12):  # many polls: the array id never shows up in squeue, but its tasks do
        await env.monitor.poll()
    await ap.drain()
    rec = env.store.get_job("6100")
    assert not rec.ended_at and rec.state == "RUNNING", rec
    assert not any("left the queue" in m.text for m in env.notifier.sent)
    assert not [j for j in env.store.recent_jobs(50) if "_" in j.job_id]  # tasks aren't jobs of their own
    status = await env.monitor.status_message(rec, next(j for j in await env.hpc.squeue() if j.job_id == "6100"))
    assert "📦 job array: 3 PENDING, 1 RUNNING" in status.text

    # All tasks end, one of them failed: the array ends FAILED, once, and wakes the conversation once.
    n = len([c for c in env.cli_log() if c["prog"] == "claude"])
    st = env.slurm()
    st["jobs"] = {k: v for k, v in st["jobs"].items() if not k.startswith("6100")}
    st["jobs"].update({f"6100_{i}": {**task, "state": "FAILED" if i == 3 else "COMPLETED",
                                     "exit": "1:0" if i == 3 else "0:0"} for i in range(1, 5)})
    (env.tmp / "slurm.json").write_text(json.dumps(st))
    await env.monitor.poll()
    await env.monitor.poll()
    await ap.drain()
    finals = [m.text for m in env.notifier.sent if "Job 6100" in m.text and "ended" in m.text]
    assert len(finals) == 1 and "FAILED" in finals[0] and "📦 job array: 3 COMPLETED, 1 FAILED" in finals[0], finals
    stdin = [c for c in env.cli_log() if c["prog"] == "claude"][n]["stdin"]
    assert "[Wake-up you set up]" in stdin and "job array of 4 tasks: 3 COMPLETED, 1 FAILED" in stdin


async def test_a_job_wrongly_taken_as_ended_can_be_tracked_again(env):
    await make_autopilot(env)
    s = env.agent.general_session()
    env.store.upsert_job(JobRecord(job_id="6200", name="sweep", state="UNKNOWN", ended_at=time.time(), ignored=1,
                                   by_agent=1, session_id=s.id))
    env.set_jobs(**{"6200_1": {"name": "sweep", "state": "RUNNING"}})
    token = env.ctl.open(s.id)
    ok, text = await env.ctl.handle(token, ["watch", "6200"])
    assert ok, text
    rec = env.store.get_job("6200")
    assert not rec.ended_at and not rec.ignored and rec.state == "RUNNING"
