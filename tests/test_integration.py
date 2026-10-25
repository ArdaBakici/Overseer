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

from overseer.core.agent import Agent
from overseer.core.config import Config, ConfigError
from overseer.hpc.diagnose import Reporter
from overseer.hpc.remote import HPC
from overseer.hpc.monitor import Monitor
from overseer.core.runners import Activity, AgentRunError, ClaudeCodeRunner, CodexRunner, display_command
from overseer.hpc.activity import JobActivity
from overseer.hpc.domain import HpcDomain
from overseer.hpc.ssh import SSHRunner
from overseer.hpc.sshlink import agent_ssh_config, overseer_ssh_config, write_configs
from overseer.core.notify import Outgoing
from overseer.core.store import JobRecord, MetricSpec, Store, WatchRecord

FAKE_SLURM = Path(__file__).parent / "fake_slurm"
FAKE_CLI = Path(__file__).parent / "fake_cli"


class LocalRunner(SSHRunner):
    def build_command(self, argv):
        return ["sh", "-c", self.remote_command(argv)]


class FakeNotifier:
    def __init__(self):
        self.sent = []
        self.cards = {}  # job id -> versions of its card: [(text, buttons)]
        self.live = {}  # handle -> list of versions of a self-updating message
        self.deleted = []

    async def send(self, msg):
        self.sent.append(msg)

    async def card(self, job_id, text, buttons):
        """A job's live card: every version of it, by job."""
        self.cards.setdefault(job_id, []).append((text, buttons))

    async def post_live(self, msg):
        handle = len(self.live) + 1
        self.live[handle] = [msg.text]
        return handle

    async def edit_live(self, handle, text):
        self.live[handle].append(text)

    async def delete_live(self, handle):
        self.deleted.append(handle)

    # messages edited later (a plan's card), and subagents' threads
    async def post(self, msg):
        self.sent.append(msg)
        return {"1": len(self.sent)}

    async def edit(self, refs, text):
        self.edits = getattr(self, "edits", []) + [(refs, text)]

    async def open_thread(self, session_id, name):
        self.threads = {**getattr(self, "threads", {}), session_id: name}

    async def close_thread(self, session_id, done_name):
        self.closed = getattr(self, "closed", []) + [session_id]
        return True

    def texts(self):
        return "\n".join(m.text for m in self.sent)

    def card_text(self, job_id) -> str:
        return self.cards[job_id][-1][0] if job_id in self.cards else ""


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
    domain = HpcDomain(cfg, store, notifier, hpc)
    agent = Agent(cfg, store, notifier, domain)
    reporter = Reporter(lambda system, prompt: agent.runner().text(system=system, prompt=prompt), True, False)
    monitor = Monitor(cfg, hpc, store, notifier, reporter)
    domain.monitor = monitor

    async def no_master(_cfg):
        return None

    monkeypatch.setattr("overseer.hpc.domain.ensure_master", no_master)

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


def test_app_builds_around_its_domain(env, monkeypatch):
    """`overseer run`: the platform wired around config.yaml's domain (hpc by default), the domain's commands
    registered next to the platform's; an unknown domain is refused."""
    from telegram.ext import CommandHandler
    from overseer.core.app import build
    from overseer.core.config import Config
    app = build(env.cfg)
    commands = {c for h in app.handlers[0] if isinstance(h, CommandHandler) for c in h.commands}
    assert {"ask", "new", "auto", "search", "dashboard"} <= commands  # the platform's
    assert {"jobs", "job", "log", "fix", "results", "cluster"} <= commands, commands  # the HPC's
    assert env.cfg.domain == "hpc" and env.cfg.ssh.hostname and env.cfg.ssh_dir.name == "ssh"
    with pytest.raises(ConfigError, match="unknown domain 'nope'"):
        Config(domain="nope")
    with pytest.raises(ConfigError, match="a lowercase name"):
        Config(domain="../x")


def test_load_config_resolves_relative_to_config_file(tmp_path):
    from overseer.core.config import load_config

    (tmp_path / "config.yaml").write_text("ssh: {hostname: login.x.edu}\ntelegram: {allowed_chat_ids: [1]}\n")
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.local.state() == (tmp_path / "state").resolve()
    assert cfg.ssh_key == (tmp_path / "state" / "ssh" / "id_ed25519").resolve()


def test_env_file(tmp_path, monkeypatch):
    from overseer.core.config import load_config, parse_env_file

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
    from overseer.hpc.compute import write_agent_tools
    from overseer.hpc.sshlink import _write
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
    from overseer.hpc.sshlink import key_permissions_ok, setup_key

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
    store.upsert_session(__import__("overseer.core.store", fromlist=["SessionRecord"]).SessionRecord(id="s1"))
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
    from overseer.hpc.activity import submitted_jobs
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
    a = JobActivity(on_job=lambda j, c: seen.append(j))
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
    from overseer.core.runners import Runner

    env.cfg.local.env_path.write_text("TELEGRAM_BOT_TOKEN=x\n")
    denied = Runner(env.cfg, env.agent.domain.sandbox()).deny_read()
    assert covered(env.cfg.ssh_dir, denied) and covered(env.cfg.local.state(), denied)
    assert str(env.cfg.local.env_path) in denied
    assert not any(Path(a) in Path(b).parents for a in denied for b in denied)  # no nested entries
    # With the key folder outside the state dir it is listed on its own.
    env.cfg.ssh.dir = str(env.tmp / "app" / "keys")
    (env.tmp / "app" / "keys").mkdir()
    assert str(env.cfg.ssh_dir) in Runner(env.cfg, env.agent.domain.sandbox()).deny_read()


def test_deny_list_uses_real_paths(env):
    """A socket reachable as /var/run/x and /run/x (/var/run -> /run) is listed once, by its real
    path: bubblewrap can't mask a path through a symlinked dir and the agent's shell wouldn't start."""
    import socket
    from overseer.core.runners import Runner

    run = env.tmp / "run"
    run.mkdir()
    (env.tmp / "var-run").symlink_to(run)
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(run / "d.sock"))
    try:
        env.cfg.agent.deny_read = [str(env.tmp / "var-run" / "d.sock"), str(run / "d.sock")]
        denied = Runner(env.cfg, env.agent.domain.sandbox()).deny_read()
        assert str(run / "d.sock") in denied and not any("var-run" in d for d in denied)
    finally:
        sock.close()


async def test_claude_runner_sandbox_and_parsing(env):
    r = ClaudeCodeRunner(env.cfg, env.agent.domain.sandbox())
    act = JobActivity()
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
    r = CodexRunner(env.cfg, env.agent.domain.sandbox())
    act = JobActivity()
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
    from overseer.core.runners import AgentRunError

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

    monkeypatch.setattr("overseer.hpc.sshlink.ensure_master", no_master)
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
    from overseer.core import live
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
    from overseer.core.live import Transcript, watch

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
    assert "⏳ pending" in env.notifier.card_text("77")  # its live card

    n = len(env.notifier.sent)
    env.set_jobs(**{"77": {**env.slurm()["jobs"]["77"], "state": "RUNNING"}})
    await env.monitor.poll()
    assert env.notifier.card_text("77").startswith("🏃 <b>77</b> train") and len(env.notifier.sent) == n  # edited
    assert ("📊 Status", "status:77") in env.notifier.cards["77"][-1][1][0]

    log.write_text("epoch 1/3 loss 1.0\nTraceback (most recent call last):\n  File \"train.py\", line 1\n"
                   "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4 GiB\n")
    env.set_jobs(**{"77": {**env.slurm()["jobs"]["77"], "state": "FAILED", "exit": "1:0"}})
    n = len(env.notifier.sent)
    await env.monitor.poll()
    # No separate report: the card's final version is the result, with what went wrong and a Fix button.
    card, buttons = env.notifier.cards["77"][-1]
    assert "FAILED after" in card and "⚠️ <b>GPU out of memory</b>" in card and "CUDA out of memory" in card, card
    assert ("🔧 Ask agent to fix", "fix:77") in buttons[0] and ("📊 Status", "status:77") not in buttons[0]
    assert len(env.notifier.sent) == n and not [c for c in env.cli_log() if "--output-format" in c["argv"]
                                                and "json" in c["argv"]]  # no LLM summary either
    rec = env.store.get_job("77")
    assert rec.ended_at and rec.batch_script.startswith("#!/bin/bash") and "GPU out of memory" in rec.report

    env.set_jobs(**{"78": {"name": "quick", "state": "COMPLETED", "workdir": proj, "stdout": str(log)}})
    await env.monitor.poll()
    card, buttons = env.notifier.cards["78"][-1]
    assert "<b>78</b> quick · finished" in card and "⚠️" not in card and ("🔧 Ask agent to fix", "fix:78") not in buttons[0]


async def test_monitor_fills_in_agent_submitted_jobs(env):
    s = env.agent.general_session()
    await env.agent.handle(s, "submit it")  # fake claude "submits" 4242
    proj = str(env.home / "proj")
    env.set_jobs(**{"4242": {"name": "fixed", "state": "RUNNING", "workdir": proj, "stdout": f"{proj}/slurm-4242.out"}})
    await env.monitor.poll()
    assert env.notifier.card_text("4242").startswith("🏃 <b>4242</b> fixed")
    rec = env.store.get_job("4242")
    assert rec.stdout == f"{proj}/slurm-4242.out" and rec.name == "fixed" and rec.session_id == s.id


async def test_monitor_track_agent_only(env):
    env.cfg.slurm.track = "agent"
    env.set_jobs(**{"90": {"name": "mine", "state": "RUNNING", "workdir": "/x"}})
    await env.monitor.poll()
    assert env.store.get_job("90") is None


# ----------------------------------------------------------------------------- bot

async def test_bot_routing_and_commands(env, monkeypatch):
    from overseer.core.bot import Bot

    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    started = []

    async def fake_run_agent(chat_id, session, text, files=None):
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
    from overseer.core.autopilot import Autopilot
    from overseer.core.ctl import ControlServer
    from overseer.core.projects import Projects
    ap = Autopilot(env.cfg, env.store, env.agent, env.notifier)  # (the domain links its job watches to the monitor)
    env.ctl = ControlServer(env.cfg, env.agent, Projects(env.cfg, env.store, env.agent), env.notifier)
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
    await asyncio.gather(*list(env.ctl._bg))
    card, buttons = env.notifier.cards[j1][-1]  # the job's card says it's watched, with a ✖ to cancel that
    w = env.store.watches(s.id)[0]
    assert "⏰ agent watching" in card and ("✖ watch", f"uw:{w.id}") in buttons[0]
    assert f"⏰ Waiting for: job {j1}" in env.notifier.sent[-1].text

    # Job 1 finishes: the conversation is woken with its report; no one-shot LLM summary.
    await run_job(env, j1)
    assert "finished" in env.notifier.card_text(j1)  # the card is the result: no report message
    assert not any(f"<b>{j1}</b>" in m.text and "finished" in m.text for m in env.notifier.sent)
    assert not any("Waking the agent" in m.text for m in env.notifier.sent)  # the agent's reply is the notice
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
    assert len(claude_calls(env)) == n
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
    from overseer.core.bot import Bot
    await make_autopilot(env)
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
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
    from overseer.core.bot import Bot
    await make_autopilot(env)
    s = env.agent.general_session()
    env.set_jobs(**{"5001": {"name": "train", "state": "RUNNING", "workdir": str(env.home / "proj")}})
    await env.agent.handle(s, "FAKE_OV ov watch 5001 --note 'compare the loss'\nFAKE_OV ov wake in 30m look at the eval")
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    w, timer = env.store.watches(s.id)
    await asyncio.gather(*list(env.ctl._bg))
    assert "⏰ agent watching" in env.notifier.card_text("5001")
    q = FakeQuery(f"uw:{w.id}")  # ✖ on the job's card: the card is updated, not replaced
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1)), None)
    assert not q.edits and env.store.get_watch(w.id).state == "cancelled"
    assert "agent watching" not in env.notifier.card_text("5001")
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1)), None)
    assert "already happened or was cancelled" in q.edits[-1][0]
    note = next(m for m in env.notifier.sent if "look again in 30m" in m.text)  # a timer: its own one-line notice
    assert note.silent and note.buttons == [[("✖ Cancel", f"uw:{timer.id}")]]
    q = FakeQuery(f"uw:{timer.id}")
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1)), None)
    assert "Cancelled" in q.edits[-1][0]
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
    assert "too long: condense" in await env.agent.domain.research_log("dreamer")


async def test_usage_report(env):
    from overseer.core.usage import usage_report
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
    from overseer.core.autopilot import human_interval, parse_duration
    assert parse_duration("2h") == 7200 and parse_duration("90m") == 5400 and parse_duration("1d") == 86400
    assert parse_duration("1.5") == 5400 and parse_duration("off") == 0 and parse_duration("soon") is None
    assert parse_duration("30 minutes") == 1800
    assert human_interval(5400) == "1h30" and human_interval(172800) == "2d" and human_interval(900) == "15m"


async def test_status_follows_the_training_log(env):
    """SLURM's stdout often only has a launcher preamble; the training output goes to a file it names."""
    from overseer.core.bot import Bot
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

    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    await bot.ext._send_status("18743802")
    text = env.notifier.sent[-1].text
    assert "🏃 <b>18743802</b> bench-dreamerv3-mixed-s0 · cn257\n⏱ 1:00:00 / 4:00:00 (25%, 3h00m left)" in text
    assert "📈 step 12k · score 3.1 · dyn 0.42" in text, text
    assert "🎮 GPU 0 A100-SXM4-80GB: 87% busy · 30.5/80 GB" in text
    assert "<code>train_18743802.log</code>, last lines:" in text and "Step 12000" in text
    assert "pygame" not in text and "UserWarning" not in text  # the preamble and harmless warnings stay out
    assert json.loads(env.store.get_job("18743802").info)["progress_log"] == str(train_log)

    # A real problem on stderr is flagged.
    stderr.write_text("Traceback (most recent call last):\n  File \"train.py\"\ntorch.OutOfMemoryError: CUDA out of memory.\n")
    await bot.ext._send_status("18743802")
    assert "⚠️ stderr: <b>" in env.notifier.sent[-1].text

    # 📜 Log shows the training log; SLURM's own files come along as attachments.
    await bot.ext._send_log("18743802", 200)
    msg = env.notifier.sent[-1]
    assert "train_18743802.log" in msg.text and "episode/score 3.1" in msg.text
    assert [name for name, _ in msg.files] == ["train_18743802.log", "job-18743802-stdout.log", "job-18743802-stderr.log"]

    # The final report reads the training log too (that's where a crash's traceback is).
    stderr.write_text("gymnasium/core.py:311: UserWarning: env.get_frame is deprecated\n")
    with train_log.open("a") as fh:
        fh.write("Traceback (most recent call last):\n  File \"dreamer.py\", line 9\nValueError: bad replay ratio\n")
    env.set_jobs(**{"18743802": {**job, "state": "FAILED", "exit": "1:0"}})
    await env.monitor.poll()
    card = env.notifier.card_text("18743802")
    assert "FAILED" in card and "ValueError: bad replay ratio" in card, card  # the decisive line, on the card


# ----------------------------------------------------------------------------- projects

def make_projects(env):
    from overseer.core.projects import Projects
    return Projects(env.cfg, env.store, env.agent)


PICKERS = []  # (text, buttons) the bot replied with directly


def topic_update(text, thread=None, reply_to=None, message_id=None):
    async def reply_text(text, reply_markup=None, **kw):
        rows = reply_markup.inline_keyboard if reply_markup else []
        PICKERS.append((text, [(b.text, b.callback_data) for row in rows for b in row]))
    msg = NS(text=text, reply_to_message=NS(message_id=reply_to) if reply_to else None,
             message_thread_id=thread, is_topic_message=thread is not None, reply_text=reply_text,
             **({"message_id": message_id} if message_id else {}))
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
    from overseer.core.config import ProjectConfig
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
    cards = {jid: versions[-1][1] for jid, versions in env.notifier.cards.items()}  # (buttons)
    assert ("📁 dreamer", "assign:303:dreamer") in [b for row in cards["303"] for b in row]
    assert all(not d.startswith("assign:") for row in cards["301"] for _, d in row)


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
        if kw.get("reply_parameters"):
            self.replies = (self.replies or {}) | {len(self.sent): kw["reply_parameters"].message_id}
        return NS(message_id=len(self.sent))

    async def send_chat_action(self, chat_id, action, message_thread_id=None, **kw):
        from telegram.error import BadRequest
        if message_thread_id in self.dead_threads:
            raise BadRequest("Message thread not found")
        return True

    replies: dict = None  # message number -> the message it replies to
    edits: list = None  # (chat_id, message_id, text)
    gone: set = None  # messages that can't be edited any more (deleted)

    async def edit_message_text(self, text, chat_id=None, message_id=None, **kw):
        from telegram.error import BadRequest
        if message_id in (self.gone or set()):
            raise BadRequest("Message to edit not found")
        if self.edits and self.edits[-1][1:] == (message_id, text):
            raise BadRequest("Message is not modified")
        self.edits = (self.edits or []) + [(chat_id, message_id, text)]
        return True

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
    from overseer.core.bot import TelegramNotifier
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
    from overseer.core.bot import Bot
    projects = make_projects(env)
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier, projects=projects)
    replies, started = [], []

    async def reply(update, text):
        replies.append(text)

    async def fake_run_agent(chat_id, session, text, files=None):
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
    await bot.ext.cmd_jobs(topic_update("/jobs", thread=77), NS(args=[]))
    assert "801" in replies[-1] and "802" not in replies[-1]
    await bot.ext.cmd_jobs(topic_update("/jobs"), NS(args=[]))
    assert "801" in replies[-1] and "📁chem" in replies[-1] and "802" in replies[-1]
    await bot.ext.cmd_assign(topic_update("/assign"), NS(args=["802", "dreamer"]))
    assert env.store.get_job("802").project == "dreamer" and "moved to project <code>dreamer</code>" in replies[-1]
    assert "moved here from the main chat" in env.notifier.sent[-1].text and env.notifier.sent[-1].job_id == "802"
    env.store.link_message(1, 555, "802", None)
    await bot.ext.cmd_assign(topic_update("/assign", reply_to=555), NS(args=["main"]))
    assert env.store.get_job("802").project is None
    await bot.ext.cmd_assign(topic_update("/assign", thread=77), NS(args=["802"]))
    assert env.store.get_job("802").project == "chem"
    await bot.ext._assign("802", "nope", None)
    assert "No active project" in replies[-1]
    s = env.agent.session_for_job("801")
    assert s.project == "chem"  # a job's conversation lives in the job's project
    env.agent.domain.move_job("801", "dreamer")
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
    from overseer.core.bot import Bot
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    started = []

    async def fake_run_agent(chat_id, session, text, files=None):
        started.append((session.id, text))

    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)
    env.store.upsert_job(JobRecord(job_id="901", name="dreamer", state="RUNNING", stdout="/s/slurm-901.out",
                                   info=json.dumps({"progress_log": "/s/train_901.log"})))
    await bot.ext._start_plot(1, "901")
    for t in list(bot._tasks):
        await t
    sid, text = started[-1]  # no plotting command yet: the agent writes one (and registers it)
    assert sid == env.agent.session_for_job("901").id
    assert "/s/train_901.log" in text and "ov plots set" in text and "PNG" in text
    assert "is plotting job" in env.notifier.sent[-1].text
    # With a plotting command: run at once, no model involved; the graphs go under the card.
    shown = []

    async def show_plots(job_id, images, repost=False):
        shown.append((job_id, [n for n, _ in images]))
    env.notifier.show_plots = show_plots
    env.store.set_metric_spec(MetricSpec(scope="job", key="901", plot=(
        "printf '\\x89PNG fake' > {out}/loss.png && printf '\\x89PNG fake' > {out}/score.png")))
    n = len(started)
    await bot.ext._start_plot(1, "901")
    assert shown == [("901", ["loss.png", "score.png"])] and len(started) == n
    assert env.monitor.cached_plots("901")[0] == ("loss.png", b"\x89PNG fake")


async def test_telegram_pictures_and_fallbacks(env):
    from telegram.error import BadRequest
    from overseer.core.bot import TelegramNotifier

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
    from overseer.core.bot import Bot, TelegramNotifier
    env.cfg.telegram.new_threads = True
    tg = FakeTelegram()
    renamed = []

    async def edit_forum_topic(chat_id, message_thread_id, name):
        renamed.append((message_thread_id, name))
    tg.edit_forum_topic = edit_forum_topic
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    projects = make_projects(env)
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=notifier, projects=projects)
    started = []

    async def fake_run_agent(chat_id, session, text, files=None):
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
    from overseer.core.bot import TelegramNotifier
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
    from overseer.core.bot import Bot, TelegramNotifier
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
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent, notifier=notifier)
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
    import overseer.core.runners as runners
    clock = [1000.0]
    monkeypatch.setattr(runners.time, "monotonic", lambda: clock[0])
    a = JobActivity(login_warn_seconds=120)
    for cmd, took in [("ssh hpc 'python make_plots.py'", 200), ("ssh hpc 'ls -la'", 2),
                      ("hpc-run --dir ~/proj -- python make_plots.py", 900), ("ssh hpc 'cd x && sbatch run.sh'", 300),
                      ("python local_thing.py", 500), ("/bin/bash -lc \"ssh hpc 'srun -p compute python x.py'\"", 400)]:
        a.command(cmd)
        clock[0] += took
        a.output(cmd, "")
    assert a.login_node == [("ssh hpc 'python make_plots.py'", 200)]


async def test_login_node_warning_in_summary(env):
    s = env.agent.general_session()
    a = JobActivity(login_warn_seconds=120)
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
    assert any("job 7777<" in t and "native7_cli.py submit" in t for t in announced), announced
    assert any("job 7778<" in t and "in the agent's reply" in t for t in announced), announced
    assert not any("job 1234" in t for t in announced)
    assert {j.job_id for j in env.store.session_jobs(s.id)} == {"7777", "7778"}
    reply = next(m.text for m in reversed(env.notifier.sent) if m.text.startswith("🤖"))
    assert "Agent submitted" not in reply


async def test_assign_links_a_job_to_the_conversation(env, monkeypatch):
    """A job the agent submitted unnoticed can be handed to its conversation: /assign <id> in the thread."""
    from overseer.core.bot import Bot
    ap = await make_autopilot(env)
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    replies = []

    async def reply(update, text):
        replies.append(text)
    monkeypatch.setattr(bot, "reply", reply)
    s = env.agent.new_conversation()
    env.store.bind_thread(1, 301, s.id)
    env.store.upsert_job(JobRecord(job_id="18746955", name="native7", state="RUNNING"))  # seen, but not linked
    await bot.ext.cmd_assign(topic_update("/assign", thread=301), NS(args=["18746955"]))
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
    assert "[Wake-up you set up]" in claude_calls(env)[-1]["stdin"] and "18746955" in claude_calls(env)[-1]["stdin"]


async def test_general_topic(env, monkeypatch):
    """With topics, anything outside every project lives in a "general" topic."""
    from overseer.core.bot import Bot, TelegramNotifier
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
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=notifier)
    started, replies = [], []

    async def fake_run_agent(chat_id, session, text, files=None):
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
    from overseer.core.ctl import ControlServer
    projects = make_projects(env)
    ctl = ControlServer(env.cfg, env.agent, projects, env.notifier)
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
    from overseer.core.ctl import ControlServer
    ctl = ControlServer(env.cfg, env.agent, make_projects(env), env.notifier)
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
    from overseer.core.bot import Bot
    env.agent.set_backend(backend)
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
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
    assert env.agent.domain.storage_warning() == ""
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
    from overseer.core.bot import Bot
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

    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
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
    assert "📦 array: 3 PENDING, 1 RUNNING" in status.text

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
    card = env.notifier.card_text("6100")
    assert "FAILED after" in card and "📦 array: 3 COMPLETED, 1 FAILED" in card, card
    stdin = [c for c in env.cli_log() if c["prog"] == "claude"][n]["stdin"]
    assert "[Wake-up you set up]" in stdin and "job array of 4 tasks: 3 COMPLETED, 1 FAILED" in stdin


def replace_jobs(env, **jobs):
    st = env.slurm()
    st["jobs"] = jobs
    (env.tmp / "slurm.json").write_text(json.dumps(st))


async def test_array_tasks_logs_are_their_own(env):
    """FB2: for an array's pending tasks SLURM puts NO_VAL (4294967294) where %a goes in the log path. The array's
    logs are its tasks' own (the running one on the card); at the end the agent gets every task's state and log, and
    all the tasks - also the ones that ended before sacct's recent window."""
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    logs = env.home / "proj" / "runs" / "slurm"
    logs.mkdir(parents=True)
    (logs / "ne-ratio-pair_8100_0.out").write_text("ratio 32: achievement_pct=41.5\n")
    (logs / "ne-ratio-pair_8100_1.out").write_text("ratio 128: achievement_pct=47.25\n")
    task = {"name": "ne-ratio-pair", "workdir": str(env.home / "proj"),
            "stdout": str(logs / "ne-ratio-pair_8100_%a.out")}
    env.set_jobs(**{"8100_[0-1]": {**task, "state": "PENDING"}})
    token = env.ctl.open(s.id)
    ok, text = await env.ctl.handle(token, ["watch", "8100"])
    assert ok, text
    await env.monitor.poll()
    rec = env.store.get_job("8100")
    assert rec.info_dict["stdout_tpl"] == str(logs / "ne-ratio-pair_8100_%a.out"), rec.info

    replace_jobs(env, **{"8100_0": {**task, "state": "RUNNING"}, "8100_[1]": {**task, "state": "PENDING"}})
    await env.monitor.poll()
    rec = env.store.get_job("8100")
    assert rec.stdout == str(logs / "ne-ratio-pair_8100_0.out") and "4294967294" not in rec.stderr, rec

    # Task 0 ended long ago (outside sacct's recent window, and scontrol forgot it); task 1 just now.
    n = len([c for c in env.cli_log() if c["prog"] == "claude"])
    replace_jobs(env, **{"8100_0": {**task, "state": "COMPLETED", "ended_long_ago": True, "scontrol_gone": True},
                    "8100_1": {**task, "state": "COMPLETED"}})
    await env.monitor.poll()
    await env.monitor.poll()
    await ap.drain()
    card = env.notifier.card_text("8100")
    assert "📦 array: 2 COMPLETED" in card, card
    stdin = [c for c in env.cli_log() if c["prog"] == "claude"][n]["stdin"]
    assert "job array of 2 tasks: 2 COMPLETED" in stdin, stdin
    assert "4294967294" not in stdin, stdin
    for i, value in ((0, "41.5"), (1, "47.25")):
        assert f"task {i}: COMPLETED {logs}/ne-ratio-pair_8100_{i}.out" in stdin, stdin
        assert value in stdin, stdin


async def test_array_records_with_no_val_logs_are_repaired(env):
    """Records stored with the NO_VAL path (before FB2 was fixed) get their tasks' logs at the end."""
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    proj = env.home / "proj"
    (proj / "pair_8200_1.out").write_text("Traceback (most recent call last):\nValueError: ratio must be > 0\n")
    (proj / "pair_8200_0.out").write_text("done: return_mean=3.5\n")
    env.store.upsert_job(JobRecord(job_id="8200", name="pair", state="RUNNING", first_seen=time.time(), by_agent=1,
                                   session_id=s.id, workdir=str(proj), stdout=str(proj / "pair_8200_4294967294.out"),
                                   stderr=str(proj / "pair_8200_4294967294.out")))
    token = env.ctl.open(s.id)
    assert (await env.ctl.handle(token, ["watch", "8200"]))[0]
    env.set_jobs(**{"8200_0": {"name": "pair", "state": "COMPLETED", "scontrol_gone": True},
                    "8200_1": {"name": "pair", "state": "FAILED", "exit": "1:0", "scontrol_gone": True}})
    n = len([c for c in env.cli_log() if c["prog"] == "claude"])
    await env.monitor.poll()
    await ap.drain()
    rec = env.store.get_job("8200")
    assert rec.stdout == str(proj / "pair_8200_1.out"), rec  # the failed task's
    card = env.notifier.card_text("8200")
    assert "FAILED after" in card and "ratio must be" in card, card
    stdin = [c for c in env.cli_log() if c["prog"] == "claude"][n]["stdin"]
    assert "ValueError: ratio must be > 0" in stdin and "return_mean=3.5" in stdin, stdin


async def test_old_records_of_array_tasks_become_the_array(env):
    """Records of single tasks (6300_2), stored before arrays were tracked as one job, never show up in squeue
    again: they become the array's record, silently, instead of "left the queue" for each task."""
    await make_autopilot(env)
    s = env.agent.general_session()
    for i in (2, 3):
        env.store.upsert_job(JobRecord(job_id=f"6300_{i}", name="mepka_ft", state="RUNNING", first_seen=time.time(),
                                       by_agent=1, session_id=s.id))
    env.store.add_watch(WatchRecord(session_id=s.id, kind="jobs", jobs="6300_2 6300_3"))
    env.set_jobs(**{"6300_2": {"name": "mepka_ft", "state": "RUNNING"},
                    "6300_3": {"name": "mepka_ft", "state": "RUNNING"}})
    for _ in range(12):
        await env.monitor.poll()
    assert not any("left the queue" in m.text for m in env.notifier.sent)
    assert not [j for j in env.store.active_jobs() if "_" in j.job_id]
    rec = env.store.get_job("6300")
    assert rec and rec.state == "RUNNING" and rec.by_agent and rec.session_id == s.id and not rec.ended_at, rec
    assert [w.jobs for w in env.store.watches(s.id)] == ["6300"]


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


# ----------------------------------------------------------------------------- live job cards, metrics, alerts

async def test_job_cards_in_jobs_topics(env):
    """Each job has one card in its project's jobs topic, edited in place; its report replies to it."""
    from overseer.core.bot import TelegramNotifier
    tg = FakeTelegram()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    env.monitor.notifier = notifier
    projects = make_projects(env)
    proj = env.home / "proj"
    projects.create("dreamer", str(proj))
    await notifier.ensure_jobs_topics(["dreamer"])
    assert [name for _, name, _ in tg.created] == ["📊 jobs", "📊 dreamer · jobs"]
    dreamer_jobs = notifier.jobs_thread(1, "dreamer")

    log = proj / "slurm-501.out"
    log.write_text("step 100 loss=0.9\n")
    job = {"name": "train", "state": "RUNNING", "workdir": str(proj), "stdout": str(log)}
    env.set_jobs(**{"501": job})
    await env.monitor.poll()
    card_msg = len(tg.sent)
    assert tg.sent[-1][1] == dreamer_jobs and tg.sent[-1][2].startswith("🏃 <b>501</b> train")
    n = len(tg.sent)
    log.write_text("step 200 loss=0.5\n")
    rec = env.store.get_job("501")
    rec.last_status_at = 0  # due for a refresh
    env.store.upsert_job(rec)
    await env.monitor.poll()
    assert len(tg.sent) == n and "loss 0.5" in tg.edits[-1][2] and tg.edits[-1][1] == card_msg  # edited in place

    tg.gone = {card_msg}  # the card was deleted: a new one is posted
    rec = env.store.get_job("501")
    rec.last_status_at = 0
    env.store.upsert_job(rec)
    await env.monitor.poll()
    assert len(tg.sent) == n + 1 and env.store.card_refs("501") == {"1": n + 1}

    env.set_jobs(**{"501": {**job, "state": "COMPLETED", "exit": "0:0"}})
    m = len(tg.sent)
    await env.monitor.poll()
    assert "finished" in tg.edits[-1][2] and tg.edits[-1][1] == n + 1  # the card shows the end, in place
    assert len(tg.sent) == m  # and nothing else is posted: no report under it


async def test_agent_defined_metrics_and_alerts(env, tmp_path):
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    proj = env.home / "proj"
    (proj / "metrics.jsonl").write_text('{"step": 1200000, "eval": {"score": 0.4}, "loss": 0.3}\n')
    log = proj / "slurm-601.out"
    log.write_text("crl/train_episode/score 0.5 / train/loss/value 0.82\n")
    job = {"name": "bench-dreamerv3-s1", "state": "RUNNING", "workdir": str(proj), "stdout": str(log),
           "elapsed": "3:00:00"}
    env.set_jobs(**{"601": job})
    await env.monitor.poll()
    token = env.ctl.open(s.id)
    ok, text = await env.ctl.handle(token, [
        "metrics", "set", "--pattern", "bench-dreamerv3-*", "--critical", "score=score",
        "--important", "train/loss/value=value-loss,loss", "--progress", "step/5000000",
        "--command", "tail -n1 {workdir}/metrics.jsonl", "--alert", "score < 1 after 2h",
        "--wake-on", "loss is nan"])
    assert ok and "cards updated: 601" in text, text
    await asyncio.gather(*list(env.ctl._bg))
    card = env.notifier.card_text("601")
    assert "step 1.2M / 5M (24%)" in card  # progress from the command's values
    assert "🎯 <b>score 0.5</b>" in card or "🎯 <b>score 0.4</b>" in card
    assert "📈 value-loss 0.82 · loss 0.3" in card, card
    ok, text = await env.ctl.handle(token, ["metrics", "show", "601"])
    assert ok and "metrics from pattern bench-dreamerv3-*" in text and "eval/score=0.4" in text and "  ok" in text
    ok, text = await env.ctl.handle(token, ["metrics", "set", "601", "--alert", "score is high"])
    assert not ok and "can't read the alert" in text

    # The alert fires once (it has run 3h > 2h and score < 1): you're pinged, under the card.
    rec = env.store.get_job("601")
    rec.last_status_at = 0
    env.store.upsert_job(rec)
    await env.monitor.poll()
    alerts = [m for m in env.notifier.sent if m.text.startswith("🚨")]
    assert len(alerts) == 1 and "score &lt; 1 after 2h" in alerts[0].text and not alerts[0].silent and alerts[0].jobs
    assert "🚨 score &lt; 1 after 2h" in env.notifier.card_text("601")
    rec = env.store.get_job("601")
    rec.last_status_at = 0
    env.store.upsert_job(rec)
    await env.monitor.poll()
    assert len([m for m in env.notifier.sent if m.text.startswith("🚨")]) == 1  # once per job

    # A --wake-on alert wakes the conversation that set it.
    (proj / "metrics.jsonl").write_text('{"step": 1300000, "loss": "nan"}\n')
    rec = env.store.get_job("601")
    rec.last_status_at = 0
    env.store.upsert_job(rec)
    await env.monitor.poll()
    await ap.drain()
    assert "[Alert you set up] Job 601" in claude_calls(env)[-1]["stdin"]
    assert "loss is nan" in claude_calls(env)[-1]["stdin"]

    # A broken command: the card says so and falls back to the log.
    await env.ctl.handle(token, ["metrics", "set", "601", "--command", "cat /nonexistent/metrics.json"])
    await asyncio.gather(*list(env.ctl._bg))
    assert "⚠️ metrics command: exit 1" in env.notifier.card_text("601")


async def test_idle_gpus_wake_the_watching_agent(env):
    """A running GPU job whose GPUs stay idle for idle_gpu_minutes: a 💤 line on its card, a message, and the
    conversation watching it woken - once, until the GPUs are busy and then idle again."""
    ap = await make_autopilot(env)
    env.cfg.slurm.idle_gpu_sample_minutes = 0  # sample at every poll
    s = env.agent.general_session()
    proj = env.home / "proj"
    (proj / "slurm-701.out").write_text("epoch 3 loss=0.4\nwaiting for rank 1...\n")
    job = {"name": "ddp-train", "state": "RUNNING", "workdir": str(proj), "stdout": str(proj / "slurm-701.out")}
    env.set_jobs(**{"701": job, "702": {**job, "name": "other"}})
    token = env.ctl.open(s.id)
    assert (await env.ctl.handle(token, ["watch", "701"]))[0]

    def gpu(jid, util, mem="40960 MiB"):
        st = env.slurm()
        st.setdefault("gpu_util", {})[jid] = [util, mem]
        (env.tmp / "slurm.json").write_text(json.dumps(st))

    async def poll():
        await env.monitor.poll()
        await env.monitor.drain()
        await ap.drain()

    def backdate(jid, minutes):  # as if the GPUs had been idle that long
        state = env.monitor.idle_state(jid)
        state["since"] -= minutes * 60
        env.store.set(f"gpuidle:{jid}", json.dumps(state))

    await poll()
    assert "701" in env.slurm()["gpu_samples"] and not env.monitor.idle_state("701")  # busy (87%)
    n = len(claude_calls(env))
    gpu("701", "0 %")
    gpu("702", "2 %")
    await poll()
    assert env.monitor.idle_state("701")["util"] == 0 and len(claude_calls(env)) == n  # not idle long enough yet
    backdate("701", 31)
    backdate("702", 31)
    await poll()
    idle = {m.text.split("</b>")[0][-3:]: m for m in env.notifier.sent if m.text.startswith("💤")}
    assert set(idle) == {"701", "702"}, [m.text for m in env.notifier.sent]
    assert "waking the agent watching it" in idle["701"].text and idle["701"].jobs
    assert "no agent is watching it" in idle["702"].text
    assert "💤 <b>GPU idle</b> for 31m" in env.notifier.card_text("701"), env.notifier.card_text("701")
    calls = claude_calls(env)[n:]
    assert len(calls) == 1 and "[Idle GPU] Job 701 (ddp-train)" in calls[0]["stdin"], calls
    assert "40.0 of 80 GB GPU memory in use" in calls[0]["stdin"] and "hang or deadlock" in calls[0]["stdin"]

    await poll()  # still idle: not again
    assert len([m for m in env.notifier.sent if m.text.startswith("💤")]) == 2 and len(claude_calls(env)) == n + 1
    gpu("701", "fail")  # a sample that fails says nothing
    await poll()
    assert env.monitor.idle_state("701").get("alerted")
    gpu("701", "95 %")  # busy again: the card loses its line, and a new idle spell is reported again
    await poll()
    assert not env.monitor.idle_state("701") and "💤" not in env.notifier.card_text("701")
    gpu("701", "1 %", "0 MiB")
    await poll()
    backdate("701", 45)
    await poll()
    calls = claude_calls(env)[n:]
    assert len(calls) == 2 and "running on the CPU" in calls[1]["stdin"], calls

    # One sample at most per job at a time, never of hpc-run's worker; none when it's off.
    env.cfg.slurm.idle_gpu_minutes = 0
    before = len(env.slurm()["gpu_samples"])
    await poll()
    assert len(env.slurm()["gpu_samples"]) == before
    # The job ends: its idle state goes.
    st = env.slurm()
    st["jobs"]["701"]["state"] = "COMPLETED"
    (env.tmp / "slurm.json").write_text(json.dumps(st))
    await poll()
    assert env.store.get("gpuidle:701") is None


async def test_cluster_for_agents_and_queued_cards(env):
    """ov cluster: free GPUs per partition, time limits, the queued jobs' estimated starts and the fairshare; the
    estimated start is also on a queued job's card, until it starts."""
    from overseer.hpc import monitor as M
    await make_autopilot(env)
    s = env.agent.general_session()
    st = env.slurm()
    st["nodes"] = {
        "gpu1": {"gres": "gpu:a100:4(S:0-1)", "alloc_tres": "cpu=16,gres/gpu=3", "cpu_alloc": 16, "state": "MIXED",
                 "partitions": "nvidia"},
        "gpu2": {"gres": "gpu:a100:4", "state": "DOWN*", "partitions": "nvidia"},
        "cpu1": {"partitions": "compute", "cpus": 128}}
    st["partitions"] = {"compute*": {}, "nvidia": {"time": "4-00:00:00"}}
    st["fairshare"] = {"ab11578_acct": "0.42"}
    (env.tmp / "slurm.json").write_text(json.dumps(st))
    clock = (await env.hpc.clock())
    from datetime import datetime, timedelta
    soon = (datetime.fromisoformat(clock) + timedelta(hours=2, minutes=1)).isoformat(timespec="seconds")
    env.set_jobs(**{"801": {"name": "big-sweep", "state": "PENDING", "reason": "Resources", "est_start": soon}})
    await env.monitor.poll()
    card = env.notifier.card_text("801")
    assert f"🕐 starts {M.when(soon, clock)} (SLURM's estimate)" in card, card

    token = env.ctl.open(s.id)
    ok, text = await env.ctl.handle(token, ["cluster"])
    assert ok, text
    assert "nvidia: 1/2 nodes up · GPUs 1/8 free (a100), at most 1 on one node · CPUs 48/128 free" in text, text
    assert "max time 4-00:00:00" in text and "compute (default): 1/1 nodes up, 1 idle" in text, text
    assert "801 big-sweep: " in text and "(in ~2h0" in text and "waiting for Resources" in text, text
    assert "Fairshare: ab11578_acct 0.42" in text, text
    ok, text = await env.ctl.handle(token, ["cluster", "-p", "nvidia"])
    assert ok and "compute" not in text.split("Your queued")[0], text
    ok, text = await env.ctl.handle(token, ["cluster", "-p", "nope"])
    assert ok and "no partition 'nope'" in text

    # It starts: the estimate leaves the card.
    st = env.slurm()
    st["jobs"]["801"]["state"] = "RUNNING"
    (env.tmp / "slurm.json").write_text(json.dumps(st))
    await env.monitor.poll()
    assert "🕐" not in env.notifier.card_text("801") and env.store.get("eststart:801") is None


async def test_morning_digest(env):
    """Once a day (digest.time, local time) in the general topic: jobs, GPU time, results and findings, what each
    agent did and waits for, tokens, what waits for the researcher. /digest any time."""
    from datetime import datetime, timedelta
    from overseer.core.digest import Digest
    from overseer.core.live import Transcript
    from overseer.core.store import FeedbackRecord, FindingRecord, ResultRecord
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    clock = datetime.fromisoformat(await env.hpc.clock())
    iso = lambda dt: dt.isoformat(timespec="seconds")  # noqa: E731
    env.set_jobs(**{
        "901": {"name": "sweep-a", "state": "COMPLETED", "start": iso(clock - timedelta(hours=3)),
                "end": iso(clock - timedelta(hours=1)), "tres": "cpu=8,gres/gpu=2"},
        "902": {"name": "sweep-b", "state": "OUT_OF_MEMORY", "start": iso(clock - timedelta(hours=30)),
                "end": iso(clock - timedelta(hours=20)), "tres": "cpu=8,gres/gpu=4"},  # 4 GPUs x 4h in the window
        "903": {"name": "long-run", "state": "RUNNING", "start": iso(clock - timedelta(hours=2))}})
    now = time.time()
    for jid, name, state in (("901", "sweep-a", "COMPLETED"), ("902", "sweep-b", "OUT_OF_MEMORY")):
        env.store.upsert_job(JobRecord(job_id=jid, name=name, state=state, first_seen=now - 9000, ended_at=now - 3600,
                                       report="out of memory" if state != "COMPLETED" else "", session_id=s.id))
    env.store.upsert_job(JobRecord(job_id="903", name="long-run", state="RUNNING", first_seen=now - 7200))
    env.store.set("gpuidle:903", json.dumps({"since": now - 4000, "alerted": now - 100}))
    env.store.upsert_job(JobRecord(job_id="800", name="old", state="COMPLETED", first_seen=now - 9e5,
                                   ended_at=now - 8e5))
    env.store.save_result(ResultRecord(id="901", name="sweep-a", verdict="promising", conclusion="lr 3e-4 helps"))
    env.store.save_result(ResultRecord(id="902", name="sweep-b"))
    env.store.add_finding(FindingRecord(project=None, kind="dead-end", text="batch 4096 OOMs on a100-40g"))
    env.store.add_usage(session_id=s.id, project=None, backend="claude_code", model="m", origin="wakeup",
                        input=150_000, cached=0, output=5_000, cost=1.25)
    Transcript(env.cfg, s.id).write({"kind": "end", "status": "done", "text": "Sweep A beat the baseline by 4%."})
    token = env.ctl.open(s.id)
    assert (await env.ctl.handle(token, ["watch", "903", "--every", "1h"]))[0]
    env.store.add_feedback(FeedbackRecord(kind="bug", text="something is off with ov", session_id=s.id))

    d = Digest(env.cfg, env.store, env.notifier, env.agent)
    text = await d.build(now - 86400, now)
    assert text.startswith("☀️ <b>Digest</b>") and "since " in text, text
    assert "✅ 1 finished · ❌ 1 failed · 🏃 1 running" in text, text
    assert "<code>902</code> sweep-b · OUT_OF_MEMORY · out of memory" in text, text
    assert "<code>903</code> long-run · running · 💤 GPU idle" in text and "old" not in text, text
    assert "🎮 <b>GPU time</b>: 22.0 GPU-hours in 3 job(s)" in text, text  # 2x2h + 4x4h + 1x2h (default 1 GPU)
    assert "<b>Results</b>: 2 new (1 promising, 1 not judged yet)" in text and "lr 3e-4 helps" in text, text
    assert "🚫 batch 4096 OOMs" in text, text
    assert f"💬 main <code>{s.id}</code> · 1 run(s) (1 wake-ups) · ⏰ job 903" in text, text
    assert "<i>Sweep A beat the baseline by 4%.</i>" in text, text
    assert "🪙 <b>Tokens</b>: 155.0k · $1.25" in text and "🛠 1 open feedback item(s)" in text, text

    # When: once a day, after digest.time; then the next covers what happened since.
    env.cfg.digest.time = "00:00"
    assert d.due()
    await d.send()
    assert not d.due() and env.notifier.sent[-1].text.startswith("☀️") and env.notifier.sent[-1].session_id is None
    env.cfg.digest.time = "23:59"
    env.store.unset("digest_day")
    assert d.due() == (time.localtime().tm_hour == 23 and time.localtime().tm_min == 59)
    env.cfg.digest.enabled = False
    assert not d.due()

    # /digest and /cluster in Telegram.
    from overseer.core.bot import Bot
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    bot.digest = d
    replies = []

    async def reply(update, text):
        replies.append(text)
    bot.reply = reply
    await bot.cmd_digest(None, NS(args=[]))
    assert replies[-1].startswith("☀️ <b>Digest</b>") and "sweep-b" in replies[-1], replies[-1]
    st = env.slurm()
    st["nodes"] = {"g1": {"gres": "gpu:h100:8", "alloc_tres": "gres/gpu=5", "partitions": "nvidia"}}
    (env.tmp / "slurm.json").write_text(json.dumps(st))
    await bot.ext.cmd_cluster(None, NS(args=[]))
    assert "🖥 <b>Cluster now</b>" in replies[-1] and "<b>nvidia</b>: 1/1 nodes up" in replies[-1], replies[-1]
    assert "GPUs 3/8 free (h100)" in replies[-1], replies[-1]


async def test_search_past_work(env):
    """ov history search / show and /search: conversations (requests, replies, commands), results, findings, jobs;
    the transcripts are indexed as they grow."""
    from overseer.core.live import Transcript
    from overseer.core.store import FindingRecord, ResultRecord, SessionRecord
    await make_autopilot(env)
    s = env.agent.general_session()
    make_projects(env)
    env.store.upsert_session(SessionRecord(id="conv-ne", created=time.time(), project="dreamer"))
    old = time.time() - 40 * 86400
    t = Transcript(env.cfg, "conv-ne")
    t.write({"kind": "turn", "text": "try the NE ratio 64 with a smaller batch", "ts": old})
    t.write({"kind": "command", "command": "ssh hpc 'cd ne && sbatch --export=RATIO=64 run.sh'", "ts": old})
    t.write({"kind": "output", "output": "secret-output-not-indexed 64", "ts": old})
    t.write({"kind": "end", "status": "done", "text": "Ratio 64 diverged after 2k steps; 32 is stable.", "ts": old})
    env.store.save_result(ResultRecord(id="9100", project="dreamer", name="ne-ratio-64", verdict="broken",
                                       conclusion="diverges at ratio 64", metrics=json.dumps({"return": 1.5})))
    env.store.add_finding(FindingRecord(project="dreamer", kind="dead-end", text="NE ratio above 32 diverges"))
    env.store.upsert_job(JobRecord(job_id="9100", name="ne-ratio-64", state="FAILED", first_seen=old,
                                   report="NaN loss", project="dreamer"))
    token = env.ctl.open(s.id)

    ok, text = await env.ctl.handle(token, ["history", "search", "ratio", "64"])
    assert ok and "match(es)" in text, text
    assert "💬 conv-ne · 📁 dreamer" in text and "«Ratio» «64» diverged" in text, text
    assert "→ ov history show conv-ne --grep 'ratio 64'" in text
    assert "📊 result 9100 · 📁 dreamer" in text and "🧾 job 9100" in text, text
    assert "secret-output-not-indexed" not in text  # command output isn't indexed
    ok, text = await env.ctl.handle(token, ["history", "search", "diverges", "--kind", "finding"])
    assert ok and text.count("\n") == 1 and "🚫 F1 · 📁 dreamer: NE ratio above 32 «diverges»" in text, text
    ok, text = await env.ctl.handle(token, ["history", "search", "ratio", "--since", "7d"])
    assert ok and "conv-ne" not in text, text  # 40 days old
    ok, text = await env.ctl.handle(token, ["history", "search", "ratio", "--project", "main"])
    assert ok and text.startswith("nothing found"), text
    ok, text = await env.ctl.handle(token, ["history", "search", "--since", "7d"])
    assert not ok

    # The transcript grows: the new turn is found at the next search.
    Transcript(env.cfg, "conv-ne").write({"kind": "turn", "text": "now try ratio 48 with warmup"})
    Transcript(env.cfg, "conv-ne").write({"kind": "end", "status": "done", "text": "Ratio 48 with warmup is stable."})
    ok, text = await env.ctl.handle(token, ["history", "search", "warmup"])
    assert ok and "«warmup»" in text, text

    ok, text = await env.ctl.handle(token, ["history", "show", "conv-ne", "--grep", "ratio 64"])
    assert ok and "👤 try the NE ratio 64 with a smaller batch" in text and "  $ ssh hpc 'cd ne && sbatch" in text, text
    assert "🤖 Ratio 64 diverged" in text, text
    ok, text = await env.ctl.handle(token, ["history", "show", "conv-ne", "--last", "1"])
    assert ok and "turn(s)" in text and "ratio 48" in text and "smaller batch" not in text, text
    ok, text = await env.ctl.handle(token, ["history", "show", "../etc"])
    assert not ok and "no conversation" in text

    # /search in Telegram (in a project's topic: that project's work).
    from overseer.core.bot import Bot
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    replies = []

    async def reply(update, text):
        replies.append(text)
    bot.reply = reply
    await bot.cmd_search(None, NS(args=["ratio", "64"]))
    assert replies[-1].startswith("🔎 <b>ratio 64</b>") and "<b>Ratio</b> <b>64</b> diverged" in replies[-1], replies[-1]
    bot._scope = lambda update: "chemistry"
    await bot.cmd_search(None, NS(args=["ratio"]))
    assert "Nothing found" in replies[-1] and "in project chemistry" in replies[-1], replies[-1]
    await bot.cmd_search(None, NS(args=[]))
    assert "Usage: /search" in replies[-1]


async def test_dashboard(env):
    """The Mini App's server: the page for anyone (it holds no data), the data only with Telegram's signed launch
    data of an allowed user; read-only; graphs, logs, results, findings and the timeline."""
    from tests.test_parsing import sign_init_data
    from overseer.core.dashboard import Dashboard
    from overseer.core.store import FindingRecord, ResultRecord
    token = "123:abc"
    env.cfg.dashboard.port = 0
    proj = env.home / "proj"
    (proj / "slurm-951.out").write_text("epoch 9 return=3.5\n")
    env.set_jobs(**{"951": {"name": "ne-run", "state": "RUNNING", "workdir": str(proj),
                            "stdout": str(proj / "slurm-951.out"), "start": "2026-10-17T08:00:00"},
                    "952": {"name": "ne-old", "state": "FAILED", "start": "2026-10-16T08:00:00",
                            "end": "2026-10-16T09:00:00"}})
    await env.monitor.poll()
    env.store.set("cardview:951", json.dumps(["🏃 <b>951</b> ne-run · n1\n🎯 <b>return 3.5</b>", []]))  # (Telegram's)
    env.store.save_result(ResultRecord(id="951", name="ne-run", verdict="promising", metrics=json.dumps({"return": 3.5})))
    env.store.save_result(ResultRecord(id="r1", name="by eye", conclusion="looks <b>fine</b>"))
    env.store.add_finding(FindingRecord(project=None, kind="insight", text="warmup helps"))
    cache = env.cfg.local.state("plots", "951")
    cache.mkdir(parents=True)
    (cache / "00-curve.png").write_bytes(b"\x89PNG fake")
    d = Dashboard(env.cfg, env.store, token, env.agent.domain)
    await d.start()
    allowed = env.cfg.telegram.allowed_chat_ids[0]

    async def get(path, init=None, method="GET"):
        r, w = await asyncio.open_connection("127.0.0.1", d.port)
        hdr = f"X-Telegram-Init-Data: {init}\r\n" if init else ""
        w.write(f"{method} {path} HTTP/1.1\r\nHost: x\r\n{hdr}\r\n".encode())
        await w.drain()
        raw = await r.read()
        w.close()
        head, _, body = raw.partition(b"\r\n\r\n")
        return int(head.split()[1]), head.decode(), body

    try:
        good = sign_init_data(token, allowed)
        status, head, body = await get("/")
        assert status == 200 and b"<nav" in body and "Content-Security-Policy" in head and "no-store" in head
        assert "'unsafe-inline' https://telegram.org" not in head  # scripts: only its own file and Telegram's
        status, head, body = await get("/app.js")
        assert status == 200 and b"X-Telegram-Init-Data" in body
        assert (await get("/api/overview"))[0] == 401
        assert (await get("/api/overview", sign_init_data(token, allowed + 1)))[0] == 401
        assert (await get("/api/overview", sign_init_data("9:other", allowed)))[0] == 401
        assert (await get("/api/overview", good, method="POST"))[0] == 405

        status, _, body = await get("/api/overview", good)
        data = json.loads(body)
        assert status == 200 and {j["id"] for j in data["jobs"]} >= {"951"}, data
        j = next(j for j in data["jobs"] if j["id"] == "951")
        assert j["state"] == "RUNNING" and j["plots"] == 1 and j["card"] == "🏃 951 ne-run · n1\n🎯 return 3.5", j
        res = json.loads((await get("/api/results", good))[2])
        assert res["metric_keys"] == ["return"] and {r["id"] for r in res["rows"]} == {"951", "952", "r1"}
        assert json.loads((await get("/api/findings", good))[2])["rows"][0]["text"] == "warmup helps"
        tl = json.loads((await get("/api/timeline?days=30", good))[2])
        assert {x["id"]: x["end"] for x in tl["jobs"]} == {"951": "", "952": "2026-10-16T09:00:00"}, tl
        status, head, body = await get("/api/plot/951/0", good)
        assert status == 200 and "image/png" in head and body == b"\x89PNG fake"
        assert (await get("/api/plot/951/1", good))[0] == 404
        assert (await get("/api/plot/..%2F..%2Fetc/0", good))[0] == 404
        assert (await get("/api/plot/../0", good))[0] == 404
        log_ = json.loads((await get("/api/log/951?lines=20", good))[2])
        assert "epoch 9 return=3.5" in log_["text"] and log_["path"].endswith("slurm-951.out"), log_
        assert "doesn't track" in json.loads((await get("/api/log/999", good))[2])["error"]
        assert (await get("/nothing", good))[0] == 404
    finally:
        await d.stop()

    # /dashboard: the button when the address is set, how to set it up otherwise.
    from overseer.core.bot import Bot
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    replies = []

    async def reply(update, text):
        replies.append(text)
    bot.reply = reply
    await bot.cmd_dashboard(None, NS(args=[]))
    assert "tailscale serve --bg 8765" in replies[-1] and "funnel" in replies[-1]
    bot.dashboard = d
    env.cfg.dashboard.url = "https://homelab.tail1234.ts.net"
    sent = []

    async def reply_text(text, reply_markup=None, **kw):
        sent.append((text, reply_markup))
    await bot.cmd_dashboard(NS(effective_message=NS(reply_text=reply_text)), NS(args=[]))
    button = sent[-1][1].inline_keyboard[0][0]
    assert button.web_app.url == "https://homelab.tail1234.ts.net" and "Open the dashboard" in button.text


def test_stalled_alert(env):
    from overseer.hpc.metrics import check_alerts, parse_alert
    from overseer.core.store import MetricSpec
    spec = MetricSpec(scope="job", key="1", alerts=json.dumps([{"when": "step stalled 30m", "wake": False}]))
    rec = JobRecord(job_id="1")
    assert check_alerts(env.store, rec, spec, {"step": "100"}, 600) == []
    st = json.loads(env.store.get("mstate:1"))
    st["last"]["step"][1] -= 1900  # unchanged for 31 minutes
    env.store.set("mstate:1", json.dumps(st))
    (a, v), = check_alerts(env.store, rec, spec, {"step": "100"}, 2500)
    assert a.text == "step stalled 30m" and v == "100"
    assert parse_alert("x >= 3").op == ">="


async def test_unwatched_agent_job_points_to_the_jobs_topic(env):
    s = env.agent.general_session()
    await env.agent.handle(s, "submit it")  # fake claude "submits" 4242 (and doesn't watch it)
    proj = env.home / "proj"
    env.set_jobs(**{"4242": {"name": "fixed", "state": "COMPLETED", "exit": "0:0", "workdir": str(proj),
                             "stdout": f"{proj}/slurm-4242.out"}})
    await env.monitor.poll()
    pointer = next(m for m in env.notifier.sent if m.session_id == s.id and "Job 4242 fixed finished" in m.text)
    assert pointer.session_id == s.id and pointer.silent and "4242 fixed finished" in pointer.text


# ----------------------------------------------------------------------------- files sent to the agent

def file_update(message_id, *, photo=None, document=None, caption=None, thread=None, group=None, size=None):
    """A Telegram message with a picture (photo=bytes) or a file (document=(name, bytes))."""
    data = photo if photo is not None else document[1]

    async def get_file():
        async def download_to_drive(path):
            Path(path).write_bytes(data)
        return NS(download_to_drive=download_to_drive)
    media = NS(file_size=size or len(data), get_file=get_file,
               file_name=document[0] if document else None)
    msg = NS(message_id=message_id, photo=[NS(file_size=1, get_file=None), media] if photo is not None else [],
             document=media if document else None, caption=caption, media_group_id=group, reply_to_message=None,
             message_thread_id=thread, is_topic_message=thread is not None, text=None)
    return NS(effective_message=msg, effective_chat=NS(id=1))


async def test_files_sent_to_the_agent(env, monkeypatch):
    """A file goes with its caption, or with the next message; an album is one message; each lands in the agent's
    ./inbox and the agent is told where."""
    import overseer.core.bot as botmod
    from overseer.core.bot import Bot
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    started, replies = [], []

    async def fake_run_agent(chat_id, session, text, files=None):
        started.append((session.id, text, [n for n, _ in files or []], files))

    async def reply(update, text):
        replies.append(text)
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)
    monkeypatch.setattr(bot, "reply", reply)
    monkeypatch.setattr(botmod, "UPLOAD_WAIT", 0.3)
    monkeypatch.setattr(botmod, "ALBUM_WAIT", 0.05)
    main = env.agent.general_session().id

    # A screenshot, then a few words about it: one message.
    await bot.on_file(file_update(10, photo=b"\x89PNG one"), None)
    assert "📎 Got it" in replies[-1] and not started
    await bot.on_text(NS(effective_message=NS(text="why is this curve flat?", reply_to_message=None),
                         effective_chat=NS(id=1)), None)
    await asyncio.sleep(0)
    assert started[-1][:3] == (main, "why is this curve flat?", ["photo-10.jpg"]), started
    await asyncio.sleep(0.4)
    assert len(started) == 1  # the timer was cancelled

    # A file with a caption, and an album: one message each, without waiting for more.
    await bot.on_file(file_update(11, document=("my config.yaml", b"lr: 1"), caption="use this config"), None)
    await asyncio.sleep(0.1)
    assert started[-1][1:3] == ("use this config", ["my config.yaml"])
    await bot.on_file(file_update(12, photo=b"a", caption="compare these", group="g1"), None)
    await bot.on_file(file_update(13, photo=b"b", group="g1"), None)
    await asyncio.sleep(0.1)
    assert started[-1][1:3] == ("compare these", ["photo-12.jpg", "photo-13.jpg"])

    # Nothing written about it: after a while it goes on its own.
    await bot.on_file(file_update(14, photo=b"c"), None)
    await asyncio.sleep(0.4)
    assert started[-1][1:3] == ("", ["photo-14.jpg"])

    # Too big for the Bot API: refused, with what to do instead.
    n = len(started)
    await bot.on_file(file_update(15, document=("ckpt.pt", b"x"), size=botmod.MAX_DOWNLOAD + 1), None)
    assert "too big" in replies[-1] and len(started) == n

    # The agent gets the files in ./inbox and is told about them; names are cleaned, nothing is overwritten.
    s = env.agent.general_session()
    files = started[1][3]
    workdir = env.agent.runner(project=None).workdir(s.id)
    (workdir / "inbox").mkdir(parents=True, exist_ok=True)
    (workdir / "inbox" / "my_config.yaml").write_text("the agent's own")
    await env.agent.handle(s, "use this config", files=files)
    stdin = [c for c in env.cli_log() if c["prog"] == "claude"][-1]["stdin"]
    assert "use this config" in stdin and "./inbox/my_config-1.yaml (1 KB)" in stdin and "Read tool" in stdin, stdin
    assert (workdir / "inbox" / "my_config-1.yaml").read_text() == "lr: 1"
    assert (workdir / "inbox" / "my_config.yaml").read_text() == "the agent's own"


async def test_a_symlinked_inbox_is_not_followed(env, tmp_path):
    """The agent can write its working dir: an inbox it replaced with a link must not make Overseer write elsewhere."""
    from overseer.core.agent import deliver_uploads
    workdir, elsewhere = tmp_path / "work", tmp_path / "elsewhere"
    workdir.mkdir()
    elsewhere.mkdir()
    (workdir / "inbox").symlink_to(elsewhere)
    staged = tmp_path / "staged"
    staged.write_bytes(b"data")
    items = deliver_uploads(workdir, [("../../x.png", staged)])
    assert items == [("./inbox/x.png", 4)]
    assert not list(elsewhere.iterdir()) and (workdir / "inbox").is_dir() and not (workdir / "inbox").is_symlink()
    assert (workdir / "inbox" / "x.png").read_bytes() == b"data"


async def test_messages_outside_the_topics_are_refused(env, monkeypatch):
    """With new threads off (the default), a message written in "All" reaches no agent: the topic Telegram made for
    it is deleted and a note in "general" says where to write. Jobs topics talk to their project's agent."""
    from overseer.core.bot import Bot, TelegramNotifier
    tg = FakeTelegram()
    deleted = []

    async def delete_forum_topic(chat_id, message_thread_id):
        deleted.append(message_thread_id)
    tg.delete_forum_topic = delete_forum_topic
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    await notifier.ensure_general_topic()
    projects = make_projects(env)
    projects.create("dreamer", "/scratch/me/rf-dreamer")
    await notifier.ensure_jobs_topics(["dreamer"])
    await notifier.ensure_topic("dreamer")
    bot = Bot(env.cfg, app=NS(bot=tg), store=env.store, agent=env.agent,
              notifier=notifier, projects=projects)
    started = []

    async def fake_run_agent(chat_id, session, text, files=None):
        started.append((session.id, text))
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)

    async def drain():
        for t in list(bot._tasks):
            await t

    # Telegram made a topic of its own for the message: deleted, and the note (with the text) is in "general".
    await bot.on_text(topic_update("why is seed 1 at zero?", thread=901), None)
    await drain()
    assert not started and deleted == [901]
    gen = notifier.general_thread(1)
    note = [t for _, th, t in tg.sent if th == gen][-1]
    assert "outside the topics" in note and "why is seed 1 at zero?" in note, note
    # No topic at all: also refused.
    await bot.on_text(topic_update("hello"), None)
    await drain()
    assert not started and "outside the topics" in tg.sent[-1][2]
    # A picture sent there: refused before it's downloaded.
    await bot.on_file(file_update(30, photo=b"x", caption="look", thread=902), None)
    assert deleted[-1] == 902 and not bot._uploads

    # Topics still work: general, a project's topic, and its jobs topic.
    await bot.on_text(topic_update("quota?", thread=gen), None)
    await bot.on_text(topic_update("dreamer status", thread=projects.get("dreamer").thread_map["1"]), None)
    await bot.on_text(topic_update("how is the sweep?", thread=notifier.jobs_thread(1, "dreamer")), None)
    await drain()
    sessions = [env.agent.get(sid).project for sid, _ in started]
    assert [t for _, t in started] == ["quota?", "dreamer status", "how is the sweep?"]
    assert sessions == [None, "dreamer", "dreamer"]

    # Turned on: a new thread is a conversation of its own again.
    env.cfg.telegram.new_threads = True
    await bot.on_text(topic_update("a new idea", thread=903), None)
    await drain()
    assert any("Which project" in t for t, _ in PICKERS[-1:]) and 903 not in deleted


# ----------------------------------------------------------------------------- runs cut off by a restart

async def test_runs_cut_off_by_a_restart_are_offered_again(env, monkeypatch):
    """Overseer stops while the agent works (and a message waits its turn): the CLI is stopped with it, and at the
    next start each conversation is offered ▶️ Continue / ✖ Dismiss. Runs that ended, or that you /stop-ped, aren't."""
    from overseer.core.agent import Agent, CONTINUE_PROMPT
    from overseer.core.bot import Bot
    from overseer.core.runners import _RUNNING, kill_running
    s = env.agent.general_session()
    running = asyncio.create_task(env.agent.handle(s, "FORCE_SLEEP train the big model"))
    for _ in range(50):
        await asyncio.sleep(0.05)
        if _RUNNING:
            break
    assert _RUNNING, "the fake CLI never started"
    queued = asyncio.create_task(env.agent.handle(s, "then plot the curves"))
    await asyncio.sleep(0.1)
    # Overseer stops (post_stop): the CLI is killed, nothing is reported, the queued message doesn't start.
    n_sent, n_calls = len(env.notifier.sent), len(env.cli_log())
    env.agent.shutting_down = True
    assert kill_running() == 1
    with pytest.raises(AgentRunError, match="was killed"):
        await running
    assert await queued is False
    assert len(env.notifier.sent) == n_sent and len(env.cli_log()) == n_calls

    # The next start: one notice per cut-off run, in the conversation, with buttons.
    agent = Agent(env.cfg, env.store, env.notifier, HpcDomain(env.cfg, env.store, env.notifier, env.hpc))
    bot = Bot(env.cfg, app=None, store=env.store, agent=agent,
              notifier=env.notifier)
    started = []

    async def fake_run_agent(chat_id, session, text, files=None):
        started.append((session.id, text))
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)
    await bot.offer_cut_off_runs()
    notices = env.notifier.sent[n_sent:]
    assert len(notices) == 2 and all(m.session_id == s.id for m in notices)
    assert "was working on your message" in notices[0].text and "train the big model" in notices[0].text
    assert "had yet to start your message" in notices[1].text and "then plot the curves" in notices[1].text
    (cont, cont_data), (_, dismiss_data) = notices[0].buttons[0]
    assert cont == "▶️ Continue"
    edits = []

    async def press(data):
        async def edit_message_text(text, **kw):
            edits.append(text)
        q = NS(data=data, message=NS(message_thread_id=None), answer=_noop, edit_message_text=edit_message_text)
        await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1), effective_message=q.message), None)
        for t in list(bot._tasks):
            await t

    # ▶️ Continue: the agent is told it was cut off, with the request; a run that never started simply runs.
    await press(cont_data)
    assert started[-1] == (s.id, CONTINUE_PROMPT + "FORCE_SLEEP train the big model")
    assert edits[-1].startswith("▶️ Continuing")
    await press(notices[1].buttons[0][0][1])
    assert started[-1] == (s.id, "then plot the curves")
    await press(cont_data)  # pressed twice: nothing more
    assert len(started) == 2 and "already" in edits[-1]
    await press(dismiss_data)
    assert "already" in edits[-1]

    # A restart now offers nothing: those were handled. Nor do finished or stopped runs leave anything.
    s = agent.get(s.id)
    await agent.handle(s, "hello")
    stop = asyncio.create_task(agent.handle(s, "FORCE_SLEEP long one"))
    for _ in range(50):
        await asyncio.sleep(0.05)
        if agent.is_running(s) and _RUNNING:
            break
    assert await agent.stop(s.id)
    await stop
    assert "⏹ Stopped." in env.notifier.sent[-1].text
    n_sent = len(env.notifier.sent)
    await bot.offer_cut_off_runs()
    assert len(env.notifier.sent) == n_sent, [m.text for m in env.notifier.sent[n_sent:]]


async def test_a_wake_up_cut_off_by_a_restart(env):
    """A wake-up running when Overseer stops doesn't pause the conversation's wake-ups: it's offered again."""
    from overseer.core.bot import Bot
    from overseer.core.runners import _RUNNING, kill_running
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    w, _ = ap.add_timer(s, 60, note="FORCE_SLEEP compare the seeds")
    w.due = time.time() - 1
    env.store.update_watch(w)
    await ap.tick()
    for _ in range(50):
        await asyncio.sleep(0.05)
        if _RUNNING:
            break
    n_sent = len(env.notifier.sent)
    env.agent.shutting_down = True
    assert kill_running() == 1  # the wake-up's CLI was running
    await ap.drain()
    assert len(env.notifier.sent) == n_sent, [m.text for m in env.notifier.sent[n_sent:]]
    assert not env.store.get(f"wakeups_paused:{s.id}")
    env.agent.shutting_down = False
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    await bot.offer_cut_off_runs()
    notice = env.notifier.sent[-1]
    assert "was working on a wake-up it had set up" in notice.text and "[Wake-up you set up]" in notice.text
    assert notice.buttons[0][0][0] == "▶️ Continue"
    await env.ctl.stop()


# ----------------------------------------------------------------------------- planner mode: subagents

async def make_subagents(env):
    from overseer.core.subagents import Subagents
    ap = await make_autopilot(env)
    return Subagents(env.cfg, env.store, env.agent, env.notifier), ap


def claude_runs(env):
    return [c for c in env.cli_log() if c["prog"] == "claude" and "--output-format" in c["argv"]
            and c["argv"][c["argv"].index("--output-format") + 1] == "stream-json"]


def model_of(call) -> str:
    return call["argv"][call["argv"].index("--model") + 1] if "--model" in call["argv"] else ""


PLAN = {"goal": "beat the baseline score 5.2", "why": "independent schedules",
        "subagents": [{"name": "cosine", "task": "try a cosine schedule, 3 seeds"},
                      {"name": "warmup", "task": "linear warmup 1k vs 5k"},
                      {"name": "grid", "task": "launch the 6-run grid", "backend": "codex", "model": "gpt-6-luna"},
                      {"name": "plots", "task": "plot the curves"}]}


async def test_a_proposed_plan_approved_runs_subagents_and_wakes_the_planner(env):
    """The agent proposes planner mode; nothing starts until it's approved. Then the subagents run on their worker
    models in their own threads (3 at once, the 4th queued), and the planner, now on its planner model, is woken
    once with all their reports."""
    subs, ap = await make_subagents(env)
    s = env.agent.general_session()
    assert "ov propose" in env.agent.system_prompt(s.id) and "Planner mode:" not in env.agent.system_prompt(s.id)
    await env.agent.handle(s, f"FAKE_OV ov propose '{json.dumps(PLAN)}'")
    assert "✓ plan" in env.notifier.sent[-1].text, env.notifier.texts()[-2000:]
    card = next(m for m in env.notifier.sent if m.text.startswith("🧭"))
    assert "beat the baseline score 5.2" in card.text and "Opus 5.5" in card.text and "Sonnet 5.5" in card.text
    assert "grid</b> · gpt-6-luna (codex)" in card.text and "3 run at a time" in card.text
    (approve, pid), _ = card.buttons[0]
    assert approve == "✅ Approve" and not env.store.subagents()  # nothing started yet

    n_runs = len(claude_runs(env))
    await subs.approve(pid.split(":", 1)[1])
    assert env.agent.planner_on(s) and "approved" in env.notifier.edits[-1][1].lower()
    recs = {r.name: r for r in env.store.subagents(parent=s.id)}
    assert set(recs) == {"cosine", "warmup", "grid", "plots"}
    assert [r.name for r in env.store.subagents(states=("queued",))] == ["plots"]  # 3 at once
    assert set(env.notifier.threads.values()) >= {"🤖 cosine", "🤖 grid"}
    await subs.drain()
    await ap.drain()
    await subs.drain()
    recs = {r.name: r for r in env.store.subagents(parent=s.id)}
    assert all(r.state == "done" and r.report for r in recs.values()), {n: r.state for n, r in recs.items()}
    runs = claude_runs(env)[n_runs:]
    workers = [c for c in runs if "[Task from your planner" in c["stdin"]]
    assert sorted(model_of(c) for c in workers) == ["claude-sonnet-5-5"] * 3
    codex = [c for c in env.cli_log() if c["prog"] == "codex" and "[Task from your planner" in c["stdin"]]
    assert len(codex) == 1 and "gpt-6-luna" in json.dumps(codex[0]["argv"])
    planner = [c for c in runs if "[Subagents you started have finished]" in c["stdin"]]
    assert len(planner) == 1 and model_of(planner[0]) == "claude-opus-5-5", [c["stdin"][:80] for c in runs]
    stdin = planner[0]["stdin"]
    assert all(f"name='{n}'" in stdin for n in recs) and "Fixed the batch size" in stdin
    assert "The researcher approved your plan" in stdin  # the note, delivered with the reports
    # Each finished subagent's thread: its summary line, with Delete / Keep.
    ends = [m for m in env.notifier.sent if m.buttons and m.buttons[0][0][0] == "🗑 Delete thread"]
    assert len(ends) == 4 and "finished" in ends[0].text
    # The planner's instructions; a subagent's (no research log, no spawning).
    assert "Planner mode: you plan" in env.agent.system_prompt(s.id)
    w = env.agent.get(recs["cosine"].id)
    wp = env.agent.system_prompt(w.id)
    assert 'You are a subagent: "cosine"' in wp and "ov propose" not in wp and "Planner mode: you plan" not in wp
    await env.ctl.stop()


async def test_writing_to_the_agent_declines_its_plan(env, monkeypatch):
    """A message while a plan is open declines it and reaches the agent with what you wrote; ✖ declines it without
    comment; a newer plan replaces the old one."""
    from overseer.core.bot import Bot
    subs, ap = await make_subagents(env)
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    s = env.agent.general_session()
    p1 = await subs.propose(s, PLAN)
    p2 = await subs.propose(s, {**PLAN, "goal": "a smaller plan", "subagents": PLAN["subagents"][:1]})
    assert "Replaced by a newer plan" in env.notifier.edits[-1][1]
    assert (await subs.approve(p1["id"])).startswith("This plan was already")  # only the newest counts
    await bot.on_text(NS(effective_message=NS(text="use 2 seeds, not 3", reply_to_message=None),
                         effective_chat=NS(id=1)), None)
    for t in list(bot._tasks):
        await t
    assert "Declined</b>: you replied “use 2 seeds, not 3”" in env.notifier.edits[-1][1]
    stdin = claude_runs(env)[-1]["stdin"]
    assert "[The researcher declined your proposed plan (goal: a smaller plan) and replied:]\nuse 2 seeds" in stdin
    assert not env.agent.planner_on(s) and not env.store.subagents()
    assert (await subs.approve(p2["id"])).startswith("This plan was already")
    # ✖ No: the agent is told at its next run; nothing runs now.
    p3 = await subs.propose(s, PLAN)
    n = len(claude_runs(env))
    await subs.decline(p3["id"])
    assert "Declined." in env.notifier.edits[-1][1] and len(claude_runs(env)) == n
    await env.agent.handle(s, "ok, what next?")
    assert "declined your plan" in claude_runs(env)[-1]["stdin"]
    await env.ctl.stop()


async def test_subagent_rules(env):
    """What's refused, and what each command does."""
    from overseer.core.subagents import SubagentError
    subs, ap = await make_subagents(env)
    s = env.agent.general_session()
    with pytest.raises(SubagentError, match="planner mode is off"):
        await subs.spawn(s, "a", "task")
    with pytest.raises(SubagentError, match="isn't a subagent model"):
        await subs.propose(s, {**PLAN, "subagents": [{"name": "a", "task": "t", "model": "claude-fable-5-1"}]})
    with pytest.raises(SubagentError, match="the planner here is claude-opus-5-5"):
        await subs.propose(s, {**PLAN, "planner": "gpt-6-astra"})
    with pytest.raises(SubagentError, match="name"):
        await subs.propose(s, {**PLAN, "subagents": [{"name": "Bad Name", "task": "t"}]})
    env.agent._set_setting("subagents", "on", None)  # /subagents on (main chat)
    assert env.agent.planner_on(s) and env.agent.session_runner(s).model == "claude-opus-5-5"
    with pytest.raises(SubagentError, match="already on"):
        await subs.propose(s, PLAN)
    rec = await subs.spawn(s, "probe", "FORCE_SLEEP look around")
    w = env.agent.get(rec.id)
    with pytest.raises(SubagentError, match="already a subagent named"):
        await subs.spawn(s, "probe", "again")
    with pytest.raises(SubagentError, match="can't start subagents"):
        await subs.spawn(w, "nested", "task")
    for _ in range(50):
        await asyncio.sleep(0.05)
        if env.agent.is_running(w):
            break
    await subs.stop(w)  # by the planner: it isn't woken for it
    await subs.drain()
    assert w.rec.state == "stopped" and not [c for c in claude_runs(env) if "[Subagents you started" in c["stdin"]]
    # A subagent waiting for its jobs isn't done yet; when its watch fires and it answers, it is.
    rec = await subs.spawn(s, "watcher", "FAKE_LOOP train and watch")
    await subs.drain()
    w = env.agent.get(rec.id)
    assert w.rec.state == "waiting" and env.store.watches(w.id), w.rec.state
    env.agent._set_setting("subagents", None, None)
    assert not env.agent.planner_on(s) and env.agent.session_runner(s).model != "claude-opus-5-5"
    await env.ctl.stop()


async def test_subagent_threads_and_the_subagents_command(env, monkeypatch):
    from overseer.core.bot import Bot
    subs, ap = await make_subagents(env)
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    replies = []

    async def reply(update, text):
        replies.append(text)
    monkeypatch.setattr(bot, "reply", reply)
    await bot.cmd_subagents(None, NS(args=["on"]))
    assert "Planner mode is <b>on</b>" in replies[-1] and "claude-opus-5-5" in replies[-1]
    s = env.agent.general_session()
    rec = await subs.spawn(s, "cosine", "try cosine")
    await subs.drain()
    await bot.cmd_subagents(None, NS(args=[]))
    assert "cosine: ✅ done · Sonnet 5.5" in replies[-1], replies[-1]
    edits = []

    async def press(data):
        async def edit_message_text(text, **kw):
            edits.append(text)
        q = NS(data=data, message=NS(message_thread_id=None), answer=_noop, edit_message_text=edit_message_text,
               edit_message_reply_markup=_noop)
        await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1), effective_message=q.message), None)
    await press(f"subdel:{rec.id}")
    assert env.notifier.closed == [rec.id]
    await bot.cmd_subagents(None, NS(args=["workers", "codex", "gpt-6-luna"]))
    assert env.agent.worker_models("codex", None) == ["gpt-6-luna"]
    await bot.cmd_subagents(None, NS(args=["off"]))
    assert not env.agent.planner_on(s) and "off" in replies[-1]
    await env.ctl.stop()


# ----------------------------------------------------------------------------- results and findings

async def test_results_are_recorded_annotated_and_queried(env):
    """A finished job gets its row (state, runtime, key metrics by the agent's labels); the agent adds what it showed,
    records results without jobs, sorts and filters them, and gives them as Markdown or CSV."""
    await make_autopilot(env)
    projects = make_projects(env)
    proj = env.home / "proj"
    projects.create("dreamer", str(proj))
    s = env.agent.general_session(project="dreamer")
    token = env.ctl.open(s.id)

    async def ov(*argv):
        ok, text = await env.ctl.handle(token, list(argv))
        return ok, text

    for jid, score, loss in (("601", "4.1", "0.9"), ("602", "5.3", "0.4"), ("603", "3.0", "1.2")):
        (proj / f"slurm-{jid}.out").write_text(f"step 1000 episode/score={score} train/loss={loss}\n")
        env.store.upsert_job(JobRecord(job_id=jid, name=f"bench-{jid}", state="RUNNING", by_agent=1, session_id=s.id,
                                       project="dreamer", first_seen=time.time()))
    ok, text = await ov("result", "601", "--hypothesis", "baseline config")
    assert ok and "601" in text, text
    await ov("metrics", "set", "--project", "--critical", "episode/score=score", "--important", "train/loss=loss")
    env.set_jobs(**{jid: {"name": f"bench-{jid}", "state": "COMPLETED", "exit": "0:0", "workdir": str(proj),
                          "stdout": str(proj / f"slurm-{jid}.out")} for jid in ("601", "602", "603")})
    await env.monitor.poll()
    r = env.store.get_result("601")
    assert r and r.state == "COMPLETED" and r.success and r.project == "dreamer" and r.hypothesis == "baseline config"
    assert json.loads(r.metrics)["score"] == "4.1" and json.loads(r.metrics)["loss"] == "0.9", r.metrics

    assert (await ov("result", "601", "--verdict", "baseline", "--conclusion", "reference"))[0]
    assert (await ov("result", "602", "--verdict", "promising", "--conclusion", "cosine helps", "--tags", "lr,cosine"))[0]
    ok, text = await ov("result", "603", "--dead-end", "--conclusion", "fp16 diverges")
    assert ok and "dead end F1" in text, text
    ok, text = await ov("result", "add", "--name", "sample quality 602 vs 601", "--verdict", "promising",
                        "--metric", "artifacts=fewer", "--conclusion", "cleaner textures")
    assert ok and "result r1" in text, text
    assert not (await ov("result", "601", "--verdict", "great"))[0]  # not a verdict
    assert not (await ov("result", "999", "--verdict", "worse"))[0]  # not a tracked job
    assert not (await ov("result", "r9", "--verdict", "worse"))[0]  # no such result

    ok, table = await ov("results", "--sort", "score")
    lines = table.splitlines()
    assert ok and lines[0].split()[:4] == ["id", "name", "state", "verdict"], table
    assert [ln.split()[0] for ln in lines[1:]] == ["602", "601", "603", "r1"], table  # by score; no score last
    ok, table = await ov("results", "--sort", "loss", "--asc", "--columns", "id,loss")
    assert [ln.split() for ln in table.splitlines()[1:4]] == [["602", "0.4"], ["601", "0.9"], ["603", "1.2"]]
    ok, table = await ov("results", "--verdict", "promising", "--format", "md")
    assert table.startswith("| id |") and "602" in table and "r1" in table and "603" not in table, table
    ok, csv_text = await ov("results", "--format", "csv")
    header = csv_text.splitlines()[0]
    assert header.startswith("id,job_id,name,state") and "score" in header and "artifacts" in header, header
    ok, table = await ov("results", "--search", "cosine")
    assert "602" in table and "601" not in table
    ok, text = await ov("result", "show", "602")
    assert "verdict: ✅ promising" in text and "score=5.3" in text and "tags: lr,cosine" in text

    # Findings: recorded, listed, removed; dead ends are announced.
    ok, text = await ov("finding", "insight", "score tracks train loss closely")
    assert ok and "F2" in text and "Insight</b> recorded" in env.notifier.sent[-1].text
    ok, text = await ov("findings")
    assert "⛔ F1 dead-end: bench-603: fp16 diverges (job 603)" in text and "💡 F2 insight" in text, text
    assert (await ov("finding", "rm", "F2"))[0] and "F2" not in (await ov("findings"))[1]
    await env.ctl.stop()


async def test_agents_see_findings_and_stalls(env):
    """A new conversation and every wake-up get the project's findings and latest results; a run of results without
    gain is pointed out, and the agent's own results still waiting for a verdict are listed."""
    from overseer.hpc.results import context
    from overseer.core.store import FindingRecord, ResultRecord
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    env.store.add_finding(FindingRecord(project=None, kind="dead-end", text="lr 1e-3 diverges"))
    for i, v in enumerate(["promising", "no-gain", "worse", "no-gain", ""]):
        env.store.save_result(ResultRecord(id=f"70{i}", job_id=f"70{i}", name=f"run{i}", session_id=s.id,
                                           state="COMPLETED", ended=time.time() + i, verdict=v))
    ctx = context(env.store, None, s.id)
    assert "⛔ F1 dead-end: lr 1e-3 diverges" in ctx
    assert "Your last 3 results brought no gain (no-gain, worse, no-gain)" in ctx, ctx
    assert "Results of yours without a verdict yet: 704 (run4)" in ctx
    await env.agent.handle(s, "hello")  # a new CLI session: they're in its first prompt
    assert "lr 1e-3 diverges" in claude_runs(env)[-1]["stdin"]
    w, _ = ap.add_timer(s, 60, note="look again")
    w.due = time.time() - 1
    env.store.update_watch(w)
    await ap.tick()
    await ap.drain()
    stdin = claude_runs(env)[-1]["stdin"]
    assert "[Wake-up you set up]" in stdin and "brought no gain" in stdin and "ov result <job>" in stdin
    await env.ctl.stop()


async def test_results_and_findings_commands(env, monkeypatch):
    from overseer.core.bot import Bot
    from overseer.core.store import FindingRecord, ResultRecord
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent,
              notifier=env.notifier)
    replies = []

    async def reply(update, text):
        replies.append(text)
    monkeypatch.setattr(bot, "reply", reply)
    await bot.ext.cmd_results(None, NS(args=[]))
    assert "No results yet" in replies[-1]
    for jid, score, verdict in (("801", "2", "baseline"), ("802", "9", "promising"), ("803", "5", "no-gain")):
        env.store.save_result(ResultRecord(id=jid, job_id=jid, name=f"r{jid}", state="COMPLETED", ended=time.time(),
                                           verdict=verdict, metrics=json.dumps({"score": score})))
    await bot.ext.cmd_results(None, NS(args=["score"]))
    msg = env.notifier.sent[-1]
    assert "by score" in msg.text and msg.markdown.index("802") < msg.markdown.index("803") < msg.markdown.index("801")
    assert "| id | name | state | verdict | score | conclusion |" in msg.markdown
    await bot.ext.cmd_results(None, NS(args=["promising"]))
    assert "802" in env.notifier.sent[-1].markdown and "801" not in env.notifier.sent[-1].markdown
    await bot.ext.cmd_results(None, NS(args=["csv"]))
    (name, data), = env.notifier.sent[-1].files
    assert name == "results.csv" and data.decode().count("\n") == 4
    env.store.add_finding(FindingRecord(project=None, kind="insight", text="bigger batch helps"))
    await bot.ext.cmd_findings(None, NS(args=[]))
    assert "💡 F1 insight: bigger batch helps" in replies[-1]


# ----------------------------------------------------------------------------- graphs and the order of job cards

class MediaTelegram(FakeTelegram):
    """FakeTelegram plus photo cards: a photo with a caption and buttons, its caption or picture edited, deletions."""
    deleted: set = None
    media_edits: list = None  # (message id, file name, caption)
    caption_edits: list = None  # (message id, caption)
    markups: dict = None  # message id -> its buttons (as (text, data) rows)

    def _keep(self, mid, markup):
        rows = [[(b.text, b.callback_data) for b in row] for row in markup.inline_keyboard] if markup else []
        self.markups = (self.markups or {}) | {mid: rows}

    async def send_photo(self, chat_id, photo, filename, message_thread_id=None, **kw):
        self.sent.append((chat_id, message_thread_id, f"<photo {filename}> {kw.get('caption', '')}"))
        self._keep(len(self.sent), kw.get("reply_markup"))
        return NS(message_id=len(self.sent))

    async def edit_message_caption(self, chat_id=None, message_id=None, caption=None, **kw):
        self.caption_edits = (self.caption_edits or []) + [(message_id, caption)]
        self._keep(message_id, kw.get("reply_markup"))
        return True

    async def edit_message_media(self, chat_id=None, message_id=None, media=None, **kw):
        self.media_edits = (self.media_edits or []) + [(message_id, media.media.filename, media.caption)]
        self._keep(message_id, kw.get("reply_markup"))
        return True

    async def delete_message(self, chat_id, message_id):
        self.deleted = (self.deleted or set()) | {message_id}
        return True


async def test_graphs_carousel_in_cards_and_running_cards_at_the_bottom(env):
    """Once the agent's plotting command produced graphs, a job's card becomes a photo card: status refreshes edit its
    caption, new graphs replace the picture, ◀ ▶ page through them - all in place. When a job ends, running cards
    above it move to the bottom."""
    from overseer.core.bot import Bot, TelegramNotifier
    tg = MediaTelegram()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    await notifier.ensure_jobs_topics([])
    env.monitor.notifier = notifier
    proj = env.home / "proj"
    env.store.set_metric_spec(MetricSpec(scope="pattern", key="sweep-*", plot=(
        "printf '\\x89PNG {job_id}' > {out}/loss.png && printf '\\x89PNG' > {out}/score.png")))
    jobs = {jid: {"name": f"sweep-{jid}", "state": "RUNNING", "workdir": str(proj), "stdout": str(proj / "o.out")}
            for jid in ("901", "902")}
    (proj / "o.out").write_text("step 1 loss=1\n")

    async def poll(due=()):
        for jid in due:  # a status refresh due (and, unless just made, its graphs)
            rec = env.store.get_job(jid)
            rec.last_status_at = 0
            env.store.upsert_job(rec)
        await env.monitor.poll()
        await env.monitor.drain()

    env.set_jobs(**jobs)
    await poll()
    text_card = env.store.card_refs("901")["1"]
    assert text_card < env.store.card_refs("902")["1"]

    # The first graphs: the card is posted again as a photo card (graph + card text + buttons); the text card goes.
    await poll(due=["901"])
    card = env.store.card_refs("901")["1"]
    assert tg.sent[card - 1][2].startswith("<photo loss.png> 🏃 <b>901</b> sweep-901") and text_card in tg.deleted
    assert tg.markups[card][0] == [("◀", "pg:901:1"), ("📈 1/2", "noop:"), ("▶", "pg:901:1")], tg.markups[card]
    n = len(tg.sent)
    # A status refresh edits its caption; new graphs replace the picture: nothing new is posted.
    env.store.set("plot_at:901", str(time.time()))
    await poll(due=["901"])
    assert len(tg.sent) == n and tg.caption_edits[-1][0] == card and "🏃 <b>901</b>" in tg.caption_edits[-1][1]
    env.store.set("plot_at:901", "0")
    await poll(due=["901"])
    assert len(tg.sent) == n and tg.media_edits[-1][:2] == (card, "loss.png")
    # ▶: the next graph, in place.
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent, notifier=notifier)
    q = NS(data="pg:901:1", message=NS(message_thread_id=None), answer=_noop, edit_message_text=_noop)
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1), effective_message=q.message), None)
    assert tg.media_edits[-1][:2] == (card, "score.png") and tg.markups[card][0][1] == ("📈 2/2", "noop:")

    # 902 ends: its card becomes a photo card with its final graphs, at the bottom; 901 is still running and now
    # above it, so 901's photo card is posted again below it (still on its 2nd graph).
    env.set_jobs(**{"901": jobs["901"], "902": {**jobs["902"], "state": "COMPLETED", "exit": "0:0"}})
    await poll()
    c902 = env.store.card_refs("902")["1"]
    assert "finished" in tg.sent[c902 - 1][2] and tg.sent[c902 - 1][2].startswith("<photo")
    new = env.store.card_refs("901")["1"]
    assert new > c902 and card in tg.deleted and tg.sent[new - 1][2].startswith("<photo score.png> 🏃 <b>901</b>")
    # 901 ends: it's the lowest already; its caption becomes the final card, in place.
    env.set_jobs(**{"901": {**jobs["901"], "state": "COMPLETED", "exit": "0:0"}})
    await poll()
    assert env.store.card_refs("901")["1"] == new and any(m == new and "finished" in c for m, c in tg.caption_edits)


async def test_ov_plots(env):
    await make_autopilot(env)
    s = env.agent.general_session()
    env.store.upsert_job(JobRecord(job_id="950", name="eval-1", state="RUNNING", by_agent=1, session_id=s.id,
                                   workdir=str(env.home / "proj"), first_seen=time.time()))
    token = env.ctl.open(s.id)

    async def ov(*argv):
        return await env.ctl.handle(token, list(argv))
    ok, text = await ov("plots", "set", "--pattern", "eval-*", "--command", "python plot.py")
    assert not ok and "{out}" in text
    ok, text = await ov("plots", "set", "--pattern", "eval-*", "--command", "printf '\\x89PNG' > {out}/curve.png")
    assert ok and "ov plots test 950" in text, text
    ok, text = await ov("plots", "test", "950")
    assert ok and "1 graph(s)" in text and "./inbox/plot-950-curve.png" in text, text
    workdir = env.agent.session_runner(s).workdir(s.id)
    assert (workdir / "inbox" / "plot-950-curve.png").read_bytes() == b"\x89PNG"
    ok, text = await ov("plots", "show", "950")
    assert "graphs from pattern eval-*" in text
    await ov("metrics", "set", "--pattern", "eval-*", "--critical", "acc")
    await ov("metrics", "clear", "--pattern", "eval-*")  # the plotting command stays
    assert env.store.get_metric_spec("pattern", "eval-*").plot
    ok, text = await ov("plots", "set", "950", "--command", "exit 3; echo {out}")
    ok, text = await ov("plots", "test", "950")
    assert not ok and "failed (exit 3)" in text, text
    assert (await ov("plots", "clear", "950"))[0] and (await ov("plots", "clear", "--pattern", "eval-*"))[0]
    assert env.store.get_metric_spec("pattern", "eval-*") is None
    await env.ctl.stop()


# ----------------------------------------------------------------------------- feedback, its review, skills

async def make_feedback(env):
    from overseer.core.addons import Addons
    from overseer.core.feedback import Feedback
    await make_autopilot(env)
    return Feedback(env.cfg, env.store, env.agent, env.notifier), Addons(env.store, env.agent, env.notifier)


async def test_feedback_cards_votes_and_answers(env, monkeypatch):
    """An agent reports what limited it; the card goes to the feedback topic; another agent adds a +1; the researcher
    answers with a button, and the agents that reported it are told."""
    from overseer.core.bot import Bot, TelegramNotifier
    fb, _ = await make_feedback(env)
    tg = FakeTelegram()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    fb.notifier = notifier
    s1, s2 = env.agent.general_session(), env.agent.new_conversation()
    t1, t2 = env.ctl.open(s1.id), env.ctl.open(s2.id)
    ok, text = await env.ctl.handle(t1, ["feedback", "ov results can't filter by array task",
                                         "--context", "comparing 40 sweep tasks"])
    assert ok and "FB1" in text, text
    thread = notifier.feedback_thread(1)
    assert thread and ("🛠 feedback" in [n for _, n, _ in tg.created])
    card = [t for _, th, t in tg.sent if th == thread][-1]
    assert "🧱 <b>FB1</b> · limitation" in card and "while: comparing 40 sweep tasks" in card, card
    ok, text = await env.ctl.handle(t2, ["feedback", "+1", "FB1"])
    assert ok and "+1 on FB1" in text and "1 other conversation(s)" in tg.edits[-1][2]
    for i in range(2):
        assert (await env.ctl.handle(t1, ["feedback", f"another limitation number {i}", "--kind", "idea"]))[0]
    ok, text = await env.ctl.handle(t1, ["feedback", "a fourth one today is too many"])
    assert not ok and "already reported 3" in text
    assert not (await env.ctl.handle(t1, ["feedback", "x", "--kind", "rant"]))[0]
    ok, text = await env.ctl.handle(t1, ["feedback", "list"])
    assert text.splitlines()[0].startswith("🧱 FB1 limitation · +1"), text  # most +1s first

    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent, notifier=notifier)
    q = NS(data="fb:1:fixed", message=NS(message_thread_id=thread), answer=_noop, edit_message_text=_noop)
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1), effective_message=q.message), None)
    assert env.store.get_feedback(1).status == "fixed" and "✅ fixed" in tg.edits[-1][2]
    for s in (s1, s2):  # both conversations hear about it at their next run
        assert any("FB1" in n and "fixed it" in n for n in env.agent.take_notes(s.id))
    await env.ctl.stop()


async def test_feedback_review_runs_in_the_feedback_topic(env, monkeypatch):
    """/feedback review: the reviewer (planner model) gets the open items and Overseer's source path; writing in the
    feedback topic talks to it."""
    from overseer.core.bot import Bot, TelegramNotifier
    from overseer.core.feedback import SOURCE
    fb, _ = await make_feedback(env)
    tg = FakeTelegram()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    await notifier.ensure_feedback_topic()
    fb.notifier = notifier
    s = env.agent.general_session()
    await fb.report(s, "no way to see a job's GPU memory over time", "idea", "debugging an OOM")
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent, notifier=notifier)
    await bot.cmd_feedback(None, NS(args=["review", "GPU", "things"]))
    for t in list(bot._tasks):
        await t
    call = claude_runs(env)[-1]
    assert "[Feedback review]" in call["stdin"] and "Focus on: GPU things" in call["stdin"]
    assert "GPU memory over time" in call["stdin"] and str(SOURCE) in call["stdin"] and model_of(call) == "claude-opus-5-5"
    reviewer = fb.reviewer()
    assert reviewer.rec.role == "reviewer" and "feedback reviewer" in env.agent.system_prompt(reviewer.id)
    # Its reply lands in the feedback topic; writing there talks to it.
    thread = notifier.feedback_thread(1)
    assert env.store.session_thread(reviewer.id, 1) == thread
    assert bot._here(topic_update("why not a plot?", thread=thread)).id == reviewer.id
    await env.ctl.stop()


SKILL_TEXT = ("---\nname: cluster-long-jobs\ndescription: How to run jobs longer than 24 h on this cluster (qos long, "
              "checkpointing). Use for jobs over 24 h.\n---\nUse --qos long (max 7 days).\n")


def test_skill_files_are_checked():
    from overseer.core.addons import SkillError, parse_skill
    assert parse_skill(SKILL_TEXT) == ("cluster-long-jobs", "How to run jobs longer than 24 h on this cluster (qos long, "
                                                            "checkpointing). Use for jobs over 24 h.")
    for bad, why in [(SKILL_TEXT.replace("description:", "allowed-tools: Bash\ndescription:"), "only name and"),
                     (SKILL_TEXT.replace("cluster-long-jobs", "Cluster Jobs"), "lowercase"),
                     (SKILL_TEXT.replace("cluster-long-jobs", "overseer"), "Overseer's own"),
                     ("no frontmatter", "starts with frontmatter"),
                     (SKILL_TEXT.split("---\nUse")[0] + "---\n", "no body")]:
        with pytest.raises(SkillError, match=why):
            parse_skill(bad)


async def test_skills_are_proposed_approved_and_installed(env, tmp_path):
    """An agent proposes a skill; once approved it's in every run of its project (only), where the CLI finds it;
    removed, it's gone at the next run. Links in the run's folder are never followed."""
    from overseer.core.addons import write_skills
    _, addons = await make_feedback(env)
    projects = make_projects(env)
    projects.create("dreamer", str(env.home / "proj"))
    s = env.agent.general_session(project="dreamer")
    token = env.ctl.open(s.id)
    ok, text = await env.ctl.handle(token, ["skill", "propose", SKILL_TEXT])
    assert ok and "for project dreamer" in text, text
    card = env.notifier.sent[-1]
    assert "🧩 <b>Skill proposal: cluster-long-jobs</b> · for project dreamer" in card.text
    (install, data), _ = card.buttons[0]
    workdir = env.agent.session_runner(s).workdir(s.id)
    await env.agent.handle(s, "hello")
    assert not (workdir / ".claude" / "skills" / "cluster-long-jobs").exists()  # not before it's approved
    await addons.approve(int(data.split(":")[1]))
    assert "installed your skill cluster-long-jobs" in " ".join(env.agent.take_notes(s.id))
    await env.agent.handle(s, "hello again")
    installed = workdir / ".claude" / "skills" / "cluster-long-jobs" / "SKILL.md"
    assert installed.read_text() == SKILL_TEXT
    other = env.agent.general_session()  # the main chat: not its project
    await env.agent.handle(other, "hi")
    assert not (env.agent.session_runner(other).workdir(other.id) / ".claude" / "skills" / "cluster-long-jobs").exists()
    ok, text = await env.ctl.handle(token, ["skill", "list"])
    assert "cluster-long-jobs (project dreamer)" in text
    addons.remove(env.store.skills()[0].id)
    await env.agent.handle(s, "and again")
    assert not installed.parent.exists()

    # A link planted where a skill goes is replaced, never written through.
    outside = tmp_path / "outside"
    outside.mkdir()
    run = tmp_path / "run"
    (run / ".agents").mkdir(parents=True)
    (run / ".agents" / "skills").symlink_to(outside)
    write_skills(run, (".agents", "skills"), {"x": "data"})
    assert not list(outside.iterdir()) and (run / ".agents" / "skills" / "x" / "SKILL.md").read_text() == "data"
    await env.ctl.stop()


def test_card_captions_fit_telegram():
    """A photo card's caption holds 1024 characters: whole lines go from the end, GPU and alert lines first."""
    from overseer.core.bot import TelegramNotifier
    text = "\n".join(["🏃 <b>901</b> sweep", "🎯 <b>score 4.1</b>", "🎮 " + "g" * 600, "📈 " + "m" * 500, "🚨 x"])
    out = TelegramNotifier._caption(text)
    assert out.startswith("🏃 <b>901</b> sweep\n🎯 <b>score 4.1</b>") and "🎮" not in out and "🚨" not in out
    assert "📈" in out and len(out) <= 1000
    assert TelegramNotifier._caption("short\ncard") == "short\ncard"


async def test_cards_read_json_records_and_skip_parameter_counts(env):
    """FB1: a diagnostic job prints its modules' parameter counts and JSON records with the metrics the agent
    configured. The card shows those (not 'reward 656'), and so does its result."""
    from test_parsing import DIAG_LOG
    proj = env.home / "proj"
    (proj / "slurm-960.out").write_text(DIAG_LOG)
    env.store.set_metric_spec(MetricSpec(scope="job", key="960", critical="achievement_pct,return_mean"))
    job = {"name": "ne-diag", "workdir": str(proj), "stdout": str(proj / "slurm-960.out")}
    env.set_jobs(**{"960": {**job, "state": "RUNNING"}})
    await env.monitor.poll()
    env.set_jobs(**{"960": {**job, "state": "COMPLETED", "exit": "0:0"}})
    await env.monitor.poll()
    card = env.notifier.card_text("960")
    assert "🎯 <b>achievement_pct 7.9 · return_mean 2.31</b>" in card and "656" not in card, card
    assert json.loads(env.store.get_result("960").metrics)["achievement_pct"] == "7.9"


async def test_a_new_hpc_project_gets_its_jobs_topic(env):
    """The jobs topics are the HPC's: made through its project_created hook (not by the platform)."""
    from overseer.core.bot import TelegramNotifier
    tg = FakeTelegram()
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    make_projects(env).create("dreamer", "/scratch/me/rf-dreamer")
    await env.agent.domain.project_created(notifier, "dreamer")
    assert [name for _, name, _ in tg.created] == ["📊 jobs", "📊 dreamer · jobs"]


async def test_telegram_network_trouble_is_one_quiet_line(env, caplog):
    """A 502 from Telegram while polling (it retries by itself) is logged as one line, not a traceback each time;
    other errors keep their traceback."""
    import logging
    from telegram.error import BadRequest, Conflict, NetworkError, TimedOut
    from overseer.core.bot import Bot
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent, notifier=env.notifier, projects=make_projects(env))
    caplog.set_level(logging.WARNING, logger="overseer")
    for err in (NetworkError("Bad Gateway"), TimedOut(), NetworkError("Bad Gateway")):
        await bot.on_error(None, NS(error=err))
    lines = [r for r in caplog.records if "unreachable" in r.getMessage()]
    assert len(lines) == 1 and "Bad Gateway" in lines[0].getMessage() and not lines[0].exc_info
    bot._net_errors = (0.0, 2)  # five minutes later: one more line, saying how many were quiet
    await bot.on_error(None, NS(error=NetworkError("Bad Gateway")))
    assert "and 2 more" in caplog.records[-1].getMessage()
    await bot.on_error(None, NS(error=Conflict("terminated by other getUpdates request")))
    assert "same token" in caplog.records[-1].getMessage()
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        await bot.on_error(NS(), NS(error=e))
    assert caplog.records[-1].exc_info and caplog.records[-1].levelname == "ERROR"
    await bot.on_error(NS(), NS(error=BadRequest("Message is too long")))  # (not network trouble: a real error)
    assert caplog.records[-1].levelname == "ERROR"


async def test_slow_media_uploads_are_given_time_and_not_sent_twice(env, caplog):
    """A GIF upload Telegram takes long to answer (it converts it first) mustn't be cut off after 5 s, and if it
    still times out it isn't sent again (it may have arrived) - one quiet line instead of a traceback."""
    import logging
    from telegram.error import TimedOut
    from overseer.core.app import build
    from overseer.core.bot import TelegramNotifier
    app = build(env.cfg)
    req = app.bot._request[1]  # (the request used for everything but getUpdates)
    assert req.read_timeout >= 60 and req._media_write_timeout >= 120
    tg = FakeTelegram()
    tries = []

    async def send_animation(chat_id, **kw):
        tries.append(kw["animation"].filename)
        raise TimedOut("Timed out")
    tg.send_animation = send_animation
    n = TelegramNotifier(NS(bot=tg), [1], env.store)
    caplog.set_level(logging.WARNING, logger="overseer")
    await n.send(Outgoing("🤖 here", media=[("animation", "shaula.gif", b"GIF89a")]))
    assert tries == ["shaula.gif"] and tg.sent[-1][2] == "🤖 here"  # once; the message itself went out once
    line = [r for r in caplog.records if "shaula.gif" in r.getMessage()][-1]
    assert "may have arrived" in line.getMessage() and not line.exc_info


async def test_messages_wait_while_telegram_is_unreachable(env, caplog):
    """The network is down (DNS): messages aren't lost after three quick tries - they wait, and go out in order,
    marked as delayed, once Telegram can be reached; each failure is one line, not a traceback."""
    import logging
    from telegram.error import NetworkError, TimedOut
    from overseer.core.bot import TelegramNotifier
    tg = FakeTelegram()
    real_send = tg.send_message
    tg.down = True

    async def send_message(chat_id, text, **kw):
        if tg.down:
            raise NetworkError("httpx.ConnectError: [Errno -3] Temporary failure in name resolution")
        return await real_send(chat_id, text, **kw)
    tg.send_message = send_message
    n = TelegramNotifier(NS(bot=tg), [1], env.store)
    n.retry_delay = 0
    n.backlog_retry = 0.01
    caplog.set_level(logging.INFO, logger="overseer")
    await n.send(Outgoing("⏰ Meds 💊"))
    await n.send(Outgoing("⏰ Water"))
    assert tg.sent == [] and len(n._backlog) == 2
    assert not any(r.exc_info for r in caplog.records)  # (no tracebacks for a network that's down)
    tg.down = False
    await asyncio.sleep(0.05)  # (the backlog's retry)
    assert [t.split("\n")[0] for _, _, t in tg.sent] == ["⏰ Meds 💊", "⏰ Water"] and n._backlog == []
    assert "delayed: Telegram couldn't be reached at" in tg.sent[0][2]
    await n.send(Outgoing("🤖 later"))
    assert tg.sent[-1][2] == "🤖 later"  # (no note on messages that weren't delayed)
    # A timeout isn't retried (it may have arrived): no second copy, nothing queued.
    calls = []

    async def slow(chat_id, text, **kw):
        calls.append(text)
        raise TimedOut()
    tg.send_message = slow
    await n.send(Outgoing("🤖 slow"))
    assert calls == ["🤖 slow"] and n._backlog == []


async def test_a_wake_up_isnt_handed_over_again_as_if_new(env, monkeypatch):
    """FB5: a wake-up run that fails after doing its work (here: it watches a new job) isn't delivered again; one
    that fails before acting is - once the researcher writes - but says it came before and may be partly done."""
    ap = await make_autopilot(env)
    s = env.agent.general_session()
    # Fails without acting: put back, and the next delivery warns.
    w = env.store.add_watch(WatchRecord(session_id=s.id, kind="timer", due=time.time() - 1, note="FORCE_APIERROR"))
    await ap.check(s.id)
    await ap.drain()
    assert ap.paused(s.id) and env.store.get_watch(w.id).state == "active"
    w = env.store.get_watch(w.id)
    w.note = "compare the seeds"
    env.store.update_watch(w)
    await env.agent.handle(s, "hi")  # (they write: wake-ups resume, and what was put back fires)
    await ap.drain()
    again = claude_calls(env)[-1]["stdin"]
    assert f"[Again: w{w.id} came to you before, at" in again and "don't resubmit" in again
    assert "compare the seeds" in again and env.store.get_watch(w.id).state == "fired"
    # Fails after acting (it set a new wake-up in that run): not handed over again; the agent is told.
    w2 = env.store.add_watch(WatchRecord(session_id=s.id, kind="timer", due=time.time() - 1, note="look at 5001"))
    real = env.agent.handle

    async def acts_then_fails(sess, prompt, **kw):
        env.store.add_watch(WatchRecord(session_id=sess.id, kind="timer", due=time.time() + 3600, note="next run"))
        return False
    monkeypatch.setattr(env.agent, "handle", acts_then_fails)
    await ap.check(s.id)
    await ap.drain()
    monkeypatch.setattr(env.agent, "handle", real)
    assert env.store.get_watch(w2.id).state == "fired" and not env.store.get(f"rewake:{w2.id}")
    notes = env.agent.take_notes(s.id)
    assert any(f"run for w{w2.id} didn't finish" in n and "aren't handed to you again" in n for n in notes)
    await env.ctl.stop()


def test_changed_instructions_are_handed_to_a_resumed_conversation(env):
    """The CLIs keep a resumed conversation's first instructions: what changed goes with the next request, once."""
    from overseer.core.agent import INSTRUCTIONS_KEY, prompt_sections
    assert list(prompt_sections("You are X.\n\n## A\none\n\n## B\ntwo\n## A\nagain")) == ["(start)", "## A", "## B",
                                                                                           "## A (2)"]
    a = env.agent
    s = a.general_session()
    sysp = "You are X.\n\n## Rules\nbe brief\n\n## Project\nnone"
    known, update = a._instructions_update(s, sysp, None)  # a new CLI session: it gets them itself
    assert update == ""
    a._instructions_known(s, known, "cli-1")
    known, update = a._instructions_update(s, sysp, "cli-1")
    assert update == ""  # unchanged
    sysp2 = sysp.replace("be brief", "be very brief") + "\n\n## Output style\nadhd"
    known, update = a._instructions_update(s, sysp2, "cli-1")
    assert "<instructions_update>" in update and "## Rules\nbe very brief" in update and "## Output style" in update
    assert "## Project" not in update and "You are X." not in update
    a._instructions_known(s, known, "cli-1")
    assert a._instructions_update(s, sysp2, "cli-1")[1] == ""  # (handed over once)
    # A section that went away is named.
    known, update = a._instructions_update(s, sysp, "cli-1")
    assert "No longer part of your instructions: ## Output style" in update
    a._instructions_known(s, known, "cli-1")
    # A compaction may drop what was handed over: everything changed since the start goes again.
    a._instructions_known(s, a._instructions_update(s, sysp2, "cli-1")[0], "cli-1", compacted=True)
    assert "## Output style" in a._instructions_update(s, sysp2, "cli-1")[1]
    # A conversation from before Overseer kept track: all of it, once.
    env.store.unset(INSTRUCTIONS_KEY + s.id)
    update = a._instructions_update(s, sysp2, "cli-1")[1]
    assert "You are X." in update and "## Project" in update



async def test_an_existing_topic_is_never_deleted_and_general_finds_its_way_back(env, monkeypatch):
    """A message in an existing topic Overseer doesn't know (here: the general topic, after Overseer made another
    one) isn't refused by deleting the topic: an older general topic is general again, and an unknown one can be
    made the general topic with a button - the message then goes to the agent."""
    from telegram.error import BadRequest
    from overseer.core.bot import Bot, TelegramNotifier
    tg = FakeTelegram()
    deleted = []

    async def delete_forum_topic(chat_id, message_thread_id):
        deleted.append(message_thread_id)
    tg.delete_forum_topic = delete_forum_topic
    notifier = TelegramNotifier(NS(bot=tg), [1], env.store)
    await notifier.detect_topics()
    await notifier.ensure_general_topic()
    old = notifier.general_thread(1)
    bot = Bot(env.cfg, app=NS(bot=tg), store=env.store, agent=env.agent, notifier=notifier,
              projects=make_projects(env))
    started = []

    async def fake_run_agent(chat_id, session, text, files=None):
        started.append((session.id, text))
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)

    async def drain():
        for t in list(bot._tasks):
            await t
    # Overseer made a new general topic meanwhile (a send failed): writing in the old one still works, and makes
    # it the general one again.
    await notifier._create_general(1)
    assert notifier.general_thread(1) != old
    await bot.on_text(topic_update("add renew mac", thread=old, message_id=old + 500), None)
    await drain()
    assert started == [(env.agent.general_session().id, "add renew mac")] and notifier.general_thread(1) == old
    assert not deleted
    # An existing topic it never knew: not deleted; a button makes it the general topic, and the message goes on.
    PICKERS.clear()
    await bot.on_text(topic_update("hello there", thread=777, message_id=4000), None)
    await drain()
    assert not deleted and len(started) == 1
    text, buttons = PICKERS[-1]
    assert "isn't one of Overseer's" in text and buttons == [("💬 Make this the general topic", "gen:777")]
    q = FakeQuery("gen:777")
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1)), None)
    await drain()
    assert notifier.general_thread(1) == 777 and "general topic now" in q.edits[-1][0]
    assert started[-1] == (env.agent.general_session().id, "hello there")
    # A topic Telegram made for a message written in "All" is still removed.
    await bot.on_text(topic_update("in All", thread=901, message_id=902), None)
    assert deleted == [901]
    # Before replacing a topic after a failed send, Overseer checks it's really gone.
    async def action_ok(chat_id, action, message_thread_id=None):
        return True
    tg.send_chat_action = action_ok
    assert await notifier.topic_exists(1, 777)

    async def action_gone(chat_id, action, message_thread_id=None):
        raise BadRequest("Message thread not found")
    tg.send_chat_action = action_gone
    assert not await notifier.topic_exists(1, 777)
