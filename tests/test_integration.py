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
from overseer.store import Store

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
    cfg.agent.claude_code.pass_env.append("FAKE_CLI_LOG")
    cfg.agent.codex.pass_env.append("FAKE_CLI_LOG")
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
    for name in ("ssh", "scp", "sftp"):
        w = cfg.local.tmp("agent-bin", name)
        assert os.access(w, os.X_OK) and f"-F {cfg.agent_ssh_config}" in w.read_text()
    assert (cfg.ssh_dir.stat().st_mode & 0o777) == 0o700
    assert (cfg.ssh_config_file.stat().st_mode & 0o777) == 0o600


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
