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
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from fake_model_server import FakeModel
from overseer.agent import Agent
from overseer.config import Config
from overseer.hpc import HPC
from overseer.runners import _toml
from overseer.ssh import SSHRunner
from overseer.sshlink import setup_key, write_configs
from overseer.store import Store

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

    async def send(self, msg):
        self.sent.append(msg)

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
    try:
        yield cfg, remote, state, app
    finally:
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


@pytest.mark.parametrize("backend,cli_name", [("claude_code", "claude"), ("codex", "codex")])
async def test_agent_end_to_end(hpc, backend, cli_name):
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
    ]
    (app / ".env").write_text("TELEGRAM_BOT_TOKEN=999:super-secret-token\n")
    (app / ".env").chmod(0o600)
    with FakeModel(script) as fm:
        configure_backend(cfg, backend, cli, fm)
        store = Store(cfg.local.db_path)
        notes = Notes()
        agent = Agent(cfg, store, notes, HPC(SSHRunner(cfg), cfg.slurm.setup))
        agent.set_backend(backend)
        s = agent.general_session()
        await asyncio.wait_for(agent.handle(s, "Run the checks."), 240)
        texts = notes.texts()
        outs = fm.outputs
        detail = f"\n\nTELEGRAM:\n{texts}\n\nTOOL OUTPUTS:\n" + "\n====\n".join(outs)
        if os.environ.get("OVERSEER_E2E_VERBOSE"):
            print(detail)
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
        assert "refused by" not in texts, detail
        assert "Agent submitted job 5001" in texts, detail
        rec = store.get_job("5001")
        assert rec and rec.by_agent == 1 and rec.session_id == s.id
        assert json.loads(state.read_text())["jobs"]["5001"]["state"] == "PENDING"
        assert "DONE" in notes.sent[-1].text, detail
        assert store.get_session(s.id).cli_session_id
        # Overseer's instructions reached the model.
        tool_requests = [r for r in fm.requests if r["body"].get("tools")]
        assert "You are Overseer's agent" in json.dumps(tool_requests[0]["body"]), "system prompt missing"
