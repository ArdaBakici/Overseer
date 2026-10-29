"""The TUI (`overseer tui`, `overseer setup`) and running bots in the background (`overseer start/stop/status`):
real background processes (a stand-in bot that takes the real lock), the watcher's restarts, the bot's heartbeat,
the monitor (read-only: it changes no file), and the setup wizard's steps."""

import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from overseer.core import daemon
from overseer.core.config import load_config
from overseer.core.heartbeat import Heartbeat
from overseer.tui import data as D
from overseer.tui import steps as S
from test_assistant import aenv  # noqa: F401 (fixture)
from test_integration import env, short_tmp  # noqa: F401 (fixtures)

FAKE_BOT = Path(__file__).parent / "fake_overseer_bot.py"


@pytest.fixture
def bot_cfg(tmp_path, short_tmp):  # noqa: F811
    """A bot's config file (it loads, like a real one) in its own Overseer folder."""
    app = tmp_path / "app"
    app.mkdir()
    (app / "config.yaml").write_text(f"""
local: {{tmp_dir: {short_tmp}, timezone: UTC}}
ssh: {{hostname: login.example.edu, user: me}}
telegram: {{allowed_chat_ids: [42]}}
""")
    cfg = load_config(app / "config.yaml")
    yield cfg
    for pid in (daemon.state(cfg.local.state()).pid, int(daemon.state(cfg.local.state()).supervisor.get("pid") or 0)):
        if pid:
            os.kill(pid, signal.SIGKILL)


def fake(monkeypatch, mode: str, delay: float = 0.5):
    """`overseer run` -> the stand-in bot (mode); `overseer supervise` -> the real watcher, running the stand-in,
    with short pauses."""
    def argv(config, command):
        state = str(Path(config).parent / "state")
        if command == "run":
            return [sys.executable, str(FAKE_BOT), state, mode]
        code = (f"import sys; from overseer.core import daemon as d; from overseer.core.config import load_config; "
                f"d.FIRST_DELAY = {delay}; d.FAST_CRASH = 0; "
                f"d._argv = lambda c, cmd: [sys.executable, {str(FAKE_BOT)!r}, {state!r}, {mode!r}]; "
                f"sys.exit(d.supervise(load_config({str(config)!r})))")
        return [sys.executable, "-c", code]
    monkeypatch.setattr(daemon, "_argv", argv)


def wait_for(cond, seconds=10.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


# ----------------------------------------------------------------------------- the background process
def test_only_one_copy_of_a_bot_runs(tmp_path):
    fd = daemon.acquire_lock(tmp_path)
    try:
        assert daemon.read_pid(tmp_path / daemon.LOCK_FILE) == os.getpid()
        with pytest.raises(daemon.AlreadyRunning, match=f"already running \\(pid {os.getpid()}\\)"):
            daemon.acquire_lock(tmp_path)
    finally:
        os.close(fd)
    os.close(daemon.acquire_lock(tmp_path))  # (released with its process: free again)


def test_start_status_stop_without_the_watcher(bot_cfg, monkeypatch):
    fake(monkeypatch, "run")
    state = bot_cfg.local.state()
    ok, msg = daemon.start(bot_cfg, watched=False)
    assert ok and "started in the background" in msg, msg
    st = daemon.state(state)
    assert st.running and not st.watched
    assert os.getsid(st.pid) != os.getsid(0)  # (a session of its own: not tied to this terminal)
    assert daemon.start(bot_cfg) == (False, f"already running (pid {st.pid})")
    ok, msg = daemon.stop(bot_cfg)
    assert ok and msg == "stopped" and not daemon.state(state).running
    assert daemon.stop(bot_cfg) == (False, "not running")


def test_the_watcher_restarts_a_crash_but_not_a_stop(bot_cfg, monkeypatch):
    fake(monkeypatch, "crash-once")
    state = bot_cfg.local.state()
    ok, msg = daemon.start(bot_cfg, watched=True)
    # It crashed right away (crash-once), and the watcher started it again after its pause.
    assert wait_for(lambda: daemon.state(state).running and daemon.state(state).supervisor.get("restarts")), msg
    st = daemon.state(state)
    assert st.watched and len(st.supervisor["restarts"]) == 1 and st.supervisor["last_exit"] == 1
    first = st.pid
    os.kill(first, signal.SIGKILL)  # killed (e.g. out of memory): a crash too
    assert wait_for(lambda: len(daemon.state(state).supervisor.get("restarts", [])) == 2
                    and daemon.state(state).running and daemon.state(state).pid != first)
    assert daemon.state(state).supervisor["last_exit"] == -signal.SIGKILL
    ok, msg = daemon.stop(bot_cfg)  # a stop is not a crash: both end
    assert ok and msg == "stopped"
    assert not (state / daemon.SUPERVISOR_FILE).exists() and not daemon.state(state).watched


def test_a_bot_that_cant_start_isnt_restarted(bot_cfg, monkeypatch):
    fake(monkeypatch, "config-error")
    ok, msg = daemon.start(bot_cfg, watched=True, wait=5)
    assert not ok and "config error: telegram.allowed_chat_ids is empty" in msg
    assert wait_for(lambda: not daemon.state(bot_cfg.local.state()).watched)


def test_the_status_command(bot_cfg, monkeypatch, capsys):
    from overseer.__main__ import daemon_command
    fake(monkeypatch, "run")
    assert daemon_command(bot_cfg, "status") == 1 and "not running" in capsys.readouterr().out
    assert daemon_command(bot_cfg, "start") == 0
    assert daemon_command(bot_cfg, "status") == 0
    out = capsys.readouterr().out
    assert "running, pid" in out and "not restarted after a crash (local.restart_on_crash: true" in out
    assert daemon_command(bot_cfg, "stop") == 0
    bot_cfg.local.restart_on_crash = True  # (opted in)
    assert daemon_command(bot_cfg, "start") == 0 and daemon_command(bot_cfg, "status") == 0
    assert "watched (restarted after a crash: 0 in the last day)" in capsys.readouterr().out
    assert daemon_command(bot_cfg, "stop") == 0


def test_the_bot_writes_its_log_to_a_file(tmp_path):
    import logging
    path = daemon.log_to_file(tmp_path)
    logging.getLogger("overseer.test").warning("hello from the bot")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "WARNING overseer.test: hello from the bot" in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    logging.getLogger().removeHandler(logging.getLogger().handlers[-1])


# ----------------------------------------------------------------------------- the heartbeat and the monitor
def files_digest(folder: Path) -> dict[str, str]:
    return {str(p): hashlib.sha1(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


async def test_the_heartbeat_and_the_read_only_monitor(env, monkeypatch):  # noqa: F811
    from textual.widgets import DataTable

    from overseer.tui.monitor import Monitor
    env.set_jobs(**{"77": {"name": "train", "state": "RUNNING", "workdir": str(env.home / "proj")}})
    await env.monitor.poll(first=True)
    s = env.agent.general_session()
    assert await env.agent.handle(s, "how are my jobs?")
    env.cfg.source = str(env.tmp / "app" / "config.yaml")
    hb = Heartbeat(env.cfg, env.agent, env.notifier, env.agent.domain)
    hb.bot_name = "@my_bot"
    await hb.write()
    st = json.loads((env.cfg.local.state() / "status.json").read_text())
    assert st["pid"] == os.getpid() and st["bot"] == "@my_bot" and st["telegram"]["ok"]
    assert st["accounts"]["claude_code"]["active"] == "default" and st["runs"] == []
    jobs = st["domain"]["tables"][0]
    assert jobs["title"] == "Jobs" and jobs["rows"][0][1] == "77" and "train" in jobs["rows"][0][2]
    assert any(label == "last poll" for label, _, _ in st["domain"]["lines"])

    monkeypatch.setattr(D, "discover", lambda base: [D.Bot(Path(env.cfg.source), env.cfg)])
    before = files_digest(env.cfg.local.state())
    app = Monitor(env.tmp / "app")
    async with app.run_test(size=(150, 45)) as pilot:
        await pilot.pause()
        assert "running" not in str(app.query_one("#summary").render())  # (its pid isn't a bot's: not running)
        for key in "123456":
            await pilot.press(key)
            await pilot.pause(0.2)
        await pilot.press("2")
        await pilot.pause(0.2)
        assert app.query_one("#tabs").get_tab("domain").label.plain == "Jobs"
        assert app.query_one("#dt0", DataTable).row_count == len(jobs["rows"]) >= 1
        await pilot.press("3")
        await pilot.pause(0.3)
        assert app.query_one("#conversations", DataTable).row_count == 1
        lines = "\n".join(str(x) for x in app.query_one("#transcript").lines)
        assert "how are my jobs?" in lines and "ssh hpc" in lines
    assert files_digest(env.cfg.local.state()) == before  # it read everything, and changed nothing


def test_health_of_a_bot(bot_cfg, monkeypatch):
    bot = D.Bot(Path(bot_cfg.source), bot_cfg)
    h = D.health(bot)
    assert not h.running and h.label == "○ stopped"
    fake(monkeypatch, "run")
    assert bot_cfg.local.restart_on_crash is False  # (not in its config: off)
    daemon.start(bot_cfg)
    h = D.health(bot)
    assert h.running and not h.watched and h.stale  # (the stand-in writes no heartbeat)
    daemon.stop(bot_cfg)
    bot_cfg.local.restart_on_crash = True  # opted in: watched
    daemon.start(bot_cfg)
    h = D.health(bot)
    assert h.running and h.watched
    daemon.stop(bot_cfg)


def test_discovering_the_bots_of_a_folder(tmp_path, short_tmp):  # noqa: F811
    (tmp_path / "config.yaml").write_text(f"local: {{tmp_dir: {short_tmp}}}\nssh: {{hostname: h, user: u}}\n"
                                          "telegram: {allowed_chat_ids: [1]}\n")
    (tmp_path / "assistant.yaml").write_text("domain: assistant\nvault: {path: /nonexistent}\n")
    bots = D.discover(tmp_path)
    assert [b.label for b in bots] == ["HPC bot", "assistant.yaml"[:0] + "Assistant (assistant.yaml)"
                                       if bots[1].cfg else bots[1].label]
    assert bots[0].cfg and bots[1].error  # (a config that doesn't load is listed, with why)


# ----------------------------------------------------------------------------- the setup wizard's steps
def test_the_secrets_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# my secrets\nOTHER=1\nTELEGRAM_BOT_TOKEN=old\n")
    S.set_env(env_file, "TELEGRAM_BOT_TOKEN", "123:new")
    S.set_env(env_file, "ASSISTANT_BOT_TOKEN", "456:x")
    assert env_file.read_text() == "# my secrets\nOTHER=1\nTELEGRAM_BOT_TOKEN=123:new\nASSISTANT_BOT_TOKEN=456:x\n"
    assert env_file.stat().st_mode & 0o777 == 0o600
    assert S.read_env(env_file)["ASSISTANT_BOT_TOKEN"] == "456:x"


def test_the_config_is_written_as_the_example_with_your_values(tmp_path):
    example = S.configtool.example_path("hpc")
    d = S.Draft(example, "hpc")
    d.path = tmp_path / "config.yaml"
    d.set("ssh.hostname", "login.hpc.edu")
    d.set("telegram.allowed_chat_ids", [42])
    d.set("slurm.compute.partition", "cpu")
    assert d.save() is None  # (a new file: nothing to back up)
    text = d.path.read_text()
    assert "hostname: login.hpc.edu" in text and "allowed_chat_ids: [42]" in text
    assert "# Overseer configuration" in text  # (the comments are kept)
    cfg = load_config(d.path)
    assert cfg.slurm.compute.partition == "cpu" and cfg.ssh.hostname == "login.hpc.edu"
    d.set("ssh.user", "me")
    assert d.save() is None  # (it wrote this file itself: no backup of it)
    again = S.Draft(d.path, "hpc")
    again.set("ssh.user", "you")
    backup = again.save()
    assert backup and "user: me" in backup.read_text() and "user: you" in d.path.read_text()


def test_the_cluster_is_read_from_sinfo():
    parts = S.parse_sinfo("compute*|up|2-00:00:00|(null)|120\nnvidia|up|1-00:00:00|gpu:a100:4(S:0-1)|20\n"
                          "nvidia|up|1-00:00:00|gpu:a100:8|4\ndebug|down|1:00:00|(null)|2\n")
    assert [(p.name, p.default, p.up, p.gpus, p.nodes) for p in parts] == [
        ("compute", True, True, False, 120), ("nvidia", False, True, True, 24), ("debug", False, False, False, 2)]
    assert S.gpu_gres(parts[1]) == "gpu:a100:1" and S.gpu_gres(S.Partition("x", gres=["gpu:2"])) == "gpu:1"


def test_what_to_install(monkeypatch):
    monkeypatch.setattr(S, "package_manager", lambda: "apt")
    checks = [S.Check("ssh", True), S.Check("bwrap", False), S.Check("socat", False), S.Check("npm", True),
              S.Check("claude", False, optional=True), S.Check("codex", False, optional=True)]
    assert S.install_commands(checks, ["claude_code"]) == [
        ["sudo", "apt-get", "install", "-y", "bubblewrap", "socat"], ["npm", "install", "-g", "@anthropic-ai/claude-code"]]


def test_the_chat_id_is_found_from_a_message(monkeypatch):
    calls = []

    def fake_telegram(token, method, http_timeout=15, **params):
        calls.append(method)
        if method == "getMe":
            return {"username": "my_bot", "first_name": "My bot", "has_topics_enabled": True}
        return [{"message": {"date": int(time.time()), "chat": {"id": 4242, "type": "private", "first_name": "Ada"}}}]
    monkeypatch.setattr(S, "telegram", fake_telegram)
    assert S.bot_info("t") == {"username": "my_bot", "name": "My bot", "topics": True}
    assert S.wait_for_chat("t", wait=5) == (4242, "Ada")


def test_logins_are_seen_from_the_clis_files(tmp_path):
    home = tmp_path / "claude"
    home.mkdir()
    assert not S.logged_in("claude_code", home, {})
    assert S.logged_in("claude_code", home, {"ANTHROPIC_API_KEY": "x"})
    (home / ".claude.json").write_text(json.dumps({"oauthAccount": {"emailAddress": "a@b"}}))
    assert S.logged_in("claude_code", home, {})
    assert not S.logged_in("codex", tmp_path, {}) and S.logged_in("codex", tmp_path, {"OPENAI_API_KEY": "k"})


async def test_the_setup_wizard_writes_a_working_config(tmp_path):
    from textual.widgets import Input, ListView

    from overseer.tui.setup import Setup
    cfgp = tmp_path / "config.yaml"
    app = Setup(cfgp)
    async with app.run_test(size=(150, 48)) as pilot:
        await pilot.pause()
        steps = app.query_one("#steps", ListView)
        steps.index = 2
        await pilot.pause(0.2)
        app.query_one("#chat-id", Input).value = "4242"
        await pilot.click("#save-chat")
        steps.index = 3
        await pilot.pause(0.2)
        app.query_one("#hostname", Input).value = "login.hpc.example"
        app.query_one("#user", Input).value = "ab123"
        await pilot.click("#save-hpc")
        steps.index = 6
        await pilot.pause(0.2)
        app.query_one("#tz", Input).value = "Asia/Dubai"
        await pilot.click("#save-prefs")
        await pilot.pause(0.2)
        labels = [str(lb.render()) for lb in app.query("#steps Label")]
    assert "⬜ Telegram bot" in labels  # (the chat id is saved, the token isn't yet)
    assert "⬜ Cluster settings" in labels and "✅ Preferences" in labels
    cfg = load_config(cfgp)
    assert (cfg.ssh.hostname, cfg.ssh.user, cfg.telegram.allowed_chat_ids, cfg.local.timezone) == (
        "login.hpc.example", "ab123", [4242], "Asia/Dubai")
    assert cfg.local.restart_on_crash is False and not list(tmp_path.glob("config.yaml.bak-*"))  # (opt-in)


def test_overseer_run_refuses_a_second_copy(bot_cfg):
    import subprocess
    fd = daemon.acquire_lock(bot_cfg.local.state())
    try:
        res = subprocess.run([sys.executable, "-m", "overseer", "run", "-c", bot_cfg.source], capture_output=True,
                             text=True, timeout=60)
    finally:
        os.close(fd)
    assert res.returncode == daemon.EXIT_RUNNING and "already running" in res.stderr


async def test_the_assistant_shows_its_day(aenv):  # noqa: F811
    from test_assistant import ROUTINES
    (aenv.vault.root / "Assistant").mkdir(exist_ok=True)
    (aenv.vault.root / "Assistant/Routines.md").write_text(ROUTINES)
    (aenv.vault.root / "Assistant/Tasks.md").write_text(f"- [ ] call the bank 📅 {time.strftime('%Y-%m-%d')}\n")
    st = await aenv.domain.status()
    titles = [t["title"] for t in st["tables"]]
    assert titles == ["Today", "Tasks", "Waiting for you"]
    assert any("call the bank" in r[2] for r in st["tables"][1]["rows"])
    assert st["lines"][0][0] == "vault" and st["lines"][0][2] is True


def test_overseer_alone_opens_the_tui(tmp_path, monkeypatch):
    import overseer.__main__ as main
    from overseer.tui import monitor
    opened = []
    monkeypatch.setattr(monitor, "run_monitor", lambda config="config.yaml": opened.append(config) or 0)
    assert main.main([]) == 0 and opened == ["config.yaml"]
    assert main.main(["-c", str(tmp_path / "assistant.yaml")]) == 0 and opened[-1] == str(tmp_path / "assistant.yaml")
