"""The personal assistant domain: Obsidian Tasks, days and reminders, the vault's history and Undo, its config and
sandbox (kept out of the HPC bot's folders), and a whole turn through the platform with the fake agent CLI."""

import json
import os
import shutil
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

import pytest

from overseer.assistant import tasks as T
from overseer.assistant.domain import AssistantDomain
from overseer.assistant.reminders import Reminders
from overseer.assistant.vault import Vault, moment_to_strftime
from overseer.assistant.when import parse_day, parse_moment
from overseer.core.agent import Agent
from overseer.core.config import Config, ConfigError
from overseer.core.ctl import ControlServer
from overseer.core.projects import Projects
from overseer.core.runners import ClaudeCodeRunner, CodexRunner
from overseer.core.store import Store

FAKE_CLI = Path(__file__).parent / "fake_cli"
HAS_GIT = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not HAS_GIT, reason="git not installed")

TODAY = date(2026, 10, 17)  # a Saturday


class FakeNotifier:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    async def post_live(self, msg):
        return 1

    async def edit_live(self, handle, text):
        pass

    async def delete_live(self, handle):
        pass

    def texts(self):
        return "\n".join(m.text for m in self.sent)


# ---------------------------------------------------------------------------- tasks
def test_task_lines_parse_like_the_tasks_plugin():
    t = T.parse_line("- [ ] Submit the review 📅 2026-10-20 ⏳ 2026-10-18 ⏫ 🔁 every week #work #work/papers", "a.md", 3)
    assert (t.text, t.due, t.scheduled, t.priority, t.recurrence) == (
        "Submit the review #work #work/papers", "2026-10-20", "2026-10-18", "high", "every week")
    assert t.tags == ["#work", "#work/papers"] and t.open and t.when == "2026-10-20"
    assert T.parse_line("  * [x] Done thing ✅ 2026-10-01").status == "x"
    assert T.parse_line("1. [/] Half done").status == "/"
    assert T.parse_line("- not a task") is None and T.parse_line("- [ ]no space") is None


def vault_with(tmp_path, notes: dict[str, str]):
    cfg = assistant_cfg(tmp_path)
    for rel, text in notes.items():
        p = cfg.vault_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return Vault(cfg)


def test_selecting_tasks_by_day_tag_and_words(tmp_path):
    v = vault_with(tmp_path, {
        "Work.md": "- [ ] Overdue report 📅 2026-10-10 #work\n- [ ] Today call ⏳ 2026-10-17\n"
                   "- [ ] Next week 📅 2026-10-23 🔽\n- [ ] Someday\n- [x] Old ✅ 2026-10-01\n",
        "Home/Chores.md": "- [ ] Water plants 📅 2026-10-19 #home\n",
        ".obsidian/x.md": "- [ ] not a note\n",
    })
    items = T.scan(v)
    texts = lambda when, **kw: [t.text for t in T.select(items, when, today=TODAY, **kw)]  # noqa: E731
    assert texts("today") == ["Overdue report #work", "Today call"]
    assert texts("overdue") == ["Overdue report #work"]
    assert texts("week") == ["Overdue report #work", "Today call", "Water plants #home", "Next week"]
    assert texts("undated") == ["Someday"] and texts("done") == ["Old"]
    assert texts("open", tag="home") == ["Water plants #home"]
    assert texts("all", search="water") == ["Water plants #home"]
    with pytest.raises(T.TaskError):
        T.select(items, "soon")
    assert "2d overdue" not in T.describe(items[0], TODAY) and "7d overdue" in T.describe(items[0], TODAY)


def test_completing_tasks_and_recurrence(tmp_path):
    v = vault_with(tmp_path, {"T.md": "# Tasks\n- [ ] Pay rent 🔁 every month 📅 2026-10-31\n"
                                      "- [ ] Stretch 🔁 every 2 days when done ⏳ 2026-10-10\n- [ ] Once 📅 2026-10-17\n"})
    by = {t.text: t for t in T.scan(v)}
    t, remark = T.complete(v, by["Once"].id, TODAY)
    assert "- [x] Once 📅 2026-10-17 ✅ 2026-10-17" in (v.root / "T.md").read_text() and not remark
    T.complete(v, by["Pay rent"].id, TODAY)
    text = (v.root / "T.md").read_text()
    assert "- [ ] Pay rent 🔁 every month 📅 2026-11-30\n- [x] Pay rent 🔁 every month 📅 2026-10-31 ✅ 2026-10-17" in text
    _, remark = T.complete(v, by["Stretch"].id, TODAY)  # "when done": from today, not from the old date
    assert "⏳ 2026-10-19" in (v.root / "T.md").read_text() and remark == "next: 2026-10-19"
    with pytest.raises(T.TaskError, match="already done"):
        T.complete(v, [t for t in T.scan(v) if t.text == "Once"][0].id, TODAY)
    with pytest.raises(T.TaskError, match="no task"):
        T.complete(v, "zzzzzz", TODAY)


def test_steps_of_recurrences():
    assert T.step("every weekday", date(2026, 10, 16)) == date(2026, 10, 19)  # Friday -> Monday
    assert T.step("every monday, thursday", date(2026, 10, 17)) == date(2026, 10, 19)
    assert T.step("every month", date(2026, 1, 31)) == date(2026, 2, 28)
    assert T.step("every 2 weeks", TODAY) == date(2026, 10, 31)
    assert T.step("every year", date(2028, 2, 29)) == date(2029, 2, 28)
    assert T.step("on the second tuesday", TODAY) is None


def test_task_writes_stay_in_the_vault_and_check_the_line(tmp_path):
    v = vault_with(tmp_path, {"A.md": "- [ ] Thing\n"})
    outside = tmp_path / "outside.md"
    outside.write_text("- [ ] Secret\n")
    (v.root / "link.md").symlink_to(outside)
    (v.root / "dir").symlink_to(tmp_path)
    for bad in ("../outside.md", "/etc/passwd.md", "link.md", "dir/outside.md", "notes.txt"):
        with pytest.raises(T.TaskError):
            T.add(v, "- [ ] x", bad)
    assert [t.text for t in T.scan(v)] == ["Thing"]  # links aren't followed when reading either
    t = T.scan(v)[0]
    (v.root / "A.md").write_text("- [ ] Thing changed\n")
    with pytest.raises(T.TaskError):
        T.complete(v, t.id, TODAY)
    assert T.add(v, T.line_for("Buy milk", due="2026-10-18", priority="high")) == "Assistant/Tasks.md"
    assert (v.root / "Assistant/Tasks.md").read_text() == "- [ ] Buy milk ⏫ 📅 2026-10-18\n"
    with pytest.raises(T.TaskError):
        T.line_for("x", due="next week")


# ---------------------------------------------------------------------------- days and moments
def test_days_and_moments():
    assert parse_day("tomorrow", TODAY) == date(2026, 10, 18)
    assert parse_day("monday", TODAY) == date(2026, 10, 19)
    assert parse_day("sat", TODAY) == date(2026, 10, 24)  # today's weekday: a week from now
    assert parse_day("+3d", TODAY) == date(2026, 10, 20) and parse_day("2w", TODAY) == date(2026, 10, 31)
    with pytest.raises(T.TaskError):
        parse_day("soonish", TODAY)
    now = datetime(2026, 10, 17, 14, 5)
    assert parse_moment("in 2h", now) == datetime(2026, 10, 17, 16, 5)
    assert parse_moment("1h30m", now) == datetime(2026, 10, 17, 15, 35)
    assert parse_moment("17:30", now) == datetime(2026, 10, 17, 17, 30)
    assert parse_moment("9:00", now) == datetime(2026, 10, 18, 9, 0)  # passed today: tomorrow
    assert parse_moment("tomorrow 9am", now) == datetime(2026, 10, 18, 9, 0)
    assert parse_moment("friday at 6pm", now) == datetime(2026, 10, 23, 18, 0)
    assert parse_moment("2026-10-20", now) == datetime(2026, 10, 20, 9, 0)
    for bad in ("2026-10-01 10:00", "tomorrow 25:00", "whenever"):
        with pytest.raises(T.TaskError):
            parse_moment(bad, now)


def test_daily_note_names_follow_obsidian(tmp_path):
    assert moment_to_strftime("YYYY-MM-DD") == "%Y-%m-%d"
    assert moment_to_strftime("dddd, MMMM Do [week] YY") == "%A, %B %-d week %y"
    v = vault_with(tmp_path, {".obsidian/daily-notes.json": json.dumps({"folder": "Daily/", "format": "YYYY/MM-DD"})})
    day = datetime(2026, 10, 17, 12).timestamp()
    assert v.daily_note(day) == "Daily/2026/10-17.md"
    v.cfg.vault.daily_folder, v.cfg.vault.daily_format = "", "DD.MM.YYYY"
    assert v.daily_note(day) == "17.10.2026.md"


# ---------------------------------------------------------------------------- the vault's history
@needs_git
async def test_vault_history_and_undo_keep_your_edits(tmp_path):
    v = vault_with(tmp_path, {"Notes.md": "one\n", ".obsidian/workspace.json": "{}"})
    await v.init()
    assert not (v.root / ".git").exists() and (v.git_dir / "HEAD").exists()  # nothing extra is synced
    (v.root / "Notes.md").write_text("one\ntwo (the assistant)\n")
    (v.root / "New.md").write_text("new\n")
    (v.root / ".obsidian/workspace.json").write_text('{"moved": 1}')  # per-device UI state: not history
    turn = await v.snapshot("Assistant: a turn")
    assert sorted(turn.files) == [("A", "New.md"), ("M", "Notes.md")]
    assert await v.snapshot("nothing") is None
    (v.root / "Other.md").write_text("mine\n")  # your edit, after the turn
    ok, what = await v.undo(turn.commit[:12])
    assert ok and "Notes.md" in what
    assert (v.root / "Notes.md").read_text() == "one\n" and not (v.root / "New.md").exists()
    assert (v.root / "Other.md").read_text() == "mine\n"
    assert [c.message for c in await v.log()][:3] == ['Revert "Assistant: a turn"', "Your edits", "Assistant: a turn"]
    # A turn whose notes you changed since can't be undone automatically: nothing is touched.
    (v.root / "Notes.md").write_text("one\nthree\n")
    turn2 = await v.snapshot("Assistant: another")
    (v.root / "Notes.md").write_text("one\nthree, edited by you\n")
    ok, what = await v.undo(turn2.commit)
    assert not ok and "changed since" in what
    assert (v.root / "Notes.md").read_text() == "one\nthree, edited by you\n"
    assert (await v.undo("HEAD; rm -rf /"))[0] is False


# ---------------------------------------------------------------------------- config and sandbox
def assistant_cfg(tmp_path, **local):
    vault = tmp_path / "Vault"
    vault.mkdir(exist_ok=True)
    app = tmp_path / "app"
    app.mkdir(exist_ok=True)
    cfg = Config(domain="assistant")
    cfg.local.base_dir = str(app)
    cfg.local.state_dir = local.get("state_dir", "./state/assistant")
    cfg.local.tmp_dir = local.get("tmp_dir", tempfile.mkdtemp(prefix="ova-", dir="/tmp"))
    cfg.local.allowed_roots = [".", "/tmp", str(tmp_path)]
    cfg.telegram.token_env = "ASSISTANT_BOT_TOKEN"
    cfg.telegram.allowed_chat_ids = [1]
    cfg.vault.path = str(vault)
    cfg.agent.claude_code.command = str(FAKE_CLI / "claude")
    cfg.agent.codex.command = str(FAKE_CLI / "codex")
    cfg.agent.claude_code.pass_env += ["FAKE_CLI_LOG"]
    cfg.validate()
    return cfg


def test_assistant_config_is_its_own_bot(tmp_path):
    cfg = assistant_cfg(tmp_path)
    assert cfg.vault_dir == (tmp_path / "Vault").resolve()
    assert cfg.vault_git_dir == tmp_path / "app/state/assistant/vault.git"
    assert cfg.agent.claude_code.home == str(tmp_path / "app/state/assistant/claude")  # its own CLI logins
    assert cfg.agent.codex.home == str(tmp_path / "app/state/assistant/codex")
    cases = [
        (lambda c: setattr(c.vault, "path", ""), "vault.path is required"),
        (lambda c: setattr(c.vault, "path", "Vault"), "absolute"),
        (lambda c: setattr(c.vault, "path", str(tmp_path / "nope")), "isn't a folder"),
        (lambda c: setattr(c.vault, "path", str(tmp_path)), "not at Overseer's folder or above it"),
        (lambda c: setattr(c.vault, "path", str(tmp_path / "app")), "not at Overseer's folder"),
        (lambda c: setattr(c.local, "state_dir", "./state"), "its own"),
        (lambda c: setattr(c.local, "tmp_dir", "/tmp/overseer"), "its own"),
        (lambda c: setattr(c.telegram, "token_env", "TELEGRAM_BOT_TOKEN"), "ASSISTANT_BOT_TOKEN"),
    ]
    for change, message in cases:
        c = assistant_cfg(tmp_path)
        change(c)
        with pytest.raises(ConfigError, match=message):
            c.validate()


def test_each_bot_is_kept_out_of_the_others_folders(tmp_path):
    hpc_tmp = tempfile.mkdtemp(prefix="ovh-", dir="/tmp")
    try:
        hpc = Config()
        hpc.local.base_dir = str(tmp_path / "app")
        (tmp_path / "app").mkdir()
        hpc.local.tmp_dir = hpc_tmp
        hpc.local.register("hpc")
        Path(hpc_tmp, "ssh").mkdir()
        cfg = assistant_cfg(tmp_path)
        cfg.local.register("assistant")
        (tmp_path / "app/.env").write_text("ASSISTANT_BOT_TOKEN=x\n")
        (tmp_path / "app/.env.assistant").write_text("X=y\n")
        Path(cfg.local.tmp_dir).mkdir(exist_ok=True)
        cfg.vault_dir.joinpath(".obsidian").mkdir()
        domain = AssistantDomain(cfg, Store(cfg.local.db_path), FakeNotifier())
        runner = ClaudeCodeRunner(cfg, domain.sandbox())
        settings = runner.settings(cfg.local.tmp("agents", "x"))
        deny = settings["sandbox"]["filesystem"]["denyRead"]
        for p in (tmp_path / "app/state", Path(hpc_tmp).resolve(), tmp_path / "app/.env",
                  tmp_path / "app/.env.assistant"):
            assert str(p.resolve()) in deny, p  # the HPC's SSH master socket is in its tmp folder
        assert str(Path(cfg.local.tmp_dir).resolve()) not in deny  # its own: its scratch folders are there
        assert str(cfg.vault_dir) in settings["sandbox"]["filesystem"]["allowWrite"]
        assert settings["sandbox"]["filesystem"]["denyWrite"] == [str(cfg.vault_dir / ".obsidian")]
        assert f"Edit(/{cfg.vault_dir / '.obsidian'}/**)" in settings["permissions"]["deny"]
        assert str(cfg.vault_dir) in settings["permissions"]["additionalDirectories"]
        fs = " ".join(CodexRunner(cfg, domain.sandbox()).overrides(
            cfg.local.tmp("agents", "x"), system=None, writable=True, network=True))
        assert f'"{cfg.vault_dir}" = "write"' in fs and f'"{cfg.vault_dir / ".obsidian"}" = "read"' in fs
        assert f'"{Path(hpc_tmp).resolve()}" = "deny"' in fs
        # And the HPC bot's agent can't read the assistant's tmp folder (its conversations' files).
        hpc_deny = ClaudeCodeRunner(hpc).deny_read()
        assert str(Path(cfg.local.tmp_dir).resolve()) in hpc_deny
        # A bot's folders can't lie inside another's: it couldn't be kept out of them.
        nested = assistant_cfg(tmp_path, tmp_dir=os.path.join(hpc_tmp, "assistant"))
        with pytest.raises(ConfigError, match="overlap"):
            nested.local.register("assistant")
    finally:
        shutil.rmtree(hpc_tmp, ignore_errors=True)


# ---------------------------------------------------------------------------- reminders
async def test_reminders_fire_once_and_snooze(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    n = FakeNotifier()
    r = Reminders(store, n)
    a = r.add(time.time() - 1, "  Call   the bank ", "s1")
    r.add(time.time() + 3600, "Later")
    assert await r.fire_due() == 1 and await r.fire_due() == 0
    assert n.sent[0].text == "⏰ Call the bank" and n.sent[0].session_id == "s1"
    assert n.sent[0].buttons[0][0] == ("💤 1h", f"rsnz:{a['id']}:3600")
    again = r.snooze(a["id"], 600)
    assert again and abs(again["at"] - time.time() - 600) < 5 and r.snooze(a["id"], 600) is None
    assert [x["text"] for x in r.pending()] == ["Call the bank", "Later"]
    assert r.remove(again["id"]) and [x["text"] for x in r.pending()] == ["Later"]
    with pytest.raises(ValueError):
        r.add(time.time(), "   ")


# ---------------------------------------------------------------------------- a whole turn
@pytest.fixture
def aenv(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "cli.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = assistant_cfg(tmp_path)
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.local.db_path)
    notifier = FakeNotifier()
    domain = AssistantDomain(cfg, store, notifier)
    domain.prepare()
    agent = Agent(cfg, store, notifier, domain)
    ctl = ControlServer(cfg, agent, Projects(cfg, store, agent), notifier)
    yield type("Env", (), {"cfg": cfg, "store": store, "notifier": notifier, "domain": domain, "agent": agent,
                           "ctl": ctl, "vault": domain.vault, "tmp": tmp_path})
    shutil.rmtree(cfg.local.tmp(), ignore_errors=True)


def test_prompt_has_the_persona_and_what_it_knows(aenv):
    v = aenv.vault
    assert (v.root / "Assistant/Persona.md").exists() and (v.root / "Assistant/Memory").is_dir()
    (v.root / "Assistant/Persona.md").write_text("# Persona\n- Name: Jarvis\n- Tone: dry humour\n")
    (v.root / "Assistant/About me.md").write_text("I live in Abu Dhabi.\n")
    prompt = aenv.agent.system_prompt(aenv.agent.general_session().id)
    assert prompt.startswith("You are Jarvis,") and "dry humour" in prompt and "Abu Dhabi" in prompt
    assert str(v.root) in prompt and "ov task" in prompt and "ov remind" in prompt
    assert "planner mode" not in prompt.lower()  # no subagents for the assistant
    assert aenv.domain.persona_name() == "Jarvis"


async def test_context_has_the_time_and_whats_due(aenv):
    (aenv.vault.root / "Todo.md").write_text(f"- [ ] Due now 📅 {date.today().isoformat()}\n- [ ] Later 📅 2099-01-01\n")
    (aenv.vault.root / "Todo.sync-conflict-20261017-101010-ABC.md").write_text("x")
    ctx = await aenv.domain.now_context()
    assert "<now>" in ctx and date.today().strftime("%B %Y") in ctx
    assert "Due now" in ctx and "Later" not in ctx and "sync_conflicts" in ctx


@needs_git
async def test_a_turn_that_changes_notes_can_be_undone(aenv):
    await aenv.ctl.start()
    try:
        v = aenv.vault
        (v.root / "Journal.md").write_text("# Journal\n")
        s = aenv.agent.general_session()
        note = v.root / "Journal.md"
        await aenv.agent.handle(s, "\n".join([
            "FAKE_OV ov task add 'Call the bank' --due tomorrow --priority high",
            "FAKE_OV ov task list --when week",
            f"FAKE_OV echo '- met Jane' >> '{note}'",
            "FAKE_OV ov remind 'tomorrow 9am' 'Bring the documents'",
        ]))
        reply = aenv.notifier.sent[-1]
        assert "✓ added to Assistant/Tasks.md: Call the bank ⏫ 📅" in reply.text
        assert "Call the bank" in reply.text.split("@@@")[1]  # it lists what it just added
        assert "📝" in reply.text and "Journal.md" in reply.text and "Assistant/Tasks.md" in reply.text
        (label, data), = reply.buttons[0]
        assert label == "↩️ Undo" and data.startswith("vundo:")
        assert [r["text"] for r in aenv.domain.reminders.pending()] == ["Bring the documents"]
        assert "⏰ Reminder set for tomorrow 09:00: Bring the documents" in aenv.notifier.texts()
        # Your edit meanwhile (from Obsidian, synced), then Undo: the turn's changes go, yours stay.
        (v.root / "Ideas.md").write_text("mine\n")
        ok, what = await v.undo(data.split(":", 1)[1])
        assert ok and note.read_text() == "# Journal\n" and not (v.root / "Assistant/Tasks.md").exists()
        assert (v.root / "Ideas.md").read_text() == "mine\n"
        # A turn that changes nothing: no Undo button.
        await aenv.agent.handle(s, "FAKE_OV ov task list")
        assert aenv.notifier.sent[-1].buttons == []
        # Ticking a task off from Telegram is undoable too.
        T.add(v, "- [ ] Water plants")
        await aenv.domain.record_before()
        t, _ = T.complete(v, T.scan(v)[0].id)
        change = await aenv.domain.record(f"Done: {t.text}")
        assert change.message == "Done: Water plants" and change.files == [("M", "Assistant/Tasks.md")]
    finally:
        await aenv.ctl.stop()


async def test_ov_task_errors_and_status(aenv):
    await aenv.ctl.start()
    try:
        s = aenv.agent.general_session()
        token = aenv.ctl.open(s.id)
        ok, text = await aenv.ctl.handle(token, ["task", "done", "abcdef"])
        assert "no task" in text
        ok, text = await aenv.ctl.handle(token, ["task", "add", "x", "--due", "someday"])
        assert "a day is" in text
        ok, text = await aenv.ctl.handle(token, ["task", "add", "x", "--note", "../escape.md"])
        assert "inside the vault" in text
        ok, text = await aenv.ctl.handle(token, ["remind", "whenever", "x"])
        assert "say e.g." in text or "a day is" in text
        ok, text = await aenv.ctl.handle(token, ["remind", "2h", "stretch", "now"])
        assert text.startswith("✓ r")
        ok, text = await aenv.ctl.handle(token, ["status"])
        assert "reminders set in this conversation: 1" in text
        ok, text = await aenv.ctl.handle(token, ["help"])
        assert "ov task list" in text and "ov job" not in text
    finally:
        await aenv.ctl.stop()


@needs_git
async def test_dashboard_digest_and_search(aenv):
    v = aenv.vault
    today = date.today().isoformat()
    (v.root / "Plans.md").write_text(f"Trip to Berlin in May.\n- [ ] Book flights 📅 {today}\n")
    await v.init()
    (v.root / "Plans.md").write_text(f"Trip to Berlin in May.\n- [ ] Book flights 📅 {today}\n- hotel?\n")
    await v.snapshot("Jarvis · s1: added a question")
    aenv.domain.reminders.add(time.time() + 60, "Stand up")
    status, ctype, body, _ = await aenv.domain.dashboard_api(["today"], {})
    data = json.loads(body)
    assert [t["text"] for t in data["due"]] == ["Book flights"] and data["reminders"][0]["text"] == "Stand up"
    data = json.loads((await aenv.domain.dashboard_api(["tasks"], {"when": "all", "q": "book"}))[2])
    assert data["total"] == 1
    data = json.loads((await aenv.domain.dashboard_api(["changes"], {}))[2])
    assert data["changes"][0]["message"] == "Jarvis · s1: added a question"
    assert await aenv.domain.dashboard_api(["nope"], {}) is None
    assert b"renderToday" in aenv.domain.dashboard_js()
    sections = "\n".join(await aenv.domain.digest_sections(time.time() - 3600, time.time() + 1))
    assert "Book flights" in sections and "Stand up" in sections and "1 change by the assistant" in sections
    rows = aenv.domain.search_rows()
    plans = [r for r in rows if r[3] == "Plans.md"][0]
    assert "Berlin" in plans[0] and plans[2] == "note"
    from overseer.core.search import Search
    hits, total = await Search(aenv.cfg, aenv.store, aenv.domain).asearch("berlin")
    assert [h.ref for h in hits] == ["Plans.md"]


def test_the_assistant_bot_loads_from_its_config_file(tmp_path, monkeypatch):
    """`overseer run -c assistant.yaml`: its token from the .env file by the name in token_env, its vault from the
    config, the platform wired around it with the assistant's commands (and none of the HPC's)."""
    from telegram.ext import CommandHandler

    from overseer.core.app import build
    from overseer.core.config import load_config

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    vault = tmp_path / "Vault"
    vault.mkdir()
    tmp = tempfile.mkdtemp(prefix="ova-", dir="/tmp")
    try:
        text = Path(__file__).parent.parent.joinpath("assistant.example.yaml").read_text()
        text = text.replace("/home/you/Vault", str(vault)).replace("/tmp/overseer-assistant", tmp)
        (app_dir / "assistant.yaml").write_text(text)
        (app_dir / ".env").write_text("ASSISTANT_BOT_TOKEN=123456:FROM-DOTENV\n")
        monkeypatch.delenv("ASSISTANT_BOT_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        cfg = load_config(app_dir / "assistant.yaml")
        assert cfg.domain == "assistant" and cfg.vault_dir == vault.resolve()
        assert cfg.telegram.token == "123456:FROM-DOTENV"
        assert cfg.local.state() == app_dir / "state/assistant"
        app = build(cfg)
        commands = {c for h in app.handlers[0] if isinstance(h, CommandHandler) for c in h.commands}
        assert {"ask", "new", "auto", "search", "dashboard"} <= commands  # the platform's
        assert {"tasks", "task", "remind", "reminders", "undo", "vault", "persona"} <= commands
        assert not {"jobs", "fix", "cluster"} & commands
        assert (vault / "Assistant/Persona.md").exists() and Path(tmp, "agent-bin/ov").exists()
        bots = [json.loads(p.read_text()) for p in (app_dir / "state/bots").glob("*.json")]
        assert bots == [{"domain": "assistant", "state": str(app_dir / "state/assistant"), "tmp": tmp}]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------- topics (telegram.topics) and projects
def topic_bot(aenv, monkeypatch, use_topics, topic_mode=True):
    from types import SimpleNamespace as NS

    from overseer.core.bot import Bot, TelegramNotifier
    from test_integration import FakeTelegram
    tg = FakeTelegram(topics=topic_mode)
    deleted = []

    async def delete_forum_topic(chat_id, message_thread_id):
        deleted.append(message_thread_id)
    tg.delete_forum_topic = delete_forum_topic
    notifier = TelegramNotifier(NS(bot=tg), [1], aenv.store, use_topics=use_topics)
    aenv.cfg.telegram.topics = use_topics
    projects = Projects(aenv.cfg, aenv.store, aenv.agent)
    projects.notifier = notifier
    bot = Bot(aenv.cfg, app=NS(bot=tg), store=aenv.store, agent=aenv.agent, notifier=notifier, projects=projects)
    started = []

    async def fake_run_agent(chat_id, session, text, files=None):
        started.append((session.id, text))
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)
    return NS(tg=tg, notifier=notifier, bot=bot, projects=projects, started=started, deleted=deleted)


async def drain(bot):
    for t in list(bot._tasks):
        await t


async def test_with_topics_the_assistant_works_like_the_hpc_bot(aenv, monkeypatch):
    """telegram.topics: true - Overseer makes the topics ("general", one per project, made by you or the
    assistant), and nothing written outside them ("All") reaches the agent."""
    from test_integration import topic_update
    t = topic_bot(aenv, monkeypatch, use_topics=True)
    assert await t.notifier.detect_topics()
    await t.notifier.ensure_general_topic()
    gen = t.notifier.general_thread(1)
    # A message in "All" (Telegram made a topic for it): refused, that topic deleted.
    await t.bot.on_text(topic_update("what's on today?", thread=901), None)
    await drain(t.bot)
    assert not t.started and t.deleted == [901] and "outside the topics" in t.tg.sent[-1][2]
    # In "general": the everyday conversation.
    await t.bot.on_text(topic_update("what's on today?", thread=gen), None)
    await drain(t.bot)
    assert t.started[-1] == (aenv.agent.general_session().id, "what's on today?")
    # /project new: no folder needed (it can get a vault folder later); it gets its own topic.
    replies = []

    async def reply(update, text):
        replies.append(text)
    monkeypatch.setattr(t.bot, "reply", reply)
    await t.bot.cmd_project(topic_update("/project new japan Trip in April", thread=gen),
                            type("C", (), {"args": ["new", "japan", "Trip", "in", "April"]})())
    p = aenv.store.get_project("japan")
    assert p.dir == "" and p.context == "Trip in April" and "japan" in [n for _, n, _ in t.tg.created]
    assert p.thread_map["1"] != gen and t.notifier.general_thread(1) == gen  # (typed in general: general stays)
    assert not [n for _, n, _ in t.tg.created if "jobs" in n]  # the HPC's jobs topics are the HPC bot's
    assert "Project <code>japan</code> created." in replies[-1] and "folder" not in replies[-1].split("\n")[0]
    await t.bot.on_text(topic_update("book the ryokan", thread=p.thread_map["1"]), None)
    await drain(t.bot)
    assert aenv.agent.get(t.started[-1][0]).project == "japan"
    # The assistant makes one itself: ov project new (and its notes folder must be in the vault).
    aenv.ctl.projects = t.projects
    aenv.ctl.notifier = t.notifier
    s = aenv.agent.general_session()
    token = aenv.ctl.open(s.id)
    ok, text = await aenv.ctl.handle(token, ["project", "new", "move", "--about", "Moving flats", "--dir",
                                             "Projects/Move/", "--move-here"])
    assert "✓ project move created (Projects/Move)" in text and "now in move" in text, text
    assert "move" in [n for _, n, _ in t.tg.created] and aenv.agent.get(s.id).project == "move"
    for bad in ("../outside", "/etc", ".obsidian/x"):
        ok, text = await aenv.ctl.handle(token, ["project", "new", "x", "--dir", bad])
        assert "inside the vault" in text, text
    prompt = aenv.agent.system_prompt(s.id)
    assert '## Project: move' in prompt and "Moving flats" in prompt and "`Projects/Move/`" in prompt
    ok, text = await aenv.ctl.handle(token, ["help"])
    assert "ov project new <name>" in text


async def test_with_topics_off_it_is_one_plain_chat(aenv, monkeypatch):
    """telegram.topics: false - even with topic mode on in @BotFather, no topics are made and every message is the
    main conversation (nothing refused); true without topic mode: the same, until it's switched on there."""
    from test_integration import topic_update
    t = topic_bot(aenv, monkeypatch, use_topics=False)
    assert not await t.notifier.detect_topics() and t.notifier.topic_mode
    await t.notifier.ensure_general_topic()
    assert t.tg.created == []
    await t.bot.on_text(topic_update("hi", thread=901), None)
    await drain(t.bot)
    assert t.started == [(aenv.agent.general_session().id, "hi")] and t.deleted == []
    assert t.bot._topics_state() == "off (telegram.topics: false)"
    t2 = topic_bot(aenv, monkeypatch, use_topics=True, topic_mode=False)
    assert not await t2.notifier.detect_topics()
    assert "threaded mode" in t2.bot._topics_state()
    t3 = topic_bot(aenv, monkeypatch, use_topics=None)  # (the HPC bot's default: the bot's own mode)
    assert await t3.notifier.detect_topics()


def test_hpc_projects_still_need_their_folder(tmp_path):
    from overseer.hpc.domain import HpcDomain
    d = HpcDomain.__new__(HpcDomain)
    assert d.check_project_dir("~/dreamer/") == "~/dreamer"
    for bad in ("", "relative/dir", "/with space"):
        with pytest.raises(ValueError, match="on the HPC"):
            d.check_project_dir(bad)
