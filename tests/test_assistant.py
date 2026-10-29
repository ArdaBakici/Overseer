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
from overseer.core.notify import Outgoing
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
async def test_followups_come_back_until_answered(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    n = FakeNotifier()
    r = Reminders(store, n)
    logged, stuck = [], []

    async def on_log(e):
        logged.append((e["state"], e["text"]))

    async def on_stuck(item):
        stuck.append(item["text"])
    r.on_log, r.on_stuck = on_log, on_stuck
    t0 = time.time()
    policy = {"after": 600, "times": 2}
    a = r.add(t0 - 1, "  Call   the bank ", "s1", policy=policy)
    r.add(t0 + 86400, "Later on")
    assert await r.fire_due(t0) == 1 and await r.fire_due(t0 + 1) == 0
    first = n.sent[0]
    assert first.text == "⏰ Call the bank" and first.session_id == "s1"
    assert first.buttons == [[("✅ Done", f"fu:{a['id']}:done"), ("💤 Later", f"fu:{a['id']}:later"),
                              ("⏭ Skip", f"fu:{a['id']}:skip")]]
    # Unanswered: again after 10 minutes, twice, then it counts as unanswered (and is logged).
    assert await r.fire_due(t0 + 601) == 1 and n.sent[-1].text == "🔁 ⏰ Call the bank"
    assert await r.fire_due(t0 + 1202) == 1
    assert await r.fire_due(t0 + 1803) == 0  # (no third time: it waits a while for an answer)
    assert [x["id"] for x in r.open()] == [a["id"]]
    assert await r.fire_due(t0 + 1202 + 3601) == 0 and r.open() == []
    assert logged == [("unanswered", "Call the bank")] and r.log()[-1]["icon"] == "🤷"
    # Done: no more follow-ups.
    b = r.add(t0 - 1, "Meds", kind="say", policy=policy)
    await r.fire_due(t0)
    assert (await r.act(b["id"], "done"))["state"] == "done" and await r.act(b["id"], "done") is None
    assert await r.fire_due(t0 + 5000) == 0 and logged[-1] == ("done", "Meds")
    # Later: back after its policy's interval; Skip: logged.
    c = r.add(t0 - 1, "Water", policy=policy)
    await r.fire_due(t0)
    later = await r.act(c["id"], "later")
    assert later["state"] == "pending" and 590 < later["at"] - time.time() < 610
    await r.act(c["id"], "skip")
    assert logged[-1] == ("skipped", "Water")
    # A check-back on a commitment: Stuck wakes the assistant.
    d = r.add(t0 - 1, "Send the report", kind="commit", policy={"after": 0, "times": 0})
    await r.fire_due(t0)
    assert "Check-in" in n.sent[-1].text and n.sent[-1].buttons[0][2] == ("🧱 Stuck", f"fu:{d['id']}:stuck")
    await r.act(d["id"], "stuck")
    assert stuck == ["Send the report"] and logged[-1] == ("stuck", "Send the report")
    # Cancel before it fires: gone, not logged.
    e = r.add(t0 + 999, "Never mind")
    assert r.remove(e["id"]) and all(x["id"] != e["id"] for x in r.pending())
    with pytest.raises(ValueError):
        r.add(time.time(), "   ")


async def test_a_focus_session_counts_down_quietly(tmp_path):
    class LiveNotifier(FakeNotifier):
        def __init__(self):
            super().__init__()
            self.posted, self.edits = [], []

        async def post(self, msg):
            self.posted.append(msg)
            return {"1": len(self.posted)}

        async def edit(self, refs, text, buttons=None):
            self.edits.append((refs, text, buttons))

    n = LiveNotifier()
    r = Reminders(Store(tmp_path / "db.sqlite"), n)
    f = await r.start_focus(25, "Write the intro", "s1")
    assert "25 min left" in n.posted[0].text and n.posted[0].silent
    with pytest.raises(ValueError, match="running already"):
        await r.start_focus(10, "another")
    assert r.log()[-1]["state"] == "started"
    await r.fire_due(time.time() + 70)  # a minute later: the same message, edited (no new one)
    assert n.edits and "min left" in n.edits[-1][1] and not n.sent
    await r.fire_due(f["at"] + 1)  # time's up: the countdown says so, and a ping asks how it went
    assert "time's up" in n.edits[-1][1] and "Focus time's up" in n.sent[-1].text
    assert [b[0] for b in n.sent[-1].buttons[0]] == ["✅ Done", "➕ 5 min", "🧱 Stuck"]
    more = await r.act(f["id"], "more")
    assert more["state"] == "pending" and len(n.posted) == 2  # a new countdown, 5 minutes
    await r.act(f["id"], "done")
    assert r.focus() is None and r.log()[-1]["state"] == "done"


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
        assert bots == [{"domain": "assistant", "state": str(app_dir / "state/assistant"), "tmp": tmp,
                         "config": str(app_dir / "assistant.yaml")}]  # (the TUI finds the bot by it)
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

    async def fake_run_agent(chat_id, session, text, files=None, steer=True):
        started.append((session.id, text))
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)
    return NS(tg=tg, notifier=notifier, bot=bot, projects=projects, started=started, deleted=deleted)


async def drain(bot):
    for t in list(bot._tasks):
        await t


async def test_with_topics_the_assistant_works_like_the_hpc_bot(aenv, monkeypatch):
    """telegram.topics: true - Overseer makes the topics ("general", one per project, made by you or the
    assistant), and what's written outside them ("All") goes to the general conversation."""
    from test_integration import topic_update
    t = topic_bot(aenv, monkeypatch, use_topics=True)
    assert await t.notifier.detect_topics()
    await t.notifier.ensure_general_topic()
    gen = t.notifier.general_thread(1)
    # A message in "All" (Telegram made a topic for it): to general, that topic deleted.
    await t.bot.on_text(topic_update("what's on today?", thread=901), None)
    await drain(t.bot)
    assert t.started == [(aenv.agent.general_session().id, "what's on today?")] and t.deleted == [901]
    assert any("From “All”" in text for _, th, text in t.tg.sent if th == gen)
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


# ---------------------------------------------------------------------------- routines, set-up, follow-ups
ROUTINES = """# Routines

Some words of mine, ignored.

## Morning planning
- when: weekdays 09:00
- do: Help me pick my top 3.
  One question at a time.

## Meds
- when: every day 09:30 and 9pm
- say: Meds 💊
- follow-up: persistent

## Weekly review
- when: sun 18:00
- do: Plan the week.
- paused until: 2026-11-01

## Gym
- when: mon, thu 7am
- say: Gym!
- off: yes

# Follow-ups
- default: again after 20m, up to 2 more times
- persistent: again after 10m, up to 5 more times
"""


def test_the_routines_note():
    from overseer.assistant import routines as R
    rs = R.parse(ROUTINES)
    assert rs.errors == []
    m = rs.get("morning planning")
    assert m.do == "Help me pick my top 3. One question at a time." and m.describe_when() == "weekdays 09:00"
    meds = rs.get("Meds")
    assert meds.describe_when() == "every day 09:30 and 21:00" and rs.policy(meds.follow_up) == {"after": 600, "times": 5}
    assert rs.policy("") == {"after": 1200, "times": 2} and rs.policy("none") == {"after": 0, "times": 0}
    assert rs.policy("again after 1h, up to 1 more time") == {"after": 3600, "times": 1}
    sat = datetime(2026, 10, 17, 8, 0)  # a Saturday
    assert m.next_time(sat) == datetime(2026, 10, 19, 9, 0)
    assert rs.get("weekly review").next_time(sat) == datetime(2026, 11, 8, 18, 0)  # paused until Nov 1st
    assert rs.get("gym").next_time(sat) is None  # off
    assert meds.occurrences(datetime(2026, 10, 17, 9, 0), datetime(2026, 10, 17, 22, 0)) == [
        datetime(2026, 10, 17, 9, 30), datetime(2026, 10, 17, 21, 0)]
    bad = R.parse("# Routines\n## A\n- do: x\n## B\n- when: sometimes 9am\n- say: y\n## C\n- when: daily 9:00\n"
                  "- do: x\n- say: y\n## D\n- when: 9:00\n- say: z\n- follow-up: brutal\n# Follow-ups\n- x: often\n")
    assert len(bad.errors) == 5 and [r.name for r in bad.routines] == ["D"], bad.errors
    assert any("brutal" in e for e in bad.errors) and any("sometimes" in e for e in bad.errors)


def with_autopilot(aenv, monkeypatch):
    from overseer.core.autopilot import Autopilot
    ap = Autopilot(aenv.cfg, aenv.store, aenv.agent, aenv.notifier)
    woken = []
    monkeypatch.setattr(ap, "wake", lambda s, prompt, kind="wakeup", quiet=False, by_user=False:
                        woken.append((s.id, kind, prompt)))
    return woken


async def test_routines_run_at_their_times(aenv, monkeypatch):
    from overseer.assistant.scheduler import CHECKED_KEY
    woken = with_autopilot(aenv, monkeypatch)
    sch = aenv.domain.scheduler
    (aenv.vault.root / "Assistant/Routines.md").write_text(ROUTINES)
    monday = datetime(2026, 10, 19, 9, 31)
    assert await sch.tick(monday) == []  # the first tick: from now on
    aenv.store.set(CHECKED_KEY, str(datetime(2026, 10, 19, 8, 59).timestamp()))
    assert await sch.tick(monday) == ["Morning planning", "Meds"]
    sid, kind, prompt = woken[-1]
    assert kind == "routine" and prompt.startswith("[Routine: Morning planning]") and "top 3" in prompt
    assert "<now>" in prompt and sid == aenv.agent.general_session().id
    say = aenv.domain.reminders.pending()[-1]
    assert say["text"] == "Meds 💊" and say["kind"] == "say" and say["policy"] == {"after": 600, "times": 5}
    assert await sch.tick(datetime(2026, 10, 19, 9, 32)) == []  # once
    # Missed while Overseer was down: run if recent, skipped if not.
    aenv.store.set(CHECKED_KEY, str(datetime(2026, 10, 19, 20, 0).timestamp()))
    assert await sch.tick(datetime(2026, 10, 19, 23, 30)) == ["Meds"]  # 21:00, 2.5 h ago
    aenv.store.set(CHECKED_KEY, str(datetime(2026, 10, 19, 20, 0).timestamp()))
    assert await sch.tick(datetime(2026, 10, 20, 1, 0)) == []  # 4 h ago: skipped
    # A broken note (an edit in Obsidian): told once, the rest still runs.
    (aenv.vault.root / "Assistant/Routines.md").write_text(ROUTINES + "\n# Routines\n## Oops\n- when: whenever\n- do: x\n")
    await sch.tick(datetime(2026, 10, 20, 1, 1))
    await sch.tick(datetime(2026, 10, 20, 1, 2))
    notices = [m.text for m in aenv.notifier.sent if "has problems" in m.text]
    assert len(notices) == 1 and "Oops" in notices[0]
    assert "Oops" in sch.check_text() and "✓ Meds (say): every day 09:30 and 21:00" in sch.check_text()


async def test_the_first_run_says_hi_and_asks_for_a_personality(aenv, monkeypatch):
    woken = with_autopilot(aenv, monkeypatch)
    sch = aenv.domain.scheduler
    v = aenv.vault
    # Nothing is set up in advance: a placeholder persona, no routines, and the two notes about the user.
    assert not aenv.domain.persona_set() and not (v.root / "Assistant/Routines.md").exists()
    for name in ("My writing style.md", "My personality.md"):
        assert "assistant: index" in (v.root / "Assistant" / name).read_text()
    assert "## Informal" in (v.root / "Assistant/My writing style.md").read_text()
    prompt = aenv.agent.system_prompt(aenv.agent.general_session().id)
    assert prompt.startswith("You are the assistant,") and "They haven't set it yet" in prompt
    assert "No routines yet" in prompt and "## How you help: you build the system" in prompt
    assert "Assistant/My writing style.md" in prompt and "Assistant/My personality.md" in prompt
    assert "Assistant/System.md" in prompt and "Assistant/Someday.md" in prompt and "[silent]" in prompt
    assert "Top 3" not in prompt  # (no fixed daily-note sections)
    assert await sch.onboard() and not await sch.onboard()  # once
    sid, kind, text = woken[0]
    assert kind == "setup" and text.startswith("[First run]") and "give you a personality" in text
    assert "Don't ask for their name" in text and "Assistant/Persona.md" in text and "# Follow-ups" not in text
    # They give it one: the instructions follow it.
    (v.root / "Assistant/Persona.md").write_text("# Persona\n- Name: Pip\n- Tone: cheeky\n")
    prompt = aenv.agent.system_prompt(aenv.agent.general_session().id)
    assert prompt.startswith("You are Pip,") and "They haven't set it yet" not in prompt
    (v.root / "Assistant/Routines.md").write_text(ROUTINES)
    prompt = aenv.agent.system_prompt(aenv.agent.general_session().id)
    assert "Their routines now: Morning planning (weekdays 09:00)" in prompt and "Gym" in prompt
    # After `overseer reset` (the flag gone), one that already has a persona isn't asked again.
    aenv.store.set("assistant:onboarding_started", "")
    assert not await sch.onboard()


def test_a_name_on_its_own_line(aenv):
    """"Name:" with nothing after it isn't a name (it doesn't take the next line's text)."""
    (aenv.vault.root / "Assistant/Persona.md").write_text("# Persona\n- Name:\n- Tone: calm\n")
    assert not aenv.domain.persona_set() and aenv.domain.persona_name() == "the assistant"


async def test_reminders_fire_once_unless_told_otherwise(aenv):
    rem = aenv.domain.reminders
    r = rem.add(time.time() - 1, "Call mum")
    assert r["policy"]["times"] == 0
    await rem.fire_due()
    assert sum("Call mum" in m.text for m in aenv.notifier.sent) == 1
    item = rem.get(r["id"])
    await rem.fire_due(item["next"] + 1)  # nobody answered: given up on, not sent again
    assert sum("Call mum" in m.text for m in aenv.notifier.sent) == 1 and rem.get(r["id"])["state"] == "unanswered"


def test_someday_tasks(aenv):
    v = aenv.vault
    T.add(v, "- [ ] Learn to draw #someday", "Assistant/Someday.md")
    T.add(v, "- [ ] Pay rent 📅 2026-11-01")
    T.add(v, "- [x] Read Dune #someday ✅ 2026-10-01", "Assistant/Someday.md")
    hits = T.select(T.scan(v), "someday")
    assert [t.text for t in hits] == ["Learn to draw #someday"]


async def test_ov_task_add_someday_and_ov_ask(aenv):
    from types import SimpleNamespace as NS
    from overseer.core.bot import Bot
    await aenv.ctl.start()
    try:
        s = aenv.agent.general_session()
        token = aenv.ctl.open(s.id)
        ok, text = await aenv.ctl.handle(token, ["task", "add", "Learn", "to", "draw", "--someday"])
        assert text == "✓ added to Assistant/Someday.md: Learn to draw #someday", text
        ok, text = await aenv.ctl.handle(token, ["task", "list", "--when", "someday"])
        assert "Learn to draw" in text
        # A question with tap buttons.
        ok, text = await aenv.ctl.handle(token, ["ask", "A daily reminder for your meds at 9:30?",
                                                 "--options", "Yes | Another time | No"])
        assert ok and text.startswith("✓ asked"), text
        msg = aenv.notifier.sent[-1]
        assert msg.text == "❓ A daily reminder for your meds at 9:30?" and msg.session_id == s.id
        labels = [label for row in msg.buttons for label, _ in row]
        assert labels == ["Yes", "Another time", "No"] and all(len(d) <= 64 for row in msg.buttons for _, d in row)
        assert [len(row) for row in msg.buttons] == [2, 1]  # ("Another time" is too long for three in a row)
        ok, text = await aenv.ctl.handle(token, ["ask", "Which?", "--options", "only one"])
        assert not ok and "2 to 6 choices" in text
        ok, text = await aenv.ctl.handle(token, ["ask", "Which?", "--options",
                                                 "PKA till 19:30 as planned then stop|Rest"])
        assert not ok and "at most 32 characters" in text and "put the detail in the question" in text
    finally:
        await aenv.ctl.stop()
    # Their tap: the message says what they chose, and the agent gets it as their answer.
    bot = Bot(aenv.cfg, app=None, store=aenv.store, agent=aenv.agent, notifier=aenv.notifier)
    started, edits = [], []

    async def fake_run_agent(chat_id, session, text, files=None, steer=True):
        started.append((session.id, text))
    bot._run_agent = fake_run_agent

    async def press(data):
        async def answer(*a, **kw):
            pass

        async def edit_message_text(text, **kw):
            edits.append(text)
        q = NS(data=data, message=NS(message_thread_id=None), answer=answer, edit_message_text=edit_message_text)
        await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1), effective_message=q.message), None)
        for t in list(bot._tasks):
            await t
    await press(msg.buttons[0][1][1])
    assert edits[-1] == "❓ A daily reminder for your meds at 9:30?\n→ <b>Another time</b>"
    assert started == [(s.id, "[They tapped “Another time” on your question: A daily reminder for your meds at 9:30?]")]
    await press(msg.buttons[0][0][1])  # answered already
    assert len(started) == 1 and "already answered" in edits[-1]


async def test_a_wake_up_with_nothing_to_say_sends_nothing(aenv):
    s = aenv.agent.general_session()
    n = len(aenv.notifier.sent)
    await aenv.agent.handle(s, "FAKE_SILENT", origin="routine")
    assert len(aenv.notifier.sent) == n
    await aenv.agent.handle(s, "FAKE_SILENT")  # (to the user's own message, it's a reply like any other)
    assert aenv.notifier.sent[-1].text.startswith("🤖 [silent]")


async def test_ov_commit_focus_followup_and_routine(aenv, monkeypatch):
    with_autopilot(aenv, monkeypatch)
    await aenv.ctl.start()
    try:
        (aenv.vault.root / "Assistant/Routines.md").write_text(ROUTINES)
        s = aenv.agent.general_session()
        token = aenv.ctl.open(s.id)
        ok, text = await aenv.ctl.handle(token, ["commit", "2h", "send", "the", "report"])
        assert text.startswith("✓ r") and "again after 20m, up to 2 more times" in text, text
        assert "🎯 I'll check in today" in aenv.notifier.texts()
        rid = text.split(":")[0][2:]
        c = aenv.domain.reminders.get(rid)
        assert c["kind"] == "commit" and c["text"] == "send the report"
        ok, text = await aenv.ctl.handle(token, ["remind", "45m", "stretch", "--follow-up", "persistent"])
        assert "again after 10m, up to 5 more times" in text
        ok, text = await aenv.ctl.handle(token, ["remind", "45m", "x", "--follow-up", "brutal"])
        assert "persistent" in text and "none" in text  # the names it knows
        ok, text = await aenv.ctl.handle(token, ["focus", "25", "write", "the", "intro"])
        assert "focus until" in text
        ok, text = await aenv.ctl.handle(token, ["focus"])
        assert "write the intro" in text
        ok, text = await aenv.ctl.handle(token, ["followup", "list"])
        assert "send the report" in text and "stretch" in text and "write the intro" in text
        # They answer a check-back in words: the agent records it.
        await aenv.domain.reminders.fire_due(c["at"] + 1)  # (the focus session and the stretch are due by then too)
        assert any("Check-in</b>: send the report" in m.text for m in aenv.notifier.sent[-3:])
        ok, text = await aenv.ctl.handle(token, ["followup", "done", rid])
        assert text == f"✓ {rid} done"
        ok, text = await aenv.ctl.handle(token, ["followup", "done", rid])
        assert "no follow-up" in text
        ok, text = await aenv.ctl.handle(token, ["focus", "stop"])
        assert "no focus session" in text  # (it ended: its ping is waiting for an answer)
        ok, text = await aenv.ctl.handle(token, ["routine", "check"])
        assert "✓ Morning planning (do): weekdays 09:00 · next" in text and "follow-up persistent" in text
        ctx = await aenv.domain.now_context()
        assert "<today_so_far>" in ctx and "done: send the report" in ctx
        waiting = ctx.split("<waiting_for_their_answer>")[1].split("</waiting_for_their_answer>")[0]
        assert "reminder: stretch" in waiting and "focus: write the intro" in waiting
    finally:
        await aenv.ctl.stop()


async def test_follow_up_buttons_work(aenv, monkeypatch):
    from types import SimpleNamespace as NS
    from overseer.assistant.telegram import AssistantBot
    with_autopilot(aenv, monkeypatch)
    rem = aenv.domain.reminders
    r = rem.add(time.time() - 1, "Meds 💊", kind="say")
    await rem.fire_due()
    edits = []

    async def edit(q, text, buttons=None):
        edits.append((text, buttons))
    bot = AssistantBot(NS(_edit=edit), aenv.domain)
    await bot._followup(None, r["id"], "done")
    assert edits[-1] == ("✅ Meds 💊", None) and rem.log()[-1]["text"] == "Meds 💊"
    await bot._followup(None, r["id"], "done")
    assert edits[-1][0] == "That one's settled already."
    later = rem.add(time.time() - 1, "Water", kind="say")
    await rem.fire_due()
    await bot._followup(None, later["id"], "later")
    assert edits[-1][0].startswith("💤 Water") and edits[-1][1] == [[("✖ Cancel", f"fu:{later['id']}:cancel")]]
    # Outcomes aren't written into the daily note: what goes there is up to the assistant.
    daily = aenv.vault.root / aenv.vault.daily_note()
    assert not daily.exists()


# ---------------------------------------------------------------------------- media, time zone, reactions, notes
GIF = b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"
PUBLIC = "93.184.216.34"


def fake_web(routes: dict):
    """An httpx transport answering by host+path (the request goes to the pinned address: Host says which)."""
    import httpx
    seen = []

    def handler(request):
        host = request.headers["host"].split(":")[0]
        seen.append((str(request.url), host, request.extensions.get("sni_hostname")))
        status, headers, body = routes.get(host + request.url.path, (404, {}, b""))
        return httpx.Response(status, headers=headers, content=body)
    return httpx.MockTransport(handler), seen


def resolver(table):
    import socket

    def resolve(host, port, type=None):
        import ipaddress
        try:
            literal = str(ipaddress.ip_address(host))  # (an address as the host: itself, like the real resolver)
        except ValueError:
            literal = None
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (literal or table.get(host, PUBLIC), port))]
    return resolve


async def test_fetching_media_is_kept_to_the_public_internet(tmp_path):
    from overseer.assistant import media
    transport, seen = fake_web({
        "pics.example/cat.gif": (200, {"content-type": "image/gif"}, GIF),
        "pics.example/page": (200, {"content-type": "text/html"}, b"<html>hi</html>"),
        "pics.example/huge.png": (200, {}, b"\x89PNG\r\n\x1a\n" + b"0" * 3000),
        "pics.example/to-lan": (302, {"location": "http://router.lan/admin.png"}, b""),
        "pics.example/hop": (301, {"location": "/cat.gif"}, b""),
    })
    resolve = resolver({"router.lan": "192.168.1.1", "tailnet.example": "100.101.102.103", "me.example": "127.0.0.1",
                        "v6.example": "::1"})
    fetch = lambda url, **kw: media.fetch(url, tmp_path, 2000, transport=transport, resolve=resolve, **kw)  # noqa: E731
    f = await fetch("https://pics.example/cat.gif")
    assert (f.kind, f.size, f.path.name) == ("gif", len(GIF), "cat.gif") and f.path.read_bytes() == GIF
    url, host, sni = seen[-1]
    assert url.startswith(f"https://{PUBLIC}/") and host == "pics.example" and sni == "pics.example"
    f = await fetch("https://pics.example/hop", name="Funny cat!")
    assert f.path.name == "Funny-cat.gif" and f.url.endswith("/cat.gif")  # redirects are followed (and checked)
    for url, why in [("https://pics.example/page", "isn't a picture"), ("https://pics.example/huge.png", "limit"),
                     ("https://pics.example/to-lan", "private address"),
                     ("http://tailnet.example/x.gif", "private address"), ("http://me.example/x", "private"),
                     ("http://v6.example/x", "private"), ("ftp://pics.example/x", "only http"),
                     ("https://user:pw@pics.example/cat.gif", "passwords")]:
        with pytest.raises(media.FetchError, match=why):
            await fetch(url)
    assert media.sniff(b"\x00\x00\x00\x18ftypmp42") == "mp4" and media.sniff(b"RIFF\x00\x00\x00\x00WEBPVP8") == "webp"


async def test_ov_fetch_and_gif_send_playable_media(aenv, monkeypatch):
    from overseer.assistant import media
    from overseer.core.agent import collect_outbox
    transport, _ = fake_web({"pics.example/cat.gif": (200, {}, GIF),
                             "media.giphy.com/media/abc/giphy.gif": (200, {}, GIF)})
    real = media.fetch
    monkeypatch.setattr(media, "fetch", lambda url, folder, max_bytes, name="": real(
        url, folder, max_bytes, name, transport=transport, resolve=resolver({})))
    await aenv.ctl.start()
    try:
        s = aenv.agent.general_session()
        token = aenv.ctl.open(s.id)
        ok, text = await aenv.ctl.handle(token, ["fetch", "https://pics.example/cat.gif"])
        assert text.startswith("✓ cat.gif (gif") and "./outbox/" in text, text
        notice = aenv.notifier.sent[-1]
        assert notice.topic == "web" and notice.silent and "cat.gif" in notice.text and "pics.example" in notice.text
        outbox = aenv.agent.session_runner(s).workdir(s.id) / "outbox"
        photos, docs, notes, played = collect_outbox(outbox)
        assert played == [("animation", "cat.gif", GIF)] and not photos and not docs
        ok, text = await aenv.ctl.handle(token, ["fetch", "http://127.0.0.1:8765/api/x"])
        assert "private address" in text
        # GIF search: off until a Giphy key is configured.
        ok, text = await aenv.ctl.handle(token, ["gif", "happy", "dance"])
        assert "GIF search is off" in text
        aenv.cfg.media.giphy_key_env = "GIPHY_TEST_KEY"
        monkeypatch.setenv("GIPHY_TEST_KEY", "k")

        async def search(query, key, rating="pg-13", limit=8, transport=None):
            assert (query, key, rating) == ("happy dance", "k", "pg-13")
            return [{"title": "Happy Dance", "url": "https://media.giphy.com/media/abc/giphy.gif", "page": ""},
                    {"title": "Other", "url": "https://media.giphy.com/media/def/giphy.gif", "page": ""}]
        monkeypatch.setattr(media, "giphy_search", search)
        ok, text = await aenv.ctl.handle(token, ["gif", "happy", "dance"])
        assert "happy-dance.gif" in text and "Happy Dance" in text and "2. Other" in text, text
        assert "GIF for “happy dance”" in aenv.notifier.sent[-1].text
        aenv.cfg.media.fetch = False
        ok, text = await aenv.ctl.handle(token, ["fetch", "https://pics.example/cat.gif"])
        assert "is off" in text
        ok, text = await aenv.ctl.handle(token, ["capabilities"])
        assert "OFF (media.fetch)" in text and "on (Giphy)" in text and ".gif" in text and "stickers" in text
    finally:
        await aenv.ctl.stop()


async def test_gifs_and_videos_play_in_telegram_and_the_web_topic(aenv):
    from types import SimpleNamespace as NS
    from overseer.core.bot import TelegramNotifier
    from test_integration import FakeTelegram
    tg = FakeTelegram()
    played = []

    async def send_animation(chat_id, animation=None, message_thread_id=None, **kw):
        played.append(("animation", animation.filename, message_thread_id))
        return NS(message_id=99)

    async def send_video(chat_id, video=None, message_thread_id=None, **kw):
        played.append(("video", video.filename, kw.get("supports_streaming")))
        return NS(message_id=98)
    tg.send_animation, tg.send_video = send_animation, send_video
    n = TelegramNotifier(NS(bot=tg), [1], aenv.store)
    await n.detect_topics()
    await n.send(Outgoing("🤖 here", media=[("animation", "cat.gif", GIF), ("video", "clip.mp4", b"x")]))
    assert tg.sent[-1][2] == "🤖 here" and played == [("animation", "cat.gif", None), ("video", "clip.mp4", True)]
    await n.send(Outgoing("🌐 Fetched cat.gif", topic="web", silent=True))
    await n.send(Outgoing("🌐 Fetched dog.gif", topic="web", silent=True))
    web = n.side_thread(1, "web")
    assert [name for _, name, _ in tg.created] == ["🌐 web"] and tg.sent[-1] == (1, web, "🌐 Fetched dog.gif")
    plain = TelegramNotifier(NS(bot=FakeTelegram(topics=False)), [1], aenv.store)
    await plain.detect_topics()
    await plain.send(Outgoing("🌐 Fetched cat.gif", topic="web"))
    assert plain.app.bot.sent[-1] == (1, None, "🌐 Fetched cat.gif")


def test_the_time_zone_is_the_users(tmp_path, monkeypatch):
    from overseer.assistant.reminders import exact_text
    from overseer.core.config import apply_timezone
    cfg = assistant_cfg(tmp_path)
    cfg.local.timezone = "Asia/Dubaii"
    with pytest.raises(ConfigError, match="Asia/Dubai"):
        cfg.validate()
    monkeypatch.setenv("TZ", "Europe/Istanbul")  # (the homelab's)
    time.tzset()
    try:
        apply_timezone("Asia/Dubai")
        assert os.environ["TZ"] == "Asia/Dubai" and time.strftime("%z") == "+0400"
        ts = datetime(2026, 10, 22, 10, 0).timestamp()
        assert exact_text(ts).startswith("Thu 2026-10-22 10:00 (+04:00, ")
    finally:
        monkeypatch.undo()
        time.tzset()


async def test_ov_react_to_the_last_message(aenv, monkeypatch):
    from types import SimpleNamespace as NS
    from overseer.core.bot import Bot
    reactions = []

    async def react(chat_id, message_id, emoji):
        reactions.append((chat_id, message_id, emoji))
    aenv.notifier.react = react
    bot = Bot(aenv.cfg, app=None, store=aenv.store, agent=aenv.agent, notifier=aenv.notifier,
              projects=Projects(aenv.cfg, aenv.store, aenv.agent))

    async def fake_run_agent(chat_id, session, text, files=None, steer=True):
        pass
    monkeypatch.setattr(bot, "_run_agent", fake_run_agent)
    await aenv.ctl.start()
    try:
        s = aenv.agent.general_session()
        token = aenv.ctl.open(s.id)
        ok, text = await aenv.ctl.handle(token, ["react", "👍"])
        assert "no message of theirs" in text
        msg = NS(text="I did it!", message_id=4242, reply_to_message=None, message_thread_id=None,
                 is_topic_message=False)
        await bot.on_text(NS(effective_message=msg, effective_chat=NS(id=1)), None)
        ok, text = await aenv.ctl.handle(token, ["react", "🎉"])
        assert text == "✓ reacted 🎉" and reactions == [(1, 4242, "🎉")]
        ok, text = await aenv.ctl.handle(token, ["react", "❤️"])  # (with the variation selector)
        assert reactions[-1][2] == "❤"
        ok, text = await aenv.ctl.handle(token, ["react", "🦖"])
        assert "only these" in text and len(reactions) == 2
    finally:
        await aenv.ctl.stop()


def test_notes_flagged_to_always_load(aenv):
    v = aenv.vault
    (v.root / "Assistant/Writing style.md").write_text("---\nassistant: always\n---\n# Writing style\nShort sentences.\n")
    (v.root / "People.md").write_text("---\nassistant: index\ndescription: who's who\n---\nJane - colleague\n")
    (v.root / "Big.md").write_text("---\nassistant: [always]\n---\n" + "x" * 30_000)
    (v.root / "Big2.md").write_text("---\nassistant: always\n---\n" + "y" * 30_000)
    (v.root / "Plain.md").write_text("---\ntags: [a]\n---\nnothing\n")
    (v.root / "Broken.md").write_text("---\nassistant: [unclosed\n---\n")
    prompt = aenv.agent.system_prompt(aenv.agent.general_session().id)
    assert "## Notes you always keep in mind" in prompt and "Short sentences." in prompt
    assert "- People.md: who's who" in prompt and "Jane - colleague" not in prompt
    assert "x" * 12_000 in prompt and "x" * 12_001 not in prompt and "[... cut here" in prompt  # (too long: cut)
    assert "- Big2.md: (flagged always, left out" in prompt and "y" * 100 not in prompt  # (over the budget)
    assert "Plain.md" not in prompt and "assistant: always" in prompt  # (and it's told how to flag one)
    assert [r for r, _, _ in v.flagged()] == ["Assistant/My personality.md", "Assistant/My writing style.md",
                                              "Assistant/Writing style.md", "Big.md", "Big2.md", "People.md"]


def test_ov_history_day(aenv):
    from overseer.core.live import Transcript
    from overseer.core.search import Search, render_day
    a = aenv.agent.general_session()
    b = aenv.agent.new_conversation()
    today = datetime.combine(date.today(), datetime.min.time()).timestamp()
    for sid, ts, text, reply in [(a.id, today - 3600, "yesterday's", "old"), (a.id, today + 3600, "plan my day",
                                 "Top 3: ..."), (b.id, today + 7200, "remind me", "Done")]:
        t = Transcript(aenv.cfg, sid)
        t.write({"kind": "turn", "text": text, "ts": ts})
        t.write({"kind": "end", "status": "done", "text": reply, "ts": ts + 5})
    out = render_day(Search(aenv.cfg, aenv.store, aenv.domain).day(today, today + 86400))
    assert "plan my day" in out and "Top 3" in out and "remind me" in out and "yesterday's" not in out
    assert out.index("plan my day") < out.index("remind me")


async def test_overseer_reset_wipes_the_flags_and_asks_about_conversations(aenv, monkeypatch, capsys):
    from overseer.__main__ import reset_command
    from overseer.core.live import Transcript
    from overseer.core.store import WatchRecord
    rem = aenv.domain.reminders
    rem.add(time.time() + 600, "Call the bank")
    aenv.store.set("assistant:onboarding_started", "1")
    aenv.store.set("assistant:log:2026-10-22", "[]")
    s = aenv.agent.general_session()
    s.rec.cli_session_id = "cli-123"
    aenv.store.upsert_session(s.rec)
    Transcript(aenv.cfg, s.id).write({"kind": "turn", "text": "my visa plan"})
    aenv.store.set(f"wakeups_paused:{s.id}", "1")
    aenv.store.set("side_topic:web:1", "77")  # (not a flag: kept)
    aenv.store.add_watch(WatchRecord(session_id=s.id, kind="timer", due=time.time() + 3600, note="check in"))
    Projects(aenv.cfg, aenv.store, aenv.agent).create("japan", "", "trip")
    claude = aenv.cfg.local.resolve(aenv.cfg.agent.claude_code.home)
    (claude / "projects" / "x").mkdir(parents=True)
    (claude / "projects" / "x" / "cli-123.jsonl").write_text("{}")
    (claude / ".credentials.json").write_text("{}")  # (the login: kept)

    def answers(*a):
        it = iter(a)
        return lambda _: next(it)
    # "no" to both: nothing changes.
    assert reset_command(aenv.cfg, yes=False, ask=answers("n", "n")) == 0
    assert aenv.store.get("assistant:onboarding_started") == "1" and aenv.store.get("general_session") == s.id
    # Flags yes, conversations no.
    assert reset_command(aenv.cfg, yes=False, ask=answers("y", "n")) == 0
    assert aenv.store.keys("assistant:") == [] and not aenv.store.get(f"wakeups_paused:{s.id}")
    assert aenv.store.watches() == [] and aenv.store.get("side_topic:web:1") == "77" and rem.pending() == []
    assert aenv.store.get_session(s.id) and (claude / "projects").exists()
    # --yes: flags only; conversations are only forgotten when asked.
    assert reset_command(aenv.cfg, yes=True) == 0
    assert "Conversations kept" in capsys.readouterr().out and aenv.store.get_session(s.id)
    # Now forget them: records, transcripts, the CLI's sessions; logins, projects and topics stay.
    assert reset_command(aenv.cfg, yes=False, ask=answers("y")) == 0  # (no flags left: just that question)
    out = capsys.readouterr().out
    assert "Forgot 1 conversation(s)" in out and "Remove these?" not in out, out
    assert aenv.store.get_session(s.id) is None and aenv.store.get("general_session") is None
    assert not aenv.cfg.local.state("transcripts").exists() and not (claude / "projects").exists()
    assert (claude / ".credentials.json").exists() and aenv.store.get_project("japan")
    assert aenv.store.get("side_topic:web:1") == "77"
    aenv.agent.sessions.clear()
    assert aenv.agent.general_session().id != s.id  # a fresh one
    await aenv.ctl.start()  # while the bot runs: refused
    try:
        aenv.store.set("assistant:x", "1")
        assert reset_command(aenv.cfg, yes=True) == 1 and aenv.store.get("assistant:x") == "1"
    finally:
        await aenv.ctl.stop()

async def test_ov_history_day_through_ov(aenv):
    """FB1: `ov history day` failed - the day's conversations were read in a worker thread that touched the
    store (SQLite objects can only be used in the thread that made them)."""
    from overseer.core.live import Transcript
    s = aenv.agent.general_session()
    t = Transcript(aenv.cfg, s.id)
    t.write({"kind": "turn", "text": "plan my day"})
    t.write({"kind": "end", "status": "done", "text": "Top 3: visa, gym, email"})
    await aenv.ctl.start()
    try:
        token = aenv.ctl.open(s.id)
        for when in ("today", "yesterday", date.today().isoformat()):
            ok, text = await aenv.ctl.handle(token, ["history", "day", when])
            assert "ProgrammingError" not in text and "thread" not in text, text
        ok, text = await aenv.ctl.handle(token, ["history", "day"])
        assert "plan my day" in text and "Top 3: visa, gym, email" in text, text
        ok, text = await aenv.ctl.handle(token, ["history", "day", "yesterday"])
        assert text == "no conversations that day"
    finally:
        await aenv.ctl.stop()



async def test_a_routine_runs_even_while_wake_ups_are_paused(aenv, monkeypatch):
    """A failed run pauses the agent's own wake-ups until the user writes - but routines are the user's schedule:
    they still run, and one that fails (e.g. the network was down) doesn't pause anything."""
    from overseer.assistant.scheduler import CHECKED_KEY
    from overseer.core.autopilot import Autopilot
    ap = Autopilot(aenv.cfg, aenv.store, aenv.agent, aenv.notifier)
    runs = []

    async def handle(s, prompt, quiet=False, origin="user", files=None):
        runs.append(origin)
        return False  # (the run failed)
    monkeypatch.setattr(aenv.agent, "handle", handle)
    s = aenv.agent.general_session()
    aenv.store.set(f"wakeups_paused:{s.id}", "1")  # (paused by an earlier failure)
    (aenv.vault.root / "Assistant/Routines.md").write_text(ROUTINES)
    aenv.store.set(CHECKED_KEY, str(datetime(2026, 10, 19, 8, 59).timestamp()))
    assert await aenv.domain.scheduler.tick(datetime(2026, 10, 19, 9, 1)) == ["Morning planning"]
    await ap.drain()
    assert runs == ["routine"]
    aenv.store.set(f"wakeups_paused:{s.id}", "")
    ap.wake(s, "[Routine: x]", kind="routine", by_user=True)
    await ap.drain()
    assert runs == ["routine", "routine"] and not ap.paused(s.id)  # (a failed routine pauses nothing)
    ap.wake(s, "agent's own", kind="wakeup")
    await ap.drain()
    assert ap.paused(s.id)  # (the agent's own wake-ups still pause on failure)


def test_the_i_have_adhd_style_is_off_unless_turned_on(aenv):
    sid = aenv.agent.general_session().id
    assert "i-have-adhd" not in aenv.agent.system_prompt(sid)
    aenv.cfg.agent.i_have_adhd = True
    prompt = aenv.agent.system_prompt(sid)
    assert prompt.endswith(prompt[prompt.index("## Output style: i-have-adhd (always on)"):])  # (last)
    assert "#### 1. Lead with the next action" in prompt and "## Their calendar" not in prompt.split("i-have-adhd")[-1]


async def test_the_instructions_dont_change_by_themselves(aenv, monkeypatch):
    """What's in the instructions is handed over again whenever it changes (agent.INSTRUCTIONS_KEY): nothing in them
    may change with the day or the hour - today's daily note goes with each request instead."""
    import overseer.assistant.vault as V
    sid = aenv.agent.general_session().id
    first = aenv.agent.system_prompt(sid)
    monkeypatch.setattr(V.Vault, "daily_note", lambda self, day=None: "Daily/2099-01-01.md")
    assert aenv.agent.system_prompt(sid) == first
    assert "<daily_note>Daily/2099-01-01.md</daily_note>" in await aenv.domain.now_context()
