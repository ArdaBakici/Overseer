"""Several logins per agent CLI (agent.<cli>.accounts): the account in use, usage limits and failover (the CLI's next
account, then the other CLI), conversations carried across accounts (the CLI session copied) and across CLIs (a
handover from Overseer's transcripts), /account, logins, and every account's home kept out of the agents' reach."""

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from overseer.core.accounts import Accounts, carry_session, handover, reset_time, usage_limit
from overseer.core.config import ConfigError
from overseer.core.runners import ClaudeCodeRunner
from test_integration import env, make_cfg, short_tmp  # noqa: F401 (fixtures)

CLAUDE = {"personal": "./state/claude-personal", "work": "./state/claude-work"}


def with_accounts(e, claude=None, codex=None):
    """The bot configured with these accounts (as if config.yaml had them when it started)."""
    if claude:
        e.cfg.agent.claude_code.accounts = dict(claude)
    if codex:
        e.cfg.agent.codex.accounts = dict(codex)
    e.cfg.validate()
    e.agent.accounts = Accounts(e.cfg, e.store)
    return e.agent.accounts


def limit(e, *homes):
    """These homes are out of usage (the fake CLIs fail like the real ones at their usage limit)."""
    (e.tmp / "cli.jsonl.limited").write_text("\n".join(str(h) for h in homes))


def runs(e):
    return [c for c in e.cli_log() if c["argv"][:1] != ["debug"] and "--ephemeral" not in c["argv"]
            and "json" not in c["argv"]]


def home_of(call):
    return call["env"].get("CLAUDE_CONFIG_DIR") or call["env"].get("CODEX_HOME")


# ----------------------------------------------------------------------------- usage limits
@pytest.mark.parametrize("text, hit", [
    ("claude: Claude AI usage limit reached|1761300000", True),
    ("claude: You've hit your limit · resets 3pm (Europe/London)", True),
    ("claude: 5-hour limit reached ∙ resets 11pm", True),
    ("codex: You've hit your usage limit. Upgrade to Pro, or try again in 2 hours 5 minutes.", True),
    ("codex: {\"type\":\"usage_limit_reached\",\"resets_in_seconds\":3600}", True),
    ("claude: Credit balance is too low", True),
    ("codex: insufficient_quota: You exceeded your current quota", True),
    ("claude: API Error: 529 overloaded", False),
    ("claude: API Error: 429 rate_limit_error: retry", False),
    ("boom: model unavailable", False),
])
def test_usage_limits_are_told_apart_from_other_errors(text, hit):
    assert usage_limit(text)[0] is hit


def test_when_a_limit_resets():
    now = datetime(2026, 10, 27, 10, 0).timestamp()
    assert reset_time("Claude AI usage limit reached|1761300000", now) == 1761300000
    assert reset_time('"resets_in_seconds": 5400', now) == now + 5400
    assert reset_time("try again in 2 hours 5 minutes.", now) == now + 2 * 3600 + 300
    assert reset_time("try again in 4 days 2 hours", now) == now + 4 * 86400 + 7200
    assert reset_time("try again in 1h 30m", now) == now + 5400
    assert datetime.fromtimestamp(reset_time("You've hit your limit · resets 3pm", now)) == datetime(2026, 10, 27, 15)
    assert datetime.fromtimestamp(reset_time("resets 9:30am", now)) == datetime(2026, 10, 28, 9, 30)  # (tomorrow)
    assert datetime.fromtimestamp(reset_time("resets Oct 30, 9am", now)) == datetime(2026, 10, 30, 9)
    utc = reset_time("resets 3pm (UTC)", now)
    assert datetime.fromtimestamp(utc, timezone.utc).hour == 15 and utc > now
    assert reset_time("usage limit reached", now) is None  # (not said: Overseer assumes an hour)


# ----------------------------------------------------------------------------- the accounts
def test_config(tmp_path, short_tmp):  # noqa: F811
    cfg = make_cfg(tmp_path, short_tmp)
    for accounts, why in (({"my work": "./state/x"}, "short name"), ({"a": "./state/x", "b": "./state/x"}, "own folder"),
                          ({"a": "/etc/claude"}, "outside the allowed"), ({"a": ""}, "the folder of its login")):
        cfg.agent.claude_code.accounts = accounts
        with pytest.raises(ConfigError, match=why):
            cfg.validate()
    cfg.agent.claude_code.accounts = dict(CLAUDE)
    cfg.validate()


async def test_the_account_in_use_and_where_to_go_when_one_is_out(env):  # noqa: F811
    acc = with_accounts(env, CLAUDE, {"default": "./state/codex"})
    assert acc.names("claude_code") == ["personal", "work"] and acc.active("claude_code") == "personal"
    assert env.cfg.agent.claude_code.home == "./state/claude-personal"
    assert ClaudeCodeRunner(env.cfg).home() == env.cfg.local.resolve("./state/claude-personal")
    acc.switch("claude_code", "work")
    assert ClaudeCodeRunner(env.cfg).home() == env.cfg.local.resolve("./state/claude-work")
    assert Accounts(env.cfg, env.store).active("claude_code") == "work"  # (kept: a restart stays on it)
    with pytest.raises(ValueError, match="no account 'nope'"):
        acc.switch("claude_code", "nope")
    # Out of usage: the CLI's next account first, then the other CLI.
    assert acc.next_after_limit("claude_code", "work") == ("claude_code", "personal")
    acc.mark_out("claude_code", "personal", time.time() + 600)
    assert acc.next_after_limit("claude_code", "work") == ("codex", "default")
    acc.mark_out("codex", "default", None)  # (no reset given: an hour)
    assert 3500 < acc.out_until("codex", "default") - time.time() <= 3600
    assert acc.next_after_limit("claude_code", "work") is None
    env.store.set("account_out:claude_code:personal", str(int(time.time() - 1)))  # (its reset has passed)
    assert acc.next_after_limit("claude_code", "work") == ("claude_code", "personal")
    # /account's names: unique ones alone, "<cli> <name>" for both.
    assert acc.find("work") == [("claude_code", "work")] and acc.find("codex default") == [("codex", "default")]
    assert acc.find("claude personal") == [("claude_code", "personal")] and acc.find("nope") == []
    assert acc.find("codex") == [("codex", "default")]
    assert acc.label("claude_code", "work") == "Claude Code (work)" and acc.label("codex") == "Codex"


def test_the_logins_are_out_of_the_agents_reach_and_get_the_skills(env):  # noqa: F811
    from overseer.core.ctl import write_skills
    from overseer.core.reset import conversation_files

    acc = with_accounts(env, {"personal": "./state/claude", "work": "./logins/claude-work"})
    for b in ("claude_code", "codex"):
        for h in acc.all_homes(b):
            h.mkdir(parents=True, exist_ok=True)
            (h / ("projects" if b == "claude_code" else "sessions")).mkdir(exist_ok=True)
    work = env.cfg.local.resolve("./logins/claude-work")
    assert str(work) in ClaudeCodeRunner(env.cfg).deny_read()  # (outside the state dir: denied all the same)
    write_skills(env.cfg, {"ov": "---\nname: ov\n---\nhi\n"})
    assert all((h / "skills" / "ov" / "SKILL.md").exists() for h in acc.all_homes("claude_code"))
    files = conversation_files(env.cfg)
    assert work / "projects" in files and env.cfg.local.resolve("./state/claude") / "projects" in files


async def test_logging_each_account_in(env, capsys):  # noqa: F811
    from overseer.__main__ import login

    with_accounts(env, CLAUDE)
    assert login(env.cfg, "claude_code", []) == 2
    assert "--account <name> (personal, work)" in capsys.readouterr().err
    assert login(env.cfg, "claude_code", [], "nope") == 2
    assert login(env.cfg, "claude_code", [], "work") == 0
    assert home_of(env.cli_log()[-1]) == str(env.cfg.local.resolve("./state/claude-work"))


# ----------------------------------------------------------------------------- carrying conversations
def test_a_cli_session_is_copied_into_the_account_it_continues_in(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    f = a / "projects" / "-tmp-agents-x" / "s1.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("turn 1\n")
    (a / "projects" / "-tmp-agents-x" / "s1" / "subagents").mkdir(parents=True)
    assert carry_session("claude_code", "s1", [a, b], b)
    assert (b / "projects" / "-tmp-agents-x" / "s1.jsonl").read_text() == "turn 1\n"
    assert (b / "projects" / "-tmp-agents-x" / "s1" / "subagents").is_dir()
    f.write_text("turn 1\nturn 2\n")  # it went on in a (newer there): copied again
    later = time.time() + 5
    import os
    os.utime(f, (later, later))
    assert carry_session("claude_code", "s1", [a, b], b)
    assert (b / "projects" / "-tmp-agents-x" / "s1.jsonl").read_text() == "turn 1\nturn 2\n"
    assert not carry_session("claude_code", "nope", [a, b], b)
    assert not carry_session("claude_code", "../x", [a, b], b)
    r = a / "sessions" / "2026" / "10" / "27" / "rollout-2026-10-27T10-00-00-t1.jsonl"
    r.parent.mkdir(parents=True)
    r.write_text("{}\n")
    assert carry_session("codex", "t1", [a, b], b)
    assert (b / r.relative_to(a)).exists()


def test_the_handover_to_the_other_cli(env):  # noqa: F811
    path = env.cfg.local.state("transcripts")
    path.mkdir(parents=True, exist_ok=True)
    evs = []
    for i in range(30):
        evs += [{"kind": "turn", "ts": 1761550000 + i * 60, "backend": "claude_code", "text": f"request {i} " + "x" * 2000},
                {"kind": "command", "command": f"ssh hpc sbatch run{i}.sh"},
                {"kind": "end", "status": "done", "text": f"reply {i}"}]
    evs += [{"kind": "turn", "ts": 1761560000, "text": "/compact"}, {"kind": "end", "status": "done", "text": "x"},
            {"kind": "turn", "ts": 1761560100, "text": "the last one"}, {"kind": "file", "path": "/w/train.py"},
            {"kind": "end", "status": "error", "text": "claude: You've hit your limit"}]
    (path / "c1.jsonl").write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    text = handover(env.cfg, "c1", "Claude Code", "Claude Code ran out of usage")
    assert text.startswith("<conversation_so_far>") and "another AI agent (Claude Code)" in text
    assert "Claude Code ran out of usage" in text and "/compact" not in text
    assert "Researcher: the last one" in text and "Files it changed: /w/train.py" in text
    assert "(The turn didn't finish - an error: claude: You've hit your limit)" in text
    assert "$ ssh hpc sbatch run29.sh" in text and "Agent: reply 29" in text  # the latest in full
    assert "researcher: request 0 xxx" in text and "→ agent: reply 0" in text  # the older ones a line each
    assert "Researcher: request 0 " not in text
    assert len(text) < 40_000
    assert handover(env.cfg, "nope", "Codex", "x") == ""


# ----------------------------------------------------------------------------- failover, end to end (fake CLIs)
async def test_out_of_usage_moves_on_to_the_next_account_then_the_other_cli(env):  # noqa: F811
    acc = with_accounts(env, CLAUDE)
    (env.tmp / "cli.jsonl.strict").write_text("")  # (resuming a session the home doesn't have fails)
    personal, work = (env.cfg.local.resolve(CLAUDE[n]) for n in ("personal", "work"))
    codex_home = env.cfg.local.resolve("./state/codex")
    s = env.agent.general_session()
    assert await env.agent.handle(s, "first request")
    sid = s.rec.cli_session_id
    assert home_of(runs(env)[-1]) == str(personal) and "claude_code (personal)" in env.notifier.texts()

    # personal runs out: the same request again on work, the session carried over (resumed there).
    limit(env, personal)
    assert await env.agent.handle(s, "second request")
    first, again = runs(env)[-2:]
    assert home_of(first) == str(personal) and home_of(again) == str(work)
    assert again["argv"][again["argv"].index("--resume") + 1] == sid
    assert "ran out of usage" in again["stdin"] and again["stdin"].endswith("second request")
    assert acc.active("claude_code") == "work" and acc.out_until("claude_code", "personal") > time.time()
    note = next(m.text for m in env.notifier.sent if m.text.startswith("🔄"))
    assert "Claude Code (personal) is out of usage until" in note and "<b>work</b>" in note
    assert env.notifier.sent[-1].text.startswith("🤖 Fixed") and "claude_code (work)" in env.notifier.sent[-1].text
    rows = env.store.usage_rows(0)
    assert [r["account"] for r in rows] == ["personal", "work"]

    # work runs out too: the whole bot moves to Codex, the conversation handed over.
    limit(env, personal, work)
    assert await env.agent.handle(s, "third request")
    last = runs(env)[-1]
    assert last["prog"] == "codex" and home_of(last) == str(codex_home)
    assert "<conversation_so_far>" in last["stdin"] and "Researcher: first request" in last["stdin"]
    assert "Claude Code ran out of usage" in last["stdin"] and last["stdin"].endswith("third request")
    assert acc.forced == "codex" and env.agent.session_backend(s) == "codex"
    other = env.agent.general_session(reset=True, project=None)
    assert env.agent.session_backend(other) == "codex"  # (every conversation)
    assert "Every conversation runs on <b>Codex</b>" in env.notifier.texts()

    # Codex runs out as well: stopped, kept for ▶️ Continue.
    limit(env, personal, work, codex_home)
    n = len(runs(env))
    assert not await env.agent.handle(s, "fourth request")
    assert len(runs(env)) == n + 1  # (tried once, then nowhere left to go)
    stop = env.notifier.sent[-1]
    assert stop.text.startswith("⏸ <b>Every account is out of usage</b>") and "personal until" in stop.text
    rid = stop.buttons[0][0][1].split(":")[1]
    assert stop.buttons[0][0] == ("▶️ Continue", f"rerun:{rid}")

    # The researcher picks personal again (it's back): Claude Code, the conversation carried back to it.
    limit(env)
    env.agent.use_account("claude_code", "personal")
    assert acc.forced is None and acc.out_until("claude_code", "personal") == 0
    text, files = env.agent.continue_prompt(env.agent.pop_cut_off_run(rid))
    assert text.endswith("fourth request") and "every account it tried was out of usage" in text
    assert await env.agent.handle(s, text)
    last = runs(env)[-1]
    assert last["prog"] == "claude" and home_of(last) == str(personal)
    assert "--resume" not in last["argv"]  # (the conversation went on in Codex since: a handover, not the old session)
    assert "Researcher: third request" in last["stdin"] and "Agent: Resubmitted as job 4343." in last["stdin"]


async def test_failover_stops_after_a_few_accounts_in_a_row(env, monkeypatch):  # noqa: F811
    from overseer.core import agent as agent_mod

    monkeypatch.setattr(agent_mod, "MAX_FAILOVERS", 1)
    with_accounts(env, {"a": "./state/ca", "b": "./state/cb", "c": "./state/cc"})
    limit(env, *(env.cfg.local.resolve(f"./state/c{x}") for x in "abc"))
    s = env.agent.general_session()
    assert not await env.agent.handle(s, "go")
    assert len(runs(env)) == 2
    assert "Usage limits on 2 accounts in a row" in env.notifier.sent[-1].text


async def test_switching_cli_hands_the_conversation_over(env):  # noqa: F811
    s = env.agent.general_session()
    assert await env.agent.handle(s, "train the baseline")
    env.agent.set_backend("codex")
    assert await env.agent.handle(s, "and now?")
    last = runs(env)[-1]
    assert last["prog"] == "codex" and "Researcher: train the baseline" in last["stdin"]
    assert "the researcher switched this conversation to Codex" in last["stdin"]
    assert "Agent: Fixed the batch size and resubmitted as job 4242." in last["stdin"]
    assert await env.agent.handle(s, "thanks")  # (resumed in Codex now: no handover again)
    assert "<conversation_so_far>" not in runs(env)[-1]["stdin"]


async def test_the_account_command(env, monkeypatch):  # noqa: F811
    from overseer.core.bot import Bot

    acc = with_accounts(env, CLAUDE)
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent, notifier=env.notifier)
    replies = []

    async def reply(update, text, buttons=None):
        replies.append((text, buttons))

    monkeypatch.setattr(bot, "reply", reply)
    await bot.cmd_account(None, NS(args=[]))
    text, rows = replies[-1]
    assert "Claude Code: ▶ <b>personal</b> · work" in text and "Codex: ▶ <b>default</b>" in text
    assert rows[0] == [("▶ Claude · personal", "acct:claude_code:personal"), ("Claude · work", "acct:claude_code:work")]
    await bot.cmd_account(None, NS(args=["work"]))
    assert "runs on <b>Claude Code (work)</b>" in replies[-1][0] and acc.active("claude_code") == "work"
    await bot.cmd_account(None, NS(args=["nope"]))
    assert replies[-1][0].startswith("⛔ No account")
    await bot.cmd_account(None, NS(args=["codex"]))
    assert acc.forced == "codex" and env.agent.backend == "codex"
    await bot.cmd_backend(None, NS(args=[]))
    assert "For now every conversation runs on" in replies[-1][0]
    await bot.cmd_backend(None, NS(args=["claude_code"]))
    assert acc.forced is None and env.agent.backend == "claude_code"
    edits = []

    async def edit(q, text, buttons=None):
        edits.append(text)

    async def answer(*a, **k):
        pass

    monkeypatch.setattr(bot, "_edit", edit)
    q = NS(data="acct:claude_code:personal", answer=answer, message=NS(message_id=5))
    await bot.on_button(NS(callback_query=q, effective_chat=NS(id=1)), None)
    assert acc.active("claude_code") == "personal" and "Claude Code (personal)" in edits[-1]
