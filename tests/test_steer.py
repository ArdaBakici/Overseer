"""Messages sent while the agent works (steering): they go into its run - it reads them at its next step - for both
CLIs; ones it can't take any more are the next request; /after waits for the run; a planner's message reaches a
working subagent the same way."""

import asyncio
import json
from types import SimpleNamespace as NS

from overseer.core.agent import LATE_NOTE, STEER_NOTE
from overseer.core.runners import RunResult
from test_integration import env, short_tmp  # noqa: F401 (fixtures)


async def started(env, s, seconds=10):  # noqa: F811
    """Until the run's CLI is up and has started its command (it takes messages from then on)."""
    path = env.cfg.local.state("transcripts") / f"{s.id}.jsonl"
    for _ in range(int(seconds * 20)):
        if s.steering and path.exists() and '"kind": "command"' in path.read_text():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the run didn't get going")


def events(env, s):  # noqa: F811
    path = env.cfg.local.state("transcripts") / f"{s.id}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


async def steer_mid_run(env, backend):  # noqa: F811
    env.agent.set_backend(backend)
    s = env.agent.general_session()
    run = asyncio.create_task(env.agent.handle(s, "FAKE_STEER train the baseline"))
    await started(env, s)
    assert await env.agent.steer(s, "use lr 1e-4 instead")
    assert await run
    reply = env.notifier.sent[-1].text
    assert "STEERED: " + STEER_NOTE["user"].strip() in reply and "use lr 1e-4 instead" in reply, reply
    steers = [e for e in events(env, s) if e["kind"] == "steer"]
    assert len(steers) == 1 and steers[0]["text"].endswith("use lr 1e-4 instead")
    assert sum(1 for e in events(env, s) if e["kind"] == "turn") == 1  # (the same request: no new turn)
    assert not await env.agent.steer(s, "too late")  # (nothing running: a request of its own)
    return s


async def test_a_message_reaches_claude_code_mid_run(env):  # noqa: F811
    await steer_mid_run(env, "claude_code")
    argv = env.cli_log()[-1]["argv"]
    assert "--input-format" in argv and "--replay-user-messages" in argv


async def test_a_message_reaches_codex_mid_run(env):  # noqa: F811
    s = await steer_mid_run(env, "codex")
    assert env.cli_log()[-1]["argv"][0] == "app-server"
    first = s.rec.cli_session_id
    assert await env.agent.handle(s, "and now?")  # (its thread goes on: resumed through the app server)
    assert s.rec.cli_session_id == first and "Resubmitted as job 4343." in env.notifier.sent[-1].text


async def test_a_message_too_late_for_the_run_is_the_next_request(env, monkeypatch):  # noqa: F811
    """The run ended before taking it (it was finishing its answer): it runs next, said to be late - not lost."""
    runner = env.agent.runner("claude_code")
    prompts = []

    async def run(prompt, *, system, session_id, key, activity, steering=None):
        prompts.append(prompt)
        if len(prompts) == 1:
            await steering.ready(lambda text: asyncio.sleep(0, True))  # (handed over, never taken)
            await asyncio.sleep(0.3)
        return RunResult(text=f"answer {len(prompts)}", session_id="sid")
    monkeypatch.setattr(runner, "run", run)
    s = env.agent.general_session()
    task = asyncio.create_task(env.agent.handle(s, "first"))
    while not s.steering:
        await asyncio.sleep(0.01)
    assert await env.agent.steer(s, "also plot it")
    assert await task
    assert len(prompts) == 2 and prompts[1].endswith(LATE_NOTE + "also plot it")
    assert [m.text.split("\n")[0] for m in env.notifier.sent[-2:]] == ["🤖 answer 1", "🤖 answer 2"]


async def test_a_stopped_run_says_what_it_didnt_read(env, monkeypatch):  # noqa: F811
    runner = env.agent.runner("claude_code")

    async def run(prompt, *, system, session_id, key, activity, steering=None):
        await steering.ready(lambda text: asyncio.sleep(0, True))
        await asyncio.sleep(30)
    monkeypatch.setattr(runner, "run", run)
    s = env.agent.general_session()
    task = asyncio.create_task(env.agent.handle(s, "long work"))
    while not s.steering:
        await asyncio.sleep(0.01)
    assert await env.agent.steer(s, "skip seed 3")
    await env.agent.stop(s.id)
    await task
    note = env.notifier.sent[-1].text
    assert note.startswith("📨 What you sent while it worked didn't reach the agent") and "skip seed 3" in note
    assert STEER_NOTE["user"].strip() not in note  # (what you wrote, not Overseer's note to the agent)


async def test_telegram_steers_by_default_and_after_waits(env, monkeypatch):  # noqa: F811
    from overseer.core.bot import STEERED_NOTE, Bot
    bot = Bot(env.cfg, app=None, store=env.store, agent=env.agent, notifier=env.notifier)
    s = env.agent.general_session()
    steered, handled = [], []

    async def steer(session, text, files=None, by="user"):
        steered.append(text)
        return True

    async def handle(session, text, **kw):
        handled.append(text)
        return True
    monkeypatch.setattr(env.agent, "steer", steer)
    monkeypatch.setattr(env.agent, "handle", handle)
    monkeypatch.setattr(env.agent, "is_running", lambda session: True)
    await bot._run_agent(1, s, "use lr 1e-4")
    assert steered == ["use lr 1e-4"] and handled == [] and env.notifier.sent[-1].text == STEERED_NOTE
    monkeypatch.setattr(bot, "_here", lambda update: s)
    monkeypatch.setattr(bot, "_outside_topics", lambda update: False)
    monkeypatch.setattr(bot, "_replied_session", lambda update: None)
    await bot.cmd_after(NS(effective_message=None, effective_chat=NS(id=1)), NS(args=["then", "plot", "it"]))
    for t in list(bot._tasks):
        await t
    assert steered == ["use lr 1e-4"] and handled == ["then plot it"]  # (queued behind the run)
    assert "yours is queued" in env.notifier.texts()


async def test_a_planner_message_reaches_a_working_subagent(env, monkeypatch):  # noqa: F811
    from overseer.core.subagents import Subagents
    subs = Subagents(env.cfg, env.store, env.agent, env.notifier)
    w = env.agent.new_conversation()
    seen = []

    async def steer(session, text, files=None, by="user"):
        seen.append((session.id, text, by))
        return True
    monkeypatch.setattr(env.agent, "steer", steer)
    assert await subs.send(w, "also try warmup 1k")
    assert seen == [(w.id, "also try warmup 1k", "planner")]
