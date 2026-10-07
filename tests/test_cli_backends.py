"""End-to-end tests of the codex / opencode backends with the REAL CLIs and a fake model server.

The CLI talks to tests/fake_model_server.py instead of OpenAI/Anthropic, starts Overseer's MCP
server, and the scripted model calls an Overseer tool through it. Skipped when the CLI isn't
installed (set OVERSEER_TEST_CLI_DIR to a node_modules/.bin containing codex / opencode).
"""

import os
import shutil

import pytest

from fake_model_server import FakeModel
from overseer.agent import Agent
from overseer.cli_agents import CodexAgent, OpencodeAgent, _toml
from test_integration import make_env


def find_cli(name):
    extra = os.environ.get("OVERSEER_TEST_CLI_DIR", "")
    return shutil.which(name, path=f"{extra}:{os.environ.get('PATH', '')}" if extra else None)


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = make_env(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    e.cfg.llm.enabled = True
    return e


def codex_setup(env, fm, cli):
    env.cfg.llm.backend = "codex"
    c = env.cfg.llm.codex
    c.command = cli
    c.pass_env = [*c.pass_env, "FAKE_KEY"]
    provider = {"name": "fake", "base_url": fm.url, "wire_api": "responses", "env_key": "FAKE_KEY",
                "request_max_retries": 0, "stream_max_retries": 0}
    c.extra_config = ['model_provider="fake"', 'model="fake-model"', f"model_providers.fake={_toml(provider)}"]
    return CodexAgent(env.cfg)


def opencode_setup(env, fm, cli):
    env.cfg.llm.backend = "opencode"
    o = env.cfg.llm.opencode
    o.command = cli
    o.model = "fake/fake-model"
    o.extra_config = {
        "provider": {"fake": {"npm": "@ai-sdk/openai-compatible", "name": "Fake",
                              "options": {"baseURL": fm.url, "apiKey": "sk-fake"},
                              "models": {"fake-model": {"name": "Fake model", "tool_call": True}}}},
        "small_model": "fake/fake-model",
        # extra_config must not be able to re-enable built-in tools:
        "tools": {"bash": True}, "permission": {"bash": "allow"},
    }
    return OpencodeAgent(env.cfg)


@pytest.mark.parametrize("backend", ["codex", "opencode"])
async def test_cli_backend_end_to_end(env, backend):
    cli = find_cli(backend)
    if not cli:
        pytest.skip(f"{backend} CLI not installed")
    await env.hpc.connect()
    with FakeModel(target_tool="read_file", target_args={"path": "~/proj/train.py"}) as fm:
        runner = (codex_setup if backend == "codex" else opencode_setup)(env, fm, cli)
        agent = Agent(env.cfg, env.hpc, env.store, None, env.notifier, env.submitter, cli=runner)
        s = agent.general_session()
        await agent.handle(s, "What batch size does train.py use?")
        texts = env.notifier.texts()
        assert "DONE: BATCH = 512" in texts, texts
        assert "read" in texts and "train.py" in texts  # reported via the "Agent actions" message
        assert s.cli_session_id

        tool_requests = [r for r in fm.requests if fm.tool_names(r["body"])]
        names = set(fm.tool_names(tool_requests[0]["body"]))
        overseer = {n for n in names if "overseer" in n}
        # MCP-resource helpers only reach MCP servers (= Overseer's, which has none); asking the user is harmless.
        harmless = {n for n in names if "mcp_resource" in n or n in ("request_user_input", "question")}
        assert any(n.endswith("read_file") for n in overseer)
        # No built-in tool that could act on this machine is offered to the model.
        dangerous = {n for n in names - overseer - harmless
                     if any(w in n.lower() for w in ("shell", "exec", "bash", "patch", "edit", "write", "read", "grep",
                                                     "glob", "list", "fetch", "browser", "computer", "image"))}
        assert not dangerous, f"built-in tools exposed: {sorted(dangerous)} (all: {sorted(names)})"

        # Second turn resumes the same CLI session.
        first_sid = s.cli_session_id
        await agent.handle(s, "thanks")
        assert s.cli_session_id == first_sid

        # The token is revoked after the run: nothing can call tools in between turns.
        assert agent.bridge._runs == {}

        # One-shot report call (no tools).
        assert (await runner.text(system="Summarise.", prompt="log")).strip() == "OK"

    # The CLI wrote nothing into the real $HOME (it gets its own HOME/CODEX_HOME/XDG dirs in the state dir).
    assert sorted(p.name for p in env.home.iterdir()) == [".bashrc", ".ssh", "proj"]
    cli_home = env.cfg.local.resolve(getattr(env.cfg.llm, backend).home)
    assert cli_home.is_dir() and env.cfg.local.check(cli_home)
    assert env.cfg.local.tmp("agents").is_dir()


async def test_mcp_server_protocol_and_tokens(env):
    """Drive `python -m overseer.mcp_server` over stdio like a CLI would (no codex/opencode needed)."""
    import asyncio
    import json
    import sys

    await env.hpc.connect()
    agent = Agent(env.cfg, env.hpc, env.store, None, env.notifier, env.submitter, cli=OpencodeAgent(env.cfg))
    s = agent.general_session()
    actions = []

    async def session(bridge_env, messages):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "overseer.mcp_server", env={**os.environ, **bridge_env},
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        data = "".join(json.dumps(m) + "\n" for m in messages).encode()
        out, _ = await proc.communicate(data)
        return [json.loads(line) for line in out.decode().splitlines()]

    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "~/proj/train.py"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "write_file", "arguments": {"path": "~/proj/x.py", "content": "x"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "nope"},
    ]
    async with agent.bridge.run(s, actions) as bridge_env:
        out = await session(bridge_env, msgs)
        assert (agent.cfg.local.tmp("bridge.sock").stat().st_mode & 0o777) == 0o600
    by_id = {m["id"]: m for m in out}
    assert by_id[1]["result"]["protocolVersion"] == "2025-03-26"
    assert {t["name"] for t in by_id[2]["result"]["tools"]} >= {"read_file", "propose_submission", "request_access"}
    assert "BATCH = 512" in by_id[3]["result"]["content"][0]["text"] and not by_id[3]["result"]["isError"]
    assert by_id[4]["result"]["isError"] and "request_access" in by_id[4]["result"]["content"][0]["text"]
    assert by_id[5]["error"]["code"] == -32601
    assert not (env.home / "proj" / "x.py").exists()
    assert any("read" in a for a in actions) and any("blocked write_file" in a for a in actions)

    # After the run the token is dead.
    out = await session(bridge_env, msgs[2:3])
    assert "invalid or expired token" in out[0]["error"]["message"]
