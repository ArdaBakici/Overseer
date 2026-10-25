"""The HPC bot's folder in the Obsidian vault (obsidian.folder): the config, the sandbox (that folder and nothing else
of the vault), research logs there (and imported from the HPC), the notes Overseer writes (findings, results,
graphs), and the assistant's view of it (read-only)."""

import json
import shutil
import tempfile
import time

import pytest

from overseer.core.agent import Agent
from overseer.core.config import ConfigError
from overseer.core.runners import ClaudeCodeRunner, CodexRunner
from overseer.core.store import FindingRecord, ProjectRecord, ResultRecord, Store
from overseer.hpc.domain import HpcDomain, hpc_sandbox
from overseer.hpc.obsidian import GENERATED
from overseer.hpc.remote import HPC
from test_integration import FakeNotifier, LocalRunner, make_cfg


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "Vault"
    (v / ".obsidian").mkdir(parents=True)
    (v / "Personal").mkdir()
    (v / "Personal" / "Journal.md").write_text("private\n")
    (v / "Ideas.md").write_text("private\n")
    (v / "HPC").mkdir()
    return v


@pytest.fixture
def henv(tmp_path, vault, monkeypatch):
    short = tempfile.mkdtemp(prefix="ovv-", dir="/tmp")
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    cfg = make_cfg(tmp_path, short)
    cfg.obsidian.folder = str(vault / "HPC")
    cfg.validate()
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.local.db_path)
    notifier = FakeNotifier()
    domain = HpcDomain(cfg, store, notifier, HPC(LocalRunner(cfg)))
    agent = Agent(cfg, store, notifier, domain)
    store.upsert_project(ProjectRecord(name="dreamer", dir=str(home / "proj"), created=time.time()))
    yield type("E", (), {"cfg": cfg, "store": store, "domain": domain, "agent": agent, "vault": vault,
                         "home": home, "folder": vault / "HPC"})
    shutil.rmtree(short, ignore_errors=True)


def test_config(tmp_path, vault):
    short = tempfile.mkdtemp(prefix="ovv-", dir="/tmp")
    try:
        cfg = make_cfg(tmp_path, short)
        assert cfg.obsidian_dir is None and hpc_sandbox(cfg).writable == []
        for folder, why in (("HPC", "absolute"), (str(vault / "Nope"), "doesn't exist"),
                            (str(vault), "isn't inside an Obsidian vault"), (str(tmp_path), "isn't inside")):
            cfg.obsidian.folder = folder
            with pytest.raises(ConfigError, match=why):
                cfg.validate()
        cfg.obsidian.folder = str(vault / "HPC")
        cfg.validate()
        assert cfg.obsidian_dir == vault / "HPC" and cfg.obsidian_vault == vault
        from overseer.hpc.config import registry
        assert registry(cfg) == {"vault_folder": str(vault / "HPC")}
    finally:
        shutil.rmtree(short, ignore_errors=True)


def test_the_sandbox_is_the_folder_and_nothing_else_of_the_vault(henv):
    v = henv.vault
    (v / "HPC" / "sub").mkdir()
    sb = hpc_sandbox(henv.cfg)
    assert sb.writable == [v / "HPC"] and sb.deny_read_shell == [v]
    assert set(sb.deny_read) >= {v / ".obsidian", v / "Personal", v / "Ideas.md"} and v / "HPC" not in sb.deny_read
    # Claude Code: the shell sandbox denies the vault (the folder allowed in it); its file tools, every other entry.
    runner = ClaudeCodeRunner(henv.cfg, sb)
    st = runner.settings(henv.cfg.local.tmp("agents", "x"))
    fs = st["sandbox"]["filesystem"]
    assert str(v) in fs["denyRead"] and str(v / "HPC") in fs["allowWrite"]
    rules = st["permissions"]["deny"]
    assert f"Read(/{v / 'Personal'}/**)" in rules and f"Read(/{v / 'Ideas.md'})" in rules
    assert not any(r.startswith(f"Read(/{v}/**)") or r == f"Read(/{v})" for r in rules)  # (would block HPC/ too)
    assert str(v / "HPC") in st["permissions"]["additionalDirectories"]
    # Codex: the vault denied, the folder writable inside it.
    over = " ".join(CodexRunner(henv.cfg, sb).overrides(henv.cfg.local.tmp("agents", "x"), system=None,
                                                         writable=True, network=False))
    assert f'"{v}" = "deny"' in over and f'"{v / "HPC"}" = "write"' in over


async def test_research_logs_live_in_the_vault(henv):
    d = henv.domain
    # The old log on the HPC is copied in once.
    (henv.home / "proj" / "RESEARCH_LOG.md").write_text("# Goal\nworld models\n## 2026-10-01\nfirst run\n")
    text = await d.research_log("dreamer")
    log = henv.folder / "dreamer" / "Research log.md"
    assert log.exists() and "first run" in log.read_text() and log.read_text().startswith("---\nproject: dreamer")
    assert f"<research_log path='{log}'>" in text and "first run" in text
    (henv.home / "proj" / "RESEARCH_LOG.md").write_text("changed on the HPC since\n")
    log.write_text("# Goal\nedited in Obsidian\n")
    assert "edited in Obsidian" in await d.research_log("dreamer")  # (the vault's copy is the log now)
    # Outside projects: General.md.
    general = await d.research_log(None)
    assert f"path='{henv.folder / 'General.md'}'" in general and "none yet" in general
    # The instructions: the folder, and the vault's log path.
    s = henv.agent.general_session()
    prompt = d.system_prompt(s)
    assert f"## Your folder in the researcher's Obsidian vault: `{henv.folder}`" in prompt
    assert f"Research log outside projects: `{henv.folder / 'General.md'}`" in prompt
    s.rec.project = "dreamer"
    assert f"Research log: `{log}`" in d.system_prompt(s)
    ctx = await d.wake_context(s)
    assert "edited in Obsidian" in ctx[0]
    assert dict((n, ok) for n, ok, _ in await d.check())["obsidian"]


def test_findings_results_and_graphs_are_written_into_the_vault(henv):
    d, store = henv.domain, henv.store
    assert d.vault.sync() == []  # nothing recorded yet: nothing written
    store.add_finding(FindingRecord(project="dreamer", kind="insight", text="lr 3e-4 is stable", created=time.time()))
    store.add_finding(FindingRecord(project="dreamer", kind="dead-end", text="no BN in the world model",
                                    job_id="5001", created=time.time()))
    store.save_result(ResultRecord(id="5001", job_id="5001", project="dreamer", name="ne-ratio32", state="COMPLETED",
                                   ended=time.time() - 60, verdict="promising", conclusion="better | than baseline",
                                   metrics=json.dumps({"return": "812", "achievement_pct": "41"})))
    store.save_result(ResultRecord(id="5002", job_id="5002", project="dreamer", name="running", state="RUNNING"))
    cache = henv.cfg.local.state("plots", "5001")
    cache.mkdir(parents=True)
    (cache / "00-return.png").write_bytes(b"\x89PNG small")
    (cache / "01-huge.png").write_bytes(b"x" * (henv.cfg.obsidian.max_figure_mb * 1024 * 1024 + 1))
    run = henv.cfg.local.state("plots", "5002")
    run.mkdir(parents=True)
    (run / "00-return.png").write_bytes(b"\x89PNG running")
    written = d.vault.sync()
    p = henv.folder / "dreamer"
    assert sorted(written) == [str(p / "Findings.md"), str(p / "Results.md")]
    findings = (p / "Findings.md").read_text()
    assert findings.startswith("---\nproject: dreamer\ngenerated: overseer") and GENERATED in findings
    assert "## 💡 Insights" in findings and "- **F1** lr 3e-4 is stable" in findings
    assert "## ⛔ Dead ends" in findings and "no BN in the world model (job 5001)" in findings
    results = (p / "Results.md").read_text()
    assert "| result | name | state | verdict |" in results and "achievement_pct" in results
    assert "[[#5001 ne-ratio32\\|5001]]" in results and "better \\| than baseline" in results  # (| escaped)
    assert "## 5001 ne-ratio32" in results and "![[5001-return.png]]" in results
    assert (p / "figures" / "5001-return.png").read_bytes() == b"\x89PNG small"
    assert not (p / "figures" / "5001-huge.png").exists()  # (over obsidian.max_figure_mb)
    assert not (p / "figures" / "5002-return.png").exists()  # (a running job's graphs: only when it ends)
    assert d.vault.sync() == []  # unchanged: not written again
    store.add_finding(FindingRecord(project="dreamer", kind="insight", text="another", created=time.time()))
    assert d.vault.sync() == [str(p / "Findings.md")]


def test_the_assistant_sees_the_folder_but_cant_change_it(henv, tmp_path, monkeypatch):
    from overseer.assistant.domain import AssistantDomain
    from test_assistant import FakeNotifier as AFake, assistant_cfg
    henv.cfg.local.register("hpc", {"vault_folder": str(henv.folder)})
    acfg = assistant_cfg(tmp_path)
    acfg.vault.path = str(henv.vault)
    acfg.local.base_dir = henv.cfg.local.base_dir  # (the same Overseer folder: they see each other's notes)
    acfg.validate()
    for dd in (acfg.local.state(), acfg.local.tmp()):
        dd.mkdir(parents=True, exist_ok=True)
    ad = AssistantDomain(acfg, Store(acfg.local.db_path), AFake())
    assert ad.bot_folders() == [henv.folder]
    assert henv.folder in ad.sandbox().deny_write and ad.sandbox().writable == [henv.vault]
    Agent(acfg, ad.store, ad.notifier, ad)
    prompt = ad.system_prompt(None)
    assert "## The HPC bot's notes: `HPC/`" in prompt and "you can't change them" in prompt
    import asyncio
    asyncio.run(ad.start(None, None))
    if ad.vault.history:
        exclude = (acfg.vault_git_dir / "info" / "exclude").read_text()
        assert "/HPC/**/figures/" in exclude
