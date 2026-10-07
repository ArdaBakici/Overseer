"""Entry point: ``python -m overseer run|check -c config.yaml``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys

from .config import Config, ConfigError, load_config


def prepare_local_dirs(cfg: Config) -> None:
    """Create the local state and tmp dirs (inside local.allowed_roots), private to this user."""
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        if d.is_symlink() or d.stat().st_uid != os.getuid():
            raise ConfigError(f"{d} must be a real directory owned by you")
        os.chmod(d, 0o700)


def build(cfg: Config):
    from telegram.ext import Application

    from .agent import Agent
    from .bot import Bot, TelegramNotifier
    from .diagnose import Reporter
    from .hpc import HPC
    from .cli_agents import make_cli_agent
    from .llm import LLM
    from .monitor import Monitor
    from .proposals import Submitter
    from .ssh import SSHRunner
    from .store import Store

    prepare_local_dirs(cfg)
    store = Store(cfg.local.db_path)
    runner = SSHRunner(cfg.ssh, lambda: hpc.write_roots, cfg.local.control_path, cfg.local.audit_log)
    hpc = HPC(cfg, runner)
    hpc.grants = store.grants()  # folders granted from Telegram earlier
    llm = LLM(cfg.llm)
    cli = make_cli_agent(cfg)
    tasks: list[asyncio.Task] = []

    async def post_init(app: Application) -> None:
        tasks.append(asyncio.create_task(monitor.run_forever(), name="monitor"))
        await app.bot.set_my_commands([
            ("jobs", "active and recent jobs"), ("job", "status of a job"), ("log", "tail a job's log"),
            ("fix", "diagnose & fix a job"), ("ask", "talk to the agent"), ("new", "fresh agent conversation"),
            ("pending", "proposals awaiting approval"), ("cancel", "cancel an Overseer job"),
            ("access", "folders Overseer may read/write"), ("grant", "grant folder access"),
            ("revoke", "revoke a granted folder"), ("health", "connection status"), ("help", "help"),
        ])

    async def post_shutdown(app: Application) -> None:
        for t in tasks:
            t.cancel()
        if agent.bridge:
            await agent.bridge.stop()

    app = (
        Application.builder()
        .token(cfg.telegram.token)
        .concurrent_updates(True)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    notifier = TelegramNotifier(app, cfg.telegram.allowed_chat_ids, store)
    reporter = Reporter(cli or llm, cfg.llm.report_effort, cfg.llm.llm_status_updates)
    monitor = Monitor(cfg, hpc, store, notifier, reporter)
    monitor.restore_read_paths()
    submitter = Submitter(cfg, hpc, store, notifier)
    submitter.monitor = monitor
    agent = Agent(cfg, hpc, store, llm, notifier, submitter, cli=cli)
    Bot(cfg, app, store, hpc, monitor, agent, submitter, notifier).register()
    return app


async def check(cfg: Config) -> int:
    """Verify SSH, SLURM, folder access and credentials without starting the bot."""
    from .hpc import HPC
    from .ssh import SSHRunner
    from .store import Store

    ok = True

    def report(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    prepare_local_dirs(cfg)
    report("local folders", True, f"state={cfg.local.state()} tmp={cfg.local.tmp()} "
                                   f"(writes limited to {', '.join(map(str, cfg.local.roots))})")
    hpc = HPC(cfg, SSHRunner(cfg.ssh, lambda: hpc.write_roots, cfg.local.control_path, cfg.local.audit_log))
    hpc.grants = Store(cfg.local.db_path).grants()
    try:
        await hpc.connect()
        report("ssh", True, f"user={hpc.user} home={hpc.home}")
    except Exception as e:
        report("ssh", False, str(e))
        return 1
    print("writable on HPC:\n  " + ("\n  ".join(hpc.write_roots) or "(none - the agent will have to ask)"))
    print("readable on HPC:\n  " + "\n  ".join(hpc.read_roots))
    for m in hpc.missing_roots:
        print(f"note: write root {m} does not exist yet (Overseer may create it)")
    for name, coro in [("squeue", hpc.squeue()), ("sacct", hpc.sacct_recent(60))]:
        try:
            jobs = await coro
            report(name, True, f"{len(jobs)} job(s)")
        except Exception as e:
            report(name, False, str(e))
    try:
        cfg.telegram.token
        report("telegram token", True)
    except ConfigError as e:
        report("telegram token", False, str(e))
    if cfg.llm.enabled and cfg.llm.backend == "claude":
        report("anthropic credentials", bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
               "set ANTHROPIC_API_KEY (or disable llm)")
    elif cfg.llm.enabled:
        from .cli_agents import make_cli_agent

        cli = make_cli_agent(cfg)
        cmd = getattr(cfg.llm, cfg.llm.backend).command
        report(f"{cfg.llm.backend} CLI", bool(shutil.which(cmd)), shutil.which(cmd) or f"{cmd!r} not on PATH")
        try:
            reply = await cli.text(system="You are a connectivity check.", prompt="Reply with exactly: OK")
            report(f"{cfg.llm.backend} model call", "OK" in reply, reply[:200])
        except Exception as e:
            report(f"{cfg.llm.backend} model call", False,
                   f"{e}\n  (log in with: python -m overseer login {cfg.llm.backend} -c <config>)")
    return 0 if ok else 1


def login(cfg: Config, backend: str, extra: list[str]) -> int:
    from .cli_agents import CodexAgent, OpencodeAgent

    if backend not in ("codex", "opencode"):
        print("usage: overseer login codex|opencode", file=sys.stderr)
        return 2
    prepare_local_dirs(cfg)
    agent = CodexAgent(cfg) if backend == "codex" else OpencodeAgent(cfg)
    argv, env = agent.login_command(extra)
    if not shutil.which(argv[0], path=env.get("PATH")):
        print(f"{argv[0]!r} not found on PATH (install it, or set llm.{backend}.command)", file=sys.stderr)
        return 1
    print(f"Running {' '.join(argv)} with its home in {cfg.local.resolve(getattr(cfg.llm, backend).home)}")
    return subprocess.call(argv, env=env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="overseer",
        epilog="login: `overseer login codex|opencode [-- extra args]` logs the CLI in using the isolated "
               "home Overseer runs it with (inside local.state_dir).")
    ap.add_argument("command", choices=["run", "check", "login"])
    ap.add_argument("backend", nargs="?", choices=["codex", "opencode"], help="for login")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:  # everything after `--` goes to the CLI's login command
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    try:
        cfg = load_config(args.config)
    except (OSError, ConfigError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    try:
        if args.command == "check":
            return asyncio.run(check(cfg))
        if args.command == "login":
            return login(cfg, args.backend or cfg.llm.backend, extra)
        app = build(cfg)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    app.run_polling(allowed_updates=["message", "callback_query"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
