"""Entry point: ``python -m overseer run|check -c config.yaml``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
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

    app = (
        Application.builder()
        .token(cfg.telegram.token)
        .concurrent_updates(True)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    notifier = TelegramNotifier(app, cfg.telegram.allowed_chat_ids, store)
    reporter = Reporter(llm, cfg.llm.report_effort, cfg.llm.llm_status_updates)
    monitor = Monitor(cfg, hpc, store, notifier, reporter)
    monitor.restore_read_paths()
    submitter = Submitter(cfg, hpc, store, notifier)
    submitter.monitor = monitor
    agent = Agent(cfg, hpc, store, llm, notifier, submitter)
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
    if cfg.llm.enabled:
        report("anthropic credentials", bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
               "set ANTHROPIC_API_KEY (or disable llm)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="overseer")
    ap.add_argument("command", choices=["run", "check"])
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
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
        app = build(cfg)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    app.run_polling(allowed_updates=["message", "callback_query"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
