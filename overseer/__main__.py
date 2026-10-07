"""Entry point: ``python -m overseer run|check -c config.yaml``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .config import Config, ConfigError, load_config


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

    store = Store(cfg.state.db_path)
    runner = SSHRunner(cfg.ssh, cfg.sandbox.root, cfg.state.audit_log)
    hpc = HPC(cfg, runner)
    llm = LLM(cfg.llm)
    tasks: list[asyncio.Task] = []

    async def post_init(app: Application) -> None:
        tasks.append(asyncio.create_task(monitor.run_forever(), name="monitor"))
        await app.bot.set_my_commands([
            ("jobs", "active and recent jobs"), ("job", "status of a job"), ("log", "tail a job's log"),
            ("fix", "diagnose & fix a job"), ("ask", "talk to the agent"), ("new", "fresh agent conversation"),
            ("pending", "proposals awaiting approval"), ("cancel", "cancel an Overseer job"),
            ("health", "connection status"), ("help", "help"),
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
    """Verify SSH, SLURM, sandbox and credentials without starting the bot."""
    from .hpc import HPC
    from .ssh import SSHRunner

    ok = True

    def report(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    hpc = HPC(cfg, SSHRunner(cfg.ssh, cfg.sandbox.root, cfg.state.audit_log))
    try:
        await hpc.connect()
        report("ssh + sandbox", True, f"user={hpc.user} home={hpc.home} sandbox={hpc.root} (mode 700, owned by you)")
    except Exception as e:
        report("ssh + sandbox", False, str(e))
        return 1
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
    if cfg.sandbox.root.startswith("/tmp/"):
        print("note: /tmp is usually local to each node. Jobs Overseer submits run from the sandbox, so compute nodes\n"
              "      must see it. If your cluster's /tmp is node-local, point sandbox.root at shared scratch\n"
              "      (e.g. /scratch/$USER/overseer). See README 'Where should the sandbox live?'.")
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
    if args.command == "check":
        return asyncio.run(check(cfg))
    app = build(cfg)
    app.run_polling(allowed_updates=["message", "callback_query"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
