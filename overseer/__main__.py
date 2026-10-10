"""Entry point.

    overseer ssh setup [--import KEY]   create Overseer's own SSH key/config in the Overseer folder
    overseer ssh connect                open the SSH connection interactively (for 2FA clusters)
    overseer ssh test                   check the connection works
    overseer login claude_code|codex    log the agent CLI in (stored in the state dir)
    overseer check                      verify everything without starting the bot
    overseer run                        start the bot
All take -c/--config (default: config.yaml).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys

from .config import BACKENDS, Config, ConfigError, load_config


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
    from .monitor import Monitor
    from .ssh import SSHRunner
    from .sshlink import write_configs
    from .store import Store

    prepare_local_dirs(cfg)
    write_configs(cfg)
    store = Store(cfg.local.db_path)
    hpc = HPC(SSHRunner(cfg), cfg.slurm.setup)
    tasks: list[asyncio.Task] = []

    async def post_init(app: Application) -> None:
        tasks.append(asyncio.create_task(monitor.run_forever(), name="monitor"))
        await app.bot.set_my_commands([
            ("jobs", "active and recent jobs"), ("job", "status of a job"), ("log", "tail a job's log"),
            ("fix", "ask the agent to fix a job"), ("ask", "talk to the agent"), ("new", "fresh agent conversation"),
            ("backend", "show/switch claude_code or codex"), ("model", "show/set the agent's model"),
            ("effort", "show/set the agent's effort"), ("stop", "stop the agent's current run"),
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
    agent = Agent(cfg, store, notifier, hpc)

    async def report_text(system: str, prompt: str) -> str:
        return await agent.runner().text(system=system, prompt=prompt)

    reporter = Reporter(report_text, cfg.agent.enabled and cfg.agent.job_reports, cfg.agent.llm_status_updates)
    monitor = Monitor(cfg, hpc, store, notifier, reporter)
    agent.monitor = monitor
    Bot(cfg, app, store, hpc, monitor, agent, notifier).register()
    return app


async def check(cfg: Config) -> int:
    """Verify SSH, SLURM, the agent CLI and credentials without starting the bot."""
    from .agent import Agent
    from .hpc import HPC
    from .ssh import SSHRunner
    from .sshlink import ensure_master, key_permissions_ok, write_configs
    from .store import Store

    ok = True

    def report(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    prepare_local_dirs(cfg)
    report("local folders", True, f"state={cfg.local.state()} tmp={cfg.local.tmp()}")
    write_configs(cfg)
    report("ssh key", key_permissions_ok(cfg), str(cfg.ssh_key) if key_permissions_ok(cfg)
           else f"missing or group/world-readable: {cfg.ssh_key} (run: overseer ssh setup)")
    try:
        await ensure_master(cfg)
        hpc = HPC(SSHRunner(cfg), cfg.slurm.setup)
        await hpc.connect()
        report("ssh connection", True, f"{cfg.ssh.host} -> {hpc.user}@{cfg.ssh.hostname} (home {hpc.home})")
    except Exception as e:
        report("ssh connection", False, str(e))
        return 1
    slurm_ok = True
    for name, coro in [("squeue", hpc.squeue()), ("sacct", hpc.sacct_recent(60))]:
        try:
            jobs = await coro
            report(name, True, f"{len(jobs)} job(s)")
        except Exception as e:
            slurm_ok = False
            report(name, False, str(e))
    if not slurm_ok:
        print("  SLURM isn't on PATH even in a login shell on the HPC. Log in and run `type squeue`;\n"
              "  if you normally need e.g. `module load slurm`, put that in slurm.setup in config.yaml.\n"
              f"  (login-shell PATH Overseer saw: {hpc.path or 'none - could not run bash -lc'})")
    try:
        cfg.telegram.token
        report("telegram token", True)
    except ConfigError as e:
        report("telegram token", False, str(e))
    if cfg.agent.enabled:
        store = Store(cfg.local.db_path)
        agent = Agent(cfg, store, None, hpc)  # type: ignore[arg-type]
        for backend in BACKENDS:
            runner = agent.runner(backend)
            cmd = runner.c.command
            if not shutil.which(cmd):
                print(f"[skip] {backend}: {cmd!r} not installed")
                continue
            try:
                reply = await asyncio.wait_for(
                    runner.text(system="You are a connectivity check.", prompt="Reply with exactly: OK"), 120)
                report(f"{backend} model call", "OK" in reply, reply[:200])
            except asyncio.TimeoutError:
                report(f"{backend} model call", False,
                       f"no answer within 2 minutes (not logged in?)\n  (log in with: overseer login {backend} -c <config>)")
            except Exception as e:
                report(f"{backend} model call", False,
                       f"{e}\n  (log in with: overseer login {backend} -c <config>)")
        print(f"active agent backend: {agent.backend}")
    return 0 if ok else 1


def ssh_command(cfg: Config, action: str, import_from: str | None) -> int:
    from .sshlink import connect_interactive, ensure_master, setup_key

    prepare_local_dirs(cfg)
    if action == "setup":
        pub = setup_key(cfg, import_from)
        print(f"Overseer's SSH files are in {cfg.ssh_dir} (config, known_hosts, {cfg.ssh.key_name}).")
        print("\nAdd this public key to ~/.ssh/authorized_keys on the HPC (one line):\n")
        print(f"  {pub}\n")
        print("For example, from a machine where you can already log in:")
        print(f"  echo '{pub}' | ssh {cfg.ssh.user + '@' if cfg.ssh.user else ''}{cfg.ssh.hostname} "
              "'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys'")
        print("\nThen run: overseer ssh test")
        return 0
    if action == "connect":
        return connect_interactive(cfg)
    if action == "test":
        try:
            asyncio.run(ensure_master(cfg))
        except ConfigError as e:
            print(f"FAIL: {e}")
            return 1
        agent_ssh = cfg.local.tmp("agent-bin", "ssh")
        out = subprocess.run([str(agent_ssh), cfg.ssh.host, "echo connected as $(whoami) on $(hostname)"],
                             capture_output=True, text=True, timeout=60)
        print(out.stdout.strip() or out.stderr.strip())
        return out.returncode
    print("usage: overseer ssh setup|connect|test", file=sys.stderr)
    return 2


def login(cfg: Config, backend: str, extra: list[str]) -> int:
    from .runners import make_runner

    if backend not in BACKENDS:
        print(f"usage: overseer login {'|'.join(BACKENDS)}", file=sys.stderr)
        return 2
    prepare_local_dirs(cfg)
    runner = make_runner(cfg, backend)
    argv, env = runner.login_command(extra)
    if not shutil.which(argv[0], path=env.get("PATH")):
        print(f"{argv[0]!r} not found on PATH (install it, or set agent.{backend}.command)", file=sys.stderr)
        return 1
    print(f"Running {' '.join(argv)} with its config in {runner.home()}")
    return subprocess.call(argv, env=env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="overseer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "check", "login", "ssh"])
    ap.add_argument("target", nargs="?", help="login: claude_code|codex; ssh: setup|connect|test")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--import", dest="import_key", help="ssh setup: copy this existing private key instead of generating one")
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
    try:
        cfg = load_config(args.config)
    except (OSError, ConfigError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    try:
        if args.command == "check":
            return asyncio.run(check(cfg))
        if args.command == "login":
            return login(cfg, args.target or cfg.agent.backend, extra)
        if args.command == "ssh":
            return ssh_command(cfg, args.target or "", args.import_key)
        app = build(cfg)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    app.run_polling(allowed_updates=["message", "callback_query"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
